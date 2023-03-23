import re
import torch
import torch.distributed

from typing import List, Optional, Type, Tuple

from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedTokenizerBase,
)
from transformers.models.opt.parallel_layers import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelRowLinear,
)

from text_generation_server.models import CausalLM
from text_generation_server.pb import generate_pb2
from text_generation_server.models.causal_lm import CausalLMBatch
from text_generation_server.utils import (
    NextTokenChooser,
    StoppingCriteria,
    initialize_torch_distributed,
    weight_files,
)

HAS_BITS_AND_BYTES = True
try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Int8Params
except Exception as e:
    HAS_BITS_AND_BYTES = False


# CREDIT: Papers with code => https://github.com/paperswithcode/galai/blob/main/galai/utils.py

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"


def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].
    Parameters
    ----------
    n : str
        Input text to split
    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", rf"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization
    Parameters
    ----------
    text : str
        Input text to split
    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)


# END CREDIT


class GalacticaCausalLMBatch(CausalLMBatch):
    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "GalacticaCausalLMBatch":
        inputs = []
        next_token_choosers = []
        stopping_criterias = []
        input_lengths = []

        # Parse batch
        max_sequence_length = 0
        padding_right_offset = 0
        for r in pb.requests:
            # Add escape_custom_split_sequence to the CausalLMBatch logic
            inputs.append(escape_custom_split_sequence(r.inputs))
            input_lengths.append(r.input_length)
            next_token_choosers.append(NextTokenChooser.from_pb(r.parameters, device))
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            max_sequence_length = max(max_sequence_length, r.input_length)
            padding_right_offset = max(
                padding_right_offset, stopping_criteria.max_new_tokens
            )

        # Tokenize batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(device)
        input_ids = tokenized_inputs["input_ids"]
        # Allocate maximum attention_mask
        attention_mask = input_ids.new_zeros(
            (pb.size, max_sequence_length + padding_right_offset)
        )
        # Copy tokenizer attention_mask into fully allocated attention_mask
        attention_mask[:, :max_sequence_length] = tokenized_inputs["attention_mask"]

        position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
        position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 1)
        all_input_ids = tokenized_inputs["input_ids"].unsqueeze(-1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            size=pb.size,
            max_sequence_length=max_sequence_length,
            padding_right_offset=padding_right_offset,
        )


class Galactica(CausalLM):
    @property
    def batch_type(self) -> Type[CausalLMBatch]:
        return GalacticaCausalLMBatch

    def decode(self, generated_ids: List[int]) -> str:
        # Do not skip special tokens as they are used for custom parsing rules of the generated text
        return self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, cleanup_tokenization_spaces=False
        )

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Overwrite forward to ignore position_ids"""

        # Model Forward
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, outputs.past_key_values


class GalacticaSharded(Galactica):
    def __init__(
        self, model_id: str, revision: Optional[str] = None, quantize: bool = False
    ):
        self.process_group, self.rank, self.world_size = initialize_torch_distributed()
        self.master = self.rank == 0
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            dtype = torch.bfloat16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )

        config = AutoConfig.from_pretrained(
            model_id, revision=revision, tp_parallel=True
        )
        tokenizer.pad_token_id = config.pad_token_id

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        torch.distributed.barrier(group=self.process_group)
        self.load_weights(
            model,
            filenames,
            quantize=quantize,
            device=device,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.model = model.eval().to(dtype)
        torch.distributed.barrier(group=self.process_group)
        super(CausalLM, self).__init__(
            tokenizer=tokenizer,
            device=device,
        )

    @staticmethod
    def load_weights(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    if name == "lm_head.weight":
                        continue

                    module_name, param_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                    current_tensor = parameters[name]

                    slice_ = f.get_slice(name)

                    if isinstance(module, TensorParallelColumnLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[0]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[start:stop]
                        else:
                            size = slice_.get_shape()[0]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[start:stop]
                    elif isinstance(module, TensorParallelRowLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[1]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[:, start:stop]
                        else:
                            tensor = slice_[:]
                            # XXX: Hack for Rowlinear to add the bias only once.
                            if rank != 0:
                                tensor = torch.zeros_like(tensor)
                    elif isinstance(module, TensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        tensor = slice_[:]

                    if current_tensor.shape != tensor.shape:
                        raise ValueError(
                            f"Name {name} -- Current {current_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous()

                    if quantize:
                        if not HAS_BITS_AND_BYTES:
                            raise ImportError(
                                "bitsandbytes is not available on your machine either because it is not installed "
                                "or you don't have a GPU.\n"
                                "You can install it with `pip install bitsandbytes`."
                            )

                        if (
                            type(module)
                            in [TensorParallelRowLinear, TensorParallelColumnLinear]
                            and param_name == "weight"
                        ):
                            tensor = Int8Params(
                                tensor,
                                has_fp16_weights=False,
                                requires_grad=False,
                            ).to(device)
                            state = bnb.MatmulLtState()
                            state.threshold = 6.0
                            state.has_fp16_weights = False
                            state.memory_efficient_backward = False
                            state.use_pool = True
                            state.CB = tensor.CB
                            state.SCB = tensor.SCB
                            tensor.CB = None
                            tensor.SCB = None

                            def replace_linear(state):
                                def linear(input, weight, bias):
                                    out = bnb.matmul(
                                        input,
                                        weight,
                                        state=state,
                                        threshold=state.threshold,
                                        bias=bias,
                                    )

                                    if state.CB is not None:
                                        # we converted 8-bit row major to turing/ampere format
                                        # in the first inference pass
                                        # we no longer need the row-major weight
                                        del state.CB
                                        weight.data = state.CxB

                                    return out

                                return linear

                            module.linear = replace_linear(state)

                        else:
                            tensor = tensor.to(device)

                    module._parameters[param_name] = tensor
                    if name == "model.decoder.embed_tokens.weight":
                        model.lm_head._parameters["weight"] = tensor

    def forward(
        self, input_ids, attention_mask, position_ids, past_key_values: Optional = None
    ):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        # Logits are sharded, so we need to gather them
        logits = [torch.empty_like(outputs.logits) for _ in range(self.world_size)]
        torch.distributed.all_gather(logits, outputs.logits, group=self.process_group)
        logits = torch.cat(logits, dim=2)

        return logits, outputs.past_key_values
