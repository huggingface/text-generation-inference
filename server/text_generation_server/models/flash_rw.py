import torch
import torch.distributed

from opentelemetry import trace
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from text_generation_server.models import FlashCausalLM
from text_generation_server.models.custom_modeling.flash_rw_modeling import (
    RWConfig,
    FlashRWForCausalLM,
)
from text_generation_server.utils import (
    initialize_torch_distributed,
    weight_files,
    Weights,
)
from transformers import BitsAndBytesConfig
from peft import PeftModel

tracer = trace.get_tracer(__name__)


class FlashRWSharded(FlashCausalLM):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
        peft_model_id: str = None,
    ):
        self.process_group, rank, world_size = initialize_torch_distributed()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16 if dtype is None else dtype
        else:
            raise NotImplementedError("FlashRW is only available on GPU")

        config = RWConfig.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        torch.distributed.barrier(group=self.process_group)
        filenames = weight_files(model_id, revision=revision, extension=".safetensors")
        weights = Weights(
            filenames,
            device,
            dtype,
            process_group=self.process_group,
            aliases={"lm_head.weight": ["transformer.word_embeddings.weight"]},
        )

        config.quantize = quantize
        if config.quantize == "gptq":
            weights._set_gptq_params(model_id)

        if peft_model_id:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                device_map="auto",
                quantization_config=bnb_config
            )

            # load Lora weights
            model = PeftModel.from_pretrained(
                model,
                peft_model_id,
                device_map="auto",
            )
            model.eval()
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision,
                padding_side="left",
                truncation_side="left",
                trust_remote_code=trust_remote_code,
            )

            model = FlashRWForCausalLM(config, weights)

        torch.distributed.barrier(group=self.process_group)
        super(FlashRWSharded, self).__init__(
            model=model.to(device),
            tokenizer=tokenizer,
            num_layers=len(model.transformer.h),
            num_kv_heads=model.transformer.cache_size,
            head_size=model.transformer.head_size,
            dtype=dtype,
            device=device,
            rank=rank,
            world_size=world_size,
        )
