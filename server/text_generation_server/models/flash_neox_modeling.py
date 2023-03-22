import torch

from torch import nn

from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_neox import GPTNeoXConfig
from einops import rearrange
from flash_attn.flash_attn_interface import (
    flash_attn_unpadded_qkvpacked_func,
    flash_attn_unpadded_kvpacked_func,
)
from flash_attn.ops.fused_dense import (
    FusedDense,
    ColumnParallelLinear,
    RowParallelLinear,
    fused_mlp_func,
)
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb_qkv_
from flash_attn.ops.layer_norm import dropout_add_layer_norm


class TensorParallelEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        process_group: torch.distributed.ProcessGroup,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        device=None,
        dtype=None,
    ):
        self.process_group = process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.original_num_embeddings = num_embeddings

        assert num_embeddings % self.tp_world_size == 0
        block_size = num_embeddings // self.tp_world_size
        # inputs in `[min_id, max_id[` are handled by `self` to get embeddings
        self.min_id = self.tp_rank * block_size
        self.max_id = (self.tp_rank + 1) * block_size

        super().__init__(
            block_size,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            device=device,
            dtype=dtype,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check
        if torch.any(
            torch.logical_or(0 > input, input >= self.original_num_embeddings)
        ):
            raise IndexError(
                f"Input is required to be in [0, {self.original_num_embeddings}[, got min: {torch.min(input)} and max: {torch.max(input)}"
            )

        # `0` if input is in the correct interval, else `1`
        input_mask = torch.logical_or(self.min_id > input, input >= self.max_id)
        # translate for [0, self.max_id - self.min_id[
        input = input - self.min_id
        # default all out of bounds values to `0`
        input[input_mask] = 0
        out = super().forward(input)
        out[input_mask] = 0.0
        torch.distributed.all_reduce(out, group=self.process_group)
        return out


class PositionRotaryEmbedding(RotaryEmbedding):
    def forward(self, qkv: torch.Tensor, position_ids: torch.Tensor):
        assert self.scale is None

        self._update_cos_sin_cache(qkv, position_ids.max() + 1)

        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]

        return apply_rotary_emb_qkv_(qkv, cos, sin, None, None)


class FlashNeoxAttention(torch.nn.Module):
    def __init__(
        self, num_heads, hidden_size, rotary_pct, rotary_emb_base, process_group=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_size = hidden_size // num_heads

        rotary_ndims = int(self.head_size * rotary_pct)
        self.rotary_emb = PositionRotaryEmbedding(rotary_ndims, base=rotary_emb_base)
        self.softmax_scale = self.head_size ** (-0.5)

        if process_group is None:
            self.query_key_value = FusedDense(hidden_size, 3 * hidden_size)
            self.dense = FusedDense(hidden_size, hidden_size)
        else:
            self.num_heads = self.num_heads // process_group.size()
            self.query_key_value = ColumnParallelLinear(
                hidden_size,
                3 * hidden_size,
                process_group=process_group,
                sequence_parallel=False,
            )
            self.dense = RowParallelLinear(
                hidden_size,
                hidden_size,
                process_group=process_group,
                sequence_parallel=False,
            )

    def forward(
        self, hidden_states, position_ids, cu_seqlens, max_s, layer_past, prefill
    ):
        qkv = self.query_key_value(hidden_states)
        qkv = rearrange(
            qkv, "... (h three d) -> ... h three d", three=3, d=self.head_size
        ).permute(0, 2, 1, 3)
        qkv_rot = self.rotary_emb(qkv.unsqueeze(0), position_ids).squeeze(0)

        if prefill:
            layer_past[...] = qkv_rot[:, 1:]

            # test flash_attn_unpadded_qkvpacked_split_func
            attn_output = flash_attn_unpadded_qkvpacked_func(
                qkv_rot, cu_seqlens, max_s, 0.0, self.softmax_scale, causal=True
            )
        else:
            query = qkv_rot[:, 0]
            layer_past[cu_seqlens[1:] - 1] = qkv_rot[:, 1:]

            attn_output = flash_attn_unpadded_kvpacked_func(
                query,
                layer_past,
                cu_seqlens_q=torch.arange(len(cu_seqlens), dtype=torch.int32).to(
                    query.device
                ),
                max_seqlen_q=torch.tensor(1, dtype=torch.int32).to(query.device),
                cu_seqlens_k=cu_seqlens,
                max_seqlen_k=max_s,
                dropout_p=0.0,
                softmax_scale=self.softmax_scale,
                causal=False,
            )

        return self.dense(rearrange(attn_output, "... h d -> ... (h d)"))


class FlashMLP(nn.Module):
    def __init__(self, act, hidden_size, intermediate_size, process_group=None):
        super().__init__()
        if "gelu" in act:
            act = "gelu_approx"
        assert act in ["gelu_approx", "relu"]
        self.act = act

        if process_group is None:
            self.dense_h_to_4h = FusedDense(hidden_size, intermediate_size)
            self.dense_4h_to_h = FusedDense(intermediate_size, hidden_size)
        else:
            self.dense_h_to_4h = ColumnParallelLinear(
                hidden_size,
                intermediate_size,
                process_group=process_group,
                sequence_parallel=False,
            )
            self.dense_4h_to_h = RowParallelLinear(
                intermediate_size,
                hidden_size,
                process_group=process_group,
                sequence_parallel=False,
            )
        self.heuristic = "auto"
        self.process_group = process_group

    def forward(self, x):
        if self.heuristic == "auto":
            if self.act == "gelu_approx":
                cuda_ver = tuple(map(int, torch.version.cuda.split(".")))
                self.heuristic = (
                    0
                    if cuda_ver >= (11, 8)
                    else (1 if x.dtype == torch.float16 else -1)
                )
            else:
                self.heuristic = 0

        out = fused_mlp_func(
            x,
            self.dense_h_to_4h.weight,
            self.dense_4h_to_h.weight,
            self.dense_h_to_4h.bias,
            self.dense_4h_to_h.bias,
            activation=self.act,
            save_pre_act=self.training,
            checkpoint_lvl=0,
            heuristic=self.heuristic,
            process_group=self.process_group,
            sequence_parallel=False,
        )
        if self.process_group is not None:
            torch.distributed.all_reduce(out, group=self.process_group)
        return out


class FlashNeoXLayer(nn.Module):
    def __init__(
        self,
        num_heads,
        act,
        hidden_size,
        intermediate_size,
        rotary_pct,
        rotary_emb_base,
        layer_norm_eps,
        use_parallel_residual,
        process_group=None,
    ):
        super().__init__()
        self.use_parallel_residual = use_parallel_residual
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.attention = FlashNeoxAttention(
            num_heads, hidden_size, rotary_pct, rotary_emb_base, process_group
        )
        self.mlp = FlashMLP(act, hidden_size, intermediate_size, process_group)

    def forward(
        self,
        hidden_states,
        residual,
        position_ids,
        cu_seqlens,
        max_s,
        layer_past,
        prefill,
    ):
        if self.use_parallel_residual:
            ln1_hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                0.0,
                self.input_layernorm.eps,
                rowscale=None,
                prenorm=False,
                residual_in_fp32=False,
            )
            attn_output = self.attention(
                ln1_hidden_states, position_ids, cu_seqlens, max_s, layer_past, prefill
            )

            ln2_hidden_states = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.bias,
                0.0,
                self.post_attention_layernorm.eps,
                rowscale=None,
                prenorm=False,
                residual_in_fp32=False,
            )
            mlp_output = self.mlp(ln2_hidden_states)
            return mlp_output + attn_output + hidden_states, None

        else:
            hidden_states, residual = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.input_layernorm.weight,
                self.input_layernorm.bias,
                0.0,
                self.input_layernorm.eps,
                rowscale=None,
                prenorm=True,
                residual_in_fp32=True,
            )

            hidden_states = self.attention(
                hidden_states, position_ids, cu_seqlens, max_s, layer_past, prefill
            )

            hidden_states, residual = dropout_add_layer_norm(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.bias,
                0.0,
                self.post_attention_layernorm.eps,
                rowscale=None,
                prenorm=True,
                residual_in_fp32=True,
            )

            mlp_output = self.mlp(hidden_states)

            return mlp_output, residual


class FlashGPTNeoXPreTrainedModel(PreTrainedModel):
    config_class = GPTNeoXConfig
    base_model_prefix = "gpt_neox"
    supports_gradient_checkpointing = False
    _no_split_modules = None


class FlashGPTNeoXModel(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config, process_group=None):
        super().__init__(config)
        self.config = config

        self.tp_embeddings = False
        if process_group is not None:
            self.tp_rank = process_group.rank()
            self.tp_world_size = process_group.size()
            if config.vocab_size % self.tp_world_size == 0:
                self.tp_embeddings = True

        if self.tp_embeddings:
            self.embed_in = TensorParallelEmbedding(
                config.vocab_size, config.hidden_size, process_group=process_group
            )
        else:
            self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [
                FlashNeoXLayer(
                    config.num_attention_heads,
                    config.hidden_act,
                    config.hidden_size,
                    config.intermediate_size,
                    config.rotary_pct,
                    config.rotary_emb_base,
                    config.layer_norm_eps,
                    config.use_parallel_residual,
                    process_group,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].attention.head_size
        self.num_heads = self.layers[0].attention.num_heads

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states = self.embed_in(input_ids)

        prefill = False
        if past_key_values is None:
            past_key_values = hidden_states.new_empty(
                (
                    len(self.layers),
                    len(hidden_states),
                    2,
                    self.num_heads,
                    self.head_size,
                )
            )
            prefill = True

        residual = None
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                position_ids,
                cu_seqlens,
                max_s,
                past_key_values[i],
                prefill,
            )

        hidden_states = dropout_add_layer_norm(
            hidden_states,
            residual,
            self.final_layer_norm.weight,
            self.final_layer_norm.bias,
            0.0,
            self.final_layer_norm.eps,
            rowscale=None,
            prenorm=False,
            residual_in_fp32=False,
        )

        return hidden_states, past_key_values


class FlashGPTNeoXForCausalLM(FlashGPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if config.tp_parallel:
            process_group = torch.distributed.distributed_c10d._get_default_group()
        else:
            process_group = None

        self.gpt_neox = FlashGPTNeoXModel(config, process_group)

        if self.gpt_neox.tp_embeddings:
            self.embed_out = FusedDense(
                config.hidden_size,
                config.vocab_size // process_group.size(),
                bias=False,
            )
        else:
            self.embed_out = FusedDense(
                config.hidden_size, config.vocab_size, bias=False
            )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        past_key_values=None,
    ):
        hidden_states, present = self.gpt_neox(
            input_ids, position_ids, cu_seqlens, max_s, past_key_values
        )
        return self.embed_out(hidden_states), present


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from flash_attn.bert_padding import unpad_input

    model = (
        FlashGPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m")
        .cuda()
        .to(torch.half)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-160m", padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_inputs = tokenizer(
        ["What is this?\n\nA:\n\nThe answer to the problem?", "hello!"],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    input_ids, indices, cu_seqlens, max_seqlen = unpad_input(
        tokenized_inputs["input_ids"].unsqueeze(-1), tokenized_inputs["attention_mask"]
    )

    position_ids = tokenized_inputs["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(tokenized_inputs["attention_mask"] == 0, 0)

    unpad_position_ids = torch.gather(position_ids.view(-1).cuda(), 0, indices)

    gen_input_ids = input_ids.squeeze(1).cuda().clone()
    gen_position_ids = unpad_position_ids.clone()
    gen_indices = indices.clone()
    gen_cu_seqlens = cu_seqlens.clone()
    gen_max_seqlen = max_seqlen

    past_key_values = None

    results = []
    with torch.no_grad():
        out, present, _ = model(
            gen_input_ids,
            gen_position_ids,
            gen_cu_seqlens,
            gen_max_seqlen,
            past_key_values=past_key_values,
        )

        futures = []
        new_gen_cu_seqlens = [0]
        new_position_ids = []
        next_token_ids = []

        for i in range(len(gen_cu_seqlens) - 1):
            start_index = gen_cu_seqlens[i]
            end_index = gen_cu_seqlens[i + 1]

            seq_logits = out[start_index:end_index]
            next_token_id = torch.argmax(seq_logits[-1:], dim=1)
            next_token_ids.append(next_token_id)

            sequence_length = end_index - start_index
            new_gen_cu_seqlens.append(new_gen_cu_seqlens[i] + sequence_length + 1)

            seq_position_ids = gen_position_ids[start_index:end_index]
            new_position_ids.append(
                torch.concat([seq_position_ids, seq_position_ids[-1:] + 1])
            )

            seq_present = present[:, start_index:end_index]
            future = torch.nn.functional.pad(seq_present, (0, 0, 0, 0, 0, 0, 0, 1))

            futures.append(future)

        past_key_values = torch.concat(futures, dim=1)
        new_position_ids = torch.concat(new_position_ids)
        new_gen_cu_seqlens = torch.tensor(
            new_gen_cu_seqlens, device=past_key_values.device, dtype=torch.int32
        )
        next_token_ids = torch.concat(next_token_ids)

        gen_max_seqlen += 1

        gen_input_ids = next_token_ids
        gen_position_ids = new_position_ids
        gen_cu_seqlens = new_gen_cu_seqlens

        print(tokenizer.batch_decode(gen_input_ids))

        for _ in range(40):
            out, present, _ = model(
                gen_input_ids,
                gen_position_ids,
                gen_cu_seqlens,
                gen_max_seqlen,
                past_key_values=past_key_values,
            )

            futures = []
            new_gen_cu_seqlens = [0]
            new_position_ids = []
            next_token_ids = []
            for i in range(len(gen_cu_seqlens) - 1):
                start_index = gen_cu_seqlens[i]
                end_index = gen_cu_seqlens[i + 1]

                seq_logits = out[i]
                next_token_id = torch.argmax(seq_logits.view(1, -1)[-1:], dim=1)
                next_token_ids.append(next_token_id)

                sequence_length = end_index - start_index
                new_gen_cu_seqlens.append(new_gen_cu_seqlens[i] + sequence_length + 1)

                seq_position_ids = gen_position_ids[start_index:end_index]
                new_position_ids.append(
                    torch.concat([seq_position_ids, seq_position_ids[-1:] + 1])
                )

                seq_present = present[:, start_index:end_index]
                future = torch.nn.functional.pad(seq_present, (0, 0, 0, 0, 0, 0, 0, 1))

                futures.append(future)

            past_key_values = torch.concat(futures, dim=1)
            new_position_ids = torch.concat(new_position_ids)
            new_gen_cu_seqlens = torch.tensor(
                new_gen_cu_seqlens, device=past_key_values.device, dtype=torch.int32
            )
            next_token_ids = torch.concat(next_token_ids)

            gen_max_seqlen += 1

            gen_input_ids = next_token_ids
            gen_position_ids = new_position_ids
            gen_cu_seqlens = new_gen_cu_seqlens

            print(tokenizer.batch_decode(gen_input_ids))
