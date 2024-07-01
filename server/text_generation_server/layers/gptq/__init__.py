from dataclasses import dataclass
from loguru import logger
import os
from typing import List, Optional, Union, Dict
from safetensors import SafetensorError
from pathlib import Path
from text_generation_server.utils.weights import Weights
import torch
from text_generation_server.utils.import_utils import (
    SYSTEM,
)
from text_generation_server.utils.log import log_once
from huggingface_hub import hf_hub_download
import json


@dataclass
class GPTQParams:
    bits: int
    checkpoint_format: Optional[str]
    groupsize: int
    desc_act: bool
    quant_method: str
    sym: bool


@dataclass
class GPTQWeight:
    qweight: torch.Tensor
    qzeros: torch.Tensor
    scales: torch.Tensor
    g_idx: Optional[torch.Tensor]
    bits: int
    groupsize: int
    use_exllama: bool

    def __post_init__(self):
        if self.scales.dtype == torch.float:
            self.scales = self.scales.half()

    @property
    def device(self) -> torch.device:
        return self.qweight.device


try:
    major, _minor = torch.cuda.get_device_capability()
except Exception:
    major = 1

HAS_EXLLAMA = False
CAN_EXLLAMA = major >= 8 or SYSTEM == "rocm"
V2 = os.getenv("EXLLAMA_VERSION", "2") == "2"
if os.getenv("DISABLE_EXLLAMA") == "True":
    HAS_EXLLAMA = False
elif CAN_EXLLAMA:
    try:
        if V2:
            from text_generation_server.layers.gptq.exllamav2 import (
                QuantLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "2"
        else:
            from text_generation_server.layers.gptq.exllama import (
                Ex4bitLinear as ExllamaQuantLinear,
                create_exllama_buffers,
                set_device,
            )

            HAS_EXLLAMA = "1"

    except ImportError:
        pass

from text_generation_server.layers.gptq.quant_linear import QuantLinear


class GPTQWeights(Weights):
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
        prefix: Optional[str] = None,
    ):
        super().__init__(
            filenames=filenames,
            device=device,
            dtype=dtype,
            process_group=process_group,
            aliases=aliases,
            prefix=prefix,
        )

    def get_weights_col_packed(
        self, prefix: str, quantize: str, block_sizes: Union[int, List[int]]
    ):
        from text_generation_server.layers.marlin import (
            can_use_gptq_marlin,
            repack_gptq_for_marlin,
        )

        try:
            qweight = self.get_packed_sharded(
                f"{prefix}.qweight", dim=1, block_sizes=block_sizes
            )
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{quantize}` weight, make sure the model is already quantized."
            )
        scales = self.get_packed_sharded(
            f"{prefix}.scales", dim=1, block_sizes=block_sizes
        )
        scales = scales.to(dtype=self.dtype)

        gptq_params = self._get_gptq_params()
        if can_use_gptq_marlin(gptq_params, quantize):
            g_idx = self.get_tensor(f"{prefix}.g_idx")
            return repack_gptq_for_marlin(
                qweight=qweight,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                desc_act=gptq_params.desc_act,
                groupsize=gptq_params.groupsize,
                sym=gptq_params.sym,
                sharded_infeatures=False,
            )

        qzeros = self.get_packed_sharded(
            f"{prefix}.qzeros", dim=1, block_sizes=block_sizes
        )
        if quantize == "gptq" and gptq_params.quant_method == "gptq":
            g_idx = self.get_tensor(f"{prefix}.g_idx")
        elif quantize == "gptq" and gptq_params.quant_method == "awq":
            log_once(
                logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
            )
            from text_generation_server.layers.awq.conversion_utils import (
                fast_awq_to_gptq,
            )

            qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
            g_idx = (
                torch.arange(
                    qweight.shape[0] * (32 // gptq_params.bits),
                    device=qweight.device,
                )
                // gptq_params.groupsize
            ).to(dtype=torch.int32)
        else:
            g_idx = None

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=gptq_params.bits,
            groupsize=gptq_params.groupsize,
            use_exllama=False,
        )

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int):
        from text_generation_server.layers.marlin import (
            can_use_gptq_marlin,
            repack_gptq_for_marlin,
        )

        try:
            qweight = torch.cat(
                [self.get_sharded(f"{p}.qweight", dim=1) for p in prefixes], dim=1
            )
        except RuntimeError:
            raise RuntimeError(
                f"Cannot load `{quantize}` weight, make sure the model is already quantized"
            )

        scales = torch.cat(
            [self.get_sharded(f"{p}.scales", dim=1) for p in prefixes], dim=1
        )

        gptq_params = self._get_gptq_params()
        if can_use_gptq_marlin(gptq_params, quantize):
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            return repack_gptq_for_marlin(
                qweight=qweight,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                desc_act=gptq_params.desc_act,
                groupsize=gptq_params.groupsize,
                sym=gptq_params.sym,
                sharded_infeatures=False,
            )

        qzeros = torch.cat(
            [self.get_sharded(f"{p}.qzeros", dim=1) for p in prefixes], dim=1
        )

        from text_generation_server.layers.gptq import HAS_EXLLAMA

        use_exllama = (
            gptq_params.bits == 4
            and HAS_EXLLAMA
            and quantize == "gptq"
            and not gptq_params.desc_act
        )

        if quantize == "gptq" and gptq_params.quant_method == "gptq":
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]
        elif quantize == "gptq" and gptq_params.quant_method == "awq":
            log_once(
                logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
            )
            from text_generation_server.layers.awq.conversion_utils import (
                fast_awq_to_gptq,
            )

            qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
            if use_exllama:
                g_idx = None
            else:
                g_idx = (
                    torch.arange(
                        qweight.shape[0] * (32 // gptq_params.bits),
                        device=qweight.device,
                    )
                    // gptq_params.groupsize
                ).to(dtype=torch.int32)
        else:
            g_idx = None

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=gptq_params.bits,
            groupsize=gptq_params.groupsize,
            use_exllama=use_exllama,
        )

    def get_multi_weights_row(self, prefix: str, quantize: str):
        from text_generation_server.layers.marlin import (
            can_use_gptq_marlin,
            repack_gptq_for_marlin,
        )

        gptq_params = self._get_gptq_params()
        if can_use_gptq_marlin(gptq_params, quantize):
            log_once(logger.info, "Using GPTQ-Marlin kernels")
            try:
                qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
            except RuntimeError:
                raise RuntimeError(
                    f"Cannot load `{quantize}` weight for GPTQ -> Marlin repacking, make sure the model is already quantized"
                )

            g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)
            if gptq_params.desc_act or gptq_params.groupsize == -1:
                scales = self.get_tensor(f"{prefix}.scales")
            else:
                scales = self.get_sharded(f"{prefix}.scales", dim=0)

            sharded_in_features = self.process_group.size() > 1

            return repack_gptq_for_marlin(
                qweight=qweight,
                scales=scales,
                g_idx=g_idx,
                bits=gptq_params.bits,
                desc_act=gptq_params.desc_act,
                groupsize=gptq_params.groupsize,
                sym=gptq_params.sym,
                sharded_infeatures=sharded_in_features,
            )

        use_exllama = True
        if gptq_params.bits != 4:
            use_exllama = False

        if gptq_params.desc_act:
            log_once(logger.warning, "Disabling exllama because desc_act=True")
            use_exllama = False

        try:
            qweight = self.get_sharded(f"{prefix}.qweight", dim=0)
        except RuntimeError:
            raise RuntimeError(
                "Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`"
            )

        if quantize == "gptq" and gptq_params.quant_method == "gptq":
            g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)
        else:
            g_idx = None

        if self.process_group.size() > 1:
            if g_idx is not None:
                if (
                    not torch.equal(
                        g_idx.cpu(),
                        torch.tensor(
                            [i // gptq_params.groupsize for i in range(g_idx.shape[0])],
                            dtype=torch.int32,
                        ),
                    )
                    and not (g_idx == 0).all()
                ):
                    # Exllama implementation does not support row tensor parallelism with act-order, as
                    # it would require to reorder input activations that are split unto several GPUs
                    use_exllama = False

        from text_generation_server.layers.gptq import (
            HAS_EXLLAMA,
            CAN_EXLLAMA,
            GPTQWeight,
        )

        if use_exllama:
            if not HAS_EXLLAMA:
                if CAN_EXLLAMA:
                    log_once(
                        logger.warning,
                        "Exllama GPTQ cuda kernels (which are faster) could have been used, but are not currently installed, try using BUILD_EXTENSIONS=True",
                    )
                use_exllama = False
            else:
                log_once(logger.info, f"Using exllama kernels v{HAS_EXLLAMA}")

        if use_exllama and gptq_params.groupsize != -1:
            qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
            scales = self.get_sharded(f"{prefix}.scales", dim=0)
        else:
            qzeros = self.get_tensor(f"{prefix}.qzeros")
            scales = self.get_tensor(f"{prefix}.scales")

        if use_exllama and g_idx is not None:
            g_idx = g_idx - g_idx[0]

        if quantize == "gptq" and gptq_params.quant_method == "awq":
            log_once(
                logger.info, "Converting AWQ model to Exllama/GPTQ packing format."
            )
            from text_generation_server.layers.awq.conversion_utils import (
                fast_awq_to_gptq,
            )

            qweight, qzeros = fast_awq_to_gptq(qweight, qzeros)
            if use_exllama:
                g_idx = None
            else:
                g_idx = (
                    torch.arange(
                        qweight.shape[0] * (32 // gptq_params.bits),
                        device=qweight.device,
                    )
                    // gptq_params.groupsize
                ).to(dtype=torch.int32)

        return GPTQWeight(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            g_idx=g_idx,
            bits=gptq_params.bits,
            groupsize=gptq_params.groupsize,
            use_exllama=use_exllama,
        )

    def _get_gptq_params(self) -> GPTQParams:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
            checkpoint_format = getattr(self, "gptq_checkpoint_format", None)
            desc_act = False
            sym = False
            quant_method = "gptq"
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
                checkpoint_format = getattr(self, "gptq_checkpoint_format", None)
                desc_act = getattr(self, "gptq_desc_act", False)
                quant_method = getattr(self, "quant_method", "gptq")
                sym = getattr(self, "sym", True)
            except Exception:
                raise e

        return GPTQParams(
            bits=bits,
            checkpoint_format=checkpoint_format,
            desc_act=desc_act,
            groupsize=groupsize,
            quant_method=quant_method,
            sym=sym,
        )

    def _set_gptq_params(self, model_id, revision):
        filename = "config.json"
        try:
            if os.path.exists(os.path.join(model_id, filename)):
                filename = os.path.join(model_id, filename)
            else:
                filename = hf_hub_download(
                    model_id, filename=filename, revision=revision
                )
            with open(filename, "r") as f:
                data = json.load(f)
            self.gptq_bits = data["quantization_config"]["bits"]
            self.gptq_groupsize = data["quantization_config"]["group_size"]
            # Order is important here, desc_act is missing on some real models
            self.quant_method = data["quantization_config"]["quant_method"]
            self.gptq_checkpoint_format = data["quantization_config"].get(
                "checkpoint_format"
            )
            self.gptq_sym = data["quantization_config"]["sym"]
            self.gptq_desc_act = data["quantization_config"]["desc_act"]
        except Exception:
            filename = "quantize_config.json"
            try:
                if os.path.exists(os.path.join(model_id, filename)):
                    filename = os.path.join(model_id, filename)
                else:
                    filename = hf_hub_download(
                        model_id, filename=filename, revision=revision
                    )
                with open(filename, "r") as f:
                    data = json.load(f)
                self.gptq_bits = data["bits"]
                self.gptq_groupsize = data["group_size"]
                self.gptq_sym = data["sym"]
                self.gptq_desc_act = data["desc_act"]
                if "version" in data and data["version"] == "GEMM":
                    self.quant_method = "awq"
            except Exception:
                filename = "quant_config.json"
                try:
                    if os.path.exists(os.path.join(model_id, filename)):
                        filename = os.path.join(model_id, filename)
                    else:
                        filename = hf_hub_download(
                            model_id, filename=filename, revision=revision
                        )
                    with open(filename, "r") as f:
                        data = json.load(f)
                    self.gptq_bits = data["w_bit"]
                    self.gptq_groupsize = data["q_group_size"]
                    self.gptq_desc_act = data["desc_act"]
                    if "version" in data and data["version"] == "GEMM":
                        self.quant_method = "awq"
                except Exception:
                    pass
