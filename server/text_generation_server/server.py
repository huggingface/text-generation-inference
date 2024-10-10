import asyncio
import os
import torch
import time
import signal

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models import Model, get_model_with_lora_adapters
from text_generation_server.utils.adapter import AdapterInfo
from text_generation_server.utils.prefill_chunking import set_max_prefill_tokens

try:
    from text_generation_server.models.pali_gemma import PaliGemmaBatch
    from text_generation_server.models.vlm_causal_lm import (
        VlmCausalLMBatch,
    )
    from text_generation_server.models.idefics_causal_lm import IdeficsCausalLMBatch
    from text_generation_server.models.mllama_causal_lm import MllamaCausalLMBatch

    VLM_BATCH_TYPES = {
        PaliGemmaBatch,
        VlmCausalLMBatch,
        IdeficsCausalLMBatch,
        MllamaCausalLMBatch,
    }
except (ImportError, NotImplementedError):
    # These imports can fail on CPU/Non flash.
    VLM_BATCH_TYPES = set()

from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor
from text_generation_server.models.globals import set_adapter_to_index


class SignalHandler:
    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def set_keep_processing(self, value: bool):
        self.KEEP_PROCESSING = value

    def exit_gracefully(self, signum, frame):
        print(f"Exiting gracefully: Signal {signum}")
        self.set_keep_processing(False)


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(
        self,
        model: Model,
        cache: Cache,
        server_urls: List[str],
    ):
        self.cache = cache
        self.model = model
        # Quantize is resolved during model loading
        self.quantize = model.quantize
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        if request.HasField("id"):
            self.cache.delete(request.id)
        else:
            self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        batch = self.cache.pop(request.batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {request.batch_id} not found in cache.")
        filtered_batch = batch.filter(request.request_ids)
        self.cache.set(filtered_batch)

        return generate_pb2.FilterBatchResponse(batch=filtered_batch.to_pb())

    async def Warmup(self, request, context):
        set_max_prefill_tokens(request.max_prefill_tokens)

        if self.quantize in {"exl2", "gptq"}:
            try:
                # When using GPTQ, Exllama kernels need some global kernels
                # For which we have the finale shapes only after the model has loaded
                # This will allocate those buffers.
                from text_generation_server.layers.gptq import (
                    create_exllama_buffers,
                    set_device,
                )

                set_device(self.model.device)
                create_exllama_buffers(request.max_prefill_tokens)
            except ImportError:
                pass

        if (
            self.model.batch_type in VLM_BATCH_TYPES
        ):  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb_processor(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.model.config,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )
        max_supported_total_tokens = self.model.warmup(batch)

        return generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )

    async def Prefill(self, request, context):
        start = time.time_ns()
        if (
            self.model.batch_type in VLM_BATCH_TYPES
        ):  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb_processor(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.model.config,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )

        concat_ns = None
        if self.model.support_chunking:
            if request.HasField("cached_batch"):
                cached_batch = self.cache.pop(request.cached_batch.id)
                if cached_batch is None:
                    raise ValueError(
                        f"Batch ID {request.cached_batch.id} not found in cache."
                    )
                start_concat = time.time_ns()
                batch = self.model.batch_type.concatenate([cached_batch, batch])
                concat_ns = time.time_ns() - start_concat

        generations, next_batch, timings = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
            concat_ns=concat_ns,
        )

    async def Decode(self, request, context):
        start = time.time_ns()
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        if len(batches) > 1:
            start_concat = time.time_ns()
            batch = self.model.batch_type.concatenate(batches)
            concat_ns = time.time_ns() - start_concat
        else:
            batch = batches[0]
            concat_ns = None

        generations, next_batch, timings = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            concat_ns=concat_ns,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )


def serve(
    model_id: str,
    lora_adapters: Optional[List[AdapterInfo]],
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    kv_cache_dtype: Optional[str],
    trust_remote_code: bool,
    uds_path: Path,
    max_input_tokens: int,
):
    async def serve_inner(
        model_id: str,
        lora_adapters: Optional[List[AdapterInfo]],
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
        speculate: Optional[int] = None,
        dtype: Optional[str] = None,
        kv_cache_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        adapter_to_index = {}
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model_with_lora_adapters(
                model_id,
                lora_adapters,
                revision,
                sharded,
                quantize,
                speculate,
                dtype,
                kv_cache_dtype,
                trust_remote_code,
                max_input_tokens,
                adapter_to_index,
            )

        except Exception:
            logger.exception("Error when initializing model")
            raise

        signal_handler = SignalHandler()

        set_adapter_to_index(adapter_to_index)
        server = aio.server(
            interceptors=[
                ExceptionInterceptor(lambda: signal_handler.set_keep_processing(False)),
                UDSOpenTelemetryAioServerInterceptor(),
            ],
            options=[
                # Set the maximum possible message length: i32::MAX
                ("grpc.max_receive_message_length", (1 << 31) - 1)
            ],
        )
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(model, Cache(), server_urls), server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))
        while signal_handler.KEEP_PROCESSING:
            await asyncio.sleep(0.5)

    asyncio.run(
        serve_inner(
            model_id,
            lora_adapters,
            revision,
            sharded,
            quantize,
            speculate,
            dtype,
            kv_cache_dtype,
            trust_remote_code,
        )
    )
