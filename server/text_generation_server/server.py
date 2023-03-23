import asyncio
import os
import torch

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models import Model, get_model, Seq2SeqLM
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.pb.generate_pb2 import ModelInfoResponse
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def Prefill(self, request, context):
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.device
        )

        generations = self.model.generate_token(batch, prefill=True)
        self.cache.set(batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
        )

    async def Decode(self, request, context):
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.batch_id)
            if batch_pb.HasField("status"):
                if batch is None:
                    raise ValueError(f"Batch ID {batch.batch_id} not found in cache.")
                batch = self.model.batch_type.prune(batch, batch_pb.status.completed_ids)
                if batch is not None:
                    batches.append(batch)

        if len(batches) == 0:
            # All batches finished, nothing to do
            return generate_pb2.DecodeResponse()

        if len(batches) > 1:
            batch = self.model.batch_type.concatenate(batches)
        else:
            batch = batches[0]

        generations = self.model.generate_token(batch)
        self.cache.set(batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch_id=batch.get_id(),
        )

    async def ModelInfo(self, _: generate_pb2.ModelInfoRequest, context) -> generate_pb2.ModelInfoResponse:
        return generate_pb2.ModelInfoResponse(
            model_type=ModelInfoResponse.ModelType.SEQ2SEQ_LM
            if isinstance(self.model, Seq2SeqLM) else ModelInfoResponse.ModelType.CAUSAL_LM,
            eos_token=self.model.tokenizer.eos_token_id,
            skip_special_tokens=self.model.skip_special_tokens,
        )

def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: bool,
    uds_path: Path,
):
    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
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
            model = get_model(model_id, revision, sharded, quantize)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
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

        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(serve_inner(model_id, revision, sharded, quantize))
