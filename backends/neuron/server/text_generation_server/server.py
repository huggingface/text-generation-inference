import asyncio
from pathlib import Path
from typing import List

from grpc import aio
from grpc_reflection.v1alpha import reflection
from loguru import logger

from .generator import Generator, NeuronGenerator
from .interceptor import ExceptionInterceptor
from .pb import generate_pb2, generate_pb2_grpc


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, generator: Generator, server_urls: List[str]):
        self.generator = generator
        self.server_urls = server_urls

    async def Info(self, request, context):
        return self.generator.info

    async def Health(self, request, context):
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        if request.HasField("id"):
            self.generator.clear(request.id)
        else:
            self.generator.clear()
        return generate_pb2.ClearCacheResponse()

    async def FilterBatch(self, request, context):
        filtered_batch = self.generator.filter(request.batch_id, request.request_ids)
        return generate_pb2.FilterBatchResponse(batch=filtered_batch)

    async def Warmup(self, request, context):
        max_tokens = self.generator.warmup(request.batch)
        return generate_pb2.WarmupResponse(max_supported_total_tokens=max_tokens)

    async def Prefill(self, request, context):
        generations, batch = self.generator.prefill(request.batch)
        return generate_pb2.PrefillResponse(generations=generations, batch=batch)

    async def Decode(self, request, context):
        generations, batch = self.generator.decode(request.batches)
        return generate_pb2.DecodeResponse(generations=generations, batch=batch)


def serve(
    model_id: str,
    revision: str,
    uds_path: Path,
):
    async def serve_inner(model_id: str, revision: str):
        unix_socket_template = "unix://{}-{}"
        local_url = unix_socket_template.format(uds_path, 0)
        server_urls = [local_url]

        try:
            generator = NeuronGenerator.from_pretrained(model_id, revision)
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(interceptors=[ExceptionInterceptor()])
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(generator, server_urls), server
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

    asyncio.run(serve_inner(model_id, revision))
