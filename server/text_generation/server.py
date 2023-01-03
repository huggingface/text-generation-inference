import asyncio
import os

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List

from text_generation.cache import Cache
from text_generation.interceptor import ExceptionInterceptor
from text_generation.models import Model, get_model
from text_generation.pb import generate_pb2_grpc, generate_pb2


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def Generate(self, request, context):
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.device
        )

        generated_texts, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.GenerateResponse(
            generated_texts=[
                generated_text.to_pb() for generated_text in generated_texts
            ],
            batch=next_batch.to_pb() if next_batch else None,
        )

    async def GenerateWithCache(self, request, context):
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) > 1:
            batch = self.model.batch_type.concatenate(batches)
        else:
            batch = batches[0]

        generated_texts, next_batch = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.GenerateWithCacheResponse(
            generated_texts=[
                generated_text.to_pb() for generated_text in generated_texts
            ],
            batch=next_batch.to_pb() if next_batch else None,
        )


def serve(
    model_name: str,
    sharded: bool,
    quantize: bool,
    uds_path: Path,
):
    async def serve_inner(
        model_name: str,
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

        model = get_model(model_name, sharded, quantize)

        server = aio.server(interceptors=[ExceptionInterceptor()])
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

    asyncio.run(serve_inner(model_name, sharded, quantize))
