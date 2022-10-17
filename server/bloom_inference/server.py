import asyncio
import os

from grpc import aio

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import Optional, List

from bloom_inference.cache import Cache
from bloom_inference.model import BLOOM, Batch, BLOOMSharded
from bloom_inference.pb import generate_pb2_grpc, generate_pb2


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: BLOOM, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def Generate(self, request, context):
        batch = Batch.from_pb(request.batch, self.model.tokenizer, self.model.device)

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
            batch = Batch.concatenate(batches)
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

    async def GenerateUntilFinished(self, request, context):
        batch = Batch.from_pb(request.batch, self.model.tokenizer, self.model.device)

        generated_texts = []
        while not generated_texts:
            generated_texts, next_batch = self.model.generate_token(batch)
            batch = next_batch
        self.cache.set(next_batch)

        return generate_pb2.GenerateUntilFinishedResponse(
            generated_texts=[
                generated_text.to_pb() for generated_text in generated_texts
            ],
            batch=next_batch.to_pb() if next_batch else None,
        )

    async def GenerateUntilFinishedWithCache(self, request, context):
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) > 1:
            batch = Batch.concatenate(batches)
        else:
            batch = batches[0]

        generated_texts = []
        while not generated_texts:
            generated_texts, next_batch = self.model.generate_token(batch)
            batch = next_batch
        self.cache.set(next_batch)

        return generate_pb2.GenerateUntilFinishedWithCacheResponse(
            generated_texts=[
                generated_text.to_pb() for generated_text in generated_texts
            ],
            batch=next_batch.to_pb() if next_batch else None,
        )


def serve(model_name, sharded, shard_directory):
    async def serve_inner(
        model_name: str,
        sharded: bool = False,
        shard_directory: Optional[Path] = None,
    ):
        unix_socket_template = "unix:///tmp/bloom-inference-{}"
        if sharded:
            if shard_directory is None:
                raise ValueError("shard_directory must be set when sharded is True")
            model = BLOOMSharded(model_name, shard_directory)
            server_urls = [
                unix_socket_template.format(rank) for rank in range(model.world_size)
            ]
            local_url = unix_socket_template.format(model.rank)
        else:
            model = BLOOM(model_name)
            local_url = unix_socket_template.format(0)
            server_urls = [local_url]

        server = aio.server()
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
        print("Server started at {}".format(local_url))
        await server.wait_for_termination()

    asyncio.run(serve_inner(model_name, sharded, shard_directory))
