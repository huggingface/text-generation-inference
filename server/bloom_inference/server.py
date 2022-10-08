import asyncio
from grpc import aio

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import Optional, List

from bloom_inference.cache import Cache
from bloom_inference.model import BLOOM, Batch, BLOOMSharded
from bloom_inference.pb import generate_pb2_grpc, generate_pb2


class TextGeneration(generate_pb2_grpc.TextGenerationServicer):
    def __init__(self, model: BLOOM, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    async def ClearCache(self, request, context):
        self.cache.clear()
        return generate_pb2.Empty()

    async def Generate(self, request, context):
        batch = Batch.from_batch_pb(request, self.model.tokenizer, self.model.device)
        finished_generations, cache_entry = self.model.generate_token(batch)
        self.cache.set(cache_entry)

        return generate_pb2.Response(
            finished=[
                finished_generation.to_pb()
                for finished_generation in finished_generations
            ],
            cache_entry=cache_entry.to_pb() if cache_entry else None,
        )

    async def GenerateWithCache(self, request, context):
        batch = Batch.from_batch_cached_pb(request, self.cache)
        finished_generations, cache_entry = self.model.generate_token(batch)
        self.cache.set(cache_entry)

        return generate_pb2.Response(
            finished=[
                finished_generation.to_pb()
                for finished_generation in finished_generations
            ],
            cache_entry=cache_entry.to_pb() if cache_entry else None,
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
        generate_pb2_grpc.add_TextGenerationServicer_to_server(
            TextGeneration(model, Cache(), server_urls), server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGeneration"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)
        await server.start()
        print("Server started at {}".format(local_url))
        await server.wait_for_termination()

    asyncio.run(serve_inner(model_name, sharded, shard_directory))


if __name__ == "__main__":
    serve("bigscience/bloom-560m", True, Path("/tmp/models"))
