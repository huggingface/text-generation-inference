# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import asyncio
import os
import sys
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
from text_generation_server.models import Model, get_model
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor
from text_generation_server.utils.version import is_driver_compatible, MIN_TGI_GAUDI_SYNAPSE_VERSION


class SignalHandler:
    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print(f"Exiting gracefully: Signal {signum}")
        self.KEEP_PROCESSING = False


signal_handler = SignalHandler()


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(
        self,
        model: Model,
        cache: Cache,
        server_urls: List[str],
    ):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        # TODO: The inferecemode set messes up the autograd op dispatch. And results in aten::matmul
        # op not optimized issue. Will investigate further.
        # if model.device.type == "hpu":
        # Force inference mode for the lifetime of TextGenerationService
        # self._inference_mode_raii_guard = torch._C._InferenceMode(True)


    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "hpu":
            torch.zeros((2, 2)).to("hpu")
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
        def batch_from_pb(batch):
            return self.model.batch_type.from_pb(
                batch, self.model.tokenizer, self.model.dtype, self.model.device
            )

        batches = [batch_from_pb(batch) for batch in request.batches]
        self.model.warmup(batches)

        return generate_pb2.WarmupResponse()

    async def Prefill(self, request, context):
        start = time.time_ns()
        batch = self.model.batch_type.from_pb(
            request.batch, self.model.tokenizer, self.model.dtype, self.model.device
        )
        generations, next_batch, timings = self.model.generate_token([batch])
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
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

        generations, next_batch, timings = self.model.generate_token(batches)
        self.cache.set(next_batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            concat_ns=None, # TODO: measure concat time
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )


def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
    uds_path: Path,
):
    # Remove default handler
    logger.remove()
    logger.add(
        sys.stdout,
        format="{message}",
        filter="text_generation_server",
        level="INFO",
        serialize=False,
        backtrace=True,
        diagnose=False,
    )

    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        speculate: Optional[int] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if not is_driver_compatible():
            logger.warning(f"Current Synapse version is lower than the minimum version supported: {MIN_TGI_GAUDI_SYNAPSE_VERSION}, this could result in failures")

        unix_socket_template = "unix://{}-{}"
        logger.info("Server:server_inner: sharded ={}".format(sharded))

        if sharded:
            rank = int(os.environ["RANK"])
            logger.info("Server:server_inner: rank ={}".format(rank))
            server_urls = [
                unix_socket_template.format(uds_path, rank) for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        logger.info("Server:server_inner: data type = {}, local_url = {}".format(dtype, local_url))
        if dtype == "bfloat16" or None:
            data_type = torch.bfloat16
        else:
            data_type = torch.float
        if revision == "None":
            revision = None
        try:
            model = get_model(
                model_id,
                revision,
                speculate,
                data_type,
                trust_remote_code
            )
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

        while signal_handler.KEEP_PROCESSING:
            await asyncio.sleep(0.5)

    asyncio.run(
        serve_inner(
            model_id, revision, sharded, speculate, dtype, trust_remote_code
        )
    )
