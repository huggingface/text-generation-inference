from typing import Any, Callable

import grpc
from google.rpc import code_pb2, status_pb2
from grpc_interceptor.server import AsyncServerInterceptor
from grpc_status import rpc_status
from loguru import logger


class ExceptionInterceptor(AsyncServerInterceptor):
    async def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        context: grpc.ServicerContext,
        method_name: str,
    ) -> Any:
        try:
            response = method(request_or_iterator, context)
            return await response
        except Exception as err:
            method_name = method_name.split("/")[-1]
            logger.exception(f"Method {method_name} encountered an error.")

            await context.abort_with_status(
                rpc_status.to_status(status_pb2.Status(code=code_pb2.INTERNAL, message=str(err)))
            )
