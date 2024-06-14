import functools
import os

from typing import Optional
import sys
import pytest

SYSTEM = os.environ.get("SYSTEM")


def is_flaky_async(
    max_attempts: int = 5,
    wait_before_retry: Optional[float] = None,
    description: Optional[str] = None,
):
    """
    To decorate flaky tests. They will be retried on failures.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        async def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count <= max_attempts:
                try:
                    return await test_func_ref(*args, **kwargs)

                except Exception as err:
                    if retry_count == max_attempts:
                        raise err

                    print(
                        f"Test failed at try {retry_count}/{max_attempts}.",
                        file=sys.stderr,
                    )
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

        return wrapper

    return decorator

def require_backend(*args):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*wrapper_args, **wrapper_kwargs):
            if SYSTEM not in args:
                pytest.skip(
                    f"Skipping as this test requires the backend {args} to be run, but current system is SYSTEM={SYSTEM}."
                )
            return func(*wrapper_args, **wrapper_kwargs)

        return wrapper

    return decorator


def require_backend_async(*args):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*wrapper_args, **wrapper_kwargs):
            if SYSTEM not in args:
                pytest.skip(
                    f"Skipping as this test requires the backend {args} to be run, but current system is SYSTEM={SYSTEM}."
                )
            return await func(*wrapper_args, **wrapper_kwargs)

        return wrapper

    return decorator
