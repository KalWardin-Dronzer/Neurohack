from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Sequence[T], workers: int) -> list[R]:
    """Order-preserving map over a thread pool, for I/O-bound LLM calls. The first exception cancels work
    that has not started (so QuotaExhausted stops a run) and is re-raised."""
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        try:
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
