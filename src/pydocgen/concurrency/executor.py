"""Concurrent batch execution pool with retries and rate limiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from pydocgen.batching.models import FunctionBatch
from pydocgen.concurrency.limiter import RateLimiter
from pydocgen.config.settings import (
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_INITIAL_BACKOFF,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_RETRIES,
)
from pydocgen.errors.exceptions import ConcurrencyError
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BatchTaskResult[T]:
    """Result of processing a single FunctionBatch task.

    Attributes:
        batch_id: Identifier of the processed FunctionBatch.
        success: Whether the batch task succeeded.
        data: Return data from worker_fn if successful.
        error: Captured exception if failed.
    """

    batch_id: str
    success: bool
    data: T | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.batch_id, str) or not self.batch_id.strip():
            raise ConcurrencyError("batch_id must be a non-empty string.")
        if self.success and self.error is not None:
            raise ConcurrencyError("Successful BatchTaskResult cannot contain an error.")
        if not self.success and self.error is None:
            raise ConcurrencyError("Failed BatchTaskResult must contain an error.")

    @property
    def error_message(self) -> str | None:
        """String representation of the error if failed, otherwise None."""
        return str(self.error) if self.error is not None else None


async def _execute_with_retry[T](
    batch: FunctionBatch,
    worker_fn: Callable[[FunctionBatch], Awaitable[T]],
    rate_limiter: RateLimiter | None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
) -> BatchTaskResult[T]:
    """Execute a single batch with exponential backoff on failure.

    Args:
        batch: The FunctionBatch to process.
        worker_fn: Async callable taking a FunctionBatch and returning generated data.
        rate_limiter: Optional RateLimiter to throttle API invocations.
        max_retries: Maximum retry attempts before marking the batch as failed.
        initial_backoff: Initial sleep duration in seconds after the first failure.
        backoff_multiplier: Multiplier for exponential backoff on subsequent retries.

    Returns:
        BatchTaskResult containing either data or error.

    Raises:
        ConcurrencyError: If parameters are invalid.
    """
    if not isinstance(batch, FunctionBatch):
        raise ConcurrencyError(f"batch must be a FunctionBatch, got {type(batch).__name__}.")
    if not callable(worker_fn):
        raise ConcurrencyError("worker_fn must be callable.")
    if rate_limiter is not None and not isinstance(rate_limiter, RateLimiter):
        raise ConcurrencyError(
            f"rate_limiter must be a RateLimiter or None, got {type(rate_limiter).__name__}."
        )
    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ConcurrencyError("max_retries must be a non-negative integer.")
    if (
        not isinstance(initial_backoff, (int, float))
        or isinstance(initial_backoff, bool)
        or initial_backoff < 0
    ):
        raise ConcurrencyError("initial_backoff must be a non-negative number.")
    if (
        not isinstance(backoff_multiplier, (int, float))
        or isinstance(backoff_multiplier, bool)
        or backoff_multiplier < 1.0
    ):
        raise ConcurrencyError("backoff_multiplier must be a number >= 1.0.")

    attempt = 0
    backoff = float(initial_backoff)

    while True:
        try:
            if rate_limiter is not None:
                await rate_limiter.acquire()

            result = await worker_fn(batch)
            logger.debug(
                "Batch task completed successfully",
                extra={"batch_id": batch.id, "attempt": attempt + 1},
            )
            return BatchTaskResult(batch_id=batch.id, success=True, data=result)

        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > max_retries:
                logger.error(
                    "Batch task failed after exhausting all retry attempts",
                    extra={
                        "batch_id": batch.id,
                        "total_attempts": attempt,
                        "max_retries": max_retries,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                    },
                    exc_info=exc,
                )
                return BatchTaskResult(
                    batch_id=batch.id,
                    success=False,
                    error=exc,
                )

            logger.warning(
                "Batch task execution failed; retrying after backoff",
                extra={
                    "batch_id": batch.id,
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "backoff_seconds": backoff,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                },
            )
            await asyncio.sleep(backoff)
            backoff *= backoff_multiplier


async def run_batch_pool[T](
    batches: Sequence[FunctionBatch],
    worker_fn: Callable[[FunctionBatch], Awaitable[T]],
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    rate_limiter: RateLimiter | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
    preserve_order: bool = False,
) -> list[BatchTaskResult[T]]:
    """Distribute batches across bounded concurrent workers.

    Args:
        batches: Sequence of FunctionBatch instances.
        worker_fn: Async callable that takes a FunctionBatch and returns generated data.
        max_concurrency: Maximum number of simultaneous active tasks.
        rate_limiter: Optional RPM rate limiter.
        max_retries: Number of retries per batch before marking it failed.
        initial_backoff: Initial retry backoff delay in seconds.
        backoff_multiplier: Exponential multiplier for successive retries.
        preserve_order: Whether to return results sorted in the order of input batches.

    Returns:
        List of BatchTaskResult instances.

    Raises:
        ConcurrencyError: If parameters are invalid.
    """
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise ConcurrencyError(
            f"batches must be a Sequence of FunctionBatch, got {type(batches).__name__}."
        )

    for idx, item in enumerate(batches):
        if not isinstance(item, FunctionBatch):
            raise ConcurrencyError(
                f"Item at index {idx} is not a FunctionBatch: got {type(item).__name__}."
            )

    if not callable(worker_fn):
        raise ConcurrencyError("worker_fn must be callable.")

    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency <= 0
    ):
        raise ConcurrencyError("max_concurrency must be an integer greater than zero.")

    if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
        raise ConcurrencyError("max_retries must be a non-negative integer.")

    if (
        not isinstance(initial_backoff, (int, float))
        or isinstance(initial_backoff, bool)
        or initial_backoff < 0
    ):
        raise ConcurrencyError("initial_backoff must be a non-negative number.")

    if (
        not isinstance(backoff_multiplier, (int, float))
        or isinstance(backoff_multiplier, bool)
        or backoff_multiplier < 1.0
    ):
        raise ConcurrencyError("backoff_multiplier must be a number >= 1.0.")

    if rate_limiter is not None and not isinstance(rate_limiter, RateLimiter):
        raise ConcurrencyError(
            f"rate_limiter must be a RateLimiter or None, got {type(rate_limiter).__name__}."
        )

    if not batches:
        logger.debug("run_batch_pool called with empty batches sequence; returning empty list.")
        return []

    worker_count = min(len(batches), max_concurrency)

    queue: asyncio.Queue[tuple[int, FunctionBatch]] = asyncio.Queue()
    for idx, batch in enumerate(batches):
        queue.put_nowait((idx, batch))

    indexed_results: list[tuple[int, BatchTaskResult[T]]] = []
    results_lock = asyncio.Lock()

    async def worker() -> None:
        while True:
            try:
                idx, batch = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                task_result = await _execute_with_retry(
                    batch=batch,
                    worker_fn=worker_fn,
                    rate_limiter=rate_limiter,
                    max_retries=max_retries,
                    initial_backoff=initial_backoff,
                    backoff_multiplier=backoff_multiplier,
                )
                async with results_lock:
                    indexed_results.append((idx, task_result))
            finally:
                queue.task_done()

    with logger.timed(
        "run_batch_pool",
        total_batches=len(batches),
        worker_count=worker_count,
        max_concurrency=max_concurrency,
    ) as metrics:
        logger.info(
            "Starting batch pool execution",
            extra={
                "total_batches": len(batches),
                "workers": worker_count,
                "max_concurrency": max_concurrency,
                "max_retries": max_retries,
                "preserve_order": preserve_order,
            },
        )

        workers = [
            asyncio.create_task(worker(), name=f"batch-pool-worker-{i}")
            for i in range(worker_count)
        ]

        try:
            await asyncio.gather(*workers)
        except BaseException:
            for w in workers:
                if not w.done():
                    w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise

        if preserve_order:
            indexed_results.sort(key=lambda item: item[0])

        results = [r for _, r in indexed_results]
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count

        metrics["successful_batches"] = successful_count
        metrics["failed_batches"] = failed_count

        logger.info(
            "Batch pool execution completed",
            extra={
                "total_batches": len(batches),
                "successful": successful_count,
                "failed": failed_count,
            },
        )

        return results


__all__ = [
    "BatchTaskResult",
    "run_batch_pool",
]
