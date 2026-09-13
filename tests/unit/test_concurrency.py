"""Unit tests for the concurrency package (RateLimiter, BatchTaskResult, run_batch_pool)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import FrozenInstanceError

import pytest

from pydocgen.batching.models import FunctionBatch
from pydocgen.concurrency import (
    BatchTaskResult,
    ConcurrencyError,
    RateLimiter,
    run_batch_pool,
)
from pydocgen.concurrency.executor import _execute_with_retry


def _make_dummy_batch(batch_id: str = "batch-001") -> FunctionBatch:
    """Helper to construct dummy FunctionBatch instances for testing."""
    return FunctionBatch(id=batch_id, functions=(), estimated_input_tokens=100)


# ==============================================================================
# RateLimiter Tests
# ==============================================================================


class TestRateLimiter:
    """Tests for RateLimiter sliding-window rate limiting."""

    def test_initialization_defaults(self) -> None:
        limiter = RateLimiter()
        assert limiter.rpm == 60
        assert limiter.window_seconds == 60.0
        assert limiter.active_requests == 0

    def test_initialization_custom(self) -> None:
        limiter = RateLimiter(requests_per_minute=120, window_seconds=30.0)
        assert limiter.rpm == 120
        assert limiter.window_seconds == 30.0
        assert limiter.active_requests == 0

    @pytest.mark.parametrize("invalid_rpm", [-1, -100, True, False, 1.5, "60"])
    def test_initialization_invalid_rpm(self, invalid_rpm: object) -> None:
        with pytest.raises(
            ConcurrencyError, match="requests_per_minute must be a non-negative integer"
        ):
            RateLimiter(requests_per_minute=invalid_rpm)  # ty: ignore[invalid-argument-type]

    @pytest.mark.parametrize("invalid_window", [0, -1, -0.5, True, False, "60"])
    def test_initialization_invalid_window(self, invalid_window: object) -> None:
        with pytest.raises(ConcurrencyError, match="window_seconds must be a positive number"):
            RateLimiter(requests_per_minute=60, window_seconds=invalid_window)  # ty: ignore[invalid-argument-type]

    async def test_zero_rpm_disables_limiting(self) -> None:
        limiter = RateLimiter(requests_per_minute=0)
        start = time.monotonic()
        for _ in range(10):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert limiter.active_requests == 0

    async def test_burst_within_rpm_limit(self) -> None:
        limiter = RateLimiter(requests_per_minute=3, window_seconds=0.1)
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.05
        assert limiter.active_requests == 3

    async def test_throttles_requests_exceeding_rpm(self) -> None:
        limiter = RateLimiter(requests_per_minute=2, window_seconds=0.1)
        start = time.monotonic()
        # Acquire slot 1
        await limiter.acquire()
        assert limiter.active_requests == 1

        # Stagger slot 2 so it doesn't expire simultaneously with slot 1
        await asyncio.sleep(0.04)
        await limiter.acquire()
        assert limiter.active_requests == 2

        # 3rd request must wait until slot 1 expires (~0.1s from start)
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed >= 0.09
        # Slot 1 expired, but slot 2 and slot 3 are active
        assert limiter.active_requests == 2

    async def test_active_requests_eviction(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, window_seconds=0.04)
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.active_requests == 2
        await asyncio.sleep(0.06)
        assert limiter.active_requests == 0

    async def test_reset(self) -> None:
        limiter = RateLimiter(requests_per_minute=5, window_seconds=1.0)
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.active_requests == 2
        limiter.reset()
        assert limiter.active_requests == 0

    async def test_concurrent_acquisitions(self) -> None:
        limiter = RateLimiter(requests_per_minute=2, window_seconds=0.06)

        async def worker() -> float:
            await limiter.acquire()
            return time.monotonic()

        start = time.monotonic()
        timestamps = await asyncio.gather(*[worker() for _ in range(4)])
        elapsed = time.monotonic() - start

        # 4 requests with rpm=2 over 0.06s window should take at least ~0.06s
        assert elapsed >= 0.05
        assert len(timestamps) == 4

    async def test_cancellation_releases_lock(self) -> None:
        limiter = RateLimiter(requests_per_minute=1, window_seconds=0.1)
        await limiter.acquire()

        # Task 2 will wait in acquire() and get cancelled
        async def waiting_task() -> None:
            await limiter.acquire()

        task = asyncio.create_task(waiting_task())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Task 3 should be able to acquire once window passes without deadlock
        await asyncio.sleep(0.1)
        await asyncio.wait_for(limiter.acquire(), timeout=0.1)


# ==============================================================================
# BatchTaskResult Tests
# ==============================================================================


class TestBatchTaskResult:
    """Tests for BatchTaskResult dataclass behavior and invariants."""

    def test_successful_result(self) -> None:
        result = BatchTaskResult(batch_id="batch-001", success=True, data={"doc": "test"})
        assert result.batch_id == "batch-001"
        assert result.success is True
        assert result.data == {"doc": "test"}
        assert result.error is None
        assert result.error_message is None

    def test_failed_result(self) -> None:
        exc = ValueError("Model rate limit exceeded")
        result: BatchTaskResult[None] = BatchTaskResult(
            batch_id="batch-002",
            success=False,
            error=exc,
        )
        assert result.batch_id == "batch-002"
        assert result.success is False
        assert result.data is None
        assert result.error is exc
        assert result.error_message == "Model rate limit exceeded"

    def test_immutability(self) -> None:
        result = BatchTaskResult(batch_id="batch-001", success=True, data="ok")
        with pytest.raises(FrozenInstanceError):
            result.data = "changed"  # ty: ignore[invalid-assignment]

    @pytest.mark.parametrize("bad_id", ["", "   ", None, 123])
    def test_invalid_batch_id(self, bad_id: object) -> None:
        with pytest.raises(ConcurrencyError, match="batch_id must be a non-empty string"):
            BatchTaskResult(batch_id=bad_id, success=True)  # ty: ignore[invalid-argument-type]

    def test_success_with_error_rejected(self) -> None:
        with pytest.raises(
            ConcurrencyError, match="Successful BatchTaskResult cannot contain an error"
        ):
            BatchTaskResult(batch_id="batch-001", success=True, error=RuntimeError("oops"))

    def test_failure_without_error_rejected(self) -> None:
        with pytest.raises(ConcurrencyError, match="Failed BatchTaskResult must contain an error"):
            BatchTaskResult(batch_id="batch-001", success=False, error=None)


# ==============================================================================
# _execute_with_retry Tests
# ==============================================================================


class TestExecuteWithRetry:
    """Tests for single batch retry logic."""

    async def test_success_on_first_attempt(self) -> None:
        batch = _make_dummy_batch("batch-001")
        calls = 0

        async def worker(b: FunctionBatch) -> str:
            nonlocal calls
            calls += 1
            return f"processed {b.id}"

        result = await _execute_with_retry(
            batch=batch,
            worker_fn=worker,
            rate_limiter=None,
        )
        assert result.success is True
        assert result.data == "processed batch-001"
        assert result.error is None
        assert calls == 1

    async def test_transient_failure_recovered(self) -> None:
        batch = _make_dummy_batch("batch-001")
        calls = 0

        async def flaking_worker(b: FunctionBatch) -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ConnectionResetError(f"transient error attempt {calls}")
            return f"recovered on attempt {calls}"

        result = await _execute_with_retry(
            batch=batch,
            worker_fn=flaking_worker,
            rate_limiter=None,
            max_retries=3,
            initial_backoff=0.001,
            backoff_multiplier=1.5,
        )
        assert result.success is True
        assert result.data == "recovered on attempt 3"
        assert calls == 3

    async def test_exhaust_all_retries(self) -> None:
        batch = _make_dummy_batch("batch-001")
        calls = 0

        async def failing_worker(b: FunctionBatch) -> str:
            nonlocal calls
            calls += 1
            raise ValueError(f"permanent failure {calls}")

        result = await _execute_with_retry(
            batch=batch,
            worker_fn=failing_worker,
            rate_limiter=None,
            max_retries=2,
            initial_backoff=0.001,
            backoff_multiplier=2.0,
        )
        assert result.success is False
        assert isinstance(result.error, ValueError)
        assert "permanent failure 3" in str(result.error)
        assert calls == 3  # initial + 2 retries

    async def test_rate_limiter_integration(self) -> None:
        batch = _make_dummy_batch("batch-001")
        limiter = RateLimiter(requests_per_minute=10, window_seconds=1.0)

        async def worker(b: FunctionBatch) -> str:
            return "done"

        result = await _execute_with_retry(
            batch=batch,
            worker_fn=worker,
            rate_limiter=limiter,
        )
        assert result.success is True
        assert limiter.active_requests == 1

    async def test_invalid_parameters(self) -> None:
        batch = _make_dummy_batch("batch-001")

        async def worker(b: FunctionBatch) -> str:
            return "ok"

        with pytest.raises(ConcurrencyError, match="batch must be a FunctionBatch"):
            await _execute_with_retry(
                batch="not-a-batch",  # ty: ignore[invalid-argument-type]
                worker_fn=worker,
                rate_limiter=None,
            )

        with pytest.raises(ConcurrencyError, match="worker_fn must be callable"):
            await _execute_with_retry(
                batch=batch,
                worker_fn="not-callable",  # ty: ignore[invalid-argument-type]
                rate_limiter=None,
            )

        with pytest.raises(ConcurrencyError, match="max_retries must be a non-negative integer"):
            await _execute_with_retry(
                batch=batch,
                worker_fn=worker,
                rate_limiter=None,
                max_retries=-1,
            )

        with pytest.raises(ConcurrencyError, match="initial_backoff must be a non-negative number"):
            await _execute_with_retry(
                batch=batch,
                worker_fn=worker,
                rate_limiter=None,
                initial_backoff=-0.5,
            )

        with pytest.raises(ConcurrencyError, match="backoff_multiplier must be a number >= 1.0"):
            await _execute_with_retry(
                batch=batch,
                worker_fn=worker,
                rate_limiter=None,
                backoff_multiplier=0.5,
            )

        with pytest.raises(ConcurrencyError, match="rate_limiter must be a RateLimiter"):
            await _execute_with_retry(
                batch=batch,
                worker_fn=worker,
                rate_limiter="invalid",  # ty: ignore[invalid-argument-type]
            )


# ==============================================================================
# run_batch_pool Tests
# ==============================================================================


class TestRunBatchPool:
    """Tests for run_batch_pool concurrent execution and error handling."""

    async def test_empty_batches_returns_empty_list(self) -> None:
        async def dummy_worker(batch: FunctionBatch) -> str:
            return "done"

        results = await run_batch_pool(batches=[], worker_fn=dummy_worker)
        assert results == []

    async def test_single_batch_success(self) -> None:
        batch = _make_dummy_batch("batch-001")

        async def worker(b: FunctionBatch) -> dict[str, str]:
            return {"batch": b.id, "status": "processed"}

        results = await run_batch_pool(batches=[batch], worker_fn=worker)
        assert len(results) == 1
        assert results[0].batch_id == "batch-001"
        assert results[0].success is True
        assert results[0].data == {"batch": "batch-001", "status": "processed"}

    async def test_concurrency_bounding(self) -> None:
        batches = [_make_dummy_batch(f"batch-{i:03d}") for i in range(6)]
        current_active = 0
        peak_active = 0
        active_lock = asyncio.Lock()

        async def worker(b: FunctionBatch) -> str:
            nonlocal current_active, peak_active
            async with active_lock:
                current_active += 1
                if current_active > peak_active:
                    peak_active = current_active
            await asyncio.sleep(0.02)
            async with active_lock:
                current_active -= 1
            return b.id

        results = await run_batch_pool(
            batches=batches,
            worker_fn=worker,
            max_concurrency=2,
        )
        assert len(results) == 6
        assert peak_active <= 2
        assert all(r.success for r in results)

    async def test_mixed_success_and_failure(self) -> None:
        batches = [_make_dummy_batch(f"batch-{i}") for i in range(4)]

        async def flaky_worker(b: FunctionBatch) -> str:
            if b.id in {"batch-1", "batch-3"}:
                raise RuntimeError(f"error in {b.id}")
            return f"data for {b.id}"

        results = await run_batch_pool(
            batches=batches,
            worker_fn=flaky_worker,
            max_concurrency=2,
            max_retries=1,
            initial_backoff=0.001,
        )
        assert len(results) == 4
        successes = [r for r in results if r.success]
        failures = [r for r in results if not r.success]

        assert len(successes) == 2
        assert len(failures) == 2
        assert {r.batch_id for r in successes} == {"batch-0", "batch-2"}
        assert {r.batch_id for r in failures} == {"batch-1", "batch-3"}
        for f in failures:
            assert f.error is not None
            assert f"error in {f.batch_id}" in str(f.error)

    async def test_preserve_order_true(self) -> None:
        batches = [_make_dummy_batch(f"batch-{i}") for i in range(4)]

        # Intentionally make earlier batches slower so they finish later
        delays = {"batch-0": 0.04, "batch-1": 0.03, "batch-2": 0.02, "batch-3": 0.01}

        async def delayed_worker(b: FunctionBatch) -> str:
            await asyncio.sleep(delays[b.id])
            return b.id

        results = await run_batch_pool(
            batches=batches,
            worker_fn=delayed_worker,
            max_concurrency=4,
            preserve_order=True,
        )
        returned_ids = [r.batch_id for r in results]
        assert returned_ids == ["batch-0", "batch-1", "batch-2", "batch-3"]

    async def test_preserve_order_false(self) -> None:
        batches = [_make_dummy_batch("slow"), _make_dummy_batch("fast")]

        async def worker(b: FunctionBatch) -> str:
            if b.id == "slow":
                await asyncio.sleep(0.04)
            else:
                await asyncio.sleep(0.005)
            return b.id

        results = await run_batch_pool(
            batches=batches,
            worker_fn=worker,
            max_concurrency=2,
            preserve_order=False,
        )
        returned_ids = [r.batch_id for r in results]
        assert returned_ids == ["fast", "slow"]

    async def test_rate_limiter_integration(self) -> None:
        batches = [_make_dummy_batch(f"batch-{i}") for i in range(3)]
        limiter = RateLimiter(requests_per_minute=2, window_seconds=0.05)

        async def worker(b: FunctionBatch) -> str:
            return b.id

        start = time.monotonic()
        results = await run_batch_pool(
            batches=batches,
            worker_fn=worker,
            max_concurrency=3,
            rate_limiter=limiter,
        )
        elapsed = time.monotonic() - start
        assert len(results) == 3
        assert elapsed >= 0.04

    async def test_worker_cancellation_cleanup(self) -> None:
        batches = [_make_dummy_batch(f"batch-{i}") for i in range(5)]

        async def hanging_worker(b: FunctionBatch) -> str:
            await asyncio.sleep(10.0)
            return b.id

        pool_task = asyncio.create_task(
            run_batch_pool(
                batches=batches,
                worker_fn=hanging_worker,
                max_concurrency=3,
            )
        )
        await asyncio.sleep(0.01)
        pool_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await pool_task

        # Verify no batch-pool-worker tasks remain pending
        active_pool_workers = [
            t
            for t in asyncio.all_tasks()
            if t.get_name().startswith("batch-pool-worker-") and not t.done()
        ]
        assert len(active_pool_workers) == 0

    @pytest.mark.parametrize(
        ("bad_batches", "match"),
        [
            ("invalid_string", "batches must be a Sequence"),
            (b"invalid_bytes", "batches must be a Sequence"),
            (12345, "batches must be a Sequence"),
            ([_make_dummy_batch(), "not_a_batch"], "is not a FunctionBatch"),
        ],
    )
    async def test_invalid_batches(self, bad_batches: object, match: str) -> None:
        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(ConcurrencyError, match=match):
            await run_batch_pool(
                batches=bad_batches,  # ty: ignore[invalid-argument-type]
                worker_fn=dummy_worker,
            )

    async def test_invalid_worker_fn(self) -> None:
        batches = [_make_dummy_batch()]
        with pytest.raises(ConcurrencyError, match="worker_fn must be callable"):
            await run_batch_pool(
                batches=batches,
                worker_fn="not-callable",  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("bad_concurrency", [0, -1, -5, True, False, 1.5, "3"])
    async def test_invalid_max_concurrency(self, bad_concurrency: object) -> None:
        batches = [_make_dummy_batch()]

        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(
            ConcurrencyError, match="max_concurrency must be an integer greater than zero"
        ):
            await run_batch_pool(
                batches=batches,
                worker_fn=dummy_worker,
                max_concurrency=bad_concurrency,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("bad_retries", [-1, True, False, "2", 1.5])
    async def test_invalid_max_retries(self, bad_retries: object) -> None:
        batches = [_make_dummy_batch()]

        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(ConcurrencyError, match="max_retries must be a non-negative integer"):
            await run_batch_pool(
                batches=batches,
                worker_fn=dummy_worker,
                max_retries=bad_retries,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("bad_backoff", [-0.1, -1, True, False, "1.0"])
    async def test_invalid_initial_backoff(self, bad_backoff: object) -> None:
        batches = [_make_dummy_batch()]

        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(ConcurrencyError, match="initial_backoff must be a non-negative number"):
            await run_batch_pool(
                batches=batches,
                worker_fn=dummy_worker,
                initial_backoff=bad_backoff,  # ty: ignore[invalid-argument-type]
            )

    @pytest.mark.parametrize("bad_multiplier", [0.9, 0.0, -1.0, True, False, "2.0"])
    async def test_invalid_backoff_multiplier(self, bad_multiplier: object) -> None:
        batches = [_make_dummy_batch()]

        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(ConcurrencyError, match="backoff_multiplier must be a number >= 1.0"):
            await run_batch_pool(
                batches=batches,
                worker_fn=dummy_worker,
                backoff_multiplier=bad_multiplier,  # ty: ignore[invalid-argument-type]
            )

    async def test_invalid_rate_limiter(self) -> None:
        batches = [_make_dummy_batch()]

        async def dummy_worker(b: FunctionBatch) -> str:
            return b.id

        with pytest.raises(ConcurrencyError, match="rate_limiter must be a RateLimiter"):
            await run_batch_pool(
                batches=batches,
                worker_fn=dummy_worker,
                rate_limiter="invalid",  # ty: ignore[invalid-argument-type]
            )


# ==============================================================================
# Concurrency Logging & Telemetry Tests
# ==============================================================================


class TestConcurrencyTelemetry:
    """Tests for logging and telemetry events in concurrency execution."""

    async def test_rate_limiter_throttling_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)
        limiter = RateLimiter(requests_per_minute=1, window_seconds=0.03)

        await limiter.acquire()
        await limiter.acquire()

        throttle_records = [r for r in caplog.records if "Rate limit ceiling reached" in r.message]
        assert len(throttle_records) >= 1
        assert throttle_records[0].levelno == logging.WARNING

    async def test_batch_pool_execution_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO)
        batch = _make_dummy_batch("batch-001")

        async def worker(b: FunctionBatch) -> str:
            return "done"

        await run_batch_pool(batches=[batch], worker_fn=worker)

        start_records = [r for r in caplog.records if "Starting batch pool execution" in r.message]
        assert len(start_records) == 1

        completed_records = [
            r for r in caplog.records if "Batch pool execution completed" in r.message
        ]
        assert len(completed_records) == 1

    async def test_retry_failure_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING)
        batch = _make_dummy_batch("batch-err")

        async def failing_worker(b: FunctionBatch) -> str:
            raise RuntimeError("forced test failure")

        await run_batch_pool(
            batches=[batch],
            worker_fn=failing_worker,
            max_retries=1,
            initial_backoff=0.001,
        )

        retry_warnings = [
            r
            for r in caplog.records
            if "Batch task execution failed; retrying after backoff" in r.message
        ]
        assert len(retry_warnings) == 1

        exhaust_errors = [
            r
            for r in caplog.records
            if "Batch task failed after exhausting all retry attempts" in r.message
        ]
        assert len(exhaust_errors) == 1


# ==============================================================================
# Package Exports Tests
# ==============================================================================


class TestPackageExports:
    """Tests for package exports in pydocgen.concurrency."""

    def test_concurrency_exports(self) -> None:
        import pydocgen.concurrency as pkg

        assert hasattr(pkg, "BatchTaskResult")
        assert hasattr(pkg, "RateLimiter")
        assert hasattr(pkg, "run_batch_pool")
        assert hasattr(pkg, "ConcurrencyError")
        assert set(pkg.__all__) == {
            "BatchTaskResult",
            "ConcurrencyError",
            "RateLimiter",
            "run_batch_pool",
        }
