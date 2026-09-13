"""Batch formation and partitioning logic for LLM documentation requests."""

import hashlib
from collections.abc import Sequence

from pydocgen.analysis.models import FunctionInfo
from pydocgen.batching.models import FunctionBatch
from pydocgen.batching.tokens import estimate_function_tokens
from pydocgen.config.settings import DEFAULT_PROMPT_OVERHEAD_TOKENS
from pydocgen.errors.exceptions import BatchError
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


def _generate_batch_id(index: int, functions: Sequence[FunctionInfo]) -> str:
    """Generate a deterministic batch identifier from batch index and function IDs.

    Args:
        index: 1-based sequential batch number.
        functions: Functions contained within the batch.

    Returns:
        Formatted identifier string (e.g. 'batch-001-a1b2c3d4').
    """
    if index < 1:
        raise ValueError(f"Batch index must be at least 1, got: {index}")

    combined_ids = ",".join(fn.id for fn in functions)
    content_hash = hashlib.sha256(combined_ids.encode("utf-8")).hexdigest()[:8]
    return f"batch-{index:03d}-{content_hash}"


def create_batches(
    functions: Sequence[FunctionInfo],
    max_batch_tokens: int = 12000,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    max_functions_per_batch: int | None = 25,
) -> list[FunctionBatch]:
    """Partition functions into batches respecting token and item limits.

    Args:
        functions: Sequence of undocumented FunctionInfo instances.
        max_batch_tokens: Hard ceiling for total estimated input tokens per request.
        prompt_overhead_tokens: Reserved budget for system prompts and schema wrappers.
        max_functions_per_batch: Optional cap on the number of functions in one batch.

    Returns:
        Ordered list of FunctionBatch instances.

    Raises:
        TypeError: If functions is not a sequence or contains non-FunctionInfo elements.
        BatchError: If parameters are invalid or a single function exceeds budget.
    """
    if max_batch_tokens <= 0:
        raise BatchError(f"max_batch_tokens ({max_batch_tokens}) must be greater than zero.")

    if prompt_overhead_tokens < 0:
        raise BatchError(f"prompt_overhead_tokens ({prompt_overhead_tokens}) cannot be negative.")

    usable_budget = max_batch_tokens - prompt_overhead_tokens
    if usable_budget <= 0:
        raise BatchError(
            f"max_batch_tokens ({max_batch_tokens}) must exceed "
            f"prompt_overhead_tokens ({prompt_overhead_tokens})."
        )

    if max_functions_per_batch is not None and max_functions_per_batch <= 0:
        raise BatchError(
            f"max_functions_per_batch ({max_functions_per_batch}) must be greater than zero."
        )

    if isinstance(functions, (str, bytes)) or not isinstance(functions, Sequence):
        raise TypeError(f"functions must be a sequence, got: {type(functions).__name__}")

    for fn in functions:
        if not isinstance(fn, FunctionInfo):
            raise TypeError(
                f"All items in functions must be FunctionInfo instances, got: {type(fn).__name__}"
            )

    if not functions:
        return []

    batches: list[FunctionBatch] = []
    current_functions: list[FunctionInfo] = []
    current_tokens = prompt_overhead_tokens

    def flush_current_batch() -> None:
        nonlocal current_tokens
        if not current_functions:
            return
        batch_id = _generate_batch_id(len(batches) + 1, current_functions)
        batch = FunctionBatch(
            id=batch_id,
            functions=tuple(current_functions),
            estimated_input_tokens=current_tokens,
        )
        batches.append(batch)
        logger.debug(
            "Created function batch",
            extra={
                "batch_id": batch_id,
                "functions_in_batch": len(batch.functions),
                "estimated_tokens": current_tokens,
            },
        )
        current_functions.clear()
        current_tokens = prompt_overhead_tokens

    with logger.timed(
        "create_batches",
        total_functions=len(functions),
        max_batch_tokens=max_batch_tokens,
        max_functions_per_batch=max_functions_per_batch,
    ) as metrics:
        for func in functions:
            func_tokens = estimate_function_tokens(func)

            if func_tokens > usable_budget:
                logger.error(
                    "Function token estimate exceeds batch budget",
                    extra={
                        "function": func.qualified_name,
                        "func_tokens": func_tokens,
                        "usable_budget": usable_budget,
                    },
                )
                raise BatchError(
                    f"Function '{func.qualified_name}' requires ~{func_tokens} tokens, "
                    f"which exceeds the maximum usable batch budget of {usable_budget} tokens."
                )

            token_overflow = (current_tokens + func_tokens) > max_batch_tokens
            count_overflow = (
                max_functions_per_batch is not None
                and len(current_functions) >= max_functions_per_batch
            )

            if current_functions and (token_overflow or count_overflow):
                flush_current_batch()

            current_functions.append(func)
            current_tokens += func_tokens

        flush_current_batch()
        metrics["batches_count"] = len(batches)
        logger.info(
            "Completed batch creation",
            extra={"batch_count": len(batches), "total_functions": len(functions)},
        )

    return batches


__all__ = ["create_batches"]
