"""Token estimation heuristics for Python code and function prompts."""

import math

from pydocgen.analysis.models import FunctionInfo
from pydocgen.config.settings import (
    CHARS_PER_TOKEN,
    PER_FUNCTION_OVERHEAD_TOKENS,
)


def estimate_text_tokens(text: str, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Estimate token count for a text string using code-tailored heuristic.

    Args:
        text: Raw string to measure.
        chars_per_token: Average characters per token (default: 3.5 for code).

    Returns:
        Estimated token count (0 for empty string, minimum 1 for non-empty text).

    Raises:
        TypeError: If text is not a string.
        ValueError: If chars_per_token is not positive.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text as string, got: {type(text).__name__}")

    if chars_per_token <= 0:
        raise ValueError(f"chars_per_token must be positive, got: {chars_per_token}")

    if not text:
        return 0

    return max(1, math.ceil(len(text) / chars_per_token))


def estimate_function_tokens(
    function: FunctionInfo, overhead_tokens: int = PER_FUNCTION_OVERHEAD_TOKENS
) -> int:
    """Estimate prompt tokens required for a function and its metadata payload.

    Args:
        function: Target FunctionInfo instance.
        overhead_tokens: Fixed structural overhead per function in prompt.

    Returns:
        Total estimated token weight.

    Raises:
        TypeError: If function is not a FunctionInfo instance.
        ValueError: If overhead_tokens is negative.
    """
    if not isinstance(function, FunctionInfo):
        raise TypeError(f"Expected FunctionInfo instance, got: {type(function).__name__}")

    if overhead_tokens < 0:
        raise ValueError(f"overhead_tokens cannot be negative, got: {overhead_tokens}")

    source_tokens = estimate_text_tokens(function.source)
    return source_tokens + overhead_tokens


__all__ = [
    "estimate_function_tokens",
    "estimate_text_tokens",
]
