"""Provider implementations for interacting with generative AI backends."""

from __future__ import annotations

from pydocgen.providers.gemini import (
    DEFAULT_MODEL,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_TIMEOUT,
    FAST_FAIL_PROBE_PROMPT,
    GeminiProvider,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROBE_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "FAST_FAIL_PROBE_PROMPT",
    "GeminiProvider",
]
