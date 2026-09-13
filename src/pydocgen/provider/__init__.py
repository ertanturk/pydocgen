"""Backward-compatibility alias for pydocgen.providers."""

from __future__ import annotations

from pydocgen.providers import *  # noqa: F403
from pydocgen.providers import __all__

__all__ = list(__all__)
