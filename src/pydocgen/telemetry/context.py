"""Contextual variable tracking for telemetry across async and sync execution."""

import contextvars
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "telemetry_context", default=None
)


def get_context() -> dict[str, Any]:
    """Return a shallow copy of the current context dictionary."""
    ctx = _CONTEXT.get()
    return dict(ctx) if ctx is not None else {}


def set_context(context: dict[str, Any]) -> None:
    """Set the context dictionary."""
    _CONTEXT.set(dict(context))


def update_context(**kwargs: Any) -> None:
    """Update current context with provided key-value pairs."""
    ctx = get_context()
    ctx.update(kwargs)
    _CONTEXT.set(ctx)


def clear_context() -> None:
    """Reset context to an empty dictionary."""
    _CONTEXT.set({})


@contextmanager
def bind_context(**kwargs: Any) -> Generator[None]:
    """Context manager to temporarily bind key-value pairs to the context.

    Args:
        **kwargs: Key-value attributes to bind for the duration of the block.
    """
    merged = {**get_context(), **kwargs}
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


__all__ = [
    "bind_context",
    "clear_context",
    "get_context",
    "set_context",
    "update_context",
]
