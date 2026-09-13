"""Data models representing Python functions, methods, and parameters."""

from dataclasses import dataclass
from enum import StrEnum


class ParameterKind(StrEnum):
    """Enumeration of parameter kinds matching Python argument conventions."""

    POSITIONAL_ONLY = "positional_only"
    POSITIONAL_OR_KEYWORD = "positional_or_keyword"
    VAR_POSITIONAL = "var_positional"
    KEYWORD_ONLY = "keyword_only"
    VAR_KEYWORD = "var_keyword"


@dataclass(frozen=True)
class ParameterInfo:
    """Represents a single parameter in a function signature."""

    name: str
    annotation: str | None = None
    default: str | None = None
    kind: ParameterKind = ParameterKind.POSITIONAL_OR_KEYWORD

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a non-empty string.")
        if isinstance(self.kind, str) and not isinstance(self.kind, ParameterKind):
            object.__setattr__(self, "kind", ParameterKind(self.kind))


@dataclass(frozen=True)
class FunctionInfo:
    """Represents an extracted Python function or method with metadata."""

    id: str
    name: str
    qualified_name: str
    source: str
    parameters: tuple[ParameterInfo, ...]
    return_annotation: str | None
    decorators: tuple[str, ...]
    line_start: int
    line_end: int
    has_docstring: bool
    is_async: bool
    is_method: bool
    docstring: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.parameters, list):
            object.__setattr__(self, "parameters", tuple(self.parameters))
        if isinstance(self.decorators, list):
            object.__setattr__(self, "decorators", tuple(self.decorators))


__all__ = [
    "FunctionInfo",
    "ParameterInfo",
    "ParameterKind",
]
