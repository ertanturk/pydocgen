"""Data models for batched function documentation requests."""

from collections.abc import Iterator
from dataclasses import dataclass

from pydocgen.analysis.models import FunctionInfo


@dataclass(frozen=True)
class FunctionBatch:
    """Represents a discrete batch of functions to be processed together by an LLM."""

    id: str
    functions: tuple[FunctionInfo, ...]
    estimated_input_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Batch id must be a non-empty string.")

        if isinstance(self.functions, list):
            object.__setattr__(self, "functions", tuple(self.functions))

        if not isinstance(self.functions, tuple):
            raise TypeError(f"functions must be a tuple, got: {type(self.functions).__name__}")

        for fn in self.functions:
            if not isinstance(fn, FunctionInfo):
                raise TypeError(
                    f"All functions in batch must be FunctionInfo instances, got: {type(fn).__name__}"
                )

        if not isinstance(self.estimated_input_tokens, int) or self.estimated_input_tokens < 0:
            raise ValueError("estimated_input_tokens must be a non-negative integer.")

    def __len__(self) -> int:
        return len(self.functions)

    def __iter__(self) -> Iterator[FunctionInfo]:
        return iter(self.functions)

    def __getitem__(self, index: int) -> FunctionInfo:
        return self.functions[index]

    @property
    def size(self) -> int:
        """Return the number of functions in the batch."""
        return len(self.functions)

    @property
    def function_ids(self) -> tuple[str, ...]:
        """Return the sequence of function IDs contained in this batch."""
        return tuple(fn.id for fn in self.functions)


__all__ = ["FunctionBatch"]
