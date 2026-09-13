from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceEdit:
    """Represents an atomic docstring insertion into a source file."""

    function_id: str
    qualified_name: str
    insert_after_line: int  # 1-based line number after which to insert
    indentation: str  # Detected indentation of the function body
    docstring: str  # Formatted docstring block
    replace_line_content: str | None = (
        None  # Replacement for header line if single-line function is split
    )
