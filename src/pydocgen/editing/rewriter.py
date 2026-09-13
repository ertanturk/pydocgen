"""Source code rewriter applying docstring edits atomically with diff previews."""

from __future__ import annotations

import difflib
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from pydocgen.analysis.models import FunctionInfo
from pydocgen.config.settings import (
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_RESET,
    CONFIRMATION_AFFIRMATIVE,
    DEFAULT_INDENTATION,
    DEFAULT_WRITE_ENCODING,
    DIFF_SEPARATOR_CHAR,
    DIFF_SEPARATOR_LENGTH,
    TEMP_FILE_SUFFIX,
)
from pydocgen.editing.models import SourceEdit
from pydocgen.errors.exceptions import SourceEditError
from pydocgen.formatting.google import GoogleFormatter
from pydocgen.generation.models import GeneratedDocumentation
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


def find_insertion_point(source_lines: list[str], function: FunctionInfo) -> tuple[int, str]:
    """Locate the exact line index and body indentation for docstring insertion.

    Scans from the function's start line past decorators until reaching the 'def'
    or 'async def' declaration, then searches for the closing colon (':') while
    safely handling multiline parameter signatures, strings, and comments.

    Args:
        source_lines: All lines of the source file (0-indexed list).
        function: FunctionInfo metadata.

    Returns:
        Tuple of (insert_after_line_1_based, body_indentation_string).

    Raises:
        SourceEditError: If the function header closing colon cannot be located.
    """
    start_idx = max(0, function.line_start - 1)
    colon_line_idx: int | None = None
    found_def = False
    paren_depth = 0
    in_quote: str | None = None

    for idx in range(start_idx, len(source_lines)):
        line = source_lines[idx]
        stripped = line.strip()

        # 1. Skip leading decorators or comments before the 'def' declaration
        if not found_def:
            if (
                stripped.startswith("def ")
                or stripped.startswith("async def ")
                or " def " in line
                or " async def " in line
            ):
                found_def = True
            else:
                continue

        # 2. Parse signature line characters ignoring strings and comments
        col = 0
        line_len = len(line)
        while col < line_len:
            char = line[col]

            # Check for comments outside strings
            if in_quote is None and char == "#":
                break

            # Check for triple quotes
            if col + 2 < line_len and line[col : col + 3] in ('"""', "'''"):
                q_token = line[col : col + 3]
                if in_quote is None:
                    in_quote = q_token
                elif in_quote == q_token:
                    in_quote = None
                col += 3
                continue

            # Check for single quotes
            if char in ("'", '"'):
                if in_quote is None:
                    in_quote = char
                elif in_quote == char:
                    # Check if escaped
                    backslashes = 0
                    b_idx = col - 1
                    while b_idx >= 0 and line[b_idx] == "\\":
                        backslashes += 1
                        b_idx -= 1
                    if backslashes % 2 == 0:
                        in_quote = None
                col += 1
                continue

            if in_quote is None:
                if char in "([{":
                    paren_depth += 1
                elif char in ")]}":
                    paren_depth = max(0, paren_depth - 1)
                elif char == ":" and paren_depth == 0:
                    colon_line_idx = idx
                    break

            col += 1

        if colon_line_idx is not None:
            break

    if colon_line_idx is None:
        raise SourceEditError(
            f"Could not find closing ':' for function '{function.qualified_name}' "
            f"starting at line {function.line_start}."
        )

    # Detect body indentation from next non-empty line or default to signature + default indentation
    def_line = source_lines[colon_line_idx]
    def_indent = def_line[: len(def_line) - len(def_line.lstrip())]
    default_indent = def_indent + DEFAULT_INDENTATION

    body_indent = default_indent
    for next_idx in range(colon_line_idx + 1, len(source_lines)):
        next_line = source_lines[next_idx]
        if next_line.strip():
            detected = next_line[: len(next_line) - len(next_line.lstrip())]
            if len(detected) > len(def_indent):
                body_indent = detected
            break

    logger.debug(
        "Located insertion point",
        extra={
            "function": function.qualified_name,
            "insert_after_line": colon_line_idx + 1,
            "indentation": repr(body_indent),
        },
    )
    return colon_line_idx + 1, body_indent


def create_source_edits(
    source: str,
    functions: Sequence[FunctionInfo],
    documentations: Sequence[GeneratedDocumentation],
    formatter: GoogleFormatter | None = None,
) -> list[SourceEdit]:
    """Calculate SourceEdit objects for all accepted function documentations.

    Handles both standard multiline functions and single-line functions
    (e.g., 'def foo(): pass') by automatically formatting and indenting
    the trailing statements.

    Args:
        source: The original source code string.
        functions: Sequence of AST FunctionInfo targets.
        documentations: Validated GeneratedDocumentation objects.
        formatter: Formatter instance (defaults to GoogleFormatter).

    Returns:
        List of calculated SourceEdit objects.
    """
    fmt = formatter or GoogleFormatter()
    source_lines = source.splitlines(keepends=True)
    fn_map = {fn.id: fn for fn in functions}
    edits: list[SourceEdit] = []

    with logger.timed("create_source_edits", doc_count=len(documentations)):
        for doc in documentations:
            if doc.status == "uncertain":
                continue

            fn = fn_map.get(doc.function_id)
            if not fn:
                continue

            insert_after, indent = find_insertion_point(source_lines, fn)
            formatted_doc = fmt.format(doc, function=fn, base_indent=indent)

            # Check if this is a single-line function with inline code after ':'
            colon_line_idx = insert_after - 1
            colon_line = source_lines[colon_line_idx]
            colon_pos = colon_line.rfind(":")
            remainder = colon_line[colon_pos + 1 :].strip() if colon_pos != -1 else ""

            if remainder and not remainder.startswith("#"):
                # Split single line into header + docstring + body
                line_end = "\n" if colon_line.endswith("\n") else ""
                replace_line = colon_line[: colon_pos + 1] + line_end
                docstring_and_body = f"{formatted_doc}\n{indent}{remainder}"
                edits.append(
                    SourceEdit(
                        function_id=fn.id,
                        qualified_name=fn.qualified_name,
                        insert_after_line=insert_after,
                        indentation=indent,
                        docstring=docstring_and_body,
                        replace_line_content=replace_line,
                    )
                )
            else:
                edits.append(
                    SourceEdit(
                        function_id=fn.id,
                        qualified_name=fn.qualified_name,
                        insert_after_line=insert_after,
                        indentation=indent,
                        docstring=formatted_doc,
                    )
                )

    logger.info("Created source edits", extra={"edits_count": len(edits)})
    return edits


def apply_edits(source: str, edits: Sequence[SourceEdit]) -> str:
    """Apply docstring edits in descending line order to prevent offset drift.

    Args:
        source: Original file source code.
        edits: Collection of SourceEdit operations.

    Returns:
        The updated source code string.
    """
    if not edits:
        return source

    lines = source.splitlines(keepends=True)

    with logger.timed("apply_edits", edit_count=len(edits)):
        # Sort strictly descending by insert_after_line (bottom-to-top)
        sorted_edits = sorted(edits, key=lambda e: e.insert_after_line, reverse=True)

        for edit in sorted_edits:
            insert_idx = edit.insert_after_line

            if edit.replace_line_content is not None:
                target_idx = insert_idx - 1
                if 0 <= target_idx < len(lines):
                    lines[target_idx] = edit.replace_line_content

            # Ensure docstring ends cleanly with a newline
            content = edit.docstring if edit.docstring.endswith("\n") else f"{edit.docstring}\n"
            lines.insert(insert_idx, content)

    return "".join(lines)


def generate_diff(
    original: str,
    modified: str,
    filename: str = "source.py",
    colorize: bool = False,
) -> str:
    """Generate a unified diff representation between original and modified source.

    Args:
        original: Unmodified source text.
        modified: Updated source text with docstrings.
        filename: Label for diff header.
        colorize: Whether to include ANSI color codes.

    Returns:
        Unified diff as a single string.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff_iter = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )

    if not colorize:
        return "".join(diff_iter)

    colored_lines: list[str] = []
    for line in diff_iter:
        if line.startswith("---") or line.startswith("+++"):
            colored_lines.append(f"{COLOR_BOLD}{line}{COLOR_RESET}")
        elif line.startswith("@@"):
            colored_lines.append(f"{COLOR_CYAN}{line}{COLOR_RESET}")
        elif line.startswith("+"):
            colored_lines.append(f"{COLOR_GREEN}{line}{COLOR_RESET}")
        elif line.startswith("-"):
            colored_lines.append(f"{COLOR_RED}{line}{COLOR_RESET}")
        else:
            colored_lines.append(line)

    return "".join(colored_lines)


def write_file_atomically(path: Path, content: str) -> None:
    """Write content to a file via a temporary file and atomic swap.

    Args:
        path: Target file path.
        content: Source content to write.

    Raises:
        SourceEditError: If file writing or renaming fails.
    """
    path = path.resolve()
    target_dir = path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    with logger.timed("write_file_atomically", path=str(path)):
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=DEFAULT_WRITE_ENCODING,
                dir=target_dir,
                delete=False,
                suffix=TEMP_FILE_SUFFIX,
            ) as tmp:
                tmp.write(content)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_path = Path(tmp.name)

            os.replace(temp_path, path)
            logger.info("Successfully wrote file atomically", extra={"path": str(path)})
        except OSError as exc:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            logger.error(
                "Atomic write failure",
                extra={"path": str(path), "error": str(exc)},
            )
            raise SourceEditError(f"Failed to atomically write to '{path}': {exc}") from exc


def preview_and_apply(
    path: Path,
    original_source: str,
    edits: Sequence[SourceEdit],
    interactive: bool = True,
    dry_run: bool = False,
    stream: sys.stdout.__class__ = sys.stdout,  # ty: ignore[invalid-type-form]
    on_preview_diff: Callable[[str], None] | None = None,
    on_prompt_confirmation: Callable[[str, int], bool] | None = None,
) -> bool:
    """Display the proposed diff and prompt the user to apply changes.

    Args:
        path: Target source file path.
        original_source: The original unmodified source code.
        edits: Sequence of edits to apply.
        interactive: If True, asks user for confirmation before writing.
        dry_run: If True, outputs preview only and does not write to disk.
        stream: Output stream for diff rendering and prompts.
        on_preview_diff: Optional custom renderer for unified diff text.
        on_prompt_confirmation: Optional custom confirmation prompt callable.

    Returns:
        True if changes were applied to disk, False if cancelled, skipped, or dry-run.
    """
    if not edits:
        if on_preview_diff is None:
            stream.write("No documentation changes to apply.\n")
        return False

    modified_source = apply_edits(original_source, edits)
    supports_color = hasattr(stream, "isatty") and stream.isatty()
    diff_text = generate_diff(
        original=original_source,
        modified=modified_source,
        filename=path.name,
        colorize=supports_color,
    )

    if on_preview_diff is not None:
        on_preview_diff(diff_text)
    else:
        separator = DIFF_SEPARATOR_CHAR * DIFF_SEPARATOR_LENGTH
        stream.write("\nProposed Changes:\n")
        stream.write(f"{separator}\n")
        stream.write(diff_text)
        stream.write(f"{separator}\n")

    if dry_run:
        if on_preview_diff is None:
            stream.write("Dry run enabled: no modifications written to disk.\n")
        logger.info("Dry run complete", extra={"path": str(path), "edits": len(edits)})
        return False

    if interactive:
        if on_prompt_confirmation is not None:
            choice = on_prompt_confirmation(path.name, len(edits))
        else:
            try:
                raw = (
                    input(f"Apply {len(edits)} docstring(s) to '{path.name}'? [y/N]: ")
                    .strip()
                    .lower()
                )
                choice = raw in CONFIRMATION_AFFIRMATIVE
            except KeyboardInterrupt, EOFError:
                stream.write("\nOperation aborted by user.\n")
                logger.info("User aborted edit operation", extra={"path": str(path)})
                return False

        if not choice:
            if on_prompt_confirmation is None:
                stream.write("Changes discarded.\n")
            logger.info("User discarded edits", extra={"path": str(path)})
            return False

    write_file_atomically(path, modified_source)
    if on_prompt_confirmation is None:
        stream.write(f"Successfully applied {len(edits)} docstring(s) to '{path.name}'.\n")
    logger.info("Applied edits successfully", extra={"path": str(path), "edits": len(edits)})
    return True
