"""Google Python Style Guide docstring formatter."""

from __future__ import annotations

from pydocgen.analysis.models import FunctionInfo, ParameterKind
from pydocgen.config.settings import (
    DEFAULT_INDENT_WIDTH,
    GOOGLE_SECTION_ARGS,
    GOOGLE_SECTION_RAISES,
    GOOGLE_SECTION_RETURNS,
)
from pydocgen.errors.exceptions import FormattingError
from pydocgen.generation.models import GeneratedDocumentation
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


class GoogleFormatter:
    """Formats GeneratedDocumentation into Google Python Style Guide docstrings."""

    def __init__(self, indent_width: int = DEFAULT_INDENT_WIDTH) -> None:
        """Initialize the Google style docstring formatter.

        Args:
            indent_width: Number of spaces per indentation level. Must be >= 1.

        Raises:
            ValueError: If indent_width is less than 1 or not an integer.
        """
        if not isinstance(indent_width, int) or isinstance(indent_width, bool) or indent_width < 1:
            raise ValueError("indent_width must be a positive integer.")
        self.indent_width = indent_width
        self._tab = " " * indent_width

    def format(
        self,
        documentation: GeneratedDocumentation,
        function: FunctionInfo | None = None,
        base_indent: str = "",
    ) -> str:
        """Format a GeneratedDocumentation instance into a Google-style docstring.

        Args:
            documentation: Validated documentation model.
            function: Optional FunctionInfo to preserve signature parameter ordering.
            base_indent: Indentation prefix to apply to the docstring lines.

        Returns:
            A formatted docstring string enclosed in triple quotes.

        Raises:
            FormattingError: If documentation is None, uncertain, or lacks a summary.
        """
        if documentation is None:
            raise FormattingError("documentation cannot be None.")

        logger.debug(
            "Formatting docstring",
            extra={"function_id": documentation.function_id, "status": documentation.status},
        )

        if documentation.status == "uncertain":
            raise FormattingError(
                f"Cannot format uncertain documentation for function '{documentation.function_id}'."
            )

        if not documentation.summary or not documentation.summary.strip():
            raise FormattingError(
                f"Documentation for function '{documentation.function_id}' has no summary."
            )

        with logger.timed("google_formatter.format", function_id=documentation.function_id):
            summary_lines = self._format_summary_lines(documentation.summary)

            sections: list[list[str]] = []

        # 1. Args Section
        if documentation.args:
            ordered_args = self._order_arguments(documentation.args, function)
            args_lines = [GOOGLE_SECTION_ARGS]
            for name, desc in ordered_args:
                clean_name = name.strip()
                if not clean_name:
                    continue
                arg_lines = desc.splitlines() if desc else []
                first_line = arg_lines[0].strip() if arg_lines else ""
                desc_part = f" {first_line}" if first_line else ""
                args_lines.append(f"{self._tab}{clean_name}:{desc_part}")
                for cont_line in arg_lines[1:]:
                    cont_clean = cont_line.strip()
                    if cont_clean:
                        args_lines.append(f"{self._tab * 2}{cont_clean}")
            if len(args_lines) > 1:
                sections.append(args_lines)

        # 2. Returns Section
        if documentation.returns and documentation.returns.strip():
            returns_lines = [GOOGLE_SECTION_RETURNS]
            ret_lines = documentation.returns.splitlines()
            for r_line in ret_lines:
                clean_r = r_line.strip()
                if clean_r:
                    returns_lines.append(f"{self._tab}{clean_r}")
            if len(returns_lines) > 1:
                sections.append(returns_lines)

        # 3. Raises Section
        if documentation.raises:
            raises_lines = [GOOGLE_SECTION_RAISES]
            for exc_entry in documentation.raises:
                exc_lines = exc_entry.splitlines() if exc_entry else []
                first_line = exc_lines[0].strip() if exc_lines else ""
                if not first_line:
                    continue
                raises_lines.append(f"{self._tab}{first_line}")
                for cont_line in exc_lines[1:]:
                    cont_clean = cont_line.strip()
                    if cont_clean:
                        raises_lines.append(f"{self._tab * 2}{cont_clean}")
            if len(raises_lines) > 1:
                sections.append(raises_lines)

        # Single-line docstring: exactly one line of summary and no sections
        if not sections and len(summary_lines) == 1:
            clean_summary = summary_lines[0].replace('"""', r"\"\"\"")
            return f'{base_indent}"""{clean_summary}"""'

        # Multi-line docstring assembly
        body_lines: list[str] = [f'"""{summary_lines[0].replace('"""', r"\"\"\"")}']
        if len(summary_lines) > 1:
            body_lines.append("")
            for s_line in summary_lines[1:]:
                body_lines.append(s_line.replace('"""', r"\"\"\""))

        # Merge sections with a blank line separator
        for sec in sections:
            body_lines.append("")
            for line in sec:
                body_lines.append(line.replace('"""', r"\"\"\""))

        body_lines.append('"""')

        # Apply base indentation prefix across all lines
        indented_lines: list[str] = []
        for line in body_lines:
            if line:
                indented_lines.append(f"{base_indent}{line}")
            else:
                indented_lines.append("")

        return "\n".join(indented_lines)

    def _format_summary_lines(self, summary: str) -> list[str]:
        raw_lines = [line.strip() for line in summary.strip().splitlines() if line.strip()]
        if not raw_lines:
            return [""]
        first_line = raw_lines[0]
        if not first_line.endswith((".", "!", "?")):
            first_line += "."
        return [first_line, *raw_lines[1:]]

    def _order_arguments(
        self,
        args: dict[str, str],
        function: FunctionInfo | None,
    ) -> list[tuple[str, str]]:
        if not function:
            return list(args.items())

        ordered: list[tuple[str, str]] = []
        remaining = dict(args)
        is_regular_method = function.is_method and "staticmethod" not in function.decorators

        for idx, param in enumerate(function.parameters):
            if is_regular_method and idx == 0 and param.name in ("self", "cls"):
                remaining.pop(param.name, None)
                remaining.pop("self", None)
                remaining.pop("cls", None)
                continue

            matched_key = None
            if param.name in remaining:
                matched_key = param.name
            elif f"*{param.name}" in remaining:
                matched_key = f"*{param.name}"
            elif f"**{param.name}" in remaining:
                matched_key = f"**{param.name}"

            if matched_key is not None:
                display_name = matched_key
                # If param was var positional or var keyword but passed without asterisks, preserve asterisks
                if param.kind == ParameterKind.VAR_POSITIONAL and not display_name.startswith("*"):
                    display_name = f"*{display_name}"
                elif param.kind == ParameterKind.VAR_KEYWORD and not display_name.startswith("**"):
                    display_name = f"**{display_name}"
                ordered.append((display_name, remaining.pop(matched_key)))

        # Append any leftover args not directly matching parameter names
        for name, desc in remaining.items():
            ordered.append((name, desc))

        return ordered
