from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from pydocgen.application.service import DocumentationReport
from pydocgen.cli.theme import (
    CLI_THEME,
    COLOR_DANGER,
    COLOR_SUCCESS,
    COLOR_WARNING,
)

# Render UI and diagnostics exclusively to stderr
err_console = Console(theme=CLI_THEME, stderr=True, highlight=False)
out_console = Console(theme=CLI_THEME, stderr=False, highlight=False)


def render_banner(path: Path, model: str) -> None:
    """Render a header panel for the documentation session."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=3)
    grid.add_column(justify="right", ratio=2)

    title_text = Text()
    title_text.append("pydocgen ", style="brand")
    title_text.append("› ", style="border")
    title_text.append(path.name, style="text")

    model_text = Text(f"model: {model}", style="text.dim")
    grid.add_row(title_text, model_text)

    err_console.print()
    err_console.print(Panel(grid, border_style="border", padding=(0, 1)))


def render_analysis_stats(total: int, documented: int, to_document: int) -> None:
    """Display an aligned, scannable table of function metrics."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="metric.label")
    table.add_column(style="metric.number", justify="right")

    table.add_row("Functions discovered", str(total))
    table.add_row("Existing docstrings", str(documented))
    table.add_row("Target candidates", str(to_document))

    err_console.print(table)
    err_console.print()


def render_batch_step(
    batch_id: str, count: int, current: int, total: int, status: str = "generating"
) -> None:
    """Print an execution line item with status indicator."""
    step_indicator = Text(f"[{current}/{total}] ", style="text.dim")
    batch_label = Text(f"{batch_id} ", style="text")
    count_label = Text(f"({count} functions) ", style="text.dim")

    if status == "success":
        status_badge = Text("✓ complete", style="status.accepted")
    elif status == "failed":
        status_badge = Text("✗ failed", style="status.rejected")
    else:
        status_badge = Text("… processing", style="status.skipped")

    err_console.print(step_indicator + batch_label + count_label + status_badge)


def render_results_summary(report: DocumentationReport) -> None:
    """Render a clean summary table of validation results."""
    err_console.print()
    err_console.print(Rule("Pipeline Evaluation", style="border", align="left"))

    table = Table.grid(padding=(0, 2))
    table.add_column(style="text.dim")
    table.add_column(justify="right")
    table.add_column(style="text.dim")

    table.add_row("Accepted", f"[{COLOR_SUCCESS}]{report.accepted_count}[/]", "Ready for injection")
    table.add_row(
        "Skipped", f"[{COLOR_WARNING}]{report.skipped_count}[/]", "Flagged uncertain by model"
    )
    table.add_row(
        "Rejected", f"[{COLOR_DANGER}]{report.rejected_count}[/]", "Failed AST semantic validation"
    )

    err_console.print(table)
    err_console.print()


def render_unified_diff(diff_text: str) -> None:
    """Render syntax-styled unified diff lines using the curated color palette."""
    if not diff_text.strip():
        return

    err_console.print(Rule("Proposed Modifications", style="border", align="left"))

    styled_diff = Text()
    for line in diff_text.splitlines():
        if line.startswith("---") or line.startswith("+++"):
            styled_diff.append(line + "\n", style="diff.header")
        elif line.startswith("@@"):
            styled_diff.append(line + "\n", style="diff.hunk")
        elif line.startswith("+"):
            styled_diff.append(line + "\n", style="diff.add")
        elif line.startswith("-"):
            styled_diff.append(line + "\n", style="diff.delete")
        else:
            styled_diff.append(line + "\n", style="text.dim")

    err_console.print(styled_diff)
    err_console.print(Rule(style="border"))


def prompt_confirmation(target_name: str, count: int) -> bool:
    """Prompt the user for interactive confirmation with styled visual indicators."""
    query = Text()
    query.append("Apply ", style="prompt.question")
    query.append(str(count), style="metric.number")
    query.append(f" docstring(s) to '{target_name}'? ", style="prompt.question")
    query.append("[y/N]: ", style="prompt.choice")

    try:
        response = err_console.input(query).strip().lower()
        return response in ("y", "yes")
    except KeyboardInterrupt, EOFError:
        err_console.print("\n[status.rejected]Aborted by user.[/]")
        return False


def render_error(title: str, details: str | Sequence[str]) -> None:
    """Render a structured error panel."""
    content = Text()
    if isinstance(details, str):
        content.append(details.strip(), style="text")
    else:
        for item in details:
            content.append(f"• {item.strip()}\n", style="text")

    err_console.print()
    err_console.print(
        Panel(content, title=f"[{COLOR_DANGER}]Error: {title}[/]", border_style=COLOR_DANGER)
    )
