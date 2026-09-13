from __future__ import annotations

from rich.style import Style
from rich.theme import Theme

COLOR_PRIMARY = "#7AA2F7"  # Crisp periwinkle / slate-blue (brand, headings)
COLOR_SECONDARY = "#BB9AF7"  # Muted lavender (metadata, decorators, accents)
COLOR_SUCCESS = "#73DACA"  # Fresh mint / sage (accepted, documented)
COLOR_WARNING = "#E0AF68"  # Warm honey / amber (uncertain, skipped)
COLOR_DANGER = "#F7768E"  # Soft crimson / rose (rejected, errors)
COLOR_MUTED = "#565F89"  # Cool slate gray (borders, framing, disabled)
COLOR_TEXT = "#C0CAF5"  # High-legibility off-white
COLOR_TEXT_DIM = "#9AA5CE"  # Subtle ash for secondary labels
COLOR_BG_DIFF_ADD = "#1A2B32"  # Diff add subtle tint
COLOR_BG_DIFF_DEL = "#2D1C24"  # Diff delete subtle tint

CLI_THEME = Theme(
    {
        # Application semantics
        "brand": Style(color=COLOR_PRIMARY, bold=True),
        "brand.subtle": Style(color=COLOR_PRIMARY),
        "accent": Style(color=COLOR_SECONDARY),
        "text": Style(color=COLOR_TEXT),
        "text.dim": Style(color=COLOR_TEXT_DIM),
        "border": Style(color=COLOR_MUTED),
        # Pipeline statuses
        "status.accepted": Style(color=COLOR_SUCCESS, bold=True),
        "status.skipped": Style(color=COLOR_WARNING, bold=True),
        "status.rejected": Style(color=COLOR_DANGER, bold=True),
        "metric.number": Style(color=COLOR_PRIMARY, bold=True),
        "metric.label": Style(color=COLOR_TEXT_DIM),
        # Diff semantics
        "diff.header": Style(color=COLOR_MUTED, bold=True),
        "diff.hunk": Style(color=COLOR_SECONDARY),
        "diff.add": Style(color=COLOR_SUCCESS),
        "diff.delete": Style(color=COLOR_DANGER),
        # Prompts & interactive states
        "prompt.question": Style(color=COLOR_TEXT, bold=True),
        "prompt.choice": Style(color=COLOR_PRIMARY),
        "prompt.default": Style(color=COLOR_TEXT_DIM),
    }
)
