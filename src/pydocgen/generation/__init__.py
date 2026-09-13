"""Generation package for AI-assisted docstring synthesis."""

from __future__ import annotations

from pydocgen.generation.models import (
    BatchGenerationResult,
    DocumentationStatus,
    GeneratedDocumentation,
)
from pydocgen.generation.prompts import (
    SCHEMA_EXAMPLE,
    SYSTEM_INSTRUCTION,
    build_batch_prompt,
)
from pydocgen.generation.service import GenerationService

__all__ = [
    "BatchGenerationResult",
    "DocumentationStatus",
    "GeneratedDocumentation",
    "GenerationService",
    "SCHEMA_EXAMPLE",
    "SYSTEM_INSTRUCTION",
    "build_batch_prompt",
]
