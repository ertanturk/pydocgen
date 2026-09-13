"""Prompt construction and instructions for Google-style docstring generation."""

from __future__ import annotations

from pydocgen.analysis.models import ParameterKind
from pydocgen.batching.models import FunctionBatch

SYSTEM_INSTRUCTION = """You are a precision Python documentation generator adhering strictly to Google Python Style Guide standards.

Primary Directives:
1. Strict Ground Truth: Document ONLY behaviors, parameters, exceptions, and side effects demonstrated in the provided source code.
2. No Hallucinations: Do not assume arguments, return values, or unraised exceptions.
3. Parameter Descriptions: Write only functional descriptions in 'args' (e.g., 'Target user identifier.'). Do NOT repeat types like 'int: Target user identifier', as type annotations are handled separately.
4. Handling Uncertainty: If a function is dynamic, relies on unresolvable runtime variables, or has ambiguous behavior, mark status="uncertain", explain why in "reason", and leave documentation fields null/empty.
5. Exact Identifiers: You MUST output exactly one result for every provided function ID, preserving the exact 'function_id' and 'batch_id' values.
"""

SCHEMA_EXAMPLE = """{
  "batch_id": "batch-001-abc12345",
  "functions": [
    {
      "function_id": "fn_101",
      "status": "documented",
      "summary": "Read and deserialize JSON payload from disk.",
      "args": {
        "file_path": "Path to the JSON file on the local filesystem."
      },
      "returns": "Parsed dictionary representing the JSON content.",
      "raises": [
        "FileNotFoundError: If the file does not exist."
      ],
      "reason": null
    },
    {
      "function_id": "fn_102",
      "status": "uncertain",
      "summary": null,
      "args": {},
      "returns": null,
      "raises": [],
      "reason": "Function relies on dynamic metaclass injection whose signature cannot be verified statically."
    }
  ]
}"""


def build_batch_prompt(batch: FunctionBatch) -> str:
    """Build a structured prompt containing instructions, target JSON schema guidance, and functions.

    Args:
        batch: The FunctionBatch containing functions to document.

    Returns:
        The formatted prompt string ready for submission to Gemini.

    Raises:
        TypeError: If batch is not a FunctionBatch instance.
    """
    if not isinstance(batch, FunctionBatch):
        raise TypeError(f"batch must be a FunctionBatch, got {type(batch).__name__}.")

    lines: list[str] = [
        f"Batch ID: {batch.id}",
        "",
        "Instructions:",
        "- Analyze each function provided below.",
        "- Generate structured documentation conforming to the Google Python Style Guide.",
        f'- You MUST return a JSON object with "batch_id": "{batch.id}" and "functions": [...].',
        "- Adhere strictly to the following target JSON structure format:",
        SCHEMA_EXAMPLE,
        "",
        "Functions to process:",
        "",
    ]

    for fn in batch.functions:
        param_descriptors: list[str] = []
        has_var_pos = any(p.kind == ParameterKind.VAR_POSITIONAL for p in fn.parameters)
        pos_only_count = sum(1 for p in fn.parameters if p.kind == ParameterKind.POSITIONAL_ONLY)
        inserted_slash = False
        inserted_star = False

        for param in fn.parameters:
            if (
                pos_only_count > 0
                and not inserted_slash
                and param.kind != ParameterKind.POSITIONAL_ONLY
            ):
                param_descriptors.append("/")
                inserted_slash = True

            if param.kind == ParameterKind.KEYWORD_ONLY and not has_var_pos and not inserted_star:
                param_descriptors.append("*")
                inserted_star = True

            prefix = ""
            if param.kind == ParameterKind.VAR_POSITIONAL:
                prefix = "*"
            elif param.kind == ParameterKind.VAR_KEYWORD:
                prefix = "**"

            descriptor = f"{prefix}{param.name}"
            if param.annotation:
                descriptor += f": {param.annotation}"
            if param.default is not None:
                descriptor += f" = {param.default}"
            param_descriptors.append(descriptor)

        if pos_only_count > 0 and not inserted_slash:
            param_descriptors.append("/")

        async_prefix = "async " if fn.is_async else ""
        return_str = f" -> {fn.return_annotation}" if fn.return_annotation else ""
        signature_str = f"{async_prefix}def {fn.name}({', '.join(param_descriptors)}){return_str}:"

        fn_header = [
            f"--- Function ID: {fn.id} ---",
            f"Qualified Name: {fn.qualified_name}",
        ]
        if fn.decorators:
            fn_header.extend([f"@{dec}" for dec in fn.decorators])
        fn_header.extend(
            [
                f"Signature: {signature_str}",
                "Source Code:",
                fn.source,
                "",
            ]
        )
        lines.extend(fn_header)

    return "\n".join(lines)
