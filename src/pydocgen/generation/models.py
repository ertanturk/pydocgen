"""Data models for structured LLM docstring generation and batch results."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DocumentationStatus = Literal["documented", "uncertain"]


class GeneratedDocumentation(BaseModel):
    """Structured documentation output for a single function."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    function_id: str = Field(
        ...,
        description="The exact identifier of the target function matching the input prompt.",
    )
    status: DocumentationStatus = Field(
        ...,
        description=(
            "Set to 'documented' if documentation was confidently generated. "
            "Set to 'uncertain' if function behavior is ambiguous, dynamic, or unverifiable."
        ),
    )
    summary: str | None = Field(
        default=None,
        description="Imperative one-sentence summary of what the function does (e.g. 'Calculate the SHA-256 digest.').",
    )
    args: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of parameter names to concise functional descriptions. Do not include type hints.",
    )
    returns: str | None = Field(
        default=None,
        description="Explanation of the return value, format, or condition. None if the function returns None.",
    )
    raises: list[str] = Field(
        default_factory=list,
        description="List of exceptions explicitly raised by the function body with their triggering conditions.",
    )
    reason: str | None = Field(
        default=None,
        description="Mandatory explanation when status is 'uncertain'. None when documented.",
    )

    @field_validator("function_id", mode="before")
    @classmethod
    def _validate_function_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("function_id cannot be None.")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("function_id cannot be empty or whitespace.")
        return cleaned

    @field_validator("summary", "returns", "reason", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    @field_validator("args", mode="before")
    @classmethod
    def _clean_args(cls, value: Any) -> dict[str, str]:
        if not value:
            return {}
        if isinstance(value, list):
            res: dict[str, str] = {}
            for item in value:
                if isinstance(item, dict):
                    k = (
                        item.get("name")
                        or item.get("arg")
                        or item.get("parameter")
                        or item.get("param")
                    )
                    v = (
                        item.get("description")
                        or item.get("desc")
                        or item.get("summary")
                        or item.get("doc")
                    )
                    if k is not None and v is not None:
                        k_str = str(k).strip()
                        v_str = str(v).strip()
                        if k_str and v_str:
                            res[k_str] = v_str
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    k_str = str(item[0]).strip()
                    v_str = str(item[1]).strip()
                    if k_str and v_str:
                        res[k_str] = v_str
            return res
        if isinstance(value, dict):
            return {
                str(k).strip(): str(v).strip()
                for k, v in value.items()
                if str(k).strip() and str(v).strip()
            }
        return {}

    @field_validator("raises", mode="before")
    @classmethod
    def _clean_raises(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            cleaned_str = value.strip()
            return [cleaned_str] if cleaned_str else []
        if isinstance(value, list):
            cleaned: list[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text and text not in cleaned:
                    cleaned.append(text)
            return cleaned
        return []

    @model_validator(mode="after")
    def _validate_status_integrity(self) -> Self:
        if self.status == "documented":
            if not self.summary:
                raise ValueError("A 'documented' function must provide a non-empty summary.")
            self.reason = None
        elif self.status == "uncertain":
            if not self.reason:
                raise ValueError("An 'uncertain' function must provide a non-empty reason.")
            self.summary = None
            self.args = {}
            self.returns = None
            self.raises = []
        return self


class BatchGenerationResult(BaseModel):
    """Aggregated documentation payload returned for an entire batch."""

    model_config = ConfigDict(extra="ignore")

    batch_id: str = Field(
        ...,
        description="The unique batch identifier matching the prompt input.",
    )
    functions: list[GeneratedDocumentation] = Field(
        default_factory=list,
        description="List of documentation results corresponding to each function in the batch.",
    )

    @field_validator("batch_id", mode="before")
    @classmethod
    def _validate_batch_id(cls, value: Any) -> str:
        if value is None:
            raise ValueError("batch_id cannot be None.")
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("batch_id cannot be empty.")
        return cleaned

    @field_validator("functions", mode="before")
    @classmethod
    def _deduplicate_functions(cls, value: Any) -> list[Any]:
        if not isinstance(value, list):
            return value
        by_id: dict[str, Any] = {}
        for item in value:
            fn_id = None
            status = None
            if isinstance(item, dict):
                fn_id = item.get("function_id")
                status = item.get("status")
            elif hasattr(item, "function_id"):
                fn_id = item.function_id
                status = getattr(item, "status", None)
            if fn_id is not None:
                fn_id_str = str(fn_id).strip()
                if fn_id_str in by_id:
                    existing = by_id[fn_id_str]
                    existing_status = (
                        existing.get("status")
                        if isinstance(existing, dict)
                        else getattr(existing, "status", None)
                    )
                    if existing_status != "documented" and status == "documented":
                        by_id[fn_id_str] = item
                    continue
                by_id[fn_id_str] = item
            else:
                by_id[str(id(item))] = item
        return list(by_id.values())

    def get_by_id(self, function_id: str) -> GeneratedDocumentation | None:
        """Find a documentation item by its unique function ID."""
        target = function_id.strip()
        for fn in self.functions:
            if fn.function_id == target:
                return fn
        return None

    def __len__(self) -> int:
        return len(self.functions)

    def __iter__(self):
        return iter(self.functions)

    def __getitem__(self, index: int) -> GeneratedDocumentation:
        return self.functions[index]
