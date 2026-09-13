"""Unit tests for the generation package (models, prompts, and GenerationService)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from pydocgen.analysis.models import FunctionInfo, ParameterInfo, ParameterKind
from pydocgen.batching.models import FunctionBatch
from pydocgen.errors.exceptions import ExitCode, ProviderResponseError
from pydocgen.generation.models import (
    BatchGenerationResult,
    GeneratedDocumentation,
)
from pydocgen.generation.prompts import (
    SCHEMA_EXAMPLE,
    SYSTEM_INSTRUCTION,
    build_batch_prompt,
)
from pydocgen.generation.service import GenerationService


def _create_mock_function(
    fn_id: str,
    name: str = "example_fn",
    parameters: tuple[ParameterInfo, ...] = (),
    return_annotation: str | None = "int",
    decorators: tuple[str, ...] = (),
    is_async: bool = False,
    is_method: bool = False,
) -> FunctionInfo:
    return FunctionInfo(
        id=fn_id,
        name=name,
        qualified_name=f"module.{name}",
        source=f"def {name}(): pass",
        parameters=parameters,
        return_annotation=return_annotation,
        decorators=decorators,
        line_start=1,
        line_end=5,
        has_docstring=False,
        is_async=is_async,
        is_method=is_method,
    )


def _create_mock_batch(
    batch_id: str = "batch-101",
    functions: list[FunctionInfo] | None = None,
) -> FunctionBatch:
    if functions is None:
        functions = [_create_mock_function("fn_1"), _create_mock_function("fn_2")]
    return FunctionBatch(
        id=batch_id,
        functions=tuple(functions),
        estimated_input_tokens=500,
    )


class TestGeneratedDocumentationModel:
    """Tests for GeneratedDocumentation model validation and cleaning."""

    def test_valid_documented_function(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Calculate sha256 checksum.",
            args={"path": "Path to target file."},
            returns="Hex digest string.",
            raises=["FileNotFoundError: Missing file."],
        )
        assert doc.function_id == "fn_1"
        assert doc.status == "documented"
        assert doc.summary == "Calculate sha256 checksum."
        assert doc.args == {"path": "Path to target file."}
        assert doc.returns == "Hex digest string."
        assert doc.raises == ["FileNotFoundError: Missing file."]
        assert doc.reason is None

    def test_documented_strips_reason(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Valid summary.",
            reason="Extraneous reason",
        )
        assert doc.reason is None

    def test_documented_missing_summary_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must provide a non-empty summary"):
            GeneratedDocumentation(
                function_id="fn_1",
                status="documented",
                summary="   ",
            )

    def test_valid_uncertain_function(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_2",
            status="uncertain",
            summary="Should be cleared",
            args={"x": "val"},
            returns="None",
            raises=["Exception"],
            reason="Relies on dynamic reflection.",
        )
        assert doc.status == "uncertain"
        assert doc.reason == "Relies on dynamic reflection."
        # Uncertainty should clear docs to prevent contradictory states
        assert doc.summary is None
        assert doc.args == {}
        assert doc.returns is None
        assert doc.raises == []

    def test_uncertain_missing_reason_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="must provide a non-empty reason"):
            GeneratedDocumentation(
                function_id="fn_2",
                status="uncertain",
                reason="",
            )

    def test_invalid_function_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="function_id cannot be empty"):
            GeneratedDocumentation(
                function_id="   ",
                status="documented",
                summary="Test",
            )

    def test_clean_args_from_list_of_dicts(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Test",
            args=[  # type: ignore[arg-type]
                {"name": "file_path", "description": "Path to file."},
                {"param": "flag", "desc": "Enable feature."},
            ],
        )
        assert doc.args == {
            "file_path": "Path to file.",
            "flag": "Enable feature.",
        }

    def test_clean_args_from_list_of_tuples(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Test",
            args=[("k1", "v1"), ("k2", "v2")],  # type: ignore[arg-type]
        )
        assert doc.args == {"k1": "v1", "k2": "v2"}

    def test_clean_raises_from_single_string(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Test",
            raises="ValueError: Invalid input.",  # type: ignore[arg-type]
        )
        assert doc.raises == ["ValueError: Invalid input."]

    def test_clean_raises_deduplication(self) -> None:
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Test",
            raises=["ValueError: Bad", "ValueError: Bad", "TypeError: Wrong type"],
        )
        assert doc.raises == ["ValueError: Bad", "TypeError: Wrong type"]


class TestBatchGenerationResultModel:
    """Tests for BatchGenerationResult container and helper methods."""

    def test_valid_batch_result(self) -> None:
        fn1 = GeneratedDocumentation(function_id="fn_1", status="documented", summary="Summary 1")
        fn2 = GeneratedDocumentation(function_id="fn_2", status="uncertain", reason="Uncertain 2")
        res = BatchGenerationResult(batch_id="batch-01", functions=[fn1, fn2])

        assert res.batch_id == "batch-01"
        assert len(res) == 2
        assert list(res) == [fn1, fn2]
        assert res[0] == fn1
        assert res.get_by_id("fn_1") == fn1
        assert res.get_by_id("fn_2") == fn2
        assert res.get_by_id("missing") is None

    def test_empty_batch_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="batch_id cannot be empty"):
            BatchGenerationResult(batch_id="   ")

    def test_functions_deduplication_by_id(self) -> None:
        res = BatchGenerationResult(
            batch_id="batch-01",
            functions=[
                {"function_id": "fn_1", "status": "documented", "summary": "First"},  # type: ignore[list-item]
                {"function_id": "fn_1", "status": "documented", "summary": "Duplicate"},  # type: ignore[list-item]
            ],
        )
        assert len(res) == 1
        assert res[0].summary == "First"


class TestPromptConstruction:
    """Tests for prompt builder and formatting."""

    def test_build_batch_prompt_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a FunctionBatch"):
            build_batch_prompt("not-a-batch")  # type: ignore[arg-type]

    def test_build_batch_prompt_basic_formatting(self) -> None:
        fn = _create_mock_function(
            "fn_1",
            name="add",
            parameters=(
                ParameterInfo(name="a", annotation="int"),
                ParameterInfo(name="b", annotation="int", default="0"),
            ),
            return_annotation="int",
        )
        batch = _create_mock_batch("batch-999", [fn])
        prompt = build_batch_prompt(batch)

        assert "Batch ID: batch-999" in prompt
        assert SYSTEM_INSTRUCTION[:30] not in prompt  # SYSTEM_INSTRUCTION is passed separately
        assert SCHEMA_EXAMPLE in prompt
        assert "--- Function ID: fn_1 ---" in prompt
        assert "Signature: def add(a: int, b: int = 0) -> int:" in prompt
        assert "Source Code:\ndef add(): pass" in prompt

    def test_build_batch_prompt_async_and_decorators(self) -> None:
        fn = _create_mock_function(
            "fn_async",
            name="fetch",
            parameters=(),
            return_annotation="dict",
            decorators=("retry", "auth_required"),
            is_async=True,
        )
        batch = _create_mock_batch("batch-async", [fn])
        prompt = build_batch_prompt(batch)

        assert "@retry" in prompt
        assert "@auth_required" in prompt
        assert "Signature: async def fetch() -> dict:" in prompt

    def test_build_batch_prompt_advanced_parameter_kinds(self) -> None:
        fn = _create_mock_function(
            "fn_complex",
            name="complex_func",
            parameters=(
                ParameterInfo(name="p1", kind=ParameterKind.POSITIONAL_ONLY),
                ParameterInfo(name="p2", kind=ParameterKind.POSITIONAL_ONLY),
                ParameterInfo(name="reg", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
                ParameterInfo(name="args", kind=ParameterKind.VAR_POSITIONAL),
                ParameterInfo(name="kw", kind=ParameterKind.KEYWORD_ONLY, default="'test'"),
                ParameterInfo(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
            ),
            return_annotation=None,
        )
        batch = _create_mock_batch("batch-complex", [fn])
        prompt = build_batch_prompt(batch)

        assert (
            "Signature: def complex_func(p1, p2, /, reg, *args, kw = 'test', **kwargs):" in prompt
        )


class TestGenerationService:
    """Tests for GenerationService orchestrator and result reconciliation."""

    def test_init_validation(self) -> None:
        with pytest.raises(ValueError, match="provider cannot be None"):
            GenerationService(provider=None)  # type: ignore[arg-type]

        mock_bad_provider = MagicMock(spec=[])
        with pytest.raises(ValueError, match="must implement an async 'generate' method"):
            GenerationService(provider=mock_bad_provider)

    @pytest.mark.asyncio
    async def test_generate_batch_type_error(self) -> None:
        service = GenerationService(provider=MagicMock(generate=AsyncMock()))
        with pytest.raises(TypeError, match="must be a FunctionBatch"):
            await service.generate_batch("invalid-batch")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_generate_empty_batch_skips_provider(self) -> None:
        mock_provider = MagicMock(generate=AsyncMock())
        service = GenerationService(provider=mock_provider)

        empty_batch = FunctionBatch(id="empty-batch", functions=(), estimated_input_tokens=0)
        result = await service.generate_batch(empty_batch)

        assert result.batch_id == "empty-batch"
        assert len(result) == 0
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_batch_success_happy_path(self) -> None:
        fn1 = _create_mock_function("fn_1")
        fn2 = _create_mock_function("fn_2")
        batch = _create_mock_batch("batch-01", [fn1, fn2])

        expected_result = BatchGenerationResult(
            batch_id="batch-01",
            functions=[
                GeneratedDocumentation(function_id="fn_1", status="documented", summary="Doc 1"),
                GeneratedDocumentation(function_id="fn_2", status="documented", summary="Doc 2"),
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=expected_result))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)

        assert result.batch_id == "batch-01"
        assert len(result) == 2
        assert result.get_by_id("fn_1").summary == "Doc 1"  # type: ignore[union-attr]
        assert result.get_by_id("fn_2").summary == "Doc 2"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_generate_batch_parses_json_string(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("batch-01", [fn1])

        json_str = (
            '{"batch_id": "batch-01", "functions": ['
            '{"function_id": "fn_1", "status": "documented", "summary": "Doc from JSON"}'
            "]}"
        )
        mock_provider = MagicMock(generate=AsyncMock(return_value=json_str))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)
        assert len(result) == 1
        assert result[0].summary == "Doc from JSON"

    @pytest.mark.asyncio
    async def test_generate_batch_parses_dict(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("batch-01", [fn1])

        dict_payload = {
            "batch_id": "batch-01",
            "functions": [
                {"function_id": "fn_1", "status": "documented", "summary": "Doc from dict"}
            ],
        }
        mock_provider = MagicMock(generate=AsyncMock(return_value=dict_payload))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)
        assert len(result) == 1
        assert result[0].summary == "Doc from dict"

    @pytest.mark.asyncio
    async def test_generate_batch_unexpected_response_type_raises(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("batch-01", [fn1])

        mock_provider = MagicMock(generate=AsyncMock(return_value=12345))
        service = GenerationService(provider=mock_provider)

        with pytest.raises(ProviderResponseError, match="Unexpected response type") as exc:
            await service.generate_batch(batch)
        assert exc.value.exit_code == ExitCode.DATAERR

    @pytest.mark.asyncio
    async def test_reconcile_corrects_batch_id_mismatch(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("real-batch-id", [fn1])

        wrong_batch_res = BatchGenerationResult(
            batch_id="hallucinated-batch-id",
            functions=[
                GeneratedDocumentation(function_id="fn_1", status="documented", summary="Summary")
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=wrong_batch_res))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)
        assert result.batch_id == "real-batch-id"

    @pytest.mark.asyncio
    async def test_reconcile_fills_missing_functions_with_uncertain(self) -> None:
        fn1 = _create_mock_function("fn_1")
        fn2 = _create_mock_function("fn_2")
        fn3 = _create_mock_function("fn_3")
        batch = _create_mock_batch("batch-01", [fn1, fn2, fn3])

        # Model omitted fn_2 and fn_3
        partial_res = BatchGenerationResult(
            batch_id="batch-01",
            functions=[
                GeneratedDocumentation(function_id="fn_1", status="documented", summary="Doc 1")
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=partial_res))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)

        assert len(result) == 3
        assert result.get_by_id("fn_1").status == "documented"  # type: ignore[union-attr]
        assert result.get_by_id("fn_2").status == "uncertain"  # type: ignore[union-attr]
        assert "omitted this function" in result.get_by_id("fn_2").reason  # type: ignore[union-attr]
        assert result.get_by_id("fn_3").status == "uncertain"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_reconcile_strips_hallucinated_functions(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("batch-01", [fn1])

        # Model returned fn_1 plus hallucinated fn_ghost
        hallucinated_res = BatchGenerationResult(
            batch_id="batch-01",
            functions=[
                GeneratedDocumentation(function_id="fn_1", status="documented", summary="Doc 1"),
                GeneratedDocumentation(
                    function_id="fn_ghost", status="documented", summary="Ghost Doc"
                ),
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=hallucinated_res))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)

        assert len(result) == 1
        assert result.get_by_id("fn_1") is not None
        assert result.get_by_id("fn_ghost") is None

    @pytest.mark.asyncio
    async def test_reconcile_preserves_exact_batch_ordering(self) -> None:
        fn_a = _create_mock_function("fn_a")
        fn_b = _create_mock_function("fn_b")
        fn_c = _create_mock_function("fn_c")
        batch = _create_mock_batch("batch-order", [fn_a, fn_b, fn_c])

        # Model returned out of order (fn_c, then fn_a, omitted fn_b)
        scrambled_res = BatchGenerationResult(
            batch_id="batch-order",
            functions=[
                GeneratedDocumentation(function_id="fn_c", status="documented", summary="Doc C"),
                GeneratedDocumentation(function_id="fn_a", status="documented", summary="Doc A"),
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=scrambled_res))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)

        # Output must match [fn_a, fn_b, fn_c]
        assert [doc.function_id for doc in result.functions] == ["fn_a", "fn_b", "fn_c"]
        assert result[0].function_id == "fn_a"
        assert result[1].function_id == "fn_b"
        assert result[1].status == "uncertain"
        assert result[2].function_id == "fn_c"

    @pytest.mark.asyncio
    async def test_reconcile_deduplicates_preferring_documented(self) -> None:
        fn1 = _create_mock_function("fn_1")
        batch = _create_mock_batch("batch-01", [fn1])

        dup_res = BatchGenerationResult(
            batch_id="batch-01",
            functions=[
                GeneratedDocumentation(
                    function_id="fn_1", status="uncertain", reason="First attempt failed"
                ),
                GeneratedDocumentation(
                    function_id="fn_1", status="documented", summary="Second attempt succeeded"
                ),
            ],
        )

        mock_provider = MagicMock(generate=AsyncMock(return_value=dup_res))
        service = GenerationService(provider=mock_provider)

        result = await service.generate_batch(batch)

        assert len(result) == 1
        assert result[0].status == "documented"
        assert result[0].summary == "Second attempt succeeded"
