"""Unit tests for the application documentation service pipeline."""

from __future__ import annotations

import io
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydocgen.application.service import (
    DocumentationService,
    PipelineExecutionError,
)
from pydocgen.errors.exceptions import AuthenticationError
from pydocgen.generation.models import BatchGenerationResult, GeneratedDocumentation


@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    target = tmp_path / "sample.py"
    target.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def multiply(x: int, y: int) -> int:\n"
        "    return x * y\n"
    )
    return target


@pytest.fixture
def documented_python_file(tmp_path: Path) -> Path:
    target = tmp_path / "documented.py"
    target.write_text(
        'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n'
    )
    return target


@pytest.fixture
def mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.fast_fail = True
    provider.verify_connection = AsyncMock(return_value=True)
    provider.close = AsyncMock()

    async def _mock_generate(prompt: str, **kwargs) -> BatchGenerationResult:
        fn_ids = re.findall(r"Function ID:\s*([a-zA-Z0-9_-]+)", prompt)
        functions = []
        for idx, fn_id in enumerate(fn_ids):
            if "multiply" in prompt and (len(fn_ids) == 1 or idx == len(fn_ids) - 1):
                functions.append(
                    GeneratedDocumentation(
                        function_id=fn_id,
                        status="documented",
                        summary="Multiply two integers together.",
                        args={"x": "First integer.", "y": "Second integer."},
                        returns="Product of x and y.",
                    )
                )
            else:
                functions.append(
                    GeneratedDocumentation(
                        function_id=fn_id,
                        status="documented",
                        summary="Add two integers together.",
                        args={"a": "First integer.", "b": "Second integer."},
                        returns="Sum of a and b.",
                    )
                )
        return BatchGenerationResult(batch_id="batch-001", functions=functions)

    provider.generate = AsyncMock(side_effect=_mock_generate)
    return provider


class TestDocumentationServiceInit:
    """Tests for service initialization and validation."""

    def test_default_initialization(self) -> None:
        service = DocumentationService()
        assert service.max_concurrency > 0
        assert service.max_batch_tokens > 0
        assert service.max_functions_per_batch > 0
        assert service.strict_raises is True

    def test_invalid_parameters_raise(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency must be at least 1"):
            DocumentationService(max_concurrency=0)

        with pytest.raises(ValueError, match="requests_per_minute cannot be negative"):
            DocumentationService(requests_per_minute=-5)

        with pytest.raises(ValueError, match="max_batch_tokens must be at least 1"):
            DocumentationService(max_batch_tokens=0)

        with pytest.raises(ValueError, match="max_functions_per_batch must be at least 1"):
            DocumentationService(max_functions_per_batch=0)


class TestDocumentFileHappyPath:
    """Tests for complete successful execution of the document_file pipeline."""

    @pytest.mark.asyncio
    async def test_document_file_success(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=sample_python_file,
            interactive=False,
            stream=buf,
        )

        assert report.is_success is True
        assert report.total_functions == 2
        assert report.already_documented == 0
        assert report.to_document == 2
        assert report.accepted_count == 2
        assert report.applied is True
        assert report.functions_modified == 2

        content = sample_python_file.read_text()
        assert '"""Add two integers together.' in content
        assert '"""Multiply two integers together.' in content

    @pytest.mark.asyncio
    async def test_already_documented_early_exit(
        self, documented_python_file: Path, mock_provider: MagicMock
    ) -> None:
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=documented_python_file,
            interactive=False,
            stream=buf,
        )

        assert report.is_success is True
        assert report.total_functions == 1
        assert report.already_documented == 1
        assert report.to_document == 0
        assert report.batches_processed == 0
        assert report.applied is False
        assert "already have docstrings" in buf.getvalue()
        mock_provider.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrite_existing_option(
        self, documented_python_file: Path, mock_provider: MagicMock
    ) -> None:
        service = DocumentationService(provider=mock_provider)

        report = await service.document_file(
            path=documented_python_file,
            interactive=False,
            overwrite_existing=True,
        )

        assert report.to_document == 1
        assert report.batches_processed == 1


class TestDocumentFileErrorHandling:
    """Tests for atomic abort and error handling across pipeline stages."""

    @pytest.mark.asyncio
    async def test_fast_fail_health_check_failure(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        mock_provider.verify_connection = AsyncMock(
            side_effect=AuthenticationError("Invalid API key")
        )
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=sample_python_file,
            interactive=False,
            raise_on_failure=False,
            stream=buf,
        )

        assert report.is_success is False
        assert report.failed_batches > 0
        assert report.applied is False
        assert "Provider connection check failed" in buf.getvalue()
        # Ensure file was not modified
        assert '"""' not in sample_python_file.read_text()

    @pytest.mark.asyncio
    async def test_fast_fail_raise_on_failure(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        mock_provider.verify_connection = AsyncMock(
            side_effect=AuthenticationError("Invalid API key")
        )
        service = DocumentationService(provider=mock_provider)

        with pytest.raises(PipelineExecutionError, match="Provider connection check failed"):
            await service.document_file(
                path=sample_python_file,
                interactive=False,
                raise_on_failure=True,
            )

    @pytest.mark.asyncio
    async def test_batch_worker_failure_aborts_atomically(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM API crashed"))
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=sample_python_file,
            interactive=False,
            raise_on_failure=False,
            stream=buf,
        )

        assert report.is_success is False
        assert report.failed_batches == 1
        assert report.applied is False
        assert "Source modification aborted to guarantee atomic behavior" in buf.getvalue()
        # Verify atomicity: file remained untouched
        assert '"""' not in sample_python_file.read_text()

    @pytest.mark.asyncio
    async def test_no_accepted_documentation_skips_preview(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        # Provider returns uncertain status for all functions
        async def _mock_uncertain(prompt: str, **kwargs) -> BatchGenerationResult:
            return BatchGenerationResult(
                batch_id="batch-001",
                functions=[
                    GeneratedDocumentation(
                        function_id="fn_add",
                        status="uncertain",
                        reason="Dynamic behavior cannot be statically analyzed.",
                    ),
                    GeneratedDocumentation(
                        function_id="fn_multiply",
                        status="uncertain",
                        reason="Ambiguous side effects.",
                    ),
                ],
            )

        mock_provider.generate = AsyncMock(side_effect=_mock_uncertain)
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=sample_python_file,
            interactive=False,
            stream=buf,
        )

        assert report.is_success is True
        assert report.accepted_count == 0
        assert report.skipped_count == 2
        assert report.applied is False
        assert "No documentation changes could be accepted" in buf.getvalue()
        assert "2 function(s) were flagged uncertain" in buf.getvalue()

    @pytest.mark.asyncio
    async def test_dry_run_leaves_file_untouched(
        self, sample_python_file: Path, mock_provider: MagicMock
    ) -> None:
        service = DocumentationService(provider=mock_provider)
        buf = io.StringIO()

        report = await service.document_file(
            path=sample_python_file,
            interactive=False,
            dry_run=True,
            stream=buf,
        )

        assert report.is_success is True
        assert report.applied is False
        assert "Dry run enabled" in buf.getvalue()
        assert '"""' not in sample_python_file.read_text()


class TestProviderLifecycleAndMultiFile:
    """Tests for provider lifecycle management and multi-file processing."""

    @pytest.mark.asyncio
    async def test_internal_provider_closed_in_finally(self, sample_python_file: Path) -> None:
        mock_internal_provider = MagicMock()
        mock_internal_provider.fast_fail = False
        mock_internal_provider.close = AsyncMock()

        async def _mock_gen(*args, **kwargs) -> BatchGenerationResult:
            return BatchGenerationResult(batch_id="b1", functions=[])

        mock_internal_provider.generate = AsyncMock(side_effect=_mock_gen)

        with patch(
            "pydocgen.application.service.GeminiProvider", return_value=mock_internal_provider
        ):
            service = DocumentationService()
            await service.document_file(sample_python_file, interactive=False)

        mock_internal_provider.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_document_files_multi_file(
        self, tmp_path: Path, mock_provider: MagicMock
    ) -> None:
        f1 = tmp_path / "file1.py"
        f1.write_text("def f1(): pass\n")
        f2 = tmp_path / "file2.py"
        f2.write_text("def f2(): pass\n")

        service = DocumentationService(provider=mock_provider)
        reports = await service.document_files([f1, f2], interactive=False)

        assert len(reports) == 2
        assert reports[0].path == f1.resolve()
        assert reports[1].path == f2.resolve()
