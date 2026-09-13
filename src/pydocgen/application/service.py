"""Application-level orchestration pipeline for Python docstring generation."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from pydocgen.analysis.extractor import extract_functions
from pydocgen.analysis.parser import parse_file
from pydocgen.batching.batcher import create_batches
from pydocgen.batching.models import FunctionBatch
from pydocgen.concurrency.executor import BatchTaskResult, run_batch_pool
from pydocgen.concurrency.limiter import RateLimiter
from pydocgen.config.settings import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_BATCH_TOKENS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_FUNCTIONS_PER_BATCH,
    DEFAULT_RPM,
    DEFAULT_STRICT_RAISES,
)
from pydocgen.editing.rewriter import create_source_edits, preview_and_apply
from pydocgen.errors.exceptions import PipelineExecutionError
from pydocgen.generation.models import BatchGenerationResult, GeneratedDocumentation
from pydocgen.generation.service import GenerationService
from pydocgen.providers.gemini import GeminiProvider
from pydocgen.telemetry import bind_context, get_logger
from pydocgen.validation.validator import SemanticValidator, ValidationReport

logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentationReport:
    """Summary of the document generation pipeline run."""

    path: Path
    total_functions: int
    already_documented: int
    to_document: int
    batches_processed: int
    accepted_count: int
    skipped_count: int
    rejected_count: int
    failed_batches: int
    applied: bool
    validation_report: ValidationReport | None = None
    batch_errors: list[str] = field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Returns True if all batches succeeded without failures."""
        return self.failed_batches == 0

    @property
    def functions_modified(self) -> int:
        """Returns the number of functions whose docstrings were applied to disk."""
        return self.accepted_count if self.applied else 0


class DocumentationService:
    """Coordinates the end-to-end documentation extraction and editing pipeline."""

    def __init__(
        self,
        provider: GeminiProvider | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_GEMINI_MODEL,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        requests_per_minute: int = DEFAULT_RPM,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        max_functions_per_batch: int = DEFAULT_MAX_FUNCTIONS_PER_BATCH,
        strict_raises: bool = DEFAULT_STRICT_RAISES,
    ) -> None:
        """Initialize the DocumentationService.

        Args:
            provider: Optional pre-configured provider adapter instance.
            api_key: Optional explicit Gemini API key if no provider is passed.
            model: Gemini model name identifier.
            max_concurrency: Number of concurrent batch worker tasks (>= 1).
            requests_per_minute: Rate limiting ceiling for API calls (>= 0).
            max_batch_tokens: Token ceiling per function batch (>= 1).
            max_functions_per_batch: Maximum functions packed per batch (>= 1).
            strict_raises: Flag to enforce strict AST exception raise matching.

        Raises:
            ValueError: If any numeric parameter is negative or out of bounds.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if requests_per_minute < 0:
            raise ValueError("requests_per_minute cannot be negative.")
        if max_batch_tokens < 1:
            raise ValueError("max_batch_tokens must be at least 1.")
        if max_functions_per_batch < 1:
            raise ValueError("max_functions_per_batch must be at least 1.")

        self.api_key = api_key
        self.model = model
        self.max_concurrency = max_concurrency
        self.max_batch_tokens = max_batch_tokens
        self.max_functions_per_batch = max_functions_per_batch
        self.strict_raises = strict_raises

        self._provider = provider
        self._rate_limiter = RateLimiter(requests_per_minute=requests_per_minute)
        self._validator = SemanticValidator(strict_raises=strict_raises)

    async def document_file(
        self,
        path: Path | str,
        *,
        interactive: bool = True,
        dry_run: bool = False,
        enforce_relative: bool = False,
        overwrite_existing: bool = False,
        raise_on_failure: bool = False,
        stream: TextIO = sys.stdout,
        on_analysis_complete: Callable[[int, int, int], None] | None = None,
        on_batch_start: Callable[[str, int, int, int], None] | None = None,
        on_batch_complete: Callable[[str, int, int, int, str], None] | None = None,
        on_validation_complete: Callable[[DocumentationReport], None] | None = None,
        on_preview_diff: Callable[[str], None] | None = None,
        on_prompt_confirmation: Callable[[str, int], bool] | None = None,
    ) -> DocumentationReport:
        """Run the full documentation generation workflow on a single Python file.

        Args:
            path: Target Python file path.
            interactive: Whether to prompt for confirmation before modifying disk.
            dry_run: If True, previews diff without writing to disk.
            enforce_relative: Rejects absolute paths if True.
            overwrite_existing: If True, targets all functions including documented ones.
            raise_on_failure: If True, raises PipelineExecutionError on unrecoverable failures.
            stream: Stream destination for diff display and confirmation prompts.
            on_analysis_complete: Optional callback invoked with (total, documented, to_document).
            on_batch_start: Optional callback invoked with (batch_id, count, current, total).
            on_batch_complete: Optional callback invoked with (batch_id, count, current, total, status).
            on_validation_complete: Optional callback invoked with preliminary DocumentationReport.
            on_preview_diff: Optional callback to render styled unified diff.
            on_prompt_confirmation: Optional callback to prompt user for confirmation.

        Returns:
            DocumentationReport detailing processing counts and outcome.

        Raises:
            FileValidationError: If target path does not exist or is invalid.
            ParseError: If target file has syntax errors.
            PipelineExecutionError: If provider or worker execution fails unrecoverably.
        """
        with (
            bind_context(file_path=str(path)),
            logger.timed("document_file", path=str(path)),
        ):
            # Parse source & extract AST functions
            valid_path, source, tree = parse_file(path, enforce_relative=enforce_relative)
            all_functions = extract_functions(source, tree)
            already_documented = sum(1 for fn in all_functions if fn.has_docstring)
            target_functions = (
                all_functions
                if overwrite_existing
                else [fn for fn in all_functions if not fn.has_docstring]
            )

            if on_analysis_complete is not None:
                on_analysis_complete(len(all_functions), already_documented, len(target_functions))

            # Early exit if nothing needs documentation
            if not target_functions:
                if all_functions:
                    stream.write(
                        f"All {len(all_functions)} function(s) in '{valid_path.name}' already have docstrings.\n"
                    )
                else:
                    stream.write(f"No functions found in '{valid_path.name}'.\n")

                return DocumentationReport(
                    path=valid_path,
                    total_functions=len(all_functions),
                    already_documented=already_documented,
                    to_document=0,
                    batches_processed=0,
                    accepted_count=0,
                    skipped_count=0,
                    rejected_count=0,
                    failed_batches=0,
                    applied=False,
                )

            # Partition target functions into token-budgeted batches
            batches = create_batches(
                target_functions,
                max_batch_tokens=self.max_batch_tokens,
                max_functions_per_batch=self.max_functions_per_batch,
            )

            logger.info(
                "Batched functions for documentation",
                extra={
                    "path": str(valid_path),
                    "functions_to_document": len(target_functions),
                    "batches": len(batches),
                },
            )

            # Setup provider with managed lifecycle
            created_provider = False
            if self._provider is not None:
                provider = self._provider
            else:
                provider = GeminiProvider(api_key=self.api_key, model=self.model)
                created_provider = True

            try:
                # Fast-fail connection verification
                if getattr(provider, "fast_fail", False) and hasattr(provider, "verify_connection"):
                    try:
                        await provider.verify_connection()
                    except Exception as exc:
                        logger.error(
                            "Fast-fail health check probe failed",
                            extra={"error": str(exc)},
                        )
                        stream.write(f"\nError: Provider connection check failed: {exc}\n")
                        if raise_on_failure:
                            raise PipelineExecutionError(
                                f"Provider connection check failed: {exc}"
                            ) from exc

                        return DocumentationReport(
                            path=valid_path,
                            total_functions=len(all_functions),
                            already_documented=already_documented,
                            to_document=len(target_functions),
                            batches_processed=0,
                            accepted_count=0,
                            skipped_count=0,
                            rejected_count=0,
                            failed_batches=len(batches),
                            applied=False,
                            batch_errors=[f"Health check failed: {exc}"],
                        )

                generation_service = GenerationService(provider)
                total_batches = len(batches)

                async def _batch_worker(batch: FunctionBatch) -> BatchGenerationResult:
                    batch_idx = next((i + 1 for i, b in enumerate(batches) if b.id == batch.id), 1)
                    if on_batch_start is not None:
                        on_batch_start(batch.id, len(batch.functions), batch_idx, total_batches)
                    try:
                        res = await generation_service.generate_batch(batch)
                        if on_batch_complete is not None:
                            on_batch_complete(
                                batch.id, len(batch.functions), batch_idx, total_batches, "success"
                            )
                        return res
                    except Exception:
                        if on_batch_complete is not None:
                            on_batch_complete(
                                batch.id, len(batch.functions), batch_idx, total_batches, "failed"
                            )
                        raise

                # Run bounded asynchronous worker pool
                results: list[BatchTaskResult[BatchGenerationResult]] = await run_batch_pool(
                    batches=batches,
                    worker_fn=_batch_worker,
                    max_concurrency=self.max_concurrency,
                    rate_limiter=self._rate_limiter,
                )

                # Check for failed batches (enforce atomic file modification)
                failed_tasks = [r for r in results if not r.success or r.data is None]
                if failed_tasks:
                    error_messages = [
                        f"Batch {t.batch_id}: {t.error}"
                        for t in failed_tasks
                        if t.error is not None
                    ]
                    stream.write(
                        f"\nError: {len(failed_tasks)} batch(es) failed to complete generation.\n"
                    )
                    for msg in error_messages:
                        stream.write(f"  - {msg}\n")
                    stream.write("Source modification aborted to guarantee atomic behavior.\n")

                    if raise_on_failure:
                        raise PipelineExecutionError(
                            f"{len(failed_tasks)} batch(es) failed: {', '.join(error_messages)}"
                        )

                    return DocumentationReport(
                        path=valid_path,
                        total_functions=len(all_functions),
                        already_documented=already_documented,
                        to_document=len(target_functions),
                        batches_processed=len(results),
                        accepted_count=0,
                        skipped_count=0,
                        rejected_count=0,
                        failed_batches=len(failed_tasks),
                        applied=False,
                        batch_errors=error_messages,
                    )

                # Aggregate successful generated documentations
                all_documentations: list[GeneratedDocumentation] = []
                for res in results:
                    if res.data:
                        all_documentations.extend(res.data.functions)

                # Semantic AST Validation
                val_report = self._validator.validate_all(target_functions, all_documentations)

                accepted_docs = [
                    item.documentation for item in val_report.accepted if item.documentation
                ]
                accepted_fns = [item.function for item in val_report.accepted]

                # Invoke validation summary callback before applying changes
                preliminary_report = DocumentationReport(
                    path=valid_path,
                    total_functions=len(all_functions),
                    already_documented=already_documented,
                    to_document=len(target_functions),
                    batches_processed=len(results),
                    accepted_count=len(val_report.accepted),
                    skipped_count=len(val_report.skipped),
                    rejected_count=len(val_report.rejected),
                    failed_batches=0,
                    applied=False,
                    validation_report=val_report,
                )
                if on_validation_complete is not None:
                    on_validation_complete(preliminary_report)

                # Handle case where no documentation was accepted
                if not accepted_docs:
                    if on_validation_complete is None:
                        stream.write(
                            f"\nNo documentation changes could be accepted for '{valid_path.name}'.\n"
                        )
                        if val_report.skipped:
                            stream.write(
                                f"  - {len(val_report.skipped)} function(s) were flagged uncertain.\n"
                            )
                        if val_report.rejected:
                            stream.write(
                                f"  - {len(val_report.rejected)} function(s) failed semantic validation.\n"
                            )

                    return preliminary_report

                # Create source edits from accepted functions
                edits = create_source_edits(
                    source=source,
                    functions=accepted_fns,
                    documentations=accepted_docs,
                )

                # Interactive preview and atomic disk modification
                applied = preview_and_apply(
                    path=valid_path,
                    original_source=source,
                    edits=edits,
                    interactive=interactive,
                    dry_run=dry_run,
                    stream=stream,
                    on_preview_diff=on_preview_diff,
                    on_prompt_confirmation=on_prompt_confirmation,
                )

                return DocumentationReport(
                    path=valid_path,
                    total_functions=len(all_functions),
                    already_documented=already_documented,
                    to_document=len(target_functions),
                    batches_processed=len(results),
                    accepted_count=len(val_report.accepted),
                    skipped_count=len(val_report.skipped),
                    rejected_count=len(val_report.rejected),
                    failed_batches=0,
                    applied=applied,
                    validation_report=val_report,
                )
            finally:
                if created_provider and hasattr(provider, "close"):
                    await provider.close()

    async def document_files(
        self,
        paths: Sequence[Path | str],
        *,
        interactive: bool = False,
        dry_run: bool = False,
        enforce_relative: bool = False,
        overwrite_existing: bool = False,
        raise_on_failure: bool = False,
        stream: TextIO = sys.stdout,
    ) -> list[DocumentationReport]:
        """Run the documentation pipeline across multiple files reusing provider resources.

        Args:
            paths: Collection of Python file paths to process.
            interactive: Whether to prompt for confirmation per file.
            dry_run: If True, previews diffs without modifying files on disk.
            enforce_relative: If True, requires all paths to be relative.
            overwrite_existing: If True, documents both documented and undocumented functions.
            raise_on_failure: If True, raises on any file pipeline failure.
            stream: Stream destination for output and confirmation prompts.

        Returns:
            List of DocumentationReport instances for all processed files.
        """
        reports: list[DocumentationReport] = []

        # Share a single provider instance across multi-file runs
        created_provider = False
        if self._provider is not None:
            shared_provider = self._provider
        else:
            shared_provider = GeminiProvider(api_key=self.api_key, model=self.model)
            created_provider = True

        service = (
            self
            if self._provider is shared_provider
            else DocumentationService(
                provider=shared_provider,
                api_key=self.api_key,
                model=self.model,
                max_concurrency=self.max_concurrency,
                requests_per_minute=self._rate_limiter.rpm,
                max_batch_tokens=self.max_batch_tokens,
                max_functions_per_batch=self.max_functions_per_batch,
                strict_raises=self.strict_raises,
            )
        )

        try:
            for path in paths:
                report = await service.document_file(
                    path=path,
                    interactive=interactive,
                    dry_run=dry_run,
                    enforce_relative=enforce_relative,
                    overwrite_existing=overwrite_existing,
                    raise_on_failure=raise_on_failure,
                    stream=stream,
                )
                reports.append(report)
        finally:
            if created_provider and hasattr(shared_provider, "close"):
                await shared_provider.close()

        return reports


__all__ = [
    "DocumentationReport",
    "DocumentationService",
    "PipelineExecutionError",
]
