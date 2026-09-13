"""Generation service orchestrating prompt synthesis and result reconciliation."""

from __future__ import annotations

from typing import Any

from pydocgen.batching.models import FunctionBatch
from pydocgen.errors.exceptions import ExitCode, ProviderResponseError
from pydocgen.generation.models import BatchGenerationResult, GeneratedDocumentation
from pydocgen.generation.prompts import SYSTEM_INSTRUCTION, build_batch_prompt
from pydocgen.providers.gemini import GeminiProvider
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


class GenerationService:
    """Service that manages prompt generation and response reconciliation with the AI provider."""

    def __init__(self, provider: GeminiProvider | Any) -> None:
        """Initialize the service with an active AI provider instance.

        Args:
            provider: Configured provider adapter with an async generate method.

        Raises:
            ValueError: If provider is None or lacks a generate method.
        """
        if provider is None:
            raise ValueError("provider cannot be None.")
        if not hasattr(provider, "generate") or not callable(provider.generate):
            raise ValueError("provider must implement an async 'generate' method.")
        self._provider = provider

    async def generate_batch(self, batch: FunctionBatch) -> BatchGenerationResult:
        """Generate structured documentation for a single batch of functions.

        Performs output reconciliation:
        - Ensures result.batch_id strictly matches batch.id.
        - Appends fallback status='uncertain' entries for any omitted functions.
        - Strips any hallucinated functions not present in the input batch.
        - Preserves deterministic function ordering matching the input batch.

        Args:
            batch: Target function batch.

        Returns:
            Validated BatchGenerationResult containing results for all functions in the batch.

        Raises:
            TypeError: If batch is not a FunctionBatch instance.
            ProviderResponseError: If the provider returns an unrecognized response type.
        """
        if not isinstance(batch, FunctionBatch):
            raise TypeError(f"batch must be a FunctionBatch, got {type(batch).__name__}.")

        # If batch has no functions, return an empty result immediately without invoking provider
        if not batch.functions:
            logger.debug(
                "Skipping provider generation for empty batch",
                extra={"batch_id": batch.id},
            )
            return BatchGenerationResult(batch_id=batch.id, functions=[])

        prompt = build_batch_prompt(batch)

        logger.debug(
            "Dispatching generation request to Gemini",
            extra={"batch_id": batch.id, "function_count": len(batch.functions)},
        )

        with logger.timed("generation.generate_batch", batch_id=batch.id):
            with logger.timed("provider_generate", batch_id=batch.id):
                raw_result = await self._provider.generate(
                    prompt=prompt,
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_schema=BatchGenerationResult,
                )

            if isinstance(raw_result, BatchGenerationResult):
                result = raw_result
            elif isinstance(raw_result, str):
                result = BatchGenerationResult.model_validate_json(raw_result)
            elif isinstance(raw_result, dict):
                result = BatchGenerationResult.model_validate(raw_result)
            else:
                raise ProviderResponseError(
                    f"Unexpected response type from provider: {type(raw_result).__name__}",
                    exit_code=ExitCode.DATAERR,
                )

            # Output reconciliation
            reconciled = self._reconcile_batch_result(batch, result)

        logger.info(
            "Batch generation completed",
            extra={
                "batch_id": batch.id,
                "total": len(reconciled.functions),
                "documented": sum(1 for f in reconciled.functions if f.status == "documented"),
                "uncertain": sum(1 for f in reconciled.functions if f.status == "uncertain"),
            },
        )
        return reconciled

    def _reconcile_batch_result(
        self,
        batch: FunctionBatch,
        result: BatchGenerationResult,
    ) -> BatchGenerationResult:
        """Ensure every requested function ID exists in the result, ordered by the batch."""
        # Enforce batch_id integrity
        if result.batch_id != batch.id:
            logger.warning(
                "Batch ID mismatch in provider response; correcting to requested batch_id",
                extra={"requested_batch_id": batch.id, "returned_batch_id": result.batch_id},
            )
            result.batch_id = batch.id

        requested_ids_list = [fn.id for fn in batch.functions]
        requested_ids_set = set(requested_ids_list)

        # Filter out hallucinated IDs and deduplicate returned functions
        returned_by_id: dict[str, GeneratedDocumentation] = {}
        hallucinated_count = 0

        for doc in result.functions:
            if doc.function_id not in requested_ids_set:
                hallucinated_count += 1
                continue
            if doc.function_id in returned_by_id:
                existing = returned_by_id[doc.function_id]
                # Prefer documented status over uncertain if duplicate
                if existing.status != "documented" and doc.status == "documented":
                    returned_by_id[doc.function_id] = doc
                continue
            returned_by_id[doc.function_id] = doc

        if hallucinated_count > 0:
            logger.warning(
                "Stripped hallucinated function IDs not requested in batch",
                extra={"batch_id": batch.id, "dropped_count": hallucinated_count},
            )

        # Create uncertain fallbacks for missing functions
        missing_ids = [fn_id for fn_id in requested_ids_list if fn_id not in returned_by_id]
        if missing_ids:
            logger.warning(
                "Gemini omitted functions from the batch response; creating uncertain fallbacks",
                extra={"batch_id": batch.id, "missing_ids": missing_ids},
            )
            for fn_id in missing_ids:
                returned_by_id[fn_id] = GeneratedDocumentation(
                    function_id=fn_id,
                    status="uncertain",
                    reason="Model response omitted this function from the output.",
                )

        # Guarantee exact order matching the input batch
        result.functions = [returned_by_id[fn_id] for fn_id in requested_ids_list]

        return result
