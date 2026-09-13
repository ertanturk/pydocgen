"""Unit tests for GeminiProvider adapter, fast-fail health checks, and error handling."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from pydocgen.errors.exceptions import (
    AuthenticationError,
    ExitCode,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    RateLimitError,
)
from pydocgen.providers.gemini import (
    DEFAULT_MODEL,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_TIMEOUT,
    GeminiProvider,
    _clean_json_markdown,
    _extract_text_from_interaction,
    _map_provider_error,
)
from pydocgen.security.credentials import Credentials

VALID_KEY = "AIzaSyD-1234567890abcdefghijklmnop"


class SampleSchema(BaseModel):
    summary: str
    item_count: int


class ItemListSchema(BaseModel):
    items: list[str]


class TestMarkdownAndExtraction:
    """Tests for markdown stripping and interaction text extraction."""

    def test_clean_json_markdown_raw_json(self) -> None:
        raw = '{"summary": "test", "item_count": 5}'
        assert _clean_json_markdown(raw) == raw

    def test_clean_json_markdown_fenced_json(self) -> None:
        raw = '```json\n{"summary": "test", "item_count": 5}\n```'
        assert _clean_json_markdown(raw) == '{"summary": "test", "item_count": 5}'

    def test_clean_json_markdown_fenced_no_lang(self) -> None:
        raw = '```\n{"summary": "test", "item_count": 5}\n```'
        assert _clean_json_markdown(raw) == '{"summary": "test", "item_count": 5}'

    def test_clean_json_markdown_fenced_array(self) -> None:
        raw = '```json\n["item1", "item2"]\n```'
        assert _clean_json_markdown(raw) == '["item1", "item2"]'

    def test_clean_json_markdown_embedded_fences(self) -> None:
        raw = '```json\n{"doc": "```python\\ndef foo(): pass\\n```"}\n```'
        assert _clean_json_markdown(raw) == '{"doc": "```python\\ndef foo(): pass\\n```"}'

    def test_clean_json_markdown_surrounding_text(self) -> None:
        raw = (
            "Here is the generated output:\n"
            "```json\n"
            '{"summary": "hello"}\n'
            "```\n"
            "Let me know if you need changes!"
        )
        assert _clean_json_markdown(raw) == '{"summary": "hello"}'

    def test_clean_json_markdown_preamble_no_fence(self) -> None:
        raw = 'Output: {"summary": "hello"} Thanks!'
        assert _clean_json_markdown(raw) == '{"summary": "hello"}'

    def test_clean_json_markdown_empty(self) -> None:
        assert _clean_json_markdown("") == ""
        assert _clean_json_markdown("   \n\t  ") == ""

    def test_extract_text_output_text(self) -> None:
        interaction = MagicMock(output_text="Primary output text")
        assert _extract_text_from_interaction(interaction) == "Primary output text"

    def test_extract_text_steps_content(self) -> None:
        part = MagicMock(type="text", text="Step content text")
        step = MagicMock(type="model_output", content=[part])
        interaction = MagicMock(output_text=None, steps=[step])
        assert _extract_text_from_interaction(interaction) == "Step content text"

    def test_extract_text_steps_direct_text(self) -> None:
        step = MagicMock(type="text", text="Direct step text", content=None)
        interaction = MagicMock(output_text=None, steps=[step])
        assert _extract_text_from_interaction(interaction) == "Direct step text"

    def test_extract_text_legacy_outputs(self) -> None:
        step = MagicMock(type="text", text="Legacy output text")
        interaction = MagicMock(output_text=None, steps=None, outputs=[step])
        assert _extract_text_from_interaction(interaction) == "Legacy output text"

    def test_extract_text_candidates_fallback(self) -> None:
        part = MagicMock(text="Candidate part text")
        candidate = MagicMock(content=MagicMock(parts=[part]))
        interaction = MagicMock(output_text=None, steps=None, outputs=None, candidates=[candidate])
        assert _extract_text_from_interaction(interaction) == "Candidate part text"

    def test_extract_text_direct_text_fallback(self) -> None:
        interaction = MagicMock(
            output_text=None, steps=None, outputs=None, candidates=None, text="Fallback text"
        )
        assert _extract_text_from_interaction(interaction) == "Fallback text"

    def test_extract_text_empty(self) -> None:
        interaction = MagicMock(
            output_text=None, steps=None, outputs=None, candidates=None, text=None
        )
        assert _extract_text_from_interaction(interaction) == ""


class TestErrorMapping:
    """Tests verifying SDK and HTTP error translation to domain exceptions."""

    def test_map_timeout_error(self) -> None:
        exc = TimeoutError()
        mapped = _map_provider_error(exc, timeout=15.0)
        assert isinstance(mapped, ProviderTimeoutError)
        assert mapped.exit_code == ExitCode.TEMPFAIL
        assert "15.0s" in str(mapped)

    def test_map_httpx_timeout_error(self) -> None:
        exc = httpx.ReadTimeout("Read timed out")
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, ProviderTimeoutError)
        assert mapped.exit_code == ExitCode.TEMPFAIL

    def test_map_auth_error_401(self) -> None:
        exc = Exception("Unauthorized")
        exc.status_code = 401  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.exit_code == ExitCode.CONFIG

    def test_map_auth_error_403(self) -> None:
        exc = Exception("Forbidden")
        exc.status_code = 403  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.exit_code == ExitCode.CONFIG

    def test_map_auth_error_gemini_400_invalid_api_key(self) -> None:
        exc = Exception("API key not valid. Please pass a valid API key.")
        exc.status_code = 400  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, AuthenticationError)
        assert mapped.exit_code == ExitCode.CONFIG

    def test_map_rate_limit_429(self) -> None:
        exc = Exception("Quota exceeded")
        exc.status_code = 429  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, RateLimitError)
        assert mapped.exit_code == ExitCode.TEMPFAIL

    def test_map_rate_limit_resource_exhausted_string(self) -> None:
        exc = Exception("Resource has been exhausted (e.g. check quota)")
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, RateLimitError)
        assert mapped.exit_code == ExitCode.TEMPFAIL

    def test_map_bad_request_400(self) -> None:
        exc = Exception("Invalid prompt format")
        exc.status_code = 400  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, ProviderError)
        assert mapped.exit_code == ExitCode.DATAERR

    def test_map_server_error_503(self) -> None:
        exc = Exception("Service unavailable")
        exc.status_code = 503  # ty: ignore[unresolved-attribute]
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, ProviderError)
        assert mapped.exit_code == ExitCode.UNAVAILABLE

    def test_map_network_error(self) -> None:
        exc = httpx.ConnectError("Connection refused")
        mapped = _map_provider_error(exc)
        assert isinstance(mapped, ProviderError)
        assert mapped.exit_code == ExitCode.UNAVAILABLE

    def test_map_existing_pydocgen_error_identity(self) -> None:
        original = AuthenticationError("Key missing", exit_code=ExitCode.CONFIG)
        assert _map_provider_error(original) is original


class TestProviderInitialization:
    """Tests verifying GeminiProvider instantiation and credential validation."""

    def test_init_with_explicit_key(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        assert provider.model == DEFAULT_MODEL
        assert provider.timeout == DEFAULT_TIMEOUT
        assert provider.probe_timeout == DEFAULT_PROBE_TIMEOUT
        assert provider.fast_fail is True
        assert provider._verified is False

    @patch.object(Credentials, "get_api_key", return_value=VALID_KEY)
    def test_init_with_stored_key(self, mock_get_key: MagicMock) -> None:
        provider = GeminiProvider()
        assert provider._client is not None
        mock_get_key.assert_called_once()

    @patch.object(Credentials, "get_api_key", return_value=None)
    def test_init_missing_key_raises_authentication_error(self, mock_get_key: MagicMock) -> None:
        with pytest.raises(AuthenticationError, match="API key not found") as exc_info:
            GeminiProvider()
        assert exc_info.value.exit_code == ExitCode.CONFIG

    def test_init_invalid_key_raises_authentication_error(self) -> None:
        with pytest.raises(AuthenticationError, match="too short") as exc_info:
            GeminiProvider(api_key="short")
        assert exc_info.value.exit_code == ExitCode.CONFIG

    def test_init_whitespace_key_raises_authentication_error(self) -> None:
        with pytest.raises(AuthenticationError, match="cannot be empty or contain only whitespace"):
            GeminiProvider(api_key="   \n  ")


class TestFastFailHealthCheck:
    """Tests verifying fast-fail 'ok' probe behavior before real requests."""

    @pytest.mark.asyncio
    async def test_fast_fail_probe_succeeds(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        mock_interaction = MagicMock(status="completed", errors=None, output_text="ok")

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ) as mock_create:
            result = await provider.verify_connection()
            assert result is True
            assert provider._verified is True
            mock_create.assert_awaited_once_with(model=DEFAULT_MODEL, input="ok")

    @pytest.mark.asyncio
    async def test_fast_fail_probe_memoized_on_subsequent_calls(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        mock_interaction = MagicMock(status="completed", errors=None, output_text="ok")

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ) as mock_create:
            await provider.verify_connection()
            await provider.verify_connection()
            await provider.ping()
            # Only dispatched once!
            assert mock_create.await_count == 1

    @pytest.mark.asyncio
    async def test_fast_fail_probe_force_rerun(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        mock_interaction = MagicMock(status="completed", errors=None, output_text="ok")

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ) as mock_create:
            await provider.verify_connection()
            await provider.verify_connection(force=True)
            assert mock_create.await_count == 2

    @pytest.mark.asyncio
    async def test_fast_fail_probe_failure_raises_authentication_error(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        error_400 = Exception("API key not valid. Please pass a valid API key.")
        error_400.status_code = 400  # ty: ignore[unresolved-attribute]

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            side_effect=error_400,
        ):
            with pytest.raises(AuthenticationError, match="authentication failed") as exc_info:
                await provider.verify_connection()
            assert exc_info.value.exit_code == ExitCode.CONFIG
            assert provider._verified is False

    @pytest.mark.asyncio
    async def test_fast_fail_probe_failure_raises_rate_limit_error(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        error_429 = Exception("Resource has been exhausted (e.g. check quota)")
        error_429.status_code = 429  # ty: ignore[unresolved-attribute]

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            side_effect=error_429,
        ):
            with pytest.raises(RateLimitError, match="rate limit exceeded") as exc_info:
                await provider.verify_connection()
            assert exc_info.value.exit_code == ExitCode.TEMPFAIL

    @pytest.mark.asyncio
    async def test_fast_fail_probe_timeout_raises_provider_timeout_error(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, probe_timeout=0.01)

        async def slow_probe(**kwargs: Any) -> Any:
            await asyncio.sleep(0.5)
            return MagicMock()

        with (
            patch.object(
                provider._client.aio.interactions,
                "create",
                side_effect=slow_probe,
            ),
            pytest.raises(ProviderTimeoutError, match="timed out"),
        ):
            await provider.verify_connection(probe_timeout=0.01)

    @pytest.mark.asyncio
    async def test_fast_fail_probe_failed_status_raises_provider_error(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY)
        mock_interaction = MagicMock(
            status="failed",
            errors=[MagicMock(message="Service overloaded")],
            output_text=None,
        )

        with (
            patch.object(
                provider._client.aio.interactions,
                "create",
                new_callable=AsyncMock,
                return_value=mock_interaction,
            ),
            pytest.raises(ProviderError, match="Fast-fail health check probe failed"),
        ):
            await provider.verify_connection()

    @pytest.mark.asyncio
    async def test_fast_fail_concurrency_single_probe(self) -> None:
        """Verify multiple concurrent generate() calls execute the probe only ONCE."""
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=True)

        call_count = 0

        async def mock_create(**kwargs: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if kwargs.get("input") == "ok":
                await asyncio.sleep(0.05)  # Simulate network latency during probe
                return MagicMock(status="completed", errors=None, output_text="ok")
            return MagicMock(status="completed", errors=None, output_text="Real result")

        with patch.object(provider._client.aio.interactions, "create", side_effect=mock_create):
            tasks = [provider.generate(f"Prompt {i}") for i in range(10)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r == "Real result" for r in results)
            # 1 probe call + 10 real calls = 11 total calls
            assert call_count == 11

    @pytest.mark.asyncio
    async def test_fast_fail_disabled_does_not_send_probe(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=MagicMock(status="completed", errors=None, output_text="Result"),
        ) as mock_create:
            result = await provider.generate("Do something")
            assert result == "Result"
            # Exactly one call, which is the real request (no probe)
            assert mock_create.await_count == 1
            assert mock_create.call_args.kwargs["input"] == "Do something"


class TestGeneration:
    """Tests verifying prompt submission, schema enforcement, and output parsing."""

    @pytest.mark.asyncio
    async def test_generate_text_success(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        mock_interaction = MagicMock(
            status="completed", errors=None, output_text="def test(): pass"
        )

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ) as mock_create:
            result = await provider.generate("Generate docstring", system_instruction="Be concise")
            assert result == "def test(): pass"
            mock_create.assert_awaited_once_with(
                model=DEFAULT_MODEL,
                input="Generate docstring",
                system_instruction="Be concise",
                generation_config={"temperature": 0.1},
            )

    @pytest.mark.asyncio
    async def test_generate_structured_success(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        json_output = '```json\n{"summary": "Calculates factorial", "item_count": 1}\n```'
        mock_interaction = MagicMock(status="completed", errors=None, output_text=json_output)

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ) as mock_create:
            result = await provider.generate("Analyze", response_schema=SampleSchema)
            assert isinstance(result, SampleSchema)
            assert result.summary == "Calculates factorial"
            assert result.item_count == 1

            # Ensure response_format passed the model JSON schema dict, NOT the class
            call_kwargs = mock_create.call_args.kwargs
            assert "response_format" in call_kwargs
            rf = call_kwargs["response_format"]
            assert rf["type"] == "text"
            assert rf["mime_type"] == "application/json"
            assert rf["schema"] == SampleSchema.model_json_schema()

    @pytest.mark.asyncio
    async def test_generate_empty_response_raises_provider_response_error(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        mock_interaction = MagicMock(status="completed", errors=None, output_text="   \n")

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ):
            with pytest.raises(ProviderResponseError, match="empty response") as exc_info:
                await provider.generate("Prompt")
            assert exc_info.value.exit_code == ExitCode.DATAERR

    @pytest.mark.asyncio
    async def test_generate_schema_validation_failure_raises(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        bad_json = '{"wrong_field": "test"}'
        mock_interaction = MagicMock(status="completed", errors=None, output_text=bad_json)

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ):
            with pytest.raises(ProviderResponseError, match="Failed to validate response") as exc:
                await provider.generate("Prompt", response_schema=SampleSchema)
            assert exc.value.exit_code == ExitCode.DATAERR

    @pytest.mark.asyncio
    async def test_generate_fallback_json_extraction_from_text(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        messy_output = 'Here is the response: {"summary": "Extracted from text", "item_count": 3} hope this helps!'
        mock_interaction = MagicMock(status="completed", errors=None, output_text=messy_output)

        with patch.object(
            provider._client.aio.interactions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_interaction,
        ):
            result = await provider.generate("Prompt", response_schema=SampleSchema)
            assert result.summary == "Extracted from text"
            assert result.item_count == 3


class TestLifecycleAndContextManager:
    """Tests verifying async context manager and resource disposal."""

    @pytest.mark.asyncio
    async def test_context_manager_probes_and_closes(self) -> None:
        mock_probe = MagicMock(status="completed", errors=None, output_text="ok")

        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=True)
        provider._client.aio.interactions.create = AsyncMock(return_value=mock_probe)
        provider._client.aio.aclose = AsyncMock()

        async with provider as p:
            assert p._verified is True
            provider._client.aio.interactions.create.assert_awaited_once_with(
                model=DEFAULT_MODEL, input="ok"
            )

        provider._client.aio.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_fallback_to_sync_close(self) -> None:
        provider = GeminiProvider(api_key=VALID_KEY, fast_fail=False)
        provider._client.aio.aclose = None  # ty: ignore[invalid-assignment]
        provider._client.close = MagicMock()

        await provider.close()
        provider._client.close.assert_called_once()
