"""Asynchronous client adapter for the Gemini Interactions API with fast-fail health checks."""

from __future__ import annotations

import asyncio
import json
import re
from types import TracebackType
from typing import Any, Self, TypeVar, overload

from google import genai

try:
    from google.genai._gaos.lib import compat_errors
except ImportError:  # pragma: no cover
    compat_errors = None  # type: ignore[assignment]

import httpx
from pydantic import BaseModel, ValidationError

from pydocgen.config.settings import (
    DEFAULT_FAST_FAIL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_PROBE_TIMEOUT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    FAST_FAIL_PROBE_PROMPT,
)
from pydocgen.errors.exceptions import (
    AuthenticationError,
    ExitCode,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    PyDocGenError,
    PydocgenError,
    RateLimitError,
)
from pydocgen.security.credentials import Credentials, InvalidCredentialError
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = DEFAULT_GEMINI_MODEL
DEFAULT_TIMEOUT = DEFAULT_REQUEST_TIMEOUT

T = TypeVar("T", bound=BaseModel)


def _clean_json_markdown(text: str) -> str:
    """Strip markdown code fences (```json ... ```) and formatting from model output."""
    trimmed = text.strip()
    if not trimmed:
        return ""

    # Case 1: Entire string is wrapped in a code fence (```json ... ``` or ``` ... ```)
    if trimmed.startswith("```"):
        lines = trimmed.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()

    # Case 2: Code fence is embedded inside surrounding explanatory text
    match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```(?:\s*$|\s*[^\`]*$)",
        trimmed,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        if (candidate.startswith("{") and candidate.endswith("}")) or (
            candidate.startswith("[") and candidate.endswith("]")
        ):
            return candidate

    # Case 3: JSON object or array with preamble or postamble without backticks
    brace_match = re.search(r"(\{.*\}|\[.*\])", trimmed, re.DOTALL)
    if brace_match:
        return brace_match.group(1).strip()

    return trimmed


def _extract_text_from_interaction(interaction: Any) -> str:
    """Extract model output text across various Google GenAI Interaction response formats."""
    # 1. Primary Interactions API output helper property
    output_text = getattr(interaction, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # 2. Modern GAOS interaction steps
    steps = getattr(interaction, "steps", None)
    if isinstance(steps, list):
        collected: list[str] = []
        for step in steps:
            if getattr(step, "type", None) == "text" and hasattr(step, "text"):
                val = step.text
                if isinstance(val, str):
                    collected.append(val)
                continue
            content = getattr(step, "content", None)
            if isinstance(content, list):
                for item in content:
                    if getattr(item, "type", None) == "text":
                        text = getattr(item, "text", None)
                        if isinstance(text, str):
                            collected.append(text)
            elif isinstance(content, str):
                collected.append(content)
        joined = "".join(collected).strip()
        if joined:
            return joined

    # 3. Legacy outputs attribute
    outputs = getattr(interaction, "outputs", None)
    if isinstance(outputs, list):
        collected = []
        for step in outputs:
            if getattr(step, "type", None) == "text":
                val = getattr(step, "text", None)
                if isinstance(val, str):
                    collected.append(val)
            elif hasattr(step, "content"):
                c = step.content
                if isinstance(c, str):
                    collected.append(c)
        joined = "".join(collected).strip()
        if joined:
            return joined

    # 4. Standard Gemini candidates format fallback
    candidates = getattr(interaction, "candidates", None)
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        if content and hasattr(content, "parts"):
            parts = [getattr(p, "text", "") for p in content.parts if getattr(p, "text", None)]
            joined = "".join(parts).strip()
            if joined:
                return joined

    direct_text = getattr(interaction, "text", None)
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    return ""


def _extract_status_and_message(exc: Exception) -> tuple[int | None, str]:
    """Extract HTTP status code and human-readable message from SDK or HTTP errors."""
    status_code: int | None = getattr(exc, "status_code", getattr(exc, "code", None))
    if not isinstance(status_code, int):
        resp = getattr(exc, "response", None)
        if resp is not None and hasattr(resp, "status_code") and isinstance(resp.status_code, int):
            status_code = resp.status_code

    message: str = getattr(exc, "message", None) or str(exc)
    return status_code, message


def _map_provider_error(exc: Exception, timeout: float | None = None) -> Exception:
    """Map Google GenAI SDK, compat errors, and network errors to domain exceptions."""
    if isinstance(exc, (PydocgenError, PyDocGenError)):
        return exc

    compat_timeout = getattr(compat_errors, "APITimeoutError", ()) if compat_errors else ()
    compat_conn = getattr(compat_errors, "APIConnectionError", ()) if compat_errors else ()

    if isinstance(exc, (asyncio.TimeoutError, httpx.TimeoutException, compat_timeout)):
        timeout_desc = f" after {timeout}s" if timeout else ""
        return ProviderTimeoutError(
            f"Gemini API timed out{timeout_desc}: {exc}",
            exit_code=ExitCode.TEMPFAIL,
        )

    status_code, message = _extract_status_and_message(exc)
    msg_upper = message.upper()

    # Google Gemini returns 400 with 'API_KEY_INVALID' or 'API key not valid' on invalid API keys,
    # or HTTP 401/403 on authorization/permission failures.
    if (
        status_code in (401, 403)
        or "API_KEY_INVALID" in msg_upper
        or "API KEY NOT VALID" in msg_upper
        or "PERMISSION_DENIED" in msg_upper
    ):
        return AuthenticationError(
            f"Gemini authentication failed ({status_code}): {message}",
            exit_code=ExitCode.CONFIG,
        )

    # Rate limiting / quota exhaustion
    if (
        status_code == 429
        or "RESOURCE_EXHAUSTED" in msg_upper
        or "RESOURCE HAS BEEN EXHAUSTED" in msg_upper
        or "RATE_LIMIT" in msg_upper
        or "RATE LIMIT" in msg_upper
        or "QUOTA" in msg_upper
    ):
        return RateLimitError(
            f"Gemini rate limit exceeded: {message}",
            exit_code=ExitCode.TEMPFAIL,
        )

    # Bad request / client payload errors
    if status_code == 400:
        return ProviderError(
            f"Gemini bad request ({status_code}): {message}",
            exit_code=ExitCode.DATAERR,
        )

    # Other API status errors (e.g. 500, 502, 503)
    if status_code is not None:
        return ProviderError(
            f"Gemini API error ({status_code}): {message}",
            exit_code=ExitCode.UNAVAILABLE,
        )

    # Network / transport errors
    if isinstance(exc, (httpx.NetworkError, compat_conn)):
        return ProviderError(
            f"Gemini connection error: {exc}",
            exit_code=ExitCode.UNAVAILABLE,
        )

    return ProviderError(
        f"Unexpected error calling Gemini: {exc}",
        exit_code=ExitCode.UNAVAILABLE,
    )


class GeminiProvider:
    """Asynchronous client adapter for the Gemini Interactions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        fast_fail: bool = DEFAULT_FAST_FAIL,
        probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
    ) -> None:
        """Initialize the Gemini client adapter.

        Args:
            api_key: Optional explicit API key. If omitted, fetched from Credentials.
            model: Gemini model identifier. Defaults to 'gemini-3.8-flash'.
            timeout: Request deadline in seconds for regular generation requests.
            temperature: Sampling temperature for model generation.
            fast_fail: When True, verifies connectivity and credentials with an
                'ok' probe before proceeding with the first real generation request.
            probe_timeout: Deadline in seconds for the fast-fail probe request.

        Raises:
            AuthenticationError: If the API key cannot be resolved or is invalid.
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.fast_fail = fast_fail
        self.probe_timeout = probe_timeout

        resolved_key = api_key or Credentials.get_api_key()
        if not resolved_key:
            raise AuthenticationError(
                "Gemini API key not found. Provide it explicitly, set the "
                "GEMINI_API_KEY environment variable, or store it via OS keyring.",
                exit_code=ExitCode.CONFIG,
            )

        try:
            resolved_key = Credentials._validate_api_key(resolved_key)
        except InvalidCredentialError as exc:
            raise AuthenticationError(
                f"Invalid Gemini API key: {exc}",
                exit_code=ExitCode.CONFIG,
            ) from exc

        self._client = genai.Client(api_key=resolved_key)
        self._verified: bool = False
        self._verify_lock: asyncio.Lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Enter async context, performing fast-fail health check if enabled."""
        if self.fast_fail and not self._verified:
            await self.verify_connection()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context and release resources."""
        await self.close()

    async def close(self) -> None:
        """Release underlying async HTTP sessions and client resources."""
        aio_client = getattr(self._client, "aio", None)
        aclose_fn = getattr(aio_client, "aclose", None)
        if callable(aclose_fn):
            await aclose_fn()
        elif hasattr(self._client, "close") and callable(self._client.close):
            self._client.close()

    async def ping(self, *, probe_timeout: float | None = None) -> bool:
        """Alias for verify_connection."""
        return await self.verify_connection(probe_timeout=probe_timeout)

    async def verify_connection(
        self,
        *,
        force: bool = False,
        probe_timeout: float | None = None,
    ) -> bool:
        """Send a lightweight probe ('ok') request to verify connectivity and credentials.

        Args:
            force: If True, bypasses the memoized status and re-probes the API.
            probe_timeout: Timeout in seconds for the probe. Defaults to self.probe_timeout.

        Returns:
            True if the probe succeeds.

        Raises:
            AuthenticationError: On bad or missing credentials (ExitCode 78).
            RateLimitError: On quota or RPM exhaustion (ExitCode 75).
            ProviderTimeoutError: If the deadline expires (ExitCode 75).
            ProviderError: On upstream server or network issues (ExitCode 69).
        """
        if self._verified and not force:
            return True

        timeout = probe_timeout if probe_timeout is not None else self.probe_timeout

        async with self._verify_lock:
            if self._verified and not force:
                return True

            logger.debug(
                "Dispatching fast-fail probe to Gemini",
                extra={"model": self.model, "probe_timeout": timeout},
            )

            try:
                probe_interaction = await asyncio.wait_for(
                    self._client.aio.interactions.create(
                        model=self.model,
                        input=FAST_FAIL_PROBE_PROMPT,
                    ),
                    timeout=timeout,
                )
            except Exception as exc:
                mapped = _map_provider_error(exc, timeout=timeout)
                logger.error(
                    "Fast-fail health check probe failed",
                    extra={"model": self.model, "error": str(mapped)},
                )
                raise mapped from exc

            status = getattr(probe_interaction, "status", None)
            errors_list = getattr(probe_interaction, "errors", None)
            if status in ("failed", "cancelled") or errors_list:
                err_msg = "; ".join(getattr(e, "message", str(e)) for e in (errors_list or []))
                raise ProviderError(
                    f"Fast-fail health check probe failed with status '{status}': {err_msg}",
                    exit_code=ExitCode.UNAVAILABLE,
                )

            self._verified = True
            logger.info(
                "Fast-fail health check probe succeeded",
                extra={"model": self.model},
            )
            return True

    @overload
    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: type[T],
    ) -> T: ...

    @overload
    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: None = None,
    ) -> str: ...

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        response_schema: type[T] | None = None,
    ) -> T | str:
        """Submit a prompt to the Interactions API and return validated data or text.

        If fast-fail is enabled, verifies connectivity with a lightweight probe before
        dispatching the real request.

        Args:
            prompt: Text prompt / batch description.
            system_instruction: Optional system instruction overriding defaults.
            response_schema: Optional Pydantic model class to constrain and validate output.

        Returns:
            An instance of response_schema if provided, otherwise the raw text string.

        Raises:
            AuthenticationError: On bad or missing credentials (ExitCode 78).
            RateLimitError: On quota or RPM exhaustion (ExitCode 75).
            ProviderTimeoutError: If the deadline expires (ExitCode 75).
            ProviderResponseError: If the response is empty or fails validation (ExitCode 65).
            ProviderError: On upstream server issues (ExitCode 69).
        """
        # Fast fail health check before proceeding with the real request
        if self.fast_fail and not self._verified:
            await self.verify_connection()

        request_params: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
        }

        if system_instruction:
            request_params["system_instruction"] = system_instruction

        if self.temperature is not None:
            request_params["generation_config"] = {"temperature": self.temperature}

        if response_schema is not None:
            schema_dict = response_schema.model_json_schema()
            # Complex schemas with nested $defs or arbitrary additionalProperties (such as
            # dynamic dictionaries or nested model hierarchies) cause Gemini Interactions
            # API constrained-decoding grammar to loop or truncate output. In those cases,
            # we enforce JSON mode via mime_type="application/json" and let client-side
            # Pydantic model_validate_json perform exact schema validation.
            has_unsupported_grammar = bool(
                schema_dict.get("$defs")
                or any("additionalProperties" in str(v) for v in schema_dict.values())
            )
            rf: dict[str, Any] = {
                "type": "text",
                "mime_type": "application/json",
            }
            if not has_unsupported_grammar:
                rf["schema"] = schema_dict
            request_params["response_format"] = rf

        try:
            interaction = await asyncio.wait_for(
                self._client.aio.interactions.create(**request_params),
                timeout=self.timeout,
            )
        except Exception as exc:
            mapped_exc = _map_provider_error(exc, timeout=self.timeout)
            raise mapped_exc from exc

        raw_text = _extract_text_from_interaction(interaction)

        if not raw_text:
            status = getattr(interaction, "status", None)
            errors_list = getattr(interaction, "errors", None)
            if errors_list:
                err_msg = "; ".join(getattr(e, "message", str(e)) for e in errors_list)
                raise ProviderResponseError(
                    f"Gemini interaction status '{status}' with errors: {err_msg}",
                    exit_code=ExitCode.DATAERR,
                )
            if status in ("failed", "cancelled"):
                raise ProviderResponseError(
                    f"Gemini interaction ended with status '{status}'.",
                    exit_code=ExitCode.DATAERR,
                )
            raise ProviderResponseError(
                "Gemini returned an empty response.",
                exit_code=ExitCode.DATAERR,
            )

        if response_schema is None:
            return raw_text

        cleaned_json = _clean_json_markdown(raw_text)
        try:
            return response_schema.model_validate_json(cleaned_json)
        except (ValidationError, json.JSONDecodeError) as exc:
            # Fallback: try finding first JSON object {...} or array [...] if preamble/postamble exists
            fallback_match = re.search(r"(\{.*\}|\[.*\])", cleaned_json, re.DOTALL)
            if fallback_match:
                try:
                    return response_schema.model_validate_json(fallback_match.group(1))
                except ValidationError, json.JSONDecodeError:
                    pass
            raise ProviderResponseError(
                f"Failed to validate response against {response_schema.__name__}: {exc}",
                exit_code=ExitCode.DATAERR,
            ) from exc


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROBE_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "FAST_FAIL_PROBE_PROMPT",
    "GeminiProvider",
    "_clean_json_markdown",
    "_extract_text_from_interaction",
    "_map_provider_error",
]
