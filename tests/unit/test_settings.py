"""Unit tests for configuration settings and constants."""

from __future__ import annotations

import pydocgen.config as config_pkg
import pydocgen.config.settings as settings


class TestSettings:
    """Tests verifying centralized settings and configuration constants."""

    def test_service_and_credential_settings(self) -> None:
        assert settings.SERVICE_NAME == "pydocgen"
        assert settings.ACCOUNT_NAME == "gemini-api-key"
        assert "GEMINI_API_KEY" in settings.ENV_API_KEY_NAMES
        assert "GOOGLE_API_KEY" in settings.ENV_API_KEY_NAMES
        assert settings.MIN_KEY_LENGTH > 0
        assert "\n" in settings.INVALID_KEY_CONTROL_CHARS
        assert "\0" in settings.INVALID_KEY_CONTROL_CHARS
        assert settings.KEY_MASK_PREFIX_LENGTH == 4
        assert settings.KEY_MASK_SUFFIX_LENGTH == 4

    def test_analysis_settings(self) -> None:
        assert settings.PYTHON_FILE_EXTENSION == ".py"
        assert settings.DEFAULT_FILE_ENCODING == "utf-8-sig"
        assert settings.DEFAULT_UNKNOWN_FILENAME == "<unknown>"
        assert settings.FUNCTION_ID_HASH_LENGTH == 12

    def test_batching_settings(self) -> None:
        assert settings.CHARS_PER_TOKEN == 3.5
        assert settings.DEFAULT_PROMPT_OVERHEAD_TOKENS == 500
        assert settings.PER_FUNCTION_OVERHEAD_TOKENS == 40
        assert settings.DEFAULT_MAX_BATCH_TOKENS == 12000
        assert settings.DEFAULT_MAX_FUNCTIONS_PER_BATCH == 25
        assert settings.BATCH_ID_HASH_LENGTH == 8

    def test_concurrency_settings(self) -> None:
        assert settings.DEFAULT_RPM == 60
        assert settings.DEFAULT_RATE_LIMIT_WINDOW_SECONDS == 60.0
        assert settings.DEFAULT_MAX_CONCURRENCY == 3
        assert settings.DEFAULT_MAX_RETRIES == 3
        assert settings.DEFAULT_INITIAL_BACKOFF == 1.0
        assert settings.DEFAULT_BACKOFF_MULTIPLIER == 2.0

    def test_telemetry_settings(self) -> None:
        assert settings.DEFAULT_LOGGER_ROOT == "pydocgen"
        assert settings.DEFAULT_LOG_LEVEL == "INFO"
        assert settings.LOG_FORMAT_CONSOLE == "console"
        assert settings.LOG_FORMAT_JSON == "json"
        assert settings.DEFAULT_LOG_FORMAT == "console"
        assert "%(asctime)s" in settings.DEFAULT_CONSOLE_LOG_FORMAT
        assert "%Y-%m-%d" in settings.DEFAULT_LOG_DATE_FORMAT
        assert settings.DEFAULT_REDACTED_MASK == "[REDACTED]"
        assert "levelname" in settings.RESERVED_LOG_RECORD_KEYS
        assert "args" in settings.RESERVED_LOG_RECORD_KEYS
        assert "api_key" in settings.SENSITIVE_KEY_NAMES

    def test_provider_settings(self) -> None:
        assert settings.DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
        assert settings.FAST_FAIL_PROBE_PROMPT == "ok"
        assert settings.DEFAULT_PROBE_TIMEOUT == 10.0
        assert settings.DEFAULT_FAST_FAIL is True
        assert settings.DEFAULT_MAX_OUTPUT_TOKENS == 8192
        assert settings.DEFAULT_TEMPERATURE == 0.1
        assert settings.DEFAULT_REQUEST_TIMEOUT == 60.0

    def test_formatting_and_validation_settings(self) -> None:
        assert settings.DEFAULT_INDENT_WIDTH == 4
        assert settings.GOOGLE_SECTION_ARGS == "Args:"
        assert settings.GOOGLE_SECTION_RETURNS == "Returns:"
        assert settings.GOOGLE_SECTION_RAISES == "Raises:"
        assert settings.DEFAULT_STRICT_RAISES is True

    def test_editing_settings(self) -> None:
        assert settings.DEFAULT_INDENTATION == "    "
        assert settings.TEMP_FILE_SUFFIX == ".pydocgen.tmp"
        assert settings.DIFF_SEPARATOR_CHAR == "─"
        assert settings.DIFF_SEPARATOR_LENGTH == 60
        assert "y" in settings.CONFIRMATION_AFFIRMATIVE
        assert settings.COLOR_RED == "\033[31m"
        assert settings.COLOR_GREEN == "\033[32m"
        assert settings.COLOR_CYAN == "\033[36m"
        assert settings.COLOR_BOLD == "\033[1m"
        assert settings.COLOR_RESET == "\033[0m"

    def test_package_exports_consistency(self) -> None:
        assert set(settings.__all__) == set(config_pkg.__all__)
        for name in settings.__all__:
            assert hasattr(settings, name)
            assert hasattr(config_pkg, name)
            assert getattr(settings, name) is getattr(config_pkg, name)

    def test_exception_exit_codes(self) -> None:
        from pydocgen.errors.exceptions import (
            AuthenticationError,
            BatchError,
            ConcurrencyError,
            ConfigurationError,
            ExitCode,
            FileValidationError,
            FormattingError,
            GenerationError,
            InvalidCredentialError,
            ParseError,
            PipelineExecutionError,
            ProviderError,
            ProviderResponseError,
            ProviderTimeoutError,
            PyDocGenError,
            PydocgenError,
            RateLimitError,
            SourceEditError,
            ValidationError,
        )

        assert issubclass(PydocgenError, PyDocGenError)
        assert issubclass(FileValidationError, PydocgenError)
        assert issubclass(ParseError, PydocgenError)
        assert issubclass(BatchError, PydocgenError)
        assert issubclass(ConcurrencyError, PydocgenError)

        assert FileValidationError("test").exit_code == ExitCode.NOINPUT
        assert ParseError("test").exit_code == ExitCode.DATAERR
        assert BatchError("test").exit_code == ExitCode.DATAERR
        assert ConcurrencyError("test").exit_code == ExitCode.SOFTWARE
        assert ConfigurationError("test").exit_code == ExitCode.CONFIG
        assert InvalidCredentialError("test").exit_code == ExitCode.USAGE
        assert AuthenticationError("test").exit_code == ExitCode.CONFIG
        assert RateLimitError("test").exit_code == ExitCode.TEMPFAIL
        assert ProviderTimeoutError("test").exit_code == ExitCode.TEMPFAIL
        assert ProviderResponseError("test").exit_code == ExitCode.DATAERR
        assert ProviderError("test").exit_code == ExitCode.UNAVAILABLE
        assert SourceEditError("test").exit_code == ExitCode.SOFTWARE
        assert PipelineExecutionError("test").exit_code == ExitCode.SOFTWARE
        assert FormattingError("test").exit_code == ExitCode.DATAERR
        assert ValidationError("test").exit_code == ExitCode.DATAERR
        assert GenerationError("test").exit_code == ExitCode.SOFTWARE
