"""Unit tests for the pydocgen telemetry and structured logging subsystem."""

import io
import json
import logging
from collections.abc import Generator
from pathlib import Path

import pytest

from pydocgen.telemetry import (
    ConsoleFormatter,
    JsonFormatter,
    SensitiveDataFilter,
    TelemetryLogger,
    bind_context,
    clear_context,
    configure_telemetry,
    get_context,
    get_logger,
    redact_sensitive_data,
    redact_sensitive_string,
    reset_telemetry,
    set_context,
    timer,
    update_context,
)


@pytest.fixture(autouse=True)
def clean_telemetry_state() -> Generator[None]:
    """Ensure each test runs with isolated context and reset logger."""
    clear_context()
    yield
    clear_context()
    reset_telemetry()


class TestContextTracking:
    """Test contextvars tracking across execution blocks."""

    def test_default_context_is_empty(self) -> None:
        assert get_context() == {}

    def test_set_and_get_context(self) -> None:
        set_context({"request_id": "req-123", "user": "alice"})
        ctx = get_context()
        assert ctx == {"request_id": "req-123", "user": "alice"}

    def test_update_context(self) -> None:
        set_context({"a": 1})
        update_context(b=2, c=3)
        assert get_context() == {"a": 1, "b": 2, "c": 3}

    def test_clear_context(self) -> None:
        set_context({"key": "val"})
        clear_context()
        assert get_context() == {}

    def test_bind_context_manager(self) -> None:
        set_context({"base": "value"})
        with bind_context(temp="temp_value", run_id=42):
            ctx = get_context()
            assert ctx == {"base": "value", "temp": "temp_value", "run_id": 42}

        # Restores previous context after with-block
        assert get_context() == {"base": "value"}

    def test_bind_context_nested(self) -> None:
        with bind_context(level=1):
            assert get_context()["level"] == 1
            with bind_context(level=2, inner="yes"):
                assert get_context()["level"] == 2
                assert get_context()["inner"] == "yes"
            assert get_context()["level"] == 1
            assert "inner" not in get_context()

    def test_bind_context_restores_on_exception(self) -> None:
        set_context({"initial": True})
        with pytest.raises(RuntimeError), bind_context(transient="val"):
            raise RuntimeError("Boom")
        assert get_context() == {"initial": True}


class TestSensitiveDataFilter:
    """Test redacting secrets from messages, dicts, and log records."""

    def test_redact_google_api_key_in_string(self) -> None:
        raw = "Error connecting with key AIzaSyD-1234567890abcdefghijklmnop to service"
        redacted = redact_sensitive_string(raw)
        assert "AIza...mnop" in redacted
        assert "1234567890abcdefghijkl" not in redacted

    def test_redact_bearer_token(self) -> None:
        raw = "Authorization: Bearer my_secret_token_12345"
        redacted = redact_sensitive_string(raw)
        assert redacted == "Authorization: Bearer [REDACTED]"

    def test_redact_sensitive_dict(self) -> None:
        data = {
            "username": "admin",
            "api_key": "AIzaSyD-1234567890abcdefghijklmnop",
            "password": "super_secret_password",
            "nested": {
                "token": "secret_jwt_token",
                "normal": "regular_value",
            },
        }
        redacted = redact_sensitive_data(data)
        assert redacted["username"] == "admin"
        assert redacted["api_key"] == "AIza...mnop"
        assert redacted["password"] == "[REDACTED]"
        assert redacted["nested"]["token"] == "[REDACTED]"
        assert redacted["nested"]["normal"] == "regular_value"

    def test_filter_modifies_log_record(self) -> None:
        log_filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Using key AIzaSyD-1234567890abcdefghijklmnop",
            args=(),
            exc_info=None,
        )
        assert log_filter.filter(record) is True
        assert "AIza...mnop" in record.msg
        assert "1234567890abcdefghijkl" not in record.msg


class TestFormatters:
    """Test JsonFormatter and ConsoleFormatter output generation."""

    def test_json_formatter_valid_json_and_keys(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="pydocgen.test",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Processing module %s",
            args=("main.py",),
            exc_info=None,
        )
        record.custom_key = "custom_value"  # type: ignore[attr-defined]

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "pydocgen.test"
        assert parsed["message"] == "Processing module main.py"
        assert "timestamp" in parsed
        assert parsed["context"]["custom_key"] == "custom_value"

    def test_json_formatter_with_exception(self) -> None:
        formatter = JsonFormatter()
        try:
            raise ValueError("Test failure")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="pydocgen.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=55,
            msg="An error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        parsed = json.loads(output)

        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test failure"
        assert len(parsed["exception"]["stack_trace"]) > 0

    def test_console_formatter_output(self) -> None:
        formatter = ConsoleFormatter()
        record = logging.LogRecord(
            name="pydocgen.analysis",
            level=logging.INFO,
            pathname=__file__,
            lineno=100,
            msg="Functions extracted",
            args=(),
            exc_info=None,
        )
        record.count = 5  # type: ignore[attr-defined]
        record.elapsed_ms = 12.5  # type: ignore[attr-defined]

        output = formatter.format(record)
        assert "[INFO]" in output
        assert "[pydocgen.analysis]" in output
        assert "Functions extracted" in output
        assert "count=5" in output
        assert "elapsed_ms=12.5" in output


class TestTelemetryLogger:
    """Test TelemetryLogger context merging, binding, and sanitization."""

    def test_logger_merges_contextvars_and_extra(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="DEBUG", format_type="json", stream=stream)
        assert isinstance(logger, TelemetryLogger)

        with bind_context(run_id="run-42"):
            logger.info("Test message", extra={"item_id": 100})

        log_line = stream.getvalue().strip()
        data = json.loads(log_line)

        assert data["message"] == "Test message"
        assert data["context"]["run_id"] == "run-42"
        assert data["context"]["item_id"] == 100

    def test_logger_bind(self) -> None:
        stream = io.StringIO()
        base_logger = configure_telemetry(level="DEBUG", format_type="json", stream=stream)

        bound_logger = base_logger.bind(component="parser", version="1.0")
        bound_logger.info("Bound event")

        data = json.loads(stream.getvalue().strip())
        assert data["context"]["component"] == "parser"
        assert data["context"]["version"] == "1.0"

    def test_logger_sanitizes_reserved_log_record_keys(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="DEBUG", format_type="json", stream=stream)

        # 'filename', 'msg', 'lineno' are reserved LogRecord attributes
        logger.info(
            "Conflict test",
            extra={"filename": "test.py", "msg": "inner", "lineno": 99},
        )

        data = json.loads(stream.getvalue().strip())
        assert data["message"] == "Conflict test"
        assert data["context"]["extra_filename"] == "test.py"
        assert data["context"]["extra_msg"] == "inner"
        assert data["context"]["extra_lineno"] == 99


class TestTimer:
    """Test execution timing context manager."""

    def test_timer_successful_operation(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="DEBUG", format_type="json", stream=stream)

        with timer(logger, "test_job", job_id=1) as metrics:
            metrics["processed_items"] = 42

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n")]
        assert len(lines) == 2

        start_log, end_log = lines[0], lines[1]
        assert start_log["message"] == "Starting test_job"
        assert start_log["context"]["status"] == "started"

        assert "Completed test_job in" in end_log["message"]
        assert end_log["context"]["status"] == "completed"
        assert end_log["context"]["processed_items"] == 42
        assert "duration_ms" in end_log["context"]
        assert end_log["context"]["duration_ms"] >= 0

    def test_timer_failure_logs_error_and_reraises(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="DEBUG", format_type="json", stream=stream)

        with pytest.raises(ZeroDivisionError), timer(logger, "failing_job"):
            _ = 1 / 0

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n")]
        assert len(lines) == 2

        err_log = lines[1]
        assert err_log["level"] == "ERROR"
        assert "Failed failing_job after" in err_log["message"]
        assert err_log["context"]["status"] == "failed"
        assert "division by zero" in err_log["context"]["error"]
        assert "duration_ms" in err_log["context"]

    def test_timed_method_on_telemetry_logger(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="INFO", format_type="json", stream=stream)

        with logger.timed("logger_timed_job"):
            pass

        lines = [json.loads(line) for line in stream.getvalue().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["message"] == "Starting logger_timed_job"
        assert lines[1]["context"]["status"] == "completed"


class TestTelemetryConfiguration:
    """Test telemetry initialization, stream routing, and file handlers."""

    def test_configure_console_stream(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="WARNING", format_type="console", stream=stream)

        logger.debug("Should not appear")
        logger.info("Should not appear")
        logger.warning("Warning message")

        output = stream.getvalue()
        assert "Should not appear" not in output
        assert "Warning message" in output
        assert "[WARNING]" in output

    def test_configure_file_output(self, tmp_path: Path) -> None:
        log_file = tmp_path / "logs" / "test.log"
        stream = io.StringIO()
        logger = configure_telemetry(
            level="INFO",
            format_type="json",
            stream=stream,
            log_file=log_file,
        )

        logger.info("File log message", extra={"target": "file"})

        assert log_file.exists()
        file_content = log_file.read_text(encoding="utf-8").strip()
        data = json.loads(file_content)
        assert data["message"] == "File log message"
        assert data["context"]["target"] == "file"

    def test_get_logger_prefixing(self) -> None:
        l1 = get_logger("custom")
        assert l1.logger.name == "pydocgen.custom"

        l2 = get_logger("pydocgen.analysis")
        assert l2.logger.name == "pydocgen.analysis"

        l3 = get_logger(None)
        assert l3.logger.name == "pydocgen"

    def test_configure_rejects_invalid_level(self) -> None:
        with pytest.raises(ValueError, match="Invalid logging level"):
            configure_telemetry(level="SUPER_CRITICAL")

    def test_configure_rejects_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unsupported format_type"):
            configure_telemetry(format_type="xml")

    def test_reset_telemetry(self) -> None:
        stream = io.StringIO()
        logger = configure_telemetry(level="DEBUG", stream=stream)
        logger.info("Before reset")
        assert "Before reset" in stream.getvalue()

        reset_telemetry()
        # After reset, root logger handles are reset to NullHandler
        root = logging.getLogger("pydocgen")
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.NullHandler)


class TestPackageExports:
    """Test that pydocgen.telemetry exports all public components."""

    def test_telemetry_exports(self) -> None:
        import pydocgen.telemetry as telemetry

        expected = [
            "ConsoleFormatter",
            "JsonFormatter",
            "SensitiveDataFilter",
            "TelemetryLogger",
            "bind_context",
            "clear_context",
            "configure_telemetry",
            "get_context",
            "get_logger",
            "redact_sensitive_data",
            "redact_sensitive_string",
            "reset_telemetry",
            "set_context",
            "timer",
            "update_context",
        ]
        for name in expected:
            assert hasattr(telemetry, name)
            assert getattr(telemetry, name) is not None
