"""Unit tests for the pydocgen CLI, command routing, and rich output formatting."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from pydocgen import __version__
from pydocgen.application.service import DocumentationReport
from pydocgen.cli import output
from pydocgen.cli.app import app, main
from pydocgen.errors.exceptions import (
    AuthenticationError,
    CredentialAlreadyExistsError,
    CredentialNotFoundError,
    ExitCode,
    InvalidCredentialError,
)

runner = CliRunner()


@pytest.fixture
def empty_python_file(tmp_path: Path) -> Path:
    target = tmp_path / "empty.py"
    target.write_text("# Just comments and variables\nx = 42\n")
    return target


@pytest.fixture
def documented_python_file(tmp_path: Path) -> Path:
    target = tmp_path / "documented.py"
    target.write_text(
        'def add(a: int, b: int) -> int:\n    """Add two integers."""\n    return a + b\n'
    )
    return target


@pytest.fixture
def undocumented_python_file(tmp_path: Path) -> Path:
    target = tmp_path / "undocumented.py"
    target.write_text("def compute(data: list[int]) -> int:\n    return sum(data)\n")
    return target


# =========================================================================
# 1. Version & Root Help Tests
# =========================================================================


class TestRootCLI:
    def test_version_flag_long(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == ExitCode.OK
        assert __version__ in result.output

    def test_version_flag_short(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == ExitCode.OK
        assert __version__ in result.output

    def test_help_menu(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == ExitCode.OK
        assert "document" in result.output
        assert "auth" in result.output


# =========================================================================
# 2. Smart Routing Tests in main()
# =========================================================================


class TestSmartRouting:
    def test_main_with_explicit_version(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == ExitCode.OK

    def test_main_routes_bare_file_to_document(self, documented_python_file: Path) -> None:
        with patch("pydocgen.cli.app._handle_check_mode") as mock_check:
            mock_check.side_effect = SystemExit(0)
            with pytest.raises(SystemExit) as exc_info:
                main(["--check", str(documented_python_file)])
            assert exc_info.value.code == 0
            assert mock_check.called

    def test_main_routes_file_first_with_flags(self, documented_python_file: Path) -> None:
        with patch("pydocgen.cli.app._handle_check_mode") as mock_check:
            mock_check.side_effect = SystemExit(0)
            with pytest.raises(SystemExit) as exc_info:
                main([str(documented_python_file), "--check"])
            assert exc_info.value.code == 0
            assert mock_check.called

    def test_main_preserves_auth_command(self) -> None:
        with (
            patch("pydocgen.security.credentials.Credentials.get_api_key", return_value=None),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["auth", "status"])
        assert exc_info.value.code == 0

    def test_main_empty_args_shows_help(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        # Typer exits with 2 on no_args_is_help or 0 depending on click version
        assert exc_info.value.code in (0, 2)


# =========================================================================
# 3. Check Mode Tests (--check)
# =========================================================================


class TestCheckMode:
    def test_check_mode_non_python_file(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello world")
        result = runner.invoke(app, ["document", str(text_file)])
        assert result.exit_code == ExitCode.NOINPUT
        assert "not a Python (.py) source file" in result.output

    def test_check_mode_empty_file_no_functions(self, empty_python_file: Path) -> None:
        result = runner.invoke(app, ["document", str(empty_python_file), "--check"])
        assert result.exit_code == ExitCode.OK
        assert "No functions found" in result.output

    def test_check_mode_fully_documented_file(self, documented_python_file: Path) -> None:
        result = runner.invoke(app, ["document", str(documented_python_file), "--check"])
        assert result.exit_code == ExitCode.OK
        assert "All 1 function(s)" in result.output

    def test_check_mode_undocumented_functions(self, undocumented_python_file: Path) -> None:
        result = runner.invoke(app, ["document", str(undocumented_python_file), "--check"])
        assert result.exit_code == 1
        assert "lack docstrings" in result.output

    def test_check_mode_quiet_fully_documented(self, documented_python_file: Path) -> None:
        result = runner.invoke(app, ["document", str(documented_python_file), "--check", "--quiet"])
        assert result.exit_code == ExitCode.OK
        assert result.output.strip() == ""


# =========================================================================
# 4. Document Command Pipeline Execution
# =========================================================================


class TestDocumentPipeline:
    def test_document_dry_run_success(self, documented_python_file: Path) -> None:
        mock_report = DocumentationReport(
            path=documented_python_file,
            total_functions=1,
            already_documented=1,
            to_document=0,
            batches_processed=0,
            accepted_count=0,
            skipped_count=0,
            rejected_count=0,
            failed_batches=0,
            applied=False,
        )

        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.return_value = mock_report
            result = runner.invoke(app, ["document", str(documented_python_file), "--dry-run"])
            assert result.exit_code == ExitCode.OK
            assert "Dry run complete" in result.output
            assert mock_doc.call_args.kwargs["dry_run"] is True

    def test_document_overwrite_flag_forwarded(self, documented_python_file: Path) -> None:
        mock_report = DocumentationReport(
            path=documented_python_file,
            total_functions=1,
            already_documented=1,
            to_document=1,
            batches_processed=1,
            accepted_count=1,
            skipped_count=0,
            rejected_count=0,
            failed_batches=0,
            applied=True,
        )

        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.return_value = mock_report
            result = runner.invoke(
                app, ["document", str(documented_python_file), "--overwrite", "--apply"]
            )
            assert result.exit_code == ExitCode.OK
            assert "Successfully updated" in result.output
            assert mock_doc.call_args.kwargs["overwrite_existing"] is True

    def test_document_pipeline_failure_batch_errors(self, undocumented_python_file: Path) -> None:
        mock_report = DocumentationReport(
            path=undocumented_python_file,
            total_functions=1,
            already_documented=0,
            to_document=1,
            batches_processed=1,
            accepted_count=0,
            skipped_count=0,
            rejected_count=0,
            failed_batches=1,
            applied=False,
            batch_errors=["Rate limit exceeded"],
        )

        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.return_value = mock_report
            result = runner.invoke(app, ["document", str(undocumented_python_file), "--apply"])
            assert result.exit_code == ExitCode.TEMPFAIL
            assert "Rate limit exceeded" in result.output

    def test_document_pipeline_rejected_validation(self, undocumented_python_file: Path) -> None:
        mock_report = DocumentationReport(
            path=undocumented_python_file,
            total_functions=1,
            already_documented=0,
            to_document=1,
            batches_processed=1,
            accepted_count=0,
            skipped_count=0,
            rejected_count=1,
            failed_batches=0,
            applied=False,
        )

        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.return_value = mock_report
            result = runner.invoke(app, ["document", str(undocumented_python_file), "--apply"])
            assert result.exit_code == ExitCode.OK
            assert "did not pass" in result.output and "semantic validation" in result.output

    def test_document_pipeline_pydocgen_error_handling(
        self, undocumented_python_file: Path
    ) -> None:
        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.side_effect = AuthenticationError(
                "Invalid API key provided", exit_code=ExitCode.CONFIG
            )
            result = runner.invoke(app, ["document", str(undocumented_python_file), "--apply"])
            assert result.exit_code == ExitCode.CONFIG
            assert "Invalid API key provided" in result.output

    def test_document_pipeline_unexpected_exception(self, undocumented_python_file: Path) -> None:
        with patch(
            "pydocgen.application.service.DocumentationService.document_file",
            new_callable=AsyncMock,
        ) as mock_doc:
            mock_doc.side_effect = RuntimeError("Fatal crash")
            result = runner.invoke(app, ["document", str(undocumented_python_file), "--apply"])
            assert result.exit_code == ExitCode.SOFTWARE
            assert "Fatal crash" in result.output


# =========================================================================
# 5. Auth Command Tests
# =========================================================================


class TestAuthCommands:
    def test_auth_status_no_key(self) -> None:
        with (
            patch("pydocgen.security.credentials.Credentials.get_api_key", return_value=None),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = runner.invoke(app, ["auth", "status"])
            assert result.exit_code == ExitCode.OK
            assert "No API key found" in result.output

    def test_auth_status_keyring_source(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.get_api_key") as mock_get:
            mock_get.return_value = "AIzaSyD-1234567890abcdefghijklmnop"
            result = runner.invoke(app, ["auth", "status"])
            assert result.exit_code == ExitCode.OK
            assert "OS Keyring" in result.output
            assert "AIza...mnop" in result.output

    def test_auth_status_env_var_source(self) -> None:
        with (
            patch("pydocgen.security.credentials.Credentials.get_api_key", return_value=None),
            patch.dict(
                "os.environ", {"GEMINI_API_KEY": "AIzaSyD-environmentkey123456789"}, clear=True
            ),
        ):
            result = runner.invoke(app, ["auth", "status"])
            assert result.exit_code == ExitCode.OK
            assert "Environment Variable" in result.output
            assert "GEMINI_API_KEY" in result.output

    def test_auth_set_explicit_key_success(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.save_api_key") as mock_save:
            result = runner.invoke(
                app, ["auth", "set", "--key", "AIzaSyD-1234567890abcdefghijklmnop"]
            )
            assert result.exit_code == ExitCode.OK
            assert "Successfully stored" in result.output
            mock_save.assert_called_once_with("AIzaSyD-1234567890abcdefghijklmnop", overwrite=True)

    def test_auth_set_invalid_key_error(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.save_api_key") as mock_save:
            mock_save.side_effect = InvalidCredentialError("Key too short")
            result = runner.invoke(app, ["auth", "set", "--key", "short"])
            assert result.exit_code == ExitCode.USAGE
            assert "Key too short" in result.output

    def test_auth_set_conflict_error_no_force(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.save_api_key") as mock_save:
            mock_save.side_effect = CredentialAlreadyExistsError("Key exists")
            result = runner.invoke(
                app, ["auth", "set", "--key", "AIzaSyD-1234567890abcdefghijklmnop", "--no-force"]
            )
            assert result.exit_code == ExitCode.CONFIG
            assert "Pass --force to overwrite" in result.output

    def test_auth_delete_existing_key(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.delete_api_key") as mock_del:
            result = runner.invoke(app, ["auth", "delete"])
            assert result.exit_code == ExitCode.OK
            assert "removed from OS keyring" in result.output
            mock_del.assert_called_once()

    def test_auth_delete_non_existing_key(self) -> None:
        with patch("pydocgen.security.credentials.Credentials.delete_api_key") as mock_del:
            mock_del.side_effect = CredentialNotFoundError("Not found")
            result = runner.invoke(app, ["auth", "delete"])
            assert result.exit_code == ExitCode.OK
            assert "No stored Gemini API key found" in result.output


# =========================================================================
# 6. Output Component Unit Tests
# =========================================================================


class TestOutputComponents:
    def test_render_banner(self, tmp_path: Path) -> None:
        p = tmp_path / "module.py"
        with patch.object(output.err_console, "print") as mock_print:
            output.render_banner(p, "gemini-3.8-flash")
            assert mock_print.called

    def test_render_analysis_stats(self) -> None:
        with patch.object(output.err_console, "print") as mock_print:
            output.render_analysis_stats(10, 4, 6)
            assert mock_print.called

    def test_render_batch_step(self) -> None:
        with patch.object(output.err_console, "print") as mock_print:
            output.render_batch_step("batch_1", 3, 1, 1, "success")
            output.render_batch_step("batch_2", 3, 1, 1, "failed")
            output.render_batch_step("batch_3", 3, 1, 1, "generating")
            assert mock_print.call_count >= 3

    def test_render_results_summary(self, tmp_path: Path) -> None:
        report = DocumentationReport(
            path=tmp_path / "test.py",
            total_functions=5,
            already_documented=1,
            to_document=4,
            batches_processed=2,
            accepted_count=3,
            skipped_count=1,
            rejected_count=0,
            failed_batches=0,
            applied=True,
        )
        with patch.object(output.err_console, "print") as mock_print:
            output.render_results_summary(report)
            assert mock_print.called

    def test_render_unified_diff(self) -> None:
        diff = "--- a/test.py\n+++ b/test.py\n@@ -1,1 +1,2 @@\n+docstring\n-old\n context"
        with patch.object(output.err_console, "print") as mock_print:
            output.render_unified_diff(diff)
            assert mock_print.called

    def test_render_unified_diff_empty(self) -> None:
        with patch.object(output.err_console, "print") as mock_print:
            output.render_unified_diff("")
            assert not mock_print.called

    def test_prompt_confirmation_affirmative(self) -> None:
        with patch.object(output.err_console, "input", return_value="y"):
            assert output.prompt_confirmation("sample.py", 2) is True

        with patch.object(output.err_console, "input", return_value="yes"):
            assert output.prompt_confirmation("sample.py", 2) is True

    def test_prompt_confirmation_negative(self) -> None:
        with patch.object(output.err_console, "input", return_value="n"):
            assert output.prompt_confirmation("sample.py", 2) is False

    def test_prompt_confirmation_eof_error(self) -> None:
        with patch.object(output.err_console, "input", side_effect=EOFError):
            assert output.prompt_confirmation("sample.py", 2) is False

    def test_prompt_confirmation_keyboard_interrupt(self) -> None:
        with patch.object(output.err_console, "input", side_effect=KeyboardInterrupt):
            assert output.prompt_confirmation("sample.py", 2) is False

    def test_render_error_string(self) -> None:
        with patch.object(output.err_console, "print") as mock_print:
            output.render_error("Test Title", "Single error message")
            assert mock_print.called

    def test_render_error_list(self) -> None:
        with patch.object(output.err_console, "print") as mock_print:
            output.render_error("Test Title", ["Error 1", "Error 2"])
            assert mock_print.called


class TestCheckModeExceptions:
    def test_check_mode_syntax_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def invalid syntax (:", encoding="utf-8")
        result = runner.invoke(app, ["document", str(bad_file), "--check"])
        assert result.exit_code == ExitCode.DATAERR

    def test_check_mode_file_not_found(self, tmp_path: Path) -> None:
        import typer

        from pydocgen.cli.app import _handle_check_mode

        missing = tmp_path / "missing.py"
        with pytest.raises(typer.Exit) as exc_info:
            _handle_check_mode(missing, quiet=True)
        assert exc_info.value.exit_code == ExitCode.NOINPUT
