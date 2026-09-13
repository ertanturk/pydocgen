"""Unit tests for source code file validation, loading, and AST parsing."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from pydocgen.analysis.parser import (
    load_source,
    parse_file,
    parse_source,
    validate_file_path,
)
from pydocgen.errors import FileValidationError, ParseError


class TestValidateFilePath:
    """Test target path validation rules and error conditions."""

    def test_valid_relative_path_string(self) -> None:
        path_str = "src/pydocgen/analysis/parser.py"
        result = validate_file_path(path_str, enforce_relative=True)
        assert isinstance(result, Path)
        assert result == Path(path_str)

    def test_valid_relative_path_pathlib(self) -> None:
        path_obj = Path("src/pydocgen/analysis/parser.py")
        result = validate_file_path(path_obj, enforce_relative=True)
        assert isinstance(result, Path)
        assert result == path_obj

    def test_valid_absolute_path_when_enforce_relative_false(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "valid_script.py"
        temp_file.write_text("x = 1\n", encoding="utf-8")

        result = validate_file_path(temp_file, enforce_relative=False)
        assert result == temp_file

    def test_rejects_non_string_and_non_path(self) -> None:
        for invalid in [None, 12345, ["path.py"], {"path": "file.py"}]:
            with pytest.raises(FileValidationError, match="must be a string or Path object"):
                validate_file_path(invalid)  # ty: ignore[invalid-argument-type]

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(FileValidationError, match="cannot be empty or contain only whitespace"):
            validate_file_path("")

    def test_rejects_whitespace_only_string(self) -> None:
        with pytest.raises(FileValidationError, match="cannot be empty or contain only whitespace"):
            validate_file_path("   \t  \n  ")

    def test_rejects_absolute_path_when_enforce_relative_true(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "module.py"
        temp_file.write_text("a = 1", encoding="utf-8")

        with pytest.raises(
            FileValidationError, match="Path must be relative to the current working directory"
        ):
            validate_file_path(temp_file, enforce_relative=True)

    def test_rejects_directory_traversal_outside_cwd(self) -> None:
        traversal_path = Path("../../outside.py")
        with pytest.raises(
            FileValidationError, match="Path must be relative to the current working directory"
        ):
            validate_file_path(traversal_path, enforce_relative=True)

    def test_rejects_nonexistent_file(self) -> None:
        with pytest.raises(FileValidationError, match="File does not exist"):
            validate_file_path("nonexistent_pydocgen_file.py")

    def test_rejects_directory_instead_of_file(self) -> None:
        with pytest.raises(FileValidationError, match="Target path is not a regular file"):
            validate_file_path("src")

    def test_rejects_directory_ending_with_py(self, tmp_path: Path) -> None:
        py_dir = tmp_path / "fake_module.py"
        py_dir.mkdir()

        with pytest.raises(FileValidationError, match="Target path is not a regular file"):
            validate_file_path(py_dir, enforce_relative=False)

    def test_rejects_non_python_extension(self, tmp_path: Path) -> None:
        text_file = tmp_path / "notes.txt"
        text_file.write_text("hello", encoding="utf-8")

        with pytest.raises(
            FileValidationError, match=r"Expected a Python source file \(\.py\), got: '\.txt'"
        ):
            validate_file_path(text_file, enforce_relative=False)

    def test_rejects_no_extension_file(self, tmp_path: Path) -> None:
        no_ext_file = tmp_path / "Makefile"
        no_ext_file.write_text("all: build", encoding="utf-8")

        with pytest.raises(
            FileValidationError,
            match=r"Expected a Python source file \(\.py\), got: 'no extension'",
        ):
            validate_file_path(no_ext_file, enforce_relative=False)

    def test_accepts_uppercase_py_extension(self, tmp_path: Path) -> None:
        upper_file = tmp_path / "script.PY"
        upper_file.write_text("y = 2\n", encoding="utf-8")

        result = validate_file_path(upper_file, enforce_relative=False)
        assert result == upper_file

    def test_rejects_hidden_dot_py_file(self, tmp_path: Path) -> None:
        hidden_file = tmp_path / ".py"
        hidden_file.write_text("# dot py", encoding="utf-8")

        with pytest.raises(
            FileValidationError,
            match=r"Expected a Python source file \(\.py\), got: 'no extension'",
        ):
            validate_file_path(hidden_file, enforce_relative=False)

    def test_handles_os_error_on_exists(self) -> None:
        with (
            patch.object(Path, "exists", side_effect=OSError("Disk failure")),
            pytest.raises(FileValidationError, match="Cannot access path"),
        ):
            validate_file_path("src/pydocgen/analysis/parser.py")

    def test_handles_os_error_on_is_file(self) -> None:
        with (
            patch.object(Path, "is_file", side_effect=PermissionError("Permission denied")),
            pytest.raises(FileValidationError, match="Cannot access path"),
        ):
            validate_file_path("src/pydocgen/analysis/parser.py")


class TestLoadSource:
    """Test source file reading, encoding resilience, and error handling."""

    def test_load_valid_file_pathlib(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "code.py"
        content = "def hello():\n    return 'world'\n"
        temp_file.write_text(content, encoding="utf-8")

        result = load_source(temp_file)
        assert result == content

    def test_load_valid_file_string(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "code.py"
        content = "VALUE = 42\n"
        temp_file.write_text(content, encoding="utf-8")

        result = load_source(str(temp_file))
        assert result == content

    def test_load_file_with_utf8_bom(self, tmp_path: Path) -> None:
        """Verify that UTF-8 BOM is stripped cleanly without AST syntax error."""
        temp_file = tmp_path / "bom_code.py"
        content_with_bom = b"\xef\xbb\xbfdef foo():\n    return True\n"
        temp_file.write_bytes(content_with_bom)

        result = load_source(temp_file)
        assert result == "def foo():\n    return True\n"
        assert not result.startswith("\ufeff")

    def test_load_empty_file(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "empty.py"
        temp_file.write_text("", encoding="utf-8")

        result = load_source(temp_file)
        assert result == ""

    def test_load_file_unicode_decode_error(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "invalid_encoding.py"
        temp_file.write_bytes(b"\x80\x81\x82\xff")

        with pytest.raises(FileValidationError, match="Failed to decode .* as UTF-8"):
            load_source(temp_file)

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        missing_file = tmp_path / "does_not_exist.py"
        with pytest.raises(FileValidationError, match="Cannot read file"):
            load_source(missing_file)

    def test_load_file_permission_denied(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "protected.py"
        temp_file.write_text("x = 1\n", encoding="utf-8")

        with (
            patch.object(Path, "read_text", side_effect=PermissionError("Permission denied")),
            pytest.raises(FileValidationError, match="Cannot read file"),
        ):
            load_source(temp_file)

    def test_load_file_os_error_without_strerror(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "error.py"
        temp_file.write_text("x = 1\n", encoding="utf-8")

        os_err = OSError("Custom I/O failure")
        os_err.strerror = None
        with (
            patch.object(Path, "read_text", side_effect=os_err),
            pytest.raises(FileValidationError, match="Cannot read file"),
        ):
            load_source(temp_file)


class TestParseSource:
    """Test AST source parsing, syntax validation, and ParseError details."""

    def test_parse_valid_code(self) -> None:
        source = "def greet(name: str) -> str:\n    return f'Hello, {name}'\n"
        tree = parse_source(source, filename="greet.py")

        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == "greet"

    def test_parse_empty_source(self) -> None:
        tree = parse_source("")
        assert isinstance(tree, ast.Module)
        assert tree.body == []

    def test_parse_comments_and_whitespace_only(self) -> None:
        tree = parse_source("# Only a comment\n   \n# Another comment\n")
        assert isinstance(tree, ast.Module)
        assert tree.body == []

    def test_parse_syntax_error_details(self) -> None:
        invalid_source = "def broken(\n"
        with pytest.raises(ParseError) as exc_info:
            parse_source(invalid_source, filename="broken.py")

        err = exc_info.value
        assert "Syntax error in 'broken.py'" in str(err)
        assert err.line is not None
        assert err.column is not None

    def test_parse_source_with_null_bytes(self) -> None:
        source_with_null = "x = 1\0"
        with pytest.raises(ParseError, match="Syntax error"):
            parse_source(source_with_null, filename="null.py")

    def test_rejects_non_string_source(self) -> None:
        for invalid in [None, 123, b"x = 1", ["code"]]:
            with pytest.raises(TypeError, match="Expected source code as string"):
                parse_source(invalid)  # ty: ignore[invalid-argument-type]


class TestParseFile:
    """Test the end-to-end parse_file pipeline."""

    def test_parse_file_success(self) -> None:
        path = "src/pydocgen/analysis/parser.py"
        valid_path, source, tree = parse_file(path, enforce_relative=True)

        assert isinstance(valid_path, Path)
        assert valid_path == Path(path)
        assert isinstance(source, str)
        assert "def parse_file" in source
        assert isinstance(tree, ast.Module)

    def test_parse_file_with_bom(self, tmp_path: Path) -> None:
        temp_file = tmp_path / "module_bom.py"
        temp_file.write_bytes(b"\xef\xbb\xbfdef add(a: int, b: int) -> int:\n    return a + b\n")

        valid_path, source, tree = parse_file(temp_file, enforce_relative=False)
        assert valid_path == temp_file
        assert source.startswith("def add")
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)

    def test_parse_file_missing_file_raises_validation_error(self) -> None:
        with pytest.raises(FileValidationError, match="File does not exist"):
            parse_file("missing_module.py")

    def test_parse_file_syntax_error_raises_parse_error(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "syntax_err.py"
        bad_file.write_text("class Unfinished:\n", encoding="utf-8")

        with pytest.raises(ParseError) as exc_info:
            parse_file(bad_file, enforce_relative=False)

        err = exc_info.value
        assert f"Syntax error in '{bad_file}'" in str(err)
        assert err.line is not None

    def test_parse_file_with_string_path(self, tmp_path: Path) -> None:
        py_file = tmp_path / "str_path.py"
        py_file.write_text("PI = 3.14159\n", encoding="utf-8")

        valid_path, source, tree = parse_file(str(py_file), enforce_relative=False)
        assert valid_path == py_file
        assert "PI = 3.14159" in source
        assert isinstance(tree, ast.Module)


class TestExceptionsExport:
    """Test exception imports and attributes."""

    def test_parse_error_exported_from_errors_package(self) -> None:
        import pydocgen.errors as errors

        assert hasattr(errors, "ParseError")
        assert errors.ParseError is ParseError
        assert hasattr(errors, "FileValidationError")
        assert errors.FileValidationError is FileValidationError

    def test_parse_error_attributes(self) -> None:
        err = ParseError(message="bad syntax", line=10, column=5)
        assert str(err) == "bad syntax"
        assert err.line == 10
        assert err.column == 5
