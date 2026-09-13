import ast
from pathlib import Path

from pydocgen.errors.exceptions import FileValidationError, ParseError


def validate_file_path(path: Path | str, enforce_relative: bool = True) -> Path:
    """Validate that the target path exists, is a file, and matches path rules.

    Args:
        path: Pathlib Path or string pointing to the target file.
        enforce_relative: If True, rejects absolute paths and paths outside the current working directory.

    Returns:
        Path: A validated Path object.

    Raises:
        FileValidationError: If the path fails any validation check.
    """
    if not isinstance(path, (Path, str)):
        raise FileValidationError(
            f"Path must be a string or Path object, got: {type(path).__name__}"
        )

    if isinstance(path, str) and not path.strip():
        raise FileValidationError("File path cannot be empty or contain only whitespace.")

    file_path = Path(path)

    if enforce_relative:
        if file_path.is_absolute():
            raise FileValidationError(
                f"Path must be relative to the current working directory: '{file_path}'"
            )
        try:
            file_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError as exc:
            raise FileValidationError(
                f"Path must be relative to the current working directory: '{file_path}'"
            ) from exc

    try:
        exists = file_path.exists()
    except OSError as exc:
        raise FileValidationError(
            f"Cannot access path '{file_path}': {exc.strerror or exc}"
        ) from exc

    if not exists:
        raise FileValidationError(f"File does not exist: '{file_path}'")

    try:
        is_file = file_path.is_file()
    except OSError as exc:
        raise FileValidationError(
            f"Cannot access path '{file_path}': {exc.strerror or exc}"
        ) from exc

    if not is_file:
        raise FileValidationError(f"Target path is not a regular file: '{file_path}'")

    if file_path.suffix.lower() != ".py":
        raise FileValidationError(
            f"Expected a Python source file (.py), got: '{file_path.suffix or 'no extension'}'"
        )

    return file_path


def load_source(path: Path | str) -> str:
    """Read file content safely handling permission and encoding failures.

    Args:
        path: Validated Path or string to read.

    Returns:
        The raw source code as a string.

    Raises:
        FileValidationError: If reading the file fails due to I/O or encoding.
    """
    file_path = Path(path)
    try:
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FileValidationError(
            f"Failed to decode '{file_path}' as UTF-8: {exc.reason} at byte {exc.start}"
        ) from exc
    except OSError as exc:
        raise FileValidationError(f"Cannot read file '{file_path}': {exc.strerror or exc}") from exc


def parse_source(source: str, filename: str = "<unknown>") -> ast.Module:
    """Parse raw Python source string into an AST Module node.

    Args:
        source: Python source code string.
        filename: Optional filename used in syntax error reporting.

    Returns:
        ast.Module root node.

    Raises:
        TypeError: If source is not a string.
        ParseError: If the source code contains syntax errors.
    """
    if not isinstance(source, str):
        raise TypeError(f"Expected source code as string, got: {type(source).__name__}")

    try:
        return ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise ParseError(
            message=f"Syntax error in '{filename}': {exc.msg or 'invalid syntax'}",
            line=exc.lineno,
            column=exc.offset,
        ) from exc


def parse_file(path: Path | str, enforce_relative: bool = True) -> tuple[Path, str, ast.Module]:
    """Validate, read, and parse a Python source file.

    Returns:
        Tuple of (validated_path, raw_source_code, ast_root).

    Raises:
        FileValidationError: On missing, invalid, or unreadable files.
        ParseError: On Python syntax errors.
    """
    valid_path = validate_file_path(path, enforce_relative=enforce_relative)
    source = load_source(valid_path)
    tree = parse_source(source, filename=str(valid_path))
    return valid_path, source, tree
