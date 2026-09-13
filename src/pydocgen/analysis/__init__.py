"""AST parsing, code analysis, and function metadata extraction."""

from pydocgen.analysis.extractor import (
    FunctionExtractor,
    extract_functions,
    extract_undocumented_functions,
)
from pydocgen.analysis.models import (
    FunctionInfo,
    ParameterInfo,
    ParameterKind,
)
from pydocgen.analysis.parser import (
    load_source,
    parse_file,
    parse_source,
    validate_file_path,
)

__all__ = [
    "FunctionExtractor",
    "FunctionInfo",
    "ParameterInfo",
    "ParameterKind",
    "extract_functions",
    "extract_undocumented_functions",
    "load_source",
    "parse_file",
    "parse_source",
    "validate_file_path",
]
