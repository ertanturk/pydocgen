"""AST visitor and extraction logic for Python functions and methods."""

import ast
import hashlib

from pydocgen.analysis.models import FunctionInfo, ParameterInfo, ParameterKind
from pydocgen.config.settings import FUNCTION_ID_HASH_LENGTH
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


class FunctionExtractor(ast.NodeVisitor):
    """Traverses Python AST to collect functions with their metadata and source code."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str):
            raise TypeError(f"Expected source code as string, got: {type(source).__name__}")
        self._source = source
        self._scope_stack: list[str] = []
        self._scope_type_stack: list[str] = []
        self.functions: list[FunctionInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_stack.append(node.name)
        self._scope_type_stack.append("class")
        self.generic_visit(node)
        self._scope_type_stack.pop()
        self._scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._process_function(node, is_async=True)

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, is_async: bool
    ) -> None:
        # Build qualified name (e.g. MyClass.process_item or outer.<locals>.inner)
        if self._scope_stack:
            qualified_name = f"{'.'.join(self._scope_stack)}.{node.name}"
        else:
            qualified_name = node.name

        # A function is a method if and only if its immediate enclosing scope is a class
        is_method = bool(self._scope_type_stack and self._scope_type_stack[-1] == "class")

        # Extract source slice cleanly
        source_segment = ast.get_source_segment(self._source, node) or ""

        # Docstring presence check (ignoring whitespace-only docstrings)
        docstring = ast.get_docstring(node, clean=True)
        has_doc = bool(docstring and docstring.strip())

        # Extract parameters and return annotation
        parameters = self._extract_parameters(node.args)
        return_annot = ast.unparse(node.returns) if node.returns else None
        decorators = tuple(ast.unparse(dec) for dec in node.decorator_list)

        # Generate deterministic function ID based on qualified name and line number
        id_seed = f"{qualified_name}:{node.lineno}:{node.col_offset}"
        fn_id = hashlib.sha256(id_seed.encode()).hexdigest()[:FUNCTION_ID_HASH_LENGTH]

        function_info = FunctionInfo(
            id=fn_id,
            name=node.name,
            qualified_name=qualified_name,
            source=source_segment,
            parameters=tuple(parameters),
            return_annotation=return_annot,
            decorators=decorators,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            has_docstring=has_doc,
            is_async=is_async,
            is_method=is_method,
            docstring=docstring if has_doc else None,
        )
        self.functions.append(function_info)

        # Push scope to handle nested functions cleanly
        self._scope_stack.append(f"{node.name}.<locals>")
        self._scope_type_stack.append("function")
        self.generic_visit(node)
        self._scope_type_stack.pop()
        self._scope_stack.pop()

    def _extract_parameters(self, args: ast.arguments) -> list[ParameterInfo]:
        params: list[ParameterInfo] = []

        def annot_str(arg: ast.arg) -> str | None:
            return ast.unparse(arg.annotation) if arg.annotation else None

        # Positional arguments (posonlyargs + args) share args.defaults aligned to the right
        all_pos = args.posonlyargs + args.args
        num_defaults = len(args.defaults)
        non_default_count = len(all_pos) - num_defaults
        padded_defaults: list[ast.expr | None] = [None] * non_default_count + list(args.defaults)

        posonly_defaults = padded_defaults[: len(args.posonlyargs)]
        regular_defaults = padded_defaults[len(args.posonlyargs) :]

        # Positional-only args
        for arg, default in zip(args.posonlyargs, posonly_defaults, strict=True):
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=annot_str(arg),
                    default=ast.unparse(default) if default is not None else None,
                    kind=ParameterKind.POSITIONAL_ONLY,
                )
            )

        # Regular args (positional or keyword)
        for arg, default in zip(args.args, regular_defaults, strict=True):
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=annot_str(arg),
                    default=ast.unparse(default) if default is not None else None,
                    kind=ParameterKind.POSITIONAL_OR_KEYWORD,
                )
            )

        # *args (var_positional)
        if args.vararg:
            params.append(
                ParameterInfo(
                    name=args.vararg.arg,
                    annotation=annot_str(args.vararg),
                    kind=ParameterKind.VAR_POSITIONAL,
                )
            )

        # Keyword-only args
        for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
            params.append(
                ParameterInfo(
                    name=arg.arg,
                    annotation=annot_str(arg),
                    default=ast.unparse(default) if default is not None else None,
                    kind=ParameterKind.KEYWORD_ONLY,
                )
            )

        # **kwargs (var_keyword)
        if args.kwarg:
            params.append(
                ParameterInfo(
                    name=args.kwarg.arg,
                    annotation=annot_str(args.kwarg),
                    kind=ParameterKind.VAR_KEYWORD,
                )
            )

        return params


def extract_functions(source: str, tree: ast.AST) -> list[FunctionInfo]:
    """Extract all functions from an AST module preserving document order."""
    if not isinstance(source, str):
        raise TypeError(f"Expected source code as string, got: {type(source).__name__}")
    if not isinstance(tree, ast.AST):
        raise TypeError(f"Expected tree as ast.AST node, got: {type(tree).__name__}")
    with logger.timed("extractor.extract_functions"):
        logger.debug("Extracting functions from AST", extra={"source_len": len(source)})
        extractor = FunctionExtractor(source)
        extractor.visit(tree)
        logger.info(
            "Extracted functions from AST",
            extra={"function_count": len(extractor.functions)},
        )
        return extractor.functions


def extract_undocumented_functions(source: str, tree: ast.AST) -> list[FunctionInfo]:
    """Extract only functions that lack a docstring."""
    all_funcs = extract_functions(source, tree)
    undocumented = [fn for fn in all_funcs if not fn.has_docstring]
    logger.info(
        "Extracted undocumented functions",
        extra={"total": len(all_funcs), "undocumented": len(undocumented)},
    )
    return undocumented


__all__ = [
    "FunctionExtractor",
    "extract_functions",
    "extract_undocumented_functions",
]
