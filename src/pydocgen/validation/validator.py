"""Semantic validator cross-referencing LLM docstrings with AST source truth."""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydocgen.analysis.models import FunctionInfo
from pydocgen.config.settings import DEFAULT_STRICT_RAISES
from pydocgen.generation.models import GeneratedDocumentation
from pydocgen.telemetry import get_logger

logger = get_logger(__name__)


class ValidationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: ValidationSeverity


@dataclass
class ValidatedFunction:
    function: FunctionInfo
    documentation: GeneratedDocumentation | None
    is_accepted: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    rejection_reason: str | None = None


@dataclass
class ValidationReport:
    accepted: list[ValidatedFunction] = field(default_factory=list)
    skipped: list[ValidatedFunction] = field(default_factory=list)
    rejected: list[ValidatedFunction] = field(default_factory=list)

    @property
    def total_processed(self) -> int:
        return len(self.accepted) + len(self.skipped) + len(self.rejected)


class _ReturnYieldVisitor(ast.NodeVisitor):
    """AST visitor detecting return and yield statements without recursing into nested scopes."""

    def __init__(self) -> None:
        self.has_valuable_return = False
        self.is_generator = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            if isinstance(node.value, ast.Constant) and node.value.value is None:
                return
            self.has_valuable_return = True

    def visit_Yield(self, node: ast.Yield) -> None:
        self.is_generator = True
        self.has_valuable_return = True

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        self.is_generator = True
        self.has_valuable_return = True


class _ExceptionVisitor(ast.NodeVisitor):
    """AST visitor extracting explicitly raised exception names within the function scope."""

    def __init__(self) -> None:
        self.raised: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        pass

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            if isinstance(node.exc, ast.Call):
                raw = ast.unparse(node.exc.func)
                self.raised.add(raw)
                self.raised.add(raw.split(".")[-1])
            elif isinstance(node.exc, ast.Name):
                self.raised.add(node.exc.id)
            elif isinstance(node.exc, ast.Attribute):
                raw = ast.unparse(node.exc)
                self.raised.add(raw)
                self.raised.add(node.exc.attr)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            has_bare_raise = any(
                isinstance(n, ast.Raise) and n.exc is None for n in ast.walk(handler)
            )
            if has_bare_raise and handler.type is not None:
                exc_text = ast.unparse(handler.type)
                self.raised.add(exc_text)
                self.raised.add(exc_text.split(".")[-1])
        self.generic_visit(node)


class SemanticValidator:
    """Cross-validates LLM-generated documentation against AST source truth."""

    def __init__(self, strict_raises: bool = DEFAULT_STRICT_RAISES) -> None:
        self.strict_raises = strict_raises

    def validate_all(
        self,
        functions: Sequence[FunctionInfo],
        documentations: Sequence[GeneratedDocumentation],
    ) -> ValidationReport:
        doc_map = {doc.function_id: doc for doc in documentations}
        report = ValidationReport()

        with logger.timed("validate_all", function_count=len(functions)):
            for fn in functions:
                doc = doc_map.get(fn.id)
                if doc is None:
                    item = ValidatedFunction(
                        function=fn,
                        documentation=None,
                        is_accepted=False,
                        rejection_reason="No generation output provided for function.",
                    )
                    report.rejected.append(item)
                    continue

                if doc.status == "uncertain":
                    item = ValidatedFunction(
                        function=fn,
                        documentation=doc,
                        is_accepted=False,
                        rejection_reason=doc.reason or "Model flagged function as uncertain.",
                    )
                    report.skipped.append(item)
                    continue

                validated_item = self.validate_function(fn, doc)
                if validated_item.is_accepted:
                    report.accepted.append(validated_item)
                else:
                    report.rejected.append(validated_item)

            logger.info(
                "Completed semantic validation",
                extra={
                    "total": len(functions),
                    "accepted": len(report.accepted),
                    "rejected": len(report.rejected),
                    "skipped": len(report.skipped),
                },
            )

        return report

    def validate_function(
        self,
        fn: FunctionInfo,
        doc: GeneratedDocumentation,
    ) -> ValidatedFunction:
        issues: list[ValidationIssue] = []

        # 1. Clean arguments: Strip self / cls only if method receiver
        expected_params = self._get_expected_parameter_names(fn)
        cleaned_args = self._sanitize_documented_args(fn, doc.args)

        doc_param_names = set(cleaned_args.keys())
        missing_params = expected_params - doc_param_names
        extra_params = doc_param_names - expected_params

        if extra_params:
            issues.append(
                ValidationIssue(
                    field="args",
                    message=f"Documented nonexistent parameter(s): {', '.join(sorted(extra_params))}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        if missing_params:
            issues.append(
                ValidationIssue(
                    field="args",
                    message=f"Missing documentation for parameter(s): {', '.join(sorted(missing_params))}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # 2. Return semantics check
        has_valuable_return = self._inspect_return_behavior(fn)
        if not has_valuable_return and doc.returns:
            issues.append(
                ValidationIssue(
                    field="returns",
                    message="Documented return value, but function returns None or lacks return statements.",
                    severity=ValidationSeverity.ERROR,
                )
            )
        elif has_valuable_return and not doc.returns:
            issues.append(
                ValidationIssue(
                    field="returns",
                    message="Function returns a value, but no Returns section is documented.",
                    severity=ValidationSeverity.WARNING,
                )
            )

        # 3. Exception checking
        if self.strict_raises and doc.raises:
            actual_raises, parsed_ok = self._extract_raised_exceptions(fn.source)
            if parsed_ok:
                for raised_entry in doc.raises:
                    exc_name = raised_entry.split(":")[0].strip()
                    base_exc_name = exc_name.split(".")[-1]
                    if exc_name not in actual_raises and base_exc_name not in actual_raises:
                        issues.append(
                            ValidationIssue(
                                field="raises",
                                message=f"Documented exception '{exc_name}' not raised in function body.",
                                severity=ValidationSeverity.ERROR,
                            )
                        )

        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        if has_errors:
            reasons = "; ".join(i.message for i in issues if i.severity == ValidationSeverity.ERROR)
            logger.warning(
                "Semantic validation failed for function",
                extra={
                    "function_id": fn.id,
                    "qualified_name": fn.qualified_name,
                    "reason": reasons,
                },
            )
            return ValidatedFunction(
                function=fn,
                documentation=doc,
                is_accepted=False,
                issues=issues,
                rejection_reason=reasons,
            )

        doc.args = cleaned_args
        logger.debug(
            "Function documentation passed semantic validation",
            extra={"function_id": fn.id, "qualified_name": fn.qualified_name},
        )
        return ValidatedFunction(
            function=fn,
            documentation=doc,
            is_accepted=True,
            issues=issues,
        )

    def _get_expected_parameter_names(self, fn: FunctionInfo) -> set[str]:
        names: set[str] = set()
        is_regular_method = fn.is_method and "staticmethod" not in fn.decorators
        for idx, param in enumerate(fn.parameters):
            if is_regular_method and idx == 0 and param.name in ("self", "cls"):
                continue
            names.add(param.name)
        return names

    def _sanitize_documented_args(self, fn: FunctionInfo, args: dict[str, str]) -> dict[str, str]:
        cleaned: dict[str, str] = {}
        is_regular_method = fn.is_method and "staticmethod" not in fn.decorators
        expected_params = {p.name for p in fn.parameters}

        for k, v in args.items():
            k_clean = k.strip()
            if (
                is_regular_method
                and k_clean in ("self", "cls")
                and fn.parameters
                and fn.parameters[0].name == k_clean
            ):
                continue
            if k_clean.startswith("**") and k_clean[2:] in expected_params:
                cleaned[k_clean[2:]] = v.strip()
            elif k_clean.startswith("*") and k_clean[1:] in expected_params:
                cleaned[k_clean[1:]] = v.strip()
            else:
                cleaned[k_clean] = v.strip()

        return cleaned

    def _inspect_return_behavior(self, fn: FunctionInfo) -> bool:
        try:
            tree = ast.parse(textwrap.dedent(fn.source))
        except SyntaxError:
            return fn.return_annotation is not None and fn.return_annotation not in (
                "None",
                "typing.NoReturn",
                "NoReturn",
            )

        fn_body = (
            tree.body[0].body
            if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            else tree.body
        )

        visitor = _ReturnYieldVisitor()
        for stmt in fn_body:
            visitor.visit(stmt)

        if visitor.has_valuable_return:
            return True

        if fn.return_annotation in ("None", "typing.NoReturn", "NoReturn"):
            return False

        return fn.return_annotation is not None

    def _extract_raised_exceptions(self, source: str) -> tuple[set[str], bool]:
        try:
            tree = ast.parse(textwrap.dedent(source))
        except SyntaxError:
            return set(), False

        fn_body = (
            tree.body[0].body
            if tree.body and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef))
            else tree.body
        )

        visitor = _ExceptionVisitor()
        for stmt in fn_body:
            visitor.visit(stmt)

        return visitor.raised, True
