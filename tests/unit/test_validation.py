"""Unit tests for the semantic validation package."""

from __future__ import annotations

import textwrap

from pydocgen.analysis.models import FunctionInfo, ParameterInfo, ParameterKind
from pydocgen.generation.models import GeneratedDocumentation
from pydocgen.validation.validator import (
    SemanticValidator,
    ValidatedFunction,
    ValidationReport,
    ValidationSeverity,
)


def _make_fn(
    fn_id: str = "fn_1",
    name: str = "test_func",
    source: str = "def test_func(): pass",
    parameters: tuple[ParameterInfo, ...] = (),
    return_annotation: str | None = None,
    decorators: tuple[str, ...] = (),
    is_method: bool = False,
    is_async: bool = False,
) -> FunctionInfo:
    return FunctionInfo(
        id=fn_id,
        name=name,
        qualified_name=f"pkg.{name}",
        source=source,
        parameters=parameters,
        return_annotation=return_annotation,
        decorators=decorators,
        line_start=1,
        line_end=5,
        has_docstring=False,
        is_async=is_async,
        is_method=is_method,
    )


class TestValidationDataModels:
    """Tests for report containers and data classes."""

    def test_severity_enum(self) -> None:
        assert ValidationSeverity.INFO == "info"
        assert ValidationSeverity.WARNING == "warning"
        assert ValidationSeverity.ERROR == "error"

    def test_validation_report_totals(self) -> None:
        report = ValidationReport()
        fn = _make_fn()
        doc = GeneratedDocumentation(function_id="fn_1", status="documented", summary="Test")

        report.accepted.append(ValidatedFunction(function=fn, documentation=doc, is_accepted=True))
        report.skipped.append(
            ValidatedFunction(
                function=fn,
                documentation=None,
                is_accepted=False,
                rejection_reason="Skipped",
            )
        )
        report.rejected.append(
            ValidatedFunction(
                function=fn,
                documentation=doc,
                is_accepted=False,
                rejection_reason="Failed",
            )
        )

        assert report.total_processed == 3
        assert len(report.accepted) == 1
        assert len(report.skipped) == 1
        assert len(report.rejected) == 1


class TestSemanticValidator:
    """Tests for SemanticValidator AST cross-referencing."""

    def test_validate_all_partitions_correctly(self) -> None:
        fn1 = _make_fn("fn_1", source="def fn_1(): pass")
        fn2 = _make_fn("fn_2", source="def fn_2(): pass")
        fn3 = _make_fn("fn_3", source="def fn_3(): pass")

        doc1 = GeneratedDocumentation(function_id="fn_1", status="documented", summary="Summary 1.")
        doc2 = GeneratedDocumentation(
            function_id="fn_2", status="uncertain", reason="Ambiguous logic."
        )
        # fn3 is omitted from documentations

        validator = SemanticValidator()
        report = validator.validate_all([fn1, fn2, fn3], [doc1, doc2])

        assert len(report.accepted) == 1
        assert report.accepted[0].function.id == "fn_1"

        assert len(report.skipped) == 1
        assert report.skipped[0].function.id == "fn_2"
        assert "Ambiguous logic." in (report.skipped[0].rejection_reason or "")

        assert len(report.rejected) == 1
        assert report.rejected[0].function.id == "fn_3"
        assert "No generation output provided" in (report.rejected[0].rejection_reason or "")

    def test_parameter_parity_happy_path(self) -> None:
        fn = _make_fn(
            parameters=(
                ParameterInfo(name="x"),
                ParameterInfo(name="y"),
            ),
            source="def test_func(x, y): pass",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Valid function.",
            args={"x": "X coordinate.", "y": "Y coordinate."},
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True
        assert len(res.issues) == 0

    def test_parameter_parity_missing_and_extra_params(self) -> None:
        fn = _make_fn(
            parameters=(ParameterInfo(name="a"), ParameterInfo(name="b")),
            source="def test_func(a, b): pass",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Valid function.",
            args={"a": "Param a.", "ghost": "Hallucinated."},
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is False
        assert any(
            i.field == "args" and "Missing documentation for parameter(s): b" in i.message
            for i in res.issues
        )
        assert any(
            i.field == "args" and "Documented nonexistent parameter(s): ghost" in i.message
            for i in res.issues
        )

    def test_regular_method_strips_self_and_cls(self) -> None:
        fn = _make_fn(
            name="method",
            is_method=True,
            parameters=(
                ParameterInfo(name="self"),
                ParameterInfo(name="val"),
            ),
            source="def method(self, val): pass",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Method summary.",
            args={"self": "The instance.", "val": "Value."},
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True
        assert "self" not in res.documentation.args  # type: ignore[union-attr]
        assert "val" in res.documentation.args  # type: ignore[union-attr]

    def test_staticmethod_does_not_strip_self_parameter(self) -> None:
        fn = _make_fn(
            name="static_method",
            is_method=True,
            decorators=("staticmethod",),
            parameters=(ParameterInfo(name="self"),),
            source="@staticmethod\ndef static_method(self): pass",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Static method.",
            args={"self": "Custom self param."},
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True
        assert "self" in res.documentation.args  # type: ignore[union-attr]

    def test_varargs_and_kwargs_matching(self) -> None:
        fn = _make_fn(
            parameters=(
                ParameterInfo(name="args", kind=ParameterKind.VAR_POSITIONAL),
                ParameterInfo(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
            ),
            source="def func(*args, **kwargs): pass",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Varargs func.",
            args={"*args": "Positional args.", "**kwargs": "Keyword args."},
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True
        assert "args" in res.documentation.args  # type: ignore[union-attr]
        assert "kwargs" in res.documentation.args  # type: ignore[union-attr]

    def test_return_semantics_indented_method_with_return(self) -> None:
        """Addresses the critical bug where indented method sources raised IndentationError in ast.parse."""
        indented_code = textwrap.indent(
            "def calculate(self, x):\n    return x * 2\n",
            "    ",
        )
        fn = _make_fn(
            is_method=True,
            source=indented_code,
            parameters=(ParameterInfo(name="self"), ParameterInfo(name="x")),
            return_annotation="int",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Calculates double.",
            args={"x": "Input number."},
            returns="Doubled value.",
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True

    def test_return_semantics_nested_function_isolation(self) -> None:
        """Inner function return must not trick outer void function into claiming a return value."""
        source = textwrap.dedent("""
            def outer_func():
                def inner_helper():
                    return 42
                inner_helper()
        """)
        fn = _make_fn(source=source, return_annotation="None")
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Outer function.",
            returns="Returns 42.",
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is False
        assert any(
            i.field == "returns"
            and "Documented return value, but function returns None" in i.message
            for i in res.issues
        )

    def test_return_semantics_generator_yield(self) -> None:
        source = textwrap.dedent("""
            def gen():
                yield 1
        """)
        fn = _make_fn(source=source)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Generator function.",
            returns="Yields integer sequence.",
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True

    def test_return_semantics_missing_returns_emits_warning_not_error(self) -> None:
        source = "def add(x, y):\n    return x + y\n"
        fn = _make_fn(
            source=source,
            parameters=(ParameterInfo(name="x"), ParameterInfo(name="y")),
            return_annotation="int",
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Sum function.",
            args={"x": "Param x.", "y": "Param y."},
            returns=None,  # Omitted returns
        )

        validator = SemanticValidator()
        res = validator.validate_function(fn, doc)
        # Warning should not reject function
        assert res.is_accepted is True
        assert any(
            i.field == "returns"
            and i.severity == ValidationSeverity.WARNING
            and "no Returns section is documented" in i.message
            for i in res.issues
        )

    def test_strict_raises_happy_path(self) -> None:
        source = textwrap.dedent("""
            def parse_data(raw):
                if not raw:
                    raise ValueError("Empty data.")
                if len(raw) > 100:
                    raise errors.DataOverflowError("Too large.")
        """)
        fn = _make_fn(source=source, parameters=(ParameterInfo(name="raw"),))
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Parse data.",
            args={"raw": "Raw string."},
            raises=[
                "ValueError: If raw is empty.",
                "DataOverflowError: If raw exceeds limit.",
            ],
        )

        validator = SemanticValidator(strict_raises=True)
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True

    def test_strict_raises_flags_unraised_exception(self) -> None:
        source = textwrap.dedent("""
            def parse_data(raw):
                if not raw:
                    raise ValueError("Empty data.")
        """)
        fn = _make_fn(source=source, parameters=(ParameterInfo(name="raw"),))
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Parse data.",
            args={"raw": "Raw string."},
            raises=[
                "ValueError: If raw is empty.",
                "KeyError: If key not found.",  # Unraised exception
            ],
        )

        validator = SemanticValidator(strict_raises=True)
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is False
        assert any(
            i.field == "raises"
            and "Documented exception 'KeyError' not raised in function body." in i.message
            for i in res.issues
        )

    def test_strict_raises_function_with_no_raises_rejects_hallucinated_raises(self) -> None:
        """Addresses the bug where if actual_raises was empty, exceptions were never verified."""
        source = "def safe_func():\n    return 42\n"
        fn = _make_fn(source=source)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Safe func.",
            returns="42.",
            raises=["RuntimeError: Unexpected issue."],
        )

        validator = SemanticValidator(strict_raises=True)
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is False
        assert any(
            "Documented exception 'RuntimeError' not raised" in i.message for i in res.issues
        )

    def test_strict_raises_disabled_allows_unraised_exceptions(self) -> None:
        source = "def func():\n    pass\n"
        fn = _make_fn(source=source)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Func summary.",
            raises=["Exception: Possible failure."],
        )

        validator = SemanticValidator(strict_raises=False)
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True

    def test_re_raised_exception_in_try_block_detected(self) -> None:
        source = textwrap.dedent("""
            def connect():
                try:
                    open_socket()
                except ConnectionError:
                    log_error()
                    raise
        """)
        fn = _make_fn(source=source)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Connect to socket.",
            raises=["ConnectionError: Socket failed."],
        )

        validator = SemanticValidator(strict_raises=True)
        res = validator.validate_function(fn, doc)
        assert res.is_accepted is True
