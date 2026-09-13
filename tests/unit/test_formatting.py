"""Unit tests for the GoogleFormatter docstring formatting package."""

from __future__ import annotations

import pytest

from pydocgen.analysis.models import FunctionInfo, ParameterInfo, ParameterKind
from pydocgen.formatting.google import GoogleFormatter
from pydocgen.generation.models import GeneratedDocumentation


def _make_fn(
    parameters: tuple[ParameterInfo, ...] = (),
    decorators: tuple[str, ...] = (),
    is_method: bool = False,
) -> FunctionInfo:
    return FunctionInfo(
        id="fn_1",
        name="target_func",
        qualified_name="pkg.target_func",
        source="def target_func(): pass",
        parameters=parameters,
        return_annotation=None,
        decorators=decorators,
        line_start=1,
        line_end=5,
        has_docstring=False,
        is_async=False,
        is_method=is_method,
    )


class TestGoogleFormatter:
    """Tests for Google Python Style Guide docstring generator."""

    def test_init_validation(self) -> None:
        formatter = GoogleFormatter(indent_width=2)
        assert formatter.indent_width == 2

        with pytest.raises(ValueError, match="indent_width must be a positive integer"):
            GoogleFormatter(indent_width=0)

        with pytest.raises(ValueError, match="indent_width must be a positive integer"):
            GoogleFormatter(indent_width=-2)

        with pytest.raises(ValueError, match="indent_width must be a positive integer"):
            GoogleFormatter(indent_width="4")  # type: ignore[arg-type]

    def test_format_invalid_inputs_raise(self) -> None:
        formatter = GoogleFormatter()

        with pytest.raises(ValueError, match="documentation cannot be None"):
            formatter.format(None)  # type: ignore[arg-type]

        uncertain_doc = GeneratedDocumentation(
            function_id="fn_1",
            status="uncertain",
            reason="Ambiguous logic.",
        )
        with pytest.raises(ValueError, match="Cannot format uncertain documentation"):
            formatter.format(uncertain_doc)

        empty_summary_doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Valid summary",
        )
        empty_summary_doc.summary = "   "
        with pytest.raises(ValueError, match="has no summary"):
            formatter.format(empty_summary_doc)

    def test_single_line_docstring(self) -> None:
        formatter = GoogleFormatter()
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Calculate the SHA-256 digest",
        )

        result = formatter.format(doc)
        # Adds trailing period for complete sentence
        assert result == '"""Calculate the SHA-256 digest."""'

        # Preserves base_indent
        indented_res = formatter.format(doc, base_indent="    ")
        assert indented_res == '    """Calculate the SHA-256 digest."""'

    def test_multiline_summary_indentation(self) -> None:
        """Addresses the bug where extended summary lines were not indented by base_indent."""
        formatter = GoogleFormatter()
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Primary summary line.\nExtended explanation line 1.\nExtended explanation line 2.",
            args={"x": "Input value."},
        )

        result = formatter.format(doc, base_indent="    ")
        lines = result.splitlines()

        assert lines[0] == '    """Primary summary line.'
        assert lines[1] == ""
        assert lines[2] == "    Extended explanation line 1."
        assert lines[3] == "    Extended explanation line 2."
        assert lines[4] == ""
        assert lines[5] == "    Args:"
        assert lines[6] == "        x: Input value."
        assert lines[7] == '    """'

    def test_multiline_docstring_with_args_returns_raises(self) -> None:
        formatter = GoogleFormatter(indent_width=4)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Perform secure data transfer.",
            args={
                "host": "Destination hostname or IP address.",
                "payload": "Binary payload to transmit.",
            },
            returns="Status code indicating success or failure.",
            raises=["ConnectionError: If network is unreachable."],
        )

        expected = (
            '    """Perform secure data transfer.\n'
            "\n"
            "    Args:\n"
            "        host: Destination hostname or IP address.\n"
            "        payload: Binary payload to transmit.\n"
            "\n"
            "    Returns:\n"
            "        Status code indicating success or failure.\n"
            "\n"
            "    Raises:\n"
            "        ConnectionError: If network is unreachable.\n"
            '    """'
        )

        result = formatter.format(doc, base_indent="    ")
        assert result == expected

    def test_multiline_arg_description_continuation_indent(self) -> None:
        formatter = GoogleFormatter(indent_width=4)
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Process batch.",
            args={
                "config": "Configuration dictionary.\nMust contain 'timeout' and 'retries' keys."
            },
        )

        result = formatter.format(doc, base_indent="    ")
        lines = result.splitlines()

        assert lines[2] == "    Args:"
        assert lines[3] == "        config: Configuration dictionary."
        assert lines[4] == "            Must contain 'timeout' and 'retries' keys."

    def test_argument_ordering_against_function_signature(self) -> None:
        fn = _make_fn(
            parameters=(
                ParameterInfo(name="first"),
                ParameterInfo(name="second"),
                ParameterInfo(name="third"),
            )
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Func with unordered docs.",
            args={
                "third": "3rd",
                "first": "1st",
                "second": "2nd",
                "extra": "Leftover arg",
            },
        )

        formatter = GoogleFormatter()
        result = formatter.format(doc, function=fn)

        lines = result.splitlines()
        assert lines[0] == '"""Func with unordered docs.'
        assert lines[1] == ""
        assert lines[2] == "Args:"
        assert lines[3] == "    first: 1st"
        assert lines[4] == "    second: 2nd"
        assert lines[5] == "    third: 3rd"
        assert lines[6] == "    extra: Leftover arg"
        assert lines[7] == '"""'

    def test_argument_ordering_regular_method_skips_self(self) -> None:
        fn = _make_fn(
            is_method=True,
            parameters=(
                ParameterInfo(name="self"),
                ParameterInfo(name="arg1"),
            ),
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Method doc.",
            args={"self": "Instance", "arg1": "Value 1"},
        )

        formatter = GoogleFormatter()
        result = formatter.format(doc, function=fn)

        assert "self:" not in result
        assert "arg1: Value 1" in result

    def test_argument_ordering_staticmethod_preserves_self(self) -> None:
        fn = _make_fn(
            is_method=True,
            decorators=("staticmethod",),
            parameters=(ParameterInfo(name="self"),),
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Static method.",
            args={"self": "Real self parameter."},
        )

        formatter = GoogleFormatter()
        result = formatter.format(doc, function=fn)

        assert "self: Real self parameter." in result

    def test_argument_ordering_varargs_and_kwargs(self) -> None:
        fn = _make_fn(
            parameters=(
                ParameterInfo(name="args", kind=ParameterKind.VAR_POSITIONAL),
                ParameterInfo(name="kwargs", kind=ParameterKind.VAR_KEYWORD),
            )
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Varargs func.",
            args={"args": "Positional.", "kwargs": "Keyword."},
        )

        formatter = GoogleFormatter()
        result = formatter.format(doc, function=fn)

        lines = result.splitlines()
        assert lines[0] == '"""Varargs func.'
        assert lines[1] == ""
        assert lines[2] == "Args:"
        # Should automatically format VAR_POSITIONAL as *args and VAR_KEYWORD as **kwargs
        assert lines[3] == "    *args: Positional."
        assert lines[4] == "    **kwargs: Keyword."
        assert lines[5] == '"""'

    def test_regular_method_with_only_self_becomes_single_line(self) -> None:
        fn = _make_fn(
            is_method=True,
            parameters=(ParameterInfo(name="self"),),
        )
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Method with only self.",
            args={"self": "The receiver instance."},
        )

        formatter = GoogleFormatter()
        result = formatter.format(doc, function=fn)
        # Since self is stripped and no other args exist, docstring collapses to single line
        assert result == '"""Method with only self."""'

    def test_empty_returns_and_empty_raises_omitted(self) -> None:
        formatter = GoogleFormatter()
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary="Check empty sections.",
            returns="   ",
            raises=["", "   "],
        )

        result = formatter.format(doc)
        assert result == '"""Check empty sections."""'
        assert "Returns:" not in result
        assert "Raises:" not in result

    def test_triple_quotes_escaped(self) -> None:
        formatter = GoogleFormatter()
        doc = GeneratedDocumentation(
            function_id="fn_1",
            status="documented",
            summary='Handles """ triple quotes safely',
        )

        result = formatter.format(doc)
        assert r"\"\"\"" in result
        assert not result.startswith('""""')
