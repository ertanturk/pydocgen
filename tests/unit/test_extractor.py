"""Unit tests for AST function extraction, parameter extraction, and analysis data models."""

import ast
from dataclasses import FrozenInstanceError

import pytest

from pydocgen.analysis import (
    FunctionExtractor,
    FunctionInfo,
    ParameterInfo,
    ParameterKind,
    extract_functions,
    extract_undocumented_functions,
)


class TestParameterInfo:
    """Test ParameterInfo model behavior, validation, and immutability."""

    def test_create_parameter_with_defaults(self) -> None:
        param = ParameterInfo(name="arg")
        assert param.name == "arg"
        assert param.annotation is None
        assert param.default is None
        assert param.kind == ParameterKind.POSITIONAL_OR_KEYWORD

    def test_create_parameter_with_custom_attributes(self) -> None:
        param = ParameterInfo(
            name="total",
            annotation="int",
            default="0",
            kind=ParameterKind.KEYWORD_ONLY,
        )
        assert param.name == "total"
        assert param.annotation == "int"
        assert param.default == "0"
        assert param.kind == ParameterKind.KEYWORD_ONLY

    def test_create_parameter_with_str_kind_coerced_to_enum(self) -> None:
        param = ParameterInfo(name="arg", kind="var_positional")  # ty: ignore[invalid-argument-type]
        assert param.kind == ParameterKind.VAR_POSITIONAL

    def test_parameter_immutability(self) -> None:
        param = ParameterInfo(name="x")
        with pytest.raises(FrozenInstanceError):
            param.name = "y"  # ty: ignore[invalid-assignment]

    def test_parameter_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ParameterInfo(name="")

    def test_parameter_rejects_non_string_name(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            ParameterInfo(name=123)  # ty: ignore[invalid-argument-type]

    def test_parameter_equality(self) -> None:
        p1 = ParameterInfo(name="x", annotation="int", default="1")
        p2 = ParameterInfo(name="x", annotation="int", default="1")
        assert p1 == p2


class TestFunctionInfo:
    """Test FunctionInfo model immutability and container conversions."""

    def test_create_function_info(self) -> None:
        param = ParameterInfo(name="x")
        fn = FunctionInfo(
            id="abc123456789",
            name="compute",
            qualified_name="compute",
            source="def compute(x): return x",
            parameters=(param,),
            return_annotation="int",
            decorators=("staticmethod",),
            line_start=1,
            line_end=1,
            has_docstring=False,
            is_async=False,
            is_method=False,
            docstring=None,
        )
        assert fn.name == "compute"
        assert fn.parameters == (param,)
        assert fn.decorators == ("staticmethod",)
        assert fn.docstring is None

    def test_function_info_immutability(self) -> None:
        fn = FunctionInfo(
            id="1",
            name="f",
            qualified_name="f",
            source="def f(): pass",
            parameters=(),
            return_annotation=None,
            decorators=(),
            line_start=1,
            line_end=1,
            has_docstring=False,
            is_async=False,
            is_method=False,
        )
        with pytest.raises(FrozenInstanceError):
            fn.name = "g"  # ty: ignore[invalid-assignment]

    def test_list_conversion_in_post_init(self) -> None:
        param = ParameterInfo(name="x")
        fn = FunctionInfo(
            id="1",
            name="f",
            qualified_name="f",
            source="def f(x): pass",
            parameters=[param],  # ty: ignore[invalid-argument-type]
            return_annotation=None,
            decorators=["memoize"],  # ty: ignore[invalid-argument-type]
            line_start=1,
            line_end=1,
            has_docstring=False,
            is_async=False,
            is_method=False,
        )
        assert isinstance(fn.parameters, tuple)
        assert fn.parameters == (param,)
        assert isinstance(fn.decorators, tuple)
        assert fn.decorators == ("memoize",)


class TestParameterExtraction:
    """Test parameter extraction across all Python argument variations."""

    def test_no_parameters(self) -> None:
        code = "def no_args(): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        assert functions[0].parameters == ()

    def test_regular_positional_or_keyword_args(self) -> None:
        code = "def regular(a, b, c): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 3
        for p, name in zip(params, ["a", "b", "c"], strict=True):
            assert p.name == name
            assert p.default is None
            assert p.kind == ParameterKind.POSITIONAL_OR_KEYWORD

    def test_positional_with_defaults(self) -> None:
        code = "def with_defaults(a, b=1, c='test'): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 3
        assert params[0].name == "a" and params[0].default is None
        assert params[1].name == "b" and params[1].default == "1"
        assert params[2].name == "c" and params[2].default == "'test'"

    def test_positional_only_and_regular_without_defaults(self) -> None:
        code = "def pos_only(a, b, /, c, d): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 4
        assert params[0].name == "a" and params[0].kind == ParameterKind.POSITIONAL_ONLY
        assert params[1].name == "b" and params[1].kind == ParameterKind.POSITIONAL_ONLY
        assert params[2].name == "c" and params[2].kind == ParameterKind.POSITIONAL_OR_KEYWORD
        assert params[3].name == "d" and params[3].kind == ParameterKind.POSITIONAL_OR_KEYWORD
        for p in params:
            assert p.default is None

    def test_positional_only_and_regular_mixed_defaults(self) -> None:
        """Verify the bug where positional-only defaults shifted regular arg defaults."""
        code = "def mixed(a, b=1, /, c=2, d=3): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 4
        assert params[0].name == "a" and params[0].default is None
        assert params[0].kind == ParameterKind.POSITIONAL_ONLY

        assert params[1].name == "b" and params[1].default == "1"
        assert params[1].kind == ParameterKind.POSITIONAL_ONLY

        assert params[2].name == "c" and params[2].default == "2"
        assert params[2].kind == ParameterKind.POSITIONAL_OR_KEYWORD

        assert params[3].name == "d" and params[3].default == "3"
        assert params[3].kind == ParameterKind.POSITIONAL_OR_KEYWORD

    def test_positional_only_all_default(self) -> None:
        code = "def all_default(a=1, b=2, /, c=3, d=4): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert params[0].name == "a" and params[0].default == "1"
        assert params[1].name == "b" and params[1].default == "2"
        assert params[2].name == "c" and params[2].default == "3"
        assert params[3].name == "d" and params[3].default == "4"

    def test_var_positional(self) -> None:
        code = "def var_pos(*args): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "args"
        assert params[0].kind == ParameterKind.VAR_POSITIONAL
        assert params[0].default is None

    def test_keyword_only_args(self) -> None:
        code = "def kw_only(*, a, b=10, c=None): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 3
        assert params[0].name == "a"
        assert params[0].default is None
        assert params[0].kind == ParameterKind.KEYWORD_ONLY

        assert params[1].name == "b"
        assert params[1].default == "10"
        assert params[1].kind == ParameterKind.KEYWORD_ONLY

        assert params[2].name == "c"
        assert params[2].default == "None"
        assert params[2].kind == ParameterKind.KEYWORD_ONLY

    def test_var_keyword(self) -> None:
        code = "def var_kw(**kwargs): pass"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        params = functions[0].parameters
        assert len(params) == 1
        assert params[0].name == "kwargs"
        assert params[0].kind == ParameterKind.VAR_KEYWORD
        assert params[0].default is None

    def test_complex_full_signature(self) -> None:
        code = (
            "def full_sig(a: int, b: str = 'x', /, c: float = 1.0, *args: int, "
            "d: bool = True, **kwargs: object) -> None: pass"
        )
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        fn = functions[0]
        assert fn.return_annotation == "None"
        params = fn.parameters
        assert len(params) == 6

        assert params[0].name == "a" and params[0].annotation == "int"
        assert params[0].default is None and params[0].kind == ParameterKind.POSITIONAL_ONLY

        assert params[1].name == "b" and params[1].annotation == "str"
        assert params[1].default == "'x'" and params[1].kind == ParameterKind.POSITIONAL_ONLY

        assert params[2].name == "c" and params[2].annotation == "float"
        assert params[2].default == "1.0" and params[2].kind == ParameterKind.POSITIONAL_OR_KEYWORD

        assert params[3].name == "args" and params[3].annotation == "int"
        assert params[3].default is None and params[3].kind == ParameterKind.VAR_POSITIONAL

        assert params[4].name == "d" and params[4].annotation == "bool"
        assert params[4].default == "True" and params[4].kind == ParameterKind.KEYWORD_ONLY

        assert params[5].name == "kwargs" and params[5].annotation == "object"
        assert params[5].default is None and params[5].kind == ParameterKind.VAR_KEYWORD


class TestFunctionExtraction:
    """Test function attributes, decorators, and determinism."""

    def test_direct_extractor_instantiation(self) -> None:
        code = "def sample(): pass\n"
        extractor = FunctionExtractor(code)
        extractor.visit(ast.parse(code))
        assert len(extractor.functions) == 1
        assert extractor.functions[0].name == "sample"

    def test_extract_simple_function(self) -> None:
        code = "def add(x: int, y: int) -> int:\n    return x + y\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        fn = functions[0]
        assert fn.name == "add"
        assert fn.qualified_name == "add"
        assert fn.line_start == 1
        assert fn.line_end == 2
        assert not fn.is_async
        assert not fn.is_method
        assert fn.return_annotation == "int"
        assert fn.source == "def add(x: int, y: int) -> int:\n    return x + y"

    def test_extract_async_function(self) -> None:
        code = "async def fetch_data():\n    return []\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        assert functions[0].is_async is True
        assert functions[0].name == "fetch_data"

    def test_extract_function_with_decorators(self) -> None:
        code = "@property\n@route('/home', methods=['GET'])\ndef home(): pass\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        fn = functions[0]
        assert len(fn.decorators) == 2
        assert fn.decorators[0] == "property"
        assert fn.decorators[1] == "route('/home', methods=['GET'])"

    def test_deterministic_function_id(self) -> None:
        code = "def f(): pass\n"
        tree = ast.parse(code)
        fn1 = extract_functions(code, tree)[0]
        fn2 = extract_functions(code, tree)[0]
        assert fn1.id == fn2.id
        assert len(fn1.id) == 12

    def test_preserves_document_order(self) -> None:
        code = "def first(): pass\ndef second(): pass\ndef third(): pass\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        names = [f.name for f in functions]
        assert names == ["first", "second", "third"]


class TestScopeAndIsMethod:
    """Test is_method accuracy and qualified_name across nested scopes."""

    def test_top_level_function_is_not_method(self) -> None:
        code = "def top(): pass\n"
        tree = ast.parse(code)
        fn = extract_functions(code, tree)[0]

        assert fn.is_method is False
        assert fn.qualified_name == "top"

    def test_nested_function_in_top_level_is_not_method(self) -> None:
        code = "def outer():\n    def inner(): pass\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 2
        assert functions[0].name == "outer"
        assert functions[0].is_method is False
        assert functions[0].qualified_name == "outer"

        assert functions[1].name == "inner"
        assert functions[1].is_method is False
        assert functions[1].qualified_name == "outer.<locals>.inner"

    def test_class_method_is_method(self) -> None:
        code = "class Calculator:\n    def add(self, x, y):\n        return x + y\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        fn = functions[0]
        assert fn.name == "add"
        assert fn.is_method is True
        assert fn.qualified_name == "Calculator.add"

    def test_nested_function_in_method_is_not_method(self) -> None:
        """Addresses the bug where helper functions inside methods were marked as is_method=True."""
        code = (
            "class Service:\n"
            "    def process(self):\n"
            "        def helper(x):\n"
            "            return x + 1\n"
            "        return helper(1)\n"
        )
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 2
        method, helper = functions[0], functions[1]

        assert method.name == "process"
        assert method.is_method is True
        assert method.qualified_name == "Service.process"

        assert helper.name == "helper"
        assert helper.is_method is False
        assert helper.qualified_name == "Service.process.<locals>.helper"

    def test_nested_class_inside_method_has_methods(self) -> None:
        code = (
            "class Outer:\n"
            "    def method(self):\n"
            "        class Inner:\n"
            "            def inner_method(self):\n"
            "                pass\n"
        )
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 2
        outer_m, inner_m = functions[0], functions[1]

        assert outer_m.name == "method"
        assert outer_m.is_method is True
        assert outer_m.qualified_name == "Outer.method"

        assert inner_m.name == "inner_method"
        assert inner_m.is_method is True
        assert inner_m.qualified_name == "Outer.method.<locals>.Inner.inner_method"

    def test_class_inside_top_level_function_has_methods(self) -> None:
        code = "def factory():\n    class Product:\n        def use(self):\n            pass\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 2
        factory_fn, use_method = functions[0], functions[1]

        assert factory_fn.name == "factory"
        assert factory_fn.is_method is False
        assert factory_fn.qualified_name == "factory"

        assert use_method.name == "use"
        assert use_method.is_method is True
        assert use_method.qualified_name == "factory.<locals>.Product.use"

    def test_class_inside_class_method(self) -> None:
        code = "class A:\n    class B:\n        def b_method(self): pass\n"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)

        assert len(functions) == 1
        fn = functions[0]
        assert fn.name == "b_method"
        assert fn.is_method is True
        assert fn.qualified_name == "A.B.b_method"


class TestDocstringExtraction:
    """Test docstring presence detection and undocumented function filtering."""

    def test_function_with_clean_docstring(self) -> None:
        code = (
            "def documented():\n"
            '    """This is a summary.\n'
            "\n"
            "    Returns:\n"
            "        None.\n"
            '    """\n'
            "    pass\n"
        )
        tree = ast.parse(code)
        fn = extract_functions(code, tree)[0]

        assert fn.has_docstring is True
        assert fn.docstring is not None
        assert fn.docstring.startswith("This is a summary.")

    def test_function_without_docstring(self) -> None:
        code = "def undocumented():\n    pass\n"
        tree = ast.parse(code)
        fn = extract_functions(code, tree)[0]

        assert fn.has_docstring is False
        assert fn.docstring is None

    def test_function_with_empty_docstring_treated_as_undocumented(self) -> None:
        code = 'def empty_doc():\n    """"""\n    pass\n'
        tree = ast.parse(code)
        fn = extract_functions(code, tree)[0]

        assert fn.has_docstring is False
        assert fn.docstring is None

    def test_function_with_whitespace_docstring_treated_as_undocumented(self) -> None:
        code = 'def ws_doc():\n    """   \\n  \\t  """\n    pass\n'
        tree = ast.parse(code)
        fn = extract_functions(code, tree)[0]

        assert fn.has_docstring is False
        assert fn.docstring is None

    def test_extract_undocumented_functions(self) -> None:
        code = (
            'def f1():\n    """Doc."""\n    pass\n'
            "def f2():\n    pass\n"
            'def f3():\n    """"""\n    pass\n'
            'def f4():\n    """Valid doc."""\n    pass\n'
        )
        tree = ast.parse(code)
        undocumented = extract_undocumented_functions(code, tree)

        names = [f.name for f in undocumented]
        assert names == ["f2", "f3"]


class TestInputValidation:
    """Test error handling on invalid function extractor inputs."""

    def test_rejects_non_string_source(self) -> None:
        tree = ast.parse("pass")
        for invalid in [None, 123, ["code"], b"def f(): pass"]:
            with pytest.raises(TypeError, match="Expected source code as string"):
                extract_functions(invalid, tree)  # ty: ignore[invalid-argument-type]

    def test_rejects_non_ast_tree(self) -> None:
        for invalid in [None, "tree", 123, []]:
            with pytest.raises(TypeError, match="Expected tree as ast.AST"):
                extract_functions("pass", invalid)  # ty: ignore[invalid-argument-type]

    def test_empty_ast_returns_empty_list(self) -> None:
        code = "# just a comment"
        tree = ast.parse(code)
        functions = extract_functions(code, tree)
        assert functions == []


class TestAnalysisPackageExports:
    """Test that pydocgen.analysis exports all necessary public symbols."""

    def test_analysis_exports(self) -> None:
        import pydocgen.analysis as analysis

        expected_exports = [
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
        for name in expected_exports:
            assert hasattr(analysis, name)
            assert getattr(analysis, name) is not None
