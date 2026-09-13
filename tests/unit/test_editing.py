"""Unit tests for the source code editing and rewriting package."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from pydocgen.analysis.models import FunctionInfo, ParameterInfo
from pydocgen.editing.models import SourceEdit
from pydocgen.editing.rewriter import (
    apply_edits,
    create_source_edits,
    find_insertion_point,
    generate_diff,
    preview_and_apply,
    write_file_atomically,
)
from pydocgen.errors.exceptions import SourceEditError
from pydocgen.generation.models import GeneratedDocumentation


def _make_fn(
    fn_id: str = "fn_1",
    name: str = "sample_func",
    line_start: int = 1,
    line_end: int = 3,
    parameters: tuple[ParameterInfo, ...] = (),
    is_async: bool = False,
    is_method: bool = False,
    decorators: tuple[str, ...] = (),
) -> FunctionInfo:
    return FunctionInfo(
        id=fn_id,
        name=name,
        qualified_name=f"pkg.{name}",
        source="def sample_func(): pass",
        parameters=parameters,
        return_annotation=None,
        decorators=decorators,
        line_start=line_start,
        line_end=line_end,
        has_docstring=False,
        is_async=is_async,
        is_method=is_method,
    )


class TestFindInsertionPoint:
    """Tests for locating function closing colon and body indentation."""

    def test_simple_function(self) -> None:
        source = "def foo():\n    pass\n"
        fn = _make_fn(line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 1
        assert indent == "    "

    def test_multiline_signature_with_annotations(self) -> None:
        source = (
            "def complex_func(\n"
            "    a: int,\n"
            '    b: dict[str, int] = {"key:colon": 1},\n'
            ") -> bool:\n"
            "    result = True\n"
            "    return result\n"
        )
        fn = _make_fn(line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 4
        assert indent == "    "

    def test_multiline_decorator_with_colons(self) -> None:
        source = '@route(\n    path="http://api:8080/action",\n)\ndef decorated():\n    pass\n'
        fn = _make_fn(line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 4
        assert indent == "    "

    def test_async_function(self) -> None:
        source = "async def fetch_data(url: str) -> dict:\n    return {}\n"
        fn = _make_fn(name="fetch_data", is_async=True, line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 1
        assert indent == "    "

    def test_comment_with_colon(self) -> None:
        source = "def target(x: int):  # note: required\n    return x\n"
        fn = _make_fn(line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 1
        assert indent == "    "

    def test_custom_two_space_indentation(self) -> None:
        source = "def short_indent():\n  x = 1\n  return x\n"
        fn = _make_fn(line_start=1)
        insert_after, indent = find_insertion_point(source.splitlines(keepends=True), fn)
        assert insert_after == 1
        assert indent == "  "

    def test_missing_colon_raises_source_edit_error(self) -> None:
        source = "def malformed(\n"
        fn = _make_fn(line_start=1)
        with pytest.raises(SourceEditError, match="Could not find closing ':'"):
            find_insertion_point(source.splitlines(keepends=True), fn)


class TestCreateSourceEdits:
    """Tests for generating SourceEdit objects from documentation models."""

    def test_create_edits_multiple_functions(self) -> None:
        source = "def fn_a():\n    return 1\n\ndef fn_b():\n    return 2\n"
        fns = [
            _make_fn(fn_id="id_a", name="fn_a", line_start=1),
            _make_fn(fn_id="id_b", name="fn_b", line_start=4),
        ]
        docs = [
            GeneratedDocumentation(function_id="id_a", status="documented", summary="Doc A."),
            GeneratedDocumentation(function_id="id_b", status="documented", summary="Doc B."),
        ]

        edits = create_source_edits(source, fns, docs)
        assert len(edits) == 2
        assert edits[0].function_id == "id_a"
        assert edits[0].insert_after_line == 1
        assert edits[1].function_id == "id_b"
        assert edits[1].insert_after_line == 4

    def test_skips_uncertain_documentation(self) -> None:
        source = "def fn_a():\n    pass\n"
        fns = [_make_fn(fn_id="id_a", name="fn_a", line_start=1)]
        docs = [GeneratedDocumentation(function_id="id_a", status="uncertain", reason="Ambiguous")]

        edits = create_source_edits(source, fns, docs)
        assert len(edits) == 0

    def test_single_line_function_split(self) -> None:
        source = "def one_liner(): pass\n"
        fns = [_make_fn(fn_id="id_1", name="one_liner", line_start=1)]
        docs = [
            GeneratedDocumentation(function_id="id_1", status="documented", summary="One liner.")
        ]

        edits = create_source_edits(source, fns, docs)
        assert len(edits) == 1
        edit = edits[0]
        assert edit.replace_line_content == "def one_liner():\n"
        assert '"""One liner."""' in edit.docstring
        assert "pass" in edit.docstring


class TestApplyEdits:
    """Tests for applying SourceEdit objects into source code strings."""

    def test_apply_edits_descending_order(self) -> None:
        source = "def first():\n    return 1\n\ndef second():\n    return 2\n"
        edits = [
            SourceEdit(
                function_id="id_1",
                qualified_name="pkg.first",
                insert_after_line=1,
                indentation="    ",
                docstring='    """First doc."""',
            ),
            SourceEdit(
                function_id="id_2",
                qualified_name="pkg.second",
                insert_after_line=4,
                indentation="    ",
                docstring='    """Second doc."""',
            ),
        ]

        result = apply_edits(source, edits)
        lines = result.splitlines()

        assert lines[0] == "def first():"
        assert lines[1] == '    """First doc."""'
        assert lines[2] == "    return 1"
        assert lines[4] == "def second():"
        assert lines[5] == '    """Second doc."""'
        assert lines[6] == "    return 2"

    def test_apply_single_line_split_edit(self) -> None:
        source = "def quick(): return 42\n"
        edit = SourceEdit(
            function_id="id_q",
            qualified_name="pkg.quick",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Quick func."""\n    return 42',
            replace_line_content="def quick():\n",
        )

        result = apply_edits(source, [edit])
        assert result == ('def quick():\n    """Quick func."""\n    return 42\n')

    def test_apply_edits_empty_list(self) -> None:
        source = "def unaltered(): pass\n"
        assert apply_edits(source, []) == source


class TestGenerateDiff:
    """Tests for generating diff text."""

    def test_generate_diff_plain(self) -> None:
        original = "def foo():\n    pass\n"
        modified = 'def foo():\n    """Doc."""\n    pass\n'
        diff = generate_diff(original, modified, filename="test.py", colorize=False)

        assert "--- a/test.py" in diff
        assert "+++ b/test.py" in diff
        assert '+    """Doc."""' in diff
        assert "\033[" not in diff

    def test_generate_diff_colorized(self) -> None:
        original = "def foo():\n    pass\n"
        modified = 'def foo():\n    """Doc."""\n    pass\n'
        diff = generate_diff(original, modified, filename="test.py", colorize=True)

        assert "\033[32m+" in diff or "\033[" in diff


class TestWriteFileAtomically:
    """Tests for atomic file writing."""

    def test_write_file_atomically_success(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "output.py"
        content = "print('hello world')\n"

        write_file_atomically(target, content)
        assert target.exists()
        assert target.read_text(encoding="utf-8") == content

    def test_write_file_atomically_failure_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "output.py"
        with (
            patch("os.replace", side_effect=OSError("Permission denied")),
            pytest.raises(SourceEditError, match="Failed to atomically write"),
        ):
            write_file_atomically(target, "content")


class TestPreviewAndApply:
    """Tests for interactive/dry-run preview and apply logic."""

    def test_no_edits_returns_false(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        applied = preview_and_apply(
            path=target,
            original_source="def foo(): pass\n",
            edits=[],
            stream=buf,
        )
        assert applied is False
        assert "No documentation changes" in buf.getvalue()

    def test_dry_run_returns_false_and_does_not_write(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        target.write_text("def foo():\n    pass\n")

        edit = SourceEdit(
            function_id="fn_1",
            qualified_name="pkg.foo",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Doc."""',
        )

        applied = preview_and_apply(
            path=target,
            original_source=target.read_text(),
            edits=[edit],
            dry_run=True,
            stream=buf,
        )

        assert applied is False
        assert "Dry run enabled" in buf.getvalue()
        assert '"""Doc."""' not in target.read_text()

    def test_non_interactive_applies_directly(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        original = "def foo():\n    pass\n"
        target.write_text(original)

        edit = SourceEdit(
            function_id="fn_1",
            qualified_name="pkg.foo",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Doc."""',
        )

        applied = preview_and_apply(
            path=target,
            original_source=original,
            edits=[edit],
            interactive=False,
            stream=buf,
        )

        assert applied is True
        assert '"""Doc."""' in target.read_text()
        assert "Successfully applied 1 docstring(s)" in buf.getvalue()

    def test_interactive_user_confirms(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        original = "def foo():\n    pass\n"
        target.write_text(original)

        edit = SourceEdit(
            function_id="fn_1",
            qualified_name="pkg.foo",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Doc."""',
        )

        with patch("builtins.input", return_value="y"):
            applied = preview_and_apply(
                path=target,
                original_source=original,
                edits=[edit],
                interactive=True,
                stream=buf,
            )

        assert applied is True
        assert '"""Doc."""' in target.read_text()

    def test_interactive_user_declines(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        original = "def foo():\n    pass\n"
        target.write_text(original)

        edit = SourceEdit(
            function_id="fn_1",
            qualified_name="pkg.foo",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Doc."""',
        )

        with patch("builtins.input", return_value="n"):
            applied = preview_and_apply(
                path=target,
                original_source=original,
                edits=[edit],
                interactive=True,
                stream=buf,
            )

        assert applied is False
        assert "Changes discarded" in buf.getvalue()
        assert '"""Doc."""' not in target.read_text()

    def test_interactive_user_aborts(self, tmp_path: Path) -> None:
        buf = io.StringIO()
        target = tmp_path / "test.py"
        original = "def foo():\n    pass\n"
        target.write_text(original)

        edit = SourceEdit(
            function_id="fn_1",
            qualified_name="pkg.foo",
            insert_after_line=1,
            indentation="    ",
            docstring='    """Doc."""',
        )

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            applied = preview_and_apply(
                path=target,
                original_source=original,
                edits=[edit],
                interactive=True,
                stream=buf,
            )

        assert applied is False
        assert "Operation aborted by user" in buf.getvalue()
