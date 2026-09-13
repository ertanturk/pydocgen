"""Unit tests for token estimation heuristics, FunctionBatch model, and batch creation."""

from dataclasses import FrozenInstanceError

import pytest

from pydocgen.analysis.models import FunctionInfo, ParameterInfo
from pydocgen.batching import (
    FunctionBatch,
    create_batches,
    estimate_function_tokens,
    estimate_text_tokens,
)
from pydocgen.batching.batcher import _generate_batch_id
from pydocgen.errors import BatchError


def _make_dummy_function(
    id_suffix: str = "1",
    name: str = "dummy",
    qualified_name: str = "dummy",
    source_len: int = 35,
) -> FunctionInfo:
    """Helper to create dummy FunctionInfo instances with predictable source size."""
    # With CHARS_PER_TOKEN = 3.5, source of len 35 yields ceil(35 / 3.5) = 10 tokens.
    # Total function tokens with overhead (40) = 50 tokens.
    padding = "x" * max(0, source_len - len(f"def {name}(): return "))
    source = f"def {name}(): return '{padding}'"
    return FunctionInfo(
        id=f"fn_{id_suffix.zfill(8)}",
        name=name,
        qualified_name=qualified_name,
        source=source,
        parameters=(),
        return_annotation=None,
        decorators=(),
        line_start=1,
        line_end=2,
        has_docstring=False,
        is_async=False,
        is_method=False,
    )


class TestTokenEstimation:
    """Test text and function token estimation heuristics."""

    def test_estimate_text_tokens_empty_string(self) -> None:
        assert estimate_text_tokens("") == 0

    def test_estimate_text_tokens_single_character(self) -> None:
        assert estimate_text_tokens("a") == 1

    def test_estimate_text_tokens_standard(self) -> None:
        # len 35 / 3.5 = 10
        text = "x" * 35
        assert estimate_text_tokens(text) == 10

    def test_estimate_text_tokens_rounding_up(self) -> None:
        # len 4 / 3.5 = 1.14 -> ceil -> 2
        text = "xxxx"
        assert estimate_text_tokens(text) == 2

    def test_estimate_text_tokens_custom_ratio(self) -> None:
        # len 20 / 2.0 = 10
        text = "x" * 20
        assert estimate_text_tokens(text, chars_per_token=2.0) == 10

    def test_estimate_text_tokens_rejects_non_string(self) -> None:
        for invalid in [None, 123, ["text"], b"code"]:
            with pytest.raises(TypeError, match="Expected text as string"):
                estimate_text_tokens(invalid)  # ty: ignore[invalid-argument-type]

    def test_estimate_text_tokens_rejects_non_positive_ratio(self) -> None:
        for invalid_ratio in [0, -1.0, -3.5]:
            with pytest.raises(ValueError, match="chars_per_token must be positive"):
                estimate_text_tokens("code", chars_per_token=invalid_ratio)

    def test_estimate_function_tokens_standard(self) -> None:
        # Function source with 35 chars -> 10 tokens + default overhead 40 = 50 tokens
        fn = _make_dummy_function(source_len=35)
        tokens = estimate_function_tokens(fn)
        assert tokens == estimate_text_tokens(fn.source) + 40

    def test_estimate_function_tokens_custom_overhead(self) -> None:
        fn = _make_dummy_function(source_len=35)
        tokens = estimate_function_tokens(fn, overhead_tokens=100)
        assert tokens == estimate_text_tokens(fn.source) + 100

    def test_estimate_function_tokens_rejects_non_function_info(self) -> None:
        for invalid in [None, "function", 123]:
            with pytest.raises(TypeError, match="Expected FunctionInfo instance"):
                estimate_function_tokens(invalid)  # ty: ignore[invalid-argument-type]

    def test_estimate_function_tokens_rejects_negative_overhead(self) -> None:
        fn = _make_dummy_function()
        with pytest.raises(ValueError, match="overhead_tokens cannot be negative"):
            estimate_function_tokens(fn, overhead_tokens=-5)


class TestFunctionBatchModel:
    """Test FunctionBatch container methods, properties, and validation."""

    def test_create_batch_success(self) -> None:
        fn = _make_dummy_function()
        batch = FunctionBatch(
            id="batch-001-a1b2c3d4",
            functions=(fn,),
            estimated_input_tokens=550,
        )
        assert batch.id == "batch-001-a1b2c3d4"
        assert batch.functions == (fn,)
        assert batch.estimated_input_tokens == 550

    def test_batch_immutability(self) -> None:
        batch = FunctionBatch(
            id="batch-001-test",
            functions=(),
            estimated_input_tokens=500,
        )
        with pytest.raises(FrozenInstanceError):
            batch.id = "modified"  # ty: ignore[invalid-assignment]

    def test_batch_len_and_size_property(self) -> None:
        fn1 = _make_dummy_function(id_suffix="1")
        fn2 = _make_dummy_function(id_suffix="2")
        batch = FunctionBatch(
            id="batch-001-test",
            functions=(fn1, fn2),
            estimated_input_tokens=600,
        )
        assert len(batch) == 2
        assert batch.size == 2

    def test_batch_iteration_and_indexing(self) -> None:
        fn1 = _make_dummy_function(id_suffix="1", name="first")
        fn2 = _make_dummy_function(id_suffix="2", name="second")
        batch = FunctionBatch(
            id="batch-001-test",
            functions=(fn1, fn2),
            estimated_input_tokens=600,
        )
        assert list(batch) == [fn1, fn2]
        assert batch[0] == fn1
        assert batch[1] == fn2

    def test_batch_function_ids_property(self) -> None:
        fn1 = _make_dummy_function(id_suffix="1")
        fn2 = _make_dummy_function(id_suffix="2")
        batch = FunctionBatch(
            id="batch-001-test",
            functions=(fn1, fn2),
            estimated_input_tokens=600,
        )
        assert batch.function_ids == (fn1.id, fn2.id)

    def test_batch_list_conversion_in_post_init(self) -> None:
        fn = _make_dummy_function()
        batch = FunctionBatch(
            id="batch-001-test",
            functions=[fn],  # ty: ignore[invalid-argument-type]
            estimated_input_tokens=550,
        )
        assert isinstance(batch.functions, tuple)
        assert batch.functions == (fn,)

    def test_batch_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="Batch id must be a non-empty string"):
            FunctionBatch(id="", functions=(), estimated_input_tokens=500)

    def test_batch_rejects_whitespace_only_id(self) -> None:
        with pytest.raises(ValueError, match="Batch id must be a non-empty string"):
            FunctionBatch(id="   ", functions=(), estimated_input_tokens=500)

    def test_batch_rejects_non_string_id(self) -> None:
        with pytest.raises(ValueError, match="Batch id must be a non-empty string"):
            FunctionBatch(id=123, functions=(), estimated_input_tokens=500)  # ty: ignore[invalid-argument-type]

    def test_batch_rejects_negative_tokens(self) -> None:
        with pytest.raises(
            ValueError, match="estimated_input_tokens must be a non-negative integer"
        ):
            FunctionBatch(id="batch-001", functions=(), estimated_input_tokens=-1)

    def test_batch_rejects_non_function_info_elements(self) -> None:
        param = ParameterInfo(name="arg")
        with pytest.raises(TypeError, match="All functions in batch must be FunctionInfo"):
            FunctionBatch(id="batch-001", functions=(param,), estimated_input_tokens=500)  # ty: ignore[invalid-argument-type]


class TestGenerateBatchId:
    """Test deterministic batch ID generation."""

    def test_deterministic_id(self) -> None:
        fn1 = _make_dummy_function(id_suffix="1")
        fn2 = _make_dummy_function(id_suffix="2")

        id1 = _generate_batch_id(1, [fn1, fn2])
        id2 = _generate_batch_id(1, [fn1, fn2])
        assert id1 == id2
        assert id1.startswith("batch-001-")

    def test_different_index_yields_different_id(self) -> None:
        fn = _make_dummy_function()
        id1 = _generate_batch_id(1, [fn])
        id2 = _generate_batch_id(2, [fn])
        assert id1 != id2
        assert id1.startswith("batch-001-")
        assert id2.startswith("batch-002-")

    def test_different_functions_yield_different_id(self) -> None:
        fn1 = _make_dummy_function(id_suffix="1")
        fn2 = _make_dummy_function(id_suffix="2")
        id1 = _generate_batch_id(1, [fn1])
        id2 = _generate_batch_id(1, [fn2])
        assert id1 != id2

    def test_rejects_index_less_than_one(self) -> None:
        fn = _make_dummy_function()
        for invalid_idx in [0, -1]:
            with pytest.raises(ValueError, match="Batch index must be at least 1"):
                _generate_batch_id(invalid_idx, [fn])


class TestCreateBatches:
    """Test partitioning logic, boundaries, and validation in create_batches."""

    def test_empty_functions_returns_empty_list(self) -> None:
        result = create_batches([])
        assert result == []

    def test_single_function_batch(self) -> None:
        fn = _make_dummy_function(id_suffix="1")
        fn_tokens = estimate_function_tokens(fn)

        batches = create_batches(
            [fn],
            max_batch_tokens=1000,
            prompt_overhead_tokens=500,
        )
        assert len(batches) == 1
        batch = batches[0]
        assert batch.size == 1
        assert batch.functions == (fn,)
        assert batch.estimated_input_tokens == 500 + fn_tokens
        assert batch.id.startswith("batch-001-")

    def test_multiple_functions_single_batch(self) -> None:
        funcs = [_make_dummy_function(id_suffix=str(i)) for i in range(3)]
        total_fn_tokens = sum(estimate_function_tokens(fn) for fn in funcs)

        batches = create_batches(
            funcs,
            max_batch_tokens=2000,
            prompt_overhead_tokens=500,
            max_functions_per_batch=10,
        )
        assert len(batches) == 1
        assert batches[0].size == 3
        assert batches[0].estimated_input_tokens == 500 + total_fn_tokens

    def test_token_overflow_splits_batches(self) -> None:
        # overhead = 500, max = 610 -> usable budget = 110
        # each function is 51 tokens -> 2 functions fit (500 + 51 + 51 = 602 <= 610)
        # 3rd function requires 51 tokens, causing overflow (602 + 51 = 653 > 610)
        funcs = [_make_dummy_function(id_suffix=str(i)) for i in range(5)]
        batches = create_batches(
            funcs,
            max_batch_tokens=610,
            prompt_overhead_tokens=500,
            max_functions_per_batch=None,
        )
        # 5 functions with 2 per batch -> 3 batches (2, 2, 1)
        assert len(batches) == 3
        assert [b.size for b in batches] == [2, 2, 1]
        assert batches[0].id.startswith("batch-001-")
        assert batches[1].id.startswith("batch-002-")
        assert batches[2].id.startswith("batch-003-")

    def test_count_overflow_splits_batches(self) -> None:
        funcs = [_make_dummy_function(id_suffix=str(i)) for i in range(5)]
        # Plentiful token budget, but capped at 2 functions per batch
        batches = create_batches(
            funcs,
            max_batch_tokens=10000,
            prompt_overhead_tokens=500,
            max_functions_per_batch=2,
        )
        assert len(batches) == 3
        assert [b.size for b in batches] == [2, 2, 1]

    def test_no_count_limit_when_max_functions_none(self) -> None:
        funcs = [_make_dummy_function(id_suffix=str(i)) for i in range(10)]
        batches = create_batches(
            funcs,
            max_batch_tokens=10000,
            prompt_overhead_tokens=500,
            max_functions_per_batch=None,
        )
        assert len(batches) == 1
        assert batches[0].size == 10

    def test_single_function_exceeds_budget_raises_batch_error(self) -> None:
        # Function source with 1000 chars -> ceil(1000 / 3.5) = 286 tokens + 40 = 326 tokens
        huge_fn = _make_dummy_function(qualified_name="HugeModule.big_func", source_len=1000)
        huge_tokens = estimate_function_tokens(huge_fn)

        # Usable budget is 600 - 500 = 100 tokens, which is smaller than huge_tokens
        with pytest.raises(BatchError) as exc_info:
            create_batches(
                [huge_fn],
                max_batch_tokens=600,
                prompt_overhead_tokens=500,
            )

        err_msg = str(exc_info.value)
        assert "HugeModule.big_func" in err_msg
        assert str(huge_tokens) in err_msg
        assert "exceeds the maximum usable batch budget of 100" in err_msg

    def test_rejects_max_batch_tokens_less_than_overhead(self) -> None:
        with pytest.raises(BatchError, match="must exceed prompt_overhead_tokens"):
            create_batches([], max_batch_tokens=500, prompt_overhead_tokens=500)

        with pytest.raises(BatchError, match="must exceed prompt_overhead_tokens"):
            create_batches([], max_batch_tokens=400, prompt_overhead_tokens=500)

    def test_rejects_non_positive_max_batch_tokens(self) -> None:
        with pytest.raises(BatchError, match="must be greater than zero"):
            create_batches([], max_batch_tokens=0, prompt_overhead_tokens=0)

        with pytest.raises(BatchError, match="must be greater than zero"):
            create_batches([], max_batch_tokens=-10, prompt_overhead_tokens=0)

    def test_rejects_negative_prompt_overhead(self) -> None:
        with pytest.raises(BatchError, match="cannot be negative"):
            create_batches([], max_batch_tokens=1000, prompt_overhead_tokens=-1)

    def test_rejects_non_positive_max_functions(self) -> None:
        with pytest.raises(BatchError, match="max_functions_per_batch.*greater than zero"):
            create_batches([], max_functions_per_batch=0)

        with pytest.raises(BatchError, match="max_functions_per_batch.*greater than zero"):
            create_batches([], max_functions_per_batch=-2)

    def test_rejects_non_sequence_functions(self) -> None:
        for invalid in [None, 123, "not-a-sequence"]:
            with pytest.raises(TypeError, match="functions must be a sequence"):
                create_batches(invalid)  # ty: ignore[invalid-argument-type]

    def test_rejects_invalid_elements_in_functions(self) -> None:
        for invalid_elem in [None, "str", 123]:
            with pytest.raises(TypeError, match="All items in functions must be FunctionInfo"):
                create_batches([invalid_elem])  # ty: ignore[invalid-argument-type]


class TestBatchingPackageExports:
    """Test public API exposure from pydocgen.batching and pydocgen.errors."""

    def test_batching_exports(self) -> None:
        import pydocgen.batching as batching

        assert hasattr(batching, "FunctionBatch")
        assert hasattr(batching, "create_batches")
        assert hasattr(batching, "estimate_function_tokens")
        assert hasattr(batching, "estimate_text_tokens")

    def test_batch_error_exported_from_errors(self) -> None:
        import pydocgen.errors as errors

        assert hasattr(errors, "BatchError")
        assert errors.BatchError is BatchError
