"""Tests for reassembling a grades array and comparing two judges."""

from __future__ import annotations

import numpy as np
import pytest
from inspect_ai.log import EvalLog, EvalSample

from grader_variance import compare_judges, grades_array
from grader_variance._reshape import MODEL_SAMPLE_KEY


def _sample(
    item_id: str, epoch: int, grade: str, model_sample: int | None = None
) -> EvalSample:
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score

    metadata = {} if model_sample is None else {MODEL_SAMPLE_KEY: model_sample}
    return EvalSample(
        id=item_id,
        epoch=epoch,
        input="q",
        target="t",
        messages=[],
        output=ModelOutput.from_content(model="mockllm/model", content="c"),
        scores={"model_graded_fact": Score(value=grade)},
        metadata=metadata,
    )


def _log_with(samples: list[EvalSample]) -> EvalLog:
    return EvalLog.model_construct(samples=samples)


def test_grades_array_2d_single_completion() -> None:
    # 2 items, 3 grader repeats each, single model completion (no #m metadata).
    samples = [
        _sample("a", 1, "C"),
        _sample("a", 2, "I"),
        _sample("a", 3, "C"),
        _sample("b", 1, "I"),
        _sample("b", 2, "I"),
        _sample("b", 3, "C"),
    ]
    arr = grades_array(_log_with(samples))
    assert arr.shape == (2, 1, 3)
    np.testing.assert_allclose(arr[0, 0], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(arr[1, 0], [0.0, 0.0, 1.0])


def test_grades_array_3d_with_model_samples() -> None:
    # 2 items, 2 model completions each (via metadata), 2 grader repeats each.
    samples = [
        _sample("a", 1, "C", model_sample=1),
        _sample("a", 2, "C", model_sample=1),
        _sample("a", 1, "I", model_sample=2),
        _sample("a", 2, "C", model_sample=2),
        _sample("b", 1, "I", model_sample=1),
        _sample("b", 2, "I", model_sample=1),
        _sample("b", 1, "C", model_sample=2),
        _sample("b", 2, "I", model_sample=2),
    ]
    arr = grades_array(_log_with(samples))
    assert arr.shape == (2, 2, 2)
    np.testing.assert_allclose(arr[0, 0], [1.0, 1.0])  # a, model-sample 1
    np.testing.assert_allclose(arr[0, 1], [0.0, 1.0])  # a, model-sample 2


def test_grades_array_ragged_raises() -> None:
    samples = [
        _sample("a", 1, "C"),
        _sample("a", 2, "I"),
        _sample("b", 1, "C"),  # only one repeat for b
    ]
    with pytest.raises(ValueError, match="ragged"):
        grades_array(_log_with(samples))


# --- compare_judges: inter-rater diagnostics incl. bias index ---------------


def _judge_arrays_from_2x2(a: int, b: int, c: int, d: int) -> tuple[list, list]:
    """Per-item 0/1 grades for two judges reproducing a 2x2 [[a,b],[c,d]] table.

    a = A-pass & B-pass, b = A-pass & B-fail, c = A-fail & B-pass, d = both fail.
    """
    ga: list[float] = []
    gb: list[float] = []
    for count, (av, bv) in zip(
        (a, b, c, d), ((1, 1), (1, 0), (0, 1), (0, 0)), strict=True
    ):
        ga += [float(av)] * count
        gb += [float(bv)] * count
    return ga, gb


def test_compare_judges_reproduces_fc1990_table2() -> None:
    # Feinstein & Cicchetti 1990 Table 2: PI=0.75, BI=0.05, PABAK=0.70, kappa=0.3182.
    ga, gb = _judge_arrays_from_2x2(80, 10, 5, 5)
    d = compare_judges(ga, gb)
    assert d["prevalence_index"] == pytest.approx(0.75)
    assert d["bias_index"] == pytest.approx(0.05)
    assert d["pabak"] == pytest.approx(0.70)
    assert d["kappa"] == pytest.approx(0.3182, abs=1e-4)


def test_compare_judges_accepts_repeat_matrices() -> None:
    # Judges given (n_items, k) matrices are reduced to per-item mean then
    # thresholded; identical judges agree perfectly.
    a = np.array([[1, 1], [0, 0], [1, 0]], dtype=float)
    d = compare_judges(a, a)
    assert d["percent_agreement"] == pytest.approx(1.0)


def test_compare_judges_requires_same_items() -> None:
    with pytest.raises(ValueError, match="same items"):
        compare_judges([1.0, 0.0, 1.0], [1.0, 0.0])
