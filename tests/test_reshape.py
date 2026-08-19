"""Tests for the unreduced-score reshaping layer."""

from __future__ import annotations

import pytest

from grader_variance._reshape import (
    RaggedRepeatsError,
    group_by_item,
    label_matrix,
    numeric_matrix,
)

from ._helpers import sample_scores


def test_group_preserves_item_order_and_repeats() -> None:
    scores = sample_scores([["C", "I"], ["C", "C"]])
    grouped = group_by_item(scores)
    assert list(grouped.keys()) == ["0", "1"]
    assert grouped["0"] == ["C", "I"]
    assert grouped["1"] == ["C", "C"]


def test_label_matrix_rectangular() -> None:
    scores = sample_scores([["C", "I", "C"], ["I", "I", "C"]])
    assert label_matrix(scores) == [["C", "I", "C"], ["I", "I", "C"]]


def test_numeric_matrix_maps_letters() -> None:
    # C/I/P -> 1.0/0.0/0.5 via value_to_float default.
    scores = sample_scores([["C", "I"], ["P", "C"]])
    assert numeric_matrix(scores) == [[1.0, 0.0], [0.5, 1.0]]


def test_ragged_repeats_detected() -> None:
    # Item 0 has 2 repeats, item 1 has 3 -> not rectangular.
    from inspect_ai.scorer import SampleScore, Score

    scores = [
        SampleScore(score=Score(value="C"), sample_id=0),
        SampleScore(score=Score(value="I"), sample_id=0),
        SampleScore(score=Score(value="C"), sample_id=1),
        SampleScore(score=Score(value="C"), sample_id=1),
        SampleScore(score=Score(value="I"), sample_id=1),
    ]
    with pytest.raises(RaggedRepeatsError):
        label_matrix(scores)


def test_rejects_nonscalar_values() -> None:
    from inspect_ai.scorer import SampleScore, Score

    scores = [SampleScore(score=Score(value=[1, 2]), sample_id=0)]
    with pytest.raises(ValueError, match="scalar"):
        group_by_item(scores)


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="no scores"):
        label_matrix([])
