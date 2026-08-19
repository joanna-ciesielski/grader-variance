"""ICC(2,1) and Krippendorff-alpha-over-repeats metrics (consolidation additions)."""

from __future__ import annotations

import numpy as np
import pytest
from raters.icc import icc as icc_fn

from grader_variance import icc_2_1, krippendorff_alpha_repeats

from ._helpers import sample_scores


class TestIcc21:
    def test_matches_library_directly(self) -> None:
        rng = np.random.default_rng(3)
        rows = np.clip(
            np.round(rng.uniform(1, 5, size=(20, 1)) + rng.normal(0, 0.5, (20, 6))),
            1,
            5,
        )
        got = icc_2_1()(sample_scores(rows.tolist()))
        assert got == pytest.approx(float(icc_fn(rows, form="2,1")))

    def test_penalizes_repeat_drift_below_icc11(self) -> None:
        # Second repeat uniformly +1: two-way absolute agreement must charge
        # the drift. (This is the property that distinguishes it from ICC(1,1)
        # on average and is why both are reported.)
        base = list(np.linspace(1.0, 5.0, 30))
        rows = [[v, v + 1.0] for v in base]
        drifted = icc_2_1()(sample_scores(rows))
        stable = icc_2_1()(sample_scores([[v, v] for v in base]))
        assert isinstance(drifted, float)
        assert isinstance(stable, float)
        assert stable == pytest.approx(1.0)
        assert drifted < stable

    def test_undefined_returns_nan(self) -> None:
        rows = [[1.0, 1.0], [1.0, 1.0]]
        result = icc_2_1()(sample_scores(rows))
        assert isinstance(result, float)
        assert np.isnan(result)


class TestAlphaRepeats:
    def test_perfect_repeatability(self) -> None:
        rows = [["C", "C", "C"], ["I", "I", "I"], ["C", "C", "C"]]
        assert krippendorff_alpha_repeats()(sample_scores(rows)) == pytest.approx(1.0)

    def test_noisy_repeats_below_one(self) -> None:
        rows = [["C", "I", "C"], ["I", "I", "C"], ["C", "C", "I"], ["I", "C", "I"]]
        value = krippendorff_alpha_repeats()(sample_scores(rows))
        assert isinstance(value, float)
        assert value < 0.5

    def test_interval_level_with_categories(self) -> None:
        rows = [[1, 2, 1], [4, 5, 4], [3, 3, 3]]
        value = krippendorff_alpha_repeats(
            level="interval", categories=[1, 2, 3, 4, 5]
        )(sample_scores(rows))
        assert isinstance(value, float)
        assert 0.5 < value <= 1.0
