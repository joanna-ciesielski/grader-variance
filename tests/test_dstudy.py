"""D-study utilities: dependability, repeats_needed, and the REML cross-check."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from grader_variance import (
    VarianceComponents,
    decompose,
    dependability,
    repeats_needed,
)

HAS_STATSMODELS = importlib.util.find_spec("statsmodels") is not None


def simulate(
    n_items: int,
    m: int,
    k: int,
    seed: int,
    s2_question: float = 0.5,
    s2_model: float = 0.2,
    s2_grader: float = 0.3,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, np.sqrt(s2_question), size=(n_items, 1, 1))
    b = rng.normal(0.0, np.sqrt(s2_model), size=(n_items, m, 1))
    e = rng.normal(0.0, np.sqrt(s2_grader), size=(n_items, m, k))
    return np.asarray(3.0 + a + b + e)


def comp(q: float, m: float, g: float) -> VarianceComponents:
    return VarianceComponents(question=q, model=m, grader=g)


class TestDependability:
    def test_closed_form(self) -> None:
        # Phi(3) = 0.7 / (0.7 + 0.1) = 0.875
        assert dependability(comp(0.5, 0.2, 0.3), 3) == pytest.approx(0.875)

    def test_monotone_in_k(self) -> None:
        c = comp(0.5, 0.2, 0.3)
        values = [dependability(c, k) for k in range(1, 30)]
        assert all(b > a for a, b in zip(values, values[1:], strict=False))

    def test_rejects_bad_k(self) -> None:
        with pytest.raises(ValueError):
            dependability(comp(0.5, 0.2, 0.3), 0)


class TestRepeatsNeeded:
    def test_tight(self) -> None:
        c = comp(0.5, 0.2, 0.3)
        for target in (0.8, 0.9, 0.95, 0.99):
            k = repeats_needed(c, target)
            assert k is not None
            assert dependability(c, k) >= target
            if k > 1:
                assert dependability(c, k - 1) < target

    def test_no_signal_returns_none(self) -> None:
        assert repeats_needed(comp(0.0, 0.0, 1.0), 0.9) is None

    def test_no_noise_returns_one(self) -> None:
        assert repeats_needed(comp(0.5, 0.2, 0.0), 0.99) == 1

    def test_cap(self) -> None:
        # Huge grader noise relative to signal: required k blows past the cap.
        assert repeats_needed(comp(1e-9, 0.0, 1.0), 0.99, max_repeats=100) is None


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
class TestREMLCrossCheck:
    def test_agrees_with_moments_estimator(self) -> None:
        from grader_variance import estimate_components_reml

        x = simulate(300, 3, 8, seed=42)
        moments = decompose(x)
        reml = estimate_components_reml(x)
        assert reml.question == pytest.approx(moments.question, abs=0.03)
        assert reml.model == pytest.approx(moments.model, abs=0.03)
        assert reml.grader == pytest.approx(moments.grader, abs=0.01)

    def test_recovers_known_components(self) -> None:
        from grader_variance import estimate_components_reml

        x = simulate(400, 3, 10, seed=7)
        reml = estimate_components_reml(x)
        assert reml.question == pytest.approx(0.5, abs=0.09)
        assert reml.model == pytest.approx(0.2, abs=0.05)
        assert reml.grader == pytest.approx(0.3, abs=0.02)

    def test_requires_multiple_model_samples(self) -> None:
        from grader_variance import estimate_components_reml

        x = simulate(20, 1, 5, seed=1)
        with pytest.raises(ValueError, match="n_model_samples >= 2"):
            estimate_components_reml(x)
