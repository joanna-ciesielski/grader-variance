"""Inspect extension entry point.

Importing this module registers the grader-variance metrics with Inspect's
registry (the ``@metric`` decorators run on import). Referenced by the
``inspect_ai`` entry point in ``pyproject.toml`` so that Inspect discovers the
metrics without an explicit user import.
"""

from __future__ import annotations

from .metrics import (
    flip_rate,
    grader_variance_share,
    icc_1_1,
    icc_2_1,
    krippendorff_alpha_repeats,
    pabak,
    prevalence_index,
    test_retest,
)

__all__ = [
    "flip_rate",
    "icc_1_1",
    "icc_2_1",
    "krippendorff_alpha_repeats",
    "test_retest",
    "pabak",
    "prevalence_index",
    "grader_variance_share",
]
