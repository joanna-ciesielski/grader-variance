"""grader-variance: Inspect-native measurement of LLM-grader variance.

An Inspect extension that isolates grader (judge) variance from model-sampling
variance, decomposes a benchmark score's spread into question / model / grader
components, and gives a stopping rule for how many grader repeats are needed.

What this adds over Inspect core. Inspect ships ``krippendorff_alpha()`` for
agreement *across multiple judges* and ``epochs`` that re-run the *model*. This
package adds the intra-grader (one judge, repeated) case: a fixed-completion
harness, intra-rater metrics, the variance decomposition, and the k-selection
stopping rule.

Prior art. Grader self-inconsistency is established (Rating Roulette,
arXiv 2510.27106; Reliability without Validity, arXiv 2606.19544). The statistics
are computed by the ``rater-agreement`` library (DOI 10.5281/zenodo.21983269).
The novel contribution is the decomposition and the grader stopping rule, not the
observation that judges are noisy.
"""

from __future__ import annotations

from .analysis import compare_judges, grades_array
from .decomposition import (
    KSelectionCurve,
    KSelectionPoint,
    VarianceComponents,
    decompose,
    k_selection_curve,
    stopping_rule_text,
)
from .dstudy import dependability, estimate_components_reml, repeats_needed
from .harness import (
    FrozenCheck,
    completion_digest,
    freeze_completions,
    regrade_frozen,
    verify_frozen,
)
from .manifest import (
    CANARY,
    RunManifest,
    dataset_revision_for_bundled_example,
    frozen_digests,
    resolve_model_id,
    verify_manifest_digests,
    write_run_manifest,
)
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

__version__ = "0.1.0"

__all__ = [
    # metrics (intra-grader)
    "flip_rate",
    "icc_1_1",
    "icc_2_1",
    "krippendorff_alpha_repeats",
    "test_retest",
    "pabak",
    "prevalence_index",
    "grader_variance_share",
    # harness
    "freeze_completions",
    "regrade_frozen",
    "verify_frozen",
    "completion_digest",
    "FrozenCheck",
    # decomposition
    "decompose",
    "k_selection_curve",
    "stopping_rule_text",
    "grades_array",
    "compare_judges",
    "VarianceComponents",
    "KSelectionCurve",
    "KSelectionPoint",
    # D-study
    "dependability",
    "repeats_needed",
    "estimate_components_reml",
    # manifest / provenance
    "CANARY",
    "RunManifest",
    "dataset_revision_for_bundled_example",
    "frozen_digests",
    "resolve_model_id",
    "verify_manifest_digests",
    "write_run_manifest",
]
