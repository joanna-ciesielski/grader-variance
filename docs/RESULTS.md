# Results — definitive study (2026-08-19, k = 12)

This run **supersedes the k = 6 demonstration of 2026-08-19 (morning)** in its
entirety. No number from the earlier run is reported anywhere; where the two
runs disagree, this one stands. Validity constraints: see `docs/VALIDITY.md`
(written before results; this file defers to it).

## Configuration (from `demo_results.manifest.json`)

| | |
|---|---|
| Benchmark | `theory_of_mind` (Inspect bundled example), model-graded (`model_graded_fact`) |
| Dataset pin | `inspect_ai-0.3.259/theory_of_mind.jsonl@sha256:d67dcc9d8f95a9e22453016f28fb99fba8cf17b1d32f6f4dee739c5b6e7fd61c` |
| Model under test | `anthropic/claude-sonnet-5` (alias = served ID at run time) |
| Judges | `anthropic/claude-haiku-4-5-20251001`; `openai/gpt-4.1-mini-2025-04-14` (resolved from `gpt-4.1-mini` by 1-token ping) |
| Design | 50 questions × 3 model epochs × 12 grader repeats per judge (1,800 grades/judge) |
| Grader temperature | 1.0 |
| Byte identity | `verify_frozen` passed (150 units × 12 repeats, per judge) **and** every regrade verified against the frozen SHA-256 digests in the run manifest |
| Canary | carried in the manifest (`PAPER-A CANARY GUID 8f4c1d2e…`) |

## Variance decomposition (nested random-effects ANOVA, method of moments)

Estimates are latent variance components of a single grade; shares are of the
component total. REML (statsmodels, `grader-variance[analysis]`) is reported
as an independent estimator cross-check.

| Judge | Question | Model sampling | **Grader re-scoring** | Grader share |
|---|---|---|---|---|
| claude-haiku-4-5-20251001 | 0.0994 (88.2%) | 0.0133 (11.8%) | **0.0000 (0.0%)** | **0.0%** |
| gpt-4.1-mini-2025-04-14 | 0.0508 (62.6%) | 0.0071 (8.8%) | **0.0232 (28.6%)** | **28.6%** |

REML cross-check: for gpt-4.1-mini, REML reproduces the moments estimates to
five decimals (0.050777 / 0.007138 / 0.023232 — shares 62.57 / 8.80 / 28.63%).
For haiku, the REML fit sits on the boundary of the parameter space
(statsmodels `ConvergenceWarning`), which is the expected behavior when a
variance component is exactly zero; the boundary fit is reported as
corroboration that the grader component is zero, and the moments estimates
(exact for balanced designs) are primary.

## Stopping rules (both forms agree)

- **Miller-form** (grader term ≤ 5% of between-item variance,
  σ²_grader/(m·k) ≤ 0.05·σ²_between): haiku **k\* = 1**; gpt-4.1-mini
  **k\* = 4**.
- **G-theory dependability** (Φ(k) ≥ 0.9, Φ(k) = (σ²_q+σ²_m)/(σ²_q+σ²_m+σ²_g/k)):
  haiku **k\* = 1** (Φ(1) = 1.0); gpt-4.1-mini **k\* = 4** (Φ(1) = 0.714,
  Φ(4) = 0.909, Φ(12) = 0.968).

Benchmark-score SE: haiku 0.0456 flat in k; gpt-4.1-mini 0.0349 (k=1) →
0.0328 (k=12), i.e. the grader term is worth ~6% of the SE at k=1 for this
judge and nothing for the other.

## Inter-judge comparison (same frozen completions)

Percent agreement 96%, Cohen's κ 0.811, prevalence index 0.76, bias index
0.04, PABAK 0.92 — no kappa paradox; the two judges agree substantially and
neither is systematically more lenient at the pass mark.

## The headline reading

Two judges, same benchmark, same frozen completions: one has **zero**
measurable re-scoring variance (k = 1 suffices), the other's re-scoring noise
is **28.6% of grade variance** (k = 4 required for Φ ≥ 0.9). Published setups
report still other values — 11–15 repeats (Coin Flip Judge, arXiv 2606.13685)
and 44% judge-share (Messing, arXiv 2604.11581). Required grader repeats are
an **empirical property of the (benchmark, judge) pair**, spanning an order of
magnitude across setups; the only way to know yours is to measure it, and this
harness makes that a one-line check.

## Honesty notes

1. **Haiku's exact zero across 1,800 re-scorings (T = 1.0).** Verified not to
   be an artifact of provider response caching: [PENDING — 20-call probe:
   distinct reasoning texts with constant grades rules caching out; record the
   probe output here before citing the zero anywhere].
2. **"Grader" is the within-cell residual.** Under the nested design
   (question / model-sample / grader-repeat) with a single grader whose
   repeats are exchangeable, the grader component and the residual are the
   same term by construction; there is no separable fourth component. Stated
   in `grader_variance/dstudy.py` and in the paper.
3. The 0.38 flip rate from earlier project notes is a seeded synthetic
   fixture (no model calls) and appears nowhere in these results.
4. Grader-share depends on the judge-induced grade distribution, so the
   question/model components legitimately differ between judges (each judge
   defines its own grade random variable over the same frozen completions).
