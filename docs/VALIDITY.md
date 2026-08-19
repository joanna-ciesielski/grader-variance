# Validity — what this study cannot tell you

This section is written **before** the demonstration study is run, deliberately.
It is not a disclaimer bolted on after the fact; it is the boundary of the claim,
fixed in advance so that the results cannot quietly expand to fill whatever the
numbers happen to allow. If the demonstration in `examples/demo_study.py`
produces a clean decomposition and a tidy stopping rule, everything below still
applies to it.

## What the study is

A single demonstration: one existing Inspect eval, one task type, one
model-graded scorer, two judge model families, run through the fixed-completion
harness to isolate grader variance, decomposed into question / model / grader
components, with a k-selection curve for the number of grader repeats. Its
purpose is to show the tooling produces a coherent, reproducible measurement and
a stopping rule stated in Miller's form — not to characterise LLM graders in
general.

## What it cannot tell you

**One benchmark, one task type.** The decomposition and the recommended k are
properties of *this* benchmark with *this* scorer. A benchmark whose questions
are more homogeneous will have smaller between-item variance, which — holding
grader noise fixed — inflates the grader's share and pushes the recommended k
up. Nothing here establishes what the numbers would be on a different dataset,
and the k-selection formula makes that dependence explicit
(`sigma2_grader / k <= tol * sigma2_between`): change either variance term and
the answer moves.

**Whether the rule transfers.** A stopping rule fitted on one benchmark is an
estimate for that benchmark. We have no second benchmark here to test transfer,
so any use of this k on another eval is extrapolation. The honest use of the
number is "this is how many repeats *this* benchmark needed," not "use k=N for
model-graded evals."

**Two judge families is not a survey of judges.** Reliability without Validity
(arXiv 2606.19544) already surveyed 21 models; this study deliberately does not.
Two families are enough to show the harness handles more than one judge and to
see whether the grader share is similar across them — not enough to say anything
general about which judges are stable.

**Temperature and seed are not separated.** Grader variance as measured here is
the total run-to-run variance of the judge at whatever sampling configuration it
was called with. We do not decompose it into a temperature component and a
seed/sampling component. A judge called at temperature 0 with a fixed seed may
show near-zero variance that is an artifact of the configuration, not evidence of
a reliable judge; a judge at temperature 1 will show more. The measured grader
share is conditional on the sampling settings and is reported as such.

**Caching semantics can silently zero out the grader term.** If the grader model
provider caches responses keyed on the prompt, repeated grading of an identical
frozen completion can return the *same* cached response every time, collapsing
measured grader variance to zero for reasons that have nothing to do with judge
reliability. The harness freezes the *completion*, not the grader's sampling; it
cannot detect provider-side caching. A grader variance of zero must therefore be
read together with the provider's caching behaviour before it is believed — and
the `verify_frozen` check guards the completion side of this, not the grader
side.

**Isolation is of grader-from-model, not grader-from-everything.** The harness
proves (by byte-identical completion digests) that the grader is scoring the same
completion across repeats. It does not control the grader's own prompt
construction, ordering effects, or any state the scorer carries; it isolates the
*model-sampling* confound specifically, which is the one Inspect's `epochs`
otherwise reintroduces.

## What a null result means

If the grader variance term is small — the plausible outcome, given a
low-temperature judge on an unambiguous task — that is a real finding and is
published as-is, not buried. "For this benchmark and this scorer, the grader
contributes a negligible share of the score variance and does not need
repeating" is exactly as useful to an eval author as a large grader term, and the
k-selection curve will show it directly (recommended k = 1). A small measured
grader term is only *uninformative* if it is an artifact of caching or a
degenerate sampling configuration — which is why the two points above matter.

## Relationship to prior claims

That LLM judges are self-inconsistent across repeated runs is established prior
art (Rating Roulette, arXiv 2510.27106; Reliability without Validity,
arXiv 2606.19544) and is never presented as a discovery here. (A 0.38 decision
flip rate cited in earlier project notes was traced to a seeded synthetic
worked example in the rater-agreement library — no model calls — and is not a
measurement; it is excluded from all claims.) Miller ("Adding
Error Bars to Evals", arXiv 2411.00640) decomposes the question and
model-sampling variance components and assumes scores are already computed. The
only new element is the grader variance term and its stopping rule; the validity
of *that* element is bounded by everything above.
