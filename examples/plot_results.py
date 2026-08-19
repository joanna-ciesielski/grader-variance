"""Render publication figures from a demo_study.py results JSON.

Produces two SVGs (and matching PNGs) next to the input file:

* ``<stem>_kcurve.svg`` -- the k-selection curve.
* ``<stem>_variance.svg`` -- question / model / grader variance shares.

Usage::

    python examples/plot_results.py demo_results.json
    python examples/plot_results.py demo_results_3way.json --out-dir figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COMPONENT_COLORS = {"question": "#2a78d6", "model": "#eb6834", "grader": "#1baf7a"}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
        }
    )


def _short(label: str) -> str:
    name = label.split("/", 1)[-1]
    parts = name.split("-")
    if parts and parts[-1].isdigit() and len(parts[-1]) >= 6:
        parts = parts[:-1]
    return "-".join(parts)


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    outputs = []
    for ext in ("svg", "png"):
        path = base.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def plot_kcurve(data: dict[str, Any], out_base: Path) -> list[Path]:
    results = data["results"]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, r in enumerate(results):
        color = SERIES[i % len(SERIES)]
        ks = [p["k"] for p in r["k_curve"]]
        ses = [p["score_se"] for p in r["k_curve"]]
        ax.plot(
            ks,
            ses,
            "-o",
            color=color,
            linewidth=2,
            markersize=6,
            label=_short(r["grader"]),
        )
        rec_k = r.get("recommended_k")
        if isinstance(rec_k, int) and rec_k <= ks[-1]:
            se_at = next((p["score_se"] for p in r["k_curve"] if p["k"] == rec_k), None)
            if se_at is not None:
                ax.plot([rec_k], [se_at], marker="D", color=color, markersize=8)
                ax.annotate(
                    f"k*={rec_k}",
                    xy=(rec_k, se_at),
                    xytext=(0, 10),
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                    ha="center",
                )
    ax.set_xlabel("grader repeats k")
    ax.set_ylabel("std. error of benchmark score")
    ax.set_title(
        "CI on the benchmark score vs. grader repeats", color=INK, fontsize=12, pad=12
    )
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    return _save(fig, out_base)


def plot_variance(data: dict[str, Any], out_base: Path) -> list[Path]:
    results = data["results"]
    labels = [_short(r["grader"]) for r in results]
    components = ["question", "model", "grader"]
    fig, ax = plt.subplots(figsize=(7.0, 0.9 * len(results) + 1.8))
    y = list(range(len(results)))
    for row, r in enumerate(results):
        dec = r["decomposition"]
        left = 0.0
        total = dec["question"] + dec["model"] + dec["grader"]
        for comp in components:
            share = (dec[comp] / total) if total > 0 else 0.0
            ax.barh(
                row,
                share,
                left=left,
                color=COMPONENT_COLORS[comp],
                edgecolor=SURFACE,
                linewidth=2,
                height=0.6,
            )
            if share >= 0.06:
                ax.text(
                    left + share / 2,
                    row,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    color="#ffffff",
                    fontsize=9,
                    fontweight="bold",
                )
            left += share
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of question + model + grader variance")
    ax.set_title("Variance decomposition by judge", color=INK, fontsize=12, pad=12)
    ax.invert_yaxis()
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COMPONENT_COLORS[c]) for c in components
    ]
    ax.legend(
        handles,
        components,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=9,
    )
    return _save(fig, out_base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, help="a demo_study.py results JSON")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="directory for the figures (default: alongside the input)",
    )
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    out_dir = args.out_dir or args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results.stem
    _apply_style()
    written: list[Path] = []
    written += plot_kcurve(data, out_dir / f"{stem}_kcurve")
    written += plot_variance(data, out_dir / f"{stem}_variance")
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
