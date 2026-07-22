#!/usr/bin/env python3
"""
plot_gp_vs_rf.py — Paired GP-vs-RF comparison for the targeted GP grid.

The GP sweep runs a subset of cells (a few datasets/reprs/acqs) with the GP
surrogate; the RF baseline for the same cells comes from the main sweep. This
script pairs them: for each (dataset, representation) it plots final-round GP vs
RF side by side, matched on the acquisitions that BOTH surrogates ran (the GP
grid omits random/diversity/retrieval, so a naive mean over acquisitions would
compare GP's {greedy,ucb} against RF's five — this restricts to the shared set).

Produces, per metric, one grouped-bar figure (representations on x, GP/RF pairs)
faceted by dataset, plus a printed delta table.

Usage
-----
python scripts/plot_gp_vs_rf.py \
    --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012 \
    --metrics topk10_recall simple_regret pool_spearman \
    --output_dir docs/figures/gp_vs_rf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

_REPR_ORDER = [
    "mutation", "physicochemical",
    "plm_mean", "plm_delta", "plm_site",
    "plm_physico", "plm_concat", "plm_retrieval",
]
_REPR_LABELS = {
    "mutation": "Mutation", "physicochemical": "Physico",
    "plm_mean": "PLM mean", "plm_delta": "PLM delta", "plm_site": "PLM site",
    "plm_physico": "PLM+physico", "plm_concat": "PLM concat",
    "plm_retrieval": "PLM+retrieval",
}
_SURR_COLORS = {"rf": "#4E79A7", "gp": "#E15759"}   # RF blue, GP red
_SURR_LABELS = {"rf": "RF", "gp": "GP"}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load every seed_*.csv (excluding selections) under results_dir."""
    dfs = []
    for seed_csv in sorted(results_dir.rglob("seed_*.csv")):
        if "selections" in seed_csv.name:
            continue
        d = pd.read_csv(seed_csv)
        d["dataset"] = seed_csv.relative_to(results_dir).parts[0]
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError(f"No seed CSV files found under {results_dir}")
    df = pd.concat(dfs, ignore_index=True)
    if "surrogate" not in df.columns:      # legacy CSVs predate the column
        df["surrogate"] = "rf"
    return df


def final_round(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last round of each (dataset,repr,acq,surrogate,seed) run."""
    keys = ["dataset", "representation", "acquisition", "surrogate", "seed"]
    last = df.groupby(keys)["round"].transform("max")
    return df[df["round"] == last]


def matched_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to rows where, within each (dataset, representation), the
    acquisition ran under BOTH gp and rf — so the GP/RF means average over the
    same acquisition set. Returns the final-round rows for those cells.
    """
    fin = final_round(df)
    kept = []
    for (ds, rep), g in fin.groupby(["dataset", "representation"]):
        surr_by_acq = g.groupby("acquisition")["surrogate"].agg(set)
        shared = [a for a, s in surr_by_acq.items() if {"gp", "rf"} <= s]
        if shared:
            kept.append(g[g["acquisition"].isin(shared)])
    if not kept:
        raise SystemExit("No (dataset, representation) cells have both GP and RF.")
    return pd.concat(kept, ignore_index=True)


def summarize(matched: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean ± sem of *metric* per (dataset, representation, surrogate)."""
    g = (matched.groupby(["dataset", "representation", "surrogate"])[metric]
         .agg(["mean", "sem"]).reset_index())
    return g


def plot_metric(matched: pd.DataFrame, metric: str, output_dir: Path) -> Path:
    """Grouped GP/RF bars, representations on x, one facet per dataset."""
    stats = summarize(matched, metric)
    datasets = sorted(stats["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 4.2 * n), 4.4), squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        sub = stats[stats["dataset"] == ds]
        reprs = [r for r in _REPR_ORDER if r in sub["representation"].unique()]
        x = np.arange(len(reprs))
        width = 0.38
        for i, surr in enumerate(["rf", "gp"]):
            s = sub[sub["surrogate"] == surr].set_index("representation")
            means = [float(s.loc[r, "mean"]) if r in s.index else np.nan for r in reprs]
            sems = [float(s.loc[r, "sem"]) if r in s.index else 0.0 for r in reprs]
            ax.bar(x + (i - 0.5) * width, means, width,
                   yerr=sems, capsize=3, color=_SURR_COLORS[surr],
                   label=_SURR_LABELS[surr])
        ax.set_xticks(x)
        ax.set_xticklabels([_REPR_LABELS.get(r, r) for r in reprs],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Representation")
    axes[0].set_ylabel(f"Final-round {metric}\n(mean ± sem, matched acqs)")
    axes[0].legend(title="Surrogate", frameon=False, loc="best")
    fig.suptitle(f"GP vs RF — {metric}", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"gp_vs_rf_{metric}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def print_delta_table(matched: pd.DataFrame, metric: str) -> None:
    """Print GP, RF, and Δ(gp−rf) per (dataset, representation)."""
    piv = (matched.groupby(["dataset", "representation", "surrogate"])[metric]
           .mean().reset_index()
           .pivot(index=["dataset", "representation"], columns="surrogate",
                  values=metric))
    if "gp" in piv and "rf" in piv:
        piv["delta(gp-rf)"] = piv["gp"] - piv["rf"]
    print(f"\n===== {metric}  (final round, matched acqs, mean over seeds) =====")
    print(piv.round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description="Paired GP-vs-RF comparison figures.")
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--output_dir", type=Path, default=Path("docs/figures/gp_vs_rf"))
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict to these datasets (default: all present).")
    p.add_argument("--metrics", nargs="*",
                   default=["topk10_recall", "simple_regret", "pool_spearman"],
                   help="Metrics to compare.")
    args = p.parse_args()

    df = load_results(args.results_dir)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
    # keep only datasets that actually have a GP run
    gp_datasets = set(df.loc[df["surrogate"] == "gp", "dataset"].unique())
    df = df[df["dataset"].isin(gp_datasets)]
    if df.empty:
        raise SystemExit("No GP results found for the requested datasets.")

    matched = matched_final(df)
    print(f"Matched cells: {matched['dataset'].nunique()} datasets, "
          f"acqs {sorted(matched['acquisition'].unique())}")

    for metric in args.metrics:
        if metric not in matched.columns:
            print(f"Skipping {metric} — column not found.")
            continue
        print_delta_table(matched, metric)
        fname = plot_metric(matched, metric, args.output_dir)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
