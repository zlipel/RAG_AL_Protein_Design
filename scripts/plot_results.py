#!/usr/bin/env python3
"""
plot_results.py — Generate learning-curve figures from benchmark results.

Reads all seed CSVs from results/<dataset>/<tag>/seed_*.csv,
aggregates mean ± std across seeds, and plots:
  - best_fitness vs. AL round (one panel per acquisition function)
  - simple_regret vs. AL round
  - topk10_recall vs. AL round

Usage
-----
python scripts/plot_results.py --dataset BLAT_ECOLX --output_dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ------------------------------------------------------------------
# Plotting style
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

_REPR_COLORS = {
    "mutation":       "#4E79A7",
    "physicochemical":"#F28E2B",
    "plm_mean":       "#E15759",
    "plm_delta":      "#76B7B2",
    "plm_site":       "#B07AA1",
    "plm_physico":    "#EDC948",
    "plm_concat":     "#9C755F",
    "plm_retrieval":  "#59A14F",
}

_REPR_LABELS = {
    "mutation":       "Mutation descriptors",
    "physicochemical":"Physicochemical",
    "plm_mean":       "PLM mean-pool",
    "plm_delta":      "PLM delta",
    "plm_site":       "PLM site",
    "plm_physico":    "PLM + physico (per-res)",
    "plm_concat":     "PLM ⊕ physico (concat)",
    "plm_retrieval":  "PLM + retrieval",
}

_ACQ_LABELS = {
    "random":       "Random",
    "greedy":       "Greedy",
    "ucb":          "UCB",
    "diversity_ucb":"Diversity UCB",
    "retrieval_ucb":"Retrieval UCB",
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_results(results_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Concatenate all seed CSVs found under results_dir/<dataset>/**/seed_*.csv.
    """
    dataset_dir = results_dir / dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No results found at {dataset_dir}")

    dfs = []
    for seed_csv in sorted(dataset_dir.rglob("seed_*.csv")):
        df = pd.read_csv(seed_csv)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No seed CSV files found under {dataset_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(
        f"Loaded {len(dfs)} seed files, "
        f"{len(combined)} total rows, "
        f"for dataset '{dataset}'."
    )
    return combined


def aggregate(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute mean ± std of *metric* across seeds, grouped by
    (representation, acquisition, round).
    """
    return (
        df.groupby(["representation", "acquisition", "round"])[metric]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    )


# ------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------

def plot_metric_by_acquisition(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    acquisitions: list[str],
    output_dir: Path,
    dataset: str,
    suffix: str = "",
) -> None:
    """
    One subplot per acquisition function; representations are different lines.
    """
    agg = aggregate(df, metric)
    n_acq = len(acquisitions)
    fig, axes = plt.subplots(
        1, n_acq, figsize=(4.5 * n_acq, 4), sharey=True, sharex=True
    )
    if n_acq == 1:
        axes = [axes]

    reprs = sorted(df["representation"].unique())

    for ax, acq in zip(axes, acquisitions):
        sub = agg[agg["acquisition"] == acq]
        for repr_name in reprs:
            r = sub[sub["representation"] == repr_name]
            if r.empty:
                continue
            color = _REPR_COLORS.get(repr_name, "grey")
            label = _REPR_LABELS.get(repr_name, repr_name)
            ax.plot(
                r["round"], r[f"{metric}_mean"],
                color=color, label=label, linewidth=2, marker="o", markersize=4,
            )
            std = r[f"{metric}_std"].fillna(0)
            ax.fill_between(
                r["round"],
                r[f"{metric}_mean"] - std,
                r[f"{metric}_mean"] + std,
                color=color, alpha=0.15,
            )
        ax.set_title(_ACQ_LABELS.get(acq, acq), fontsize=11)
        ax.set_xlabel("AL Round")
        ax.set_xticks(sorted(df["round"].unique()))

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(len(reprs), 5),
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
        )
    fig.suptitle(f"{dataset} — {ylabel}", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = output_dir / f"{dataset}_{metric}{suffix}_by_acq.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


def plot_metric_by_representation(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    representations: list[str],
    output_dir: Path,
    dataset: str,
    suffix: str = "",
) -> None:
    """
    One subplot per representation; acquisition functions are different lines.
    """
    agg = aggregate(df, metric)
    n_repr = len(representations)
    fig, axes = plt.subplots(
        1, n_repr, figsize=(4.5 * n_repr, 4), sharey=True, sharex=True
    )
    if n_repr == 1:
        axes = [axes]

    acqs = sorted(df["acquisition"].unique())
    palette = sns.color_palette("tab10", n_colors=len(acqs))
    acq_colors = dict(zip(acqs, palette))

    for ax, repr_name in zip(axes, representations):
        sub = agg[agg["representation"] == repr_name]
        for acq in acqs:
            r = sub[sub["acquisition"] == acq]
            if r.empty:
                continue
            color = acq_colors[acq]
            label = _ACQ_LABELS.get(acq, acq)
            ax.plot(
                r["round"], r[f"{metric}_mean"],
                color=color, label=label, linewidth=2, marker="o", markersize=4,
            )
            std = r[f"{metric}_std"].fillna(0)
            ax.fill_between(
                r["round"],
                r[f"{metric}_mean"] - std,
                r[f"{metric}_mean"] + std,
                color=color, alpha=0.15,
            )
        ax.set_title(_REPR_LABELS.get(repr_name, repr_name), fontsize=10)
        ax.set_xlabel("AL Round")
        ax.set_xticks(sorted(df["round"].unique()))

    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(len(acqs), 6),
            bbox_to_anchor=(0.5, -0.12),
            frameon=False,
        )
    fig.suptitle(f"{dataset} — {ylabel}", y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = output_dir / f"{dataset}_{metric}{suffix}_by_repr.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RAG-AL benchmark learning curves."
    )
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--surrogate", type=str, default="rf",
        help="Surrogate to plot: 'rf' (default) or 'gp'. Filters the CSVs so "
             "RF and GP curves for the same cell are never blended.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.results_dir, args.dataset)
    if "surrogate" not in df.columns:        # legacy CSVs predate the column
        df["surrogate"] = "rf"
    df = df[df["surrogate"] == args.surrogate]
    if df.empty:
        raise SystemExit(
            f"No '{args.surrogate}' results for {args.dataset}."
        )
    # keep RF (default) filenames stable; tag GP/other explicitly
    suffix = "" if args.surrogate == "rf" else f"_{args.surrogate}"

    acquisitions = sorted(df["acquisition"].unique())
    representations = sorted(df["representation"].unique())

    metrics = [
        ("best_fitness",       "Best observed fitness"),
        ("simple_regret",      "Simple regret"),
        ("topk10_recall",      "Top-10 recall"),
        ("topk50_recall",      "Top-50 recall"),
        ("batch_mean_fitness", "Batch mean fitness"),
        ("pool_spearman",      "Pool Spearman ρ"),
    ]

    for metric, ylabel in metrics:
        if metric not in df.columns:
            print(f"Skipping {metric} — column not found.")
            continue
        plot_metric_by_acquisition(
            df, metric, ylabel, acquisitions, args.output_dir, args.dataset, suffix
        )
        plot_metric_by_representation(
            df, metric, ylabel, representations, args.output_dir, args.dataset, suffix
        )

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
