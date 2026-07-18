#!/usr/bin/env python3
"""
plot_aggregate.py — Cross-dataset aggregation figures for the RAG-AL benchmark.

Reads results/<dataset>/<tag>/seed_*.csv for every dataset found in results/,
then produces:

  1. Heatmap of mean final-round topk10_recall per (representation, acquisition)
     pair, averaged across all datasets and seeds.

  2. Bar chart ranking representations by mean topk10_recall, split into
     difficulty quartiles derived from per-dataset fitness_std.

Usage
-----
python scripts/plot_aggregate.py --results_dir results --data_dir data/curated \
    --output_dir figures/aggregate --metric topk10_recall
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

_REPR_ORDER = [
    "mutation", "physicochemical",
    "plm_mean", "plm_delta", "plm_site",
    "plm_physico", "plm_concat", "plm_retrieval",
]
_ACQ_ORDER  = ["random", "greedy", "ucb", "diversity_ucb", "retrieval_ucb"]

_REPR_LABELS = {
    "mutation":        "Mutation",
    "physicochemical": "Physico-\nchem",
    "plm_mean":        "PLM\nmean",
    "plm_delta":       "PLM\ndelta",
    "plm_site":        "PLM\nsite",
    "plm_physico":     "PLM+\nphysico",
    "plm_concat":      "PLM\nconcat",
    "plm_retrieval":   "PLM +\nretrieval",
}
_ACQ_LABELS = {
    "random":        "Random",
    "greedy":        "Greedy",
    "ucb":           "UCB",
    "diversity_ucb": "Div-UCB",
    "retrieval_ucb": "Ret-UCB",
}


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load every seed_*.csv (excluding selections) across all datasets."""
    dfs = []
    for seed_csv in sorted(results_dir.rglob("seed_*.csv")):
        if "selections" in seed_csv.name:
            continue
        df = pd.read_csv(seed_csv)
        # dataset name is the immediate subdirectory of results_dir
        df["dataset"] = seed_csv.relative_to(results_dir).parts[0]
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No seed CSV files found under {results_dir}")

    combined = pd.concat(dfs, ignore_index=True)
    print(
        f"Loaded {len(dfs)} seed files across "
        f"{combined['dataset'].nunique()} datasets, "
        f"{len(combined)} total rows."
    )
    return combined


def compute_difficulty(data_dir: Path) -> pd.DataFrame:
    """
    Compute per-dataset difficulty statistics from curated CSVs.
    Returns a DataFrame with columns: dataset, n_variants, fitness_std,
    fitness_skew, wt_len.
    """
    rows = []
    for csv in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        if "fitness" not in df.columns:
            continue
        y = df["fitness"]
        wt_len = len(df["wt_sequence"].iloc[0]) if "wt_sequence" in df.columns else np.nan
        rows.append({
            "dataset":      csv.stem,
            "n_variants":   len(df),
            "fitness_std":  float(y.std()),
            "fitness_skew": float(stats.skew(y)),
            "wt_len":       int(wt_len),
        })
    return pd.DataFrame(rows)


def final_round_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    For each (dataset, representation, acquisition, surrogate, seed), extract the
    final-round value of *metric*. Grouping includes ``surrogate`` so RF and GP
    runs of the same cell are never conflated.
    """
    keys = ["dataset", "representation", "acquisition", "surrogate", "seed"]
    last_round = (
        df.groupby(keys)["round"]
        .max()
        .reset_index()
        .rename(columns={"round": "last_round"})
    )
    merged = df.merge(last_round, on=keys)
    return merged[merged["round"] == merged["last_round"]].copy()


# ------------------------------------------------------------------
# Figure 1 — Heatmap (repr × acq), averaged across all datasets + seeds
# ------------------------------------------------------------------

def plot_heatmap(
    final_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    suffix: str = "",
) -> None:
    reprs = [r for r in _REPR_ORDER if r in final_df["representation"].unique()]
    acqs  = [a for a in _ACQ_ORDER  if a in final_df["acquisition"].unique()]

    cell_means = (
        final_df
        .groupby(["representation", "acquisition"])[metric]
        .mean()
        .reset_index()
    )
    mat = cell_means.pivot(index="representation", columns="acquisition", values=metric)
    mat = mat.reindex(index=reprs, columns=acqs)

    fig, ax = plt.subplots(figsize=(len(acqs) * 1.5 + 1.5, len(reprs) * 1.0 + 1.5))
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": f"Mean {metric} (final round)"},
        xticklabels=[_ACQ_LABELS.get(a, a) for a in acqs],
        yticklabels=[_REPR_LABELS.get(r, r).replace("\n", " ") for r in reprs],
    )
    ax.set_title(
        f"Mean final-round {metric}\n(averaged across all datasets & seeds)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Acquisition function")
    ax.set_ylabel("Representation")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()

    fname = output_dir / f"aggregate_{metric}{suffix}_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 2 — Representation ranking by difficulty quartile
# ------------------------------------------------------------------

def plot_repr_by_difficulty(
    final_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    suffix: str = "",
) -> None:
    merged = final_df.merge(
        difficulty_df[["dataset", "fitness_std"]],
        on="dataset",
        how="left",
    )
    if merged["fitness_std"].isna().all():
        print("No difficulty data — skipping difficulty plot.")
        return

    n_datasets = merged["dataset"].nunique()
    if n_datasets < 2:
        print("Only one dataset present — skipping difficulty split plot.")
        return

    # Assign difficulty halves based on fitness_std
    merged["difficulty_q"] = pd.qcut(
        merged["fitness_std"],
        q=min(2, n_datasets),
        labels=["Narrow\n(low std)", "Wide\n(high std)"],
        duplicates="drop",
    )

    reprs = [r for r in _REPR_ORDER if r in merged["representation"].unique()]
    groups = merged["difficulty_q"].cat.categories.tolist()

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 4.5), sharey=True)
    if n_groups == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", n_colors=len(reprs))

    for ax, grp in zip(axes, groups):
        sub = merged[merged["difficulty_q"] == grp]
        means = (
            sub.groupby("representation")[metric]
            .mean()
            .reindex(reprs)
            .dropna()
        )
        sems = (
            sub.groupby("representation")[metric]
            .sem()
            .reindex(means.index)
            .fillna(0)
        )
        # Sort by mean metric descending within each panel
        order = means.sort_values(ascending=False).index.tolist()
        colors = [palette[reprs.index(r)] for r in order]

        ax.barh(
            range(len(order)),
            means[order].values,
            xerr=sems[order].values,
            color=colors,
            height=0.6,
            capsize=3,
        )
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(
            [_REPR_LABELS.get(r, r).replace("\n", " ") for r in order],
            fontsize=9,
        )
        ax.set_xlabel(f"Mean {metric}")
        ax.set_title(f"Fitness std: {grp}", fontsize=11)
        ax.invert_yaxis()

    axes[0].set_ylabel("Representation")
    fig.suptitle(
        f"Representation ranking by landscape difficulty\n(metric: {metric})",
        y=1.02, fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    fname = output_dir / f"aggregate_{metric}{suffix}_repr_by_difficulty.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
# Figure 3 — Per-dataset final-round metric, grouped bar by representation
# ------------------------------------------------------------------

def plot_per_dataset_bar(
    final_df: pd.DataFrame,
    difficulty_df: pd.DataFrame,
    metric: str,
    output_dir: Path,
    suffix: str = "",
) -> None:
    """
    One group of bars per dataset (sorted by fitness_std), each bar a
    representation. Averaged across seeds and acquisitions.
    """
    cell = (
        final_df
        .groupby(["dataset", "representation"])[metric]
        .mean()
        .reset_index()
    )

    # Sort datasets by fitness_std if available
    if not difficulty_df.empty:
        cell = cell.merge(difficulty_df[["dataset", "fitness_std"]], on="dataset", how="left")
        dataset_order = (
            cell.groupby("dataset")["fitness_std"]
            .first()
            .sort_values()
            .index.tolist()
        )
    else:
        dataset_order = sorted(cell["dataset"].unique())

    reprs = [r for r in _REPR_ORDER if r in cell["representation"].unique()]
    n_datasets = len(dataset_order)
    x = np.arange(n_datasets)
    width = 0.8 / len(reprs)
    palette = sns.color_palette("Set1", n_colors=len(reprs))

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 1.6), 4.5))
    for i, repr_name in enumerate(reprs):
        sub = cell[cell["representation"] == repr_name].set_index("dataset")
        vals = [float(sub.loc[d, metric]) if d in sub.index else np.nan for d in dataset_order]
        offset = (i - len(reprs) / 2 + 0.5) * width
        ax.bar(
            x + offset, vals,
            width=width,
            color=palette[i],
            label=_REPR_LABELS.get(repr_name, repr_name).replace("\n", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [d.replace("_", "\n") for d in dataset_order],
        fontsize=8,
    )
    ax.set_ylabel(f"Mean final-round {metric}\n(avg over seeds & acquisitions)")
    ax.set_title(
        f"{metric} per dataset (sorted by fitness_std)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()

    fname = output_dir / f"aggregate_{metric}{suffix}_per_dataset.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


# ------------------------------------------------------------------
# Text summaries (inspect the numbers without generating figures)
# ------------------------------------------------------------------

def print_summary_tables(final_df: pd.DataFrame, metric: str) -> None:
    """
    Print the aggregated final-round *metric* as plain tables:
      1. representation x acquisition, averaged across datasets & seeds
      2. dataset x representation, averaged across acquisitions & seeds
      3. per-dataset best (representation, acquisition) cell
    """
    reprs = [r for r in _REPR_ORDER if r in final_df["representation"].unique()]
    acqs  = [a for a in _ACQ_ORDER  if a in final_df["acquisition"].unique()]

    print("\n" + "=" * 78)
    print(f"[{metric}]  representation x acquisition  (mean over datasets & seeds)")
    print("=" * 78)
    mat = (final_df.groupby(["representation", "acquisition"])[metric]
                   .mean().reset_index()
                   .pivot(index="representation", columns="acquisition", values=metric)
                   .reindex(index=reprs, columns=acqs))
    mat["MEAN"] = mat.mean(axis=1)
    print(mat.round(3).to_string())

    print("\n" + "=" * 78)
    print(f"[{metric}]  dataset x representation  (mean over acquisitions & seeds)")
    print("=" * 78)
    dr = (final_df.groupby(["dataset", "representation"])[metric]
                  .mean().reset_index()
                  .pivot(index="dataset", columns="representation", values=metric)
                  .reindex(columns=reprs))
    print(dr.round(3).to_string())

    print("\n" + "=" * 78)
    print(f"[{metric}]  best (representation x acquisition) cell per dataset")
    print("=" * 78)
    cell = (final_df.groupby(["dataset", "representation", "acquisition"])[metric]
                    .mean().reset_index())
    for ds in sorted(cell["dataset"].unique()):
        sub = cell[cell.dataset == ds].sort_values(metric, ascending=False)
        best = sub.iloc[0]
        print(f"  {ds:28s} {best.representation:15s} x {best.acquisition:14s} "
              f"= {best[metric]:.3f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-dataset aggregation plots for RAG-AL benchmark."
    )
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--data_dir",    type=Path, default=Path("data/curated"))
    parser.add_argument("--output_dir",  type=Path, default=Path("figures/aggregate"))
    parser.add_argument(
        "--metric", type=str, default="topk10_recall",
        help="Metric column to aggregate (default: topk10_recall)",
    )
    parser.add_argument(
        "--surrogate", type=str, default="rf",
        help="Surrogate to analyze: 'rf', 'gp', or 'all' to loop over each "
             "surrogate present (default: rf).",
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Only print the numeric summary tables; skip figure generation.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Restrict analysis to these datasets. Use to compare representations "
             "on a common set of datasets when the grid is ragged (e.g. reps that "
             "only ran on a subset). Default: all datasets present.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_results(args.results_dir)
    if "surrogate" not in df.columns:      # legacy CSVs predate the column
        df["surrogate"] = "rf"
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]
        if df.empty:
            raise SystemExit(f"No results for datasets {args.datasets}.")
        print(f"Restricted to {df['dataset'].nunique()} datasets: "
              f"{sorted(df['dataset'].unique())}")

    present = sorted(df["surrogate"].unique())
    if args.surrogate == "all":
        surrogates = present
    elif args.surrogate in present:
        surrogates = [args.surrogate]
    else:
        raise SystemExit(
            f"No results for surrogate '{args.surrogate}'. Present: {present}"
        )

    difficulty_df = pd.DataFrame()
    if args.data_dir.exists():
        difficulty_df = compute_difficulty(args.data_dir)
        print("\nDataset difficulty statistics:")
        print(difficulty_df.to_string(index=False))
        print()

    for surrogate in surrogates:
        sdf = df[df["surrogate"] == surrogate]
        final_df = final_round_metric(sdf, args.metric)
        # suffix keeps RF (default) filenames stable; tags GP/other explicitly
        suffix = "" if surrogate == "rf" else f"_{surrogate}"

        print(f"\n########## surrogate = {surrogate} "
              f"({sdf['dataset'].nunique()} datasets) ##########")
        print_summary_tables(final_df, args.metric)

        if args.no_plots:
            continue
        plot_heatmap(final_df, args.metric, args.output_dir, suffix)
        plot_repr_by_difficulty(final_df, difficulty_df, args.metric, args.output_dir, suffix)
        plot_per_dataset_bar(final_df, difficulty_df, args.metric, args.output_dir, suffix)

    if not args.no_plots:
        print(f"\nAll aggregate figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
