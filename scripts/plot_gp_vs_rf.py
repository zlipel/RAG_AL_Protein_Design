#!/usr/bin/env python3
"""
plot_gp_vs_rf.py — Paired surrogate comparison for the targeted GP grid.

Compares final-round metrics across surrogate variants, matched on the
acquisitions that EVERY compared variant ran (the GP grid omits
random/diversity/retrieval, so a naive mean over acquisitions would compare
GP's {greedy,ucb} against RF's five — this restricts to the shared set).

The surrogate CSV column is 'gp' for both the isotropic and ARD kernels; they
differ only by the result-dir tag (`_gp` vs `_gp_ard`), so the variant is
derived from the path. Supported variants: rf, gp (isotropic), gp_ard.

Produces, per metric, one grouped-bar figure (representations on x, one bar per
variant) faceted by dataset, plus a printed delta table.

Usage
-----
# RF vs isotropic GP (default: all variants present)
python scripts/plot_gp_vs_rf.py \
    --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012 \
    --metrics topk10_recall simple_regret pool_spearman \
    --output_dir docs/figures/gp_vs_rf

# ARD A/B: isotropic vs ARD GP (delta = gp_ard − gp)
python scripts/plot_gp_vs_rf.py --variants gp_ard gp \
    --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012
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
_SURR_COLORS = {"rf": "#4E79A7", "gp": "#E15759", "gp_ard": "#59A14F"}  # RF blue, GP red, ARD green
_SURR_LABELS = {"rf": "RF", "gp": "GP (iso)", "gp_ard": "GP (ARD)"}


def _variant_from_tag(tag: str, col_value: str) -> str:
    """Derive the surrogate variant from the run's directory tag.

    The CSV `surrogate` column is 'gp' for both isotropic and ARD kernels — they
    differ only by the path suffix (`_gp` vs `_gp_ard`). Reading the suffix keeps
    isotropic and ARD GP runs distinct. Safe while GP is the only kernel-variant
    surrogate; revisit (add a CSV column) if more variants appear.
    """
    if tag.endswith("_gp_ard"):
        return "gp_ard"
    if tag.endswith("_gp"):
        return "gp"
    return col_value


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load every seed_*.csv (excluding selections); label the surrogate variant
    from the directory tag so `_gp` (isotropic) and `_gp_ard` are distinct."""
    dfs = []
    for seed_csv in sorted(results_dir.rglob("seed_*.csv")):
        if "selections" in seed_csv.name:
            continue
        d = pd.read_csv(seed_csv)
        d["dataset"] = seed_csv.relative_to(results_dir).parts[0]
        col = d["surrogate"].iloc[0] if "surrogate" in d.columns and len(d) else "rf"
        d["surrogate"] = _variant_from_tag(seed_csv.parent.name, col)
        dfs.append(d)
    if not dfs:
        raise FileNotFoundError(f"No seed CSV files found under {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def final_round(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the last round of each (dataset,repr,acq,surrogate,seed) run."""
    keys = ["dataset", "representation", "acquisition", "surrogate", "seed"]
    last = df.groupby(keys)["round"].transform("max")
    return df[df["round"] == last]


def matched_final(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """
    Restrict to rows where, within each (dataset, representation), the
    acquisition ran under EVERY requested variant — so per-variant means average
    over the same acquisition set. Returns the final-round rows for those cells.
    """
    fin = final_round(df)
    want = set(variants)
    kept = []
    for (ds, rep), g in fin.groupby(["dataset", "representation"]):
        surr_by_acq = g.groupby("acquisition")["surrogate"].agg(set)
        shared = [a for a, s in surr_by_acq.items() if want <= s]
        if shared:
            kept.append(g[g["acquisition"].isin(shared)])
    if not kept:
        raise SystemExit(f"No (dataset, representation) cells ran all of {variants}.")
    return pd.concat(kept, ignore_index=True)


def summarize(matched: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean ± sem of *metric* per (dataset, representation, surrogate)."""
    g = (matched.groupby(["dataset", "representation", "surrogate"])[metric]
         .agg(["mean", "sem"]).reset_index())
    return g


def plot_metric(
    matched: pd.DataFrame, metric: str, output_dir: Path, variants: list[str]
) -> Path:
    """Grouped per-variant bars, representations on x, one facet per dataset."""
    stats = summarize(matched, metric)
    datasets = sorted(stats["dataset"].unique())
    n = len(datasets)
    nv = len(variants)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 4.2 * n), 4.4), squeeze=False)
    axes = axes[0]

    for ax, ds in zip(axes, datasets):
        sub = stats[stats["dataset"] == ds]
        reprs = [r for r in _REPR_ORDER if r in sub["representation"].unique()]
        x = np.arange(len(reprs))
        width = 0.8 / nv
        for i, surr in enumerate(variants):
            s = sub[sub["surrogate"] == surr].set_index("representation")
            means = [float(s.loc[r, "mean"]) if r in s.index else np.nan for r in reprs]
            sems = [float(s.loc[r, "sem"]) if r in s.index else 0.0 for r in reprs]
            ax.bar(x + (i - (nv - 1) / 2) * width, means, width,
                   yerr=sems, capsize=3, color=_SURR_COLORS.get(surr, "grey"),
                   label=_SURR_LABELS.get(surr, surr))
        ax.set_xticks(x)
        ax.set_xticklabels([_REPR_LABELS.get(r, r) for r in reprs],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(ds, fontsize=10)
        ax.set_xlabel("Representation")
    axes[0].set_ylabel(f"Final-round {metric}\n(mean ± sem, matched acqs)")
    axes[0].legend(title="Surrogate", frameon=False, loc="best")
    title = " vs ".join(_SURR_LABELS.get(v, v) for v in variants)
    fig.suptitle(f"{title} — {metric}", y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fname = output_dir / f"{'_vs_'.join(variants)}_{metric}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def print_delta_table(matched: pd.DataFrame, metric: str, variants: list[str]) -> None:
    """Print per-variant means (and, for a 2-variant compare, their delta)."""
    piv = (matched.groupby(["dataset", "representation", "surrogate"])[metric]
           .mean().reset_index()
           .pivot(index=["dataset", "representation"], columns="surrogate",
                  values=metric))
    piv = piv.reindex(columns=[v for v in variants if v in piv.columns])
    if len(variants) == 2 and all(v in piv.columns for v in variants):
        a, b = variants
        piv[f"delta({a}-{b})"] = piv[a] - piv[b]
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
    p.add_argument("--variants", nargs="*", default=None,
                   help="Surrogate variants to compare, e.g. 'gp gp_ard' (ARD A/B) "
                        "or 'rf gp'. Derived from the result dir tag. For a "
                        "2-variant compare the delta column is first−second. "
                        "Default: all variants present.")
    args = p.parse_args()

    df = load_results(args.results_dir)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)]

    present = sorted(df["surrogate"].unique())
    variants = args.variants or present
    missing = [v for v in variants if v not in present]
    if missing:
        raise SystemExit(f"Variants not present: {missing}. Present: {present}")
    if len(variants) < 2:
        raise SystemExit(f"Need ≥2 variants to compare; got {variants}.")

    matched = matched_final(df, variants)
    print(f"Comparing {variants} on {matched['dataset'].nunique()} datasets, "
          f"acqs {sorted(matched['acquisition'].unique())}")

    for metric in args.metrics:
        if metric not in matched.columns:
            print(f"Skipping {metric} — column not found.")
            continue
        print_delta_table(matched, metric, variants)
        fname = plot_metric(matched, metric, args.output_dir, variants)
        print(f"Saved: {fname}")


if __name__ == "__main__":
    main()
