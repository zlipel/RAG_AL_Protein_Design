"""
Tests for Bug #1 fix (correct batch_y after reveal) and Gap #1 (selection logging).

Uses a small synthetic dataset so selected global indices, local indices, and
revealed fitness values can be verified by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rag_al.data.al_dataset import ALDataset
from rag_al.loop.runner import run_al_loop


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_WT = "ACDEFGHIKLM"   # 11-AA wild-type sequence
_N_TOTAL = 20
_N_INIT = 5
_BATCH_SIZE = 3
_N_ROUNDS = 2
_SEED = 42


def _make_synthetic_df(n: int = _N_TOTAL, rng_seed: int = 0) -> pd.DataFrame:
    """Build a tiny synthetic dataset with deterministic fitness values."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    for i in range(n):
        # Each variant differs from WT at position 0 only (simple substitution)
        mut_aa = aa[i % len(aa)]
        mutated = mut_aa + _WT[1:]
        rows.append({
            "variant_id": f"var_{i:04d}",
            "mutant": f"{_WT[0]}1{mut_aa}",
            "mutated_sequence": mutated,
            "wt_sequence": _WT,
            "fitness": float(rng.uniform(0, 1)),
        })
    return pd.DataFrame(rows)


class _IdentityEncoder:
    """Trivial encoder: uses position-0 AA one-hot as 20-dim feature."""
    _AA = list("ACDEFGHIKLMNPQRSTVWY")

    def fit(self, df, y):
        pass

    def transform_labeled(self, df) -> np.ndarray:
        return self.transform(df)

    def transform(self, df) -> np.ndarray:
        out = np.zeros((len(df), 20), dtype=float)
        for i, seq in enumerate(df["mutated_sequence"]):
            aa_idx = self._AA.index(seq[0]) if seq[0] in self._AA else 0
            out[i, aa_idx] = 1.0
        return out


class _ConstantSurrogate:
    """Surrogate that predicts mu=fitness mean, sigma=0 (greedy will pick first)."""
    def fit(self, X, y):
        self._mean = float(y.mean())

    def predict(self, X):
        mu = np.full(len(X), self._mean)
        sigma = np.zeros(len(X))
        return mu, sigma


class _RandomAcquisition:
    """Acquisition that always selects the first batch_size pool variants."""
    def select_batch(self, mu, sigma, batch_size, *, pool_X, labeled_X, labeled_y, rng):
        return np.arange(min(batch_size, len(mu)), dtype=int)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_batch_y_matches_global_selected():
    """
    Regression test for Bug #1.

    After reveal(), batch_y must contain the fitness scores for the
    SELECTED global indices — not a tail-slice of labeled_y.
    """
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    # Manually track what fitness values the selected variants should have
    true_fitness = df["fitness"].to_numpy()

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    # For each row in selections, the fitness must match the true fitness at
    # that global_index (the ground truth, which we can check here in tests)
    for _, row in selections.iterrows():
        expected = true_fitness[int(row["global_index"])]
        assert abs(row["fitness"] - expected) < 1e-10, (
            f"batch_y mismatch at global_index={row['global_index']}: "
            f"got {row['fitness']:.6f}, expected {expected:.6f}"
        )


def test_selections_shape():
    """Selections DataFrame has one row per (round × batch variant)."""
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    assert len(selections) == _N_ROUNDS * _BATCH_SIZE
    assert list(selections.columns) == ["round", "global_index", "variant_id", "fitness"]


def test_selections_global_indices_not_in_initial_labeled():
    """
    All global_indices in round 0 selections must NOT have been in
    the initial labeled set — i.e., they were pool variants before round 1.
    """
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    # Record initial labeled indices before running the loop
    initial_labeled = set(dataset.labeled_indices.tolist())

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    round0_selected = set(
        selections.loc[selections["round"] == 0, "global_index"].tolist()
    )
    overlap = initial_labeled & round0_selected
    assert len(overlap) == 0, (
        f"Round-0 selections overlap with initial labeled set: {overlap}"
    )


def test_selections_global_indices_unique_across_rounds():
    """No variant should be selected more than once across all rounds."""
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    all_selected = selections["global_index"].tolist()
    assert len(all_selected) == len(set(all_selected)), (
        "Duplicate global_index values found in selections — a variant was selected twice."
    )


def test_variant_ids_consistent_with_global_indices():
    """variant_id in selections must match df.iloc[global_index]['variant_id']."""
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    for _, row in selections.iterrows():
        expected_vid = df.iloc[int(row["global_index"])]["variant_id"]
        assert row["variant_id"] == expected_vid, (
            f"variant_id mismatch at global_index={row['global_index']}: "
            f"got {row['variant_id']!r}, expected {expected_vid!r}"
        )


def test_results_shape():
    """Results DataFrame has one row per round with expected columns."""
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    assert len(results) == _N_ROUNDS
    for col in ("round", "n_labeled", "best_fitness", "simple_regret",
                "topk10_recall", "topk50_recall", "batch_mean_fitness",
                "pool_spearman"):
        assert col in results.columns, f"Missing column in results: {col}"


def test_pool_spearman_in_bounds():
    """pool_spearman must be in [-1, 1] or NaN each round."""
    import math
    df = _make_synthetic_df()
    dataset = ALDataset(df, n_init=_N_INIT, seed=_SEED)

    results, _ = run_al_loop(
        dataset=dataset,
        encoder=_IdentityEncoder(),
        surrogate=_ConstantSurrogate(),
        acquisition=_RandomAcquisition(),
        n_rounds=_N_ROUNDS,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
    )

    for val in results["pool_spearman"]:
        assert math.isnan(val) or (-1.0 <= val <= 1.0), (
            f"pool_spearman out of range: {val}"
        )
