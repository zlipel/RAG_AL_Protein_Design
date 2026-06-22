"""
Tests for Bug #3 fix: self-neighbor exclusion in RetrievalAugmentedEncoder.

The key invariant: when transform_labeled() is called on the same points
used to build the kNN index, each point's nearest neighbor must be a
*different* labeled variant — not itself (distance 0).

Uses mock ESM embeddings so tests run without GPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from rag_al.representations.retrieval import RetrievalAugmentedEncoder


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_N_LABELED = 6
_N_POOL = 4
_D = 8          # embedding dimension
_K = 2          # n_neighbors


def _make_embeddings(n: int, d: int, scale: float = 1.0) -> np.ndarray:
    """Distinct embeddings: scaled identity rows (cycled if n > d)."""
    eye = np.eye(d) * scale
    rows = [eye[i % d] for i in range(n)]
    return np.vstack(rows).astype(float)


def _make_labeled_df(n: int) -> pd.DataFrame:
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    rows = [
        {
            "variant_id": f"var_{i:04d}",
            "mutant": f"A1{aa[i % len(aa)]}",
            "mutated_sequence": aa[i % len(aa)] + "ACDEFGHIKL",
            "wt_sequence": "AACDEFGHIKL",
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def _make_encoder_with_mock_esm(
    labeled_embs: np.ndarray,
    pool_embs: np.ndarray,
    n_neighbors: int = _K,
) -> RetrievalAugmentedEncoder:
    """
    Build a RetrievalAugmentedEncoder whose ESMEncoder is mocked.
    The mock returns labeled_embs when called on labeled_df and
    pool_embs when called on pool_df (distinguished by DataFrame length).
    """
    mock_esm = MagicMock()
    mock_esm.n_features = labeled_embs.shape[1]

    def side_effect(df):
        if len(df) == len(labeled_embs):
            return labeled_embs.copy()
        return pool_embs.copy()

    mock_esm.transform.side_effect = side_effect
    mock_esm.fit.return_value = None

    enc = RetrievalAugmentedEncoder(esm_encoder=mock_esm, n_neighbors=n_neighbors)
    return enc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transform_labeled_d_min_nonzero():
    """
    Regression test for Bug #3.

    After transform_labeled(), d_min (retrieval feature index 2, at column D+2)
    must be > 0 for every labeled point. Self-distance = 0, so any d_min == 0
    indicates self is still the nearest neighbor.
    """
    labeled_embs = _make_embeddings(_N_LABELED, _D, scale=10.0)
    pool_embs = _make_embeddings(_N_POOL, _D, scale=5.0)
    labeled_df = _make_labeled_df(_N_LABELED)
    y_labeled = np.arange(_N_LABELED, dtype=float)

    enc = _make_encoder_with_mock_esm(labeled_embs, pool_embs)
    enc.fit(labeled_df, y_labeled)

    X = enc.transform_labeled(labeled_df)
    d_min = X[:, _D + 2]   # retrieval feature: min distance (normalized)

    assert np.all(d_min > 0), (
        f"transform_labeled() returned d_min=0 for some labeled points: {d_min}"
    )


def test_transform_self_neighbor_present():
    """
    Documents that transform() (the pool path) DOES include self when called on
    the labeled set. This is intentional — pool variants never query themselves.
    This test confirms the uncorrected behavior is still present on the base path.
    """
    labeled_embs = _make_embeddings(_N_LABELED, _D, scale=10.0)
    pool_embs = _make_embeddings(_N_POOL, _D, scale=5.0)
    labeled_df = _make_labeled_df(_N_LABELED)
    y_labeled = np.arange(_N_LABELED, dtype=float)

    enc = _make_encoder_with_mock_esm(labeled_embs, pool_embs)
    enc.fit(labeled_df, y_labeled)

    X = enc.transform(labeled_df)
    d_min = X[:, _D + 2]

    # Self is included → at least one d_min should be ≈ 0
    assert np.any(d_min < 1e-10), (
        "Expected transform() to include self-neighbor (d_min≈0) for labeled set, "
        f"but got d_min={d_min}"
    )


def test_mean_y_not_self_referential():
    """
    After transform_labeled(), no row's mean_y (retrieval feature index 0)
    should equal that row's own fitness when k=1. With k=1 and self excluded,
    the single neighbor is a different labeled point — mean_y cannot equal self-y.
    """
    labeled_embs = _make_embeddings(_N_LABELED, _D, scale=10.0)
    pool_embs = _make_embeddings(_N_POOL, _D, scale=5.0)
    labeled_df = _make_labeled_df(_N_LABELED)
    y_labeled = np.arange(_N_LABELED, dtype=float) * 2.0 + 1.0  # distinct, non-zero

    enc = _make_encoder_with_mock_esm(labeled_embs, pool_embs, n_neighbors=1)
    enc.fit(labeled_df, y_labeled)

    X = enc.transform_labeled(labeled_df)
    mean_y = X[:, _D + 0]   # retrieval feature: mean neighbor fitness

    for i in range(_N_LABELED):
        assert abs(mean_y[i] - y_labeled[i]) > 1e-10, (
            f"Row {i}: mean_y={mean_y[i]:.4f} equals own fitness={y_labeled[i]:.4f} "
            "— self-neighbor is still included."
        )


def test_dist_scale_positive_after_fit():
    """
    _dist_scale must be meaningfully positive after fit().

    Without the fix, the self-distance column of zeros pulls the median to ~0
    when k=1 (only neighbor found is self → dists all 0 → median = 0).
    With the fix, the median is computed over true inter-point distances.
    """
    labeled_embs = _make_embeddings(_N_LABELED, _D, scale=10.0)
    pool_embs = _make_embeddings(_N_POOL, _D, scale=5.0)
    labeled_df = _make_labeled_df(_N_LABELED)
    y_labeled = np.ones(_N_LABELED)

    enc = _make_encoder_with_mock_esm(labeled_embs, pool_embs, n_neighbors=1)
    enc.fit(labeled_df, y_labeled)

    assert enc._dist_scale > 1e-7, (
        f"_dist_scale={enc._dist_scale} is near zero — self-distance is polluting "
        "the median normalization."
    )


def test_pool_transform_d_min_nonzero():
    """
    Pool variants are never in the kNN index, so transform() should always
    return d_min > 0 for pool queries — no self-inclusion is possible.
    This confirms the pool path is unaffected by the Bug #3 fix.
    """
    labeled_embs = _make_embeddings(_N_LABELED, _D, scale=10.0)
    pool_embs = _make_embeddings(_N_POOL, _D, scale=5.0)
    labeled_df = _make_labeled_df(_N_LABELED)
    pool_df = _make_labeled_df(_N_POOL)
    y_labeled = np.arange(_N_LABELED, dtype=float)

    enc = _make_encoder_with_mock_esm(labeled_embs, pool_embs)
    enc.fit(labeled_df, y_labeled)

    X_pool = enc.transform(pool_df)
    d_min = X_pool[:, _D + 2]

    assert np.all(d_min > 0), (
        f"Pool transform() returned d_min=0 — pool variant matched itself: {d_min}"
    )
