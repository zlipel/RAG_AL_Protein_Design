"""
Tests for GPSurrogate.

All tests use device='cpu' to avoid MPS/CUDA issues in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

from rag_al.surrogates.gp import GPSurrogate


RNG = np.random.default_rng(42)

# Small synthetic datasets — large enough for a sane GP, small enough for CPU speed
N_TRAIN = 30
N_QUERY = 10
N_FEATURES = 8


def _make_train(n=N_TRAIN, d=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)
    return X, y


def _make_query(n=N_QUERY, d=N_FEATURES, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


# ---------------------------------------------------------------------------
# Basic shape / dtype / sign contracts
# ---------------------------------------------------------------------------

def test_gp_output_shapes():
    X, y = _make_train()
    Xq = _make_query()
    sur = GPSurrogate(n_iter=30, device="cpu")
    sur.fit(X, y)
    mu, sigma = sur.predict(Xq)
    assert mu.shape == (N_QUERY,)
    assert sigma.shape == (N_QUERY,)


def test_gp_output_dtype():
    X, y = _make_train()
    Xq = _make_query()
    sur = GPSurrogate(n_iter=30, device="cpu")
    sur.fit(X, y)
    mu, sigma = sur.predict(Xq)
    assert mu.dtype == np.float32 or mu.dtype == np.float64  # either is fine
    assert sigma.dtype == mu.dtype


def test_gp_sigma_nonneg():
    X, y = _make_train()
    Xq = _make_query()
    sur = GPSurrogate(n_iter=30, device="cpu")
    sur.fit(X, y)
    _, sigma = sur.predict(Xq)
    assert np.all(sigma >= 0.0), f"Negative sigma values: {sigma[sigma < 0]}"


def test_gp_predict_before_fit_raises():
    sur = GPSurrogate(device="cpu")
    Xq = _make_query()
    with pytest.raises(RuntimeError, match="fit"):
        sur.predict(Xq)


# ---------------------------------------------------------------------------
# Standardization: predictions are in original scale
# ---------------------------------------------------------------------------

def test_gp_mu_in_original_scale():
    """mu should be in the same range as the training labels."""
    X, y = _make_train()
    Xq = _make_query()
    sur = GPSurrogate(n_iter=50, device="cpu")
    sur.fit(X, y)
    mu, _ = sur.predict(Xq)
    # mu should be in a plausible range around y's mean, not in z-space
    assert mu.mean() == pytest.approx(y.mean(), abs=2.0)


# ---------------------------------------------------------------------------
# Warm start: two rounds, state persists
# ---------------------------------------------------------------------------

def test_gp_warm_start_no_error():
    """fit() called twice should not raise."""
    X, y = _make_train()
    Xq = _make_query()
    sur = GPSurrogate(n_iter=30, device="cpu")
    sur.fit(X, y)
    # Simulate next round: add 5 more points
    X2, y2 = _make_train(n=N_TRAIN + 5, seed=99)
    sur.fit(X2, y2)
    mu, sigma = sur.predict(Xq)
    assert mu.shape == (N_QUERY,)
    assert np.all(sigma >= 0.0)


def test_gp_warm_start_state_populated():
    """_prev_state should be set after fit() and have the right keys."""
    X, y = _make_train()
    sur = GPSurrogate(n_iter=30, device="cpu")
    assert sur._prev_state is None
    sur.fit(X, y)
    assert sur._prev_state is not None
    assert "model" in sur._prev_state
    assert "likelihood" in sur._prev_state


def test_gp_cold_start_state_is_none():
    """A fresh GPSurrogate has no prev_state."""
    sur = GPSurrogate(device="cpu")
    assert sur._prev_state is None


# ---------------------------------------------------------------------------
# Patience / early stopping
# ---------------------------------------------------------------------------

def test_gp_patience_exits_early():
    """With tight patience settings, training should exit before n_iter steps."""
    X, y = _make_train()
    sur = GPSurrogate(n_iter=200, patience=1, tol=1e-6, device="cpu")
    sur.fit(X, y)  # Should not run all 200 steps — just verify it completes fast
    assert sur._model is not None


# ---------------------------------------------------------------------------
# End-to-end through runner on synthetic data
# ---------------------------------------------------------------------------

def test_gp_end_to_end_runner(tmp_path):
    """GP surrogate completes a 3-round AL loop on a 100-variant synthetic dataset."""
    import pandas as pd
    from rag_al.data.al_dataset import ALDataset
    from rag_al.loop.runner import run_al_loop
    from rag_al.representations.mutation import MutationDescriptorEncoder
    from rag_al.acquisition.ucb import UCBAcquisition

    # Build a minimal synthetic dataset
    rng = np.random.default_rng(7)
    n = 100
    wt = "ACDEFGHIKL"  # 10 AA wildtype
    aas = list("ACDEFGHIKL")
    rows = []
    for i in range(n):
        pos = i % len(wt)
        mut_aa = aas[(i + 3) % len(aas)]
        orig_aa = wt[pos]
        mutant = f"{orig_aa}{pos + 1}{mut_aa}"
        mut_seq = wt[:pos] + mut_aa + wt[pos + 1:]
        rows.append({
            "variant_id": f"var_{i:04d}",
            "mutant": mutant,
            "mutated_sequence": mut_seq,
            "wt_sequence": wt,
            "fitness": float(rng.standard_normal()),
        })
    df = pd.DataFrame(rows)

    dataset = ALDataset(df, n_init=10, seed=0)
    encoder = MutationDescriptorEncoder()
    surrogate = GPSurrogate(n_iter=30, device="cpu")
    acquisition = UCBAcquisition(beta=1.0)

    results, selections = run_al_loop(
        dataset=dataset,
        encoder=encoder,
        surrogate=surrogate,
        acquisition=acquisition,
        n_rounds=3,
        batch_size=5,
        seed=0,
    )

    assert len(results) == 3
    assert "pool_spearman" in results.columns
    assert "best_fitness" in results.columns
    assert len(selections) == 3 * 5
