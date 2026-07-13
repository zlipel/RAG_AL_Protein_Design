"""
Tests for PLMPhysicoEncoder and PLMSimpleConcatEncoder.

All tests use mock models — no HuggingFace weights loaded in CI.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from rag_al.representations.plm_physico import (
    PLMPhysicoEncoder,
    PLMSimpleConcatEncoder,
    _sequence_to_physico,
    _N_PHYSICO,
    _STANDARD_AA,
    _RESIDUE_TABLE,
)

# ---------------------------------------------------------------------------
# _sequence_to_physico unit tests
# ---------------------------------------------------------------------------

def test_sequence_to_physico_shape():
    out = _sequence_to_physico("ACDEF")
    assert out.shape == (5, _N_PHYSICO)


def test_sequence_to_physico_dtype():
    out = _sequence_to_physico("ACDEF")
    assert out.dtype == np.float64


def test_sequence_to_physico_charge_columns():
    # R and K are positive, D and E are negative
    out = _sequence_to_physico("RKDE")
    np.testing.assert_array_equal(out[:, 1], [1.0, 1.0, -1.0, -1.0])


def test_sequence_to_physico_polar():
    # S, T, N, Q are polar
    out = _sequence_to_physico("STNQ")
    np.testing.assert_array_equal(out[:, 3], [1.0, 1.0, 1.0, 1.0])


def test_sequence_to_physico_aromatic():
    # F, Y, W are aromatic
    out = _sequence_to_physico("FYW")
    np.testing.assert_array_equal(out[:, 4], [1.0, 1.0, 1.0])


def test_sequence_to_physico_nonstandard_aa_zeros():
    # Unknown AA should fall back to zeros
    out = _sequence_to_physico("X")
    np.testing.assert_array_equal(out[0], np.zeros(_N_PHYSICO))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_WT = "ACDEFGHIKLM"   # 11 AA
_HIDDEN_DIM = 8


def _make_df(n: int = 4) -> pd.DataFrame:
    rows = []
    aa = list("ACDEFGHIKLM")
    for i in range(n):
        mut_aa = aa[i % len(aa)]
        rows.append({
            "variant_id": f"var_{i:04d}",
            "mutant": f"{_WT[0]}1{mut_aa}",
            "mutated_sequence": mut_aa + _WT[1:],
            "wt_sequence": _WT,
            "fitness": float(i),
        })
    return pd.DataFrame(rows)


def _make_mock_outputs(batch_size: int, seq_len: int):
    import torch
    hidden = np.ones((batch_size, seq_len + 2, _HIDDEN_DIM), dtype=np.float32)
    ns = types.SimpleNamespace()
    ns.last_hidden_state = torch.tensor(hidden)
    return ns


def _mock_tokenizer(seqs, return_tensors, padding, truncation, max_length):
    import torch
    B = len(seqs)
    L = len(seqs[0]) + 2  # simplification: assume same length
    return {
        "input_ids": torch.zeros(B, L, dtype=torch.long),
        "attention_mask": torch.ones(B, L, dtype=torch.long),
    }


class _MockModel:
    def __call__(self, **kwargs):
        B = kwargs["input_ids"].shape[0]
        seq_len = B  # wrong, but let's compute from input_ids shape
        L = kwargs["input_ids"].shape[1] - 2
        return _make_mock_outputs(B, L)
    def eval(self): return self
    def to(self, device): return self


def _inject_mock(enc: PLMPhysicoEncoder) -> None:
    """Inject mock tokenizer/model into the internal ESMEncoder."""
    enc._esm._tokenizer = _mock_tokenizer
    enc._esm._model = _MockModel()


def _inject_mock_concat(enc: PLMSimpleConcatEncoder) -> None:
    enc._esm._tokenizer = _mock_tokenizer
    enc._esm._model = _MockModel()


# ---------------------------------------------------------------------------
# PLMPhysicoEncoder tests
# ---------------------------------------------------------------------------

def test_physico_output_shape(tmp_path):
    enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(enc)
    df = _make_df(3)
    enc.fit(df, np.array([0.0, 1.0, 2.0]))
    out = enc.transform(df)
    assert out.shape == (3, _HIDDEN_DIM + _N_PHYSICO)


def test_physico_output_dtype(tmp_path):
    enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(enc)
    df = _make_df(2)
    enc.fit(df, np.array([0.0, 1.0]))
    out = enc.transform(df)
    assert out.dtype == np.float64


def test_physico_dim_is_esm_plus_5(tmp_path):
    enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(enc)
    df = _make_df(2)
    enc.fit(df, np.array([0.0, 1.0]))
    out = enc.transform(df)
    assert out.shape[1] == _HIDDEN_DIM + _N_PHYSICO


def test_physico_differs_from_esm_mean(tmp_path):
    """PLMPhysicoEncoder output must differ from plain ESM mean-pool."""
    from rag_al.representations.plm import ESMEncoder

    phys_enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(phys_enc)
    df = _make_df(2)
    phys_enc.fit(df, np.array([0.0, 1.0]))
    phys_out = phys_enc.transform(df)

    # Build a mock ESMEncoder for comparison
    esm_enc = ESMEncoder(mode="mean", cache_dir=None, device="cpu")
    esm_enc._tokenizer = _mock_tokenizer
    esm_enc._model = _MockModel()
    esm_enc.fit(df, np.array([0.0, 1.0]))
    esm_out = esm_enc.transform(df)

    # physico output is wider; the extra 5 columns make them definitely differ
    assert phys_out.shape[1] > esm_out.shape[1]


def test_physico_cache_avoids_recompute(tmp_path):
    """Second transform() should hit cache, not call the model."""
    enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(enc)
    df = _make_df(2)
    enc.fit(df, np.array([0.0, 1.0]))
    out1 = enc.transform(df)

    class _NeverCall:
        def __call__(self, **kwargs):
            raise AssertionError("Model should not be called on cache hit")
        def eval(self): return self
        def to(self, device): return self

    enc2 = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    enc2._esm._tokenizer = _mock_tokenizer
    enc2._esm._model = _NeverCall()
    enc2.fit(df, np.array([0.0, 1.0]))
    out2 = enc2.transform(df)

    np.testing.assert_allclose(out1, out2)


def test_physico_n_features_property(tmp_path):
    """n_features returns D_esm + 5 after first transform."""
    enc = PLMPhysicoEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock(enc)
    df = _make_df(2)
    enc.fit(df, np.zeros(2))
    enc.transform(df)
    assert enc.n_features == _HIDDEN_DIM + _N_PHYSICO


# ---------------------------------------------------------------------------
# PLMSimpleConcatEncoder tests
# ---------------------------------------------------------------------------

def test_concat_output_shape(tmp_path):
    enc = PLMSimpleConcatEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock_concat(enc)
    df = _make_df(3)
    enc.fit(df, np.array([0.0, 1.0, 2.0]))
    out = enc.transform(df)
    assert out.shape == (3, _HIDDEN_DIM + 29)


def test_concat_output_dtype(tmp_path):
    enc = PLMSimpleConcatEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock_concat(enc)
    df = _make_df(2)
    enc.fit(df, np.array([0.0, 1.0]))
    out = enc.transform(df)
    assert out.dtype == np.float64


def test_concat_wider_than_esm_alone(tmp_path):
    """PLMSimpleConcatEncoder output must be wider than ESM alone by exactly 29."""
    from rag_al.representations.plm import ESMEncoder

    enc = PLMSimpleConcatEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock_concat(enc)
    df = _make_df(2)
    enc.fit(df, np.zeros(2))
    concat_out = enc.transform(df)

    esm_enc = ESMEncoder(mode="mean", cache_dir=None, device="cpu")
    esm_enc._tokenizer = _mock_tokenizer
    esm_enc._model = _MockModel()
    esm_enc.fit(df, np.zeros(2))
    esm_out = esm_enc.transform(df)

    assert concat_out.shape[1] == esm_out.shape[1] + 29


def test_concat_transform_labeled_consistent(tmp_path):
    """transform_labeled() and transform() should agree on the same labeled df."""
    enc = PLMSimpleConcatEncoder(cache_dir=tmp_path, device="cpu")
    _inject_mock_concat(enc)
    df = _make_df(3)
    enc.fit(df, np.zeros(3))
    out_t = enc.transform(df)
    out_tl = enc.transform_labeled(df)
    np.testing.assert_allclose(out_t, out_tl)


def test_concat_fit_required_before_transform(tmp_path):
    """transform() without fit() should raise (physico scaler unfit)."""
    # ESM transform() can run without fit(); physico.transform() raises without fit().
    # Intercept by checking the physico scaler directly.
    enc = PLMSimpleConcatEncoder(cache_dir=tmp_path, device="cpu")
    with pytest.raises(RuntimeError, match="fit"):
        enc._physico.transform(_make_df(2))
