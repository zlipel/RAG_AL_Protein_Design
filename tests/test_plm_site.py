"""
Tests for ESMEncoder(mode='site') — plm_site representation.

All tests use a mock model so no HuggingFace weights are loaded in CI.
"""
from __future__ import annotations

import re
import types
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from rag_al.representations.plm import ESMEncoder, _parse_mutant_positions


# ---------------------------------------------------------------------------
# Unit tests for _parse_mutant_positions
# ---------------------------------------------------------------------------

def test_parse_single_site():
    assert _parse_mutant_positions("A23V") == [22]


def test_parse_multi_site():
    assert _parse_mutant_positions("A23V:G45L") == [22, 44]


def test_parse_position_one_indexed():
    # Position 1 in the mutant string → index 0 in the sequence
    assert _parse_mutant_positions("M1A") == [0]


def test_parse_three_sites():
    assert _parse_mutant_positions("A1V:G2L:K3R") == [0, 1, 2]


# ---------------------------------------------------------------------------
# Fixture: a tiny synthetic df with one single-site variant
# ---------------------------------------------------------------------------

_WT = "ACDEFGHIKLM"  # 11 AA, 0-indexed positions 0–10


def _make_df(mutant_strs: list[str]) -> pd.DataFrame:
    rows = []
    aa = list("ACDEFGHIKLMNPQRSTVWY")
    for i, mut in enumerate(mutant_strs):
        # Apply the first mutation to build mutated_sequence
        pos = _parse_mutant_positions(mut)[0]
        new_aa = re.search(r"[A-Z]$", mut.split(":")[0]).group()
        seq = list(_WT)
        seq[pos] = new_aa
        rows.append({
            "variant_id": f"var_{i:04d}",
            "mutant": mut,
            "mutated_sequence": "".join(seq),
            "wt_sequence": _WT,
            "fitness": float(i),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Mock ESM-2 model/tokenizer that returns deterministic hidden states
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 8    # tiny hidden dim for testing
_SEQ_LEN = 11      # length of _WT


def _make_mock_outputs(batch_size: int, seq_len: int) -> types.SimpleNamespace:
    """Return a SimpleNamespace mimicking HuggingFace model output."""
    # Each position gets a distinct vector: position p → all-p float32
    hidden = np.zeros((batch_size, seq_len + 2, _HIDDEN_DIM), dtype=np.float32)
    for b in range(batch_size):
        for p in range(seq_len + 2):
            hidden[b, p, :] = float(p)  # token index p → value p at all dims

    import torch
    ns = types.SimpleNamespace()
    ns.last_hidden_state = torch.tensor(hidden)
    return ns


def _make_mock_encoder(tmp_path) -> ESMEncoder:
    """Return an ESMEncoder(mode='site') with model/tokenizer mocked out."""
    enc = ESMEncoder(
        model_name="mock/esm2",
        mode="site",
        embed_batch_size=4,
        device="cpu",
        cache_dir=tmp_path,
    )

    # Mock tokenizer: returns attention_mask of all 1s, shape (B, seq+2)
    def mock_tokenizer(seqs, return_tensors, padding, truncation, max_length):
        import torch
        B = len(seqs)
        L = _SEQ_LEN + 2
        return {
            "input_ids": torch.zeros(B, L, dtype=torch.long),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }

    class MockModel:
        def __call__(self, **kwargs):
            B = kwargs["input_ids"].shape[0]
            return _make_mock_outputs(B, _SEQ_LEN)
        def eval(self): return self
        def to(self, device): return self

    enc._tokenizer = mock_tokenizer
    enc._model = MockModel()
    return enc


# ---------------------------------------------------------------------------
# Tests for transform() in site mode
# ---------------------------------------------------------------------------

def test_site_output_shape(tmp_path):
    """transform() returns (N, D) for site mode."""
    enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A1V", "C2G", "D3E"])
    enc.fit(df, np.array([0.1, 0.2, 0.3]))
    out = enc.transform(df)
    assert out.shape == (3, _HIDDEN_DIM), f"Expected (3, {_HIDDEN_DIM}), got {out.shape}"


def test_site_output_dtype(tmp_path):
    """Output is float64."""
    enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A1V"])
    enc.fit(df, np.array([0.0]))
    out = enc.transform(df)
    assert out.dtype == np.float64


def test_site_extracts_mutated_position(tmp_path):
    """
    The site embedding for 'A1V' (position 0, token index 1) should be
    the vector at token 1, i.e. all ones (since mock uses value=token_index).
    """
    enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A1V"])   # position 0 → token 1 → value 1.0
    enc.fit(df, np.array([0.0]))
    out = enc.transform(df)
    expected = np.ones(_HIDDEN_DIM, dtype=np.float64)
    np.testing.assert_allclose(out[0], expected, err_msg="Site embedding should be token[1] = 1.0")


def test_site_differs_from_mean(tmp_path):
    """
    Site-mode and mean-mode should give different vectors.
    Mean pool averages all tokens; site extracts only the mutated position.
    """
    site_enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A5V"])   # position 4, token 5
    site_enc.fit(df, np.array([0.0]))
    site_out = site_enc.transform(df)

    mean_enc = ESMEncoder(
        model_name="mock/esm2",
        mode="mean",
        embed_batch_size=4,
        device="cpu",
        cache_dir=None,
    )

    def mock_tokenizer(seqs, return_tensors, padding, truncation, max_length):
        import torch
        B = len(seqs)
        L = _SEQ_LEN + 2
        return {
            "input_ids": torch.zeros(B, L, dtype=torch.long),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }

    class MockModel:
        def __call__(self, **kwargs):
            B = kwargs["input_ids"].shape[0]
            return _make_mock_outputs(B, _SEQ_LEN)
        def eval(self): return self
        def to(self, device): return self

    mean_enc._tokenizer = mock_tokenizer
    mean_enc._model = MockModel()
    mean_enc.fit(df, np.array([0.0]))
    mean_out = mean_enc.transform(df)

    assert not np.allclose(site_out, mean_out), (
        "site and mean embeddings should differ for A5V"
    )


def test_site_multisite_averaging(tmp_path):
    """
    For 'A1V:C2G' (positions 0 and 1, tokens 1 and 2), site embedding
    should be mean([1.0, 2.0]) = 1.5 at all dims.
    """
    enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A1V:C2G"])
    enc.fit(df, np.array([0.0]))
    out = enc.transform(df)
    expected = np.full(_HIDDEN_DIM, 1.5, dtype=np.float64)
    np.testing.assert_allclose(out[0], expected, err_msg="Multi-site should average token[1] and token[2]")


def test_site_raises_without_mutant_column(tmp_path):
    """transform() raises ValueError if 'mutant' column is absent."""
    enc = _make_mock_encoder(tmp_path)
    df_no_mutant = _make_df(["A1V"]).drop(columns=["mutant"])
    enc.fit(df_no_mutant.assign(mutant=["A1V"]), np.array([0.0]))
    with pytest.raises(ValueError, match="mutant"):
        enc.transform(df_no_mutant)


def test_site_cache_avoids_recomputation(tmp_path):
    """Second transform() call should load from cache, not call the model."""
    enc = _make_mock_encoder(tmp_path)
    df = _make_df(["A3K"])
    enc.fit(df, np.array([0.0]))
    out1 = enc.transform(df)

    # Replace model with a function that raises if called
    class _NeverCall:
        def __call__(self, **kwargs):
            raise AssertionError("Model should not be called — cache hit expected")
        def eval(self): return self
        def to(self, device): return self

    enc2 = ESMEncoder(
        model_name="mock/esm2",
        mode="site",
        embed_batch_size=4,
        device="cpu",
        cache_dir=tmp_path,
    )
    enc2._tokenizer = enc._tokenizer
    enc2._model = _NeverCall()
    enc2.fit(df, np.array([0.0]))
    out2 = enc2.transform(df)

    np.testing.assert_allclose(out1, out2, err_msg="Cached and fresh site embeddings must match")


def test_site_mode_not_accepted_by_mean_init():
    """ESMEncoder with an invalid mode raises ValueError at init."""
    with pytest.raises(ValueError, match="mode must be"):
        ESMEncoder(mode="invalid_mode")
