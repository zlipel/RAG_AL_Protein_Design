"""
Tests for ESMEncoder sequence-hash cache (plm.py).

_embed_sequences is mocked throughout — no model weights are loaded.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from rag_al.representations.plm import ESMEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D = 8  # tiny embedding dim for tests


def _make_df(sequences: list[str], wt: str = "AAAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mutated_sequence": sequences,
            "wt_sequence": [wt] * len(sequences),
            "variant_id": [f"v{i}" for i in range(len(sequences))],
        }
    )


def _fake_embed(sequences: list[str]) -> np.ndarray:
    """Deterministic fake embeddings: each row = ord of first char repeated D times."""
    return np.array(
        [[float(ord(s[0]))] * _D for s in sequences], dtype=np.float64
    )


# ---------------------------------------------------------------------------
# Test 1: cache_dir=None — no disk I/O, correct shape
# ---------------------------------------------------------------------------

def test_no_cache_dir_mean(tmp_path):
    encoder = ESMEncoder(mode="mean", cache_dir=None)
    df = _make_df(["AAAA", "BBBB", "CCCC"])
    with patch.object(encoder, "_embed_sequences", side_effect=_fake_embed):
        result = encoder.transform(df)
    assert result.shape == (3, _D)
    # In-memory cache is populated but no file was written
    assert encoder._cache_path() is None
    assert list(tmp_path.iterdir()) == []


def test_no_cache_dir_does_not_create_file(tmp_path):
    encoder = ESMEncoder(mode="mean", cache_dir=None)
    df = _make_df(["AAAA"])
    with patch.object(encoder, "_embed_sequences", side_effect=_fake_embed):
        encoder.transform(df)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Test 2: Full compute + save
# ---------------------------------------------------------------------------

def test_full_compute_saves_pkl(tmp_path):
    encoder = ESMEncoder(mode="mean", cache_dir=tmp_path)
    df = _make_df(["AAAA", "BBBB", "CCCC"])
    with patch.object(encoder, "_embed_sequences", side_effect=_fake_embed) as mock_emb:
        result = encoder.transform(df)
    assert result.shape == (3, _D)
    # _embed_sequences called once with all 3 sequences
    mock_emb.assert_called_once()
    assert len(mock_emb.call_args[0][0]) == 3
    # .pkl file was written
    pkl_files = list(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == 1


# ---------------------------------------------------------------------------
# Test 3: Full cache hit — no model call on reload
# ---------------------------------------------------------------------------

def test_cache_hit_no_embed_call(tmp_path):
    # First encoder writes the cache
    enc1 = ESMEncoder(mode="mean", cache_dir=tmp_path)
    df = _make_df(["AAAA", "BBBB"])
    with patch.object(enc1, "_embed_sequences", side_effect=_fake_embed):
        emb1 = enc1.transform(df)

    # Second encoder (fresh instance) reads from cache
    enc2 = ESMEncoder(mode="mean", cache_dir=tmp_path)
    with patch.object(enc2, "_embed_sequences", side_effect=_fake_embed) as mock_emb:
        emb2 = enc2.transform(df)

    mock_emb.assert_not_called()
    np.testing.assert_array_equal(emb1, emb2)


# ---------------------------------------------------------------------------
# Test 4: Partial miss — only new sequences are embedded
# ---------------------------------------------------------------------------

def test_partial_miss_embeds_only_new(tmp_path):
    enc1 = ESMEncoder(mode="mean", cache_dir=tmp_path)
    df_small = _make_df(["AAAA", "BBBB"])
    with patch.object(enc1, "_embed_sequences", side_effect=_fake_embed):
        enc1.transform(df_small)

    enc2 = ESMEncoder(mode="mean", cache_dir=tmp_path)
    df_big = _make_df(["AAAA", "BBBB", "CCCC", "DDDD"])
    with patch.object(enc2, "_embed_sequences", side_effect=_fake_embed) as mock_emb:
        result = enc2.transform(df_big)

    # Only 2 new sequences should have been passed to _embed_sequences
    mock_emb.assert_called_once()
    embedded_seqs = mock_emb.call_args[0][0]
    assert set(embedded_seqs) == {"CCCC", "DDDD"}
    assert result.shape == (4, _D)


# ---------------------------------------------------------------------------
# Test 5: Delta mode WT caching
# ---------------------------------------------------------------------------

def test_delta_wt_is_cached_and_reused(tmp_path):
    wt = "WWWW"
    df = _make_df(["AAAA", "BBBB"], wt=wt)

    enc1 = ESMEncoder(mode="delta", cache_dir=tmp_path)
    with patch.object(enc1, "_embed_sequences", side_effect=_fake_embed) as mock_emb:
        enc1.fit(df, np.zeros(2))
        enc1.transform(df)

    # WT hash should be in the cache
    wt_key = enc1._seq_hash(wt)
    assert wt_key in enc1._embedding_cache

    # Second encoder: fit() should NOT call _embed_sequences for WT
    enc2 = ESMEncoder(mode="delta", cache_dir=tmp_path)
    with patch.object(enc2, "_embed_sequences", side_effect=_fake_embed) as mock_emb2:
        enc2.fit(df, np.zeros(2))

    # _embed_sequences should not have been called for WT (loaded from cache)
    for call in mock_emb2.call_args_list:
        seqs = call[0][0]
        assert wt not in seqs, f"WT sequence re-embedded unexpectedly: {seqs}"


# ---------------------------------------------------------------------------
# Test 6: Return order matches df row order
# ---------------------------------------------------------------------------

def test_transform_preserves_row_order(tmp_path):
    seqs = ["CCCC", "AAAA", "BBBB"]
    df = _make_df(seqs)
    encoder = ESMEncoder(mode="mean", cache_dir=tmp_path)
    with patch.object(encoder, "_embed_sequences", side_effect=_fake_embed):
        result = encoder.transform(df)

    # Row i should correspond to seqs[i]
    for i, seq in enumerate(seqs):
        expected = float(ord(seq[0]))
        assert result[i, 0] == expected, f"Row {i} mismatch: got {result[i,0]}, want {expected}"
