"""
Cache-path isolation tests.

The PLM representations share one cache directory but must write to distinct
files, otherwise embeddings of different shapes/semantics would silently
collide. The only thing keeping mean (D,) and physico (D+5,) apart is the
filename suffix — both key by sha256(seq) — so this invariant is worth pinning
down explicitly. No model weights are loaded here (paths only).
"""

from __future__ import annotations

from pathlib import Path

from rag_al.representations.plm import ESMEncoder
from rag_al.representations.plm_physico import PLMPhysicoEncoder

_MODEL = "facebook/esm2_t33_650M_UR50D"
_SAFE = "facebook__esm2_t33_650M_UR50D"


def test_mean_delta_share_one_cache(tmp_path):
    """mean and delta intentionally share cache_{model}.pkl."""
    mean = ESMEncoder(model_name=_MODEL, mode="mean", cache_dir=tmp_path, device="cpu")
    delta = ESMEncoder(model_name=_MODEL, mode="delta", cache_dir=tmp_path, device="cpu")
    assert mean._cache_path() == delta._cache_path()
    assert mean._cache_path() == tmp_path / f"cache_{_SAFE}.pkl"


def test_site_and_physico_have_distinct_files(tmp_path):
    """site and physico each get their own cache file, distinct from mean."""
    mean = ESMEncoder(model_name=_MODEL, mode="mean", cache_dir=tmp_path, device="cpu")
    site = ESMEncoder(model_name=_MODEL, mode="site", cache_dir=tmp_path, device="cpu")
    physico = PLMPhysicoEncoder(model_name=_MODEL, cache_dir=tmp_path)

    mean_path = mean._cache_path()
    site_path = site._site_cache_path()
    physico_path = physico._cache_path()

    assert site_path == tmp_path / f"cache_{_SAFE}_site.pkl"
    assert physico_path == tmp_path / f"cache_{_SAFE}_physico.pkl"

    # The core invariant: three distinct files, no overlap.
    paths = {mean_path, site_path, physico_path}
    assert len(paths) == 3


def test_physico_and_mean_key_same_seq_but_never_share_a_file(tmp_path):
    """
    mean and physico both key by sha256(seq); isolation depends solely on the
    filename. Guard against a regression that drops the suffix.
    """
    mean = ESMEncoder(model_name=_MODEL, mode="mean", cache_dir=tmp_path, device="cpu")
    physico = PLMPhysicoEncoder(model_name=_MODEL, cache_dir=tmp_path)

    seq = "MKTAYIAKQR"
    assert mean._seq_hash(seq) == physico._seq_hash(seq)  # same key
    assert mean._cache_path() != physico._cache_path()    # different file


def test_no_cache_dir_yields_no_paths(tmp_path):
    """With cache_dir=None every cache path resolves to None (in-memory only)."""
    mean = ESMEncoder(model_name=_MODEL, mode="mean", cache_dir=None, device="cpu")
    site = ESMEncoder(model_name=_MODEL, mode="site", cache_dir=None, device="cpu")
    physico = PLMPhysicoEncoder(model_name=_MODEL, cache_dir=None)
    assert mean._cache_path() is None
    assert site._site_cache_path() is None
    assert physico._cache_path() is None
