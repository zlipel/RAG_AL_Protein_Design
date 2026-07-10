from __future__ import annotations

import hashlib
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AbstractEncoder
from .physicochemical import (
    PhysicochemicalEncoder,
    _AA_HYDROPATHY,
    _AA_MW,
    _POLAR,
    _AROMATIC,
    _POSITIVE,
    _NEGATIVE,
)
from .plm import ESMEncoder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-residue physicochemical lookup — 5 properties per AA
# ---------------------------------------------------------------------------
# [0] hydropathy    (Kyte-Doolittle, continuous)
# [1] charge        (+1 R/K, -1 D/E, 0 otherwise)
# [2] molecular weight (Da, continuous)
# [3] polar         (1 if S/T/N/Q, else 0)
# [4] aromatic      (1 if F/Y/W, else 0)

_STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")

def _build_residue_table() -> np.ndarray:
    """Return (20, 5) float64 lookup table for _STANDARD_AA."""
    table = np.zeros((20, 5), dtype=np.float64)
    for i, aa in enumerate(_STANDARD_AA):
        charge = 1.0 if aa in _POSITIVE else (-1.0 if aa in _NEGATIVE else 0.0)
        table[i] = [
            _AA_HYDROPATHY.get(aa, 0.0),
            charge,
            _AA_MW.get(aa, 110.0),
            1.0 if aa in _POLAR else 0.0,
            1.0 if aa in _AROMATIC else 0.0,
        ]
    return table

_AA_IDX = {aa: i for i, aa in enumerate(_STANDARD_AA)}
_RESIDUE_TABLE = _build_residue_table()   # (20, 5)
_UNK_ROW = np.zeros(5, dtype=np.float64)  # fallback for non-standard AA

_N_PHYSICO = 5


def _sequence_to_physico(sequence: str) -> np.ndarray:
    """Map a sequence to a (L, 5) per-residue physicochemical matrix."""
    L = len(sequence)
    out = np.empty((L, _N_PHYSICO), dtype=np.float64)
    for i, aa in enumerate(sequence):
        idx = _AA_IDX.get(aa.upper())
        out[i] = _RESIDUE_TABLE[idx] if idx is not None else _UNK_ROW
    return out


# ---------------------------------------------------------------------------
# PLMPhysicoEncoder — per-residue ESM concat then pool
# ---------------------------------------------------------------------------

class PLMPhysicoEncoder(AbstractEncoder):
    """
    Per-residue fusion of ESM-2 hidden states and physicochemical properties.

    For each residue i in a sequence:
        fused_i = [h_i | p_i]
    where h_i is the ESM-2 hidden state (D-dim) and p_i is a 5-dim
    physicochemical lookup vector [hydropathy, charge, MW, polar, aromatic].

    The fused vectors are mean-pooled across all residues (CLS/EOS excluded),
    producing a (D + 5)-dim representation per variant.

    This differs from PLMSimpleConcatEncoder, which concatenates *sequence-level*
    ESM mean-pool and physicochemical features post-hoc. The per-residue fusion
    here preserves positional co-occurrence between PLM context and AA properties —
    e.g., whether a high-hydropathy residue also has a strong contextual embedding.

    Caching
    -------
    Fused vectors are cached to disk at cache_dir/cache_{model}_physico.pkl,
    keyed by sha256(sequence). The cache stores the final (D+5,) vectors,
    not intermediate hidden states.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
    embed_batch_size : int
        Sequences per forward pass.
    device : str or None
        'cuda', 'mps', 'cpu', or None (auto-detect via ESMEncoder).
    cache_dir : Path or None
        Directory for on-disk cache. If None, only in-memory caching.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        embed_batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.cache_dir = cache_dir

        # ESMEncoder used solely for model/tokenizer loading and device config.
        # We call esm._load_model() then access esm._tokenizer / esm._model directly.
        self._esm = ESMEncoder(
            model_name=model_name,
            mode="mean",           # mode doesn't matter — we bypass transform()
            embed_batch_size=embed_batch_size,
            device=device,
            cache_dir=None,        # we manage our own cache below
        )

        self._physico_cache: Optional[dict[str, np.ndarray]] = None
        self._hidden_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"cache_{safe}_physico.pkl"

    def _seq_hash(self, seq: str) -> str:
        return hashlib.sha256(seq.encode()).hexdigest()

    def _load_cache(self) -> None:
        if self._physico_cache is not None:
            return
        cp = self._cache_path()
        if cp is not None and cp.exists():
            with open(cp, "rb") as f:
                self._physico_cache = pickle.load(f)
            log.info("Loaded physico cache (%d entries) from %s", len(self._physico_cache), cp)
        else:
            self._physico_cache = {}

    def _save_cache(self) -> None:
        cp = self._cache_path()
        if cp is None:
            return
        assert self._physico_cache is not None
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(dir=cp.parent, suffix=".tmp")
        tmp = Path(tmp_str)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self._physico_cache, f)
            tmp.replace(cp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Saved physico cache (%d entries) to %s", len(self._physico_cache), cp)

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_physico(self, sequences: list[str]) -> np.ndarray:
        """
        Compute fused [ESM | physico] mean-pool embeddings. Returns (N, D+5).

        Runs a batched ESM-2 forward pass, then for each sequence:
          1. Extract residue hidden states (skip CLS/EOS tokens).
          2. Look up the 5-dim physico vector for each AA.
          3. Concat [h_i | p_i] per residue → (L_i, D+5).
          4. Mean-pool over residues → (D+5,).
        """
        import torch

        self._esm._load_model()
        tokenizer = self._esm._tokenizer
        model = self._esm._model
        device = self._esm.device
        bs = self.embed_batch_size

        all_vecs: list[np.ndarray] = []
        n = len(sequences)

        for start in range(0, n, bs):
            batch_seqs = sequences[start : start + bs]
            inputs = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # (B, L+2, D)
            hidden = outputs.last_hidden_state.cpu().numpy()

            if self._hidden_size is None:
                self._hidden_size = hidden.shape[-1]

            for i, seq in enumerate(batch_seqs):
                L = len(seq)
                # Token indices 1..L+1 are the residue tokens (0=CLS, L+1=EOS)
                # attention_mask includes both, so we slice by sequence length.
                # Clamp to actual token length in case of truncation.
                actual_L = min(L, hidden.shape[1] - 2)  # tokens available for residues
                h = hidden[i, 1 : actual_L + 1, :]       # (actual_L, D)
                p = _sequence_to_physico(seq[:actual_L])  # (actual_L, 5)
                fused = np.concatenate([h, p], axis=1)    # (actual_L, D+5)
                all_vecs.append(fused.mean(axis=0))        # (D+5,)

        return np.vstack(all_vecs).astype(np.float64)

    # ------------------------------------------------------------------
    # AbstractEncoder interface
    # ------------------------------------------------------------------

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """No-op: physico lookup table is fixed; no statistics to fit."""
        pass

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants as per-residue fused [ESM | physico] mean-pool.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'mutated_sequence'.

        Returns
        -------
        np.ndarray
            Shape (n_variants, D_esm + 5), float64.
        """
        sequences = list(df["mutated_sequence"])
        keys = [self._seq_hash(seq) for seq in sequences]

        self._load_cache()
        assert self._physico_cache is not None

        missing_indices = [i for i, k in enumerate(keys) if k not in self._physico_cache]

        if missing_indices:
            seen: dict[str, str] = {}  # hash → seq (deduplicate)
            for i in missing_indices:
                k = keys[i]
                if k not in seen:
                    seen[k] = sequences[i]
            unique_keys = list(seen.keys())
            unique_seqs = list(seen.values())
            log.info(
                "Computing PLM-physico embeddings (%s) for %d new sequences.",
                self.model_name, len(unique_seqs),
            )
            new_embs = self._embed_physico(unique_seqs)
            for k, emb in zip(unique_keys, new_embs):
                self._physico_cache[k] = emb
            self._save_cache()

        return np.stack([self._physico_cache[k] for k in keys]).astype(np.float64)

    @property
    def n_features(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size + _N_PHYSICO
        _KNOWN_ESM = {
            "facebook/esm2_t6_8M_UR50D": 320,
            "facebook/esm2_t12_35M_UR50D": 480,
            "facebook/esm2_t30_150M_UR50D": 640,
            "facebook/esm2_t33_650M_UR50D": 1280,
            "facebook/esm2_t36_3B_UR50D": 2560,
        }
        base = _KNOWN_ESM.get(self.model_name, -1)
        return base + _N_PHYSICO if base > 0 else -1


# ---------------------------------------------------------------------------
# PLMSimpleConcatEncoder — sequence-level post-hoc concat
# ---------------------------------------------------------------------------

class PLMSimpleConcatEncoder(AbstractEncoder):
    """
    Post-hoc concatenation of sequence-level ESM-2 mean-pool and
    29-dim physicochemical features.

    Unlike PLMPhysicoEncoder, the two representations are computed
    independently at the sequence level and joined after pooling.
    This is the simplest possible PLM + physico baseline.

    Output dimension: D_esm + 29.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for ESM-2.
    embed_batch_size : int
        Sequences per ESM-2 forward pass.
    device : str or None
        'cuda', 'mps', 'cpu', or None (auto-detect).
    cache_dir : Path or None
        Directory for ESM-2 on-disk embedding cache.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        embed_batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self._esm = ESMEncoder(
            model_name=model_name,
            mode="mean",
            embed_batch_size=embed_batch_size,
            device=device,
            cache_dir=cache_dir,
        )
        self._physico = PhysicochemicalEncoder()

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        self._esm.fit(df_labeled, y_labeled)
        self._physico.fit(df_labeled, y_labeled)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants as [ESM mean-pool | physico descriptors].

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'mutated_sequence' (and 'wt_sequence' for ESM delta).

        Returns
        -------
        np.ndarray
            Shape (n_variants, D_esm + 29), float64.
        """
        esm_feats = self._esm.transform(df)        # (N, D_esm)
        physico_feats = self._physico.transform(df) # (N, 29)
        return np.hstack([esm_feats, physico_feats]).astype(np.float64)

    def transform_labeled(self, df: pd.DataFrame) -> np.ndarray:
        esm_feats = self._esm.transform_labeled(df)
        physico_feats = self._physico.transform(df)
        return np.hstack([esm_feats, physico_feats]).astype(np.float64)

    @property
    def n_features(self) -> int:
        base = self._esm.n_features
        return base + 29 if base > 0 else -1
