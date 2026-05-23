from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AbstractEncoder

log = logging.getLogger(__name__)

# Lazy imports — torch and transformers are only needed when PLM is used
_torch = None
_AutoTokenizer = None
_AutoModel = None


def _lazy_imports() -> None:
    global _torch, _AutoTokenizer, _AutoModel
    if _torch is None:
        import torch
        from transformers import AutoTokenizer, AutoModel
        _torch = torch
        _AutoTokenizer = AutoTokenizer
        _AutoModel = AutoModel


def _best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on available hardware."""
    _lazy_imports()
    if _torch.cuda.is_available():
        return "cuda"
    if _torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ESMEncoder(AbstractEncoder):
    """
    Protein language model encoder using ESM-2 (via HuggingFace transformers).

    Three embedding modes
    ---------------------
    'mean'   Mean pool of all residue hidden states. Shape: (N, D).
    'delta'  Mean pool(mutant) − mean pool(WT). Shape: (N, D).
             Captures the representation shift induced by mutations.
    'site'   Not yet implemented; falls back to 'mean'.

    Embeddings are computed once and cached to disk as .npy files.
    On subsequent calls the cache is loaded instead of re-running the model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
    mode : str
        One of 'mean', 'delta'.
    embed_batch_size : int
        Number of sequences per forward pass.
    device : str or None
        'cuda', 'mps', 'cpu', or None (auto-detect).
    cache_dir : Path or None
        Directory for embedding cache. If None, no caching.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        mode: str = "mean",
        embed_batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if mode not in ("mean", "delta"):
            raise ValueError(f"mode must be 'mean' or 'delta', got {mode!r}")

        self.model_name = model_name
        self.mode = mode
        self.embed_batch_size = embed_batch_size
        self.device = device or _best_device()
        self.cache_dir = cache_dir

        self._wt_sequence: Optional[str] = None
        self._wt_embedding: Optional[np.ndarray] = None   # shape (D,)
        self._hidden_size: Optional[int] = None

        # Model/tokenizer loaded lazily on first use
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        if self._tokenizer is not None:
            return
        _lazy_imports()
        log.info("Loading ESM-2 tokenizer and model: %s", self.model_name)
        self._tokenizer = _AutoTokenizer.from_pretrained(self.model_name)
        self._model = _AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)
        log.info("ESM-2 loaded on device: %s", self.device)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"embeddings_{safe}_{self.mode}.npy"

    def _ids_cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"variant_ids_{safe}_{self.mode}.npy"

    def _try_load_cache(
        self, variant_ids: list[str]
    ) -> Optional[np.ndarray]:
        cp = self._cache_path()
        ip = self._ids_cache_path()
        if cp is None or not cp.exists() or ip is None or not ip.exists():
            return None
        cached_ids: np.ndarray = np.load(ip, allow_pickle=True)
        if list(cached_ids) != variant_ids:
            log.debug("Cache variant order mismatch — recomputing embeddings.")
            return None
        log.info("Loading cached embeddings from %s", cp)
        return np.load(cp)

    def _save_cache(
        self, embeddings: np.ndarray, variant_ids: list[str]
    ) -> None:
        cp = self._cache_path()
        ip = self._ids_cache_path()
        if cp is None:
            return
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.save(cp, embeddings)
        np.save(ip, np.array(variant_ids, dtype=object))
        log.info("Saved embeddings cache to %s", cp)

    # ------------------------------------------------------------------
    # Core embedding computation
    # ------------------------------------------------------------------

    def _embed_sequences(self, sequences: list[str]) -> np.ndarray:
        """
        Compute mean-pooled ESM-2 embeddings for a list of sequences.

        Parameters
        ----------
        sequences : list of str
            Amino acid sequences.

        Returns
        -------
        np.ndarray
            Shape (N, D), float32 → upcast to float64.
        """
        self._load_model()
        _lazy_imports()

        all_embeddings: list[np.ndarray] = []
        n = len(sequences)
        bs = self.embed_batch_size

        for start in range(0, n, bs):
            batch = sequences[start : start + bs]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with _torch.no_grad():
                outputs = self._model(**inputs)

            # outputs.last_hidden_state: (batch, seq_len+2, D)
            # tokens 0 and -1 are <cls>/<eos> — mask them out
            hidden = outputs.last_hidden_state  # (B, L+2, D)
            attention_mask = inputs["attention_mask"]  # (B, L+2)
            # Exclude special tokens: positions 0 and last non-pad token
            # Simple approach: mask out position 0 (cls) and rely on
            # attention_mask to handle padding
            # We zero the cls position
            mask = attention_mask.clone().float()
            mask[:, 0] = 0.0  # zero out <cls>
            # For each sequence, also zero out the last non-padding token (<eos>)
            for i, length in enumerate(attention_mask.sum(dim=1)):
                mask[i, length - 1] = 0.0

            mask_expanded = mask.unsqueeze(-1)  # (B, L+2, 1)
            sum_hidden = (hidden * mask_expanded).sum(dim=1)  # (B, D)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)   # (B, 1)
            mean_hidden = (sum_hidden / count).cpu().numpy()    # (B, D)
            all_embeddings.append(mean_hidden)

        result = np.vstack(all_embeddings).astype(np.float64)
        if self._hidden_size is None:
            self._hidden_size = result.shape[1]
        return result

    # ------------------------------------------------------------------
    # AbstractEncoder interface
    # ------------------------------------------------------------------

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """
        Store the wild-type sequence for delta-embedding computation.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Must contain 'wt_sequence' and 'mutated_sequence'.
        y_labeled : np.ndarray
            Ignored (no fitness information used by this encoder).
        """
        wt = df_labeled["wt_sequence"].iloc[0]
        if self._wt_sequence is None:
            self._wt_sequence = wt
        if self.mode == "delta" and self._wt_embedding is None:
            log.info("Computing WT embedding for delta mode.")
            self._wt_embedding = self._embed_sequences([wt])[0]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants using ESM-2.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'mutated_sequence' and 'variant_id'.

        Returns
        -------
        np.ndarray
            Shape (n_variants, D), float64.
            D = 320 for esm2_t6_8M, 1280 for esm2_t33_650M, etc.
        """
        sequences = list(df["mutated_sequence"])
        variant_ids = list(df["variant_id"])

        # Try cache first
        cached = self._try_load_cache(variant_ids)
        if cached is not None:
            emb = cached
        else:
            log.info(
                "Computing ESM-2 (%s) embeddings for %d sequences.",
                self.model_name, len(sequences),
            )
            emb = self._embed_sequences(sequences)
            self._save_cache(emb, variant_ids)

        if self.mode == "mean":
            return emb
        elif self.mode == "delta":
            if self._wt_embedding is None:
                raise RuntimeError(
                    "WT embedding not computed. Call fit() before transform()."
                )
            return emb - self._wt_embedding[np.newaxis, :]
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")

    @property
    def n_features(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size
        # Return known sizes for common ESM-2 models
        _KNOWN_SIZES = {
            "facebook/esm2_t6_8M_UR50D": 320,
            "facebook/esm2_t12_35M_UR50D": 480,
            "facebook/esm2_t30_150M_UR50D": 640,
            "facebook/esm2_t33_650M_UR50D": 1280,
            "facebook/esm2_t36_3B_UR50D": 2560,
        }
        return _KNOWN_SIZES.get(self.model_name, -1)
