from __future__ import annotations

import hashlib
import logging
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import AbstractEncoder

log = logging.getLogger(__name__)

# ESM-2 uses 1024 token positions; <cls> and <eos> occupy two of them.
_ESM_MAX_RESIDUES = 1022

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


def _parse_mutant_positions(mutant_str: str) -> list[int]:
    """Parse a mutant string into 0-indexed sequence positions.

    'A23V'       → [22]
    'A23V:G45L'  → [22, 44]
    """
    return [int(re.search(r"\d+", m).group()) - 1 for m in mutant_str.split(":")]


class ESMEncoder(AbstractEncoder):
    """
    Protein language model encoder using ESM-2 (via HuggingFace transformers).

    Embedding modes
    ---------------
    'mean'   Mean pool of all residue hidden states (CLS/EOS excluded).
             Shape: (N, D).
    'delta'  mean_pool(mutant) − mean_pool(WT). Shape: (N, D).
             Captures the representation shift induced by mutations.
    'site'   Average of hidden states at mutated residue positions only.
             Shape: (N, D). Requires a 'mutant' column in the DataFrame
             (e.g. 'A23V' or 'A23V:G45L'). Designed for single-site
             datasets; for multi-site, averaging may dilute signal —
             consider 'delta' as an alternative.

    Caching
    -------
    Mean-pool embeddings are cached to disk as a {seq_hash → embedding}
    pickle dict at cache_dir/cache_{model}.pkl. 'delta' reuses this cache.
    Site embeddings use a separate cache at cache_dir/cache_{model}_site.pkl,
    keyed by hash(seq + "::" + mutant_str), since the extracted vector
    depends on which positions are mutated.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
    mode : str
        One of 'mean', 'delta', 'site'.
    embed_batch_size : int
        Number of sequences per forward pass.
    device : str or None
        'cuda', 'mps', 'cpu', or None (auto-detect).
    cache_dir : Path or None
        Directory for on-disk embedding caches. If None, embeddings are
        cached in memory only for the lifetime of this encoder instance.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        mode: str = "mean",
        embed_batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        if mode not in ("mean", "delta", "site"):
            raise ValueError(f"mode must be 'mean', 'delta', or 'site', got {mode!r}")

        self.model_name = model_name
        self.mode = mode
        self.embed_batch_size = embed_batch_size
        self.device = device or _best_device()
        self.cache_dir = cache_dir

        self._wt_sequence: Optional[str] = None
        self._wt_embedding: Optional[np.ndarray] = None   # shape (D,)
        self._hidden_size: Optional[int] = None

        # Mean-pool cache: seq_hash → (D,) embedding. Shared by mean + delta.
        self._embedding_cache: Optional[dict[str, np.ndarray]] = None

        # Site cache: site_key → (D,) embedding. Separate because the vector
        # depends on both the sequence and the mutation positions.
        self._site_cache: Optional[dict[str, np.ndarray]] = None

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
    # Cache helpers — mean-pool cache
    # ------------------------------------------------------------------

    def _cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"cache_{safe}.pkl"

    def _seq_hash(self, seq: str) -> str:
        return hashlib.sha256(seq.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Populate _embedding_cache from disk. No-op if already loaded."""
        if self._embedding_cache is not None:
            return
        cp = self._cache_path()
        if cp is not None and cp.exists():
            with open(cp, "rb") as f:
                loaded: dict[str, np.ndarray] = pickle.load(f)
            self._embedding_cache = loaded
            log.info("Loaded embedding cache (%d entries) from %s", len(loaded), cp)
        else:
            self._embedding_cache = {}

    def _save_cache(self) -> None:
        cp = self._cache_path()
        if cp is None:
            return
        assert self._embedding_cache is not None
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(dir=cp.parent, suffix=".tmp")
        tmp = Path(tmp_str)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self._embedding_cache, f)
            tmp.replace(cp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Saved embedding cache (%d entries) to %s", len(self._embedding_cache), cp)

    # ------------------------------------------------------------------
    # Cache helpers — site cache
    # ------------------------------------------------------------------

    def _site_cache_path(self) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe = self.model_name.replace("/", "__")
        return self.cache_dir / f"cache_{safe}_site.pkl"

    def _site_key(self, seq: str, mutant_str: str) -> str:
        return hashlib.sha256((seq + "::" + mutant_str).encode()).hexdigest()

    def _load_site_cache(self) -> None:
        if self._site_cache is not None:
            return
        cp = self._site_cache_path()
        if cp is not None and cp.exists():
            with open(cp, "rb") as f:
                loaded: dict[str, np.ndarray] = pickle.load(f)
            self._site_cache = loaded
            log.info("Loaded site embedding cache (%d entries) from %s", len(loaded), cp)
        else:
            self._site_cache = {}

    def _save_site_cache(self) -> None:
        cp = self._site_cache_path()
        if cp is None:
            return
        assert self._site_cache is not None
        cp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_str = tempfile.mkstemp(dir=cp.parent, suffix=".tmp")
        tmp = Path(tmp_str)
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self._site_cache, f)
            tmp.replace(cp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        log.info("Saved site embedding cache (%d entries) to %s", len(self._site_cache), cp)

    # ------------------------------------------------------------------
    # Core embedding computation
    # ------------------------------------------------------------------

    def _embed_sequences(self, sequences: list[str]) -> np.ndarray:
        """Mean-pooled ESM-2 embeddings. Returns (N, D) float64."""
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

            # outputs.last_hidden_state: (B, L+2, D)
            # tokens 0 and -1 are <cls>/<eos> — mask them out
            hidden = outputs.last_hidden_state  # (B, L+2, D)
            attention_mask = inputs["attention_mask"]  # (B, L+2)
            mask = attention_mask.clone().float()
            mask[:, 0] = 0.0  # zero out <cls>
            for i, length in enumerate(attention_mask.sum(dim=1)):
                mask[i, length - 1] = 0.0  # zero out <eos>

            mask_expanded = mask.unsqueeze(-1)  # (B, L+2, 1)
            sum_hidden = (hidden * mask_expanded).sum(dim=1)  # (B, D)
            count = mask_expanded.sum(dim=1).clamp(min=1.0)   # (B, 1)
            mean_hidden = (sum_hidden / count).cpu().numpy()    # (B, D)
            all_embeddings.append(mean_hidden)

        result = np.vstack(all_embeddings).astype(np.float64)
        if self._hidden_size is None:
            self._hidden_size = result.shape[1]
        return result

    def _embed_sequences_site(
        self, sequences: list[str], mutant_strs: list[str]
    ) -> np.ndarray:
        """
        Site-averaged ESM-2 embeddings. Returns (N, D) float64.

        For each variant, extracts the hidden state at each mutated residue
        position and averages across sites. Token index = sequence position + 1
        because ESM-2 prepends a <cls> token.
        """
        self._load_model()
        _lazy_imports()

        all_embeddings: list[np.ndarray] = []
        n = len(sequences)
        bs = self.embed_batch_size

        for start in range(0, n, bs):
            batch_seqs = sequences[start : start + bs]
            batch_muts = mutant_strs[start : start + bs]

            inputs = self._tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with _torch.no_grad():
                outputs = self._model(**inputs)

            # (B, L+2, D) — keep on CPU as numpy for per-variant indexing
            hidden = outputs.last_hidden_state.cpu().numpy()

            for i, mutant_str in enumerate(batch_muts):
                positions = _parse_mutant_positions(mutant_str)
                # +1 because <cls> occupies token index 0
                token_idxs = [p + 1 for p in positions]
                site_embs = hidden[i, token_idxs, :]   # (n_sites, D)
                all_embeddings.append(site_embs.mean(axis=0))  # (D,)

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
            wt_key = self._seq_hash(wt)
            self._load_cache()
            assert self._embedding_cache is not None
            if wt_key in self._embedding_cache:
                self._wt_embedding = self._embedding_cache[wt_key]
                log.debug("WT embedding loaded from cache.")
            else:
                log.info("Computing WT embedding for delta mode.")
                self._wt_embedding = self._embed_sequences([wt])[0]
                self._embedding_cache[wt_key] = self._wt_embedding
                self._save_cache()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants using ESM-2.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain 'mutated_sequence'. For mode='site', must also
            contain 'mutant' (e.g. 'A23V' or 'A23V:G45L').

        Returns
        -------
        np.ndarray
            Shape (n_variants, D), float64.
            D = 320 for esm2_t6_8M, 1280 for esm2_t33_650M, etc.
        """
        sequences = list(df["mutated_sequence"])

        too_long = [s for s in sequences if len(s) > _ESM_MAX_RESIDUES]
        if too_long:
            raise ValueError(
                f"{len(too_long)} sequence(s) exceed the ESM-2 limit of "
                f"{_ESM_MAX_RESIDUES} residues "
                f"(longest: {max(len(s) for s in too_long)}). "
                "Truncation is disabled — shorten or filter sequences before embedding."
            )

        if self.mode == "site":
            return self._transform_site(sequences, df)

        # ---- mean / delta path ------------------------------------------
        keys = [self._seq_hash(seq) for seq in sequences]

        self._load_cache()
        assert self._embedding_cache is not None

        missing_indices = [i for i, k in enumerate(keys) if k not in self._embedding_cache]

        if missing_indices:
            seen: dict[str, str] = {}
            for i in missing_indices:
                k = keys[i]
                if k not in seen:
                    seen[k] = sequences[i]
            unique_keys = list(seen.keys())
            unique_seqs = list(seen.values())
            log.info(
                "Computing ESM-2 (%s) embeddings for %d new sequences.",
                self.model_name, len(unique_seqs),
            )
            new_embs = self._embed_sequences(unique_seqs)
            for k, emb in zip(unique_keys, new_embs):
                self._embedding_cache[k] = emb
            self._save_cache()

        emb = np.stack([self._embedding_cache[k] for k in keys])
        if self._hidden_size is None:
            self._hidden_size = emb.shape[1]

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

    def _transform_site(self, sequences: list[str], df: pd.DataFrame) -> np.ndarray:
        """Site-extraction path for transform(). Requires 'mutant' column."""
        if "mutant" not in df.columns:
            raise ValueError(
                "ESMEncoder(mode='site') requires a 'mutant' column in the DataFrame "
                "(e.g. 'A23V' or 'A23V:G45L'). Check that your dataset CSV includes "
                "the mutant string."
            )
        mutant_strs = list(df["mutant"])

        self._load_site_cache()
        assert self._site_cache is not None

        site_keys = [self._site_key(seq, mut) for seq, mut in zip(sequences, mutant_strs)]
        missing_indices = [i for i, k in enumerate(site_keys) if k not in self._site_cache]

        if missing_indices:
            # Deduplicate by site_key — same (seq, mutant) pair need not be embedded twice.
            seen: dict[str, tuple[str, str]] = {}  # site_key → (seq, mutant_str)
            for i in missing_indices:
                k = site_keys[i]
                if k not in seen:
                    seen[k] = (sequences[i], mutant_strs[i])
            unique_keys = list(seen.keys())
            unique_seqs = [seen[k][0] for k in unique_keys]
            unique_muts = [seen[k][1] for k in unique_keys]
            log.info(
                "Computing ESM-2 (%s) site embeddings for %d new variants.",
                self.model_name, len(unique_seqs),
            )
            new_embs = self._embed_sequences_site(unique_seqs, unique_muts)
            for k, emb in zip(unique_keys, new_embs):
                self._site_cache[k] = emb
            self._save_site_cache()

        result = np.stack([self._site_cache[k] for k in site_keys])
        if self._hidden_size is None:
            self._hidden_size = result.shape[1]
        return result

    @property
    def n_features(self) -> int:
        if self._hidden_size is not None:
            return self._hidden_size
        _KNOWN_SIZES = {
            "facebook/esm2_t6_8M_UR50D": 320,
            "facebook/esm2_t12_35M_UR50D": 480,
            "facebook/esm2_t30_150M_UR50D": 640,
            "facebook/esm2_t33_650M_UR50D": 1280,
            "facebook/esm2_t36_3B_UR50D": 2560,
        }
        return _KNOWN_SIZES.get(self.model_name, -1)
