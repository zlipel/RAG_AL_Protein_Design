from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import validate_schema

# Columns exposed to the AL loop (no fitness leak)
_FEATURE_COLS = ("variant_id", "mutant", "mutated_sequence", "wt_sequence")


class LeakageError(RuntimeError):
    """
    Raised when code attempts to access hidden fitness labels through
    an unauthorized path (i.e., not via ``reveal()`` or metric helpers).
    """


class ALDataset:
    """
    Pool-based active learning dataset with strict leakage enforcement.

    The full label array is stored privately and is never exposed directly.
    The only way to move labels from hidden → labeled is via ``reveal()``.
    Metric helpers (``global_optimum``, ``top_k_global_indices``) are
    intentionally public because the leakage rules explicitly permit the
    full label array to be accessed for evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataset DataFrame (all columns including 'fitness').
    n_init : int
        Number of variants to reveal as the initial labeled set.
    seed : int
        Random seed for reproducible initialization.
    """

    def __init__(self, df: pd.DataFrame, n_init: int, seed: int = 0) -> None:
        validate_schema(df)

        self._df: pd.DataFrame = df.reset_index(drop=True).copy()
        # Private fitness array — name-mangled to make accidental access obvious
        self.__fitness: np.ndarray = self._df["fitness"].to_numpy(dtype=float)
        self._labeled_mask: np.ndarray = np.zeros(len(self._df), dtype=bool)

        if n_init > len(self._df):
            raise ValueError(
                f"n_init={n_init} exceeds dataset size {len(self._df)}"
            )

        rng = np.random.default_rng(seed)
        init_idx = rng.choice(len(self._df), size=n_init, replace=False)
        self._labeled_mask[init_idx] = True

        # Precompute for metric helpers (full-label access is allowed here)
        self._global_optimum: float = float(self.__fitness.max())

    # ------------------------------------------------------------------
    # Labeled set (safe for the AL loop)
    # ------------------------------------------------------------------

    @property
    def labeled_df(self) -> pd.DataFrame:
        """Feature columns (no fitness) for currently labeled variants."""
        return self._df.loc[self._labeled_mask, list(_FEATURE_COLS)].copy()

    @property
    def labeled_y(self) -> np.ndarray:
        """
        Fitness scores for currently labeled variants.

        Shape: (n_labeled,)
        """
        return self.__fitness[self._labeled_mask].copy()

    @property
    def labeled_indices(self) -> np.ndarray:
        """Global indices (into the full dataset) of labeled variants."""
        return np.where(self._labeled_mask)[0]

    # ------------------------------------------------------------------
    # Unlabeled pool (safe for the AL loop — no fitness exposed)
    # ------------------------------------------------------------------

    @property
    def pool_df(self) -> pd.DataFrame:
        """Feature columns (no fitness) for currently unlabeled pool variants."""
        return self._df.loc[~self._labeled_mask, list(_FEATURE_COLS)].copy()

    @property
    def pool_indices(self) -> np.ndarray:
        """
        Global indices (into the full dataset) of unlabeled pool variants.

        Acquisition functions work with *local* pool indices (0 … n_pool-1).
        Use this array to map back to global indices before calling ``reveal()``.
        """
        return np.where(~self._labeled_mask)[0]

    # ------------------------------------------------------------------
    # Reveal — the only authorized path to expose hidden labels
    # ------------------------------------------------------------------

    def reveal(self, pool_local_indices: np.ndarray | list[int]) -> None:
        """
        Reveal fitness labels for a batch of pool variants.

        Parameters
        ----------
        pool_local_indices : array-like of int
            *Local* indices into the current pool (0-based). These are
            converted to global dataset indices internally.
        """
        pool_local_indices = np.asarray(pool_local_indices, dtype=int)
        global_indices = self.pool_indices[pool_local_indices]
        if self._labeled_mask[global_indices].any():
            raise LeakageError(
                "Some requested indices are already labeled. "
                "This should not happen — check the acquisition function."
            )
        self._labeled_mask[global_indices] = True

    # ------------------------------------------------------------------
    # Metric helpers (full-label access is allowed for evaluation)
    # ------------------------------------------------------------------

    @property
    def global_optimum(self) -> float:
        """
        Best fitness score in the full dataset.

        Intended for computing simple regret after each round.
        Must NOT be used inside acquisition functions.
        """
        return self._global_optimum

    def top_k_global_indices(self, k: int) -> np.ndarray:
        """
        Global indices of the top-*k* variants ranked by fitness.

        Used for computing top-k recall. Must NOT be used inside
        acquisition functions.

        Parameters
        ----------
        k : int
            Number of top variants to return.

        Returns
        -------
        np.ndarray
            Shape (k,), global indices in descending fitness order.
        """
        return np.argsort(self.__fitness)[::-1][:k]

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    @property
    def n_labeled(self) -> int:
        return int(self._labeled_mask.sum())

    @property
    def n_pool(self) -> int:
        return int((~self._labeled_mask).sum())

    @property
    def n_total(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"ALDataset(n_total={self.n_total}, n_labeled={self.n_labeled}, "
            f"n_pool={self.n_pool})"
        )
