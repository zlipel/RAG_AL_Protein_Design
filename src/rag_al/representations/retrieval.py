from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .base import AbstractEncoder
from .plm import ESMEncoder


class RetrievalAugmentedEncoder(AbstractEncoder):
    """
    PLM embeddings augmented with nearest-neighbor label context.

    At each active learning round, ``fit()`` is called with the current
    labeled set.  ``transform()`` then retrieves the k nearest labeled
    neighbors (in PLM embedding space) for each query variant and
    appends retrieval-derived statistics as additional features.

    This encoder is ALWAYS leakage-safe:
    - The kNN index is built ONLY from the labeled set.
    - Labels used for retrieval features come ONLY from ``y_labeled``.
    - The pool's fitness scores are never accessed.

    Appended retrieval features (5 dims)
    --------------------------------------
    [0]  mean fitness of k nearest labeled neighbors
    [1]  std  fitness of k nearest labeled neighbors
    [2]  min  distance to any labeled neighbor (normalized)
    [3]  mean distance to k nearest labeled neighbors (normalized)
    [4]  max  fitness of k nearest labeled neighbors

    Total output dimension: PLM_dim + 5

    Parameters
    ----------
    esm_encoder : ESMEncoder
        A (possibly pre-loaded) ESMEncoder instance.
    n_neighbors : int
        Number of nearest labeled neighbors to retrieve (k).
    """

    def __init__(
        self,
        esm_encoder: ESMEncoder,
        n_neighbors: int = 5,
    ) -> None:
        self._esm = esm_encoder
        self.n_neighbors = n_neighbors

        # Set by fit()
        self._labeled_embeddings: Optional[np.ndarray] = None  # (n_lab, D)
        self._labeled_y: Optional[np.ndarray] = None           # (n_lab,)
        self._knn: Optional[NearestNeighbors] = None
        self._dist_scale: float = 1.0   # normalization: mean inter-point distance

    # ------------------------------------------------------------------
    # AbstractEncoder interface
    # ------------------------------------------------------------------

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """
        Fit the ESM encoder and build the kNN index over labeled embeddings.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Feature columns for labeled variants (no 'fitness').
        y_labeled : np.ndarray
            Fitness scores for labeled variants. Shape: (n_labeled,).
            Used ONLY to populate retrieval feature statistics.
        """
        self._esm.fit(df_labeled, y_labeled)
        self._labeled_embeddings = self._esm.transform(df_labeled)
        self._labeled_y = y_labeled.copy()

        k = min(self.n_neighbors, len(y_labeled))
        self._knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        self._knn.fit(self._labeled_embeddings)

        # Scale distances by median pairwise distance (robust normalization)
        if len(self._labeled_embeddings) >= 2:
            dists, _ = self._knn.kneighbors(self._labeled_embeddings)
            self._dist_scale = max(float(np.median(dists)), 1e-8)
        else:
            self._dist_scale = 1.0

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute PLM embeddings and append retrieval features.

        Parameters
        ----------
        df : pd.DataFrame
            Feature columns for variants to encode (no 'fitness').

        Returns
        -------
        np.ndarray
            Shape (n_variants, D + 5), float64.
        """
        if self._knn is None or self._labeled_y is None:
            raise RuntimeError("Call fit() before transform().")

        plm_embeddings = self._esm.transform(df)       # (N, D)
        retrieval_feats = self._retrieval_features(plm_embeddings)  # (N, 5)
        return np.hstack([plm_embeddings, retrieval_feats])

    # ------------------------------------------------------------------
    # Retrieval feature computation
    # ------------------------------------------------------------------

    def _retrieval_features(self, query_embeddings: np.ndarray) -> np.ndarray:
        """
        For each query embedding, retrieve k labeled neighbors and compute
        summary statistics of their fitness values and distances.

        Parameters
        ----------
        query_embeddings : np.ndarray
            Shape (N, D).

        Returns
        -------
        np.ndarray
            Shape (N, 5): [mean_y, std_y, d_min, d_mean, max_y].
        """
        k = min(self.n_neighbors, len(self._labeled_y))
        dists, idxs = self._knn.kneighbors(query_embeddings, n_neighbors=k)
        # dists: (N, k),  idxs: (N, k)

        neighbor_y = self._labeled_y[idxs]   # (N, k)

        mean_y = neighbor_y.mean(axis=1)
        std_y = neighbor_y.std(axis=1)
        max_y = neighbor_y.max(axis=1)
        d_min = dists.min(axis=1) / self._dist_scale
        d_mean = dists.mean(axis=1) / self._dist_scale

        return np.column_stack([mean_y, std_y, d_min, d_mean, max_y])

    @property
    def n_features(self) -> int:
        base = self._esm.n_features
        if base < 0:
            return -1   # unknown until first forward pass
        return base + 5
