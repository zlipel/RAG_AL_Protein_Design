from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import AbstractAcquisition


class RetrievalUCBAcquisition(AbstractAcquisition):
    """
    Retrieval-augmented UCB acquisition.

    Augments the standard UCB score with a retrieval-derived term R(x)
    computed from the k nearest labeled neighbors in feature space::

        a(x) = μ(x) + β·σ(x) + λ·R(x)

    where::

        R(x) = mean fitness of k nearest labeled neighbors

    This is computed from ``labeled_X`` and ``labeled_y`` only — it never
    accesses hidden pool fitness values (leakage-safe).

    The retrieval signal biases acquisition toward regions of the landscape
    where nearby experimentally measured variants have high fitness.

    Parameters
    ----------
    beta : float
        UCB exploration weight (default 1.0).
    lam : float
        Retrieval score weight λ (default 0.5).
    n_neighbors : int
        Number of nearest labeled neighbors to retrieve (default 5).
    """

    def __init__(
        self,
        beta: float = 1.0,
        lam: float = 0.5,
        n_neighbors: int = 5,
    ) -> None:
        self.beta = beta
        self.lam = lam
        self.n_neighbors = n_neighbors

    def select_batch(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        batch_size: int,
        *,
        pool_X: Optional[np.ndarray] = None,
        labeled_X: Optional[np.ndarray] = None,
        labeled_y: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        pool_X : np.ndarray
            Encoded features for pool variants. Shape: (n_pool, D). Required.
        labeled_X : np.ndarray
            Encoded features for labeled variants. Shape: (n_lab, D). Required.
        labeled_y : np.ndarray
            Fitness scores for labeled variants. Shape: (n_lab,). Required.
            Only labeled scores — never pool scores.
        """
        if pool_X is None or labeled_X is None or labeled_y is None:
            # Graceful fallback to standard UCB if required arrays are missing
            scores = mu + self.beta * sigma
            batch_size = min(batch_size, len(scores))
            return np.argsort(scores)[::-1][:batch_size]

        retrieval_scores = self._retrieval_scores(pool_X, labeled_X, labeled_y)
        scores = mu + self.beta * sigma + self.lam * retrieval_scores
        batch_size = min(batch_size, len(scores))
        return np.argsort(scores)[::-1][:batch_size]

    def _retrieval_scores(
        self,
        pool_X: np.ndarray,
        labeled_X: np.ndarray,
        labeled_y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute R(x) = mean fitness of k nearest labeled neighbors.

        Parameters
        ----------
        pool_X : np.ndarray
            Shape (n_pool, D).
        labeled_X : np.ndarray
            Shape (n_lab, D).
        labeled_y : np.ndarray
            Shape (n_lab,) — labeled fitness scores ONLY.

        Returns
        -------
        np.ndarray
            Shape (n_pool,). Mean neighbor fitness for each pool variant.
        """
        k = min(self.n_neighbors, len(labeled_y))
        knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
        knn.fit(labeled_X)
        _, idxs = knn.kneighbors(pool_X)      # (n_pool, k)
        neighbor_y = labeled_y[idxs]           # (n_pool, k)
        return neighbor_y.mean(axis=1)          # (n_pool,)
