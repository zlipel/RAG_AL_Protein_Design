from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AbstractAcquisition


class DiversityUCBAcquisition(AbstractAcquisition):
    """
    Diversity-penalized UCB acquisition (greedy set-cover variant).

    Iteratively builds the batch by:
    1. Scoring all pool candidates with UCB.
    2. Selecting the highest-scoring candidate.
    3. Penalizing remaining candidates by their maximum cosine similarity
       to any already-selected candidate.
    4. Repeating until the batch is full.

    This encourages selecting a diverse set of high-UCB candidates rather
    than a cluster of similar high-scoring variants.

    Acquisition score (per iteration)::

        a(x) = μ(x) + β·σ(x) − γ · max_{s ∈ selected} cos_sim(x, s)

    Parameters
    ----------
    beta : float
        UCB exploration weight (default 1.0).
    gamma : float
        Diversity penalty strength (default 1.0).
        Set to 0 to recover standard UCB.
    """

    def __init__(self, beta: float = 1.0, gamma: float = 1.0) -> None:
        self.beta = beta
        self.gamma = gamma

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
            Encoded features for pool variants. Required for diversity.
            Shape: (n_pool, D). If None, falls back to standard UCB.
        """
        n_pool = len(mu)
        batch_size = min(batch_size, n_pool)
        ucb_scores = mu + self.beta * sigma

        if pool_X is None or self.gamma == 0.0:
            return np.argsort(ucb_scores)[::-1][:batch_size]

        # Normalize rows for cosine similarity
        norms = np.linalg.norm(pool_X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_norm = pool_X / norms  # (n_pool, D)

        selected: list[int] = []
        available = np.ones(n_pool, dtype=bool)
        # max cosine similarity to any selected candidate
        max_cos_sim = np.zeros(n_pool, dtype=float)

        for _ in range(batch_size):
            # Penalized score
            penalized = ucb_scores - self.gamma * max_cos_sim
            penalized[~available] = -np.inf
            chosen = int(np.argmax(penalized))
            selected.append(chosen)
            available[chosen] = False

            # Update max_cos_sim for remaining candidates
            cos_sim = X_norm @ X_norm[chosen]  # (n_pool,)
            np.maximum(max_cos_sim, cos_sim, out=max_cos_sim)

        return np.array(selected, dtype=int)
