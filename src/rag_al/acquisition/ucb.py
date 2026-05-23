from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AbstractAcquisition


class UCBAcquisition(AbstractAcquisition):
    """
    Upper Confidence Bound (UCB) acquisition.

    Balances exploitation and exploration::

        a(x) = μ(x) + β · σ(x)

    Parameters
    ----------
    beta : float
        Exploration weight. Higher values favour high-uncertainty regions.
        Default 1.0.
    """

    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

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
        scores = mu + self.beta * sigma
        batch_size = min(batch_size, len(scores))
        return np.argsort(scores)[::-1][:batch_size]
