from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AbstractAcquisition


class GreedyAcquisition(AbstractAcquisition):
    """
    Greedy exploitation: select the variants with the highest predicted mean.

    Acquisition score::

        a(x) = μ(x)
    """

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
        batch_size = min(batch_size, len(mu))
        # argsort descending; take top batch_size
        return np.argsort(mu)[::-1][:batch_size]
