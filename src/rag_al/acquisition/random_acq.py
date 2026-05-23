from __future__ import annotations

from typing import Optional

import numpy as np

from .base import AbstractAcquisition


class RandomAcquisition(AbstractAcquisition):
    """
    Random acquisition baseline.

    Selects a uniformly random batch from the unlabeled pool.
    No surrogate predictions are used.
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
        if rng is None:
            rng = np.random.default_rng()
        n_pool = len(mu)
        batch_size = min(batch_size, n_pool)
        return rng.choice(n_pool, size=batch_size, replace=False)
