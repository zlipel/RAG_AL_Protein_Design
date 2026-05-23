from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractAcquisition(ABC):
    """
    Base class for all acquisition functions.

    ``select_batch`` is the single required method.  It receives the
    surrogate's predicted mean (mu) and uncertainty (sigma) for all
    pool variants, plus optional extra arrays needed by retrieval-based
    variants, and returns the LOCAL pool indices of the selected batch.

    Leakage contract
    ----------------
    Acquisition functions MUST NOT access hidden fitness labels.
    Only ``mu``, ``sigma``, and the optional ``pool_X`` / ``labeled_X`` /
    ``labeled_y`` keyword arguments (which contain ONLY currently labeled
    data) are permitted inputs.
    """

    @abstractmethod
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
        Select a batch of pool variants to acquire.

        Parameters
        ----------
        mu : np.ndarray
            Predicted mean fitness. Shape: (n_pool,).
        sigma : np.ndarray
            Predicted uncertainty.  Shape: (n_pool,).
        batch_size : int
            Number of variants to select.
        pool_X : np.ndarray, optional
            Encoded feature matrix for pool variants. Shape: (n_pool, D).
            Required by diversity-based acquisitions.
        labeled_X : np.ndarray, optional
            Encoded feature matrix for labeled variants. Shape: (n_lab, D).
            Required by retrieval-based acquisitions.
        labeled_y : np.ndarray, optional
            Fitness scores for labeled variants. Shape: (n_lab,).
            Required by retrieval-based acquisitions.
        rng : np.random.Generator, optional
            For reproducible tie-breaking / random acquisition.

        Returns
        -------
        np.ndarray
            LOCAL pool indices (0-based) of the selected variants.
            Shape: (batch_size,).
        """
