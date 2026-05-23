from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractSurrogate(ABC):
    """
    Base class for all surrogate models.

    A surrogate is fit on the currently labeled set and predicts
    mean fitness (μ) and uncertainty (σ) for unlabeled pool variants.

    Subclasses must implement ``fit`` and ``predict``.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the surrogate on labeled data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for labeled variants. Shape: (n_labeled, n_features).
        y : np.ndarray
            Fitness scores. Shape: (n_labeled,).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and uncertainty for query variants.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for query variants. Shape: (n_query, n_features).

        Returns
        -------
        mu : np.ndarray
            Predicted mean fitness. Shape: (n_query,).
        sigma : np.ndarray
            Predicted uncertainty (std dev). Shape: (n_query,). Must be ≥ 0.
        """
