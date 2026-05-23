from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractEncoder(ABC):
    """
    Base class for all sequence encoders.

    The interface mirrors scikit-learn's fit/transform pattern with one
    key constraint: ``fit()`` receives only labeled data (no hidden labels
    in the pool), ensuring leakage-free feature normalization.

    Subclasses must implement ``fit`` and ``transform``.  The DataFrame
    passed to both methods has columns:
        variant_id, mutant, mutated_sequence, wt_sequence
    (no 'fitness' column).
    """

    @abstractmethod
    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """
        Fit any stateful components (e.g., scalers, kNN index) using
        ONLY the currently labeled set.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Feature columns for labeled variants (no 'fitness').
        y_labeled : np.ndarray
            Fitness scores for labeled variants. Shape: (n_labeled,).
            Used by retrieval encoders to store label context.
            Must NOT be stored and leaked to transform of unlabeled data.
        """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode a set of variants into a fixed-length feature matrix.

        Parameters
        ----------
        df : pd.DataFrame
            Feature columns for variants to encode (no 'fitness').

        Returns
        -------
        np.ndarray
            Shape (n_variants, n_features). dtype float64.
        """

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Dimensionality of the output feature vector."""

    def fit_transform(
        self, df_labeled: pd.DataFrame, y_labeled: np.ndarray
    ) -> np.ndarray:
        """Convenience: fit on labeled data and immediately transform it."""
        self.fit(df_labeled, y_labeled)
        return self.transform(df_labeled)
