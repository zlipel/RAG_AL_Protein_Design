from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import AbstractSurrogate


class RFSurrogate(AbstractSurrogate):
    """
    Random Forest surrogate model.

    Predicts mean fitness (μ) and uncertainty (σ) for pool variants.
    Uncertainty is estimated as the standard deviation of per-tree
    predictions across the ensemble, following the approach used in
    random forest-based Bayesian optimization.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs for forest training and prediction.
        -1 uses all available cores.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 0,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._rf: RandomForestRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the random forest on labeled data.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_labeled, n_features).
        y : np.ndarray
            Shape (n_labeled,).
        """
        self._rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._rf.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std of per-tree predictions.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_query, n_features).

        Returns
        -------
        mu : np.ndarray
            Shape (n_query,). Mean over tree predictions.
        sigma : np.ndarray
            Shape (n_query,). Std dev over tree predictions; always ≥ 0.
        """
        if self._rf is None:
            raise RuntimeError("Call fit() before predict().")

        # Collect per-tree predictions: shape (n_estimators, n_query)
        tree_preds = np.array(
            [tree.predict(X) for tree in self._rf.estimators_]
        )
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        return mu, sigma
