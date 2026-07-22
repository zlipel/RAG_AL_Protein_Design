"""
Gaussian Process surrogate — single-task ExactGP with Matérn 3/2 kernel.

Training protocol: per-round warm-start from previous round's hyperparameters,
MLL patience stopping. No validation split — train on all labeled data.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import AbstractSurrogate

log = logging.getLogger(__name__)

try:
    import torch
    import gpytorch
except ImportError as _e:
    raise ImportError(
        "GPSurrogate requires gpytorch and torch. Install with:\n"
        "  pip install gpytorch torch"
    ) from _e


class _GPRegressionModel(gpytorch.models.ExactGP):
    """Single-task ExactGP: ConstantMean + ScaleKernel(Matérn 3/2).

    ard=True gives the Matérn kernel one lengthscale per input dimension
    (ard_num_dims = feature count) instead of a single shared lengthscale.
    """

    def __init__(
        self,
        train_x: "torch.Tensor",
        train_y: "torch.Tensor",
        likelihood: "gpytorch.likelihoods.GaussianLikelihood",
        ard: bool = False,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ard_num_dims = train_x.size(-1) if ard else None
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
        )

    def forward(self, x: "torch.Tensor") -> "gpytorch.distributions.MultivariateNormal":
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPSurrogate(AbstractSurrogate):
    """
    Single-task GP surrogate with round-to-round warm start.

    X is standardized per-dimension; y is standardized before fitting.
    Predictions are un-standardized back to the original fitness scale.

    Hyperparameters (lengthscale, outputscale, noise) are carried over
    between fit() calls via _prev_state, so the optimizer warm-starts
    from the previous round's converged values rather than from scratch.

    Parameters
    ----------
    n_iter : int
        Max Adam steps per round. With warm-starting, typically exits
        well before this via patience.
    lr : float
        Adam learning rate for MLL maximization.
    patience : int
        Number of patience checks before early stopping. Each check
        covers check_interval=20 steps, so minimum steps before stopping
        is patience * check_interval.
    tol : float
        MLL improvement threshold to reset patience counter.
    device : str or None
        'cuda', 'cpu', or None (auto-detect CUDA; falls back to CPU).
    predict_batch_size : int
        Pool rows scored per forward pass in predict(). Caps predict memory
        at O(batch * n_train) instead of the O(n_pool^2) exact-GP variance
        transient on large pools.
    ard : bool
        If True, use per-dimension lengthscales (ARD) in the Matérn kernel so
        the GP learns each feature's relevance. Adds one hyperparameter per
        input dim — well-suited to low-dim interpretable features; on very
        high-dim inputs (e.g. 1280-d PLM) the MLL fit is over-parameterized.
    """

    _CHECK_INTERVAL: int = 20  # steps between patience checks

    def __init__(
        self,
        n_iter: int = 200,
        lr: float = 0.1,
        patience: int = 3,
        tol: float = 1e-4,
        device: Optional[str] = None,
        predict_batch_size: int = 4096,
        ard: bool = False,
    ) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.patience = patience
        self.tol = tol
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.predict_batch_size = predict_batch_size
        self.ard = ard

        self._model: Optional[_GPRegressionModel] = None
        self._likelihood: Optional["gpytorch.likelihoods.GaussianLikelihood"] = None
        self._prev_state: Optional[dict] = None

        self._X_mean: Optional[np.ndarray] = None
        self._X_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

    # ------------------------------------------------------------------
    # AbstractSurrogate interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit GP on labeled data.

        On first call: cold-start from gpytorch defaults.
        On subsequent calls: warm-start from the previous round's
        hyperparameters, which typically cuts step count significantly.
        """
        # Per-dim X standardization + y standardization
        self._X_mean = X.mean(axis=0)
        self._X_std = X.std(axis=0).clip(min=1e-8)
        self._y_mean = float(y.mean())
        y_std = float(y.std())
        self._y_std = y_std if y_std > 1e-8 else 1.0

        Xs = torch.tensor(
            (X - self._X_mean) / self._X_std, dtype=torch.float32
        ).to(self.device)
        ys = torch.tensor(
            (y - self._y_mean) / self._y_std, dtype=torch.float32
        ).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        model = _GPRegressionModel(Xs, ys, likelihood, ard=self.ard).to(self.device)

        if self._prev_state is not None:
            model.load_state_dict(self._prev_state["model"])
            likelihood.load_state_dict(self._prev_state["likelihood"])
            log.debug("GPSurrogate: warm-starting from previous round.")

        model.train()
        likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        best_mll = float("-inf")
        patience_count = 0

        for step in range(self.n_iter):
            opt.zero_grad()
            loss = -mll(model(Xs), ys)
            loss.backward()
            opt.step()

            if (step + 1) % self._CHECK_INTERVAL == 0:
                current_mll = -loss.item()
                if current_mll - best_mll > self.tol:
                    best_mll = current_mll
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        log.debug(
                            "GPSurrogate: early stop at step %d (patience=%d).",
                            step + 1, self.patience,
                        )
                        break

        model.eval()
        likelihood.eval()
        self._model = model
        self._likelihood = likelihood

        # Save for next round's warm start
        self._prev_state = {
            "model": {k: v.clone() for k, v in model.state_dict().items()},
            "likelihood": {k: v.clone() for k, v in likelihood.state_dict().items()},
        }

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std for pool variants.

        Returns mu and sigma in the original (un-standardized) fitness scale.
        sigma is always ≥ 0 (gpytorch guarantees this via the GaussianLikelihood).
        """
        if self._model is None or self._likelihood is None:
            raise RuntimeError("Call fit() before predict().")

        n = X.shape[0]
        if n == 0:
            empty = np.empty(0, dtype=float)
            return empty, empty
        # Score the pool in chunks so peak memory stays O(bs * n_train) rather
        # than the O(n_pool^2) exact-GP variance transient. Marginals only, so
        # chunking is exact for the mean and per-point sigma we consume.
        bs = max(1, min(self.predict_batch_size, n))

        mus: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        # Use the latent posterior p(f*|x*,X,y), not likelihood(model(x)) which adds
        # observation noise. AL acquisition cares about epistemic uncertainty only.
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, n, bs):
                Xs = torch.tensor(
                    (X[start : start + bs] - self._X_mean) / self._X_std,
                    dtype=torch.float32,
                ).to(self.device)
                pred = self._model(Xs)
                mus.append(pred.mean.cpu().numpy())
                sigmas.append(pred.stddev.cpu().numpy())

        mu = np.concatenate(mus) * self._y_std + self._y_mean
        sigma = np.concatenate(sigmas) * self._y_std
        return mu, sigma
