# rag_al/surrogates

Surrogate models that predict fitness mean (μ) and uncertainty (σ) for
unlabeled pool variants. The surrogate is the core of the Bayesian
optimization / active learning strategy — it is refitted at every round
on the current labeled set.

---

## base.py

### AbstractSurrogate
All surrogates inherit from this base class. Two methods must be implemented:

fit(X, y)
    Fit the model on labeled data.
    X : np.ndarray, shape (n_labeled, n_features) — encoded feature matrix
    y : np.ndarray, shape (n_labeled,)            — fitness scores

predict(X) -> (mu, sigma)
    Predict mean and uncertainty for query variants.
    X   : np.ndarray, shape (n_query, n_features)
    mu  : np.ndarray, shape (n_query,) — predicted mean fitness
    sigma : np.ndarray, shape (n_query,) — predicted uncertainty, always >= 0

The surrogate never sees the pool fitness labels directly. It receives
encoded features (X) from the encoder and fitness labels (y) from
ALDataset.labeled_y only.

---

## random_forest.py

### RFSurrogate
Scikit-learn RandomForestRegressor wrapped to produce both mean and
uncertainty predictions.

### Why Random Forest?
- No GPU required — runs on CPU on both laptop and cluster
- Scales well with n_estimators via joblib parallelism (n_jobs=-1)
- Uncertainty estimate is easy and reliable: the standard deviation of
  per-tree predictions across the ensemble
- Well-calibrated for moderate-dimensional feature spaces
- Fast fit and predict even for thousands of labeled variants
- No hyperparameter tuning required to get reasonable starting results

### Uncertainty estimation
For each query point x, all n_estimators trees produce a prediction.
The reported uncertainty is:
    sigma(x) = std( [tree_1(x), tree_2(x), ..., tree_K(x)] )

This is sometimes called "jackknife uncertainty" or "forest variance".
It reflects disagreement among trees — high sigma indicates a region
of feature space that is sparsely covered by training data, which is
exactly what we want from an exploration signal.

### Parameters
n_estimators   — number of trees (default: 100, set via BenchmarkConfig)
random_state   — seed for reproducibility (set to run seed in benchmark.py)
n_jobs         — parallelism; -1 uses all available cores

### Notes
- sigma is always >= 0 by construction (std cannot be negative)
- sigma can be exactly 0 if all trees agree (e.g., a point that is
  identical to a training point) — this is correct behavior
- The surrogate is re-fit from scratch at every AL round (no warm-starting)
  because the labeled set changes in composition each round

### Potential extensions
The AbstractSurrogate interface makes it straightforward to swap in other
models later:
- Gaussian Process (GPyTorch) — exact uncertainty, O(n^3) cost
- Neural network ensemble — more expressive, higher compute cost
- Bayesian linear regression — fast, linear model baseline
