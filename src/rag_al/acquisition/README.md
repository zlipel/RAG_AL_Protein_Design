# rag_al/acquisition

Acquisition functions that decide which unlabeled pool variants to measure
next. Each acquisition function receives surrogate predictions (μ, σ) and
returns the local pool indices of the selected batch.

All acquisition functions are leakage-safe: they access labeled fitness
values only through the explicitly passed labeled_y argument, and they
never access hidden pool fitness scores.

---

## base.py

### AbstractAcquisition
All acquisition functions inherit from this base class. One method must
be implemented:

select_batch(mu, sigma, batch_size, *, pool_X, labeled_X, labeled_y, rng)
    Select a batch of pool variants to acquire.

    mu          — predicted mean fitness, shape (n_pool,)
    sigma       — predicted uncertainty, shape (n_pool,)
    batch_size  — number of variants to select
    pool_X      — encoded pool features, shape (n_pool, D); optional,
                  required by diversity-based methods
    labeled_X   — encoded labeled features, shape (n_lab, D); optional,
                  required by retrieval-based methods
    labeled_y   — labeled fitness scores, shape (n_lab,); optional,
                  required by retrieval-based methods
    rng         — numpy random generator for reproducible tie-breaking

    Returns: np.ndarray of local pool indices, shape (batch_size,)

The optional keyword arguments use a keyword-only interface so that
simple acquisition functions (random, greedy, UCB) do not need to
handle arrays they don't use, while complex ones (retrieval UCB) can
declare exactly what they need.

---

## random_acq.py

### RandomAcquisition
Selects a uniformly random batch from the pool. Uses the rng argument
for reproducibility — with the same seed, the same batch is selected.

This is the baseline acquisition function. A method that cannot beat
random acquisition on a DMS landscape provides no useful signal.

No surrogate predictions are used (mu and sigma are ignored).

Score: none — uniform random selection

---

## greedy.py

### GreedyAcquisition
Selects the batch_size variants with the highest predicted mean fitness.

    a(x) = μ(x)

This is pure exploitation: it always picks the variants the surrogate
currently thinks are best. It can get stuck if the surrogate has poor
coverage and confidently mispredicts, but it is a strong baseline when
the surrogate is accurate.

Score: descending rank by μ

---

## ucb.py

### UCBAcquisition
Upper Confidence Bound acquisition. Balances exploitation (high μ) with
exploration (high σ):

    a(x) = μ(x) + β · σ(x)

The β parameter (ucb_beta in BenchmarkConfig) controls the trade-off.
Higher β favors exploration (picking uncertain variants); lower β favors
exploitation (picking predicted-best variants). β = 1.0 is the default.

UCB has a well-studied theoretical basis in bandit problems and is the
most commonly used acquisition function in Bayesian optimization.

Score: descending rank by μ + β·σ

---

## diversity_ucb.py

### DiversityUCBAcquisition
A greedy set-cover variant of UCB that penalizes batches of similar
variants. This encourages selecting a spread of high-UCB candidates
rather than a cluster of nearly identical sequences.

### Algorithm (iterative)
1. Compute UCB scores for all pool variants.
2. Select the highest-UCB variant as the first batch member.
3. Penalize all remaining variants by their max cosine similarity to
   any already-selected variant.
4. Repeat: pick the variant with the highest (UCB - γ · max_cos_sim).
5. Continue until the batch is full.

    a(x) = μ(x) + β·σ(x) − γ · max_{s ∈ selected} cos_sim(x, s)

γ = 0 recovers standard UCB. γ = 1.0 (default) applies a meaningful
diversity penalty. Cosine similarity is computed in the encoder's
feature space (pool_X), so the notion of "similar" depends on which
representation is used.

### Why this matters
Standard UCB on a batch often selects many near-duplicate variants
because similar sequences tend to have correlated μ and σ. The diversity
penalty spreads the batch across different regions of sequence space,
giving more information per round.

pool_X is required. If not provided, falls back to standard UCB.

---

## retrieval_ucb.py

### RetrievalUCBAcquisition
Augments UCB with a retrieval-derived term R(x) computed from the current
labeled set:

    a(x) = μ(x) + β·σ(x) + λ·R(x)

    R(x) = mean fitness of k nearest labeled neighbors (in feature space)

### How R(x) is computed
1. Build a NearestNeighbors index over labeled_X (labeled feature vectors).
2. For each pool variant, find its k nearest labeled neighbors.
3. R(x) = mean of their labeled fitness scores.

The kNN index is built fresh at every call to select_batch(), using only
labeled_X and labeled_y — never pool fitness values.

### Intuition
R(x) provides a local fitness estimate from nearby experimentally measured
variants. This is complementary to the surrogate's global fit (μ):
- In early rounds when the surrogate is poorly calibrated, R(x) provides
  a data-driven signal from the labeled neighborhood.
- In later rounds, R(x) continues to refine the acquisition signal by
  incorporating the most recently revealed high-fitness regions.

### Parameters
beta   — UCB exploration weight (default 1.0, shared with UCBAcquisition)
lam    — retrieval score weight λ (default 0.5, set by retrieval_lambda)
n_neighbors — k for kNN (default 5, set by n_neighbors in config)

### Fallback
If pool_X, labeled_X, or labeled_y are None, falls back to standard UCB.
This allows the same acquisition object to be used even in edge cases where
feature arrays are unavailable.

### Distinction from RetrievalAugmentedEncoder
There are two ways to incorporate retrieval into this pipeline:

1. Representation-level (plm_retrieval representation):
   RetrievalAugmentedEncoder appends kNN label features to the PLM embedding
   before the surrogate sees it. The surrogate learns to use this signal
   as part of its prediction.

2. Acquisition-level (retrieval_ucb acquisition):
   RetrievalUCBAcquisition adds R(x) directly to the acquisition score
   after the surrogate has made its predictions. The surrogate is unaware
   of the retrieval signal.

Both are valid. The benchmark compares them (and their combination) directly.
