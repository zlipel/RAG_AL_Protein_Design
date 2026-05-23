# rag_al/loop

The active learning loop and metric computation. This is the orchestration
layer that connects the data, encoder, surrogate, and acquisition function
into a complete experiment.

---

## runner.py

### run_al_loop(dataset, encoder, surrogate, acquisition, n_rounds, batch_size, seed, log)

The main AL loop function. Runs n_rounds of pool-based active learning
and returns a DataFrame of per-round metrics.

### Arguments
dataset     — ALDataset (initialized with n_init labeled variants)
encoder     — AbstractEncoder (will be fit each round on the labeled set)
surrogate   — AbstractSurrogate (will be fit each round on labeled features)
acquisition — AbstractAcquisition (selects the batch each round)
n_rounds    — number of AL rounds to run
batch_size  — number of variants to acquire per round
seed        — random seed for acquisition tie-breaking (passed as rng)
log         — optional logger; defaults to module logger if None

### Per-round procedure
Each round executes these steps in order:

Step 1 — Encode labeled set
    encoder.fit(dataset.labeled_df, dataset.labeled_y) is called to fit
    any stateful components (StandardScaler, kNN index, etc.) using only
    labeled data. Then encoder.transform(dataset.labeled_df) produces
    X_labeled, the feature matrix for labeled variants.

Step 2 — Encode pool
    encoder.transform(dataset.pool_df) produces X_pool. The encoder has
    been fit on labeled data; it now applies those statistics (e.g.,
    scaler means and variances) to the pool without refitting.

Step 3 — Fit surrogate
    surrogate.fit(X_labeled, dataset.labeled_y) retrains the surrogate
    from scratch on all currently labeled data.

Step 4 — Predict for pool
    mu, sigma = surrogate.predict(X_pool) produces predicted mean and
    uncertainty for all pool variants.

Step 5 — Select batch
    acquisition.select_batch(mu, sigma, batch_size, pool_X=X_pool,
    labeled_X=X_labeled, labeled_y=dataset.labeled_y, rng=rng) returns
    local pool indices of the selected batch.

Step 6 — Reveal
    Before calling reveal(), the batch sequences are read from the dataset
    (for metric computation). Then dataset.reveal(selected_local) moves
    the selected variants from the pool to the labeled set.

Step 7 — Compute metrics
    compute_round_metrics() is called with the now-updated labeled set.
    Results are appended to a list of dicts, which is converted to a
    DataFrame at the end.

### Leakage enforcement in the loop
- encoder.fit() receives only dataset.labeled_df (no fitness) and
  dataset.labeled_y (labeled fitness only).
- dataset.pool_df contains no fitness column.
- acquisition.select_batch() receives only labeled_y (labeled scores).
- reveal() is called BEFORE reading batch fitness for metrics.
- global_optimum and top_k_global_indices are precomputed once before
  the loop and used only by metrics — never by any acquisition function.

### Output DataFrame columns
round                — round index (0-based)
n_labeled            — total labeled set size after this round
best_fitness         — max fitness in labeled set after reveal
simple_regret        — global_optimum - best_fitness
topk10_recall        — fraction of top-10 variants acquired so far
topk50_recall        — fraction of top-50 variants acquired so far
batch_mean_fitness   — mean fitness of the newly acquired batch
batch_diversity      — mean pairwise Hamming distance among batch sequences
mean_dist_wt         — mean normalized Hamming distance from WT

---

## metrics.py

Individual metric functions, plus a convenience aggregator.

### best_fitness(labeled_y) -> float
    max(labeled_y)
    The best fitness observed so far.

### simple_regret(labeled_y, global_optimum) -> float
    global_optimum - best_fitness(labeled_y)
    How far we are from the global best. Decreases as the AL loop succeeds.
    A well-performing strategy should drive this toward zero quickly.

### topk_recall(labeled_indices, top_k_indices) -> float
    |labeled_indices ∩ top_k_indices| / k
    Fraction of the top-k variants (by true fitness) that have been
    acquired. Measures how well the strategy identifies truly elite variants.
    Computed separately for k=10 and k=50.

### batch_mean_fitness(batch_y) -> float
    mean(batch_y)
    Average fitness of the most recently acquired batch. A high-performing
    strategy should acquire increasingly high-fitness batches over rounds.

### batch_diversity(batch_sequences) -> float
    Mean pairwise normalized Hamming distance among the batch sequences.
    Computed over all pairs i < j. Returns 0.0 for batches of size <= 1.
    
    Sequences of different lengths are right-padded with a null character
    (treated as a mismatch), so the function handles variable-length
    sequences gracefully.
    
    Value in [0, 1]. 0 = all identical, 1 = no shared positions across
    any pair. This measures whether the batch is a cluster of similar
    variants or a spread across sequence space.

### mean_dist_from_wt(batch_sequences, wt_sequence) -> float
    Mean normalized Hamming distance from the wild-type sequence.
    Measures how far into sequence space the strategy is venturing.
    A greedy strategy close to the optimum will have low WT distance
    if high-fitness variants are near-WT; exploration strategies may
    venture further.

### compute_round_metrics(...)
    Convenience aggregator that calls all metric functions and returns
    a flat dict suitable for appending to the results DataFrame.
    This is what run_al_loop() calls at the end of each round.
