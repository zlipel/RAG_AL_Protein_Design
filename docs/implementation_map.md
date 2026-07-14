# Implementation Map
## RAG-AL Protein Design — Module-by-Module Reference

For each module: main classes/functions, inputs/outputs, where fitness labels
are accessed, whether label access is authorized, leakage risks, and tests
to run. See `docs/bugs.md` for specific bugs found during this audit.

---

## Audit Questions (docs/audit_plan.md)

| # | Question | Status | Detail |
|---|----------|--------|--------|
| 1 | Labels revealed only after selection? | ✅ | Fixed — Bug #1 |
| 2 | Surrogate trains on labeled only? | ✅ | `surrogate.fit(X_labeled, y_labeled)` |
| 3 | Retrieval uses labeled set only? | ✅ | `_labeled_y` from `y_labeled` in `fit()` |
| 4 | Representation retrieval avoids self-label leak? | ✅ | Fixed — Bug #3 |
| 5 | Acquisition functions receive hidden labels? | ✅ | Only `mu`, `sigma`, `labeled_X`, `labeled_y` |
| 6 | Embeddings computed without fitness labels? | ✅ | `ESMEncoder.fit()` ignores `y_labeled` |
| 7 | Indices, configs, metrics saved per run? | ✅ | Fixed — Gap #1 |
| 8 | Random seeds controlled? | ✅ | `ALDataset`, `run_al_loop` rng, RF `random_state` |

---

## Data Layer

---

### `src/rag_al/data/schema.py`

**Main functions**
- `validate_schema(df: pd.DataFrame) -> None`

**Inputs / Outputs**
- In: full DataFrame (all columns including fitness)
- Out: None; raises `SchemaError` on failure

**Fitness label access**
- YES — checks that `fitness` column is numeric and non-null. Does not read
  individual values; only checks dtype and NaN presence.

**Authorized:** YES — pre-loop validation; no AL logic here.

**Leakage risks:** NONE — purely structural.

**Tests**
- Valid CSV passes without error
- Missing column → `SchemaError` naming the column
- Non-numeric `fitness` → `SchemaError`
- NaN in any required column → `SchemaError`
- Non-standard AA character in sequence → `SchemaError`

---

### `src/rag_al/data/loader.py`

**Main functions**
- `load_dataset(path) -> pd.DataFrame`

**Inputs / Outputs**
- In: path to curated CSV
- Out: validated DataFrame, integer index, string columns cast to `str`

**Fitness label access**
- YES — returned DataFrame contains the `fitness` column. Downstream,
  `ALDataset` extracts and hides it.

**Authorized:** YES — pre-loop loading; protection begins in `ALDataset`.

**Leakage risks:** NONE at this layer.

**Tests**
- `FileNotFoundError` for missing file
- `SchemaError` propagated from `validate_schema`
- String columns correctly cast
- Returns clean integer index

---

### `src/rag_al/data/al_dataset.py`  ⭐ CRITICAL

**Main classes**
- `LeakageError(RuntimeError)`
- `ALDataset`

**`ALDataset.__init__(df, n_init, seed)`**
- In: validated DataFrame, init size, seed
- Out: `ALDataset` with `n_init` randomly labeled variants
- Label access: YES — extracts `__fitness` (name-mangled); precomputes
  `_global_optimum`

**`labeled_df` (property)**
- Out: DataFrame with `_FEATURE_COLS` only — NO fitness column ✅

**`labeled_y` (property)**
- Out: `np.ndarray` of fitness scores for labeled variants only ✅
- This is the AUTHORIZED path to labeled fitness during the loop.

**`pool_df` (property)**
- Out: DataFrame with `_FEATURE_COLS` only — pool fitness NEVER exposed ✅

**`pool_indices` (property)**
- Out: global indices of unlabeled pool variants
- No label access ✅

**`reveal(pool_local_indices)`**
- In: local pool indices (0-based into current pool)
- Out: updates `_labeled_mask` in place
- This is the ONLY authorized path to move hidden → labeled
- Raises `LeakageError` if any index is already labeled

**`global_optimum` (property)**
- Label access: YES — returns precomputed max of full `__fitness`
- Authorized: YES, for metric computation only

**`top_k_global_indices(k)`**
- Label access: YES — argsorts full `__fitness`
- Authorized: YES, for metric computation only

**Authorized:** Partially.
- `__fitness`: accessible only via `reveal`, `global_optimum`, `top_k_global_indices`
- `labeled_y`: authorized path during loop
- Pool fitness: never exposed

**Leakage risks**
- **Risk A (Bug #2):** `_df` is single-underscore and contains `fitness`.
  `dataset._df["fitness"]` would expose all labels. Runner currently reads
  `_df` for sequences only, but this is fragile. → Add `wt_sequence` property
  and `get_sequences()` method.
- **Risk B:** `_labeled_mask` is mutable and single-underscore. External code
  could flip all to `True`. Low practical risk.
- **Risk C:** `global_optimum` and `top_k_global_indices` access the full
  label array. Enforced by convention only — no code barrier prevents calling
  them inside an acquisition function.

**Tests**
- `n_init` variants labeled after construction; rest unlabeled
- `labeled_df` has NO `fitness` column
- `pool_df` has NO `fitness` column
- `labeled_y` correct length and values
- `reveal()` moves variants from pool to labeled
- `reveal()` raises `LeakageError` on already-labeled index
- `labeled_y` correct values AFTER reveal (catches Bug #1 fix)
- `global_optimum` equals `max(all fitness)`
- `top_k_global_indices(10)` returns correct 10 indices

---

## Representations Layer

---

### `src/rag_al/representations/base.py`

**Main classes:** `AbstractEncoder` (ABC)

**Interface**
- `fit(df_labeled, y_labeled) -> None`
- `transform(df) -> np.ndarray shape (N, n_features)`
- `fit_transform(df_labeled, y_labeled) -> np.ndarray`

**Fitness label access**
- `fit()` receives `y_labeled` (labeled fitness only). Subclasses decide whether
  to use it. Must NOT leak it into `transform()` of pool data.

**Authorized:** YES (labeled only, in `fit()`).

**Leakage risks:** Interface-level. Pool fitness is never passed in; the contract
relies on subclasses not storing `y_labeled` and using it in `transform()`.

---

### `src/rag_al/representations/mutation.py`

**Main classes:** `MutationDescriptorEncoder`

**`fit(df_labeled, y_labeled)`**
- In: labeled DataFrame (uses `mutant`, `mutated_sequence`), `y_labeled` (IGNORED)
- Out: fits `StandardScaler` on 49-dim raw features
- Label access: NONE ✅

**`transform(df)`**
- In: DataFrame with `mutant`, `mutated_sequence`
- Out: `np.ndarray` shape `(N, 49)`, standardized
- Label access: NONE ✅

**Feature layout (49 dims)**
- [0] n_mutations
- [1] mean_position (normalized by seq length)
- [2] std_position (normalized)
- [3–5] sum Δhydropathy, Δcharge, Δvolume/100
- [6–8] mean Δhydropathy, Δcharge, Δvolume/100
- [9:29] WT AA counts (20-dim)
- [29:49] mutant AA counts (20-dim)

**Authorized:** N/A — does not use labels.

**Leakage risks:** NONE. Scaler fit on labeled features only (correct).

**Tests**
- Parse `"A23V"` → n_muts=1, wt=A, pos=23, mut=V
- Parse `"A23V:G45L"` → n_muts=2
- Output shape `(N, 49)`
- Malformed token raises `ValueError`
- Scaler from labeled set applies correctly to pool

---

### `src/rag_al/representations/physicochemical.py`

**Main classes:** `PhysicochemicalEncoder`

**`fit(df_labeled, y_labeled)`**
- In: labeled DataFrame (`mutated_sequence`), `y_labeled` (IGNORED)
- Out: fits `StandardScaler` on 29-dim raw features
- Label access: NONE ✅

**`transform(df)`**
- Out: `np.ndarray` shape `(N, 29)`, standardized
- Label access: NONE ✅

**Feature layout (29 dims)**
- [0:20] AA composition (frequency of each of 20 AAs)
- [20] net charge / length
- [21] mean Kyte-Doolittle hydropathy
- [22] aromatic fraction (F+Y+W)
- [23] polar fraction (S+T+N+Q)
- [24] charged fraction (R+K+D+E)
- [25] positive fraction (R+K)
- [26] negative fraction (D+E)
- [27] log(length+1)
- [28] Shannon entropy

**Authorized:** N/A — does not use labels.

**Leakage risks:** NONE.

**Known bug (Bug #4):** Dead code line `net_charge += _AA_HYDROPATHY.get(aa, 0.0) * 0.0`
always evaluates to 0.0. Delete it.

**Tests**
- Output shape `(N, 29)`
- AA composition sums to 1.0 per variant
- Net charge = 0 for balanced sequence (e.g., RKDE)
- Shannon entropy > 0 for diverse, = 0 for poly-A

---

### `src/rag_al/representations/plm.py`

**Main classes:** `ESMEncoder`

**`fit(df_labeled, y_labeled)`**
- In: labeled DataFrame (`wt_sequence`), `y_labeled` (IGNORED)
- Out: stores `_wt_sequence`; computes `_wt_embedding` for delta mode
- Label access: NONE ✅

**`transform(df)`**
- In: DataFrame with `mutated_sequence`, `variant_id`
- Out: `np.ndarray` shape `(N, D)` — tries disk cache, else runs ESM-2
- Label access: NONE ✅

**Common ESM-2 model dimensions**
- `esm2_t6_8M_UR50D` → D = 320 (use on M3 / CPU)
- `esm2_t33_650M_UR50D` → D = 1280 (use on A100/H100)

**Authorized:** NO label access.

**Leakage risks**
- WT embedding computed once and reused. Consistent across rounds if WT sequence
  is the same in all rows (guaranteed by schema validation).

**Performance (fixed):** Cache is now a `{sha256(seq) → embedding}` pickle dict.
Subset calls (labeled/pool splits) look up hashes individually; only misses hit
the model. Saves are atomic via `tempfile + os.rename`.

**Tests**
- `fit()` in delta mode sets `_wt_embedding`
- `transform()` mean mode → shape `(N, D)`
- `transform()` delta mode → shape `(N, D)`, different values from mean
- Cache hit: second call returns same result without running model
- `transform()` without `fit()` in delta mode raises `RuntimeError`

---

### `src/rag_al/representations/retrieval.py`  ⭐ CRITICAL

**Main classes:** `RetrievalAugmentedEncoder`

**`fit(df_labeled, y_labeled)`**
- In: labeled DataFrame, `y_labeled` (labeled fitness ONLY)
- Out: stores `_labeled_embeddings`, `_labeled_y`; builds kNN index
- Label access: YES — `_labeled_y = y_labeled.copy()`. ONLY labeled fitness stored. ✅

**`transform(df)`**
- In: DataFrame (no fitness column)
- Out: `np.ndarray` shape `(N, D+5)` — PLM embeddings + 5 retrieval features
- Label access: YES (indirectly) — `_labeled_y` used to compute neighbor fitness stats.
  Only labeled fitness; no pool fitness. ✅

**Retrieval features (5 dims appended)**
- [0] mean fitness of k nearest labeled neighbors
- [1] std fitness of k nearest labeled neighbors
- [2] min distance to any labeled neighbor (normalized)
- [3] mean distance to k nearest labeled neighbors (normalized)
- [4] max fitness of k nearest labeled neighbors

**Authorized:** YES (labeled only, stored from `fit()`).

**Leakage risks**
- kNN querying pool against labeled index is correct and intended — not leakage.

**Self-label inclusion (fixed — Bug #3):** `transform_labeled()` override fetches
`k+1` neighbors and discards column 0 (self). Runner calls `transform_labeled()`
for the labeled set and `transform()` for the pool.

**Tests**
- `fit()` builds index from labeled embeddings only
- `transform(pool_df)` → shape `(N_pool, D+5)`, no pool fitness used
- When `transform(labeled_df)` called, nearest neighbor distance ≈ 0 (self) — documents Bug #3
- k clamps to `n_labeled` without error
- Retrieval features update correctly each round as labeled set grows

---

## Surrogate Layer

---

### `src/rag_al/surrogates/random_forest.py`

**Main classes:** `RFSurrogate`

**`fit(X, y)`**
- In: `X` shape `(N_lab, D)` labeled features; `y` shape `(N_lab,)` labeled fitness
- Out: fits `RandomForestRegressor`
- Label access: YES — `y` is labeled fitness, used for training ✅

**`predict(X) -> (mu, sigma)`**
- In: `X` shape `(N_pool, D)` pool features
- Out: `mu` shape `(N_pool,)`, `sigma` shape `(N_pool,)` — std of per-tree predictions
- Label access: NONE ✅

**Authorized:** YES (labeled only, in `fit()`).

**Leakage risks:** NONE. Surrogate only receives what caller passes.

**Tests**
- `fit()` then `predict()` runs without error
- `sigma >= 0` always
- `sigma > 0` for points not in training set
- `sigma ≈ 0` for points identical to training points

---

## Acquisition Layer

---

### `src/rag_al/acquisition/base.py`

**Interface:** `select_batch(mu, sigma, batch_size, *, pool_X, labeled_X, labeled_y, rng) -> np.ndarray`

- `mu`, `sigma` — surrogate predictions for pool, shape `(N_pool,)`
- `pool_X` — encoded pool features `(N_pool, D)` — optional, for diversity methods
- `labeled_X` — encoded labeled features `(N_lab, D)` — optional, for retrieval
- `labeled_y` — labeled fitness `(N_lab,)` — optional, for retrieval
- `rng` — `np.random.Generator` for reproducibility
- Returns: local pool indices, shape `(batch_size,)`

**Label access:** `labeled_y` passed in (labeled only). Pool fitness never passed.

---

### `src/rag_al/acquisition/random_acq.py`

- Uses only `rng` — no labels, no surrogate predictions
- Uniform random selection
- Label access: NONE ✅
- Leakage risks: NONE

---

### `src/rag_al/acquisition/greedy.py`

- Score: `a(x) = μ(x)` — rank by predicted mean
- Label access: NONE ✅

---

### `src/rag_al/acquisition/ucb.py`

- Score: `a(x) = μ(x) + β·σ(x)`
- `β = ucb_beta` from config (default 1.0)
- Label access: NONE ✅

---

### `src/rag_al/acquisition/diversity_ucb.py`

- Score: `a(x) = μ(x) + β·σ(x) − γ · max_{s∈selected} cos_sim(x, s)`
- Greedy set-cover: iteratively picks highest penalized score
- Requires `pool_X` for cosine similarity; falls back to UCB if None
- Label access: NONE ✅

---

### `src/rag_al/acquisition/retrieval_ucb.py`

- Score: `a(x) = μ(x) + β·σ(x) + λ·R(x)`
- `R(x)` = mean fitness of k nearest labeled neighbors in feature space
- Requires `pool_X`, `labeled_X`, `labeled_y`; falls back to UCB if any is None
- Label access: YES — `labeled_y` only ✅

**Leakage risks:** If caller passes pool fitness as `labeled_y`, that would be
leakage. Runner always passes `dataset.labeled_y`, which is guaranteed labeled-only.

**Tests**
- `R(x)` equals mean labeled neighbor fitness
- Fallback to UCB when required arrays are None
- k clamps to `n_labeled`

---

## Loop Layer

---

### `src/rag_al/loop/metrics.py`

**Main functions**

| Function | Inputs | Output |
|----------|--------|--------|
| `best_fitness(labeled_y)` | labeled fitness | `max(labeled_y)` |
| `simple_regret(labeled_y, global_optimum)` | labeled fitness, global max | `global_optimum - best_fitness` |
| `topk_recall(labeled_indices, top_k_indices)` | index sets | `|intersection| / k` |
| `batch_mean_fitness(batch_y)` | batch fitness | `mean(batch_y)` |
| `batch_diversity(batch_sequences)` | sequences | mean pairwise Hamming |
| `mean_dist_from_wt(batch_sequences, wt)` | sequences, WT | mean Hamming to WT |
| `compute_round_metrics(...)` | all of the above | flat dict for results CSV |

**Label access:** YES — all functions use fitness. Authorized: YES (metrics).

**Leakage risks:** NONE. Metrics evaluate post-reveal fitness only.

**Tests**
- `simple_regret = 0` when labeled set contains the global optimum
- `topk_recall = 1.0` when all top-k acquired; `0.0` when none acquired
- `batch_diversity = 0.0` for identical sequences
- `mean_dist_from_wt = 0.0` for a batch of WT copies

---

### `src/rag_al/loop/runner.py`  ⭐ CRITICAL

**Main functions:** `run_al_loop(...) -> pd.DataFrame`

**Per-round execution order**

```
1. encoder.fit(labeled_df, labeled_y)          labeled only ✅
2. encoder.transform(labeled_df) → X_labeled   no labels ✅
3. encoder.transform(pool_df)    → X_pool       no labels ✅
4. surrogate.fit(X_labeled, labeled_y)         labeled only ✅
5. surrogate.predict(X_pool) → (mu, sigma)     no labels ✅
6. acquisition.select_batch(..., labeled_y=)   labeled only ✅
7. read batch_sequences via get_sequences()    public API, no fitness ✅
8. dataset.reveal(selected_local)              ONLY authorized reveal ✅
9. batch_y = dataset.fitness_at(global_sel)   correct post-reveal lookup ✅
10. compute_round_metrics(...)                 post-reveal, authorized ✅
```

**Fitness label access**
- `labeled_y` (line 103): authorized ✅
- `global_optimum` (line 81): authorized for metrics ✅
- `top_k_global_indices` (lines 82–83): authorized for metrics ✅
- `dataset._df` (lines 84, 148–149): semi-private, no fitness read but fragile ⚠ (Bug #2)

**Authorized:** YES (labeled only, plus metric helpers).

**Leakage risks:** None remaining.
- Bug #1 fixed: `batch_y` now uses `dataset.fitness_at(global_selected)`.
- Bug #2 fixed: runner uses `dataset.wt_sequence`, `get_sequences()`, `get_variant_ids()` — no direct `_df` access.

**Tests**
- Results DataFrame has `n_rounds` rows
- `best_fitness` is monotone non-decreasing
- `n_labeled` increases by `batch_size` each round
- `batch_y` contains correct fitness for selected variants (catches Bug #1 fix)
- Same seed → identical results (determinism)
- Pool exhaustion stops loop without error

---

## CLI Layer

---

### `src/rag_al/cli/embed.py`  (`rag-embed`)

**Purpose:** Pre-compute and cache ESM-2 embeddings for a full dataset (GPU job).

**What it does**
1. Loads dataset CSV
2. Constructs `ESMEncoder` for each requested mode (mean, delta)
3. Calls `encoder.transform(df)` — triggers computation and saves `.npy` cache

**Label access:** Dataset loaded with fitness, but `ESMEncoder` ignores fitness entirely.

**CLI flags:** `--dataset`, `--esm_model`, `--embed_batch_size`, `--modes`

---

### `src/rag_al/cli/benchmark.py`  (`rag-benchmark`)

**Purpose:** Run one (dataset × representation × acquisition × surrogate × seed) cell.

**What it does**
1. Parses CLI → `BenchmarkConfig`
2. `cfg.ensure()` → validates, creates dirs
3. Loads dataset → `ALDataset` (fitness hidden from this point forward)
4. Builds encoder via `_build_encoder(cfg)`
5. Builds surrogate via `_build_surrogate(cfg)` → `RFSurrogate` or `GPSurrogate`
6. Builds acquisition via `_build_acquisition(cfg)`
7. Calls `run_al_loop(...)` → `(results_df, selections_df)`
8. Prepends metadata columns; writes `seed_<N>.csv` and `seed_<N>_selections.csv`

**Label access:** Only via `ALDataset`'s authorized interface after construction.

**Leakage risks:** NONE at CLI layer.

---

## Cluster Scripts

---

### `scripts/submit_embed.sh`

SLURM batch script — 1 GPU, 2h, 32GB. Runs `rag-embed` for one dataset.
Configure `DATASET`, `ESM_MODEL`, `EMBED_BATCH_SIZE` at top of file.
Run once before benchmark sweep.

---

### `scripts/submit_benchmark.sh`

SLURM array job — default 90 tasks (5 repr × 6 acq × 3 seeds).
Array index decoded as:
```
repr_idx = TASK_ID / (N_ACQS * N_SEEDS)
acq_idx  = (TASK_ID / N_SEEDS) % N_ACQS
seed     = TASK_ID % N_SEEDS
```
Resources per task: 8 CPUs, 16GB, 4h.

---

### `scripts/plot_results.py`

Aggregates all `seed_*.csv` files under `results/<dataset>/`.
Groups by `(representation, acquisition, round)` → mean ± std across seeds.
Produces learning-curve PNG figures — one per metric per view
(`_by_acq.png` and `_by_repr.png`).

---

## Sprint 2 Modules — Implemented

The following modules were added in Sprint 2. All are live on `main`.

---

### `src/rag_al/representations/plm_physico.py`

**Classes:** `PLMPhysicoEncoder`, `PLMSimpleConcatEncoder`

**`PLMPhysicoEncoder`** (`"plm_physico"`, output dim D+5):
Per-residue fusion. For each residue i: `fused_i = [h_i | p_i]` where `h_i` is the
ESM-2 hidden state and `p_i` is a 5-dim lookup [hydropathy, charge, MW, polar, aromatic].
Mean-pooled to (D+5,). Separate disk cache `cache_{model}_physico.pkl` keyed by `sha256(seq)`.
Internal ESMEncoder has `cache_dir=None` to avoid writing an unused mean-pool cache.
`fit()` is a no-op (lookup table is fixed).

**`PLMSimpleConcatEncoder`** (`"plm_concat"`, output dim D+29):
Post-hoc `np.hstack([ESM mean-pool, PhysicochemicalEncoder output])`. ESMEncoder
passes `cache_dir` through so it shares `cache_{model}.pkl` with `plm_mean`.

**Label access:** NONE in `transform()`. `fit()` receives `y_labeled` but ignores it for physico
lookup; physico scaler fits on labeled features only (correct).

---

### `src/rag_al/representations/plm.py` — `mode='site'` addition

**`ESMEncoder(mode='site')`** (`"plm_site"`, output dim D):
Extracts ESM-2 hidden states only at mutated residue positions and averages across sites.
For `mutant="A23V"`, extracts token index 23 (0-indexed residue 22 + 1 for CLS).
Multi-site: average across all mutated positions. Same output shape as `plm_mean`.
Requires `mutant` column in `df` — raises `ValueError` if absent.
Separate cache `cache_{model}_site.pkl` keyed by `sha256(seq + "::" + mutant_str)`.

---

### `src/rag_al/surrogates/gp.py`

**Class:** `GPSurrogate(AbstractSurrogate)`

Single-task ExactGP with `ConstantMean + ScaleKernel(MaternKernel(nu=1.5))`.

**Key implementation details:**
- `fit()`: standardizes X per-dim and y; likelihood used via `ExactMarginalLogLikelihood`
  for training only. Warm-starts from `_prev_state` on subsequent rounds.
- `predict()`: returns **latent posterior** `model(Xs)` — not `likelihood(model(Xs))`.
  This gives pure epistemic σ; observation noise is intentionally excluded for acquisition.
- MLL patience: check every 20 steps, stop if improvement < 1e-4 for 3 checks. `n_iter=200` cap.
- `fast_pred_var` (CG-based) avoids O(n²) covariance matrix.
- Auto-detects CUDA; falls back to CPU.

**Registered as:** `--surrogate gp` via `surrogate` field in `BenchmarkConfig`.

---

### `src/rag_al/loop/metrics.py` — `pool_spearman`

`pool_spearman` added to `compute_round_metrics()`:
- `float(spearmanr(mu, dataset.fitness_at(pool_indices))[0])` (uses `[0]` not `.statistic` for scipy < 1.9)
- Oracle metric: accesses hidden pool fitness for evaluation only. Same category as `topk_recall`.
- Must NOT be passed to the acquisition function.
- Computed in `runner.py` after `surrogate.predict(X_pool)`, before acquisition.

---

## Planned Modules — Sprint 3

---

### `src/rag_al/representations/hf_plm.py` *(planned)*

**Class:** `HFPLMEncoder(AbstractEncoder)`

Generalized HuggingFace protein LM encoder supporting ProtT5, Ankh, Profluent E1,
and any BERT-style encoder. Same hash-based disk cache as `ESMEncoder`.

**Priority order for outreach:** Profluent E1 (ESM-like, trivial) → Ankh → ProtT5

**Preprocessing per model:**
- ProtT5 (`Rostlab/prot_t5_xl_uniref50`, ~3GB): space-separate AAs, B/Z/U/O→X, uppercase; `T5EncoderModel`
- Ankh (`ElnaggarLab/ankh-base`, ~450MB): `tokenizer([list(seq) for seq in seqs], is_split_into_words=True, ...)`; `AutoModel`
- Profluent E1 (`Profluent-Bio/E1-600m`): no preprocessing; `AutoModel` — same as ESM-like

**CLI:** add `"prot_t5"`, `"ankh"`, `"profluent_e1"` to `REPRESENTATIONS` in `config.py`;
add cases to `_build_encoder()` in `benchmark.py`.

**BRCA1 guard:** ProtT5 ≤1022 AA, Ankh ≤2048 AA. Same length-guard pattern as `ESMEncoder`.

---

### `scripts/plot_learning_curves.py` *(planned)*

Crossover analysis: at what `n_labeled` does PLM first beat mutation?
- x-axis: `n_labeled` (not round); reads from existing `seed_*.csv` results
- Line plot per representation, averaged over seeds and acquisitions
- Annotate crossover round per dataset
- Inputs: existing results in `results/` — no new cluster runs needed

**Gate for Step 6b (low n_init sweep):** only run if crossover ≤ n_init=50 on any dataset.

---

### Low n_init sweep *(planned, gated on plot_learning_curves.py findings)*

Config-only: `n_init ∈ {25, 50}`, `batch_size=25`, `n_rounds=30`.
Datasets: BLAT_Deng, PABP, GB1 (SPG1). No code changes — new `submit_small_init.sh`.

---

## Cross-Cutting Test Plan

**Critical — must pass before any experiment**
- `ALDataset`: `labeled_y`, `pool_df` (no fitness), `reveal`, `LeakageError`
- `runner.py`: `batch_y` correct values after `reveal()` (Bug #1 fix)
- End-to-end smoke test on synthetic data (no ESM needed)

**Important — before benchmark sweep**
- `RetrievalAugmentedEncoder`: pool retrieval uses only labeled fitness
- `RFSurrogate`: `sigma > 0` on unseen data
- Each acquisition function: correct local index return, no label access

**Informational**
- `PhysicochemicalEncoder`: AA composition sums to 1.0
- `MutationDescriptorEncoder`: multi-site mutant parsing
- `ESMEncoder`: cache hit/miss behavior
