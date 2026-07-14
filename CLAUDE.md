# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Install (from project root)
pip install -e ".[dev]"

# Lint
ruff check src/

# Type check
mypy src/

# Run full test suite (64 tests as of Sprint 2)
pytest

# Run a single test file
pytest tests/test_gp_surrogate.py -v

# Pre-compute ESM-2 embeddings (GPU; run once per dataset before benchmark)
# Curated CSVs live in data/curated/ — data_dir defaults there
# plm_site uses the same cache as plm_mean (both stored in cache_{model}.pkl)
rag-embed --dataset BLAT_ECOLX_Jacquier_2013 --esm_model facebook/esm2_t33_650M_UR50D

# Run one benchmark cell interactively (RF surrogate, default)
rag-benchmark --dataset BLAT_ECOLX_Jacquier_2013 --representation plm_mean --acquisition ucb --seed 0

# Run with GP surrogate
rag-benchmark --dataset PABP_YEAST_Melamed_2013 --representation plm_mean \
  --acquisition ucb --surrogate gp --seed 0

# Cluster: full RF benchmark sweep for one dataset (8 reprs × 5 acqs × 3 seeds)
sbatch scripts/submit_benchmark.sh PABP_YEAST_Melamed_2013

# Cluster: targeted GP vs RF comparison (PABP + BLAT_Deng, 36 GP + 36 RF cells)
sbatch scripts/submit_gp_benchmark.sh

# Aggregate results and plot learning curves
python scripts/plot_results.py --dataset BLAT_ECOLX_Jacquier_2013 --output_dir figures/
```

---

## Rules

### Planning and scope control

- Before modifying code, summarize the requested change in 3–6 bullets.
- If the request is ambiguous, ask clarifying questions before editing.
- If the request is overspecified or conflicts with the current architecture, say so explicitly and propose the smallest safe alternative.
- Do not implement broad rewrites unless explicitly requested. Prefer small, auditable patches.

### Branch and version-control discipline

- Implement each fix or feature on a dedicated `fix/<name>` or `feature/<name>` branch.
- Do not merge directly into `main`.
- The integration path is:

  `fix/<name>` or `feature/<name>` → `audit/agent-scaffold` → `main`

- Only merge to `main` after:
  1. the relevant tests pass,
  2. `ruff check src/` passes,
  3. `mypy src/` passes or known type issues are documented,
  4. the change is documented in `/docs/agent_log.md`,
  5. no known leakage invariant is violated.

### Leakage and scientific correctness

- Never access `dataset._df` outside `ALDataset` unless explicitly auditing legacy code.
- Never expose or pass hidden pool fitness labels to encoders, surrogates, or acquisition functions.
- `global_optimum` and `top_k_global_indices()` are metric-only oracle quantities. They must not be used for initialization, model fitting, retrieval, acquisition, or candidate selection.
- `reveal()` is the only authorized hidden-label exposure point during active learning.
- `pool_spearman` is computed from `dataset.fitness_at(pool_indices)` — this is a metric-only oracle read, same category as `topk_recall`. It must never be passed to an acquisition function.
- Retrieval features that use labels may only use the currently labeled set.
- If a method cannot run safely without required arrays, raise an error rather than silently falling back, unless the fallback is explicitly documented and tested.

### Testing requirements

- Every bug fix must include or update a test that would have failed before the fix.
- For active-learning logic, include tests on a small synthetic dataset where selected local indices, global indices, and revealed labels can be manually verified.
- Before claiming a fix is complete, run the relevant targeted tests. If full `pytest`, `ruff`, or `mypy` are not run, state that explicitly.

### Documentation requirements

- For every meaningful change, update `/docs/agent_log.md` with:
  - task summary
  - files changed
  - reason for change
  - tests run
  - remaining concerns
- For architectural changes, add or update an ADR in `/docs/decisions/`.
- Keep documentation concise. Do not create duplicate readmes unless the module changed substantially.

### Large jobs and generated artifacts

- Do not launch large ESM embedding jobs, SLURM sweeps, or full benchmark runs unless explicitly instructed.
- Do not commit generated data, embedding caches, logs, results, or large artifacts.
- Use tiny synthetic/local smoke tests before scaling to ProteinGym or cluster runs.

---

## Architecture

### Two-stage workflow

**Stage 1 — Embedding** (`rag-embed`, GPU, run once per dataset):
Computes ESM-2 embeddings for all variants and saves to `data/embeddings/<dataset>/`.
Cache is a `{sha256(sequence) → embedding}` pickle dict (`cache_{model}.pkl`), keyed per
sequence so labeled/pool subset calls are pure lookups after the first run.

**Stage 2 — Benchmark** (`rag-benchmark`, CPU, array-parallelizable):
Runs one (dataset × representation × acquisition × surrogate × seed) cell. The full
RF benchmark grid is 8 representations × 5 acquisitions × N seeds, submitted as a
GNU-parallel job via `scripts/submit_benchmark.sh`. Results go to
`results/<dataset>/<repr>_<acq>/seed_<N>.csv` (metrics) and
`results/<dataset>/<repr>_<acq>/seed_<N>_selections.csv` (per-round acquisition log).

A separate targeted GP grid is submitted via `scripts/submit_gp_benchmark.sh`.

### Leakage enforcement — the critical invariant

`ALDataset` (`src/rag_al/data/al_dataset.py`) is the leakage boundary. Every rule flows from it:

- `self.__fitness` is name-mangled (`_ALDataset__fitness`). Accidental external access via `dataset.__fitness` fails with `AttributeError`.
- `labeled_df` and `pool_df` return only `_FEATURE_COLS = ("variant_id", "mutant", "mutated_sequence", "wt_sequence")` — the `fitness` column is never included.
- `reveal(pool_local_indices)` is the **only** authorized path from hidden → labeled. It raises `LeakageError` on double-reveal.
- `labeled_y` exposes fitness only for the currently labeled subset.
- `global_optimum` and `top_k_global_indices()` access the full label array — authorized for metric computation only; must never be called inside acquisition functions.
- `_df` (single underscore) still holds the full DataFrame including `fitness`. Use the public helpers instead: `wt_sequence` property, `get_sequences(global_indices)`, `get_variant_ids(global_indices)`, `fitness_at(global_indices)`. Never read `_df["fitness"]` outside `ALDataset`.

### AL loop execution order (`src/rag_al/loop/runner.py`)

Each round in `run_al_loop()`:
1. `encoder.fit(labeled_df, labeled_y)` — fit on labeled only
2. `encoder.transform_labeled(labeled_df)` → `X_labeled` (uses self-exclusion for retrieval encoder)
3. `encoder.transform(pool_df)` → `X_pool` (pool has no fitness)
4. `surrogate.fit(X_labeled, labeled_y)`
5. `surrogate.predict(X_pool)` → `(mu, sigma)`
6. `pool_spearman = spearmanr(mu, dataset.fitness_at(pool_indices))[0]` — **metric only, oracle read**
7. `acquisition.select_batch(mu, sigma, ..., labeled_y=labeled_y)` → local pool indices
8. `global_selected = pool_indices[selected_local]` (save before reveal)
9. `batch_sequences = dataset.get_sequences(global_selected)` — sequences before reveal
10. `dataset.reveal(selected_local)` — the only label exposure point
11. `batch_y = dataset.fitness_at(global_selected)` — fitness by global index, post-reveal
12. Log selections (round, global_index, variant_id, fitness) to `selections` list
13. `compute_round_metrics(...)` — records best_fitness, simple_regret, topk_recall, pool_spearman, etc.

`run_al_loop()` returns `(results_df, selections_df)`. `benchmark.py` saves both.

### Config pattern

`BenchmarkConfig` (`src/rag_al/core/config.py`) is a frozen dataclass. `from_cli()` auto-generates
argparse flags for all dataclass fields — adding a new field automatically exposes it as a CLI flag.
Call `cfg.ensure()` to validate and create output directories before running.

Key config groups:
- **AL settings:** `n_init=50`, `n_rounds=5`, `batch_size=20`, `seed`
- **Surrogate:** `surrogate="rf"` | `"gp"`, `n_estimators`, `rf_n_jobs`, `gp_n_iter=200`, `gp_lr=0.1`, `gp_patience=3`
- **Acquisition:** `ucb_beta=1.0`, `retrieval_lambda=0.5`, `n_neighbors=5`
- **PLM:** `esm_model`, `embed_batch_size=32`

### Representation encoders (`src/rag_al/representations/`)

All implement `AbstractEncoder` with `fit(df_labeled, y_labeled)` / `transform(df) → np.ndarray`:

| Encoder | CLI name | Output dim | Notes |
|---------|----------|------------|-------|
| `MutationDescriptorEncoder` | `mutation` | 49 | Hand-crafted features from mutant string |
| `PhysicochemicalEncoder` | `physicochemical` | 29 | AA composition, charge, hydropathy, entropy |
| `ESMEncoder(mode='mean')` | `plm_mean` | D_esm | Mean-pool all residue hidden states |
| `ESMEncoder(mode='delta')` | `plm_delta` | D_esm | mutant mean-pool − WT mean-pool |
| `ESMEncoder(mode='site')` | `plm_site` | D_esm | Hidden states only at mutated positions, averaged across sites. Requires `mutant` column. Separate cache `cache_{model}_site.pkl` keyed by `sha256(seq + "::" + mutant_str)`. |
| `PLMPhysicoEncoder` | `plm_physico` | D_esm + 5 | Per-residue `[h_i \| p_i]` concat then mean-pool. `p_i` = 5-dim lookup [hydropathy, charge, MW, polar, aromatic]. Separate cache `cache_{model}_physico.pkl`. |
| `PLMSimpleConcatEncoder` | `plm_concat` | D_esm + 29 | Post-hoc `hstack([ESM mean-pool, physico])`. Shares ESM cache with `plm_mean`. |
| `RetrievalAugmentedEncoder` | `plm_retrieval` | D_esm + 5 | ESM mean-pool + 5 kNN label-context features from labeled set. Calls `transform_labeled()` (self-excluding kNN) for labeled set, `transform()` for pool. |

**ESM-2 hidden sizes:** 8M → 320, 35M → 480, 150M → 640, 650M → 1280.
**Default model:** `facebook/esm2_t6_8M_UR50D` (local). Cluster uses `facebook/esm2_t33_650M_UR50D`.

### Surrogate models (`src/rag_al/surrogates/`)

Both implement `AbstractSurrogate` with `fit(X, y)` / `predict(X) → (mu, sigma)`.

**`RFSurrogate`** (`--surrogate rf`, default):
Wraps sklearn `RandomForestRegressor`. σ = std of per-tree predictions. Fast, no hyperparameter tuning needed.

**`GPSurrogate`** (`--surrogate gp`):
Single-task ExactGP with `ConstantMean + ScaleKernel(MaternKernel(nu=1.5))`. Key design:
- Per-dim X standardization + y standardization in `fit()`; un-standardized in `predict()`
- `predict()` returns the **latent posterior** `p(f*|x*)` — not `likelihood(model(x))`. This gives pure epistemic σ for acquisition functions (no observation noise mixed in).
- Round-to-round warm start: `_prev_state` carries `model.state_dict()` + `likelihood.state_dict()` between `fit()` calls, so optimizer resumes from previous round's converged hypers.
- MLL patience: check every 20 steps, stop if improvement < 1e-4 for 3 consecutive checks. `n_iter=200` is a cap, not a target.
- `fast_pred_var` (CG-based, avoids O(n²) covariance matrix) in `predict()`.
- Auto-detects CUDA; falls back to CPU.

### Acquisition functions (`src/rag_al/acquisition/`)

All implement `select_batch(mu, sigma, batch_size, *, pool_X, labeled_X, labeled_y, rng) → np.ndarray`
returning **local pool indices** (0-based into the current pool):
- `RandomAcquisition` — uniform random
- `GreedyAcquisition` — argmax(μ)
- `UCBAcquisition` — argmax(μ + β·σ)
- `DiversityUCBAcquisition` — UCB + diversity penalty (cosine distance to already-selected batch members)
- `RetrievalUCBAcquisition` — UCB + λ·R(x) where R(x) = mean kNN fitness from labeled set

---

## Known Bugs and Audit Status

See `docs/bugs.md` for full details. All audit findings are resolved.

| Item | Status | Resolution |
|------|--------|------------|
| Bug #1 — wrong `batch_y` after reveal | ✅ Fixed | `runner.py` uses `dataset.fitness_at(global_selected)` |
| Bug #2 — `dataset._df` accessed from runner | ✅ Fixed | `runner.py` uses `get_sequences()` / `get_variant_ids()` |
| Gap #1 — selections not logged | ✅ Fixed | `run_al_loop()` returns `(results, selections)`; saved to `seed_N_selections.csv` |
| Bug #3 — self-neighbor in `RetrievalAugmentedEncoder` | ✅ Fixed | `transform_labeled()` fetches k+1 neighbors, discards self |
| Bug #4 — dead code in `physicochemical.py` | ✅ Fixed | Deleted `net_charge += ... * 0.0` line |
| Performance — ESMEncoder cache misses | ✅ Fixed | Hash-map cache keyed by `sha256(seq)`, atomic save |

---

## Development Workflow

Branches: `fix/<name>` or `feature/<name>` → `audit/agent-scaffold` → `main`.
See `docs/workflow.md` for the full per-fix process.

---

## Sprint Status

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1 — core AL loop + ESM-2 representations | ✅ Complete | 6 datasets benchmarked; key finding: PLM+retrieval+UCB best overall (0.920 topk10_recall); PABP anomaly (PLM underperforms mutation on flat landscape) |
| Sprint 2 — new representations + GP surrogate | ✅ Complete | plm_site, plm_physico, plm_concat, GPSurrogate; 64/64 tests; pushed to main |
| Sprint 3 — HFPLMEncoder + analysis | 🔄 Next | ProtT5, Ankh, Profluent E1; plot_learning_curves.py crossover analysis |

**Cluster runs pending (submitted/to-submit):**
- Full RF benchmark on 8 datasets × 8 reprs × 5 acqs × 3 seeds (`submit_benchmark.sh`)
- Targeted GP vs RF comparison on PABP + BLAT_Deng (`submit_gp_benchmark.sh`)

---

## Data

Raw ProteinGym CSVs are **not** used directly. Data must be curated to the 5-column schema:
`variant_id, mutant, mutated_sequence, wt_sequence, fitness`. Curated CSVs live in `data/curated/`.
Schema validation runs automatically when `ALDataset` is constructed.

**Curated datasets (8 total in `data/curated/`):**

| Dataset | n_variants | WT length | Notes |
|---------|-----------|-----------|-------|
| `BLAT_ECOLX_Jacquier_2013` | 989 | 263 AA | Small, ordinal; PLM neutral vs mutation |
| `BLAT_ECOLX_Deng_2012` | 4,996 | 263 AA | PLM clearly wins; largest PLM gain |
| `BLAT_ECOLX_Firnberg_2014` | 4,783 | 263 AA | Deceptive outlier (F58N); model-based < random |
| `BLAT_ECOLX_Stiffler_2015` | ~2,000 | 263 AA | |
| `PABP_YEAST_Melamed_2013` | ~21,000 | 75 AA | PABP anomaly — PLM underperforms mutation; RF miscalibrated on flat landscape; primary GP motivation |
| `BRCA1_HUMAN_Findlay_2018` | ~4,000 | 1,863 AA | Non-PLM only (exceeds ESM-2 1022-residue limit) |
| `GFP_AEQVI_Sarkisyan_2016` | ~51,000 | 238 AA | Multi-site (median ~4 muts) |
| `SPG1_STRSG_Wu_2016` (GB1) | ~149,000 | 448 AA | 4-site combinatorial; canonical multi-site benchmark |

**Next steps for data:** sync `data/curated/` to cluster via rsync before running benchmarks.
