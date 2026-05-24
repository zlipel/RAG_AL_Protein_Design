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

pytest

# Run a single test file
pytest tests/test_runner_batch_y.py -v

# Pre-compute ESM-2 embeddings (GPU; run once per dataset before benchmark)
rag-embed --dataset BLAT_ECOLX --esm_model facebook/esm2_t6_8M_UR50D

# Run one benchmark cell interactively
rag-benchmark --dataset BLAT_ECOLX --representation plm_mean --acquisition ucb --seed 0

# Aggregate results and plot learning curves
python scripts/plot_results.py --dataset BLAT_ECOLX --output_dir figures/
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
  4. the change is documented in `/docs/agent_log.md` or `/docs/status.md`,
  5. no known leakage invariant is violated.

### Leakage and scientific correctness

- Never access `dataset._df` outside `ALDataset` unless explicitly auditing legacy code.
- Never expose or pass hidden pool fitness labels to encoders, surrogates, or acquisition functions.
- `global_optimum` and `top_k_global_indices()` are metric-only oracle quantities. They must not be used for initialization, model fitting, retrieval, acquisition, or candidate selection.
- `reveal()` is the only authorized hidden-label exposure point during active learning.
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

## Architecture

### Two-stage workflow

**Stage 1 — Embedding** (`rag-embed`, GPU, run once per dataset):
Computes ESM-2 embeddings for all variants and saves to `data/embeddings/<dataset>/`.
Cache is keyed by variant_id list; the benchmark loop loads from cache instead of re-running the model.

**Stage 2 — Benchmark** (`rag-benchmark`, CPU, array-parallelizable):
Runs one (dataset × representation × acquisition × seed) cell. The full benchmark grid is
5 representations × 5 acquisitions × N seeds, submitted as a SLURM array job via
`scripts/submit_benchmark.sh`. Results go to `results/<dataset>/<repr>_<acq>/seed_<N>.csv`.

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
2. `encoder.transform(labeled_df)` → `X_labeled`
3. `encoder.transform(pool_df)` → `X_pool` (pool has no fitness)
4. `surrogate.fit(X_labeled, labeled_y)`
5. `surrogate.predict(X_pool)` → `(mu, sigma)`
6. `acquisition.select_batch(mu, sigma, ..., labeled_y=labeled_y)` → local pool indices
7. `global_selected = pool_indices[selected_local]` (save before reveal)
8. `batch_sequences = dataset.get_sequences(global_selected)` — sequences before reveal
9. `dataset.reveal(selected_local)` — the only label exposure point
10. `batch_y = dataset.fitness_at(global_selected)` — fitness by global index, post-reveal
11. Log selections (round, global_index, variant_id, fitness) to `selections` list
12. `compute_round_metrics(...)` — records best_fitness, simple_regret, topk_recall, etc.

`run_al_loop()` returns `(results_df, selections_df)`. `benchmark.py` saves both:
`seed_<N>.csv` (metrics) and `seed_<N>_selections.csv` (per-round acquisition log).

### Config pattern

`BenchmarkConfig` (`src/rag_al/core/config.py`) is a frozen dataclass. `from_cli()` auto-generates argparse flags for all dataclass fields — adding a new field automatically exposes it as a CLI flag. Call `cfg.ensure()` to validate and create output directories before running.

### Representation encoders (`src/rag_al/representations/`)

All implement `AbstractEncoder` with `fit(df_labeled, y_labeled)` / `transform(df) → np.ndarray`:
- `MutationDescriptorEncoder` — 49-dim hand-crafted features from mutant string (e.g. `A23V:G45L`)
- `PhysicochemicalEncoder` — 29-dim: AA composition, charge, hydropathy, entropy
- `ESMEncoder` — ESM-2 mean-pool or delta (mutant − WT) embeddings with disk cache
- `RetrievalAugmentedEncoder` — wraps ESMEncoder; appends 5 kNN label-context features: `[mean_y, std_y, d_min, d_mean, max_y]` from labeled neighbors

**Known bug in `RetrievalAugmentedEncoder`**: when `transform()` is called on the labeled set itself, sklearn's `NearestNeighbors` returns each point as its own nearest neighbor (distance 0), so labeled features include self-fitness. Pool features are clean. See Bug #3 in `docs/bugs.md`.

### Surrogate (`src/rag_al/surrogates/random_forest.py`)

`RFSurrogate` wraps sklearn `RandomForestRegressor`. Uncertainty σ = std of per-tree predictions via `estimators_`. Fit on labeled features only; never sees pool fitness.

### Acquisition functions (`src/rag_al/acquisition/`)

All implement `select_batch(mu, sigma, batch_size, *, pool_X, labeled_X, labeled_y, rng) → np.ndarray` returning **local pool indices** (0-based into the current pool):
- `RandomAcquisition` — uniform random
- `GreedyAcquisition` — argmax(μ)
- `UCBAcquisition` — argmax(μ + β·σ)
- `DiversityUCBAcquisition` — UCB + diversity penalty (distance to already-selected batch members)
- `RetrievalUCBAcquisition` — UCB + λ·R(x) where R(x) = mean kNN fitness from labeled set

---

## Known Bugs

See `docs/bugs.md` for full details and fix history.

| Bug | Status | Location |
|---|---|---|
| Bug #1 — wrong `batch_y` after reveal | ✅ Fixed | `runner.py` — now uses `dataset.fitness_at(global_selected)` |
| Bug #2 — `dataset._df` accessed from runner | ✅ Fixed | `runner.py` — now uses `get_sequences()` / `get_variant_ids()` |
| Gap #1 — selections not logged | ✅ Fixed | `runner.py` returns `(results, selections)`; saved to `seed_N_selections.csv` |
| Bug #3 — self-neighbor in `RetrievalAugmentedEncoder` | ❌ Open | `retrieval.py:86` — kNN returns self at distance 0; fix on `fix/retrieval-self-neighbor` |
| Bug #4 — dead code in `physicochemical.py` | ❌ Open | `physicochemical.py` — `net_charge += ... * 0.0` line |

---

## Development Workflow

Branches: `fix/<name>` → `audit/agent-scaffold` → `main`. See `docs/workflow.md` for the
full per-fix process and the ordered fix queue.

---

## Data

Raw ProteinGym CSVs are **not** used directly. Data must be curated to the 5-column schema:
`variant_id, mutant, mutated_sequence, wt_sequence, fitness`. Place curated CSVs in `data/`.
Schema validation runs automatically when `ALDataset` is constructed.

The `data/DMS_ProteinGym_indels/` directory contains the raw ProteinGym indel CSVs for reference.
These do **not** match the required schema without curation.
