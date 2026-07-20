# Agent Log
## RAG-AL Protein Design — Running Change History

---

## 2026-07-20 — GP OOM root cause: exact-GP pool-variance transient; fix = chunked predict

### Task summary
The GP grid kept OOM-killing even at 10 GB/cell (12 concurrent, 4 cpus-per-cell). Traced
it — with local memory probes — to `GPSurrogate.predict()` scoring the **entire pool in
one forward pass**. Exact-GP predictive variance carries an `O(n_test^2)`-sized transient;
at PABP scale (pool ≈ 37,658, n_train ≈ 2,610, dim 1280) a single `predict()` peaks at
**10.63 GB**, just over the cap. Fixed by chunking the pool loop.

### Why it was the GP and not the encoder / fit / pool_spearman
- **Not the encoder:** `plm_physico` caches pooled `(D+5,)` vectors (~0.4–0.8 GB), like
  `plm_mean`. Mutation cells (49-d) OOM'd too → not representation-driven.
- **Not the fit:** Cholesky on n_train ≤ 2,610 is ~27 MB.
- **Not `pool_spearman`:** `runner.py:137` computes `predict(X_pool)` once; both
  `pool_spearman` (l.142, reuses `mu`) and the acquisition `select_batch` (l.145) consume
  it. The full-pool predict is intrinsic to acquisition on a fixed pool — removing the
  metric would not have helped.
- **Why RF survived the same loop:** RF predict is per-tree, `O(n_test · n_trees)` ≈ 30 MB,
  no covariance. The pool-sized variance transient is GP-specific. (Supersedes the earlier
  "torch base + requested RAM" explanation — that was a contributor, not the driver.)
- **Driver is n_test (pool), not n_train (revealed set):** chunking the *pool* fixed it,
  which it could not if the growing labeled set were the cause. Over a 20-round loop the
  n_train contribution adds only ~0.85 GB (chunked: 1.52 → 2.37 GB across rounds; no
  cross-round accumulation — peak = worst single round).

### Fix
Chunk the pool in `predict()`: `bs = max(1, min(predict_batch_size, n))`, standardize +
score per chunk, concatenate `mu`/`sigma`. Marginals only (per-point mean + stddev), so
chunking is exact for what acquisition uses (no cross-covariance needed). Shipped
`predict()` at PABP scale now peaks **1.12 GB** (was 10.63). Chunked vs single-chunk `mu`
is identical; `sigma` differs only within LOVE's approximation noise.

### Files changed
- `src/rag_al/surrogates/gp.py` — `predict_batch_size` param (default 4096); chunked
  `predict()` with per-chunk standardization + empty-pool guard.
- `src/rag_al/core/config.py` — `gp_predict_batch_size: int = 4096` (auto CLI flag).
- `src/rag_al/cli/benchmark.py` — `_build_surrogate` threads it into `GPSurrogate`.
- `tests/test_gp_surrogate.py` — chunked-vs-unchunked invariance (mu tight, sigma loose);
  default value + empty-pool contract.

### Tests / checks
`pytest` 77 passed (75 + 2 new). `ruff check` on changed files clean. `mypy` on `gp.py`:
only the pre-existing `gpytorch` missing-stubs note (documented). Memory verified via
throwaway probes (not committed).

### Remaining
- `MEM_PER_CELL=8G` in `submit_gp_benchmark.sh` is now generous (peak ~1–2 GB); left as-is.
  Chunking makes the per-cell memory isolation a safety margin rather than a necessity, and
  removes the memory cliff for GB1 (~149K pool) if it ever enters the GP grid.
- Dropped `#SBATCH --exclusive` from the header (whole-node reservation no longer warranted
  once predict is chunked). The per-cell `srun --exclusive` steps now draw from a normal
  allocation, so the CPU/mem the job requests at submit time gates concurrency
  (`MAX_CONCURRENT` defaults to `SLURM_CPUS_ON_NODE`).

---

## 2026-07-18 — GB1 RF rerun complete; Sprint 1/2 results docs + figures

### Task summary
The GB1 (`SPG1_STRSG_Wu_2016`) RF rerun completed and synced — now a full 8-rep grid
(120 CSVs), replacing the earlier partial/broken data (`plm_delta` was random-only,
regret 3.6). Wrote two numbers-backed results docs and generated supporting figures.

### GB1 now complete — sharpest PLM win in the sweep
Final-round `topk10_recall` (mean over acqs & seeds): mutation **0.153**, physico 0.393,
PLM **0.64–0.79** (`plm_retrieval` 0.793; best cell `plm_site × ucb` = 1.000). Every PLM
rep reaches `simple_regret = 0` (finds the 4-site optimum); mutation stays 1.15–1.92.
Confirms the PLM advantage scales with multi-site/epistatic complexity.

Grid completeness now: **6 of 8 datasets full 8-rep** (4 BLAT + PABP + GB1). GFP still
5-rep (site/physico/concat pending) — provisional; BRCA1 2-rep by design. This resolves
the GB1 half of the "GFP + GB1 incomplete" caveat in the 2026-07-15 analysis entry.

### Files changed
- `docs/sprint1_results.md` (new) — core 5-rep benchmark: PLM ≫ hand-crafted,
  `plm_retrieval` best, per-dataset table, PABP anomaly, Firnberg 8M→650M reversal.
- `docs/sprint2_results.md` (new) — Sprint-2 reps competitive-not-superior (with the
  completeness confound), GB1 multi-site win, `pool_spearman` locating the PABP
  top-of-landscape failure, GP status.
- `docs/figures/*` (generated via `plot_aggregate.py` / `plot_results.py`, restricted to
  the complete datasets): topk10 heatmap + per-dataset bar; pool_spearman; PABP/GB1/Deng
  learning curves.

### Remaining
- Run the GP grid; full GFP re-run (task queue).

---

## 2026-07-18 — GP benchmark deploy: srun-isolated cells, right-sized memory

### Task summary
The GP-only sweep OOM'd on the cluster (`sacct State=OUT_OF_MEMORY, ExitCode 0:125,
MaxRSS ≈ 33.5 GB` against `--mem=32G`). Reworked `scripts/submit_gp_benchmark.sh` to
run each cell as an isolated `srun --exclusive` step under a whole-node allocation,
replacing the single shared-cgroup GNU-parallel design.

### Root cause (it was memory, not threads)
Not a large model or expensive inference. 16 concurrent cells shared one 32 GB
allocation cgroup; a few PABP `plm_*` cells (~2–2.5 GB each) pushed the total past
32 GB and the OOM-killer took down whatever was resident — including tiny 49-dim
`mutation` cells (collateral kills, which is why even those "failed"). An earlier
thread-oversubscription theory was wrong; `sacct` settled it.

### Why RF survived and GP didn't (same AL loop)
Both do encode → fit on revealed data → predict over the full pool → acquire; the
encoder cost is identical. Three differences:
1. torch/gpytorch base ≈ 0.4–0.5 GB/process vs sklearn ≈ 0.15 GB.
2. GP `predict()` forms cross-covariance K(pool, train) ≈ 0.4 GB/round (RF's
   tree-averaging has no analog); GNU-parallel cells run in lock-step so the spikes
   coincide.
3. The GP job requested **half** the memory: RF = 64 GB / 48 cores; GP = 32 GB / 16.
So RF peaked ~49 GB < 64; GP peaked ~33.5 GB > 32. Same logic, heavier runtime,
lower budget.

### srun --exclusive vs GNU parallel (decision)
GNU parallel runs all cells in one allocation cgroup, so one cell's OOM can kill
siblings (the failure here). `srun --exclusive` gives each cell its own core + memory
cgroup → an OOM is contained to that cell, with per-cell `sacct` accounting. Chose
srun. (GNU parallel's edge is ergonomics/portability, not relevant on-cluster.)

### Files changed
- `scripts/submit_gp_benchmark.sh` — sbatch `--nodes=1 --exclusive` (was
  `--cpus-per-task=16 --mem=32G`); per-cell `srun --exclusive --mem=$MEM_PER_CELL`
  steps with a `MAX_CONCURRENT` throttle + per-cell exit-code summary, replacing the
  GNU-parallel dispatch; OMP/BLAS threads pinned to 1. Knobs: `MEM_PER_CELL=8G`,
  `CPUS_PER_CELL=1`, `MAX_CONCURRENT`=node cores. Grid unchanged (PABP + BLAT_Deng ×
  {mutation, plm_mean, plm_physico} × {greedy, ucb} × 3 seeds = 36 cells; no GB1).
- `scripts/README.md` — GP section updated (srun isolation + memory knobs).

### Verification
- `bash -n scripts/submit_gp_benchmark.sh` clean.
- Dispatch harness simulated locally (true/false stand-in cells): throttle, per-cell
  rc capture, failure count, and `set -e` safety all correct (reported 2/4 on a
  deliberate 2-failure case).
- Confirmed the paired RF baseline for all 36 cells already exists in `results/`
  (main sweep ran all reps/acqs on both datasets) → GP-vs-RF is directly comparable.

### Remaining
- Submit on cluster; compare GP vs RF (topk10 + pool_spearman), focused on PABP's
  top-of-landscape calibration.

---

## 2026-07-15 — 650M full-grid analysis (RF): findings + surrogate/repr-aware plotting

**Branch:** `feature/analysis-650m` (cut from `audit/agent-scaffold`).

### Task summary
Analyzed the synced ESM-2 650M RF benchmark (741 seed CSVs, 8 datasets) and amended
the analysis scripts to match the richer grid. `plot_aggregate.py` / `plot_results.py`
predated the Sprint-2 representations (`plm_site`, `plm_physico`, `plm_concat`) and the
`surrogate` axis, so both were silently dropping the new reps and would have blended
RF/GP. Now they know all 8 reps, filter by `--surrogate` (default `rf`), and the
aggregate script gained a `--no_plots` table mode and a `--datasets` subset filter used
for the fair comparison below.

### Grid completeness (important caveat)
- **Full 8-rep grid:** 4× BLAT (Deng, Firnberg, Jacquier, Stiffler) + PABP.
- **BRCA1:** 2 non-PLM reps only (WT 1863 AA > ESM-2 limit) — correct; saturates at 1.0.
- **GFP:** only the 5 original reps (missing `plm_site`/`plm_physico`/`plm_concat`).
- **GB1 (SPG1):** PLM grid partial/failed — `plm_delta` is random-only (mean
  simple_regret 3.6), `plm_mean_retrieval_ucb` has 0 seeds, several 1-seed cells.
- ⇒ **GFP + GB1 numbers are provisional**; they need a full Sprint-2-grid re-run before
  any multi-site conclusion. All cross-rep comparisons below are on the 5 complete datasets.

### Key findings (topk10_recall, final round, ESM-2 650M, RF)

**Q1 — best representation (fair comparison, 5 fully-gridded datasets):**

| repr | random | greedy | ucb | div_ucb | ret_ucb | **MEAN** |
|---|---|---|---|---|---|---|
| plm_retrieval | 0.567 | 0.867 | **0.920** | 0.887 | 0.867 | **0.821** |
| plm_delta | 0.567 | 0.860 | 0.893 | 0.873 | 0.893 | 0.817 |
| plm_physico | 0.567 | 0.853 | 0.873 | 0.873 | 0.860 | 0.805 |
| plm_site | 0.567 | 0.833 | 0.860 | 0.900 | 0.853 | 0.803 |
| plm_mean | 0.567 | 0.860 | 0.847 | 0.840 | 0.893 | 0.801 |
| plm_concat | 0.567 | 0.833 | 0.867 | 0.853 | 0.867 | 0.797 |
| mutation | 0.567 | 0.800 | 0.840 | 0.853 | 0.833 | 0.779 |
| physicochemical | 0.567 | 0.633 | 0.640 | 0.573 | 0.633 | 0.609 |

All six PLM variants cluster at **0.80–0.82 (within seed noise)**, above mutation
(0.779) and well above physicochemical (0.609). **The Sprint-2 reps (site/physico/concat)
do NOT beat the Sprint-1 reps on equal footing** — the naive all-8-dataset aggregate that
appeared to rank them on top was a completeness artifact (they skip GFP/GB1, which have
recall ≈ 0.2 and drag the older reps' means down). Sprint 1's headline replicates at 650M:
**PLM ≫ non-PLM; `plm_retrieval` marginally the strongest single variant.** (Sanity check:
`random` gives an identical 0.567 for every rep, as expected — random selection ignores
the surrogate, so recall is representation-independent.)

**Q2 — where PLM helps / fails:**
- **Largest PLM gain — BLAT_Deng:** physico 0.42 / mutation 0.68 → PLM ~0.85–0.90. Confirmed.
- **Saturation:** Jacquier (all reps 1.000) and BRCA1 (non-PLM 1.000) are too easy to
  discriminate at this budget — uninformative for ranking reps.
- **PABP anomaly — confirmed and sharpened.** topk10: mutation **0.507** > every PLM
  (best `plm_retrieval`/`plm_delta` ≈ 0.40–0.41). But `pool_spearman` is the *opposite*:
  PLM **0.61–0.64** > mutation 0.517. So with PLM features the surrogate ranks the *bulk*
  of the pool better yet recalls the true **top-10 worse** — the failure is localized to
  the **top of the landscape**, not global rank. This is the precise, concrete motivation
  for the GP surrogate (tail/uncertainty calibration), not just "PLM is worse on PABP."

**Q3 — retrieval augmentation:** `plm_retrieval` is the strongest single PLM variant on
the fair subset (0.821) and the clear winner on **Firnberg** (topk10 0.933; mean
simple_regret 0.16 vs mutation 1.04). **Firnberg update:** the 8M "deceptive-outlier"
pathology (all model-based methods stuck at regret 1.1995, never finding F58N) does **not**
replicate at 650M — higher-capacity embeddings + retrieval locate the isolated optimum.
Worth a per-seed confirmation, but a notable reversal from the Sprint-1 (8M) conclusion.

**Acquisition / metric interpretation:** `random` yields the *lowest* topk10 (~0.567) but
the *highest* pool_spearman for every rep (e.g. plm_site: random 0.732 vs greedy 0.432) —
a clean exploration/exploitation signature. Unbiased sampling → best global rank
correlation but worst top-k discovery; `ucb`/`retrieval_ucb` → best top-k. Confirms
**topk_recall (not pool_spearman) is the right primary objective** for the "find the best
variant" goal.

**Surrogate:** every synced cell is RF (`surrogate` column uniformly `rf`); the GP grid is
not yet run. The PABP top-vs-global split above is the specific case GP is meant to fix.

### Files changed
- `scripts/plot_aggregate.py` — added `plm_site`/`plm_physico`/`plm_concat` to
  `_REPR_ORDER`/`_REPR_LABELS`; `final_round_metric` now groups by `surrogate`;
  `main` gained `--surrogate` (`rf`|`gp`|`all`), `--no_plots` (print tables only),
  `--datasets` (subset filter); figures get a `_<surrogate>` suffix for non-RF.
- `scripts/plot_results.py` — colors/labels for the 3 new reps; `--surrogate` filter
  (default `rf`) so RF/GP curves never blend; `_<surrogate>` filename suffix; removed
  unused `numpy` import.

### Verification
- `ruff check scripts/plot_aggregate.py scripts/plot_results.py` → clean.
- Regenerated `figures/aggregate/` (all 8 reps now appear) and confirmed `--no_plots`
  and `--datasets` produce the tables above. (Figures are gitignored.)

### Remaining concerns / next steps
1. **Re-run GFP + GB1 on the full Sprint-2 grid** — site/physico/concat missing on both;
   GB1's PLM cells are partial/failed. No multi-site conclusion until then.
2. **Run the GP grid** (`submit_gp_benchmark.sh`, PABP + BLAT_Deng); compare topk10 *and*
   pool_spearman vs RF, focusing on the top of the PABP landscape.
3. `plot_learning_curves.py` (n_labeled crossover) still to build.

---

## 2026-07-15 — Milestone: 650M RF sweep complete; results synced; docs refreshed

**Branches:** `fix/gitignore-results-backup`, `docs/session-progress-update`
(both cut from `audit/agent-scaffold`).

### Summary
Operational milestone rather than a code feature. The full ESM-2 650M RF benchmark
finished on the cluster and results were synced back locally. Prepared the repo for
the analysis phase.

- **Results archived + synced.** Renamed the old ESM-2 8M prototype results
  (`results/`, 816 CSVs, dated 2026-06-26/28, pre-`surrogate`/`pool_spearman` schema)
  to `results_sprint1_8M/`; `results/` now holds the 650M full grid. The result path
  does not encode model size, so keeping them separate avoids silent overwrites.
- **gitignore.** Added `results_*/` so model-tagged result backups can't be committed.
- **Docs refreshed** to reflect the current state (this update): `CLAUDE.md` (test
  count 75, embed 4-mode note, GP-only benchmark, Sprint status, PABP = 577 AA /
  37,708 variants, length-based PLM guard), `docs/workflow.md` (current phase),
  `docs/implementation_map.md` (embed dispatch, surrogate-namespaced paths, GNU-parallel
  benchmark — corrected the stale "SLURM array job" description), `scripts/README.md`
  (same array→parallel correction, 4-mode embed, GP-only benchmark, run_embed/run_benchmark).

### Next steps (Sprint 3 — analysis)
1. Analyze synced 650M results (`plot_results.py`; cross-dataset comparison).
2. `plot_learning_curves.py` — crossover analysis (n_labeled where PLM beats mutation).
3. Run + analyze the GP-only benchmark; add `surrogate` to plot grouping.
4. Then `HFPLMEncoder`, low n_init sweep, ESM-2 size sweep.

### Remaining concerns
- `plot_results.py` still groups by (repr, acq) only — must add `surrogate` before
  plotting GP vs RF (tracked in implementation_map.md and scripts/README.md).

---

## 2026-07-15 — Benchmark safety: surrogate-namespaced results + length-based PLM guard

**Branch:** `feature/benchmark-safety-guards` (cut from `audit/agent-scaffold`)

### Task summary
Two pre-run hazards before launching the cluster sweep:
1. **GP/RF file clash.** `seed_results_csv` = `results/<dataset>/<repr>_<acq>[_bβ]/seed_N.csv`
   did not encode the surrogate, and the CSV had no `surrogate` column. So GP and
   RF runs of the same (repr, acq, seed) overwrote each other's files, and even
   in separate dirs `plot_results.py` (rglob + CSV columns) couldn't tell them
   apart. `submit_gp_benchmark.sh` ran `SURROGATES=(rf gp)` — clashing with itself.
2. **Too-long sequences.** The PLM-exclusion for BRCA1 was a hardcoded dataset-name
   match in `submit_benchmark.sh`. Any other long dataset would slip through and
   ESMEncoder would raise on >1022-residue sequences, aborting the whole GNU-parallel
   job (`--halt soon,fail=1`).

### Files changed
- `src/rag_al/core/paths.py` — `_tag()` gains a `surrogate` arg; appends
  `_{surrogate}` only when non-`rf`. `BenchmarkPaths` gains a `surrogate="rf"`
  field. RF paths are unchanged (backward compatible with Sprint 1 results).
- `src/rag_al/core/config.py` — `paths` property threads `surrogate` through.
- `src/rag_al/cli/benchmark.py` — insert a `surrogate` column into results +
  selections CSVs (position 3, before seed).
- `scripts/submit_benchmark.sh` — length-probe the CSV (`max(len(mutated_sequence))`)
  and drop PLM reps when it exceeds `ESM_MAX_RESIDUES=1022`, replacing the
  hardcoded BRCA1 name check.
- `scripts/submit_gp_benchmark.sh` — GP-only (`SURROGATES=(gp)`); RF baseline now
  comes from the main sweep. Added the same per-dataset length guard so the grid
  is safe to extend to long datasets. Header/grid comments updated (36 GP cells).
- `tests/test_result_paths.py` (new) — 7 tests: RF tag backward-compat, GP suffix,
  RF≠GP `seed_results_csv`/selections paths, config→path threading.
- `CLAUDE.md` — documented the `<tag>` format, surrogate suffix, and length guard.

### Verification
- `pytest -q` → 75 passed (68 + 7 new). `ruff check src/` clean.
  `mypy` clean on paths.py / config.py / benchmark.py.
- Length probe on real CSVs: BRCA1 (max 1863) → EXCLUDE PLM; PABP (577), GB1 (448),
  GFP (238), BLAT_Deng (286) → include PLM. Matches the old BRCA1-only behavior.
- End-to-end: ran RF and GP cells for the same (Jacquier, mutation, ucb, seed 0) →
  landed in `mutation_ucb_b1.0/` (surrogate=`rf`) and `mutation_ucb_b1.0_gp/`
  (surrogate=`gp`). No overwrite; rows distinguishable.
- `bash -n` on both submit scripts → syntax OK.

### Remaining concerns
- `plot_results.py` aggregates by (repr, acq) via rglob; once GP results exist it
  should also group by the new `surrogate` column (or it will mix RF/GP curves).
  Deferred — GP plotting is a Sprint 3 analysis task, and no GP results exist yet.
- Note: curated PABP `mutated_sequence` max length is 577 (not the 75 AA quoted in
  CLAUDE.md's dataset table) — still well under the ESM-2 limit, so no action; flag
  for a later doc correction.

---

## 2026-07-14 — GB1 embed submitted separately with longer wall time

**Branch:** `feature/gb1-embed-walltime` (cut from `audit/agent-scaffold`)

### Task summary
With `rag-embed` now doing three heavy forward passes (mean, site, physico) per
dataset, GB1 (`SPG1_STRSG_Wu_2016`, ~149K variants × ~448 AA) can exceed
`submit_embed.sh`'s 2h `#SBATCH --time` ceiling. Split GB1 out of the
`run_embed.sh` batch loop and submit it separately with a CLI `--time` override
(default 8h, overridable via `GB1_WALLTIME`). All other datasets stay at 2h —
GFP (~51K × 238 AA), the next largest, fits comfortably.

### Files changed
- `scripts/run_embed.sh` — batch loop now covers the 6 smaller PLM datasets;
  GB1 gets a dedicated `sbatch --time="${GB1_WALLTIME:-08:00:00}"` submission.

### Tests run
- `bash -n scripts/run_embed.sh` → syntax OK (no unit tests — SLURM wrapper only).

### Remaining concerns
- 8h is a conservative default; if the GB1 job still hits the wall, bump with
  `GB1_WALLTIME=12:00:00 bash scripts/run_embed.sh`.

---

## 2026-07-14 — Embed pipeline: precompute site + physico caches; cache-isolation audit

**Branch:** `feature/embed-plm-modes` (cut from `audit/agent-scaffold`)

### Task summary
`rag-embed` only precomputed `mean`/`delta`, so `plm_site` and `plm_physico` were
never cached ahead of the benchmark. On the CPU benchmark nodes those two
representations would have triggered full ESM-2 forward passes at run time
(21K–149K variants for PABP/GFP/GB1), defeating the two-stage design. Extended the
embed step to cover all four PLM caches and pinned down the cache-isolation invariant.

### Files changed
- `src/rag_al/cli/embed.py` — `--modes` now accepts `{mean, delta, site, physico}`
  and defaults to all four; dispatch `physico` → `PLMPhysicoEncoder`, others →
  `ESMEncoder`. Annotated the loop var `encoder: AbstractEncoder` (mypy union fix).
- `scripts/submit_embed.sh` — passes `--modes mean delta site physico`; added a
  multi-site caveat note (GFP/GB1 site-averaging may dilute signal).
- `scripts/run_embed.sh` — comment noting each job now precomputes four caches.
- `tests/test_cache_paths.py` (new) — 4 tests asserting mean/site/physico resolve
  to distinct files, that mean+delta share one file, and that mean/physico share a
  key function but never share a file (guards against a dropped suffix).

### Cache-isolation audit (the question that motivated this)
Distinct files in `data/embeddings/<dataset>/`, no overlap:
- `cache_{model}.pkl` — mean/delta/retrieval/concat, key `sha256(seq)`, `(D,)`
- `cache_{model}_site.pkl` — site, key `sha256(seq + "::" + mutant)`, `(D,)`
- `cache_{model}_physico.pkl` — physico, key `sha256(seq)`, `(D+5,)`

Note: mean and physico use the *same* key function; isolation rests solely on the
filename suffix — now covered by a regression test.

### Tests run
- `pytest -q` → 68 passed (64 + 4 new cache-path tests)
- `ruff check src/` → clean; `mypy src/rag_al/cli/embed.py` → clean
- End-to-end: `rag-embed --modes mean delta site physico` on a 6-row subset with
  ESM-2 8M produced 3 cache files with shapes `(320,)`, `(320,)`, `(325,)` as expected.
- `mypy src/` baseline unchanged: 37 pre-existing errors in 13 files (lazy-import
  `None`-callable pattern in `plm.py`/`plm_physico.py`/`retrieval.py`; missing
  pandas/scipy/sklearn stubs). `embed.py` — the only file changed here — is clean.

### Remaining concerns
- `plm_site` on multi-site datasets (GFP median ~4 muts, GB1 4-site) averages across
  mutated positions and may wash out signal; kept for comparison, `plm_delta` is the
  multi-site baseline. Documented in `submit_embed.sh` and `docs/sprint2_plan.md`.

---

## 2026-06-26 — PLM benchmark results: Deng 2012 & Firnberg 2014 analysis

**Branch:** `audit/agent-scaffold`

### Task summary
Ran the full 5-repr × 5-acq × 3-seed benchmark grid (n_rounds=20, batch_size=128,
n_init=50, ESM-2 8M, β=1.0) locally on Deng 2012 and Firnberg 2014.
Jacquier 2013 results from the prior session were included in cross-dataset comparison.

### Key findings

**Deng 2012 (n=4996, fitness std=1.53) — PLM clearly wins**

| repr | greedy | ucb_b1.0 | diversity_ucb | retrieval_ucb |
|---|---|---|---|---|
| mutation | 0% | 67% | 67% | 33% |
| physicochemical | 67% | 67% | 0% | 33% |
| plm_delta | **100%** | **100%** | **100%** | **100%** |
| plm_mean | **100%** | **100%** | **100%** | **100%** |
| plm_retrieval | 67% | **100%** | **100%** | **100%** |

% seeds finding the global optimum (simple_regret = 0 at final round).
PLM representations with any non-random acquisition find the global optimum in every seed.
Speed: `plm_{delta,mean} + retrieval_ucb` reaches regret=0 in ~4 rounds on average;
mutation greedy never finds it. Mean simple_regret: PLM=0.055, non-PLM=0.242.

**Firnberg 2014 (n=4783, fitness std=0.45) — deceptive outlier pathology**

Global optimum is F58N with fitness 2.9024 — isolated 1.1995 units above the next cluster
(1.7029, shared by 8 variants). With budget=2610 labels (54.6% of all variants), every
model-based method (greedy, UCB, diversity_ucb, retrieval_ucb) across all representations
achieves simple_regret = 1.1995, meaning none ever selects F58N. The surrogate exploits
the 1.7029 cluster (rich, well-supported region) and persistently underestimates F58N
because it is a feature-space outlier. Random achieves regret ≈ 0.40 — it has ~42%
probability of stumbling on F58N by chance given the budget.

Diagnosis: this is a **deceptive local optimum** failure mode. AL with model-based
acquisition is worse than random on this landscape. Mitigation strategies include
ε-greedy exploration, Thompson sampling, or ensemble disagreement-based selection.

**Jacquier 2013 (n=989, fitness std=1.95) — ordinal landscape, small n**

Top-10% threshold = 0.0 (38% of variants qualify), making topk10_recall uninformative.
Simple regret shows some methods find the optimum (mutation + diversity_ucb, plm_retrieval
+ diversity_ucb), but the dataset is too small and ordinal for reliable differentiation.
PLM mean regret (0.19) is slightly worse than non-PLM (0.13), likely because ESM-2 8M
features add noise for single-mutant antibiotic resistance on a dataset of only 989 variants.

**Cross-dataset PLM vs non-PLM (mean simple_regret, final round):**

| Dataset | non-PLM | PLM |
|---|---|---|
| Deng_2012 | 0.242 | 0.055 |
| Firnberg_2014 | 1.040 | 0.906 |
| Jacquier_2013 | 0.133 | 0.189 |

PLM helps on Deng (large, continuous landscape), is uniformly ineffective on Firnberg
(deceptive outlier, not a representation problem), and is neutral/slightly harmful on Jacquier.

### Files changed
- `environment.yml` — Python 3.12, added ipykernel/ipywidgets; torch/gpytorch/botorch as manual step
- `scripts/setup_cluster.sh` — PyTorch 2.x + torchvision, CUDA 12.x (cu121 default)

### Remaining datasets to run
Stiffler 2015 (killed partway through), PABP_YEAST_Melamed_2013, BRCA1_HUMAN_Findlay_2018
(non-PLM only, WT len=1863 > ESM-2 limit) — move to cluster.

---

## 2026-07-12 — Sprint 2 Step 3: GPSurrogate

**Branch:** `feature/sprint2-repr-surrogate`

### Task summary
Implemented `GPSurrogate` — a single-task ExactGP with Matérn 3/2 kernel, round-to-round
warm start, and MLL patience stopping. Wired into `benchmark.py` via `_build_surrogate(cfg)`;
activated with `--surrogate gp`.

### Design
- **Model:** `_GPRegressionModel(ExactGP)` with `ConstantMean` + `ScaleKernel(MaternKernel(nu=1.5))`.
  Defined directly in `gp.py` — does not import from al_active_dev (which uses a multitask model).
- **Standardization:** per-dim X mean/std and y mean/std computed in `fit()`, un-standardized
  in `predict()`. Handles mixed-scale features (e.g., plm_physico's ESM + physico dims) at
  training time, so encoders stay scale-agnostic.
- **Warm start:** `_prev_state` stores `model.state_dict()` + `likelihood.state_dict()` after
  each `fit()`. On next call, a new ExactGP is created with the enlarged labeled set and hypers
  loaded from `_prev_state` before continuing Adam steps.
- **Patience:** check MLL every 20 steps; if improvement < 1e-4 for 3 consecutive checks,
  stop early. `n_iter=200` is a cap, not a target — warm starts typically exit well before it.
- **Prediction:** `fast_pred_var` (CG-based) avoids materializing the O(n²) covariance matrix.
- **Device:** auto-detects CUDA; falls back to CPU. `device="cpu"` used in all tests.

### Config / CLI changes
Added to `BenchmarkConfig`:
- `surrogate: str = "rf"` — use `--surrogate gp` to select GP
- `gp_n_iter: int = 200`, `gp_lr: float = 0.01`, `gp_patience: int = 3`
- `validate()` checks `surrogate in ("rf", "gp")`

`benchmark.py` `_run()` now calls `_build_surrogate(cfg)` instead of hardcoding RFSurrogate.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/surrogates/gp.py` | New: `_GPRegressionModel`, `GPSurrogate` |
| `src/rag_al/core/config.py` | Added `surrogate`, `gp_n_iter`, `gp_lr`, `gp_patience`; validate() check |
| `src/rag_al/cli/benchmark.py` | Replaced hardcoded `RFSurrogate` with `_build_surrogate(cfg)` |
| `tests/test_gp_surrogate.py` | New: 10 tests (shapes, sigma ≥ 0, warm-start state, predict-before-fit, end-to-end runner) |

### Tests run
```
pytest tests/test_gp_surrogate.py -v   → 10/10 passed
pytest tests/ -q                       → 64/64 passed
ruff check src/                        → All checks passed
```

### Remaining concerns
- NumericalWarning (`Negative variance values detected. Rounding to 1e-06`) appears when
  predicting with very few labeled points (≤10). This is gpytorch's internal floor — benign
  for Sprint 2. If it persists at larger n_labeled, add a jitter via `gpytorch.settings.cholesky_jitter`.
- Calibration quality (σ vs. actual error) is not validated here — diagnosed at run-time via
  `pool_spearman` once cluster runs complete.

---

## 2026-07-10 — Sprint 2 architecture decisions (GP surrogate design)

**Branch:** `feature/sprint2-repr-surrogate`

### Decisions recorded (no code change)

**GP training: `fit()` vs. separate `Trainer` class**
Decision: keep training logic inside `GPSurrogate.fit()` for Sprint 2.
- The warm-start protocol is inherently stateful across rounds (`_prev_state` persists
  between `fit()` calls). A stateless Trainer loses this; a stateful one is equivalent
  to inlining the loop.
- One training protocol in Sprint 2 — extraction is premature without a second protocol.
- Future: when the emergent property retrospective needs k-fold, add
  `AbstractSurrogateTrainer` with `WarmStartTrainer` / `KFoldGPTrainer` subclasses.

**GP risk clarification**
The feasibility note "standardization and n_iter need empirical tuning" was imprecise:
- Standardization is deterministic (per-dim mean/std from labeled X). No tuning.
- `n_iter=200` is a cap; patience early-stopping makes it adaptive.
- The actual unknowns are calibration quality (does σ meaningfully correlate with
  prediction error?) and gpytorch device quirks. Both diagnosed at run-time:
  `pool_spearman` tracks calibration; device issues surface in the smoke test.

**`sprint2_plan.md` updated** with status table (Steps 0–2b ✅, Step 3 🔄) and
corrected GP risk note.

---

## 2026-07-10 — Sprint 2 Step 2b: PLMPhysicoEncoder and PLMSimpleConcatEncoder

**Branch:** `feature/sprint2-repr-surrogate`

### Task summary
Added two new encoder variants combining PLM embeddings with physicochemical features:

**PLMPhysicoEncoder** (`plm_physico`): per-residue fusion. For each residue i, concats
`[h_i | p_i]` where `h_i` is the ESM-2 hidden state and `p_i` is a 5-dim lookup vector
[hydropathy, charge, MW, polar, aromatic]. Mean-pooled to `(D_esm + 5)`. Separate cache
`cache_{model}_physico.pkl` keyed by `sha256(seq)`.

**PLMSimpleConcatEncoder** (`plm_concat`): post-hoc sequence-level concat of ESM-2
mean-pool and the 29-dim PhysicochemicalEncoder output. Output dim: `D_esm + 29`.
The physico scaler is fit on the labeled set only (leakage-safe).

### Design note — feature scale and GP normalization
The physico features (hydropathy, MW, etc.) are on different scales than ESM-2 hidden
states. For the RF surrogate this is benign (trees are scale-invariant). For the GP
surrogate (Step 3), the GP input standardization (per-dimension X_mean/X_std in
`GPSurrogate.fit()`) will handle this at training time — no need to normalize in the
encoder. The per-residue physico vector in `PLMPhysicoEncoder` is raw (not pre-scaled);
this is correct because the GP standardizes the full fused vector, not each sub-part.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/representations/plm_physico.py` | New: `_sequence_to_physico()`, `PLMPhysicoEncoder`, `PLMSimpleConcatEncoder` |
| `src/rag_al/core/config.py` | Added `"plm_physico"`, `"plm_concat"` to `REPRESENTATIONS` |
| `src/rag_al/cli/benchmark.py` | Added cases for `"plm_physico"` and `"plm_concat"` in `_build_encoder()` |
| `tests/test_plm_physico.py` | New: 17 tests covering per-residue properties, output shapes, cache reuse, concat width, and fit-before-transform guard |

### Tests run
```
pytest tests/test_plm_physico.py -v   → 17/17 passed
pytest tests/ -v                      → 54/54 passed
ruff check src/                       → All checks passed
```

---

## 2026-07-09 — Sprint 2 Step 2a: plm_site mode in ESMEncoder

**Branch:** `feature/sprint2-repr-surrogate`

### Task summary
Added `mode='site'` to `ESMEncoder`. Instead of mean-pooling all residue hidden states,
site mode extracts only the ESM-2 hidden states at the mutated residue positions and
averages across sites. Designed for single-site datasets; for multi-site combinatorial
(e.g., GB1 4-site), `mode='delta'` is the recommended baseline since site-averaging
across 4 distant positions may wash out signal.

### Design decisions
- **Separate cache**: site embeddings keyed by `sha256(seq + "::" + mutant_str)` in
  `cache_{model}_site.pkl`. Cannot reuse the mean-pool cache since the extracted vector
  depends on which positions are mutated, not just the sequence.
- **Token indexing**: ESM-2 prepends `<cls>` at token 0, so 0-indexed residue position
  `p` maps to token index `p+1` in `last_hidden_state`.
- **`mutant` column required**: `transform()` raises `ValueError` with a clear message
  if the `mutant` column is absent. `labeled_df` and `pool_df` both include `mutant`
  (it is in `_FEATURE_COLS`), so this is only a guard for misuse.
- **Multi-site**: averaging across sites is the first-pass approach; dedicated per-site
  delta (`h_mut[i] - h_wt[i]`) is deferred to a follow-up.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/representations/plm.py` | Added `import re`, `_parse_mutant_positions()` helper, `_site_cache` / `_site_cache_path()` / `_load_site_cache()` / `_save_site_cache()`, `_embed_sequences_site()`, `_transform_site()` helper, updated `__init__` mode validation, updated `transform()` dispatch |
| `src/rag_al/core/config.py` | Added `"plm_site"` to `REPRESENTATIONS` tuple |
| `src/rag_al/cli/benchmark.py` | Added `"plm_site"` case in `_build_encoder()` |
| `tests/test_plm_site.py` | New: 12 tests covering position parsing, output shape/dtype, correct token extraction, multi-site averaging, missing-mutant error, disk cache reuse, and invalid mode guard |

### Tests run
```
pytest tests/test_plm_site.py -v   → 12/12 passed
pytest tests/ -v                   → 37/37 passed
ruff check src/                    → All checks passed
```

### Remaining concerns
- For multi-site (GB1, GFP), try this mode first. If it underperforms `plm_delta`,
  implement per-site delta (`h_mut[i] - h_wt[i]`) as a follow-up.
- `_embed_sequences_site()` iterates over the batch dimension in Python after the GPU
  forward pass (for per-variant position indexing). This is slightly slower than the
  fully vectorized mean-pool path but correct and cache-friendly.

---

## 2026-06-17 — PLM cache: replace order-validated array cache with seq-hash dict

**Branch:** `fix/plm-cache-hashmap`

### Task summary
`ESMEncoder` cached embeddings as a dense `.npy` array validated by an exact, ordered
list of `variant_ids`. Any subset call (the AL loop calls `transform(labeled_df)` and
`transform(pool_df)` every round with different slices) failed the ordering check and
re-ran the ESM model from scratch, making the `rag-embed` pre-compute step useless.
Additionally, `transform()` had a `SyntaxError` (bare `for` on line 262) from an
incomplete earlier fix, making any cache-hit path crash immediately.

The fix replaces the two-file array cache (`embeddings_*.npy` + `variant_ids_*.npy`)
with a single `{sha256(sequence) → embedding}` pickle dict (`cache_*.pkl`). Subset
calls now look up each hash individually and only call `_embed_sequences()` for misses.
The WT embedding for delta mode is stored in the same dict via `fit()`. The in-memory
`_embedding_cache` dict is loaded once and kept alive for the lifetime of the encoder
instance; `_save_cache()` writes to disk only when new embeddings are computed.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/representations/plm.py` | Replaced `_cache_path`/`_ids_cache_path`/`_try_load_cache`/`_save_cache` with `_cache_path`/`_seq_hash`/`_load_cache`/`_save_cache`; rewrote `transform()` and updated `fit()` for delta WT caching; added `hashlib`/`pickle` imports |
| `tests/test_esm_cache.py` | New: 7 tests covering `cache_dir=None`, full compute+save, full hit, partial miss, delta WT caching, and row-order preservation |

### Tests run

```
pytest tests/test_esm_cache.py -v   →  7/7 passed
pytest tests/ -v                    →  18/18 passed
ruff check src/rag_al/representations/plm.py  →  All checks passed
mypy src/rag_al/representations/plm.py        →  Same pre-existing lazy-import errors; no new errors
```

### Remaining concerns
- Pickle format is opaque and version-sensitive; a future migration to `.npz` or HDF5
  would be needed for very large datasets or cross-Python-version portability.
- Existing `.npy`/`_ids.npy` cache files from prior `rag-embed` runs are silently
  ignored (no migration). Re-run `rag-embed` to populate the new `.pkl` cache.

---

## 2026-06-14 — Fix Bug #3 (self-neighbor in retrieval) and Bug #4 (dead code)

**Branch:** `fix/bug3-retrieval-self-neighbor-bug4-dead-code`
**Commit:** `1a95adf`

### Task summary
- **Bug #3 (Medium):** `RetrievalAugmentedEncoder.transform(labeled_df)` included each labeled point as its own nearest neighbor (distance = 0) during surrogate training. `mean_y`, `std_y`, and `max_y` retrieval features for labeled points were self-referential while pool features were clean, creating a train/predict asymmetry that distorted surrogate calibration for `plm_retrieval`. The `_dist_scale` normalization in `fit()` was also biased — self-distance zeros pulled the median toward 0.
- **Bug #4 (Minor):** Dead code line in `physicochemical.py`: `net_charge += _AA_HYDROPATHY.get(aa, 0.0) * 0.0` always evaluates to `0.0`. Deleted.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/representations/base.py` | Added `transform_labeled()` concrete method (default: delegates to `transform()`); updated `fit_transform()` to call `transform_labeled()` |
| `src/rag_al/representations/retrieval.py` | Added `transform_labeled()` override with `exclude_self=True`; updated `_retrieval_features()` to accept `exclude_self` kwarg; fixed `fit()` `_dist_scale` to exclude self column |
| `src/rag_al/loop/runner.py` | Changed `encoder.transform(labeled_df)` → `encoder.transform_labeled(labeled_df)` |
| `src/rag_al/representations/physicochemical.py` | Deleted dead code line |
| `tests/test_retrieval_self_neighbor.py` | New: 5 tests covering Bug #3 fix |
| `tests/test_runner_batch_y.py` | Added `transform_labeled()` to `_IdentityEncoder` stub |

### Tests run

```
pytest tests/ -v   →  11/11 passed (1.27s)
ruff check src/    →  All checks passed
mypy src/          →  Same 28 pre-existing errors; no new errors
```

### Remaining concerns
- **ESMEncoder cache** — Cache key mismatch means embeddings are recomputed every round. Performance issue; not correctness.

---

## 2026-06-14 — Fix Bug #1, Bug #2, Gap #1

**Branch:** `fix/reveal-result-selection-loggin`
**Commit:** `407fcf8`

### Task summary
- **Bug #1 (Critical):** `batch_y` in `runner.py` was extracted via `labeled_y[-batch_size:]` after `reveal()`. Because `labeled_y` is ordered by original dataset row index (not insertion order), this slice returned the wrong variants' fitness values silently every round, corrupting `batch_mean_fitness` metrics.
- **Bug #2 (Medium):** `runner.py` accessed `dataset._df` directly for `wt_sequence` and `batch_sequences`. `_df` contains the full DataFrame including the `fitness` column — a latent leakage risk if any future edit reads `_df["fitness"]`.
- **Gap #1 (Completeness):** No record of which specific variants were selected each round, making experiments non-auditable post-hoc.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/data/al_dataset.py` | Added `wt_sequence` property, `get_sequences()`, `get_variant_ids()`, `fitness_at()` |
| `src/rag_al/loop/runner.py` | Fixed `batch_y` extraction; replaced `_df` access with public helpers; added `selections_rows` logging |
| `src/rag_al/cli/benchmark.py` | Unpacked `(results_df, selections_df)` return; writes `seed_N_selections.csv` |
| `src/rag_al/core/logging.py` | (scaffolded for structured logging) |
| `src/rag_al/core/paths.py` | (scaffolded path helpers) |
| `src/rag_al/representations/retrieval.py` | Minor adjustments from scaffold |
| `src/rag_al/loop/metrics.py` | Minor adjustments from scaffold |
| `tests/test_runner_batch_y.py` | New: 6 tests covering Bug #1 fix and Gap #1 selection logging |
| `CLAUDE.md` | Updated architecture/bug docs |
| `docs/workflow.md` | Added workflow reference |

### Reason for change
Bug #1 silently corrupts a metric logged every round. Bug #2 is a latent leakage risk that would undermine the scientific integrity of the benchmark. Gap #1 is required to reproduce or audit any experiment.

### Tests run

```
pytest tests/ -v
```

**Result: 6/6 passed (2.21s)**

| Test | Status |
|------|--------|
| `test_batch_y_matches_global_selected` | PASSED |
| `test_selections_shape` | PASSED |
| `test_selections_global_indices_not_in_initial_labeled` | PASSED |
| `test_selections_global_indices_unique_across_rounds` | PASSED |
| `test_variant_ids_consistent_with_global_indices` | PASSED |
| `test_results_shape` | PASSED |

```
ruff check src/
```
**Result: All checks passed**

```
mypy src/
```
**Result: 28 errors — all pre-existing scaffold issues (not introduced by this fix)**

Known mypy issues (documented, not blocking):
- `import-untyped` for `pandas`, `sklearn`, `transformers` — missing stubs in environment; install `pandas-stubs` to resolve
- `union-attr` / `attr-defined` / `misc` errors in `plm.py` — scaffold's lazy `torch`/`transformers` import pattern uses `Optional` assignments that mypy cannot narrow through `if torch is not None` guards
- `index` / `union-attr` in `retrieval.py:130–134` — `_knn` and `_labeled_embeddings` declared as `Optional` but accessed without narrowing after `fit()`

### Remaining concerns
- **Bug #3** — `RetrievalAugmentedEncoder` self-neighbor inclusion when `transform(labeled_df)` called. Not blocking correctness for non-retrieval representations but distorts surrogate calibration for `plm_retrieval`. Next fix.
- **Bug #4** — Dead code in `physicochemical.py` (`* 0.0`). Minor; no correctness impact.
- **ESMEncoder cache** — Cache key mismatch means embeddings are recomputed every round. Performance issue; not correctness.

---

## 2026-06-25 — Multi-dataset benchmark setup and local sweep

### Task summary

Set up the full 6-dataset benchmark panel and ran non-PLM + PLM experiments locally
using the M3 MacBook (MPS GPU, 12 cores).

### Files changed

- `src/rag_al/core/config.py` — `data_dir` default changed to `data/curated/`; added
  `rf_n_jobs: int = 1` field so parallel local runs don't oversubscribe cores
- `src/rag_al/cli/benchmark.py` — pass `cfg.rf_n_jobs` to `RFSurrogate`
- `src/rag_al/cli/embed.py` — `data_dir` default changed to `data/curated/`
- `scripts/curate_proteingym.py` — new: converts ProteinGym substitution CSVs to
  5-column pipeline schema via WT reconstruction from mutant strings
- `scripts/run_local.py` — new: parallel local runner using `ThreadPoolExecutor`;
  accepts `--dataset`, `--reprs`, `--acqs`, `--n_seeds`, `--n_rounds`,
  `--batch_size`, `--ucb_beta`, `--esm_model`, `--workers`
- `scripts/plot_aggregate.py` — new: cross-dataset heatmap (repr × acq),
  difficulty-split bar chart, per-dataset grouped bar chart
- `scripts/submit_benchmark.sh` — rewritten: one CPU job per dataset using
  `run_local.py --workers 48` instead of 450-task SLURM array
- `scripts/submit_embed.sh` — accepts dataset as positional arg; updated paths
- `CLAUDE.md` — updated example commands to use full dataset names
- `data/curated/` — curated all 6 datasets (989–37708 variants each)
- `data/embeddings/` — ESM2-8M caches for 5 PLM-compatible datasets

### Reason for change

Pipeline previously had no real datasets or benchmark infrastructure. This adds
the full experimental scaffold needed to evaluate AL methods across diverse fitness
landscapes before cluster submission.

### Key findings from local runs (non-PLM, n_rounds=20, batch_size=128)

- AL signal is clear on PABP_YEAST and BLAT_ECOLX_Stiffler: UCB/greedy
  batch_mean_fitness is consistently higher than random; best_fitness improves
  monotonically; several acquisitions reach regret=0.000.
- BLAT_ECOLX_Deng_2012 and Jacquier_2013 have very discrete fitness landscapes
  (rational-fraction values, large neutral cluster at 0.0) — batch_mean_fitness
  stays negative because most variants are deleterious; best_fitness still
  improves meaningfully for non-random acquisitions.
- β=0.1 tested on Deng: worse than β=1.0 (regret 0.327 vs 0.109). Late-round
  batch quality decline is pool depletion, not excess exploration.
- PLM (ESM2-8M, MPS): 45 cells for Jacquier in ~2 min wall time. PLM embeddings
  do not help on discrete single-mutant landscapes; greedy + PLM is worse than
  random (overconfident exploitation in embedding space).

### Cluster deployment design

- Embed: one GPU job per dataset (`submit_embed.sh $DATASET`)
- Benchmark: one 48-core CPU job per dataset (`submit_benchmark.sh $DATASET`)
  → 10 total SLURM jobs instead of 450 array tasks
- PLM cells are cache-hit-only after embed; no GPU needed for benchmark step

### Tests run

```
conda run -n torch-protein-M1 pytest tests/ -v
```
**Result: 24/24 passed**

```
ruff check src/
```
**Result: All checks passed**

### Remaining concerns

- BRCA1_HUMAN_Findlay_2018 (WT len 1863) excluded from PLM representations.
  ESM-2 hard limit is 1022 residues. To include BRCA1 with PLM, would need
  a long-context model (ESM-3) or sequence truncation.
- PLM (ESM2-8M) results for larger datasets (Deng, Firnberg, Stiffler, PABP)
  pending — running now.
- All results are with ESM2-8M (320-dim). Cluster run should use 650M (1280-dim)
  for higher-quality embeddings.

---
