# scripts/

Cluster submission scripts and post-processing utilities. These are
standalone scripts that sit outside the Python package.

---

## submit_embed.sh  (+ run_embed.sh)

SLURM batch script for pre-computing PLM embeddings on a GPU node. Takes the
dataset as a positional arg: `sbatch scripts/submit_embed.sh <DATASET>`.

### Resources requested
- 1 GPU (A100 on Della; adjust #SBATCH --gres for Tiger H100)
- 8 CPUs, 32GB memory
- 2 hour time limit (override per-job with `sbatch --time=...`)

### What it computes
Runs `rag-embed --modes mean delta site physico`, precomputing every PLM cache
the benchmark grid needs. Each mode family writes a distinct file under
`data/embeddings/<dataset>/`:
- `cache_{model}.pkl` — mean/delta (also serves plm_retrieval, plm_concat)
- `cache_{model}_site.pkl` — site
- `cache_{model}_physico.pkl` — physico

Set `ESM_MODEL` / `EMBED_BATCH_SIZE` at the top; update module/conda lines for
your cluster. Run ONCE per dataset before the benchmark sweep.

### Submit all PLM datasets at once
    bash scripts/run_embed.sh

Loops the 6 smaller PLM datasets at the default 2h and submits **GB1**
(`SPG1_STRSG_Wu_2016`) separately at 8h — it does 3 heavy forward passes over
~149K long sequences. Override with `GB1_WALLTIME=12:00:00 bash scripts/run_embed.sh`.
BRCA1 is excluded (WT 1863 AA > ESM-2 limit).

---

## submit_benchmark.sh  (+ run_benchmark.sh)

Runs all `(representation × acquisition × seed)` cells for **one** dataset on a
single 48-core node via **GNU parallel** (not a SLURM array). Falls back to
`run_local.py` if GNU parallel is unavailable.

### Grid
Up to 8 representations × 5 acquisitions × 3 seeds. PLM reps are **auto-dropped**
for any dataset whose `max(len(mutated_sequence)) > 1022` (ESM-2 limit) — probed
from the CSV at submit time, so BRCA1 (and any future long dataset) runs only
mutation/physicochemical.

### Resources
- 48 CPUs, 64GB, 4h per dataset job (no GPU — embeddings are cache hits)

### What to configure
`N_ROUNDS`, `BATCH_SIZE`, `N_SEEDS`, `N_INIT`, `UCB_BETA`, `ESM_MODEL` at the top
(should match the embed job). Pass the dataset as a positional arg.

### Submit
    sbatch scripts/submit_benchmark.sh <DATASET>   # one dataset
    bash   scripts/run_benchmark.sh                # all datasets, one job each

### Checking progress
    squeue -u $USER
    cat logs/bench_<JOBID>_<JOBNAME>.out

Results appear in `results/<dataset>/<tag>/seed_<N>.csv`, where
`<tag> = <repr>_<acq>[_b<beta>][_<surrogate>]` (surrogate suffix omitted for RF).

---

## submit_gp_benchmark.sh

Targeted **GP-only** benchmark: `GPSurrogate` on PABP + BLAT_Deng (3 reprs × 2
acqs × 3 seeds = 36 cells). Motivated by the PABP calibration anomaly. The RF
baseline for the same cells comes from the main sweep, so RF is **not** re-run
here; GP results land in `_gp`-suffixed dirs and never overwrite RF. The same
length-based PLM guard applies, so it's safe to extend `DATASETS`/`REPRS`.

Runs on CPU. Requests a whole node (`--exclusive`) and launches each cell as its
own `srun --exclusive` step with a per-cell memory cgroup, so a cell that OOMs is
killed in isolation rather than taking down the grid, and `sacct` reports per-cell
usage. Env-overridable knobs: `MEM_PER_CELL` (default `8G` — raise for larger
pools), `CPUS_PER_CELL` (default `1`), `MAX_CONCURRENT` (default = node cores).

    sbatch scripts/submit_gp_benchmark.sh

---

## plot_results.py

Post-processing script that aggregates all seed CSVs and generates
learning-curve figures. Run this locally after syncing results from the
cluster.

### What it produces
For each metric (best_fitness, simple_regret, topk10_recall, topk50_recall,
batch_mean_fitness, pool_spearman), it generates two figures:

1. <metric>_by_acq.png
   One subplot per acquisition function. Each subplot shows all
   representations as different colored lines (mean ± std shaded band).
   Use this to see: for a given acquisition strategy, which representation
   performs best?

2. <metric>_by_repr.png
   One subplot per representation. Each subplot shows all acquisition
   functions as different colored lines.
   Use this to see: for a given representation, which acquisition strategy
   performs best?

### Usage
    python scripts/plot_results.py --dataset BLAT_ECOLX --output_dir figures/

### Flags
--dataset       Dataset name (must match the subdirectory under results/)
--results_dir   Where to find seed CSVs (default: results/)
--output_dir    Where to save PNG figures (default: figures/)

### How aggregation works
The script finds all files matching results/<dataset>/**/seed_*.csv,
reads the dataset, representation, acquisition, surrogate, and seed columns
written by rag-benchmark, and groups by (representation, acquisition, round) to
compute mean and std across seeds. Shaded bands show ±1 std.

Pass `--surrogate rf|gp` to select which surrogate's curves to plot (default
`rf`); GP filenames get a `_gp` suffix so RF and GP figures never collide. For a
paired GP-vs-RF comparison on the same axes, use `plot_gp_vs_rf.py` (below).

### Syncing results from cluster
    # results/ and results_*/ are gitignored. The current results/ holds the
    # 650M full grid; old 8M prototypes live in results_sprint1_8M/.
    rsync -avz <cluster>:<proj>/results/ ./results/
    python scripts/plot_results.py --dataset BLAT_ECOLX_Deng_2012

---

## plot_gp_vs_rf.py

Paired **surrogate-variant** comparison for the targeted GP grid. Compares each
(dataset, representation) cell across variants, **matched on the acquisitions
every compared variant ran** (the GP grid omits random/diversity/retrieval, so
this avoids comparing GP's {greedy,ucb} against RF's five). Per metric it prints
a per-variant table (with a delta column for a 2-variant compare) and saves one
grouped-bar figure faceted by dataset.

**Variants** are `rf`, `gp` (isotropic), `gp_ard`. The CSV `surrogate` column is
`gp` for both GP kernels, so the variant is **derived from the result-dir tag**
(`_gp` vs `_gp_ard`). Pick with `--variants` (default: all present).

### Usage
    # RF vs isotropic GP (all present variants)
    python scripts/plot_gp_vs_rf.py \
        --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012 \
        --metrics topk10_recall simple_regret pool_spearman best_fitness \
        --output_dir docs/figures/gp_vs_rf

    # ARD A/B: isotropic vs ARD (delta = gp_ard − gp)
    python scripts/plot_gp_vs_rf.py --variants gp_ard gp \
        --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012

`plot_aggregate.py` applies the same tag-derivation, so `--surrogate gp_ard`
works there too. See `docs/sprint3_gp_results.md` for the isotropic findings.
