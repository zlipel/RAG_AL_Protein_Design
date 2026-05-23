# scripts/

Cluster submission scripts and post-processing utilities. These are
standalone scripts that sit outside the Python package.

---

## submit_embed.sh

SLURM batch script for pre-computing ESM-2 embeddings on a GPU node.

### Resources requested
- 1 GPU (A100 on Della; adjust #SBATCH --gres for Tiger H100)
- 8 CPUs, 32GB memory
- 2 hour time limit

### What to configure before submitting
At the top of the script, set:
    DATASET         — dataset name (must match a CSV in data/)
    ESM_MODEL       — HuggingFace model ID for ESM-2
    EMBED_BATCH_SIZE — sequences per forward pass (64 is good for A100)

Also update the module load and conda activate lines to match your
cluster environment.

### Submit
    sbatch scripts/submit_embed.sh

Run this ONCE per dataset before submitting the benchmark sweep. The
embeddings are saved to data/embeddings/<dataset>/ and reused by all
subsequent benchmark runs.

---

## submit_benchmark.sh

SLURM array job that sweeps all (representation × acquisition × seed)
combinations in parallel. Each array task is one cell of the grid.

### Array job layout
Default grid: 5 representations × 6 acquisitions × 3 seeds = 90 tasks
    #SBATCH --array=0-89

Each task's array index is decoded as:
    repr_idx = TASK_ID / (N_ACQS * N_SEEDS)
    acq_idx  = (TASK_ID / N_SEEDS) % N_ACQS
    seed     = TASK_ID % N_SEEDS

So tasks 0-2 are (mutation, random, seeds 0-2), tasks 3-5 are
(mutation, greedy, seeds 0-2), and so on.

### Resources per task
- 8 CPUs (for RandomForest n_jobs=-1), 16GB memory, 4 hours

### What to configure before submitting
At the top of the script, set:
    DATASET     — must match a CSV in data/
    N_ROUNDS    — AL rounds per run (default: 5)
    BATCH_SIZE  — variants acquired per round (default: 20)
    N_SEEDS     — seeds per (repr, acq) combination (default: 3)
    ESM_MODEL   — should match the model used in submit_embed.sh

To run a smaller test sweep (e.g., just 4 repr × 4 acq × 1 seed = 16 tasks),
adjust the REPRS and ACQS arrays and set --array=0-15.

### Submit (after embed job completes)
    sbatch scripts/submit_benchmark.sh

### Checking progress
    squeue -u $USER                      # see running tasks
    cat logs/bench_<JOBID>_<TASKID>.out  # see output for one task

Results appear in results/<dataset>/<tag>/seed_<N>.csv as each task finishes.

---

## plot_results.py

Post-processing script that aggregates all seed CSVs and generates
learning-curve figures. Run this locally after syncing results from the
cluster.

### What it produces
For each metric (best_fitness, simple_regret, topk10_recall, topk50_recall,
batch_mean_fitness), it generates two figures:

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
reads the dataset, representation, acquisition, and seed columns written
by rag-benchmark, and groups by (representation, acquisition, round) to
compute mean and std across seeds. Shaded bands show ±1 std.

### Syncing results from cluster
    rsync -av della:/path/to/RAG_AL_Protein_Design/results/ ./results/
    python scripts/plot_results.py --dataset BLAT_ECOLX
