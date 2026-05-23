#!/bin/bash
#SBATCH --job-name=rag_bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # RF parallelism
#SBATCH --mem=16G
#SBATCH --array=0-89            # 5 reprs × 6 acqs × 3 seeds = 90 cells

# -----------------------------------------------------------------------
# rag-benchmark: Full benchmark sweep as a SLURM array job.
# Each task runs one (representation × acquisition × seed) cell.
#
# Prerequisites:
#   1. Run submit_embed.sh first to cache PLM embeddings.
#   2. Place curated dataset CSV at data/<DATASET>.csv.
#
# Usage:
#   sbatch scripts/submit_benchmark.sh
#
# Adjust DATASET, N_ROUNDS, BATCH_SIZE, N_SEEDS below.
# -----------------------------------------------------------------------

set -euo pipefail

# ---- Experiment grid ----------------------------------------------------
DATASET="BLAT_ECOLX"
N_INIT=50
N_ROUNDS=5
BATCH_SIZE=20
N_SEEDS=3
ESM_MODEL="facebook/esm2_t33_650M_UR50D"
UCB_BETA=1.0

REPRS=(mutation physicochemical plm_mean plm_delta plm_retrieval)
ACQS=(random greedy ucb diversity_ucb retrieval_ucb)

N_REPRS=${#REPRS[@]}   # 5
N_ACQS=${#ACQS[@]}     # 6
# Total tasks = N_REPRS × N_ACQS × N_SEEDS = 5 × 6 × 3 = 90

# ---- Decode array index -------------------------------------------------
# Index layout: task_id = repr_idx * (N_ACQS * N_SEEDS) + acq_idx * N_SEEDS + seed
TASK_ID=${SLURM_ARRAY_TASK_ID}
repr_idx=$(( TASK_ID / (N_ACQS * N_SEEDS) ))
acq_idx=$(( (TASK_ID / N_SEEDS) % N_ACQS ))
seed=$(( TASK_ID % N_SEEDS ))

REPR=${REPRS[$repr_idx]}
ACQ=${ACQS[$acq_idx]}

# ---- Environment setup --------------------------------------------------
module purge
module load anaconda3/2023.3           # adjust to cluster
conda activate rag_al

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "============================================"
echo "Job array: $SLURM_ARRAY_JOB_ID  task: $TASK_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Cell:  repr=$REPR  acq=$ACQ  seed=$seed"
echo "============================================"

rag-benchmark \
    --dataset        "$DATASET" \
    --representation "$REPR" \
    --acquisition    "$ACQ" \
    --seed           "$seed" \
    --n_init         "$N_INIT" \
    --n_rounds       "$N_ROUNDS" \
    --batch_size     "$BATCH_SIZE" \
    --esm_model      "$ESM_MODEL" \
    --ucb_beta       "$UCB_BETA"

echo "Task $TASK_ID complete."
