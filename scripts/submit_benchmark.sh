#!/bin/bash
#SBATCH --job-name=rag_bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8       # RF parallelism
#SBATCH --mem=16G
#SBATCH --array=0-449           # 6 datasets × 5 reprs × 5 acqs × 3 seeds = 450 cells

# -----------------------------------------------------------------------
# rag-benchmark: Full benchmark sweep as a SLURM array job.
# Each task runs one (dataset × representation × acquisition × seed) cell.
#
# Prerequisites:
#   1. Run submit_embed.sh for each PLM-compatible dataset to cache embeddings.
#   2. Curated CSVs must exist in data/curated/<DATASET>.csv.
#
# Usage:
#   sbatch --array=0-449 scripts/submit_benchmark.sh
#
# Dataset × representation constraints:
#   BRCA1_HUMAN_Findlay_2018 — WT length 1863 exceeds ESM-2 limit (1022).
#   PLM representations (plm_mean, plm_delta, plm_retrieval) will fail for
#   BRCA1. The script skips those cells and exits cleanly (see below).
# -----------------------------------------------------------------------

set -euo pipefail

# ---- Experiment grid ----------------------------------------------------
DATASETS=(
    BLAT_ECOLX_Jacquier_2013
    BLAT_ECOLX_Deng_2012
    BLAT_ECOLX_Firnberg_2014
    BLAT_ECOLX_Stiffler_2015
    BRCA1_HUMAN_Findlay_2018
    PABP_YEAST_Melamed_2013
)
REPRS=(mutation physicochemical plm_mean plm_delta plm_retrieval)
ACQS=(random greedy ucb diversity_ucb retrieval_ucb)

N_DATASETS=${#DATASETS[@]}  # 6
N_REPRS=${#REPRS[@]}        # 5
N_ACQS=${#ACQS[@]}          # 5
N_SEEDS=3
# Total = 6 × 5 × 5 × 3 = 450

N_INIT=50
N_ROUNDS=10
BATCH_SIZE=10
ESM_MODEL="facebook/esm2_t33_650M_UR50D"
UCB_BETA=1.0

# ---- Decode array index -------------------------------------------------
# Layout: task_id = ds_idx*(N_REPRS*N_ACQS*N_SEEDS) + repr_idx*(N_ACQS*N_SEEDS) + acq_idx*N_SEEDS + seed
TASK_ID=${SLURM_ARRAY_TASK_ID}
ds_idx=$(( TASK_ID / (N_REPRS * N_ACQS * N_SEEDS) ))
repr_idx=$(( (TASK_ID / (N_ACQS * N_SEEDS)) % N_REPRS ))
acq_idx=$(( (TASK_ID / N_SEEDS) % N_ACQS ))
seed=$(( TASK_ID % N_SEEDS ))

DATASET=${DATASETS[$ds_idx]}
REPR=${REPRS[$repr_idx]}
ACQ=${ACQS[$acq_idx]}

# ---- Skip PLM cells for BRCA1 (sequence too long for ESM-2) ------------
if [[ "$DATASET" == "BRCA1_HUMAN_Findlay_2018" ]] && \
   [[ "$REPR" == plm_mean || "$REPR" == plm_delta || "$REPR" == plm_retrieval ]]; then
    echo "Skipping PLM representation for BRCA1_HUMAN_Findlay_2018 (WT len 1863 > ESM-2 limit 1022)."
    exit 0
fi

# ---- Environment setup --------------------------------------------------
module purge
module load anaconda3/2023.3           # adjust to cluster
conda activate rag_al

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "============================================"
echo "Job array: $SLURM_ARRAY_JOB_ID  task: $TASK_ID"
echo "Node:  $SLURMD_NODENAME"
echo "Cell:  dataset=$DATASET  repr=$REPR  acq=$ACQ  seed=$seed"
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
