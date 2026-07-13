#!/bin/bash
#SBATCH --job-name=rag_bench
#SBATCH --output=logs/bench_%j_%x.out
#SBATCH --error=logs/bench_%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
# No GPU needed — embeddings loaded from cache pre-computed by submit_embed.sh

# -----------------------------------------------------------------------
# Run all (representation × acquisition × seed) cells for ONE dataset on
# a single 48-core node. Dispatch via GNU parallel (preferred) or
# run_local.py (fallback).
#
# Why not srun?  srun is the right tool when spreading tasks across
# multiple nodes. Here we fill one node's 48 cores with independent
# Python processes — GNU parallel handles the batching natively without
# SLURM step overhead.
#
# Prerequisites:
#   1. submit_embed.sh already ran for this dataset.
#   2. Curated CSV exists at data/curated/<DATASET>.csv.
#
# Usage:
#   for D in BLAT_ECOLX_Jacquier_2013 BLAT_ECOLX_Deng_2012 \
#             BLAT_ECOLX_Firnberg_2014 BLAT_ECOLX_Stiffler_2015 \
#             PABP_YEAST_Melamed_2013 BRCA1_HUMAN_Findlay_2018 \
#             GFP_AEQVI_Sarkisyan_2016 SPG1_STRSG_Wu_2016; do
#       sbatch --job-name="bench_$D" scripts/submit_benchmark.sh $D
#   done
# -----------------------------------------------------------------------

set -eo pipefail

DATASET="${1:-BLAT_ECOLX_Jacquier_2013}"
N_ROUNDS=20
BATCH_SIZE=128
N_SEEDS=3
N_INIT=50
UCB_BETA=1.0
ESM_MODEL="facebook/esm2_t33_650M_UR50D"
WORKERS=${SLURM_CPUS_PER_TASK:-48}

# BRCA1 sequences exceed the ESM-2 1022-residue limit
if [[ "$DATASET" == "BRCA1_HUMAN_Findlay_2018" ]]; then
    REPRS=(mutation physicochemical)
else
    REPRS=(mutation physicochemical plm_mean plm_delta plm_retrieval)
fi
ACQS=(random greedy ucb diversity_ucb retrieval_ucb)

# --- Environment setup ---------------------------------------------------
module purge
module load anaconda3/2024.6
conda activate rag_al

# SLURM_SUBMIT_DIR is set by sbatch to wherever the user ran sbatch from.
# Always submit from the project root: cd /home/.../rag_pipeline && sbatch scripts/submit_benchmark.sh
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. Submit with sbatch from the project root." >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(pwd)"

# Absolute paths — safe regardless of subprocess cwd
DATA_DIR="${PROJECT_ROOT}/data/curated"
EMBED_CACHE_DIR="${PROJECT_ROOT}/data/embeddings"
RESULTS_DIR="${PROJECT_ROOT}/results"

# Use cached model weights only — compute nodes have no outbound internet.
# HF_HOME defaults to ~/.cache/huggingface/ (home). Override in ~/.bashrc if needed.
export HF_HUB_OFFLINE=1

echo "============================================"
echo "Job: $SLURM_JOB_ID   Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET"
echo "Reprs: ${REPRS[*]}   Acqs: ${ACQS[*]}"
echo "n_rounds=$N_ROUNDS  batch_size=$BATCH_SIZE  n_seeds=$N_SEEDS  workers=$WORKERS"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

# Build the full command list (one line per cell)
CMDS=()
for repr in "${REPRS[@]}"; do
    for acq in "${ACQS[@]}"; do
        for seed in $(seq 0 $((N_SEEDS - 1))); do
            CMDS+=("rag-benchmark \
--dataset $DATASET \
--representation $repr \
--acquisition $acq \
--seed $seed \
--n_rounds $N_ROUNDS \
--batch_size $BATCH_SIZE \
--n_init $N_INIT \
--ucb_beta $UCB_BETA \
--esm_model $ESM_MODEL \
--rf_n_jobs 1 \
--data_dir ${DATA_DIR} \
--embed_cache_dir ${EMBED_CACHE_DIR} \
--results_dir ${RESULTS_DIR}")
        done
    done
done

N_CELLS=${#CMDS[@]}
echo "Submitting $N_CELLS cells with $WORKERS workers"

if command -v parallel &>/dev/null; then
    # GNU parallel — preferred on HPC clusters
    printf '%s\n' "${CMDS[@]}" | parallel --jobs "$WORKERS" --halt soon,fail=1
else
    # Fallback: run_local.py (works anywhere Python is available)
    echo "GNU parallel not found — falling back to run_local.py"
    python scripts/run_local.py \
        --dataset    "$DATASET" \
        --reprs      "${REPRS[@]}" \
        --n_rounds   "$N_ROUNDS" \
        --batch_size "$BATCH_SIZE" \
        --n_init     "$N_INIT" \
        --n_seeds    "$N_SEEDS" \
        --ucb_beta   "$UCB_BETA" \
        --esm_model  "$ESM_MODEL" \
        --workers    "$WORKERS"
fi

echo "Benchmark complete for $DATASET."
