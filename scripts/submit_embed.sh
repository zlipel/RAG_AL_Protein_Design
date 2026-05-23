#!/bin/bash
#SBATCH --job-name=rag_embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1          # 1 A100 on Della; adjust for Tiger

# -----------------------------------------------------------------------
# rag-embed: Pre-compute ESM-2 embeddings for one dataset (GPU job).
# Run this ONCE before submitting the benchmark sweep.
#
# Usage:
#   sbatch scripts/submit_embed.sh
#
# Adjust DATASET and ESM_MODEL below, then submit.
# -----------------------------------------------------------------------

set -euo pipefail

DATASET="BLAT_ECOLX"                          # dataset CSV stem in data/
ESM_MODEL="facebook/esm2_t33_650M_UR50D"      # use 8M model for testing
DATA_DIR="data"
EMBED_CACHE_DIR="data/embeddings"
EMBED_BATCH_SIZE=64                            # increase for A100/H100

# --- Environment setup (adjust to your cluster module/conda setup) -----
module purge
module load anaconda3/2023.3                   # or your Python module
conda activate rag_al                          # your conda env name

# Navigate to project root (assumes script submitted from project root)
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "============================================"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET"
echo "Model: $ESM_MODEL"
echo "============================================"

rag-embed \
    --dataset          "$DATASET" \
    --data_dir         "$DATA_DIR" \
    --embed_cache_dir  "$EMBED_CACHE_DIR" \
    --esm_model        "$ESM_MODEL" \
    --embed_batch_size "$EMBED_BATCH_SIZE" \
    --modes mean delta

echo "Embedding job complete."
