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
# Run this ONCE per dataset before submitting the benchmark sweep.
#
# Usage (submit one job per PLM-compatible dataset):
#   for D in BLAT_ECOLX_Jacquier_2013 BLAT_ECOLX_Deng_2012 \
#             BLAT_ECOLX_Firnberg_2014 BLAT_ECOLX_Stiffler_2015 \
#             PABP_YEAST_Melamed_2013; do
#       sbatch scripts/submit_embed.sh $D
#   done
#
# NOTE: BRCA1_HUMAN_Findlay_2018 (WT len 1863) exceeds the ESM-2 1022-residue
# limit — skip it for embedding. Use mutation/physicochemical only for BRCA1.
# -----------------------------------------------------------------------

set -eo pipefail

DATASET="${1:-BLAT_ECOLX_Jacquier_2013}"      # pass as positional arg
ESM_MODEL="facebook/esm2_t33_650M_UR50D"      # use 8M for prototyping, 650M for cluster
EMBED_BATCH_SIZE=64                            # increase for A100/H100

# --- Environment setup ---------------------------------------------------
module purge
module load anaconda3/2024.6
conda activate rag_al

# Always submit from the project root: cd /home/.../rag_pipeline && sbatch scripts/submit_embed.sh
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. Submit with sbatch from the project root." >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(pwd)"

# Absolute paths — immune to subprocess cwd changes
DATA_DIR="${PROJECT_ROOT}/data/curated"
EMBED_CACHE_DIR="${PROJECT_ROOT}/data/embeddings"

# Use cached model weights only — compute nodes have no outbound internet.
# Run scripts/download_models.sh from the login node first.
# HF_HOME defaults to ~/.cache/huggingface/ (home has ~27 GB free, sufficient
# for 650M model). Override by setting HF_HOME in ~/.bashrc if needed.
export HF_HUB_OFFLINE=1

echo "============================================"
echo "Job: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Dataset: $DATASET"
echo "Model: $ESM_MODEL"
echo "============================================"

# Precompute every cache the benchmark grid needs:
#   mean    -> cache_{model}.pkl        (also serves plm_retrieval, plm_concat)
#   delta   -> cache_{model}.pkl        (reuses mean cache + WT forward pass)
#   site    -> cache_{model}_site.pkl   (plm_site)
#   physico -> cache_{model}_physico.pkl (plm_physico)
# NOTE: for multi-site datasets (GFP, GB1) 'site' averages across mutated
# positions, which may dilute signal — kept for comparison; 'delta' is the
# multi-site baseline. See docs/sprint2_plan.md.
rag-embed \
    --dataset          "$DATASET" \
    --data_dir         "$DATA_DIR" \
    --embed_cache_dir  "$EMBED_CACHE_DIR" \
    --esm_model        "$ESM_MODEL" \
    --embed_batch_size "$EMBED_BATCH_SIZE" \
    --modes mean delta site physico

echo "Embedding job complete."
