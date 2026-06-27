#!/bin/bash
# Download HuggingFace protein LM weights to scratch.
# Run ONCE from the login node (has internet access) before submitting jobs.
#
# Usage:
#   bash scripts/download_models.sh
#
# Princeton Della setup:
#   1. Add to ~/.bashrc:
#        export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface/
#   2. Run this script from the login node to populate the scratch cache.
#   3. Submit jobs — compute nodes load from scratch, no internet needed.

set -eo pipefail

# Activate the conda environment
module load anaconda3/2024.6
conda activate rag_al

# Storage note: ESM-2 650M ~ 2.5 GB. Home has ~27 GB free which is sufficient
# for Sprint 1. If you later add ProtT5 (~3 GB) + Ankh (~0.5 GB) for Sprint 2,
# reassess. To redirect to scratch instead:
#   export HF_HOME=/scratch/gpfs/$USER/.cache/huggingface/
# or add that line to ~/.bashrc before running this script.

echo "HF cache will land in: $(python -c 'from transformers.utils.hub import hf_cache_home; print(hf_cache_home)')"

python - << 'EOF'
from transformers import AutoTokenizer, AutoModel

# Sprint 1: only 650M needed on cluster.
# Add esm2_t30_150M_UR50D and esm2_t6_8M_UR50D when running the size sweep.
MODELS = [
    "facebook/esm2_t33_650M_UR50D",   # ~2.5 GB
]

for model_id in MODELS:
    print(f"\n==> Downloading {model_id} ...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    print(f"    Done.")

print("\nDownload complete. Jobs can now run with HF_HUB_OFFLINE=1.")
EOF
