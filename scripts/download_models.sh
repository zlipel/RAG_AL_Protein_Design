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

# Mirror the HF_HOME that submit scripts will use at job time.
# Edit this if your scratch path differs.
export HF_HOME="/scratch/gpfs/${USER}/.cache/huggingface/"

echo "HF_HOME set to: $HF_HOME"
mkdir -p "$HF_HOME"

# Activate the conda environment
module load anaconda3/2024.6
conda activate rag_al

python - << 'EOF'
import os
from transformers import AutoTokenizer, AutoModel
from transformers.utils.hub import hf_cache_home

if "/scratch" not in hf_cache_home:
    raise EnvironmentError(
        f"HF cache is at '{hf_cache_home}' — not on scratch. "
        "Set HF_HOME=/scratch/gpfs/$USER/.cache/huggingface/ before running."
    )

MODELS = [
    "facebook/esm2_t6_8M_UR50D",      # 8M  — prototyping / local
    "facebook/esm2_t30_150M_UR50D",   # 150M — scale sweep
    "facebook/esm2_t33_650M_UR50D",   # 650M — cluster default
]

for model_id in MODELS:
    print(f"\n==> Downloading {model_id} ...")
    AutoTokenizer.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)
    print(f"    Cached to {os.environ['HF_HOME']}")

print("\nAll models downloaded. Jobs can now run with HF_HUB_OFFLINE=1.")
EOF
