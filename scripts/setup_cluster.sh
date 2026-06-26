#!/bin/bash
# setup_cluster.sh — First-time cluster environment setup for RAG-AL.
#
# Run this once from the project root after cloning the repo:
#   bash scripts/setup_cluster.sh
#
# Assumptions:
#   - Slurm cluster with anaconda3/2024.6 module
#   - CUDA 12.1 (change CUDA_TAG below if your cluster differs;
#     run `nvcc --version` or `module avail cuda` to check)
#   - Project cloned to /home/zl4808/PROJECTS/rag_pipeline/
#   - Scratch space at /scratch/gpfs/WEBB/zl4808/PROJECTS/rag_pipeline/
#
# After running this script, the workflow is:
#   1. Edit code locally, push to GitHub
#   2. `git pull` on cluster — editable install picks up changes immediately,
#      no reinstall needed
#   3. Submit jobs with scripts/submit_embed.sh and scripts/submit_benchmark.sh
#   4. rsync results back to local machine

set -euo pipefail

# Always run from the project root, regardless of where this script is called from.
# BASH_SOURCE[0] is the script file itself; its parent is scripts/, its grandparent is root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "==> Project root: $PROJECT_ROOT"

CUDA_TAG="cu121"   # adjust: cu118, cu121, cu124, etc.

HOME_PROJECT="/home/zl4808/PROJECTS/rag_pipeline"
SCRATCH_PROJECT="/scratch/gpfs/WEBB/zl4808/PROJECTS/rag_pipeline"

# ---- 1. Load module and create conda env --------------------------------
echo "==> Loading anaconda module"
module load anaconda3/2024.6

echo "==> Creating conda environment from environment.yml"
conda env create -f environment.yml          # relative to PROJECT_ROOT, now safe
# If env already exists: conda env update -f environment.yml --prune

# ---- 2. Activate and install PyTorch + torchvision (CUDA) ---------------
echo "==> Installing PyTorch 2.x + torchvision with CUDA ($CUDA_TAG)"
conda run -n rag_al pip install torch torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---- 3. Install BoTorch + GPyTorch (depend on torch already being present) --
echo "==> Installing BoTorch and GPyTorch"
conda run -n rag_al pip install botorch gpytorch

# ---- 4. Install this package in editable mode ---------------------------
echo "==> Installing rag-al in editable mode"
conda run -n rag_al pip install -e ".[dev]"

# ---- 5. Create scratch directory layout ---------------------------------
echo "==> Setting up scratch directories"
mkdir -p "${SCRATCH_PROJECT}/data/embeddings"
mkdir -p "${SCRATCH_PROJECT}/results"
mkdir -p "${SCRATCH_PROJECT}/logs"

# ---- 6. Symlink scratch dirs into the repo so paths resolve correctly ---
# The package defaults (data/embeddings, results, logs) point into the repo
# root. Symlinking redirects large outputs to scratch without changing code.
echo "==> Symlinking scratch dirs into repo"
for DIR in data/embeddings results logs; do
    REPO_PATH="${HOME_PROJECT}/${DIR}"
    SCRATCH_PATH="${SCRATCH_PROJECT}/${DIR}"
    if [ -L "$REPO_PATH" ]; then
        echo "    $DIR symlink already exists — skipping"
    elif [ -d "$REPO_PATH" ]; then
        echo "    WARNING: $DIR is a real directory in repo — not symlinking"
        echo "    Move its contents to scratch manually if needed"
    else
        ln -s "$SCRATCH_PATH" "$REPO_PATH"
        echo "    Linked $REPO_PATH -> $SCRATCH_PATH"
    fi
done

# ---- 7. Verify install --------------------------------------------------
echo ""
echo "==> Verifying installation"
conda run -n rag_al python -c "
import torch, transformers, sklearn, botorch, gpytorch
print(f'  torch      {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')
print(f'  transformers {transformers.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  botorch    {botorch.__version__}')
print(f'  gpytorch   {gpytorch.__version__}')
"
conda run -n rag_al rag-benchmark --help > /dev/null && echo "  rag-benchmark CLI: OK"
conda run -n rag_al rag-embed --help    > /dev/null && echo "  rag-embed CLI:     OK"

echo ""
echo "==> Setup complete. Activate with: conda activate rag_al"
echo ""
echo "==> Cluster workflow:"
echo "    Local:   git push origin main"
echo "    Cluster: git pull origin main   (editable install auto-picks up changes)"
echo "    Embed:   for D in <datasets>; do sbatch scripts/submit_embed.sh \$D; done"
echo "    Bench:   for D in <datasets>; do sbatch scripts/submit_benchmark.sh \$D; done"
echo "    Rsync:   rsync -avz ${SCRATCH_PROJECT}/results/ ./results/"
echo "             rsync -avz ${SCRATCH_PROJECT}/data/embeddings/ ./data/embeddings/"
