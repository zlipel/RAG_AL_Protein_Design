#!/bin/bash
#SBATCH --job-name=rag_gp
#SBATCH --output=logs/gp_%j_%x.out
#SBATCH --error=logs/gp_%j_%x.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
# No GPU requested — GP runs on CPU for the embedding sizes in this study.
# Increase to --gres=gpu:1 and set gp_device=cuda if training becomes a bottleneck.

# -----------------------------------------------------------------------
# Targeted GP surrogate benchmark: runs GPSurrogate on the datasets and
# representations where surrogate calibration matters most.
#
# Primary motivation: PABP anomaly (RF σ miscalibrated on flat landscape).
# Secondary: BLAT_Deng as a high-signal baseline for comparison.
#
# GP-only by design. The RF baseline for the same cells comes from the main
# sweep (submit_benchmark.sh), so we do NOT re-run RF here. GP results are
# written to surrogate-namespaced paths (…_<repr>_<acq>_bβ_gp/) by paths._tag,
# so they never overwrite the RF results from the main sweep.
#
# Grid: 3 reprs × 2 acqs × 3 seeds × 2 datasets = 36 GP cells.
# To extend: add datasets to DATASETS / reprs to REPRS below. PLM reps are
# auto-dropped for any dataset whose sequences exceed the ESM-2 limit.
#
# Prerequisites:
#   - ESM-2 embeddings pre-computed (submit_embed.sh ran for each dataset).
#   - Curated CSVs exist in data/curated/.
#   - Main sweep (submit_benchmark.sh) run for the same datasets → RF baseline.
#
# Usage (submit from project root):
#   sbatch scripts/submit_gp_benchmark.sh
# -----------------------------------------------------------------------

set -eo pipefail

DATASETS=(PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012)
REPRS=(mutation plm_mean plm_physico)
ACQS=(greedy ucb)
SURROGATES=(gp)          # GP-only; RF baseline comes from submit_benchmark.sh
N_ROUNDS=20
BATCH_SIZE=128
N_SEEDS=3
N_INIT=50
UCB_BETA=1.0
ESM_MODEL="facebook/esm2_t33_650M_UR50D"
ESM_MAX_RESIDUES=1022    # drop PLM reps for datasets exceeding this
WORKERS=${SLURM_CPUS_PER_TASK:-16}

# GP hyperparameters — defaults match GPSurrogate.__init__
GP_N_ITER=200
GP_LR=0.1
GP_PATIENCE=3

# --- Environment setup ---------------------------------------------------
module purge
module load anaconda3/2024.6
conda activate rag_al

if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not set. Submit with sbatch from the project root." >&2
    exit 1
fi
cd "$SLURM_SUBMIT_DIR"
PROJECT_ROOT="$(pwd)"

DATA_DIR="${PROJECT_ROOT}/data/curated"
EMBED_CACHE_DIR="${PROJECT_ROOT}/data/embeddings"
RESULTS_DIR="${PROJECT_ROOT}/results"

export HF_HUB_OFFLINE=1

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-local}   Node: ${SLURMD_NODENAME:-local}"
echo "Datasets: ${DATASETS[*]}"
echo "Reprs:    ${REPRS[*]}"
echo "Acqs:     ${ACQS[*]}"
echo "Surrogates: ${SURROGATES[*]}"
echo "n_rounds=$N_ROUNDS  batch_size=$BATCH_SIZE  n_seeds=$N_SEEDS  workers=$WORKERS"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

CMDS=()
for dataset in "${DATASETS[@]}"; do
    # Drop PLM reps for datasets whose sequences exceed the ESM-2 limit.
    MAX_LEN=$(python -c "import pandas as pd; print(int(pd.read_csv('${DATA_DIR}/${dataset}.csv', usecols=['mutated_sequence'])['mutated_sequence'].str.len().max()))")
    dataset_reprs=()
    for repr in "${REPRS[@]}"; do
        if [[ "$repr" == plm_* ]] && (( MAX_LEN > ESM_MAX_RESIDUES )); then
            echo "Skipping $repr for $dataset (max len $MAX_LEN > $ESM_MAX_RESIDUES)."
            continue
        fi
        dataset_reprs+=("$repr")
    done
    for surrogate in "${SURROGATES[@]}"; do
        for repr in "${dataset_reprs[@]}"; do
            for acq in "${ACQS[@]}"; do
                for seed in $(seq 0 $((N_SEEDS - 1))); do
                    CMDS+=("rag-benchmark \
--dataset $dataset \
--representation $repr \
--acquisition $acq \
--surrogate $surrogate \
--seed $seed \
--n_rounds $N_ROUNDS \
--batch_size $BATCH_SIZE \
--n_init $N_INIT \
--ucb_beta $UCB_BETA \
--esm_model $ESM_MODEL \
--gp_n_iter $GP_N_ITER \
--gp_lr $GP_LR \
--gp_patience $GP_PATIENCE \
--rf_n_jobs 1 \
--data_dir ${DATA_DIR} \
--embed_cache_dir ${EMBED_CACHE_DIR} \
--results_dir ${RESULTS_DIR}")
                done
            done
        done
    done
done

N_CELLS=${#CMDS[@]}
echo "Submitting $N_CELLS cells with $WORKERS workers"

if command -v parallel &>/dev/null; then
    printf '%s\n' "${CMDS[@]}" | parallel --jobs "$WORKERS" --halt soon,fail=1
else
    echo "GNU parallel not found — falling back to sequential execution"
    for cmd in "${CMDS[@]}"; do
        eval "$cmd"
    done
fi

echo "GP benchmark complete."
