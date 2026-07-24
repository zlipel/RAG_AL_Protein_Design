#!/bin/bash
#SBATCH --job-name=rag_gp
#SBATCH --output=logs/gp_%j_%x.out
#SBATCH --error=logs/gp_%j_%x.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
# GP runs on CPU. Each cell is its own srun --exclusive step with a per-cell
# CPU + memory cap (CPUS_PER_CELL / MEM_PER_CELL below).

# -----------------------------------------------------------------------
# GP surrogate benchmark: runs GPSurrogate on the flagship multi-site
# landscapes (GB1, GFP) where PLM+GP is expected to win. Defaults are
# env-overridable (GP_DATASETS / GP_REPRS) to also target PABP/BLAT_Deng.
#
# Kernel is representation-conditional (per-rep, not a global flag):
#   - non-PLM reps (mutation, physicochemical) → ARD Matérn (--gp_ard),
#     which rescues the isotropic collapse on low-dim hand-crafted features.
#   - PLM reps (plm_*) → isotropic Matérn. ARD on 1280-d PLM is ill-conditioned
#     → CG non-convergence → wall-timeout (see docs/agent_log.md, job 11517601).
# So mutation cells write to …_gp_ard/ and PLM cells to …_gp/; both are
# surrogate-namespaced by paths._tag and never overwrite the RF main sweep.
#
# GP-only by design. The RF baseline for the same cells comes from the main
# sweep (submit_benchmark.sh), so we do NOT re-run RF here.
#
# Grid: 3 reprs × 2 acqs × 3 seeds × 2 datasets = 36 GP cells. PLM reps are
# auto-dropped for any dataset whose sequences exceed the ESM-2 limit.
#
# Prerequisites:
#   - ESM-2 embeddings pre-computed (submit_embed.sh ran for each dataset).
#   - Curated CSVs exist in data/curated/.
#   - Main sweep (submit_benchmark.sh) run for the same datasets → RF baseline.
#
# Usage (submit from project root):
#   sbatch scripts/submit_gp_benchmark.sh
#   GP_DATASETS="PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012" sbatch scripts/submit_gp_benchmark.sh
# -----------------------------------------------------------------------

set -eo pipefail

# Flagship multi-site landscapes by default; override with GP_DATASETS / GP_REPRS
# (space-separated). PABP/BLAT_Deng already have isotropic _gp results.
DATASETS=(${GP_DATASETS:-SPG1_STRSG_Wu_2016 GFP_AEQVI_Sarkisyan_2016})
REPRS=(${GP_REPRS:-mutation plm_mean plm_physico})
ACQS=(greedy ucb)
SURROGATES=(gp)          # GP-only; RF baseline comes from submit_benchmark.sh
N_ROUNDS=20
BATCH_SIZE=128
N_SEEDS=3
N_INIT=50
UCB_BETA=1.0
ESM_MODEL="facebook/esm2_t33_650M_UR50D"
ESM_MAX_RESIDUES=1022    # drop PLM reps for datasets exceeding this

# Per-cell srun step resources. 1 core each (single-threaded); 8 GB covers the
# GB1 149K-pool cell (ESM cache + pool matrix ≈ 3-4 GB). Raise for larger pools.
MEM_PER_CELL="${MEM_PER_CELL:-8G}"
CPUS_PER_CELL="${CPUS_PER_CELL:-1}"
MAX_CONCURRENT="${MAX_CONCURRENT:-${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || echo 4)}}"

# GP hyperparameters — defaults match GPSurrogate.__init__
GP_N_ITER=200
GP_LR=0.1
GP_PATIENCE=3
GP_BATCH_SIZE="${GP_BATCH_SIZE:-4096}"
# ARD is applied per-rep in the loop below (non-PLM → --gp_ard, PLM → isotropic),
# not as a global toggle. See the header note for the rationale.

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

# One BLAS/OMP thread per cell (each cell gets 1 core). torch reads these at import.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "Job: ${SLURM_JOB_ID:-local}   Node: ${SLURMD_NODENAME:-local}"
echo "Datasets: ${DATASETS[*]}"
echo "Reprs:    ${REPRS[*]}"
echo "Acqs:     ${ACQS[*]}"
echo "Surrogates: ${SURROGATES[*]}"
echo "n_rounds=$N_ROUNDS  batch_size=$BATCH_SIZE  n_seeds=$N_SEEDS"
echo "Per cell: cpus=$CPUS_PER_CELL mem=$MEM_PER_CELL"
echo "Concurrency: up to $MAX_CONCURRENT cells running at once (global throttle)"
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
            # Kernel policy: ARD for low-dim hand-crafted reps, isotropic for PLM
            # (ARD on 1280-d PLM is ill-conditioned → CG divergence → timeout).
            gp_ard_flag=""; [[ "$repr" != plm_* ]] && gp_ard_flag="--gp_ard"
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
--gp_predict_batch_size $GP_BATCH_SIZE \
$gp_ard_flag \
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
echo "Dispatching $N_CELLS cells (up to $MAX_CONCURRENT at once)"

# Run one cell. Under SLURM it becomes its own srun step with an isolated core
# and memory cgroup; off SLURM it is a plain subshell.
run_cell() {
    if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun &>/dev/null; then
        srun --exclusive --ntasks=1 --cpus-per-task="$CPUS_PER_CELL" \
             --mem="$MEM_PER_CELL" bash -c "$1"
    else
        bash -c "$1"
    fi
}

RC_DIR="$(mktemp -d)"
trap 'rm -rf "$RC_DIR"' EXIT

idx=0
for cmd in "${CMDS[@]}"; do
    while (( $(jobs -rp | wc -l) >= MAX_CONCURRENT )); do sleep 2; done
    ( set +e; run_cell "$cmd"; echo $? > "${RC_DIR}/${idx}" ) &
    idx=$((idx + 1))
done
wait

FAILED=0
for f in "${RC_DIR}"/*; do
    rc=$(cat "$f")
    if [[ "$rc" != "0" ]]; then
        FAILED=$((FAILED + 1))
        echo "  FAILED (rc=${rc}): ${CMDS[${f##*/}]}" >&2
    fi
done
echo "GP benchmark: $((N_CELLS - FAILED))/${N_CELLS} cells succeeded."
