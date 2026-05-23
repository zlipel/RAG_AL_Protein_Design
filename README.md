# Retrieval-Augmented Active Learning for Protein Fitness Landscapes

Retrospective active learning benchmark comparing sequence representations and
acquisition functions on deep mutational scanning (DMS) datasets.

## Scientific Question

In sparse protein fitness landscapes, do foundation-model embeddings and
retrieval-augmented acquisition improve active learning compared with standard
sequence descriptors and standard Bayesian optimization strategies?

---

## Project Structure

```
RAG_AL_Protein_Design/
├── data/                    # curated dataset CSVs + embedding caches (gitignored)
├── results/                 # benchmark outputs (gitignored)
├── logs/                    # run logs (gitignored)
├── figures/                 # learning-curve plots
├── scripts/
│   ├── submit_embed.sh      # SLURM GPU job: pre-compute ESM-2 embeddings
│   ├── submit_benchmark.sh  # SLURM array job: full benchmark sweep
│   └── plot_results.py      # aggregate results → learning-curve figures
└── src/rag_al/
    ├── core/                # config, paths, logging
    ├── data/                # schema, loader, ALDataset
    ├── representations/     # mutation, physicochemical, PLM, retrieval
    ├── surrogates/          # random forest
    ├── acquisition/         # random, greedy, UCB, diversity-UCB, retrieval-UCB
    ├── loop/                # AL runner + metrics
    └── cli/                 # rag-embed, rag-benchmark entry points
```

---

## Data Schema

Place curated CSV files in `data/` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `variant_id` | str | Unique identifier |
| `mutant` | str | Mutation string, e.g. `A23V` or `A23V:G45L` |
| `mutated_sequence` | str | Full amino acid sequence of the variant |
| `wt_sequence` | str | Wild-type sequence (same for all rows in a dataset) |
| `fitness` | float | Measured fitness / variant-effect score |

---

## Installation

```bash
# From project root
pip install -e ".[dev]"
```

---

## Workflow

### Step 1 — Pre-compute embeddings (GPU, run once)

```bash
# On cluster (Della/Tiger) or locally with MPS:
rag-embed \
    --dataset       BLAT_ECOLX \
    --esm_model     facebook/esm2_t33_650M_UR50D \
    --embed_batch_size 64

# Or submit as SLURM job:
sbatch scripts/submit_embed.sh
```

### Step 2 — Run benchmark

```bash
# Single cell (local dev / prototyping):
rag-benchmark \
    --dataset        BLAT_ECOLX \
    --representation plm_mean \
    --acquisition    ucb \
    --seed           0

# Full sweep on cluster (90 cells = 5 repr × 6 acq × 3 seeds):
sbatch scripts/submit_benchmark.sh
```

### Step 3 — Plot results

```bash
python scripts/plot_results.py --dataset BLAT_ECOLX --output_dir figures/
```

---

## Benchmark Grid

| Representations | Acquisition Functions |
|---|---|
| Mutation descriptors | Random |
| Physicochemical descriptors | Greedy |
| PLM mean-pool (ESM-2) | UCB |
| PLM delta (mutant − WT) | Diversity UCB |
| PLM + retrieval features | Retrieval UCB |

---

## Leakage Rules

The pipeline enforces strict leakage prevention:

1. Surrogate trains only on labeled fitness scores.
2. Acquisition functions cannot access hidden pool fitness.
3. Retrieval features retrieve only from the labeled set.
4. Embedding computation uses sequences only — no fitness labels.
5. Label normalization statistics are fit on the labeled set only.
6. `ALDataset._ALDataset__fitness` is name-mangled; `LeakageError` is raised
   on unauthorized access paths.

---

## Key Configuration

All parameters are exposed as CLI flags (see `rag-benchmark --help`):

```
--n_init          50      Initial labeled set size
--n_rounds        5       AL rounds (increase for full runs)
--batch_size      20      Acquisitions per round
--ucb_beta        1.0     UCB exploration weight β
--retrieval_lambda 0.5    Retrieval score weight λ
--n_neighbors     5       kNN for retrieval
--esm_model       facebook/esm2_t6_8M_UR50D  (override on cluster)
```
