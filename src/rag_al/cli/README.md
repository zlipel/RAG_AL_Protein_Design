# rag_al/cli

Command-line entry points. These are the scripts you actually call on the
command line or submit to SLURM. They are thin wrappers that parse CLI
arguments, construct the appropriate objects from core/data/representations/
surrogates/acquisition/loop, run the experiment, and save results.

Entry points are registered in pyproject.toml under [project.scripts]:
    rag-embed     -> rag_al.cli.embed:main
    rag-benchmark -> rag_al.cli.benchmark:main

After pip install -e ., both commands are available system-wide.

---

## embed.py  (rag-embed)

Pre-computes ESM-2 embeddings for all variants in a dataset and caches
them to disk. This is a GPU job meant to be run once before the benchmark
sweep.

### Why run this separately?
Computing ESM-2 embeddings is expensive (minutes to hours depending on
dataset size and model size). By caching them before the benchmark, every
subsequent rag-benchmark call can load cached embeddings from disk instead
of re-running the model. This decouples GPU usage from the benchmark sweep,
which runs on CPU.

### What it does
1. Loads the curated dataset CSV.
2. Constructs an ESMEncoder for each requested mode (mean, delta).
3. Calls encoder.transform(df) which triggers embedding computation and
   automatically saves to disk (via ESMEncoder's cache logic).

### CLI flags
--dataset           Dataset name (CSV stem in data_dir)
--data_dir          Path to data directory (default: data/)
--embed_cache_dir   Where to write .npy cache files (default: data/embeddings/)
--esm_model         HuggingFace model ID (default: facebook/esm2_t6_8M_UR50D)
--embed_batch_size  Sequences per forward pass (default: 32; increase for A100)
--modes             Which modes to compute: mean, delta, or both (default: both)

### Typical usage
On laptop (small model for testing):
    rag-embed --dataset BLAT_ECOLX --esm_model facebook/esm2_t6_8M_UR50D

On cluster (large model for actual runs):
    rag-embed --dataset BLAT_ECOLX --esm_model facebook/esm2_t33_650M_UR50D \
              --embed_batch_size 64

Or submit as SLURM job:
    sbatch scripts/submit_embed.sh

---

## benchmark.py  (rag-benchmark)

Runs one complete active learning benchmark cell: one (dataset ×
representation × acquisition × seed) combination. Saves results to
results/<dataset>/<tag>/seed_<N>.csv.

This is designed to be called as a SLURM array task, with the array index
mapping to a specific (repr, acq, seed) combination via the submit script.
It can also be run interactively for single-cell testing.

### What it does
1. Parses CLI args into a BenchmarkConfig.
2. Calls cfg.ensure() to validate config and create output directories.
3. Creates a run logger (writes to logs/ and stdout).
4. Loads the dataset CSV and constructs ALDataset.
5. Builds the encoder specified by --representation.
6. Builds RFSurrogate with the config's n_estimators and seed.
7. Builds the acquisition function specified by --acquisition.
8. Calls run_al_loop() and collects the results DataFrame.
9. Prepends metadata columns (dataset, representation, acquisition, seed).
10. Writes results to results/<dataset>/<tag>/seed_<N>.csv.

### CLI flags
All BenchmarkConfig fields are exposed as CLI flags. Required flags:
    --dataset, --representation, --acquisition

Important optional flags:
    --seed           Random seed (default: 0)
    --n_rounds       AL rounds (default: 5)
    --batch_size     Variants per round (default: 20)
    --n_init         Initial labeled set size (default: 50)
    --ucb_beta       UCB β parameter (default: 1.0)
    --esm_model      ESM-2 model ID (default: esm2_t6_8M_UR50D)

Run rag-benchmark --help to see all available flags.

### Encoder construction (_build_encoder)
A private helper function maps the --representation flag to the
appropriate encoder object:
    mutation       -> MutationDescriptorEncoder()
    physicochemical -> PhysicochemicalEncoder()
    plm_mean       -> ESMEncoder(mode='mean', cache_dir=...)
    plm_delta      -> ESMEncoder(mode='delta', cache_dir=...)
    plm_retrieval  -> RetrievalAugmentedEncoder(ESMEncoder(mode='mean'), k=n_neighbors)

For PLM-based encoders, the cache_dir is set to the dataset's embedding
directory so that pre-computed embeddings are loaded automatically.

### Acquisition construction (_build_acquisition)
A private helper function maps the --acquisition flag to the appropriate
acquisition object:
    random         -> RandomAcquisition()
    greedy         -> GreedyAcquisition()
    ucb            -> UCBAcquisition(beta=ucb_beta)
    diversity_ucb  -> DiversityUCBAcquisition(beta=ucb_beta)
    retrieval_ucb  -> RetrievalUCBAcquisition(beta=ucb_beta, lam=retrieval_lambda,
                                              n_neighbors=n_neighbors)

### Typical usage
Single interactive run:
    rag-benchmark --dataset BLAT_ECOLX --representation plm_mean \
                  --acquisition ucb --seed 0

Full sweep via SLURM:
    sbatch scripts/submit_benchmark.sh
