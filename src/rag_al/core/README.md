# rag_al/core

Core infrastructure shared across the entire pipeline: configuration,
path management, and logging. These three modules are the foundation that
every other module imports from. They are intentionally kept simple and
dependency-free (no numpy, no torch).

---

## config.py

Defines `BenchmarkConfig`, a frozen dataclass that holds every parameter
for one benchmark run. "Frozen" means the object cannot be mutated after
construction, which prevents accidental parameter changes mid-run.

### Required fields (must be provided via CLI or constructor)
- `dataset`        — name of the curated CSV file in data_dir (e.g. "BLAT_ECOLX")
- `representation` — which encoder to use; must be one of:
                     mutation | physicochemical | plm_mean | plm_delta | plm_retrieval
- `acquisition`    — which acquisition function to use; must be one of:
                     random | greedy | ucb | diversity_ucb | retrieval_ucb

### Optional fields (all have defaults)
- `data_dir`             Path to the data directory (default: "data/")
- `results_dir`          Where benchmark CSVs are written (default: "results/")
- `log_dir`              Where log files are written (default: "logs/")
- `embed_cache_dir`      Where PLM embedding caches live (default: "data/embeddings/")
- `n_init`               Initial labeled set size (default: 50)
- `n_rounds`             Number of active learning rounds (default: 5)
- `batch_size`           Variants acquired per round (default: 20)
- `seed`                 Random seed for reproducibility (default: 0)
- `n_estimators`         Number of trees in the Random Forest (default: 100)
- `ucb_beta`             UCB exploration weight β (default: 1.0)
- `retrieval_lambda`     Retrieval score weight λ (default: 0.5)
- `n_neighbors`          k for kNN retrieval (default: 5)
- `esm_model`            HuggingFace model ID for ESM-2 (default: esm2_t6_8M_UR50D)
- `embed_batch_size`     Sequences per ESM forward pass (default: 32)

### Key methods
- `validate()`  — checks all field values for correctness; raises ValueError or
                  FileNotFoundError with descriptive messages before any work starts.
- `ensure()`    — calls validate(), then creates all output directories via
                  BenchmarkPaths.ensure_dirs(). Returns the BenchmarkPaths object.
- `from_cli()`  — class method that builds a BenchmarkConfig from command-line
                  arguments using argparse. Required fields become required CLI flags;
                  all optional fields get auto-generated --flag_name arguments with
                  their defaults. This mirrors the ALConfig.from_cli() pattern from
                  MODEL_COMPARISON_GIT exactly.
- `data_csv`    — property returning the full path to the dataset CSV file.
- `paths`       — property returning the BenchmarkPaths object for this run.

### Usage pattern
Every CLI entry point follows this pattern:
    cfg = BenchmarkConfig.from_cli()
    p   = cfg.ensure()   # validate + make dirs
    log = get_run_logger(cfg)
    ...

---

## paths.py

Defines `BenchmarkPaths`, a frozen dataclass that computes every file and
directory path used by a run as a Python property. No path is hardcoded
anywhere else in the codebase — everything goes through this object.

### Tag generation
Each run is identified by a short tag string built from the strategy choices:
    <representation>_<acquisition>[_b<ucb_beta>]
Examples:
    "mutation_random"
    "plm_mean_ucb_b1.0"
    "plm_retrieval_retrieval_ucb_b1.0"

The tag is used in directory and file names so that different strategy
combinations never overwrite each other's outputs.

### Path properties
Results:
    dataset_results_dir   results/<dataset>/
    run_results_dir       results/<dataset>/<tag>/
    seed_results_csv      results/<dataset>/<tag>/seed_<N>.csv

Logs:
    dataset_log_dir       logs/<dataset>/
    run_log_dir           logs/<dataset>/<tag>/
    seed_log              logs/<dataset>/<tag>/seed_<N>.log

Embeddings:
    dataset_embed_dir     data/embeddings/<dataset>/
    embed_cache(model)    data/embeddings/<dataset>/embeddings_<model>_<mode>.npy
    embed_ids_cache(model) data/embeddings/<dataset>/variant_ids_<model>_<mode>.npy

### ensure_dirs(p)
Module-level function that creates all output directories (results, logs,
embeddings) for a given BenchmarkPaths instance.

---

## logging.py

Sets up context-aware logging that injects the benchmark identity
(dataset, representation, acquisition, seed) into every log line.
Mirrors the _ContextFilter / get_master_logger pattern from MODEL_COMPARISON_GIT.

### Log format
    2025-05-23 14:32:10 | INFO | rag_al.BLAT_ECOLX.plm_mean.ucb.s0 |
    dataset=BLAT_ECOLX repr=plm_mean acq=ucb seed=0 | Starting AL loop ...

The context fields (dataset, repr, acq, seed) are injected by _ContextFilter
into every LogRecord, so they appear automatically on every line without
passing them explicitly to each log call.

### Key functions
- `get_run_logger(cfg, also_stdout=True)` — creates a logger for one run.
  Writes to logs/<dataset>/<tag>/seed_<N>.log and optionally to stdout.
  Handlers are not duplicated if the logger already exists (safe to call
  multiple times).
- `with_context(logger, **kwargs)` — wraps a logger in a LoggerAdapter
  that appends extra key=value pairs to the extra_kv section of the log
  line. Useful for adding round=3 or phase=encode to specific sections.

### Handler deduplication
If get_run_logger() is called multiple times with the same log path (e.g.,
on module reimport during testing), it detects the existing FileHandler
and returns the already-configured logger rather than adding duplicate
handlers.
