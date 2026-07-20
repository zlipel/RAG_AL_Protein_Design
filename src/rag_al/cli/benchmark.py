"""
rag-benchmark — Run one active learning benchmark cell.

Each invocation runs a single (dataset × representation × acquisition × seed)
combination and writes results to results/<dataset>/<tag>/seed_<s>.csv.

Designed to be called as a SLURM array job — one task per cell.

Usage
-----
rag-benchmark --dataset BLAT_ECOLX --representation plm_mean --acquisition ucb --seed 0
"""

from __future__ import annotations

import sys


def main() -> None:
    # Lazy imports so --help is fast
    from ..core.config import BenchmarkConfig
    from ..core.logging import get_run_logger

    cfg = BenchmarkConfig.from_cli()
    p = cfg.ensure()   # validates + creates dirs

    log = get_run_logger(cfg, also_stdout=True)
    log.info("=" * 60)
    log.info("RAG-AL Benchmark Run")
    log.info("  dataset        : %s", cfg.dataset)
    log.info("  representation : %s", cfg.representation)
    log.info("  acquisition    : %s", cfg.acquisition)
    log.info("  seed           : %d", cfg.seed)
    log.info("  n_init         : %d", cfg.n_init)
    log.info("  n_rounds       : %d", cfg.n_rounds)
    log.info("  batch_size     : %d", cfg.batch_size)
    log.info("  results → %s", p.seed_results_csv)
    log.info("=" * 60)

    try:
        results, selections = _run(cfg, log)
    except Exception as e:
        log.exception("Run failed: %s", e)
        sys.exit(1)

    results.to_csv(p.seed_results_csv, index=False)
    log.info("Results written to %s", p.seed_results_csv)

    selections.to_csv(p.seed_selections_csv, index=False)
    log.info("Selections written to %s", p.seed_selections_csv)


def _run(cfg, log):
    from ..data.loader import load_dataset
    from ..data.al_dataset import ALDataset
    from ..loop.runner import run_al_loop

    # ---- Load dataset -------------------------------------------------------
    log.info("Loading dataset from %s", cfg.data_csv)
    df = load_dataset(cfg.data_csv)
    log.info("Dataset: %d variants", len(df))

    dataset = ALDataset(df, n_init=cfg.n_init, seed=cfg.seed)
    log.info("ALDataset initialized: %r", dataset)

    # ---- Build encoder ------------------------------------------------------
    encoder = _build_encoder(cfg, log)

    # ---- Build surrogate ----------------------------------------------------
    surrogate = _build_surrogate(cfg)

    # ---- Build acquisition --------------------------------------------------
    acquisition = _build_acquisition(cfg)

    # ---- Run loop -----------------------------------------------------------
    results, selections = run_al_loop(
        dataset=dataset,
        encoder=encoder,
        surrogate=surrogate,
        acquisition=acquisition,
        n_rounds=cfg.n_rounds,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        log=log,
    )

    # Attach metadata columns for easy downstream aggregation. `surrogate` is
    # recorded so RF and GP rows for the same (repr, acq, seed) stay
    # distinguishable after results are concatenated across runs.
    results.insert(0, "dataset", cfg.dataset)
    results.insert(1, "representation", cfg.representation)
    results.insert(2, "acquisition", cfg.acquisition)
    results.insert(3, "surrogate", cfg.surrogate)
    results.insert(4, "seed", cfg.seed)

    selections.insert(0, "dataset", cfg.dataset)
    selections.insert(1, "representation", cfg.representation)
    selections.insert(2, "acquisition", cfg.acquisition)
    selections.insert(3, "surrogate", cfg.surrogate)
    selections.insert(4, "seed", cfg.seed)

    return results, selections


def _build_encoder(cfg, log):
    """Construct the encoder specified in cfg.representation."""
    from ..representations.mutation import MutationDescriptorEncoder
    from ..representations.physicochemical import PhysicochemicalEncoder
    from ..representations.plm import ESMEncoder
    from ..representations.retrieval import RetrievalAugmentedEncoder

    p = cfg.paths
    repr_name = cfg.representation

    if repr_name == "mutation":
        log.info("Encoder: MutationDescriptorEncoder")
        return MutationDescriptorEncoder()

    if repr_name == "physicochemical":
        log.info("Encoder: PhysicochemicalEncoder")
        return PhysicochemicalEncoder()

    if repr_name == "plm_mean":
        log.info("Encoder: ESMEncoder(mode='mean', model=%s)", cfg.esm_model)
        return ESMEncoder(
            model_name=cfg.esm_model,
            mode="mean",
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )

    if repr_name == "plm_delta":
        log.info("Encoder: ESMEncoder(mode='delta', model=%s)", cfg.esm_model)
        return ESMEncoder(
            model_name=cfg.esm_model,
            mode="delta",
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )

    if repr_name == "plm_site":
        log.info("Encoder: ESMEncoder(mode='site', model=%s)", cfg.esm_model)
        return ESMEncoder(
            model_name=cfg.esm_model,
            mode="site",
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )

    if repr_name == "plm_physico":
        from ..representations.plm_physico import PLMPhysicoEncoder
        log.info("Encoder: PLMPhysicoEncoder(model=%s)", cfg.esm_model)
        return PLMPhysicoEncoder(
            model_name=cfg.esm_model,
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )

    if repr_name == "plm_concat":
        from ..representations.plm_physico import PLMSimpleConcatEncoder
        log.info("Encoder: PLMSimpleConcatEncoder(model=%s)", cfg.esm_model)
        return PLMSimpleConcatEncoder(
            model_name=cfg.esm_model,
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )

    if repr_name == "plm_retrieval":
        log.info(
            "Encoder: RetrievalAugmentedEncoder(k=%d, model=%s)",
            cfg.n_neighbors, cfg.esm_model,
        )
        esm = ESMEncoder(
            model_name=cfg.esm_model,
            mode="mean",
            embed_batch_size=cfg.embed_batch_size,
            cache_dir=p.dataset_embed_dir,
        )
        return RetrievalAugmentedEncoder(esm_encoder=esm, n_neighbors=cfg.n_neighbors)

    raise ValueError(f"Unknown representation: {repr_name!r}")


def _build_surrogate(cfg):
    """Construct the surrogate specified in cfg.surrogate."""
    if cfg.surrogate == "rf":
        from ..surrogates.random_forest import RFSurrogate
        return RFSurrogate(
            n_estimators=cfg.n_estimators,
            random_state=cfg.seed,
            n_jobs=cfg.rf_n_jobs,
        )
    if cfg.surrogate == "gp":
        from ..surrogates.gp import GPSurrogate
        return GPSurrogate(
            n_iter=cfg.gp_n_iter,
            lr=cfg.gp_lr,
            patience=cfg.gp_patience,
            predict_batch_size=cfg.gp_predict_batch_size,
        )
    raise ValueError(f"Unknown surrogate: {cfg.surrogate!r}")


def _build_acquisition(cfg):
    """Construct the acquisition function specified in cfg.acquisition."""
    from ..acquisition.random_acq import RandomAcquisition
    from ..acquisition.greedy import GreedyAcquisition
    from ..acquisition.ucb import UCBAcquisition
    from ..acquisition.diversity_ucb import DiversityUCBAcquisition
    from ..acquisition.retrieval_ucb import RetrievalUCBAcquisition

    acq_name = cfg.acquisition

    if acq_name == "random":
        return RandomAcquisition()
    if acq_name == "greedy":
        return GreedyAcquisition()
    if acq_name == "ucb":
        return UCBAcquisition(beta=cfg.ucb_beta)
    if acq_name == "diversity_ucb":
        return DiversityUCBAcquisition(beta=cfg.ucb_beta)
    if acq_name == "retrieval_ucb":
        return RetrievalUCBAcquisition(
            beta=cfg.ucb_beta,
            lam=cfg.retrieval_lambda,
            n_neighbors=cfg.n_neighbors,
        )

    raise ValueError(f"Unknown acquisition: {acq_name!r}")


if __name__ == "__main__":
    main()
