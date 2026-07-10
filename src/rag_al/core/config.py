from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, MISSING
from pathlib import Path
from typing import get_type_hints

from .paths import BenchmarkPaths, ensure_dirs


REPRESENTATIONS: tuple[str, ...] = (
    "mutation",
    "physicochemical",
    "plm_mean",
    "plm_delta",
    "plm_site",
    "plm_physico",
    "plm_concat",
    "plm_retrieval",
)

ACQUISITIONS: tuple[str, ...] = (
    "random",
    "greedy",
    "ucb",
    "diversity_ucb",
    "retrieval_ucb",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Frozen configuration for one benchmark run.

    Required arguments (no defaults) must be supplied via CLI or constructor.
    All other arguments have sensible defaults and can be overridden.

    Parameters
    ----------
    dataset : str
        Dataset name — must match the CSV stem in data_dir
        (e.g., 'BLAT_ECOLX' for data/BLAT_ECOLX.csv).
    representation : str
        Encoder to use. One of REPRESENTATIONS.
    acquisition : str
        Acquisition function to use. One of ACQUISITIONS.
    """

    # ---- Required -------------------------------------------------------
    dataset: str
    representation: str
    acquisition: str

    # ---- Paths ----------------------------------------------------------
    data_dir: Path = Path("data/curated")
    results_dir: Path = Path("results")
    log_dir: Path = Path("logs")
    embed_cache_dir: Path = Path("data/embeddings")

    # ---- AL settings ----------------------------------------------------
    n_init: int = 50        # initial labeled set size
    n_rounds: int = 5       # number of AL rounds (start small; scale up via CLI)
    batch_size: int = 20    # variants acquired per round
    seed: int = 0

    # ---- Surrogate ------------------------------------------------------
    n_estimators: int = 100  # RF trees
    rf_n_jobs: int = 1       # cores per RF fit; set >1 only when running single-cell

    # ---- Acquisition hyperparameters ------------------------------------
    ucb_beta: float = 1.0           # UCB exploration weight β
    retrieval_lambda: float = 0.5   # retrieval score weight λ
    n_neighbors: int = 5            # kNN for retrieval features

    # ---- PLM ------------------------------------------------------------
    esm_model: str = "facebook/esm2_t6_8M_UR50D"  # override on cluster
    embed_batch_size: int = 32

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def data_csv(self) -> Path:
        return self.data_dir / f"{self.dataset}.csv"

    @property
    def paths(self) -> BenchmarkPaths:
        return BenchmarkPaths(
            results_dir=self.results_dir,
            log_dir=self.log_dir,
            embed_cache_dir=self.embed_cache_dir,
            dataset=self.dataset,
            representation=self.representation,
            acquisition=self.acquisition,
            seed=self.seed,
            ucb_beta=self.ucb_beta,
        )

    # ------------------------------------------------------------------
    # Validation + ensure
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError/FileNotFoundError on invalid configuration."""
        if self.representation not in REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {REPRESENTATIONS}, got {self.representation!r}"
            )
        if self.acquisition not in ACQUISITIONS:
            raise ValueError(
                f"acquisition must be one of {ACQUISITIONS}, got {self.acquisition!r}"
            )
        if self.n_init <= 0:
            raise ValueError(f"n_init must be > 0, got {self.n_init}")
        if self.n_rounds <= 0:
            raise ValueError(f"n_rounds must be > 0, got {self.n_rounds}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if not self.data_csv.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.data_csv}")

    def ensure(self) -> BenchmarkPaths:
        """Validate config and create all required output directories."""
        self.validate()
        p = self.paths
        ensure_dirs(p)
        return p

    # ------------------------------------------------------------------
    # CLI constructor (mirrors ALConfig.from_cli pattern)
    # ------------------------------------------------------------------

    @classmethod
    def from_cli(cls) -> BenchmarkConfig:
        """Parse CLI arguments and return a validated BenchmarkConfig."""
        parser = argparse.ArgumentParser(
            description="RAG-AL: Retrieval-Augmented Active Learning Benchmark"
        )

        # Required positional-style arguments
        parser.add_argument(
            "--dataset", required=True, type=str,
            help="Dataset name (CSV stem in --data_dir)",
        )
        parser.add_argument(
            "--representation", required=True, type=str,
            choices=list(REPRESENTATIONS),
        )
        parser.add_argument(
            "--acquisition", required=True, type=str,
            choices=list(ACQUISITIONS),
        )

        _CLI_EXCLUDE = {"dataset", "representation", "acquisition"}
        type_hints = get_type_hints(cls)

        # Auto-generate flags for all remaining dataclass fields
        for f in fields(cls):
            if f.name in _CLI_EXCLUDE:
                continue
            default = f.default
            if default is MISSING:
                continue

            arg = f"--{f.name}"
            t = type_hints.get(f.name, f.type)

            if t is bool:
                action = "store_true" if default is False else "store_false"
                parser.add_argument(arg, action=action, help=f"(default: {default})")
                continue

            if t is Path:
                parser.add_argument(arg, type=Path, default=default)
                continue

            if t in (int, float, str):
                parser.add_argument(arg, type=t, default=default)
                continue

            # Fallback: pass as string
            parser.add_argument(arg, type=str, default=str(default))

        args = parser.parse_args()
        cfg = cls(**vars(args))
        cfg.validate()
        return cfg
