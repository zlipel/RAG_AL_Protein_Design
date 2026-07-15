from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _tag(
    representation: str,
    acquisition: str,
    ucb_beta: float,
    surrogate: str = "rf",
) -> str:
    """
    Generate a reproducible run tag from strategy choices.

    Parameters
    ----------
    representation : str
        Encoder name (e.g., 'plm_mean').
    acquisition : str
        Acquisition function name (e.g., 'ucb').
    ucb_beta : float
        UCB exploration weight (included in tag for UCB-family acquisitions).
    surrogate : str
        Surrogate model name (e.g., 'rf', 'gp'). Non-default surrogates are
        appended to the tag so their results never collide with the default
        RF run for the same (representation, acquisition, seed). 'rf' keeps the
        historical suffix-free tag for backward compatibility.

    Returns
    -------
    str
        Tag string used for directory and file naming.
    """
    tag = f"{representation}_{acquisition}"
    if acquisition in ("ucb", "diversity_ucb", "retrieval_ucb"):
        tag += f"_b{ucb_beta}"
    if surrogate != "rf":
        tag += f"_{surrogate}"
    return tag


@dataclass(frozen=True)
class BenchmarkPaths:
    """
    Frozen dataclass computing all file and directory paths for one benchmark run.
    All paths are derived from the base directories and the run identity
    (dataset, representation, acquisition, seed).
    """

    results_dir: Path
    log_dir: Path
    embed_cache_dir: Path

    dataset: str
    representation: str
    acquisition: str
    seed: int
    ucb_beta: float = 1.0
    surrogate: str = "rf"

    # ------------------------------------------------------------------
    # Tag
    # ------------------------------------------------------------------

    @property
    def tag(self) -> str:
        return _tag(self.representation, self.acquisition, self.ucb_beta, self.surrogate)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def dataset_results_dir(self) -> Path:
        return self.results_dir / self.dataset

    @property
    def run_results_dir(self) -> Path:
        return self.dataset_results_dir / self.tag

    @property
    def seed_results_csv(self) -> Path:
        return self.run_results_dir / f"seed_{self.seed}.csv"

    @property
    def seed_selections_csv(self) -> Path:
        """Per-round selection log: which variants were acquired each round."""
        return self.run_results_dir / f"seed_{self.seed}_selections.csv"

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    @property
    def dataset_log_dir(self) -> Path:
        return self.log_dir / self.dataset

    @property
    def run_log_dir(self) -> Path:
        return self.dataset_log_dir / self.tag

    @property
    def seed_log(self) -> Path:
        return self.run_log_dir / f"seed_{self.seed}.log"

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    @property
    def dataset_embed_dir(self) -> Path:
        return self.embed_cache_dir / self.dataset

    def embed_cache(self, model_name: str) -> Path:
        """Path to cached embedding array (.npy) for a given ESM model."""
        safe = model_name.replace("/", "__")
        return self.dataset_embed_dir / f"embeddings_{safe}.npy"

    def embed_ids_cache(self, model_name: str) -> Path:
        """Path to variant_id order array matching the embedding cache."""
        safe = model_name.replace("/", "__")
        return self.dataset_embed_dir / f"variant_ids_{safe}.npy"


def ensure_dirs(p: BenchmarkPaths) -> None:
    """
    Create all output directories required for a benchmark run.

    Parameters
    ----------
    p : BenchmarkPaths
        Paths for this run.
    """
    p.run_results_dir.mkdir(parents=True, exist_ok=True)
    p.run_log_dir.mkdir(parents=True, exist_ok=True)
    p.dataset_embed_dir.mkdir(parents=True, exist_ok=True)
