"""
Result-path namespacing tests.

GP and RF results for the same (representation, acquisition, seed) must land in
distinct files, otherwise one surrogate silently overwrites the other. RF keeps
the historical suffix-free tag (backward compatibility with Sprint 1 results);
non-default surrogates get a `_{surrogate}` suffix.
"""

from __future__ import annotations

from pathlib import Path

from rag_al.core.config import BenchmarkConfig
from rag_al.core.paths import BenchmarkPaths, _tag


# ---------------------------------------------------------------------------
# _tag
# ---------------------------------------------------------------------------

def test_rf_tag_is_backward_compatible():
    """Default surrogate (rf) must keep the historical suffix-free tag."""
    assert _tag("plm_mean", "ucb", 1.0) == "plm_mean_ucb_b1.0"
    assert _tag("plm_mean", "ucb", 1.0, "rf") == "plm_mean_ucb_b1.0"
    assert _tag("mutation", "greedy", 1.0) == "mutation_greedy"  # no beta for greedy


def test_gp_tag_gets_suffix():
    assert _tag("plm_mean", "ucb", 1.0, "gp") == "plm_mean_ucb_b1.0_gp"
    # greedy has no beta term, but still gets the surrogate suffix
    assert _tag("mutation", "greedy", 1.0, "gp") == "mutation_greedy_gp"


def test_rf_and_gp_tags_differ():
    rf = _tag("plm_physico", "ucb", 1.0, "rf")
    gp = _tag("plm_physico", "ucb", 1.0, "gp")
    assert rf != gp


# ---------------------------------------------------------------------------
# BenchmarkPaths
# ---------------------------------------------------------------------------

def _paths(surrogate: str) -> BenchmarkPaths:
    return BenchmarkPaths(
        results_dir=Path("results"),
        log_dir=Path("logs"),
        embed_cache_dir=Path("data/embeddings"),
        dataset="PABP_YEAST_Melamed_2013",
        representation="plm_mean",
        acquisition="ucb",
        seed=0,
        ucb_beta=1.0,
        surrogate=surrogate,
    )


def test_seed_results_csv_differs_by_surrogate():
    rf = _paths("rf").seed_results_csv
    gp = _paths("gp").seed_results_csv
    assert rf != gp
    assert rf.name == gp.name == "seed_0.csv"      # same filename...
    assert rf.parent.name == "plm_mean_ucb_b1.0"   # ...different tag dir
    assert gp.parent.name == "plm_mean_ucb_b1.0_gp"


def test_selections_csv_also_namespaced():
    rf = _paths("rf").seed_selections_csv
    gp = _paths("gp").seed_selections_csv
    assert rf != gp
    assert rf.parent != gp.parent


def test_paths_default_surrogate_is_rf():
    """Omitting surrogate must reproduce the legacy RF path."""
    p = BenchmarkPaths(
        results_dir=Path("results"),
        log_dir=Path("logs"),
        embed_cache_dir=Path("data/embeddings"),
        dataset="d",
        representation="plm_mean",
        acquisition="ucb",
        seed=0,
    )
    assert p.tag == "plm_mean_ucb_b1.0"


# ---------------------------------------------------------------------------
# BenchmarkConfig threads surrogate through to paths
# ---------------------------------------------------------------------------

def test_config_surrogate_reaches_path():
    rf_cfg = BenchmarkConfig(
        dataset="d", representation="plm_mean", acquisition="ucb", surrogate="rf",
    )
    gp_cfg = BenchmarkConfig(
        dataset="d", representation="plm_mean", acquisition="ucb", surrogate="gp",
    )
    assert rf_cfg.paths.seed_results_csv != gp_cfg.paths.seed_results_csv
    assert gp_cfg.paths.seed_results_csv.parent.name.endswith("_gp")
