from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import BenchmarkConfig


_DEFAULT_FMT = (
    "%(asctime)s | %(levelname)s | %(name)s | "
    "dataset=%(dataset)s repr=%(representation)s acq=%(acquisition)s seed=%(seed)s"
    "%(extra_kv)s | %(message)s"
)
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _ContextFilter(logging.Filter):
    """
    Injects benchmark context (dataset / repr / acq / seed) into every
    log record, plus optional arbitrary key/value pairs (e.g., round=3).
    """

    def __init__(self, base_ctx: Mapping[str, Any]) -> None:
        super().__init__()
        self.base_ctx = dict(base_ctx)

    def filter(self, record: logging.LogRecord) -> bool:
        record.dataset = self.base_ctx.get("dataset", "-")
        record.representation = self.base_ctx.get("representation", "-")
        record.acquisition = self.base_ctx.get("acquisition", "-")
        record.seed = self.base_ctx.get("seed", "-")

        extras = {
            k: v for k, v in self.base_ctx.items()
            if k not in ("dataset", "representation", "acquisition", "seed")
        }
        record.extra_kv = (
            " " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""
        )
        return True


def _make_file_handler(log_path: Path, level: int) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    return fh


def _make_stream_handler(level: int) -> logging.Handler:
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(_DEFAULT_FMT, datefmt=_DEFAULT_DATEFMT))
    return sh


def _get_or_create_logger(
    name: str,
    *,
    log_path: Path,
    ctx: Mapping[str, Any],
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    """
    Return a logger writing to log_path (and optionally stdout).
    Handlers are not duplicated on repeated calls with the same log_path.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicating handlers if logger is already configured for this file
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename) == log_path:
                    return logger
            except Exception:
                pass

    logger.handlers.clear()
    logger.addFilter(_ContextFilter(ctx))
    logger.addHandler(_make_file_handler(log_path, level))
    if also_stdout:
        logger.addHandler(_make_stream_handler(level))

    return logger


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

@dataclass(frozen=True)
class LogPaths:
    seed_log: Path


def get_log_paths(cfg: BenchmarkConfig) -> LogPaths:
    return LogPaths(seed_log=cfg.paths.seed_log)


def get_run_logger(
    cfg: BenchmarkConfig,
    *,
    level: int = logging.INFO,
    also_stdout: bool = True,
) -> logging.Logger:
    """
    Create / retrieve the logger for one benchmark run (dataset × repr × acq × seed).

    Log format::

        2025-05-23 14:32:10 | INFO | rag_al.BLAT_ECOLX.plm_mean.ucb.s0 |
        dataset=BLAT_ECOLX repr=plm_mean acq=ucb seed=0 | Starting AL loop ...

    Parameters
    ----------
    cfg : BenchmarkConfig
        Configuration for this run.
    level : int
        Logging level (default INFO).
    also_stdout : bool
        Whether to also write to stdout (default True).
    """
    p = cfg.paths
    lp = get_log_paths(cfg)
    ctx: dict[str, Any] = {
        "dataset": cfg.dataset,
        "representation": cfg.representation,
        "acquisition": cfg.acquisition,
        "seed": cfg.seed,
    }
    name = (
        f"rag_al.{cfg.dataset}.{cfg.representation}.{cfg.acquisition}.s{cfg.seed}"
    )
    return _get_or_create_logger(
        name,
        log_path=lp.seed_log,
        ctx=ctx,
        level=level,
        also_stdout=also_stdout,
    )


def with_context(logger: logging.Logger, **kwargs: Any) -> logging.LoggerAdapter:
    """
    Wrap a logger with extra key/value context (e.g., round=3).
    The extra keys appear in the extra_kv portion of the log line.
    """
    return logging.LoggerAdapter(logger, extra=kwargs)
