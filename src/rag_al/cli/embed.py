"""
rag-embed — Pre-compute and cache ESM-2 embeddings for a dataset.

Run this once (on GPU) before running the benchmark sweep.
Cached embeddings are saved to data/embeddings/<dataset>/ and
reused automatically by ESMEncoder.

Usage
-----
rag-embed --dataset BLAT_ECOLX --esm_model facebook/esm2_t33_650M_UR50D
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute ESM-2 embeddings for a curated dataset CSV."
    )
    parser.add_argument("--dataset", required=True, type=str,
                        help="Dataset name (CSV stem in --data_dir)")
    parser.add_argument("--data_dir", type=Path, default=Path("data"),
                        help="Directory containing curated CSVs (default: data/)")
    parser.add_argument("--embed_cache_dir", type=Path,
                        default=Path("data/embeddings"),
                        help="Directory to write embedding caches (default: data/embeddings/)")
    parser.add_argument("--esm_model", type=str,
                        default="facebook/esm2_t6_8M_UR50D",
                        help="HuggingFace ESM-2 model ID")
    parser.add_argument("--embed_batch_size", type=int, default=32,
                        help="Sequences per forward pass (default: 32)")
    parser.add_argument("--modes", nargs="+",
                        default=["mean", "delta"],
                        choices=["mean", "delta"],
                        help="Embedding modes to compute (default: mean delta)")
    return parser.parse_args()


def _setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("rag_al.embed")


def main() -> None:
    args = _parse_args()
    log = _setup_logging()

    # Import here to keep startup fast for --help
    from ..data.loader import load_dataset
    from ..representations.plm import ESMEncoder

    data_csv = args.data_dir / f"{args.dataset}.csv"
    log.info("Loading dataset: %s", data_csv)
    df = load_dataset(data_csv)
    log.info("Dataset size: %d variants", len(df))

    cache_dir = args.embed_cache_dir / args.dataset
    cache_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        log.info("Computing %s embeddings with %s", mode, args.esm_model)

        encoder = ESMEncoder(
            model_name=args.esm_model,
            mode=mode,
            embed_batch_size=args.embed_batch_size,
            cache_dir=cache_dir,
        )

        # fit() needs WT sequence for delta mode
        encoder.fit(df, np.zeros(len(df)))  # y is not used by ESMEncoder.fit

        # transform() computes and caches all embeddings
        embeddings = encoder.transform(df)

        log.info(
            "Mode=%s  shape=%s  cached to %s",
            mode, embeddings.shape, cache_dir,
        )

    log.info("All embeddings computed and cached.")


if __name__ == "__main__":
    main()
