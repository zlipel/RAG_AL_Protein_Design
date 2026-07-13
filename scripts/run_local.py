#!/usr/bin/env python3
"""
run_local.py — Run a local benchmark sweep in parallel using subprocess workers.

Generates all (representation × acquisition × seed) cells for a given dataset
and runs them concurrently as rag-benchmark subprocesses.

Usage
-----
python scripts/run_local.py \
    --dataset    BLAT_ECOLX_Jacquier_2013 \
    --reprs      mutation physicochemical \
    --acqs       random greedy ucb diversity_ucb retrieval_ucb \
    --n_seeds    3 \
    --n_rounds   10 \
    --batch_size 10 \
    --workers    12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path


_ALL_REPRS = ["mutation", "physicochemical", "plm_mean", "plm_delta", "plm_retrieval"]
_ALL_ACQS  = ["random", "greedy", "ucb", "diversity_ucb", "retrieval_ucb"]


def build_cmd(
    *,
    dataset: str,
    representation: str,
    acquisition: str,
    seed: int,
    n_rounds: int,
    batch_size: int,
    n_init: int,
    n_estimators: int,
    rf_n_jobs: int,
    ucb_beta: float,
    esm_model: str,
    python: str,
) -> list[str]:
    return [
        python, "-m", "rag_al.cli.benchmark",
        "--dataset",        dataset,
        "--representation", representation,
        "--acquisition",    acquisition,
        "--seed",           str(seed),
        "--n_rounds",       str(n_rounds),
        "--batch_size",     str(batch_size),
        "--n_init",         str(n_init),
        "--n_estimators",   str(n_estimators),
        "--rf_n_jobs",      str(rf_n_jobs),
        "--ucb_beta",       str(ucb_beta),
        "--esm_model",      esm_model,
    ]


def run_cell(cmd: list[str], label: str) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    elapsed = time.perf_counter() - t0
    return label, result.returncode, elapsed, result.stderr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local multi-cell RAG-AL benchmark sweep in parallel."
    )
    parser.add_argument("--dataset",    required=True, type=str)
    parser.add_argument("--reprs",      nargs="+", default=["mutation", "physicochemical"],
                        choices=_ALL_REPRS, metavar="REPR")
    parser.add_argument("--acqs",       nargs="+", default=_ALL_ACQS,
                        choices=_ALL_ACQS, metavar="ACQ")
    parser.add_argument("--n_seeds",    type=int, default=3)
    parser.add_argument("--n_rounds",   type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_init",     type=int, default=50)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--ucb_beta",   type=float, default=1.0,
                        help="UCB exploration weight β (default: 1.0)")
    parser.add_argument("--esm_model",  type=str, default="facebook/esm2_t6_8M_UR50D",
                        help="HuggingFace ESM-2 model ID for PLM representations")
    parser.add_argument("--workers",    type=int, default=12,
                        help="Max parallel subprocesses (default: 12)")
    parser.add_argument("--python",     type=str, default=sys.executable,
                        help="Python interpreter to use (default: current env)")
    args = parser.parse_args()

    cells = list(product(args.reprs, args.acqs, range(args.n_seeds)))
    n_total = len(cells)
    print(
        f"Submitting {n_total} cells "
        f"({len(args.reprs)} reprs × {len(args.acqs)} acqs × {args.n_seeds} seeds) "
        f"with {args.workers} workers\n"
    )

    futures = {}
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for repr_name, acq, seed in cells:
            label = f"{repr_name}/{acq}/s{seed}"
            cmd = build_cmd(
                dataset=args.dataset,
                representation=repr_name,
                acquisition=acq,
                seed=seed,
                n_rounds=args.n_rounds,
                batch_size=args.batch_size,
                n_init=args.n_init,
                n_estimators=args.n_estimators,
                rf_n_jobs=1,            # 1 core per cell; workers fill all cores
                ucb_beta=args.ucb_beta,
                esm_model=args.esm_model,
                python=args.python,
            )
            futures[pool.submit(run_cell, cmd, label)] = label


        n_done = 0
        n_failed = 0
        for future in as_completed(futures):
            label, rc, elapsed, stderr = future.result()
            n_done += 1
            status = "OK" if rc == 0 else "FAIL"
            if rc != 0:
                n_failed += 1
            print(f"[{n_done:>3}/{n_total}] {status}  {label}  ({elapsed:.1f}s)")
            if rc != 0:
                # Print last 5 lines of stderr for failed cells
                lines = stderr.strip().splitlines()
                for line in lines[-5:]:
                    print(f"       {line}")

    total_elapsed = time.perf_counter() - t_start
    print(
        f"\nDone: {n_total - n_failed}/{n_total} cells succeeded "
        f"in {total_elapsed:.1f}s wall time."
    )
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
