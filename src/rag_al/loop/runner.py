from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..data.al_dataset import ALDataset
from ..representations.base import AbstractEncoder
from ..surrogates.base import AbstractSurrogate
from ..acquisition.base import AbstractAcquisition
from .metrics import compute_round_metrics


def run_al_loop(
    dataset: ALDataset,
    encoder: AbstractEncoder,
    surrogate: AbstractSurrogate,
    acquisition: AbstractAcquisition,
    n_rounds: int,
    batch_size: int,
    seed: int = 0,
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Run a pool-based retrospective active learning loop.

    At each round:
      1. Fit the encoder on the current labeled set (no fitness leak to pool).
      2. Encode the labeled set and the unlabeled pool.
      3. Fit the surrogate on labeled features + fitness.
      4. Predict (μ, σ) for all pool variants.
      5. Select a batch via the acquisition function.
      6. Reveal the selected variants' fitness labels.
      7. Record metrics.

    Leakage guarantees
    ------------------
    - The encoder's ``fit()`` receives only ``dataset.labeled_df`` and
      ``dataset.labeled_y`` — no pool fitness.
    - ``dataset.pool_df`` contains no fitness column.
    - The acquisition function receives only the labeled arrays as context.
    - ``dataset.reveal()`` is the only path through which pool labels are
      moved to the labeled set.

    Parameters
    ----------
    dataset : ALDataset
        Active learning dataset (already initialized with n_init labeled samples).
    encoder : AbstractEncoder
        Sequence encoder (fit on labeled set each round).
    surrogate : AbstractSurrogate
        Surrogate model (fit on labeled features each round).
    acquisition : AbstractAcquisition
        Acquisition function.
    n_rounds : int
        Number of active learning rounds to run.
    batch_size : int
        Number of variants to acquire per round.
    seed : int
        Random seed for acquisition tie-breaking.
    log : logging.Logger, optional
        Logger instance. If None, uses module-level logger.

    Returns
    -------
    pd.DataFrame
        One row per round. Columns:
        round, n_labeled, best_fitness, simple_regret,
        topk10_recall, topk50_recall, batch_mean_fitness,
        batch_diversity, mean_dist_wt.
    """
    if log is None:
        log = logging.getLogger(__name__)

    rng = np.random.default_rng(seed)

    # Precompute metric helpers from the full dataset
    # (allowed: metric computation uses the full label array)
    global_optimum = dataset.global_optimum
    top10_idx = dataset.top_k_global_indices(10)
    top50_idx = dataset.top_k_global_indices(50)
    wt_sequence: str = dataset._df["wt_sequence"].iloc[0]

    rows: list[dict] = []

    log.info(
        "Starting AL loop: n_rounds=%d  batch_size=%d  n_init=%d  n_pool=%d",
        n_rounds, batch_size, dataset.n_labeled, dataset.n_pool,
    )

    for round_idx in range(n_rounds):
        log.info("Round %d / %d  (labeled=%d, pool=%d)",
                 round_idx + 1, n_rounds, dataset.n_labeled, dataset.n_pool)

        if dataset.n_pool == 0:
            log.warning("Pool exhausted at round %d — stopping early.", round_idx)
            break

        # ---- 1. Encode labeled set ------------------------------------------
        df_labeled = dataset.labeled_df
        y_labeled = dataset.labeled_y

        try:
            encoder.fit(df_labeled, y_labeled)
        except Exception as e:
            log.exception("Encoder fit failed at round %d: %s", round_idx, e)
            raise

        X_labeled = encoder.transform(df_labeled)   # (n_lab, D)

        # ---- 2. Encode pool --------------------------------------------------
        df_pool = dataset.pool_df
        X_pool = encoder.transform(df_pool)          # (n_pool, D)

        # ---- 3. Fit surrogate ------------------------------------------------
        try:
            surrogate.fit(X_labeled, y_labeled)
        except Exception as e:
            log.exception("Surrogate fit failed at round %d: %s", round_idx, e)
            raise

        # ---- 4. Predict for pool --------------------------------------------
        mu, sigma = surrogate.predict(X_pool)        # (n_pool,), (n_pool,)

        # ---- 5. Select batch ------------------------------------------------
        selected_local = acquisition.select_batch(
            mu, sigma, batch_size,
            pool_X=X_pool,
            labeled_X=X_labeled,
            labeled_y=y_labeled,
            rng=rng,
        )
        selected_local = np.asarray(selected_local, dtype=int)
        actual_batch = min(batch_size, dataset.n_pool)
        if len(selected_local) > actual_batch:
            selected_local = selected_local[:actual_batch]

        log.debug(
            "Round %d: selected %d variants (pool local indices: %s …)",
            round_idx, len(selected_local), selected_local[:5],
        )

        # ---- 6. Reveal -------------------------------------------------------
        # Convert local → global to get batch sequences and fitness AFTER reveal
        global_selected = dataset.pool_indices[selected_local]
        batch_sequences = list(
            dataset._df.loc[global_selected, "mutated_sequence"]
        )
        dataset.reveal(selected_local)

        # Fitness of the newly revealed batch (allowed post-reveal)
        batch_y = dataset.labeled_y[-len(selected_local):]

        # ---- 7. Metrics ------------------------------------------------------
        row = compute_round_metrics(
            round_idx=round_idx,
            labeled_y=dataset.labeled_y,
            labeled_indices=dataset.labeled_indices,
            batch_y=batch_y,
            batch_sequences=batch_sequences,
            wt_sequence=wt_sequence,
            global_optimum=global_optimum,
            top10_indices=top10_idx,
            top50_indices=top50_idx,
        )
        rows.append(row)
        log.info(
            "Round %d done — best=%.4f  regret=%.4f  top10_recall=%.3f",
            round_idx, row["best_fitness"], row["simple_regret"], row["topk10_recall"],
        )

    results = pd.DataFrame(rows)
    log.info("AL loop complete. Final best fitness: %.4f", results["best_fitness"].max())
    return results
