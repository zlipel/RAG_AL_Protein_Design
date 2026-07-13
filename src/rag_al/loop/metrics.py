from __future__ import annotations

import itertools

import numpy as np


def best_fitness(labeled_y: np.ndarray) -> float:
    """
    Best observed fitness in the current labeled set.

        max_{x ∈ L_t} f(x)

    Parameters
    ----------
    labeled_y : np.ndarray
        Fitness scores of currently labeled variants. Shape: (n_labeled,).

    Returns
    -------
    float
    """
    return float(labeled_y.max())


def simple_regret(labeled_y: np.ndarray, global_optimum: float) -> float:
    """
    Gap between the global optimum and the best observed fitness.

        regret = f(x*) − max_{x ∈ L_t} f(x)

    Parameters
    ----------
    labeled_y : np.ndarray
        Fitness scores of currently labeled variants.
    global_optimum : float
        Best fitness in the full dataset (``ALDataset.global_optimum``).

    Returns
    -------
    float
        Non-negative regret value.
    """
    return float(global_optimum - best_fitness(labeled_y))


def topk_recall(
    labeled_indices: np.ndarray,
    top_k_indices: np.ndarray,
) -> float:
    """
    Fraction of top-k variants that have been acquired.

        recall = |L_t ∩ TopK(D)| / K

    Parameters
    ----------
    labeled_indices : np.ndarray
        Global indices of currently labeled variants.
    top_k_indices : np.ndarray
        Global indices of the top-k variants by fitness
        (``ALDataset.top_k_global_indices(k)``).

    Returns
    -------
    float
        Value in [0, 1].
    """
    k = len(top_k_indices)
    if k == 0:
        return 0.0
    overlap = len(set(labeled_indices.tolist()) & set(top_k_indices.tolist()))
    return overlap / k


def batch_mean_fitness(batch_y: np.ndarray) -> float:
    """
    Mean fitness of the most recently acquired batch.

    Parameters
    ----------
    batch_y : np.ndarray
        Fitness scores of the acquired batch. Shape: (batch_size,).

    Returns
    -------
    float
    """
    return float(batch_y.mean())


def batch_diversity(batch_sequences: list[str]) -> float:
    """
    Mean pairwise Hamming distance among variants in the acquired batch,
    normalized by sequence length.

    Hamming distance is defined only for sequences of equal length.
    If sequences have different lengths, pairs are padded to the longer
    length with a null character (treated as mismatch).

    Parameters
    ----------
    batch_sequences : list of str
        Amino acid sequences of the acquired variants.

    Returns
    -------
    float
        Mean pairwise normalized Hamming distance. 0 = identical, 1 = fully diverse.
        Returns 0.0 for batches of size ≤ 1.
    """
    n = len(batch_sequences)
    if n <= 1:
        return 0.0

    max_len = max(len(s) for s in batch_sequences)
    # Pad shorter sequences with null character
    padded = [s.ljust(max_len, "\x00") for s in batch_sequences]
    arrs = np.array([[ord(c) for c in s] for s in padded], dtype=np.int32)

    total = 0.0
    count = 0
    for i, j in itertools.combinations(range(n), 2):
        total += float(np.mean(arrs[i] != arrs[j]))
        count += 1

    return total / count if count > 0 else 0.0


def mean_dist_from_wt(batch_sequences: list[str], wt_sequence: str) -> float:
    """
    Mean normalized Hamming distance of the acquired batch from the wild-type.

    Parameters
    ----------
    batch_sequences : list of str
        Amino acid sequences of the acquired variants.
    wt_sequence : str
        Wild-type amino acid sequence.

    Returns
    -------
    float
        Mean normalized Hamming distance from WT. Value in [0, 1].
    """
    if not batch_sequences:
        return 0.0

    max_len = max(len(wt_sequence), max(len(s) for s in batch_sequences))
    wt_arr = np.array(
        [ord(c) for c in wt_sequence.ljust(max_len, "\x00")], dtype=np.int32
    )

    dists = []
    for seq in batch_sequences:
        seq_arr = np.array(
            [ord(c) for c in seq.ljust(max_len, "\x00")], dtype=np.int32
        )
        dists.append(float(np.mean(seq_arr != wt_arr)))

    return float(np.mean(dists))


def compute_round_metrics(
    round_idx: int,
    labeled_y: np.ndarray,
    labeled_indices: np.ndarray,
    batch_y: np.ndarray,
    batch_sequences: list[str],
    wt_sequence: str,
    global_optimum: float,
    top10_indices: np.ndarray,
    top50_indices: np.ndarray,
    pool_spearman: float = float("nan"),
) -> dict[str, float | int]:
    """
    Compute all metrics for one active learning round.

    Returns a flat dict suitable for appending to a results DataFrame.

    Parameters
    ----------
    round_idx : int
        Current AL round index (0-based).
    labeled_y : np.ndarray
        All labeled fitness scores after this round's reveal.
    labeled_indices : np.ndarray
        Global indices of all labeled variants after this round.
    batch_y : np.ndarray
        Fitness scores of the newly acquired batch.
    batch_sequences : list of str
        Sequences of the newly acquired batch.
    wt_sequence : str
        Wild-type sequence (for distance computation).
    global_optimum : float
        Best fitness in the full dataset.
    top10_indices : np.ndarray
        Global indices of the top-10 variants.
    top50_indices : np.ndarray
        Global indices of the top-50 variants.
    pool_spearman : float
        Spearman ρ between surrogate predictions and oracle fitness over the
        full unlabeled pool. Metric-only oracle read — same category as
        topk_recall. Defaults to NaN if not provided (e.g. in older tests).

    Returns
    -------
    dict
    """
    return {
        "round": round_idx,
        "n_labeled": len(labeled_y),
        "best_fitness": best_fitness(labeled_y),
        "simple_regret": simple_regret(labeled_y, global_optimum),
        "topk10_recall": topk_recall(labeled_indices, top10_indices),
        "topk50_recall": topk_recall(labeled_indices, top50_indices),
        "batch_mean_fitness": batch_mean_fitness(batch_y),
        "batch_diversity": batch_diversity(batch_sequences),
        "mean_dist_wt": mean_dist_from_wt(batch_sequences, wt_sequence),
        "pool_spearman": pool_spearman,
    }
