from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import AbstractEncoder


# ------------------------------------------------------------------
# Amino acid physicochemical property tables (Kyte-Doolittle / standard)
# ------------------------------------------------------------------

_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}

_AA_HYDROPATHY: dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_AA_CHARGE: dict[str, float] = {
    "R": 1.0, "K": 1.0, "D": -1.0, "E": -1.0,
    **{aa: 0.0 for aa in "ACFGHILMNPQSTVWY"},
}

_AA_VOLUME: dict[str, float] = {
    "A": 88.6,  "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1,  "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0,  "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


class _ParsedMutation(NamedTuple):
    wt_aa: str
    position: int     # 1-indexed
    mut_aa: str


def _parse_mutant_string(mutant: str) -> list[_ParsedMutation]:
    """
    Parse a mutation string (e.g., 'A23V' or 'A23V:G45L') into a list
    of (wt_aa, position, mut_aa) named tuples.

    Raises
    ------
    ValueError
        If any individual mutation token is malformed.
    """
    tokens = mutant.split(":")
    parsed: list[_ParsedMutation] = []
    for tok in tokens:
        m = _MUTATION_RE.match(tok.strip())
        if m is None:
            raise ValueError(
                f"Could not parse mutation token: {tok!r}. "
                "Expected format: <WT_AA><1-indexed position><MUT_AA>, e.g. 'A23V'."
            )
        parsed.append(_ParsedMutation(m.group(1), int(m.group(2)), m.group(3)))
    return parsed


def _encode_one(mutant: str, seq_len: int) -> np.ndarray:
    """
    Encode a single variant into a fixed-length mutation descriptor vector.

    Feature layout (49 dims total)
    --------------------------------
    [0]       n_mutations
    [1]       mean_position (normalized by seq_len)
    [2]       std_position  (normalized; 0.0 for single mutations)
    [3]       sum_delta_hydropathy
    [4]       sum_delta_charge
    [5]       sum_delta_volume  (normalized by 100.0)
    [6]       mean_delta_hydropathy
    [7]       mean_delta_charge
    [8]       mean_delta_volume (normalized by 100.0)
    [9:29]    wt_aa_counts  (20-dim, one-hot accumulated over mutation sites)
    [29:49]   mut_aa_counts (20-dim, one-hot accumulated)

    Parameters
    ----------
    mutant : str
        Mutation string, e.g. 'A23V' or 'A23V:G45L'.
    seq_len : int
        Length of the protein sequence (for position normalization).

    Returns
    -------
    np.ndarray
        Shape (49,), dtype float64.
    """
    muts = _parse_mutant_string(mutant)
    n = len(muts)

    positions = np.array([m.position for m in muts], dtype=float)
    delta_h = np.array(
        [_AA_HYDROPATHY.get(m.mut_aa, 0.0) - _AA_HYDROPATHY.get(m.wt_aa, 0.0)
         for m in muts]
    )
    delta_c = np.array(
        [_AA_CHARGE.get(m.mut_aa, 0.0) - _AA_CHARGE.get(m.wt_aa, 0.0)
         for m in muts]
    )
    delta_v = np.array(
        [((_AA_VOLUME.get(m.mut_aa, 0.0) - _AA_VOLUME.get(m.wt_aa, 0.0)) / 100.0)
         for m in muts]
    )

    wt_counts = np.zeros(20, dtype=float)
    mut_counts = np.zeros(20, dtype=float)
    for m in muts:
        if m.wt_aa in _AA_INDEX:
            wt_counts[_AA_INDEX[m.wt_aa]] += 1.0
        if m.mut_aa in _AA_INDEX:
            mut_counts[_AA_INDEX[m.mut_aa]] += 1.0

    feat = np.empty(49, dtype=float)
    feat[0] = float(n)
    feat[1] = float(positions.mean()) / max(seq_len, 1)
    feat[2] = float(positions.std()) / max(seq_len, 1) if n > 1 else 0.0
    feat[3] = float(delta_h.sum())
    feat[4] = float(delta_c.sum())
    feat[5] = float(delta_v.sum())
    feat[6] = float(delta_h.mean())
    feat[7] = float(delta_c.mean())
    feat[8] = float(delta_v.mean())
    feat[9:29] = wt_counts
    feat[29:49] = mut_counts
    return feat


class MutationDescriptorEncoder(AbstractEncoder):
    """
    Fixed-length mutation descriptor encoder.

    Parses the 'mutant' column to extract per-site amino acid identity
    and physicochemical property changes (hydropathy, charge, volume),
    then aggregates over all mutation sites in a variant.

    Continuous features are standardized using statistics fit on the
    labeled set only (leakage-safe).

    Output dimension: 49
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None
        self._seq_len: int = 0

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """
        Fit the StandardScaler on labeled variants.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Labeled feature columns (must contain 'mutant', 'mutated_sequence').
        y_labeled : np.ndarray
            Ignored (no fitness information used in this encoder).
        """
        self._seq_len = len(df_labeled["mutated_sequence"].iloc[0])
        X = self._raw_transform(df_labeled)
        self._scaler = StandardScaler()
        self._scaler.fit(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants using mutation descriptors.

        Parameters
        ----------
        df : pd.DataFrame
            Feature columns (must contain 'mutant', 'mutated_sequence').

        Returns
        -------
        np.ndarray
            Shape (n_variants, 49), standardized float64.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before transform().")
        X = self._raw_transform(df)
        return self._scaler.transform(X)

    def _raw_transform(self, df: pd.DataFrame) -> np.ndarray:
        seq_len = len(df["mutated_sequence"].iloc[0])
        rows = [_encode_one(row.mutant, seq_len) for row in df.itertuples(index=False)]
        return np.vstack(rows)

    @property
    def n_features(self) -> int:
        return 49
