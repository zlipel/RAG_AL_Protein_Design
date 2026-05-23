from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import AbstractEncoder


# ------------------------------------------------------------------
# Amino acid property tables
# ------------------------------------------------------------------

_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
_AA_INDEX = {aa: i for i, aa in enumerate(_AA_ORDER)}

_AA_HYDROPATHY: dict[str, float] = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_AA_MW: dict[str, float] = {
    "A": 89.1,  "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "Q": 146.2, "E": 147.1, "G": 75.0,  "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

_AROMATIC = frozenset("FYW")
_POLAR = frozenset("STNQ")
_POSITIVE = frozenset("RK")
_NEGATIVE = frozenset("DE")
_CHARGED = _POSITIVE | _NEGATIVE


def _encode_one(sequence: str) -> np.ndarray:
    """
    Encode a single amino acid sequence into a 29-dimensional
    physicochemical feature vector.

    Feature layout
    --------------
    [0:20]  amino acid composition (frequency of each of the 20 standard AAs)
    [20]    net charge  (sum of charge per residue / length)
    [21]    mean hydropathy (Kyte-Doolittle)
    [22]    aromatic fraction  (F + Y + W) / length
    [23]    polar fraction     (S + T + N + Q) / length
    [24]    charged fraction   (R + K + D + E) / length
    [25]    positive fraction  (R + K) / length
    [26]    negative fraction  (D + E) / length
    [27]    log sequence length  log(length + 1)
    [28]    Shannon entropy

    Parameters
    ----------
    sequence : str
        Amino acid sequence (single-letter, uppercase).

    Returns
    -------
    np.ndarray
        Shape (29,), dtype float64.
    """
    seq = sequence.upper()
    L = len(seq)
    if L == 0:
        return np.zeros(29, dtype=float)

    counts = np.zeros(20, dtype=float)
    net_charge = 0.0
    hydropathy_sum = 0.0
    n_aromatic = 0
    n_polar = 0
    n_positive = 0
    n_negative = 0
    mw_sum = 0.0

    for aa in seq:
        idx = _AA_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1.0
        net_charge += _AA_HYDROPATHY.get(aa, 0.0) * 0.0  # placeholder replaced below
        # Charge from +1/-1 residues
        if aa in _POSITIVE:
            net_charge += 1.0
            n_positive += 1
        elif aa in _NEGATIVE:
            net_charge -= 1.0
            n_negative += 1
        hydropathy_sum += _AA_HYDROPATHY.get(aa, 0.0)
        if aa in _AROMATIC:
            n_aromatic += 1
        if aa in _POLAR:
            n_polar += 1
        mw_sum += _AA_MW.get(aa, 110.0)

    # Composition (frequency)
    composition = counts / L

    # Shannon entropy over AA distribution
    probs = composition[composition > 0]
    entropy = float(-np.sum(probs * np.log(probs)))

    feat = np.empty(29, dtype=float)
    feat[0:20] = composition
    feat[20] = net_charge / L
    feat[21] = hydropathy_sum / L
    feat[22] = n_aromatic / L
    feat[23] = n_polar / L
    feat[24] = (n_positive + n_negative) / L    # charged fraction
    feat[25] = n_positive / L
    feat[26] = n_negative / L
    feat[27] = math.log(L + 1)
    feat[28] = entropy
    return feat


class PhysicochemicalEncoder(AbstractEncoder):
    """
    Physicochemical sequence descriptor encoder.

    Computes a 29-dimensional feature vector from the full amino acid
    sequence of each variant (AA composition, net charge, hydropathy,
    aromatic/polar/charged fractions, sequence length, Shannon entropy).

    Continuous features are standardized using statistics fit on the
    labeled set only (leakage-safe).

    Output dimension: 29
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler | None = None

    def fit(self, df_labeled: pd.DataFrame, y_labeled: np.ndarray) -> None:
        """
        Fit the StandardScaler on labeled variants.

        Parameters
        ----------
        df_labeled : pd.DataFrame
            Labeled feature columns (must contain 'mutated_sequence').
        y_labeled : np.ndarray
            Ignored (no fitness information used in this encoder).
        """
        X = self._raw_transform(df_labeled)
        self._scaler = StandardScaler()
        self._scaler.fit(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encode variants using physicochemical descriptors.

        Parameters
        ----------
        df : pd.DataFrame
            Feature columns (must contain 'mutated_sequence').

        Returns
        -------
        np.ndarray
            Shape (n_variants, 29), standardized float64.
        """
        if self._scaler is None:
            raise RuntimeError("Call fit() before transform().")
        X = self._raw_transform(df)
        return self._scaler.transform(X)

    def _raw_transform(self, df: pd.DataFrame) -> np.ndarray:
        rows = [_encode_one(seq) for seq in df["mutated_sequence"]]
        return np.vstack(rows)

    @property
    def n_features(self) -> int:
        return 29
