#!/usr/bin/env python3
"""
curate_proteingym.py — Convert a ProteinGym substitution CSV to the pipeline schema.

Input format (data/DMS_ProteinGym_substitutions/<name>.csv):
    mutant, mutated_sequence, DMS_score, DMS_score_bin

Output format (data/<name>.csv):
    variant_id, mutant, mutated_sequence, wt_sequence, fitness

Usage
-----
python scripts/curate_proteingym.py \\
    --input       data/DMS_ProteinGym_substitutions/BLAT_ECOLX_Jacquier_2013.csv \\
    --output_dir  data/curated
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _reconstruct_wt(mutant_str: str, mutated_seq: str) -> str | None:
    """
    Reconstruct the WT sequence by reverting each substitution in mutant_str.

    Parameters
    ----------
    mutant_str : str
        Colon-separated mutation tokens, e.g. "H24Y" or "H24Y:G45L".
    mutated_seq : str
        Full mutated amino-acid sequence.

    Returns
    -------
    str or None
        Reconstructed WT sequence, or None if any token is unparseable or the
        expected mutant AA is not present at the stated position (position-offset
        mismatch; caller should log and skip the row).
    """
    seq = list(mutated_seq)
    for token in mutant_str.split(":"):
        m = _MUTATION_RE.match(token)
        if not m:
            log.warning("Cannot parse mutation token %r — skipping row.", token)
            return None
        wt_aa, pos, mut_aa = m.group(1), int(m.group(2)), m.group(3)
        idx = pos - 1  # 1-indexed → 0-indexed
        if idx < 0 or idx >= len(seq):
            log.warning(
                "Position %d out of range for sequence of length %d — skipping row.",
                pos, len(seq),
            )
            return None
        if seq[idx] != mut_aa:
            log.warning(
                "Position mismatch at %d: expected %r (mutant), found %r — "
                "possible numbering offset; skipping row.",
                pos, mut_aa, seq[idx],
            )
            return None
        seq[idx] = wt_aa
    return "".join(seq)


def curate(input_path: Path, output_dir: Path) -> pd.DataFrame:
    """
    Load a ProteinGym substitution CSV, derive missing schema fields, validate,
    and write to output_dir/<input_stem>.csv.

    Returns the curated DataFrame.
    """
    output_path = output_dir / (input_path.stem + ".csv")
    df = pd.read_csv(input_path)
    log.info("Loaded %d rows from %s", len(df), input_path)

    # ------------------------------------------------------------------
    # 1. Check required source columns
    # ------------------------------------------------------------------
    for col in ("mutant", "mutated_sequence", "DMS_score"):
        if col not in df.columns:
            raise ValueError(
                f"Input CSV is missing required column '{col}'. "
                f"Found: {list(df.columns)}"
            )

    # ------------------------------------------------------------------
    # 2. Drop rows with missing DMS_score
    # ------------------------------------------------------------------
    n_before = len(df)
    df = df.dropna(subset=["DMS_score"]).reset_index(drop=True)
    if len(df) < n_before:
        log.info("Dropped %d rows with NaN DMS_score.", n_before - len(df))

    # ------------------------------------------------------------------
    # 3. Reconstruct wt_sequence for each row
    # ------------------------------------------------------------------
    wt_col: list[str | None] = [
        _reconstruct_wt(row["mutant"], row["mutated_sequence"])
        for _, row in df.iterrows()
    ]

    n_failed = sum(1 for w in wt_col if w is None)
    if n_failed:
        log.warning(
            "%d row(s) could not be reconstructed (numbering mismatch or "
            "parse error) — dropping them.", n_failed,
        )
        df = df[[w is not None for w in wt_col]].reset_index(drop=True)
        wt_col = [w for w in wt_col if w is not None]

    df["wt_sequence"] = wt_col

    # ------------------------------------------------------------------
    # 4. Verify WT consistency across all rows
    # ------------------------------------------------------------------
    unique_wt = set(df["wt_sequence"])
    if len(unique_wt) != 1:
        raise ValueError(
            f"Reconstructed {len(unique_wt)} distinct WT sequences — "
            "expected exactly 1. This may indicate a position-numbering "
            "inconsistency in the source file."
        )
    wt_seq = next(iter(unique_wt))
    log.info("WT sequence length: %d residues.", len(wt_seq))

    # ------------------------------------------------------------------
    # 5. Build output columns
    # ------------------------------------------------------------------
    dataset_stem = input_path.stem
    df["variant_id"] = dataset_stem + "__" + df["mutant"]
    df["fitness"] = df["DMS_score"]

    out = df[["variant_id", "mutant", "mutated_sequence", "wt_sequence", "fitness"]].copy()

    # ------------------------------------------------------------------
    # 6. Schema validation
    # ------------------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from rag_al.data.schema import validate_schema
    validate_schema(out)
    log.info("Schema validation passed.")

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("Wrote %d variants to %s", len(out), output_path)

    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Curate a ProteinGym substitution CSV into the pipeline schema."
    )
    parser.add_argument("--input",      required=True, type=Path,
                        help="Path to the raw ProteinGym substitution CSV.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/curated"),
                        help="Directory to write the curated CSV (default: data/curated).")
    args = parser.parse_args()

    curate(args.input, args.output_dir)


if __name__ == "__main__":
    main()
