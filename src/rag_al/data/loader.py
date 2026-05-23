from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import validate_schema


def load_dataset(path: Path | str) -> pd.DataFrame:
    """
    Load a curated dataset CSV and validate its schema.

    The CSV must conform to the target schema defined in ``schema.py``
    (columns: variant_id, mutant, mutated_sequence, wt_sequence, fitness).
    No ProteinGym-specific logic is applied here; it is the caller's
    responsibility to pre-process raw data into the target format.

    Parameters
    ----------
    path : Path or str
        Path to the curated CSV file.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with a clean integer index.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at *path*.
    SchemaError
        If required columns are missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    df = pd.read_csv(path)
    df = df.reset_index(drop=True)

    # Ensure sequences are stored as plain strings (no accidental float parsing)
    for col in ("mutated_sequence", "wt_sequence", "mutant", "variant_id"):
        if col in df.columns:
            df[col] = df[col].astype(str)

    validate_schema(df)
    return df
