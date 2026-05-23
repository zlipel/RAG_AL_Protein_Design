from __future__ import annotations

import pandas as pd


# ------------------------------------------------------------------
# Expected column names in any curated dataset CSV
# ------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    "variant_id",        # str  — unique identifier
    "mutant",            # str  — e.g. "A23V" or "A23V:G45L"
    "mutated_sequence",  # str  — full amino acid sequence of the variant
    "wt_sequence",       # str  — wild-type amino acid sequence
    "fitness",           # float — measured fitness / variant-effect score
)


class SchemaError(ValueError):
    """Raised when a dataset DataFrame does not conform to the target schema."""


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that *df* conforms to the curated dataset schema.

    Checks
    ------
    - All REQUIRED_COLUMNS are present.
    - ``fitness`` column is numeric.
    - ``mutated_sequence`` and ``wt_sequence`` contain only upper-case
      single-letter amino acid codes (no gaps, no unknowns that would
      silently break encoders).
    - No NaN values in any required column.

    Raises
    ------
    SchemaError
        If any check fails, with a descriptive message.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Dataset is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Numeric fitness
    if not pd.api.types.is_numeric_dtype(df["fitness"]):
        raise SchemaError(
            "Column 'fitness' must be numeric. "
            f"Got dtype: {df['fitness'].dtype}"
        )

    # No NaNs in required columns
    for col in REQUIRED_COLUMNS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            raise SchemaError(
                f"Column '{col}' contains {n_nan} NaN value(s). "
                "Please clean the dataset before use."
            )

    # Basic sequence alphabet check (warn-level; don't block on unusual AAs)
    _STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
    for col in ("mutated_sequence", "wt_sequence"):
        sample = df[col].iloc[0]
        unknown = set(sample.upper()) - _STANDARD_AA
        if unknown:
            raise SchemaError(
                f"Column '{col}' contains non-standard characters: {unknown}. "
                "Sequences must use standard single-letter amino acid codes."
            )
