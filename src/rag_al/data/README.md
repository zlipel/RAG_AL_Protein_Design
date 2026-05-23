# rag_al/data

Data loading, schema validation, and the active learning dataset object.
This layer sits between raw CSV files on disk and the rest of the pipeline.

---

## schema.py

Defines the target data schema that every curated dataset CSV must conform to.
The pipeline is intentionally agnostic about data source (ProteinGym or otherwise).
The user is responsible for converting raw data into this format before use.

### Required columns

| Column           | Type  | Description                                              |
|------------------|-------|----------------------------------------------------------|
| variant_id       | str   | Unique identifier for each variant                       |
| mutant           | str   | Mutation string, e.g. "A23V" or "A23V:G45L"             |
| mutated_sequence | str   | Full amino acid sequence of the variant                  |
| wt_sequence      | str   | Wild-type amino acid sequence (same for all rows)        |
| fitness          | float | Measured fitness or variant-effect score                 |

### validate_schema(df)
Checks a DataFrame against the schema. Raises SchemaError (a subclass of
ValueError) with descriptive messages if:
- Any required column is missing
- The fitness column is non-numeric
- Any required column contains NaN values
- mutated_sequence or wt_sequence contains characters outside the standard
  20 single-letter amino acid codes

SchemaError messages are designed to be actionable — they tell you exactly
which column is problematic and why, so you can fix your data file.

### Mutant string format
Single mutations:  "A23V"   (WT residue, 1-indexed position, mutant residue)
Multiple mutations: "A23V:G45L"  (colon-separated, same format per site)

This format is standard in ProteinGym and most DMS databases.

---

## loader.py

### load_dataset(path) -> pd.DataFrame
Reads a CSV from disk and validates it against the schema. Returns a clean
DataFrame with a contiguous integer index (0 to N-1).

String columns (variant_id, mutant, mutated_sequence, wt_sequence) are
explicitly cast to str to prevent pandas from misinterpreting them as
numeric types (which can happen with certain mutation strings).

Raises FileNotFoundError if the CSV does not exist, or SchemaError if
the schema check fails.

This function contains no ProteinGym-specific parsing logic. If your raw
data has different column names or needs any transformations (e.g., renaming
"DMS_score" to "fitness", extracting wt_sequence from a metadata file),
do that in a separate preprocessing script before calling load_dataset.

---

## al_dataset.py

### ALDataset
The central data structure for the active learning loop. It holds all N
variants from a dataset, tracks which ones are currently labeled, and
enforces the leakage rules defined in the project spec.

### Construction
    dataset = ALDataset(df, n_init=50, seed=0)

This randomly selects n_init variants as the initial labeled set (using
the given random seed for reproducibility) and treats the rest as the
unlabeled pool. The full fitness array is stored as a private, name-mangled
attribute (ALDataset._ALDataset__fitness) that cannot be accidentally
accessed from outside the class.

### Properties available to the AL loop (leakage-safe)

labeled_df
    A DataFrame with feature columns (variant_id, mutant, mutated_sequence,
    wt_sequence) for currently labeled variants. The fitness column is
    intentionally excluded.

labeled_y
    Numpy array of fitness scores for the currently labeled variants.
    Shape: (n_labeled,). This is the only legitimate path to labeled
    fitness values during the loop.

labeled_indices
    Global indices (0 to N-1) of currently labeled variants.

pool_df
    Feature columns (no fitness) for unlabeled pool variants. This is
    what encoders receive for the pool.

pool_indices
    Global indices of unlabeled pool variants. Acquisition functions work
    with local pool indices (0 to n_pool-1); use pool_indices to map
    those back to global indices.

n_labeled, n_pool, n_total
    Size helpers.

### reveal(pool_local_indices)
The only authorized path to move labels from hidden to labeled. Takes
local pool indices (0-based into the current pool order), maps them to
global dataset indices, and marks them as labeled.

If any requested index is already labeled, a LeakageError is raised.
This guards against double-acquisition bugs.

### Metric helpers (full label access allowed for evaluation)

global_optimum
    Best fitness in the full dataset. Used to compute simple regret.
    This is precomputed at construction time and should only be used
    by metrics — never inside an acquisition function.

top_k_global_indices(k)
    Global indices of the k best variants by fitness, in descending order.
    Used to compute top-k recall. Same restriction as global_optimum.

### LeakageError
A custom RuntimeError subclass raised when a leakage rule is violated.
Currently triggered by reveal() if already-labeled indices are passed.
The name-mangling of __fitness provides a second layer of protection
against accidental direct access.

### Why local vs global indices?
The pool changes every round as variants are acquired. "Local" indices
(0 to n_pool-1) are relative to the current pool order at a given round;
"global" indices (0 to N-1) are fixed for the lifetime of the dataset.
Acquisition functions return local indices. reveal() accepts local indices
and translates internally. This avoids index confusion bugs.
