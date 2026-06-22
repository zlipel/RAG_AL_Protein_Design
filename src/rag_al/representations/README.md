# rag_al/representations

Sequence encoders that transform protein variant DataFrames into fixed-length
numeric feature vectors for the surrogate model. Each encoder follows the same
fit/transform interface and is leakage-safe by design.

---

## base.py

### AbstractEncoder
All encoders inherit from this base class. The interface has two methods:

fit(df_labeled, y_labeled)
    Fit any stateful components using ONLY the currently labeled set.
    df_labeled contains feature columns (no fitness). y_labeled is the
    fitness array for labeled variants. Most encoders use y_labeled only
    to store context for retrieval features (RetrievalAugmentedEncoder);
    others ignore it entirely.
    
    Leakage rule: fit() must never receive or store pool fitness values.
    Feature normalization statistics (e.g., StandardScaler means and
    variances) are fit here — on labeled data only.

transform(df) -> np.ndarray
    Encode a DataFrame of variants into a float64 feature matrix.
    Shape: (n_variants, n_features). The df contains feature columns
    (no fitness) and may be either the labeled set or the pool.
    
    Leakage rule: transform() operates on sequences only. It does not
    receive or access any fitness values.

fit_transform(df_labeled, y_labeled) -> np.ndarray
    Convenience method: fits on df_labeled then transforms it. Not used
    during the AL loop (where fit and transform are called separately on
    labeled and pool data respectively), but useful for testing.

n_features (property)
    Returns the output dimensionality. Used for sanity checks.

---

## mutation.py

### MutationDescriptorEncoder
Parses the 'mutant' column to extract per-site information about each
amino acid substitution, then aggregates across all mutation sites into
a fixed-length vector.

This encoder does not use the full sequence — only the mutation string
and sequence length. It is fast (no model loading) and interpretable.

### Mutation string parsing
Each mutation string is split on ":" to get individual tokens.
Each token is matched against the regex ^([A-Z])(\d+)([A-Z])$ to extract:
    wt_aa    — single-letter code of the wild-type residue
    position — 1-indexed position in the sequence
    mut_aa   — single-letter code of the introduced residue

### Feature vector (49 dimensions)
[0]      n_mutations        — total number of mutation sites
[1]      mean_position      — mean 1-indexed position, normalized by seq length
[2]      std_position       — std of positions, normalized by seq length
[3]      sum_delta_hydropathy  — sum of (mut - wt) Kyte-Doolittle hydropathy
[4]      sum_delta_charge      — sum of (mut - wt) formal charge
[5]      sum_delta_volume      — sum of (mut - wt) residue volume / 100
[6]      mean_delta_hydropathy — per-site average of hydropathy change
[7]      mean_delta_charge     — per-site average of charge change
[8]      mean_delta_volume     — per-site average of volume change / 100
[9:29]   wt_aa_counts       — 20-dim count of which WT residues are being mutated
[29:49]  mut_aa_counts      — 20-dim count of which mutant residues are introduced

Property tables used: Kyte-Doolittle hydropathy, formal charge (+1 for R/K,
-1 for D/E), and side-chain volume from standard references.

### Normalization
A StandardScaler is fit on the labeled set in fit() and applied in
transform(). This means feature means and variances are estimated from
labeled data only — no leakage.

---

## physicochemical.py

### PhysicochemicalEncoder
Computes whole-sequence physicochemical descriptors from the
'mutated_sequence' column. Does not use the mutation string.

This encoder is fast, interpretable, and captures global sequence
composition rather than local mutation effects.

### Feature vector (29 dimensions)
[0:20]  aa_composition     — frequency of each of the 20 standard amino acids
[20]    net_charge         — (count_R + count_K - count_D - count_E) / length
[21]    mean_hydropathy    — mean Kyte-Doolittle hydropathy across all residues
[22]    aromatic_fraction  — (count_F + count_Y + count_W) / length
[23]    polar_fraction     — (count_S + count_T + count_N + count_Q) / length
[24]    charged_fraction   — (count_R + count_K + count_D + count_E) / length
[25]    positive_fraction  — (count_R + count_K) / length
[26]    negative_fraction  — (count_D + count_E) / length
[27]    log_length         — log(sequence_length + 1)
[28]    shannon_entropy    — -sum(p_i * log(p_i)) over non-zero AA frequencies

### Normalization
Same as MutationDescriptorEncoder: StandardScaler fit on labeled set only.

---

## plm.py

### ESMEncoder
Computes protein language model embeddings using ESM-2 via the HuggingFace
transformers library. ESM-2 is a protein-specific transformer trained on
250 million protein sequences from UniRef50.

### Model selection
The model is specified by a HuggingFace model ID in BenchmarkConfig.esm_model.
Common choices and their embedding dimensions:

    facebook/esm2_t6_8M_UR50D      — 320 dims   (fast, good for prototyping on CPU/MPS)
    facebook/esm2_t12_35M_UR50D    — 480 dims
    facebook/esm2_t30_150M_UR50D   — 640 dims
    facebook/esm2_t33_650M_UR50D   — 1280 dims  (recommended for cluster runs)
    facebook/esm2_t36_3B_UR50D     — 2560 dims

### Embedding modes
'mean'   — Mean pool of all residue hidden states (excluding cls/eos tokens).
            Captures the overall sequence context.

'delta'  — mean_pool(mutant) - mean_pool(WT).
            Captures the representation shift caused by the mutations.
            The WT embedding is computed once in fit() and reused.

### Lazy model loading
The tokenizer and model are loaded on the first call to _embed_sequences(),
not at construction time. This keeps initialization fast and allows
BenchmarkConfig validation to proceed even if transformers is slow to import.

### Disk caching
Embeddings are expensive to compute. ESMEncoder caches raw mean-pooled
embeddings as a {sha256(sequence) -> embedding} dict, serialised to a
single pickle file. The cache is shared between 'mean' and 'delta' modes
because the underlying ESM-2 vectors are identical for both.

Cache path:
    data/embeddings/<dataset>/cache_<model>.pkl

On each transform() call the encoder:
    1. Loads the cache from disk into memory (no-op on subsequent calls).
    2. Looks up each requested sequence by its hash.
    3. Embeds only the sequences that are missing from the cache.
    4. Writes the updated cache atomically (tempfile + rename) if new
       entries were added.

This means subset calls — e.g. transform(labeled_df) and transform(pool_df)
on each AL round — are served from the in-memory dict without re-running the
model. When cache_dir=None no file is written, but the dict still lives in
memory for the lifetime of the encoder instance.

### Device handling
_best_device() checks for CUDA, then MPS (Apple Silicon), then falls back
to CPU. The device can also be set explicitly via the device parameter.
On M3 MacBook: MPS is used automatically.
On Della/Tiger: CUDA is used automatically.

### Leakage safety
fit() stores only the WT sequence (for delta mode). It does not use or
store y_labeled. transform() encodes sequences only.

---

## retrieval.py

### RetrievalAugmentedEncoder
Combines PLM embeddings with nearest-neighbor label context from the
current labeled set. This is the representation-level retrieval method —
it augments the input features to the surrogate rather than modifying
the acquisition score.

### How it works
fit(df_labeled, y_labeled):
    1. Calls ESMEncoder.fit() to store WT sequence.
    2. Computes ESM embeddings for all labeled variants.
    3. Stores labeled embeddings and y_labeled (labeled fitness scores only).
    4. Builds a NearestNeighbors (kNN) index over the labeled embeddings.
    5. Computes a distance normalization scale (median inter-point distance).

transform(df):
    1. Computes ESM embeddings for all input variants (pool or labeled).
    2. For each variant, retrieves k nearest labeled neighbors using the
       kNN index built in fit().
    3. Appends 5 retrieval features per variant:
       - mean fitness of k nearest labeled neighbors
       - std  fitness of k nearest labeled neighbors
       - min  distance to any labeled neighbor (normalized)
       - mean distance to k nearest labeled neighbors (normalized)
       - max  fitness of k nearest labeled neighbors
    4. Returns concatenated [PLM_embedding, retrieval_features].

### Output dimension
PLM_dim + 5 (e.g., 320 + 5 = 325 for esm2_t6_8M_UR50D)

### Leakage safety
The kNN index is built ONLY from labeled embeddings. The y_labeled stored
in fit() contains ONLY labeled fitness scores. The pool's fitness values
are never accessed. Distance computation uses sequences only.

### Interaction with the AL loop
Because fit() rebuilds the kNN index each round, the retrieval features
automatically reflect the current state of the labeled set. As more high-
fitness variants are labeled, the retrieval signal becomes more informative.
