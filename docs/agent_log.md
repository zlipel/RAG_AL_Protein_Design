# Agent Log
## RAG-AL Protein Design — Running Change History

---

## 2026-06-14 — Fix Bug #1, Bug #2, Gap #1

**Branch:** `fix/reveal-result-selection-loggin`
**Commit:** `407fcf8`

### Task summary
- **Bug #1 (Critical):** `batch_y` in `runner.py` was extracted via `labeled_y[-batch_size:]` after `reveal()`. Because `labeled_y` is ordered by original dataset row index (not insertion order), this slice returned the wrong variants' fitness values silently every round, corrupting `batch_mean_fitness` metrics.
- **Bug #2 (Medium):** `runner.py` accessed `dataset._df` directly for `wt_sequence` and `batch_sequences`. `_df` contains the full DataFrame including the `fitness` column — a latent leakage risk if any future edit reads `_df["fitness"]`.
- **Gap #1 (Completeness):** No record of which specific variants were selected each round, making experiments non-auditable post-hoc.

### Files changed
| File | Change |
|------|--------|
| `src/rag_al/data/al_dataset.py` | Added `wt_sequence` property, `get_sequences()`, `get_variant_ids()`, `fitness_at()` |
| `src/rag_al/loop/runner.py` | Fixed `batch_y` extraction; replaced `_df` access with public helpers; added `selections_rows` logging |
| `src/rag_al/cli/benchmark.py` | Unpacked `(results_df, selections_df)` return; writes `seed_N_selections.csv` |
| `src/rag_al/core/logging.py` | (scaffolded for structured logging) |
| `src/rag_al/core/paths.py` | (scaffolded path helpers) |
| `src/rag_al/representations/retrieval.py` | Minor adjustments from scaffold |
| `src/rag_al/loop/metrics.py` | Minor adjustments from scaffold |
| `tests/test_runner_batch_y.py` | New: 6 tests covering Bug #1 fix and Gap #1 selection logging |
| `CLAUDE.md` | Updated architecture/bug docs |
| `docs/workflow.md` | Added workflow reference |

### Reason for change
Bug #1 silently corrupts a metric logged every round. Bug #2 is a latent leakage risk that would undermine the scientific integrity of the benchmark. Gap #1 is required to reproduce or audit any experiment.

### Tests run

```
pytest tests/ -v
```

**Result: 6/6 passed (2.21s)**

| Test | Status |
|------|--------|
| `test_batch_y_matches_global_selected` | PASSED |
| `test_selections_shape` | PASSED |
| `test_selections_global_indices_not_in_initial_labeled` | PASSED |
| `test_selections_global_indices_unique_across_rounds` | PASSED |
| `test_variant_ids_consistent_with_global_indices` | PASSED |
| `test_results_shape` | PASSED |

```
ruff check src/
```
**Result: All checks passed**

```
mypy src/
```
**Result: 28 errors — all pre-existing scaffold issues (not introduced by this fix)**

Known mypy issues (documented, not blocking):
- `import-untyped` for `pandas`, `sklearn`, `transformers` — missing stubs in environment; install `pandas-stubs` to resolve
- `union-attr` / `attr-defined` / `misc` errors in `plm.py` — scaffold's lazy `torch`/`transformers` import pattern uses `Optional` assignments that mypy cannot narrow through `if torch is not None` guards
- `index` / `union-attr` in `retrieval.py:130–134` — `_knn` and `_labeled_embeddings` declared as `Optional` but accessed without narrowing after `fit()`

### Remaining concerns
- **Bug #3** — `RetrievalAugmentedEncoder` self-neighbor inclusion when `transform(labeled_df)` called. Not blocking correctness for non-retrieval representations but distorts surrogate calibration for `plm_retrieval`. Next fix.
- **Bug #4** — Dead code in `physicochemical.py` (`* 0.0`). Minor; no correctness impact.
- **ESMEncoder cache** — Cache key mismatch means embeddings are recomputed every round. Performance issue; not correctness.

---
