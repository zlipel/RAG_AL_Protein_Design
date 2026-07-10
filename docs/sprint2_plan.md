# Sprint 2 Plan — Representations, Surrogate, and Accuracy Metric

> **How to use this file:** Edit any section or add comments inline. When done,
> tell Claude to read it and we will implement according to your feedback.

---

## Context

Sprint 1 is complete. ESM-2 650M results are in across all 6 datasets. Key findings
from the aggregate heatmap (topk10_recall, averaged over seeds × acquisitions):

- PLM + retrieval + UCB: **0.920** (best overall)
- PLM delta/mean + UCB: 0.84–0.89
- Physicochemical: 0.64–0.70 (weakest non-random)
- **PABP anomaly**: PLM (0.35–0.41) underperforms mutation (0.51) — flat landscape
  where RF uncertainty is miscalibrated. Primary motivation for GP surrogate.
- BLAT Deng: largest PLM gain, physico 0.42 → PLM 0.89
- BRCA1 and Jacquier: saturate ~1.0 for all reps (too easy)
- Firnberg deceptive outlier: PLM+retrieval 0.93 best, but random also competitive

Planned work (all slots into existing `AbstractEncoder`/`AbstractSurrogate` interfaces):

1. Per-round Spearman ρ — surrogate ranking accuracy metric
2. Enhanced PLM representations — site extraction, per-residue physico fusion
3. GP surrogate — proper Bayesian UQ, motivated by PABP anomaly
4. Additional PLM backends — ProtT5, Ankh, Profluent E1
5. ESM-2 size sweep — config-only once backends work
6. Minimum data size study — when do LM embeddings start winning?

---

## Design Notes

### GP Surrogate — Protocol Discussion

**Existing code:**
- `ActiveLearningAndIDPs/models/gpr_model.py` — Matérn 3/2, `ConstantMean`, `ScaleKernel`
- `ActiveLearningAndIDPs/PROJECTS/al_active_dev/al_pipeline/training/kfold_training.py` — k-fold protocol
- `ActiveLearningAndIDPs/PROJECTS/al_active_dev/al_pipeline/training/trainers.py` — GPRTrainer (patience)

**What the existing k-fold protocol does well:**
- Averages hyperparameters ({lengthscale, outputscale, noise_var}) across folds — regularizes
  against any single fold's noise
- Warm-starts the full-data retrain from averaged hypers before running without early stopping
- `fast_pred_var` in evaluation (CG-based, avoids O(n²) covariance)

**Issue for per-round AL:**
k-fold is O(k × epochs × rounds). With k=5, 500 epochs, 20 rounds → 50K epoch-equivalents
per experiment. More fundamentally: in AL the labeled set grows by one batch each round, so
GP hypers barely change from round k to k+1. k-fold hyperparameter search is redundant for
all rounds after the first.

**Recommended protocol: round-to-round warm start**

Instead of k-fold each round, carry the final hyperparameters forward:
- Round 0 (cold start): 200 MLL steps from default initialization
- Round k+1: load round k's {model, likelihood} state dict → run up to 200 more steps
  with MLL-patience early stopping (check every 20 steps; stop if MLL improvement < 1e-4
  for 3 consecutive checks)

Cost: O(rounds × ~50–100 steps) vs O(rounds × k × 500). In practice, warm starts
converge in ~60 steps since the hypers are already near-optimal.

k-fold remains valuable as a **one-time offline diagnostic** (not per-round).

**On more epochs + patience:**
Yes, patience-based stopping works well here — use MLL (on train data) as the loss, not
val-loss, so no holdout set is needed. MLL is bounded above and converges smoothly near
the optimum. Up to 1k max steps is fine; in practice you'll stop much earlier when warm-starting.

#### Proposed `GPSurrogate` skeleton

```python
# src/rag_al/surrogates/gp.py
# _GPRegressionModel copied from ActiveLearningAndIDPs/models/gpr_model.py (Matérn 3/2)

import numpy as np, torch, gpytorch
from .base import AbstractSurrogate

class GPSurrogate(AbstractSurrogate):
    def __init__(
        self,
        n_iter: int = 200,        # max Adam steps (warm start → exits early in practice)
        lr: float = 0.01,
        patience_steps: int = 3,  # patience checks (each = 20 steps → 60 min steps)
        tol: float = 1e-4,        # MLL improvement threshold per check
        device: str | None = None,
    ):
        ...
        self._prev_state: dict | None = None   # warm start across rounds

    def fit(self, X, y):
        # 1. Standardize X (per-dim) and y (global)
        # 2. Build _GPRegressionModel + GaussianLikelihood
        # 3. If self._prev_state is not None: load_state_dict (warm start)
        # 4. Adam loop with MLL patience (no val split needed)
        # 5. Save state_dicts to self._prev_state for next round

    def predict(self, X):
        # Apply same standardization, run fast_pred_var, un-standardize
        # Return (mu, sigma) as numpy arrays
```

**Key differences from existing trainer:**
- No k-fold, no val split — train on all labeled data each round
- Round-to-round warm start replaces k-fold averaging
- MLL patience replaces val-loss patience
- 200 max steps → ~60 in practice when warm-starting

---

### PLM Backends

| Backend | Org | Priority | HF model ID | Preprocessing |
|---------|-----|----------|-------------|---------------|
| ESM-2 | Meta | **Done** | `facebook/esm2_t33_650M_UR50D` | As-is |
| ProtT5 | Rostlab | High | `Rostlab/prot_t5_xl_uniref50` (~3 GB) | Space-separate AAs; B/Z/U/O→X; `T5EncoderModel` |
| Ankh | ElnaggarLab | High | `ElnaggarLab/ankh-base` (~450 MB) | List of char lists; `is_split_into_words=True` |
| **Profluent E1** | **Profluent** | **High** | **`Profluent-Bio/E1-600m`** | **As-is — ESM-like; standard `AutoModel`** |
| ESM-3 | EvolutionaryScale | Medium | `esm3-sm-open-v1` | `esm` pip package — defer (different API) |
| ProGen2 | Salesforce | Low | `lhallee/progen2-base` | Causal LM — mean pooling less principled |

**Profluent E1:** No special preprocessing — single-letter AA codes, `AutoTokenizer` +
`AutoModel`. Slots into the default branch of `HFPLMEncoder` without model-specific logic.
Strong outreach choice: Profluent is an active biodesign company.

**Preprocessing sketches:**

```python
def _preprocess(self, sequences):
    if "prot_t5" in self.model_name.lower():
        # Space-separate, uppercase, replace ambiguous AAs
        cleaned = [re.sub(r"[BZUO]", "X", s.upper()) for s in sequences]
        return [" ".join(list(s)) for s in cleaned]
    elif "ankh" in self.model_name.lower():
        # Ankh tokenizer expects list of char lists
        return [list(s) for s in sequences]
    else:
        # ESM-2, Profluent E1, and most others: as-is
        return list(sequences)

def _get_model_and_tokenizer(self):
    if "prot_t5" in self.model_name.lower():
        from transformers import T5EncoderModel, T5Tokenizer
        return T5Tokenizer.from_pretrained(..., do_lower_case=False), T5EncoderModel.from_pretrained(...)
    else:
        from transformers import AutoTokenizer, AutoModel
        return AutoTokenizer.from_pretrained(...), AutoModel.from_pretrained(...)
```

**Ankh tokenizer call requires `is_split_into_words=True`:**
```python
inputs = tokenizer(char_lists, is_split_into_words=True, return_tensors="pt",
                   padding=True, truncation=True)
```

**Max-length guards** (same pattern as `_ESM_MAX_RESIDUES = 1022` in `plm.py`):
```python
_PLM_MAX_RESIDUES = {"prot_t5": 1022, "ankh": 2046}
```
BRCA1 (1863 residues): ProtT5 will fail; Ankh and E1 would fit.

---

### Representation Combination

The simple post-hoc concat `[plm_mean | physico_29]` discards positional information.
Three options, ordered by effort:

#### 1. `plm_site` mode — mutation-site extraction (easiest, most targeted)

Extract ESM hidden state only at the mutated position(s) rather than mean-pooling all
residues. For `mutant="A23V"` → position 22; for `"A23V:G45L"` → average positions 22, 44.

```python
def _parse_mutant_positions(mutant_str):
    """'A23V:G45L' → [22, 44]"""
    return [int(re.search(r"\d+", m).group()) - 1 for m in mutant_str.split(":")]
```

For single-point mutant datasets (all BLAT variants), this is well-motivated — the
mutation site carries essentially all the fitness signal. Output shape: same (N, D).

#### 2. Per-residue physico concat → pool (more principled than post-hoc)

At each token position, append a per-residue physicochemical vector `p_i` (5 properties:
hydropathy, charge, volume, disorder propensity, pI) to ESM hidden state `h_i`, then
mean-pool `[h_i | p_i]`. This encodes "which physico traits appear where" — positional
co-occurrence that post-hoc concat loses. Output dim: D_esm + 5.

`p_i` is a lookup table (20 AAs × 5 properties) — no learned parameters.

#### 3. Simple post-hoc concat baseline

`np.hstack([plm_mean, physico_29])`. Output dim: D_esm + 29. Needed as ablation baseline
to isolate the value of the per-residue approach.

#### Future (Sprint 3+)

- **Physicochemical-weighted pooling**: weight each ESM position by
  `|physico_mutant_i − physico_wt_i|` before pooling. Zero extra parameters.
- **Reduced-vocabulary IDP LM**: cluster AAs into 5–8 macro-groups (hydrophobic /
  charged+/− / polar / special), pretrain a small transformer on DisProt + CALVADOS.
  The PABP anomaly (PLM < mutation) suggests ESM-2 misses IDP-specific grammar.
- **LoRA per-round fine-tuning**: rank 4–8 LoRA on labeled data each round. Risk:
  overfitting at low n_labeled. Investigate after GP validates.

---

## Implementation Steps

### Step 1 — Per-round Spearman ρ metric

Smallest change, highest diagnostic value. Measures how well the surrogate ranks
the hidden pool — currently invisible.

**`src/rag_al/loop/runner.py`** — after `surrogate.predict(X_pool)`, before acquisition:

```python
from scipy.stats import spearmanr
# metric-only oracle read — same category as topk_recall / global_optimum
pool_fitness_oracle = dataset.fitness_at(pool_indices)
rho = float(spearmanr(mu, pool_fitness_oracle).statistic)
```

Pass `pool_spearman=rho` to `compute_round_metrics()`.

**`src/rag_al/loop/metrics.py`** — add `pool_spearman: float` to `compute_round_metrics()`
signature and include it in the returned dict.

**Leakage note:** `dataset.fitness_at(pool_indices)` reads hidden labels for scoring only
(not model fitting). Same oracle category as `global_optimum` / `top_k_global_indices()`.

**Test:** assert `pool_spearman` column exists in `results_df` and values ∈ [-1, 1].

---

### Step 2 — Enhanced PLM representations

**2a. `plm_site` mode** — add to `src/rag_al/representations/plm.py`:
- New `mode="site"` in `ESMEncoder.__init__`
- `_parse_mutant_positions(mutant_str)` helper (parses `mutant` column)
- `transform()` branches: extract site hidden states instead of mean-pooling
- Cache: full hidden states stored per sequence hash; site extraction is post-cache
- CLI: `--representation plm_site`

**2b. Per-residue physico fusion** — new `src/rag_al/representations/plm_physico.py`:
- `PLMPhysicoEncoder` wraps `ESMEncoder`, overrides the pooling step to concat `p_i`
- AA lookup table (20×5): hydropathy (Kyte-Doolittle), charge, volume,
  disorder propensity, pI
- `n_features = esm_hidden_size + 5`
- CLI: `--representation plm_physico`

**2c. Simple concat baseline** — in same file:
- `PLMSimpleConcatEncoder`: `np.hstack([plm_mean, physico_29])`
- `n_features = esm_hidden_size + 29`
- CLI: `--representation plm_concat`

---

### Step 3 — GP surrogate

New file `src/rag_al/surrogates/gp.py`. Full skeleton in design notes above.

Config additions (`src/rag_al/core/config.py`):
```python
surrogate: str = "rf"          # "rf" | "gp"
gp_n_iter: int = 200
gp_lr: float = 0.01
gp_patience: int = 3
```

Dispatch in `src/rag_al/cli/benchmark.py` on `cfg.surrogate`.

**Scaling:** Exact GP fit is O(n³). At n_labeled=2610 max (50 init + 20×128), 100 warm
steps takes ~5–15s on CPU — acceptable on cluster. Pool prediction uses CG (fast). Upgrade
path if bottlenecked: `gpytorch.models.ApproximateGP` with inducing points.

---

### Step 4 — Additional PLM backends (ProtT5, Ankh, Profluent E1)

New file `src/rag_al/representations/hf_plm.py`. `HFPLMEncoder` generalizes `ESMEncoder`
for any HF protein LM. Cache block (~40 lines) copied from `ESMEncoder`.

CLI registration:
- `"prot_t5"` → `HFPLMEncoder("Rostlab/prot_t5_xl_uniref50")`
- `"ankh"` → `HFPLMEncoder("ElnaggarLab/ankh-base")`
- `"profluent_e1"` → `HFPLMEncoder("Profluent-Bio/E1-600m")`

ESM-3: deferred — requires separate `esm` pip package with a different API.

---

### Step 5 — ESM-2 size sweep (config-only)

No code changes. Use existing `--esm_model` flag. Cache is keyed by sanitized model name,
so 8M / 150M / 650M produce separate cache files automatically.

---

### Step 6 — Minimum data size study (original motivation)

**Core question:** At what `n_labeled` do LM embeddings start outperforming simpler
representations? What is the crossover point, and is it consistent across datasets?

#### Part A — Re-analyze existing results (no new experiments needed)

Each `seed_N.csv` has one row per round with `n_labeled`. Plot `topk10_recall` (and
`pool_spearman` from Step 1) vs `n_labeled` for each representation, per dataset. Mark
the crossover round where PLM first beats mutation.

**New script: `scripts/plot_learning_curves.py`**
- Input: `results/<dataset>/<repr>_<acq>/seed_*.csv`
- Per-dataset: line plot per representation vs `n_labeled`
- Annotate crossover point with `n_labeled` value
- Summary table: crossover n for each dataset

Expected findings:
- BLAT datasets: PLM likely overtakes mutation early (rounds 1–3, n_labeled ~178–434)
- PABP: PLM may never cross over — confirms anomaly is not a data-size issue
- BRCA1: can't test (no PLM runs)

#### Part B — Low n_init sweep (gated on Part A)

**Gate:** Only run Part B if Part A shows the crossover at or before round 0 (i.e., PLM
already wins at n_labeled = n_init = 50). If the crossover is visible within the existing
round data, Part A is sufficient and Part B is unnecessary.

**If gated open:** Run `n_init ∈ {25, 50}` with `batch_size=25`, `n_rounds=30`.
n_init=50 overlaps with Sprint 1 as a calibration anchor; n_init=25 is the lower tier.
Avoid n_init=10 unless n_init=25 also shows PLM winning from round 0 — at 10 labeled
points the RF surrogate is near-random and results are dominated by initialization
noise. If n_init=10 is needed, use 10–15 seeds rather than 5 to average out the noise.

New script: `scripts/run_small_init.sh`. Focus on BLAT_Deng (largest PLM gain) and
PABP (anomaly). Datasets chosen to bracket the two extremes seen in Sprint 1.

**Key sub-question:** In the low-data regime, do augmented representations (`plm_physico`,
`plm_site`) win even when plain PLM doesn't?

---

## Branch and Merge Order

```
feature/sprint2-repr-surrogate → audit/agent-scaffold → main
```

One commit per step. Merge to `audit/agent-scaffold` after each step passes
`pytest tests/ -v` and `ruff check src/`.

---

## Feasibility

| Item | Effort | Risk |
|------|--------|------|
| Per-round Spearman | S (30 min) | Low |
| `plm_site` mode | S (1 hr) | Low |
| Per-residue physico fusion | M (2 hr) | Low-medium |
| Simple concat baseline | XS (30 min) | Low |
| GP surrogate (warm-start + MLL patience) | M (3–4 hr) | Medium |
| ProtT5 backend | M (2 hr) | Low-medium — T5EncoderModel quirks |
| Ankh backend | M (1.5 hr) | Low-medium — is_split_into_words |
| Profluent E1 backend | S (30 min) | Low — AutoModel, no preprocessing |
| ESM-2 size sweep | XS (config) | Low |
| Step 6a — learning curve plots | S (1 hr) | Low |
| Step 6b — low n_init sweep | XS (config) | Low |

**Deferred:**
- ESM-3: different pip package (`esm`), separate encoder class, defer
- LoRA fine-tuning: Sprint 3+
- Reduced-vocabulary IDP LM: Sprint 3+ research project

---

## Verification

```bash
# After each step
pytest tests/ -v
ruff check src/

# Smoke-test representations
rag-benchmark --dataset BLAT_ECOLX_Jacquier_2013 \
              --representation plm_site --acquisition ucb \
              --seed 0 --n_rounds 3 --batch_size 10

rag-benchmark --dataset BLAT_ECOLX_Jacquier_2013 \
              --representation plm_physico --acquisition ucb \
              --seed 0 --n_rounds 3 --batch_size 10

# Smoke-test GP surrogate
rag-benchmark --dataset BLAT_ECOLX_Jacquier_2013 \
              --representation mutation --acquisition ucb \
              --surrogate gp --seed 0 --n_rounds 3 --batch_size 10

# Verify pool_spearman column
python -c "
import pandas as pd
df = pd.read_csv('results/BLAT_ECOLX_Jacquier_2013/mutation_ucb/seed_0.csv')
print(df[['round', 'n_labeled', 'pool_spearman']].head())
"
```
