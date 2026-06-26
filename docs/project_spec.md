# Project Specification

## Project Title

Retrieval-Augmented Active Learning for Sparse Protein Sequence Optimization

## Purpose

This project benchmarks how well protein language model representations and
retrieval-augmented acquisition strategies improve active learning data efficiency
across sparse protein fitness landscapes. It uses retrospective deep mutational
scanning (DMS) datasets as a controlled testbed and is designed to be extended
toward emergent biophysical properties (Phase 2) and multi-objective optimization
(Phase 3).

A secondary goal is **breadth of model coverage**: evaluating as many openly
accessible protein LMs as possible (ESM-2 at multiple scales, ProtT5, Ankh, CARP,
ESM-C, and eventually closed-model APIs) both to generate rigorous comparative
results and to serve as a concrete artifact for industry outreach.

## Scientific Question

In sparse protein fitness landscapes, do foundation-model embeddings and
retrieval-augmented acquisition improve active learning data efficiency relative to
standard sequence descriptors and standard acquisition functions? And to what extent
does the answer depend on model scale, architecture, and the structure of the
landscape?

## Task Definition

Given a DMS dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ where $x_i$ is a
protein variant and $y_i = f(x_i)$ is its measured fitness, we simulate pool-based
active learning:

1. Hide all but $n_{\text{init}}$ labels.
2. Each round: fit surrogate on labeled set → score unlabeled pool → acquire a batch
   → reveal labels → update.
3. Repeat for $T$ rounds.

No de novo generation; candidate selection is from a fixed pool of experimentally
measured variants.

---

## Experimental Axes

### Representations

| Tier | Name | Dims | Notes |
|------|------|------|-------|
| Baseline | Mutation descriptors | 49 | Position, AA change, Δphysicochemical |
| Baseline | Physicochemical | 29 | Composition, charge, hydropathy, entropy |
| PLM | ESM-2 8M | 320 | HuggingFace; local / CPU |
| PLM | ESM-2 150M | 640 | Scale sweep |
| PLM | ESM-2 650M | 1280 | Cluster default |
| PLM | ProtT5-XL | 1024 | `Rostlab/prot_t5_xl_uniref50`; T5Encoder |
| PLM | Ankh-base / large | 768/1536 | `ElnaggarLab/ankh-base`; longer context |
| PLM | CARP | varies | Microsoft `sequence-models` CNN; no MSA |
| PLM | ESM-C / ESM3-open | varies | EvolutionaryScale; gated license |
| Concat | PLM + physicochemical | D+29 | Concatenation of frozen PLM + physico |
| Retrieval | PLM + retrieval context | D+5 | kNN label features appended to PLM mean |

ESM-2 size sweep (8M / 150M / 650M) is config-only — no code change.
ProtT5 requires space-separated uppercase AA preprocessing.
CARP requires the `sequence-models` pip package (separate install).
ESM-C requires the EvolutionaryScale `esm` package (gated license — good outreach hook).

### Surrogates

| Name | Uncertainty | Scales to | Notes |
|------|-------------|-----------|-------|
| Random Forest (current) | Ensemble σ | ~10K labeled | Fast; baseline |
| Gaussian Process | Posterior σ | ~2K exact; larger with approx | Use gpytorch; approximate inference preferred (sparse GP / SGPR / inducing points); avoid exact matrix inversion for large n_labeled |
| NN ensemble | Ensemble σ | Any | Optional Optuna per-round architecture search; user has existing code |

GP implementation should use approximate inference (e.g., `gpytorch.models.ApproximateGP`
with inducing points) rather than exact MLL inversion to scale to n_labeled > 1K.

### Acquisition Functions

| Name | Score |
|------|-------|
| Random | Uniform |
| Greedy | $\mu(x)$ |
| UCB | $\mu(x) + \beta \sigma(x)$ |
| Diversity-UCB | UCB − γ · diversity penalty |
| Retrieval-UCB | UCB + λ · R(x); R from labeled kNN |
| Expected Improvement *(planned)* | $\mathbb{E}[\max(f(x)-f_{\text{best}}, 0)]$ |

### Datasets (Sprint 1 — ProteinGym)

| Dataset | N | Protein | Notes |
|---------|---|---------|-------|
| BLAT_ECOLX_Jacquier_2013 | 989 | β-lactamase | Small, ordinal |
| BLAT_ECOLX_Deng_2012 | 4996 | β-lactamase | Continuous, PLM wins strongly |
| BLAT_ECOLX_Firnberg_2014 | 4783 | β-lactamase | Deceptive outlier (F58N); adversarial for model-based acq |
| BLAT_ECOLX_Stiffler_2015 | 4996 | β-lactamase | Similar to Deng |
| PABP_YEAST_Melamed_2013 | 37708 | RNA-binding | Large; PLM TBD |
| BRCA1_HUMAN_Findlay_2018 | 1837 | Tumor suppressor | WT len 1863 > ESM-2 limit; no PLM |

### Datasets (Phase 2 — Biophysical / Proprietary)

- Lindorff-Larsen IDRome (public; Rg, diffusivity from CALVADOS)
- User's simulation data: ~600 seq × diffusivity/density/expenditure density + 2034-seq Rg/B2 set

---

## Metrics

| Metric | Description |
|--------|-------------|
| `best_fitness` | $\max_{x \in L_t} f(x)$ — best found vs. round |
| `simple_regret` | $f(x^*) - \max_{x \in L_t} f(x)$ |
| `topk_recall` | $|L_t \cap \text{Top-}K| / K$ at k=10%, 50% |
| `batch_mean_fitness` | Mean fitness of the current acquisition batch |
| `batch_diversity` | Mean pairwise Hamming in batch |
| `pool_spearman` *(planned)* | Spearman ρ between surrogate $\mu$ and true pool fitness — measures surrogate calibration each round; oracle metric, not used for acquisition |

---

## Leakage Rules

These rules are invariants enforced throughout the codebase. See `docs/bugs.md`
for the audit history.

1. Surrogate trains only on the currently labeled set.
2. Acquisition functions receive only $(\mu, \sigma, X_{\text{labeled}}, y_{\text{labeled}})$ — no pool fitness.
3. Retrieval features use labeled neighbors only.
4. Embedding computation ignores fitness labels.
5. `reveal()` is the only authorized hidden-label exposure path during the loop.
6. `global_optimum` and `top_k_global_indices()` are metric-only oracle quantities.
7. Feature normalization fit on labeled set only.

---

## Sprint Roadmap

### Sprint 1 (complete)
- Core AL loop, leakage discipline, schema validation
- 5 representations: mutation, physicochemical, plm_mean, plm_delta, plm_retrieval
- 5 acquisitions: random, greedy, UCB, diversity-UCB, retrieval-UCB
- RF surrogate; ESM-2 hash-based embedding cache
- 6-dataset ProteinGym benchmark on cluster (ESM-2 650M)
- Cluster deployment: SLURM scripts, GNU parallel, environment.yml

### Sprint 2 (next)
- `pool_spearman` per-round metric (runner + metrics)
- `PLMPhysicoEncoder` — pLM + physicochemical concat representation
- `GPSurrogate` — approximate GP via gpytorch (sparse/inducing-point preferred)
- `HFPLMEncoder` — generalized HuggingFace backend (ProtT5, Ankh)
- `CARPEncoder` — Microsoft sequence-models CNN adapter
- ESM-2 size sweep cluster runs (8M / 150M / 650M comparison)

### Phase 2 (mid-July → mid-Aug)
- Ingest simulation data + IDRome into `ALDataset` (multi-property columns)
- Multi-objective acquisition (hypervolume, MOBO)
- CV learning-curve harness for small datasets
- ALBATROSS-style BiLSTM baseline

### Phase 3 (Aug → Sep)
- pLM + IDP physics descriptor concat (κ, SCD, SHD via localCIDER/SPARROW)
- pLM + structure features (AlphaFold, folded ProteinGym only — not IDPs)
- pLM + text/function embeddings (UniProt annotations; stretch)
- LoRA fine-tuning (conditional on Phase 2 showing headroom)
