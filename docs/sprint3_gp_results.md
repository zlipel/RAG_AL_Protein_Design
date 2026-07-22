# Sprint 3 Results — GP vs RF Surrogate (targeted grid)

**Scope.** The targeted GP grid: **2 datasets** (PABP, BLAT_Deng) × **3 reprs**
(mutation, plm_mean, plm_physico) × **2 acqs** (greedy, ucb) × 3 seeds = **36 GP
cells**, all completed (20 rounds each; no OOM after the predict-chunking fix).
The RF baseline for the same cells comes from the main 650M sweep. This asks the
question Sprint 2 raised: **does a better-calibrated surrogate (exact GP) fix the
PABP top-of-landscape failure?**

All numbers below are **final-round, mean over seeds, matched on the acquisitions
both surrogates ran** ({greedy, ucb}) — the GP grid omits random/diversity/retrieval,
so a naive mean over acquisitions would compare GP's 2 acqs against RF's 5. Figures:
`docs/figures/gp_vs_rf/` (regenerate with `scripts/plot_gp_vs_rf.py`).

## 1. Headline: GP helps PLM find the peak, hurts hand-crafted features

| dataset | repr | metric | RF | GP | Δ(gp−rf) |
|---|---|---|---|---|---|
| PABP | plm_mean | simple_regret | 0.279 | **0.040** | **−0.239** |
| PABP | plm_physico | simple_regret | 0.188 | **0.000** | **−0.188** |
| PABP | plm_mean | pool_spearman | 0.533 | **0.668** | +0.135 |
| PABP | plm_physico | pool_spearman | 0.550 | **0.634** | +0.084 |
| PABP | plm_mean | best_fitness | 2.590 | **2.829** | +0.239 |
| PABP | plm_physico | best_fitness | 2.681 | **2.869** | +0.188 |
| PABP | **mutation** | simple_regret | **0.000** | **0.861** | **+0.861** |
| PABP | **mutation** | topk10_recall | **0.600** | **0.067** | **−0.533** |

Two opposite effects, both large:

- **GP + PLM on PABP is a real win where the RF was miscalibrated.** Both PLM reps
  reach `simple_regret ≈ 0` under GP (plm_physico exactly 0 — finds the global
  optimum every matched cell), vs RF's 0.19–0.28. `best_fitness` rises ~0.2 and
  `pool_spearman` rises 0.08–0.14. This **validates the Sprint-2 hypothesis**: the
  PABP miss was partly RF σ miscalibration on the flat landscape, and a latent-
  posterior GP fixes the *peak-finding* half of it.
- **GP + `mutation` on PABP collapses.** `simple_regret` 0.00 → 0.86, `topk10`
  0.60 → 0.07. It is **consistent, not a fluke**: 5 of 6 GP-mutation PABP cells
  never approach the optimum (regret ~1.0). This is the numerical failure behind
  the `NumericalWarning: Negative variance` / `CG terminated … residual > tol`
  messages in the run log — an exact Matérn GP is **ill-conditioned on the sparse,
  quasi-discrete 49-dim hand-crafted descriptors**. `pool_spearman` confirms it
  (GP 0.24 vs RF 0.43): the GP mutation surrogate ranks *worse*, so acquisition
  walks off. The same degradation shows on BLAT_Deng mutation (topk10 0.43 vs 0.65,
  ρ 0.08 vs 0.34).

## 2. topk10_recall — GP does NOT recover the top-*set*

| PABP | mutation | plm_mean | plm_physico |
|---|---|---|---|
| RF topk10 | 0.600 | 0.400 | 0.433 |
| GP topk10 | 0.067 | 0.400 | 0.417 |

For PLM, `topk10_recall` is **unchanged** by GP (~0.40) even though regret goes to
0 and best_fitness rises. **GP finds *the* best variant but not the whole top-10
set.** That gap is a **representation-resolution limit** (the embedding can't
finely separate the very top variants), not a surrogate-calibration one — so a
better surrogate can't close it. This sharpens the Sprint-2 reading: the PABP
"anomaly" is two distinct failures, and GP fixes only the calibration one.

Note this *inverts* the surface-level PABP anomaly (Sprint 1/2: mutation topk10 >
PLM). Under GP, PLM (0.40) ≫ mutation (0.07) — but because GP **wrecks the mutation
surrogate**, not because PLM improved. It is a regime change, not a clean fix.

## 3. BLAT_Deng — saturated, GP neither helps nor hurts (on PLM)

RF already solves BLAT_Deng with PLM (topk10 ≈ 1.0, regret 0). GP matches it
(topk10 0.95/0.93, regret 0.00) — marginally noisier, no room to improve.
`pool_spearman` is a tie (GP ≈ RF, ~0.31). So Deng is a **negative control**: it
confirms GP doesn't *break* PLM on an easy landscape, and isolates the PABP PLM
gains as real rather than seed noise.

## Takeaways

1. **GP is a representation-conditional surrogate, not a universal upgrade.** On
   smooth PLM embeddings it improves peak-finding and pool ranking (PABP: regret
   → 0, ρ +0.1); on hand-crafted mutation descriptors it is *ill-conditioned and
   actively harmful*. Pair the GP with PLM features, not hand-crafted ones.
2. **The PABP anomaly is two failures.** GP fixes the calibration half (finds the
   optimum) but not the top-*set* recall half (representation resolution). A
   better surrogate is necessary but not sufficient for top-k recall.
3. **Actionable:** for a GP path, drop `mutation`/`physicochemical` from the grid
   (or add a jitter/PCA/whitening pre-step to condition them); report GP results
   on PLM reps. The negative-variance/CG warnings are a reliable red flag for the
   ill-conditioned cells and could be surfaced as a fit-quality guard.

## Reproduce

```bash
# numeric tables (RF vs GP, matched acqs)
python scripts/plot_aggregate.py --surrogate all --no_plots \
  --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012 --metric topk10_recall
# paired GP-vs-RF figures + delta tables
python scripts/plot_gp_vs_rf.py \
  --datasets PABP_YEAST_Melamed_2013 BLAT_ECOLX_Deng_2012 \
  --metrics topk10_recall simple_regret pool_spearman best_fitness
# GP-only learning curves
python scripts/plot_results.py --dataset PABP_YEAST_Melamed_2013 --surrogate gp \
  --output_dir docs/figures/PABP_YEAST_Melamed_2013
```
