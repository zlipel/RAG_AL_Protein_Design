# Development Workflow

## Branch Hierarchy

```
main
  └── audit/agent-scaffold
        └── fix/<description>
```

| Branch | Purpose |
|--------|---------|
| `main` | Final, clean, production-ready code. Receives only fully-audited merges from `audit/agent-scaffold`. |
| `audit/agent-scaffold` | Integration branch for all audit fixes. The running target during the audit phase. Never push directly to this branch during active fix work. |
| `fix/<description>` | One branch per bug, gap, or tightly coupled set of fixes. Cut from `audit/agent-scaffold`, merged back when the fix is reviewed and satisfactory. |

---

## Per-Fix Process

1. **Cut** the fix branch from `audit/agent-scaffold`:
   ```bash
   git checkout audit/agent-scaffold
   git checkout -b fix/<description>
   ```

2. **Implement** the fix. Keep commits focused — one logical change per commit.
   Commit message format:
   ```
   fix(<module>): short description

   Longer explanation if needed. Reference the bug number from docs/bugs.md.
   ```
   Example: `fix(runner): correct batch_y extraction after reveal (Bug #1)`

3. **Review** — look at `docs/bugs.md` and `docs/implementation_map.md` to confirm
   the fix addresses the documented problem. Run any relevant tests.

4. **Merge** back to `audit/agent-scaffold`:
   ```bash
   git checkout audit/agent-scaffold
   git merge --no-ff fix/<description>
   ```
   The `--no-ff` flag preserves the branch history as a labeled merge commit.

5. **Move on** — cut the next fix branch from the newly updated `audit/agent-scaffold`.

---

## Fix Queue

All documented findings have been addressed.

| Priority | Branch | Covers | Status |
|----------|--------|--------|--------|
| 1 | `fix/reveal-result-selection-loggin` | Bug #1 (wrong `batch_y`) + Gap #1 (selection logging) | ✅ Merged — `audit/agent-scaffold` |
| 2 | `fix/dataset-private-access` | Bug #2 (`dataset._df` accessed directly from runner) | ✅ Merged — `audit/agent-scaffold` |
| 3 | `fix/retrieval-self-neighbor` | Bug #3 (self-label inclusion in `RetrievalAugmentedEncoder`) | ✅ Merged — `audit/agent-scaffold` |
| 4 | `fix/dead-code-physicochemical` | Bug #4 (dead code in `physicochemical.py`) | ✅ Merged — `audit/agent-scaffold` |
| 5 | `fix/plm-cache-hashmap` | Performance: ESMEncoder cache (hashmap + atomic save) | ✅ Complete, pending merge |

> Note: `fix/reveal-result-selection-loggin` has a deliberate typo in the branch name.
> `fix/dataset-private-access` was folded into the same commit as Bug #1/Gap #1.

---

## Current State (as of 2026-07-15)

**Sprint 1 + 2 complete; embed/benchmark hardening complete; 650M RF sweep done.**

All original audit findings (Bugs #1–4, Gap #1, ESMEncoder performance) resolved.
Sprint 2 delivered: `plm_site`, `plm_physico`, `plm_concat` representations;
`pool_spearman` metric; `GPSurrogate` (ExactGP, warm-start, MLL patience).

Post-Sprint-2 hardening (merged to `main` this week):
- `rag-embed` precomputes all four PLM caches (mean/delta/site/physico); GB1
  submitted separately at a longer wall time (`feature/embed-plm-modes`, `feature/gb1-embed-walltime`).
- Result paths namespaced by surrogate (`_gp` suffix) + `surrogate` CSV column;
  length-based PLM exclusion replaces the hardcoded BRCA1 check
  (`feature/benchmark-safety-guards`). `results_*/` gitignored.

75/75 tests passing. `ruff check src/` clean.

**Active branches:**
- `main` / `audit/agent-scaffold` — in sync, all work above merged and pushed.

**Cluster / results status:**
- ✅ Full RF 650M benchmark complete; synced to `results/`. Old 8M runs archived
  in `results_sprint1_8M/`.
- ⏳ GP-only benchmark (`submit_gp_benchmark.sh`, PABP + BLAT_Deng) — to run.

**Current phase — Sprint 3 analysis:**
1. Analyze the synced 650M results: `plot_results.py` per dataset; cross-dataset
   heatmaps; confirm/refine Sprint 1 findings (PABP anomaly, BLAT_Deng PLM gain).
2. `plot_learning_curves.py` — crossover analysis (x-axis n_labeled: when does PLM
   beat mutation?). Reads existing results; no new cluster runs.
3. Run + analyze the GP benchmark; add `surrogate` to plot grouping.
4. Then: `HFPLMEncoder` (ProtT5, Ankh, Profluent E1); low n_init sweep; ESM-2 size sweep.

Start new feature work from `audit/agent-scaffold`:
```bash
git checkout audit/agent-scaffold
git checkout -b feature/<name>
```

---

## Notes

- `docs/bugs.md` is the source of truth for what needed fixing and what was done.
- `docs/implementation_map.md` is the source of truth for why each module is
  structured the way it is and what the authorized label-access paths are.
- `docs/agent_log.md` has the detailed per-fix change history with test results.
