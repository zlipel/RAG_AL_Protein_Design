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

## Current State (as of 2026-07-19)

**Sprint 1 + 2 + hardening complete; 650M RF sweep done, analyzed, and written up.**

All original audit findings (Bugs #1–4, Gap #1, ESMEncoder performance) resolved.
Sprint 2 delivered `plm_site`/`plm_physico`/`plm_concat`, `pool_spearman`, and
`GPSurrogate`. RF 650M sweep is complete and documented in `docs/sprint1_results.md`,
`docs/sprint2_results.md`, `docs/figures/`.

Latest work (merged to `main`, pushed):
- Analysis scripts made surrogate/representation-aware: `plot_aggregate.py` /
  `plot_results.py` now know all 8 reps and take `--surrogate rf|gp|all`
  (`plot_aggregate` also `--no_plots` table mode + `--datasets` subset filter).
- **GP benchmark deploy fixed** (`fix/gp-benchmark-deploy`): the old shared-cgroup
  GNU-parallel design OOM'd (`sacct OUT_OF_MEMORY`); now `--exclusive` node + per-cell
  `srun --exclusive --mem=8G` steps.
- GB1 RF rerun completed → 6/8 datasets full 8-rep grid; GFP still 5-rep.

`ruff check src/` clean; `src/` untouched this session (test suite unchanged).

**Active branches:** `main` (`171526e`) / `audit/agent-scaffold` (`be5febc`) — in sync,
all above merged and pushed.

**Cluster / results status:**
- ✅ RF 650M benchmark complete + analyzed. 6/8 datasets full grid (GFP 5-rep pending).
- ⏳ GP-only benchmark (`submit_gp_benchmark.sh`, 36 cells) — deploy fixed, **being
  submitted now**; results land in `_gp/` dirs.

**Current phase — Sprint 3:**
1. **Run + analyze the GP grid** (immediate): once `_gp/` results sync, run
   `plot_aggregate.py --surrogate all --no_plots` — does GP fix PABP's top-of-landscape
   failure? (Plotting is already surrogate-aware; no code change needed.)
2. Full **GFP re-run** (5-rep → 8-rep grid).
3. `plot_learning_curves.py` — n_labeled crossover (reads existing results).
4. Then: `HFPLMEncoder` (Profluent E1, Ankh, ProtT5); low n_init sweep; ESM-2 size sweep.

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
