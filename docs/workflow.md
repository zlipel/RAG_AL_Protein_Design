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

## Current State

All original audit findings (Bugs #1–4, Gap #1, ESMEncoder performance) are resolved.
The scaffold is ready for a final merge to `main` once `fix/plm-cache-hashmap` is merged
into `audit/agent-scaffold`:

```bash
# 1. Merge the PLM cache branch
git checkout audit/agent-scaffold
git merge --no-ff fix/plm-cache-hashmap

# 2. Verify full test suite
pytest tests/ -v
ruff check src/

# 3. Merge to main
git checkout main
git merge --no-ff audit/agent-scaffold -m "merge: audited agent scaffold into main"
```

---

## Notes

- `docs/bugs.md` is the source of truth for what needed fixing and what was done.
- `docs/implementation_map.md` is the source of truth for why each module is
  structured the way it is and what the authorized label-access paths are.
- `docs/agent_log.md` has the detailed per-fix change history with test results.
