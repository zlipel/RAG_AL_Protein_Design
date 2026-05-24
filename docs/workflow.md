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

## Fix Queue (from docs/bugs.md)

Work through these in priority order. Each gets its own fix branch.

| Priority | Branch name | Covers |
|----------|-------------|--------|
| 1 | `fix/reveal-result-selection-loggin` | Bug #1 (wrong `batch_y`) + Gap #1 (selection logging). These are coupled: the Bug #1 fix captures `global_selected` before `reveal()`, and Gap #1 logs those same indices. |
| 2 | `fix/dataset-private-access` | Bug #2 (`dataset._df` accessed directly from runner) |
| 3 | `fix/retrieval-self-neighbor` | Bug #3 (self-label inclusion in `RetrievalAugmentedEncoder`) |
| 4 | `fix/dead-code-physicochemical` | Bug #4 (dead code in `physicochemical.py`) |
| 5 | `fix/esm-cache-subset-lookup` | Performance: ESMEncoder cache misses every round |

---

## Final Merge to Main

When all fixes are merged into `audit/agent-scaffold` and the scaffold is in a
satisfactory state (smoke test passes, all documented bugs resolved):

```bash
git checkout main
git merge --no-ff audit/agent-scaffold -m "merge: audited agent scaffold into main"
```

---

## Notes

- The first fix branch (`fix/reveal-result-selection-loggin`) has a deliberate
  typo in the name. The name is carried through as-is.
- `docs/bugs.md` is the source of truth for what needs fixing.
- `docs/implementation_map.md` is the source of truth for why each module is
  structured the way it is and what the authorized label-access paths are.
- Update `docs/bugs.md` as each fix is completed (mark resolved, note the commit).
