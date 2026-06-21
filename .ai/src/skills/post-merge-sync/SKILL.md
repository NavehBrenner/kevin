---
name: post-merge-sync
description: "After the user merges a PR, reconcile state: refresh local master, close the Linear issue, sync the wiki, then optionally tee up the next issue. Use for 'I merged, update everything', 'wrap up so we're ready for the next session'."
trigger: /post-merge-sync
---

# /post-merge-sync

Run the standard "I just merged a PR" reconciliation so the repo, Linear, and wiki all
agree and the next session can start clean.

## Usage

```
/post-merge-sync                 # full reconcile
/post-merge-sync --next          # reconcile, then run /next-issue to start the next one
```

## Context this needs

- **Repo:** `NavehBrenner/ai-teleop`, default `master` (only `code/` is the git repo).
- **Linear** (MCP): team **Lab**, project resolved at runtime. `Fixes LAB-NN` in the PR
  usually auto-moves the issue to Done — verify rather than assume.
- **Wiki:** `project-wiki/` per `project-wiki/CLAUDE.md` (index.md + log.md bookkeeping).
- `gh` and `git` both work — use `gh` (e.g. `gh pr view`) for any PR read/write.

## Procedure

1. **Refresh master:**
   ```
   git -C code switch master && git -C code pull
   ```
   Optionally delete the merged local feature branch (`git -C code branch -d <branch>`).
2. **Linear:** confirm the issue is Done (`get_issue`); if the auto-link didn't fire, set
   it with `save_issue`. Note any follow-up issues the merge surfaced.
3. **Wiki sync:** ensure durable findings from the merged work are captured in
   `project-wiki/` (most should already be there from /ship-pr); update `index.md`,
   append a `log.md` entry. Skip if nothing durable.
4. **State report:** one-paragraph "where we are" — what landed, what's next.
5. If `--next`, hand off to **/next-issue**.

## Notes

- The wiki half is mandated by the root `CLAUDE.md` for every session anyway. If you find
  yourself running this purely to satisfy that, consider a **Stop hook** (via
  `/update-config`) so the wiki check fires automatically instead of by hand.
- End-of-MILESTONE (not per-PR) work is bigger — see the "project state review" process
  (recurring LAB issues); that's its own pass, not this one.
