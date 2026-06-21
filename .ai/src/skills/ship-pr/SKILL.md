---
name: ship-pr
description: "Wrap the current feature branch into a GitHub PR: gate (ruff/mypy/pytest), update wiki, push, open PR via gh. Use for 'open the PR', 'wrap this branch up', 'ship it'."
trigger: /ship-pr
---

# /ship-pr

Take the current feature branch from "code done" to "PR open for review". Does **not**
merge — by project rule the user merges manually in the GitHub UI.

## Usage

```
/ship-pr                 # gate, update wiki, push, open PR
/ship-pr --draft         # open as draft
/ship-pr --no-wiki       # skip the wiki pass
```

## Context this needs

- **Repo:** `NavehBrenner/ai-teleop` (public), default branch `master`. Only the `code/`
  tree is the git repo.
- **Open the PR with `gh`** (`gh pr create`). `git push` is fine.
- **The gate** (same as CI): `uv run poe check` from `code/` (= ruff lint + mypy + pytest).
  Tools run via `python -m` inside poe because the relocated `.venv` has stale shebangs.
- **PR↔Linear linking:** branch name carries `lab-NN`, and/or put `Fixes LAB-NN` in the PR
  body (`Fixes`/`Closes` moves the issue to Done on merge). See `code/CLAUDE.md`.

## Procedure

1. **Sanity:** confirm on a feature branch (not `master`); `git -C code status` clean-ish.
   Identify the `LAB-NN` from the branch name.
2. **Update the wiki** (unless `--no-wiki`): record durable, non-obvious findings from this
   work in `project-wiki/` (entities/concepts/synthesis), update `index.md`, append to
   `log.md`. Follow `project-wiki/CLAUDE.md`. (Implementation diaries do NOT go here.)
3. **Gate:** `cd code && uv run poe check`. If red, fix or report — do not open a PR over a
   failing gate (CI will block it anyway).
4. **Push:** `git -C code push -u origin <branch>`.
5. **Write the PR body** to a temp file (summary, what/why, test evidence, `Fixes LAB-NN`).
6. **Open the PR with `gh`:**
   ```
   gh pr create --repo NavehBrenner/ai-teleop --base master --head <branch> \
     --title "<title> (LAB-NN)" --body-file /tmp/pr_body.md
   ```
   Add `--draft` if requested. Print the PR URL for the user to review & merge.
7. Remind the user that merge is theirs (UI only); after they merge, run **/post-merge-sync**.

## Notes

- CI installs `.[dev,scenegen]` and runs mypy + pytest on PRs into master.
- Never merge via CLI; never push feature work straight to `master`.
