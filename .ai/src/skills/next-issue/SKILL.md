---
name: next-issue
description: "Pick the next logical Linear issue for the workshop and start it (branch + in-progress). Use for 'what's the next issue', 'implement the next logical issue', 'lets implement lab-NN', 'start the next one'."
trigger: /next-issue
---

# /next-issue

Pick the next Linear issue to work on (or take a specific `lab-NN`) and set up to implement it.

## Usage

```
/next-issue                 # choose the next logical issue, confirm, then start
/next-issue lab-42          # start a specific issue
/next-issue --plan-only     # just tell me what's next + why, don't create the branch
```

## Context this needs

- **Linear** (MCP): team **Lab** (key `LAB`), workspace `linear.app/naveh-brenner`.
  Resolve the team + project IDs at runtime with `list_teams` / `list_projects`
  (do NOT trust hardcoded URLs — the project was migrated off `katsir-consulting`).
- Epics = milestones M1–M9 + checkpoints D1/D2, mapped to `code/docs/milestones.md`.
- Git workflow + PR↔Linear linking conventions: `code/CLAUDE.md` → *Git workflow*.

## Procedure

1. **Find the active milestone.** `list_issues` for the project; look at statuses and
   the current milestone (per `code/docs/milestones.md`). Honor milestone order — don't
   pull future-milestone work forward (anti-scope rule in `code/CLAUDE.md`).
2. **Pick the next issue.** Prefer: in-progress > unblocked todo in the active milestone,
   lowest dependency depth first. Skip blocked/needs-info issues. If a `lab-NN` was given,
   use it.
3. **Confirm with the user** which issue and why (one line). Stop here if `--plan-only`.
4. **Read the issue** in full (`get_issue`) plus its milestone spec in `code/docs/`.
5. **Set up the branch** from up-to-date `master`:
   ```
   git -C code switch master && git -C code pull
   git -C code switch -c feat/lab-<NN>-<short-slug>
   ```
   The branch name MUST embed `lab-<NN>` so the PR auto-links in Linear.
6. **Move the issue to In Progress** (`save_issue` / `get_issue_status`).
7. Begin implementing per the milestone spec. When done, hand off to **/ship-pr**.

## Notes

- One branch + one PR per issue.
- `git switch`, never `git checkout` / `git branch` for navigation.
- Ad-hoc probes go in `code/scripts/dev/`, run with `uv run python ...` (never `python -c`).
