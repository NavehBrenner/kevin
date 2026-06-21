---
name: spec-milestone
description: "Turn a workshop milestone (or a component) into a reviewed set of Linear issues. Use for 'spec M5', 'create the issues for M6', 'go over <component>, figure out what's missing and create issues'."
trigger: /spec-milestone
---

# /spec-milestone

Break a milestone (or a component) into concrete Linear issues — proposed to the user
for approval **before** anything is created.

## Usage

```
/spec-milestone M5                  # spec milestone M5 into issues
/spec-milestone "backbone controller"   # gap-analyze a component into issues
```

## Context this needs

- **Source of truth for scope:** `code/project-scope.md` and the milestone spec in
  `code/docs/` (`milestone-<N>-spec.md`, `milestones.md`). Respect each milestone's
  **anti-scope** — don't pull future work forward.
- **Current code state:** read `code/src/ai_teleop/` + `code/tests/` to find what already
  exists vs. what the spec calls for (the gap = the issues).
- **Linear** (MCP): team **Lab** / project resolved at runtime via `list_teams` /
  `list_projects`. Milestones are Linear *project milestones* — attach issues to the
  right one (`list_milestones`).

## Procedure

1. Read the milestone spec (or, for a component, the relevant scope section + code).
2. Diff spec-against-reality: list what's missing, in dependency order.
3. **Propose the issue list to the user first** — title + one-line scope + dependencies
   for each. This is a hard rule the user has stated ("validate against me what issues
   you are creating"). Wait for approval / edits.
4. On approval, create them with `save_issue`: attach to the milestone, set a sensible
   order, keep each issue single-feature (one branch / one PR each — see /next-issue).
5. Don't create the whole project up front — only the requested milestone/component.
6. Summarize created issues with their `LAB-NN` ids.

## Notes

- Title issues so the branch name `feat/lab-<NN>-<slug>` reads naturally.
- If specing reveals scope drift, update `code/project-scope.md` rather than letting
  code and spec diverge.
