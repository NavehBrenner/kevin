---
name: project-status
description: "Give a grounded state-of-the-project tour: what's implemented, what each test/script demonstrates, and what's next. Use for 'recap what's done', 'quick tour of the current state', 'figure out where we are', session start / handoff."
trigger: /project-status
---

# /project-status

Produce an honest, current snapshot of the workshop: what exists, how to see it working,
and what's next. Good at session start, after a break, or when handing off.

## Usage

```
/project-status              # full tour
/project-status --brief      # just: current milestone, last thing landed, next issue
```

## Context this needs

- **Code reality:** `code/src/ai_teleop/` (module map in `__init__.py` docstring),
  `code/tests/`, `code/scripts/` (runnable harnesses) and `code/scripts/dev/` (probes).
- **Progress:** `git -C code log --oneline -20`, open PRs via
  `gh pr list --repo NavehBrenner/ai-teleop`,
  and Linear (team **Lab**, project resolved at runtime — milestone + issue statuses).
- **Design intent:** `code/project-scope.md`, `code/docs/milestones.md`.
- **Accumulated knowledge:** `project-wiki/index.md` (catalog) — and the graphify graph in
  `graphify-out/` for cross-cutting "which script exercises which control law" questions
  (navigation aid only; wiki is source of truth).

## Procedure

1. **Where we are:** current milestone (from `milestones.md` + Linear), last few merges,
   open PRs.
2. **What's implemented:** walk the package modules; note what's real vs. stubbed.
3. **How to see it work — the demo map:** for each major component, name the test/script
   that exercises it and the exact command (prefer `uv run poe <task>`; e.g. `poe sim`,
   `poe smoke`, or `scripts/run_episode.py`, `scripts/dev/record_*.py`). This is the part
   the user asks for most ("what tests let me see what's actually happening"). For visual
   ones, point at /demo-component.
4. **What's next:** next issue(s) per Linear (defer to /next-issue for picking).
5. Keep it grounded — verify claims against current code, don't recite stale memory.

## Notes

- If the wiki is out of date with what you discover, flag it (and update per
  `project-wiki/CLAUDE.md`).
