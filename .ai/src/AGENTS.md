# CLAUDE.md — Autonomous Systems Workshop

**Always begin every response with the user's full name: "Naveh Brenner".**

Project-wide guidance for any agent working anywhere in this repository. Subtree
`CLAUDE.md` files add domain-specific rules — see `code/CLAUDE.md` (implementation)
and `project-wiki/CLAUDE.md` (knowledge base).

## What this project is

AI-assisted robotic teleoperation for precision peg-in-hole insertion, built in
MuJoCo for the Franka Emika Panda. A human gives coarse 6-DoF commands; a
vision-conditioned residual policy (behavioral-cloning–trained) supplies real-time
micro-corrections. Solo project for OpenU course 20973 (*Workshop in Autonomous
Systems Simulation*), fall 2026. Final deadline 2026-08-31.

## Repository layout

This folder (the workspace root) holds three things:

- `workshop-booklet.pdf` — the course booklet. **Copyrighted course material — never
  commit it to a public repo.** Read-only reference.
- `code/` — all implementation and project specs. The **public** showcase repo. See
  `code/CLAUDE.md` for code conventions and `code/project-scope.md` for the
  authoritative project definition.
- `project-wiki/` — a **private**, LLM-maintained knowledge base (literature,
  techniques, design rationale). See `project-wiki/CLAUDE.md` for how it operates.

## Project-wide conventions

### Keep the wiki current

Whenever a working session produces **durable, non-obvious knowledge about a
tool, system, or concept** — an undocumented MuJoCo behaviour, a Panda
kinematics convention worth remembering, a technique pulled from a paper —
record it in `project-wiki/` before ending the session. Specifically:

- New tool / library / hardware facts → `project-wiki/entities/<thing>.md`.
- New techniques or learned principles → `project-wiki/concepts/<concept>.md`.
- Cross-page insights or comparisons → `project-wiki/synthesis/<topic>.md`.
- Always update `project-wiki/index.md` (one-line catalog entry) and append
  to `project-wiki/log.md` (`## [YYYY-MM-DD] <op> | <subject>`).
- Follow the conventions in `project-wiki/CLAUDE.md` (frontmatter,
  `[[wikilinks]]`, source citations or an explicit provenance note when the
  knowledge came from direct observation rather than a `raw/` source).

What **does not** belong in the wiki: implementation diaries, milestone
status, "what we did this session". Those live in commit history and in
`code/docs/`. The wiki captures *transferable knowledge* that the next
session (or future-you) would otherwise have to rediscover.

### Knowledge graph — a navigation aid, not a source of truth

A [graphify](https://github.com/sponsors/safishamsi) knowledge graph of the
**whole workspace** (code + docs + wiki + render images) lives in
`graphify-out/` at the workspace root: `graph.json` (data), `graph.html`
(interactive view), `GRAPH_REPORT.md` (audit). Use it to navigate cross-cutting
structure that the wiki's own `[[wikilinks]]` can't show — e.g. which debug
script exercises which control law, or how a test maps to a spec concept:

```
/graphify query "how does the lock state machine reach SimEnv"
/graphify explain "SimEnv"
```

Rules of use:

- **The wiki is the source of truth, not the graph.** The graph is *extracted*
  (lossy, carries INFERRED/AMBIGUOUS edges, includes noise like bare `float`/`int`
  nodes). Treat it as a map for exploration; trust `project-wiki/` for facts.
- **Rebuild before trusting it — it is a snapshot.** After code or doc changes
  run `/graphify --update` (code-only changes need no LLM; doc/image changes do).
  The graph's quality depends on the curated wiki being current, so keep the wiki
  current first (see above).
- **Never let `graphify-out/` migrate into the public `code/` repo.** `graph.json`
  is built over `workshop-booklet.pdf`, so it is copyrighted-derived. It belongs at
  the (non-public) workspace root only.
