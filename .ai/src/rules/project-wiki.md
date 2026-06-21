---
paths:
  - "project-wiki/**"
---

# CLAUDE.md — Project Wiki Schema

This directory is an **LLM-maintained knowledge base** for the AI-assisted
teleoperation project. The pattern: instead of re-deriving knowledge from raw
sources on every question, the agent incrementally builds and maintains a
persistent, interlinked wiki of markdown files. You curate sources and ask
questions; the agent does the reading, summarizing, cross-referencing, filing, and
bookkeeping.

**You (the human) rarely write these pages. The agent writes and maintains them.**

## Three layers

1. **Raw sources** (`raw/`) — immutable source documents: papers, articles, MuJoCo
   docs, clipped web pages, notes. The agent reads these but never edits them. The
   source of truth.
2. **The wiki** (everything else here) — agent-generated markdown: source summaries,
   concept pages, entity pages, synthesis pages. The agent owns this layer entirely.
3. **The schema** (this file) — how the wiki is structured and the workflows to
   follow. Co-evolved over time as we learn what works.

## Directory structure

```
project-wiki/
├── CLAUDE.md      # this schema
├── index.md       # content catalog — every page, one-line summary, by category
├── log.md         # append-only chronological log of ingests / queries / lints
├── raw/           # immutable raw sources (+ raw/assets/ for downloaded images)
├── sources/       # one summary page per ingested raw source
├── concepts/      # concept pages (shared autonomy, impedance control, BC, …)
├── entities/      # entity pages (tools, papers, people, systems)
└── synthesis/     # higher-level synthesis / the evolving project thesis
```

## Page conventions

- **Filenames**: kebab-case, descriptive — e.g. `concepts/residual-policy-learning.md`.
- **Frontmatter** (YAML) on every page:
  ```yaml
  ---
  title: Residual Policy Learning
  type: concept            # source | concept | entity | synthesis
  tags: [shared-autonomy, imitation-learning]
  created: 2026-05-21
  updated: 2026-05-21
  sources: [sources/foo-2023.md]   # which source pages back this page
  ---
  ```
- **Links**: use `[[wikilinks]]` to connect pages (Obsidian-compatible). Link
  liberally; a link to a not-yet-written page is fine — it marks a page worth
  creating.
- **Citations**: claims should reference the `sources/` page they came from, so every
  assertion is traceable to a raw source.

## Operations

### Ingest (add a source)

When a new source lands in `raw/`:
1. Read it (text first; then view referenced images separately if needed).
2. Discuss the key takeaways with the human.
3. Write a summary page in `sources/` (frontmatter + structured summary + key claims).
4. Update or create relevant `concepts/` and `entities/` pages, integrating the new
   information — note where it agrees with, extends, or **contradicts** existing claims.
5. Add/repair `[[wikilinks]]` across affected pages.
6. Update `index.md`.
7. Append an entry to `log.md`.

A single ingest may touch 10–15 pages. That bookkeeping is the agent's job.

### Query (ask a question)

1. Read `index.md` to find relevant pages.
2. Read them; synthesize an answer with citations to `sources/`.
3. **File good answers back into the wiki** — a useful comparison, analysis, or
   discovered connection should become a new page (often `synthesis/` or `concepts/`),
   not vanish into chat history.
4. Append a brief entry to `log.md`.

### Lint (health-check)

Periodically, on request:
- Find contradictions between pages.
- Flag stale claims superseded by newer sources.
- Find orphan pages (no inbound links) and missing cross-references.
- Find important concepts mentioned but lacking their own page.
- Suggest gaps to fill with new sources or web searches.
- Append a lint summary to `log.md`.

## index.md and log.md

- **index.md** is content-oriented: a catalog of every page, grouped by category, each
  with a link and one-line summary. Read it first when answering a query; update it on
  every ingest.
- **log.md** is chronological and append-only. Each entry starts with a consistent
  prefix so it's greppable: `## [YYYY-MM-DD] <op> | <subject>` where `<op>` ∈
  {ingest, query, lint, meta}. e.g. `grep "^## \[" log.md | tail -5` shows recent activity.

## Notes

- The wiki is a git repo of markdown — version history for free. Kept **private**
  (a personal KB, distinct from the public `code/` repo).
- Obsidian-compatible: open this folder as an Obsidian vault to browse links and the
  graph view while the agent edits.
