# Docs audit — LAB-42 stage 1B (D-2)

Coverage and navigability of the **user-facing** documentation: `README.md`,
`project-scope.md`, `docs/**.md`. Review scaffolding (`docs/review/`) is excluded — it
documents the review, not the project.

Everything here is reproducible: `uv run python scripts/dev/lab42_docs_coverage.py`
(new, kept) computes the reachability BFS, the broken-link list and the three coverage
matrices. Run date: **2026-07-22**.

---

## Headline

A grader landing on `README.md` **cannot reach the results.** `docs/phase-1-results.md` — the
only document in the repo containing a measured outcome — is not linked from anything
reachable from the README. Neither are the M2, M4, M5 and M6 specs. The repo documents *how
it is built* well and *what it achieved* not at all, and the one file that does the latter is
unreachable by following links.

Three concrete failures, in the order a stranger hits them:

1. **No path to the results** (below).
2. **`kvn train` is documented nowhere** — not in `docs/cli.md`, not in the README. Training a
   policy is the project's central operation.
3. **One broken link at depth 1**: `docs/cli.md:52` → `../CLAUDE.md`, which does not exist.

---

## 1. Reachability from `README.md`

Link hops by BFS over relative markdown links. `-` means *unreachable by following links at
all* — findable only by `ls docs/` or grep.

| Doc | Hops | |
|---|---|---|
| `README.md` | 0 | |
| `project-scope.md` | 1 | |
| `docs/cli.md` | 1 | |
| `docs/milestone-1-spec.md` | 1 | |
| `docs/data-schema.md` | 2 | |
| `docs/design/{expert-corrections,human-generation,policy-model,problem-structure,teleop-input}.md` | 2 | |
| `docs/design/evaluation-protocol.md` | 3 | |
| `docs/milestones.md` | 3 | the project's status page, 3 hops deep |
| `docs/milestone-3-spec.md` | 4 | |
| **`docs/phase-1-results.md`** | **–** | **the results** |
| `docs/milestone-{2,4,5,6}-spec.md` | – | four of six milestone specs |
| `docs/acronym-dictionary.md` | – | referenced by `CLAUDE.md`, not by any doc |
| `docs/lab-78-scripted-human-realism.md` | – | |
| `docs/recentering-handoff.md` | – | a handoff note, arguably shouldn't ship |

The three navigation questions 1B was told to test:

| A stranger asks… | Reachable in ≤2 clicks? |
|---|---|
| *How do I run an episode?* | ✅ `README` → `docs/cli.md` |
| *How do I train a policy?* | ❌ nothing documents `kvn train`; the recipe exists only in `docs/phase-1-results.md:159`, which is unreachable |
| *What were the results?* | ❌ unreachable |

**Verdict: FIX in Phase 3.** This is cheap — a *Results* and a *Documentation map* section in
the README, plus links from `docs/milestones.md` to each milestone spec. D-5
(`docs/policy-guide.md`) answers question 2 by existing; it must be linked from the README the
day it lands, or it inherits this same problem.

---

## 2. Broken links

| Source | Target | Status |
|---|---|---|
| `docs/cli.md:52` | `../CLAUDE.md` | **missing** |

`kevin/CLAUDE.md` does not exist. The workspace root's `CLAUDE.md` also names it ("see
`kevin/CLAUDE.md` for code conventions"), and the conventions actually live in the workspace's
generated `.claude/rules/kevin.md`, outside the public repo. So the public showcase repo ships
**no committed agent/contributor instructions** and two files point at the gap. Either commit
a `kevin/CLAUDE.md` (the sibling `stereohand/` commits its `AGENTS.md`, so the pattern exists)
or drop the reference. **Recommend committing it** — a public repo whose contributor
conventions live in a private sibling is a real gap, not a formatting nit.

---

## 3. Module coverage — 25 of 49 modules are named in no user doc

Named at least once: 24. Named nowhere: **25**.

| Undocumented | Why it matters |
|---|---|
| `policy/train.py` | **The training pipeline.** Just promoted out of `scripts/` by G-1 and still invisible to the docs. |
| `policy/{config,losses,image_encoder,run_artifacts}.py` | The whole policy package except `model.py` / `residual_policy.py` — including `losses.py`, which owns the action-rate penalty the results doc recommends. |
| `dagger.py` | An entire technique with a wiki page and zero repo docs (see §4). |
| `expert/expert.py` | The privileged expert — the thing the policy clones. |
| `sim/scenegen/*` (9 modules) | The procedural wall generator: the largest subpackage in the repo, undocumented end to end. |
| `data/{schema,step_callbacks,images}.py` | `schema.py` is the on-disk contract; `docs/data-schema.md` describes the *format* without naming the module that defines it. |
| `common/{geometry,seating}.py` | `seating.py` is the shared definition of "success" (H-4). |
| `sim/{config,env_setup,scene_source}.py` | |

Two caveats before this is read as 25 problems:

- **Naming a module is not the bar.** `docs/design/*` describe subsystems by behaviour; a
  reader can understand the expert without `expert/expert.py` appearing verbatim. The matrix
  measures *findability from a filename*, which is what you need when reading code.
- **Milestone specs carry most of the load.** 20 of the 24 documented modules are documented
  only in a `milestone-N-spec.md` — historical documents that describe what *was built*, four
  of which are unreachable from the README. That is the deeper finding: **module documentation
  is concentrated in specs, and the specs are neither current nor navigable.**

D-3 (`docs/architecture-tour.md`, stage 1D) is the right fix for both — one navigable
document that names every module and says why it exists — not 25 new pages.

---

## 4. Scripts and CLI commands

**Scripts** — 8 of 9 named in a user doc. The exception: **`scripts/dagger.py`**, which is
absent from every doc *and* from `APP_COMMANDS`, so it is reachable only as
`python scripts/dagger.py`. It implements the LAB-105 DAgger loop — a whole M7 arc.

**`kvn` commands** (7 app + 5 poe):

| Command | Documented in |
|---|---|
| `sim`, `smoke`, `episode` | `README.md` + `docs/cli.md` ✅ |
| `harness`, `gen` | `docs/cli.md` ✅ |
| `evaluate` | **only `docs/milestone-6-spec.md`** — itself unreachable |
| **`train`** | **nowhere** ❌ |
| `fmt`, `lint`, `typecheck`, `test`, `check` | `docs/cli.md` ✅ |

So of the two commands that do the project's actual work, one is undocumented and the other is
documented only in an unreachable historical spec. This confirms the planning-stage seed
finding with exact evidence, and adds `train` (the seed finding listed only `train`/`evaluate`
as missing from `docs/cli.md`; `train` turns out to be missing from *everything*).

---

## Ranked fix list for Phase 3

| # | Fix | Effort | Why ranked here |
|---|---|---|---|
| 1 | README gains a **Results** link + a documentation map | XS | The single highest-value doc change in the repo; a grader's first question currently has no answer |
| 2 | `docs/cli.md` gains `train`, `evaluate`, and `dagger`/`report_results` | S | The CLI reference omits the CLI's two most important commands |
| 3 | `docs/milestones.md` links each milestone spec; README links `milestones.md` | XS | Makes 4 orphaned specs reachable, and fixes the status page being 3 hops deep |
| 4 | Commit `kevin/CLAUDE.md` (or drop the two references) | S | Public repo currently has no contributor instructions and one broken link |
| 5 | D-3 architecture tour names every module | M | The real fix for §3; already scheduled as stage 1D |
| 6 | Decide `docs/recentering-handoff.md` and `docs/lab-78-*.md` | XS | Orphaned working notes in a showcase repo — link them or delete them |

Not recommended: writing a page per undocumented module. The tour (D-3) plus the policy guide
(D-5) cover the same ground in two navigable documents instead of twenty-five stubs.
