---
name: add-convention
description: "Codify a new rule/convention into the correct CLAUDE.md in the hierarchy (and supporting files like the acronym dictionary). Use for 'add a dev rule that...', 'add this to the appropriate CLAUDE.md', 'always do X from now on'."
trigger: /add-convention
---

# /add-convention

Record a new project convention in the right place, at the right scope, in the existing
style — instead of scattering it or putting it at the wrong level.

## Usage

```
/add-convention always use git switch over git checkout
/add-convention PR titles must carry the LAB-NN id
/add-convention shorten <abbr> only if registered in the dictionary
```

## Context this needs — the CLAUDE.md hierarchy

- **Root** `~/autonomous-systems-workshop/CLAUDE.md` — project-wide rules (apply
  everywhere: booklet handling, wiki-currency, graph usage, response conventions).
- **`code/CLAUDE.md`** — implementation rules (git workflow, uv/poe, naming, testing,
  scripts/dev). Most engineering conventions go here.
- **`project-wiki/CLAUDE.md`** — wiki schema/workflow rules only.
- **Supporting files:** variable-naming abbreviations → `code/docs/acronym-dictionary.md`.

## Procedure

1. **Classify the rule** → pick the narrowest CLAUDE.md whose scope it fits (engineering →
   `code/`; cross-cutting → root; wiki → wiki). Naming abbreviations also touch the
   acronym dictionary.
2. **Match existing style:** these files use short sections, tables, and terse bullets —
   slot the rule into the relevant existing section rather than appending a stray line.
3. **Write it**, then show the user the diff for confirmation.
4. **Behavioral "always/whenever/before-after" rules are usually hooks, not prose.** If the
   rule is "from now on, automatically do X on event Y" (e.g. run a formatter on save, post
   a message on Stop), Claude can't self-enforce text reliably — recommend `/update-config`
   to add a hook in `settings.json` instead of (or in addition to) the CLAUDE.md line.

## Notes

- Don't duplicate a rule across levels; put it once at the broadest level it's true and
  reference it from narrower files if needed.
- Personal preferences about how *Claude* should behave (not codebase rules) may belong in
  auto-memory instead — ask if it's a repo convention or a working-style preference.
