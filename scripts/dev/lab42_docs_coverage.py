"""LAB-42 stage 1B: doc-coverage matrix over modules, scripts, and CLI commands.

For every `src/ai_teleop/*` module, every `scripts/*.py`, and every `kvn` command, report:

* **documented?** — named in any `docs/**.md` or `README.md` (excluding `docs/review/`,
  which is review scaffolding, not user documentation).
* **where** — which doc files mention it.
* **reachable?** — for docs: how many link-hops from `README.md`. A stranger who lands on
  the README should reach "run an episode", "train a policy" and "what were the results"
  without grepping; anything at depth `-` is unreachable by following links at all.

Read-only. Run: `uv run python scripts/dev/lab42_docs_coverage.py`
"""

from __future__ import annotations

import re
from pathlib import Path

DOC_ROOT = Path("docs")
REVIEW = DOC_ROOT / "review"
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def user_docs() -> list[Path]:
    """Every user-facing markdown doc (README + docs/, minus review scaffolding)."""
    docs = [Path("README.md"), Path("project-scope.md")]
    docs += [p for p in sorted(DOC_ROOT.rglob("*.md")) if REVIEW not in p.parents]
    return [p for p in docs if p.exists()]


def link_depth() -> tuple[dict[Path, int], list[str]]:
    """BFS over relative markdown links from README.md; also collect broken links."""
    depth = {Path("README.md"): 0}
    frontier = [Path("README.md")]
    broken: list[str] = []
    while frontier:
        nxt = []
        for page in frontier:
            for target in LINK.findall(page.read_text(encoding="utf-8")):
                target = target.split("#")[0].strip()
                if not target or "://" in target:
                    continue
                resolved = (page.parent / target).resolve()
                try:
                    relative = resolved.relative_to(Path.cwd())
                except ValueError:
                    continue
                if not resolved.exists():
                    broken.append(f"{page} -> {target}")
                    continue
                if relative.suffix == ".md" and relative not in depth:
                    depth[relative] = depth[page] + 1
                    nxt.append(relative)
        frontier = nxt
    return depth, broken


def mentions(needle: str, docs: list[Path]) -> list[str]:
    return [str(d) for d in docs if needle in d.read_text(encoding="utf-8")]


def report(title: str, names: list[tuple[str, str]], docs: list[Path]) -> None:
    print(f"\n### {title}")
    undocumented = []
    for label, needle in names:
        hits = mentions(needle, docs)
        if hits:
            print(f"  {label:<34} {len(hits):>2} doc(s): {', '.join(hits[:3])}")
        else:
            undocumented.append(label)
    print(
        f"  -- UNDOCUMENTED ({len(undocumented)}/{len(names)}): {', '.join(undocumented) or 'none'}"
    )


def main() -> None:
    docs = user_docs()
    depth, broken = link_depth()

    print(f"### Broken relative links reachable from README ({len(broken)})")
    for item in broken:
        print(f"  {item}")

    print("\n### Reachability from README.md (link hops; '-' = unreachable by links)")
    for doc in docs:
        hops = depth.get(doc)
        print(f"  {str(doc):<44} {hops if hops is not None else '-'}")

    modules = [
        (str(p.relative_to("src/ai_teleop")), str(p.relative_to("src/ai_teleop")))
        for p in sorted(Path("src/ai_teleop").rglob("*.py"))
        if "__pycache__" not in str(p) and p.name != "__init__.py"
    ]
    report("Modules (src/ai_teleop/**) named in a user doc", modules, docs)

    scripts = [(p.name, p.name) for p in sorted(Path("scripts").glob("*.py"))]
    report("Scripts (scripts/*.py) named in a user doc", scripts, docs)

    cli_source = Path("src/ai_teleop/cli.py").read_text(encoding="utf-8")
    commands = re.findall(r'^\s{4}"([a-z_]+)":', cli_source, re.MULTILINE)
    report("kvn commands documented", [(c, f"kvn {c}") for c in commands], docs)


if __name__ == "__main__":
    main()
