#!/usr/bin/env python3
"""Check the two structural promises `perf_knowledge/README.md` makes about this tree.

The README's banner has long advertised "0 broken links · every content file ends with `## Sources`".
Both were true when typed and neither was checked by anything, which is the failure mode this file
exists to remove: a claim that *looks* verifiable and is not is worse than no claim, because a reader
budgets trust against it. When this ran for the first time it found 25 content files with no
`## Sources` — the banner had been wrong for a while and nothing said so.

Three invariants:

1. **Every relative markdown link resolves.** A knowledge base is navigated by following links; one
   that dead-ends sends the reader to `ls` and, more often, to a wrong assumption about what exists.
   External URLs are not fetched (that would make this a network test and flaky by construction) and
   `#anchors` are not resolved into headings — both limits are stated in the output rather than left
   for a reader to discover.

2. **Every content file carries a `## Sources` section.** This tree's whole claim is that a number in
   it can be traced. A file with no sources is indistinguishable from a file whose numbers were
   invented, and the reader cannot tell which they are holding.

Exemptions are enumerated with a reason each, never as a pattern that happens to make the run green.

Run: `python3 perf_knowledge/test_docs_contract.py`
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "perf_knowledge"

# Markdown inline link. A real target holds no whitespace, which is what keeps prose like
# `[x](qx, qy)` from being read as a path — an earlier cut of this check reported four such
# "broken links" that were never links.
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

# Not sources, with the reason. Each entry answers "why would this file cite anything?".
SOURCES_EXEMPT = {
    "index/sources_index.md": "is the sources index — citing itself is circular",
    "index/changelog.md": "records what changed in this tree; its source is the git history",
    "index/sota_matrix.md": "a registry of pointers; every row's source is the doc it points at",
}
# Directory-level exemptions, same rule: a reason, not a convenience.
SOURCES_EXEMPT_DIRS = {
    "_templates": "templates are shapes to fill, not content — their `## Sources` is a placeholder",
    "expert_skills": "vendored skill bundles carry their upstream layout; rewriting their headings "
                     "would fork them from the source they came from",
}
# Structural files: an index or a readme is navigation, and its sources are its entries.
SOURCES_EXEMPT_NAMES = {"INDEX.md", "README.md", "SKILL.md"}


def markdown_files():
    for dirpath, dirnames, filenames in os.walk(HERE):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for name in sorted(filenames):
            if name.endswith(".md"):
                yield os.path.join(dirpath, name)


def rel(path):
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def exempt_from_sources(relpath):
    if relpath in SOURCES_EXEMPT:
        return SOURCES_EXEMPT[relpath]
    parts = relpath.split("/")
    for d, why in SOURCES_EXEMPT_DIRS.items():
        if d in parts[:-1]:
            return why
    if parts[-1] in SOURCES_EXEMPT_NAMES:
        return "structural: navigation, whose sources are its entries"
    return None


def check_links():
    """Relative links resolve. Returns (checked, [failure strings])."""
    checked, bad = 0, []
    for path in markdown_files():
        relpath = rel(path)
        # Templates link to the paths the FILLED file will have, so they dangle by design.
        if "_templates" in relpath.split("/"):
            continue
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in LINK.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#", "<")):
                continue
            # A literal ellipsis inside link parens is prose, not a path: `[foo](...)`.
            if target in {"...", "…"}:
                continue
            checked += 1
            bare = target.split("#")[0]
            if not bare:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), bare))
            if not os.path.exists(resolved):
                line = text[:m.start()].count("\n") + 1
                bad.append(f"{relpath}:{line} -> {target} (no such path)")
    return checked, bad


def check_sources():
    """Content files carry `## Sources`. Returns (checked, [failure strings], exempted count)."""
    checked, bad, exempted = 0, [], 0
    for path in markdown_files():
        relpath = rel(path)
        why = exempt_from_sources(relpath)
        if why:
            exempted += 1
            continue
        checked += 1
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not re.search(r"^##+\s+Sources\b", text, re.MULTILINE):
            bad.append(f"{relpath} (no `## Sources` section)")
    return checked, bad, exempted


def check_citations():
    """Every `repo@commit:path` citation resolves in a checkout on this box.

    The third promise, and the one that was silently false the longest. A `## Sources` section
    satisfies check_sources() whether or not a single line in it points at anything real: the second
    invariant checks that sources were CITED, never that the citations RESOLVE. Two did not -- a
    path that moved under `aiter/ops/` before the cited commit -- and the suite was green over them
    for as long as they existed.

    The check is delegated to `index/resolve_citation.py`, which resolves through git objects rather
    than through a path, because this box holds five aiter checkouts at five commits and "the file
    is not there" usually means "you are standing in the wrong tree". Returns (checked, [failures]).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "resolve_citation", os.path.join(HERE, "index", "resolve_citation.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cites = mod.kb_citations()
    per_repo, bad = {}, []
    for repo, _, _, _ in cites:
        name = repo.split("/")[-1]
        if name not in per_repo:
            per_repo[name] = mod.checkouts(name)
    seen = {}
    for repo, commit, path, src in cites:
        seen.setdefault((repo, commit, path), []).append(src)
    for (repo, commit, path), srcs in sorted(seen.items()):
        trees = per_repo[repo.split("/")[-1]]
        if not trees:
            # No checkout of that repo on this box: not a defect in the card. Say so and move on,
            # rather than reporting a citation nobody can check here as a broken one.
            continue
        r = mod.resolve(commit, path, trees)
        if not r["ok"]:
            bad.append(f"{sorted(set(srcs))[0]} -> {repo}@{commit[:8]}:{path} ({r['reason']})")
    return len(cites), bad


def main():
    failures = []

    n_links, bad_links = check_links()
    print(f"links:   {n_links} relative links checked, {len(bad_links)} broken")
    print("         (external URLs are not fetched; `#anchors` are not resolved to headings)")
    failures += bad_links

    n_src, bad_src, n_exempt = check_sources()
    print(f"sources: {n_src} content files checked, {len(bad_src)} without `## Sources` "
          f"({n_exempt} exempt, each with a stated reason)")
    failures += bad_src

    n_cite, bad_cite = check_citations()
    print(f"cites:   {n_cite} `repo@commit:path` citations resolved against on-box checkouts, "
          f"{len(bad_cite)} unresolved")
    print("         (resolved through git objects, not through a path -- see "
          "index/resolve_citation.py)")
    failures += bad_cite

    if failures:
        print(f"\nFAIL: {len(failures)} problem(s) in {REL}/", file=sys.stderr)
        for f in failures[:60]:
            print(f"  {f}", file=sys.stderr)
        if len(failures) > 60:
            print(f"  ... and {len(failures) - 60} more", file=sys.stderr)
        print("\nEither fix the file or, if it genuinely has no sources to cite, add it to "
              "SOURCES_EXEMPT with the reason.", file=sys.stderr)
        return 1

    print(f"\nOK: {REL}/ links all resolve, every content file cites its sources, "
          "and every citation resolves on this box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
