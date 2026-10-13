#!/usr/bin/env python3
"""test_card_arch_coverage.py — a learned/ README that states its arch coverage must still be right.

`kernel_workflow/knowledge/learned/README.md` tells a reader, in prose, that every card in the folder
was measured on gfx950 and none on gfx942, and what to do about that on an MI300X. That sentence is
load-bearing: it is the difference between treating a card's `effect:` as a number and treating it as
a hypothesis. It is also the kind of sentence that is true the day it is written and quietly false a
month later, the first time a campaign runs on another machine -- and false in the direction that
hides the new evidence, since a reader who believes "there is nothing for gfx942 here" will not look.

So the prose carries a machine-readable marker:

    <!-- arch-coverage: gfx950 -->            (space- or comma-separated, one or more gfx ids)

and this check fails when the marker and the cards disagree in either direction. Adding the first
gfx942 card is then a two-line change -- the card, and the marker -- instead of a silent drift.

Scope, deliberately narrow:

* Only sinks whose cards actually declare `platforms:` are checked. `e2e_workflow/knowledge/learned/`
  uses a lighter frontmatter (`key`/`type`/`confidence`/`confirms`/`effect`/`last_seen`) and encodes
  the arch in the filename and the `key:` line instead. Parsing an arch out of a filename is a guess,
  and a gate built on a guess is worse than no gate, so that sink is reported as unchecked rather
  than policed badly.
* A sink with no marker is not a failure. The marker is opt-in: it means "this README makes a claim
  about coverage, hold me to it". Only the claim is enforced, never the claiming.
* `lifecycle: archived` cards are excluded. An archived card is not advice, and a refuted gfx942 card
  should not make the README assert gfx942 coverage.

    python3 kernel_workflow/scripts/test_card_arch_coverage.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SINKS = [
    os.path.join(ROOT, "kernel_workflow", "knowledge", "learned"),
    os.path.join(ROOT, "e2e_workflow", "knowledge", "learned"),
]

# README.md documents the schema with a `platforms: [<gfx>]` placeholder line, and _archive.md holds
# the frontmatter of retired cards; neither is a card.
NOT_A_CARD = {"README.md", "INDEX.md", "_archive.md"}

MARKER = re.compile(r"<!--\s*arch-coverage:\s*([^>]*?)\s*-->")
GFX = re.compile(r"gfx[0-9a-z]+")


def read_marker(sink):
    """The arch set the sink's README claims, or None if it makes no claim."""
    path = os.path.join(sink, "README.md")
    if not os.path.isfile(path):
        return None
    m = MARKER.search(open(path, encoding="utf-8").read())
    if not m:
        return None
    return set(GFX.findall(m.group(1)))


def card_platforms(sink):
    """{arch -> [card, ...]} over active cards, plus the cards that declare no platforms at all."""
    found, undeclared = {}, []
    for name in sorted(os.listdir(sink)):
        if not name.endswith(".md") or name in NOT_A_CARD:
            continue
        text = open(os.path.join(sink, name), encoding="utf-8").read()
        if re.search(r"^lifecycle:\s*archived\s*$", text, re.M):
            continue
        line = re.search(r"^platforms:\s*(.+)$", text, re.M)
        if not line:
            undeclared.append(name)
            continue
        archs = GFX.findall(line.group(1))
        if not archs:                      # e.g. a literal `[<gfx>]` placeholder left in a card
            undeclared.append(name)
            continue
        for a in archs:
            found.setdefault(a, []).append(name)
    return found, undeclared


def main():
    fails, checked = [], 0

    for sink in SINKS:
        rel = os.path.relpath(sink, ROOT)
        if not os.path.isdir(sink):
            print(f"  skip  {rel} (not in this checkout)")
            continue

        claimed = read_marker(sink)
        found, undeclared = card_platforms(sink)

        if claimed is None:
            print(f"  skip  {rel}/README.md states no arch coverage (no `arch-coverage:` marker)")
            continue
        if not found:
            print(f"  skip  {rel} claims {sorted(claimed)} but no card declares `platforms:` "
                  f"({len(undeclared)} card(s) use the lighter schema) -- nothing to check against")
            continue

        checked += 1
        actual = set(found)
        for a in sorted(actual - claimed):
            fails.append(
                f"{rel}: {len(found[a])} card(s) are measured on {a}, which the README's "
                f"`arch-coverage:` marker does not list -- e.g. {found[a][0]}. Update the marker and "
                f"the paragraph around it; a reader trusting that paragraph will skip these cards.")
        for a in sorted(claimed - actual):
            fails.append(
                f"{rel}: the README's `arch-coverage:` marker claims {a}, but no active card declares "
                f"it. Either the {a} cards were archived/removed, or the marker was written ahead of "
                f"the evidence. The marker records what was measured, not what is planned.")
        if undeclared:
            fails.append(
                f"{rel}: {len(undeclared)} active card(s) declare no `platforms:` in a sink where the "
                f"rest do, so coverage cannot be read off the cards -- e.g. {undeclared[0]}.")

        if not fails:
            total = sum(len(v) for v in found.values())
            print(f"  ok    {rel}: {total} active card(s), platforms {sorted(actual)} == marker")

    for f in fails:
        print(f"  FAIL  {f}")
    if fails:
        print(f"\n{len(fails)} arch-coverage mismatch(es). A README that states its coverage is read "
              f"as fact; when it drifts it hides evidence rather than merely aging.")
        return 1

    if not checked:
        print("\nno sink makes an arch-coverage claim -- nothing to enforce")
        return 0
    print("\nall green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
