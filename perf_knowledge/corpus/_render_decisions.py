#!/usr/bin/env python3
"""Render actionable GEMM development cards from curated decisions and shipped tuning evidence.

The source-evidence index answers "what is written in AITER?". This renderer answers the next
question an author actually has: "given my conditions, what should I try, why, what are the
alternatives, and how strong is the evidence?". It does not infer that from regex hits. Curated cards
state those semantics explicitly, and every source citation is checked against the extracted evidence.

AITER's shipped tuning groups are safe to derive mechanically: a knob shared by every selected config
in a `(gfx, variant, M bucket)` group is a useful value to seed; a knob that varied is one to sweep.
No benchmark archive is attached here, so "selected" never becomes "winner" or a ranking.
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)

sys.path.insert(0, HERE)
import _render_facts as source_renderer

DECISIONS = os.path.join(HERE, "decisions", "gemm.yaml")
SOURCE_EVIDENCE = os.path.join(HERE, "evidence", "gemm_source.yaml")
TUNED_EVIDENCE = os.path.join(HERE, "evidence", "gemm_tuned_configs.yaml")
DOC = os.path.join(HERE, "gemm_decisions.md")

REQUIRED = {
    "id", "question", "evidence_level", "status", "conditions", "actions", "alternatives", "why",
    "source_evidence", "measurement_evidence", "limitations",
}
LEVELS = {"source_observed"}
GFX_ORDER = {"gfx950": 0, "gfx942": 1, "gfx1250": 2, "gfx1201": 3}
LEVEL_TEXT = {
    "source_observed": (
        "Source-observed candidate — the cited implementation exists; no performance preference is "
        "implied."
    ),
}


def validate(cards, source_evidence):
    """Return validation errors; never silently render an ungrounded decision card."""
    errors = []
    evidence_ids = [record.get("evidence_id") for record in source_evidence]
    available = {record.get("evidence_id"): record for record in source_evidence}
    if None in available:
        errors.append("source evidence record has no evidence_id; regenerate with the current extractor")
    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("source evidence contains duplicate evidence_id values")
    ids = set()
    for i, card in enumerate(cards):
        label = card.get("id") or f"card[{i}]"
        missing = sorted(REQUIRED - set(card))
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue
        if label in ids:
            errors.append(f"{label}: duplicate id")
        ids.add(label)
        if card["evidence_level"] not in LEVELS:
            errors.append(f"{label}: unknown evidence_level {card['evidence_level']!r}")
        for field in ("conditions", "actions", "why", "limitations"):
            if not card.get(field):
                errors.append(f"{label}: {field} must not be empty")
        if not card.get("source_evidence"):
            errors.append(f"{label}: source_observed card needs source_evidence")
        if card.get("measurement_evidence"):
            errors.append(
                f"{label}: measured evidence belongs in learned cards/expert skills behind their "
                "feature switch, not in the always-on corpus"
            )
        for ref in card.get("source_evidence") or []:
            if not re.fullmatch(r"src_[0-9a-f]{16}", str(ref)):
                errors.append(f"{label}: source_evidence must use a content-bound `src_…` ID: {ref}")
            elif ref not in available:
                errors.append(f"{label}: source evidence ID not present in extracted evidence: {ref}")
    return errors


def _bullets(items):
    return [f"- {item}" for item in items]


def render_card(card, evidence_by_id):
    level = card["evidence_level"]
    out = [
        f"## {card['question']}",
        "",
        f"**Card:** `{card['id']}` · **evidence:** `{level}` · **status:** `{card['status']}`",
        "",
        LEVEL_TEXT[level],
        "",
        "### Use when",
        "",
        *_bullets(card["conditions"]),
        "",
        "### Try",
        "",
        *_bullets(card["actions"]),
        "",
        "### Why this is a candidate",
        "",
        *_bullets(card["why"]),
    ]
    if card.get("alternatives"):
        out += ["", "### Keep as alternatives", "", *_bullets(card["alternatives"])]
    out += ["", "### Evidence", ""]
    for ref in card.get("source_evidence") or []:
        record = evidence_by_id[ref]
        out.append(
            f"- `{ref}` — `{record['category']}` `{source_renderer.value_of(record)}` at "
            f"`{record['file']}:{record['line']}`"
        )
    out += ["", "### Limits", "", *_bullets(card["limitations"]), ""]
    return out


def _fixed_values(group):
    return group.get("same_across_configs") or group.get("knobs") or {}


def render_tuning_table(tuned):
    """Turn each shipped config group into a concrete seed-candidate / vary-next instruction."""
    out = [
        "## Shipped configuration seeds",
        "",
        ("These rows are generated from AITER's shipped selected configs. Match all three condition "
         "columns before using a row. **Seed candidate** means every config in that group carries the "
         "value; **vary next** lists knobs that changed by shape. No benchmark archive or rejected "
         "alternatives are attached, so these are concrete candidates, not measured winners."),
        "",
    ]
    by_gfx = {}
    for group in tuned:
        by_gfx.setdefault(str(group.get("gfx")), []).append(group)
    for gfx in sorted(by_gfx, key=lambda value: (GFX_ORDER.get(value, 99), value)):
        groups = sorted(by_gfx[gfx], key=lambda g: (str(g.get("tags")), str(g.get("m_bucket"))))
        out += [
            f"### `{gfx}`",
            "",
            "| decision ref | variant | M bucket | seed candidate | vary next | shipped support |",
            "|---|---|---|---|---|---|",
        ]
        for group in groups:
            fixed = _fixed_values(group)
            shapes = int(group.get("shape_configs") or 0)
            if fixed:
                seed = ", ".join(f"`{k}={v}`" for k, v in sorted(fixed.items()))
            else:
                seed = "none shared by all configs"
            varies = group.get("varies_by_shape") or {}
            if varies:
                vary = ", ".join(f"`{k}`" for k in sorted(varies))
            elif shapes == 1:
                vary = "all exposed knobs; one config cannot establish agreement"
            else:
                vary = "no varying knob recorded"
            source_count = len(group.get("source_files") or [])
            out.append(
                f"| `{group.get('config_id')}` | `{group.get('tags')}` | "
                f"`{group.get('m_bucket')}` | {seed} | {vary} | "
                f"{shapes} selected shape config(s) in {source_count} JSON file(s) |"
            )
        out.append("")
    return out


def render(decision_data, source_data, tuned_data):
    schema = (decision_data.get("provenance") or {}).get("schema_version")
    if schema != 2:
        raise ValueError(f"unsupported decisions schema_version {schema!r}; expected 2")
    cards = decision_data.get("cards") or []
    source_evidence = source_data.get("source_evidence") or []
    tuned = tuned_data.get("tuned_configs") or []
    errors = validate(cards, source_evidence)
    if errors:
        raise ValueError("\n".join(errors))

    provenance = source_data.get("provenance") or {}
    evidence_by_id = {record["evidence_id"]: record for record in source_evidence}
    out = [
        "# GEMM development decision cards",
        "",
        ("This is the actionable layer. It does not dump regex matches and ask the reader to infer a "
         "recommendation. Every curated card states **when it applies, what to try, why, alternatives, "
         "evidence strength and limits**. The raw, reproducible source observations remain in "
         "[`gemm_source_evidence.md`](gemm_source_evidence.md)."),
        "",
        (f"Source baseline: AITER `{provenance.get('aiter_commit', '?')}` · "
         f"{len(cards)} curated decision card(s) · {len(source_evidence)} source-evidence records · "
         f"{len(tuned)} shipped tuning groups."),
        "",
        "Evidence levels:",
        "",
        "- `source_observed`: implementation precedent only; add a candidate and measure it.",
        ("- `shipped_config`: parameter seed selected in AITER's source tree; alternatives and benchmark "
         "results are not attached, so vary and measure locally."),
        ("- Measured guidance is deliberately not copied here: it stays in learned cards/expert skills "
         "behind their existing feature switches."),
        "",
    ]
    for card in cards:
        out += render_card(card, evidence_by_id)
    out += render_tuning_table(tuned)
    out += [
        "## Sources",
        "",
        "- Curated cards: [`decisions/gemm.yaml`](decisions/gemm.yaml).",
        "- Source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml).",
        ("- Shipped tuning evidence: "
         "[`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml)."),
        ("- Generated by [`_render_decisions.py`](_render_decisions.py); edit the cards or evidence, "
         "never this file."),
        "",
    ]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true", help="write gemm_decisions.md")
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if gemm_decisions.md differs from its inputs")
    args = parser.parse_args()

    for path in (DECISIONS, SOURCE_EVIDENCE, TUNED_EVIDENCE):
        if not os.path.isfile(path):
            print(f"missing decision input: {os.path.relpath(path, ROOT)}", file=sys.stderr)
            return 2
    try:
        text = render(
            source_renderer.load_yaml(DECISIONS),
            source_renderer.load_yaml(SOURCE_EVIDENCE),
            source_renderer.load_yaml(TUNED_EVIDENCE),
        )
    except ValueError as exc:
        print(f"invalid GEMM decision cards:\n{exc}", file=sys.stderr)
        return 1

    if args.check:
        old = ""
        if os.path.isfile(DOC):
            with open(DOC, encoding="utf-8") as handle:
                old = handle.read()
        if old != text:
            print("STALE: perf_knowledge/corpus/gemm_decisions.md; re-run with --emit",
                  file=sys.stderr)
            return 1
        print(f"OK: {os.path.relpath(DOC, ROOT)} matches decisions and evidence")
        return 0
    if not args.emit:
        sys.stdout.write(text)
        return 0
    with open(DOC, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"OK: {len(text.splitlines())} lines -> {os.path.relpath(DOC, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
