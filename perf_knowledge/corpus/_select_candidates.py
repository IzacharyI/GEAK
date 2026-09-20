#!/usr/bin/env python3
"""Select conditioned corpus cards without turning source precedent into a verdict.

The selector is deliberately static.  It answers "which cards apply to this task?" from the
machine-readable `match` block; it never claims that a matching performance candidate is fast.
Measured outcome summaries may rank the returned candidates later, while constraints and the
target-box verifier remain separate gates.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

import _normalize_performance_decisions as PERFORMANCE


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "catalog.yaml"
TYPE_ORDER = {"constraint": 0, "performance_candidate": 1, "semantic": 2}
DTYPE_ALIASES = {
    "float16": "f16",
    "fp16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level mapping")
    return data


def normalize_dtype(value: str | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    return DTYPE_ALIASES.get(value, value)


def normalize_gfx(value: str | None) -> str | None:
    return str(value).split(":", 1)[0] if value is not None else None


def resolve_family(catalog: dict[str, Any], requested: str) -> dict[str, Any]:
    matches = []
    for family in catalog.get("families") or []:
        names = {str(family.get("id") or "")}
        names.update(str(value) for value in family.get("patterns") or [])
        if requested in names:
            matches.append(family)
    if not matches:
        raise ValueError(f"no corpus family publishes operator/pattern {requested!r}")
    if len(matches) != 1:
        raise ValueError(f"operator/pattern {requested!r} is ambiguous across corpus families")
    return matches[0]


def _dimension(match: dict[str, Any], field: str, value: str | None) -> tuple[str, str | None]:
    allowed = match.get(field) or []
    if not allowed:
        return "eligible", None
    if value is None:
        return "deferred", f"missing {field}"
    normalized = normalize_dtype(value) if field == "dtypes" else (
        normalize_gfx(value) if field == "gfx" else value
    )
    expected = {
        normalize_dtype(item) if field == "dtypes"
        else normalize_gfx(item) if field == "gfx"
        else str(item)
        for item in allowed
    }
    if normalized not in expected:
        return "rejected", f"{field}={value!r} not in {sorted(expected)!r}"
    return "eligible", None


def classify_card(card: dict[str, Any], context: dict[str, Any]) -> tuple[str, list[str]]:
    """Return eligible/deferred/rejected plus explicit reasons."""
    match = card.get("match") or {}
    reasons: list[str] = []

    dimensions = (
        ("operator_families", context.get("operator_family")),
        ("target_languages", context.get("target_language")),
        ("gfx", context.get("gfx")),
        ("dtypes", context.get("dtype")),
        ("regimes", context.get("regime")),
        ("bottlenecks", context.get("bottleneck")),
    )
    statuses = []
    for field, value in dimensions:
        status, reason = _dimension(match, field, value)
        statuses.append(status)
        if reason:
            reasons.append(reason)

    gfx = normalize_gfx(context.get("gfx"))
    excluded_gfx = {normalize_gfx(value) for value in match.get("exclude_gfx") or []}
    if excluded_gfx and gfx is None:
        statuses.append("deferred")
        reasons.append("missing gfx required to evaluate exclusion")
    elif gfx in excluded_gfx:
        statuses.append("rejected")
        reasons.append(f"gfx={gfx!r} is explicitly excluded")

    required = set(match.get("requires") or [])
    available = set(context.get("traits") or [])
    missing_traits = sorted(required - available)
    if missing_traits:
        statuses.append("deferred")
        reasons.append(f"missing required traits {missing_traits!r}")

    shape = match.get("shape") or {}
    for axis in ("m", "n", "k"):
        low, high = shape.get(f"{axis}_min"), shape.get(f"{axis}_max")
        if low is None and high is None:
            continue
        value = context.get(axis)
        if value is None:
            statuses.append("deferred")
            reasons.append(f"missing shape axis {axis}")
            continue
        if low is not None and value < low:
            statuses.append("rejected")
            reasons.append(f"{axis}={value} is below {low}")
        if high is not None and value > high:
            statuses.append("rejected")
            reasons.append(f"{axis}={value} exceeds {high}")

    if "rejected" in statuses:
        return "rejected", reasons
    if "deferred" in statuses:
        return "deferred", reasons
    return "eligible", reasons


def _op_spec_value(op_spec: dict[str, Any], key: str):
    if key in op_spec:
        return op_spec[key]
    shapes = op_spec.get("shapes")
    if isinstance(shapes, dict):
        return shapes.get(key)
    return None


def outcome_context_matches(measured: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Require the durable environment keys and reject any stated task mismatch."""
    if measured.get("observed_language") != requested.get("target_language"):
        return False
    if requested.get("gfx") and normalize_gfx(measured.get("gfx")) != normalize_gfx(requested.get("gfx")):
        return False
    for requested_field, measured_field in (
        ("flydsl_version", "flydsl_version"),
        ("provider", "provider"),
        ("metric_kind", "metric_kind"),
        ("aiter_commit", "aiter_commit"),
        ("rocm_version", "rocm_version"),
        ("protocol_fingerprint", "protocol_fingerprint"),
    ):
        value = requested.get(requested_field)
        if value is not None and measured.get(measured_field) != value:
            return False
    op_spec = measured.get("op_spec")
    if not isinstance(op_spec, dict):
        op_spec = {}
    for key in ("dtype", "regime", "m", "n", "k"):
        want = requested.get(key)
        got = _op_spec_value(op_spec, key)
        if want is None:
            continue
        if got is None:
            return False
        if key == "dtype":
            if normalize_dtype(str(got)) != normalize_dtype(str(want)):
                return False
        elif str(got) != str(want):
            return False
    return True


def measured_priors(outcomes: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build safe priors from single-ref directions only; bundles never rank one card."""
    values: dict[str, list[float]] = {}
    absolute_values: dict[str, list[float]] = {}
    runs: dict[str, set[str]] = {}
    for row in outcomes.get("single_ref_direction_results") or []:
        decisions = row.get("decisions") or []
        if (
            len(decisions) != 1
            or row.get("attribution_scope") != "single_ref_direction"
            or row.get("quality_eligible_for_ranking") is not True
            or not outcome_context_matches(row.get("context") or {}, context)
        ):
            continue
        incremental = row.get("incremental_vs_incumbent")
        if (
            isinstance(incremental, bool)
            or not isinstance(incremental, (int, float))
            or not math.isfinite(float(incremental))
            or incremental <= 0
        ):
            continue
        decision = str(decisions[0])
        values.setdefault(decision, []).append(float(incremental))
        absolute = row.get("speedup_vs_frozen_baseline")
        if (
            not isinstance(absolute, bool)
            and isinstance(absolute, (int, float))
            and math.isfinite(float(absolute))
            and absolute > 0
        ):
            absolute_values.setdefault(decision, []).append(float(absolute))
        runs.setdefault(decision, set()).add(str(row.get("run") or ""))
    priors = {}
    for decision, samples in values.items():
        count = len(samples)
        priors[decision] = {
            "metric": "incremental_vs_incumbent",
            "basis": "single_ref_direction; planner attribution, not causal proof",
            "count": count,
            "unique_runs": len(runs[decision]),
            "median": statistics.median(samples),
            "min": min(samples),
            "max": max(samples),
            "confidence": "high" if count >= 5 else "medium" if count >= 3 else "low",
        }
        if absolute_values.get(decision):
            priors[decision]["absolute_median_vs_frozen_baseline"] = statistics.median(
                absolute_values[decision]
            )
    return priors


def compatible_bundles(outcomes: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in outcomes.get("bundled_direction_results") or []:
        if (
            row.get("quality_eligible_for_ranking") is True
            and outcome_context_matches(row.get("context") or {}, context)
        ):
            rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("incremental_vs_incumbent") or 0),
            tuple(row.get("decisions") or []),
            str(row.get("run") or ""),
        ),
    )


def select(
    catalog_path: Path,
    context: dict[str, Any],
    outcomes_path: Path | None = None,
) -> dict[str, Any]:
    catalog = load_yaml(catalog_path)
    if catalog.get("schema_version") != 1:
        raise ValueError(f"{catalog_path}: unsupported catalog schema")
    family = resolve_family(catalog, context["operator_family"])
    decisions_path = catalog_path.parent / family["decisions"]
    decision_data = load_yaml(decisions_path)
    axes_path = (
        catalog_path.parent / family["performance_axes"]
        if family.get("performance_axes")
        else None
    )
    performance_model = (
        PERFORMANCE.model_summary(load_yaml(axes_path))
        if axes_path is not None
        else {"axes": [], "comparison_context": {}, "dependencies": [], "contract": {}}
    )

    buckets: dict[str, list[dict[str, Any]]] = {
        "eligible": [],
        "deferred": [],
        "rejected": [],
    }
    outcomes = load_yaml(outcomes_path) if outcomes_path else {}
    priors = measured_priors(outcomes, context) if outcomes else {}
    for card in decision_data.get("cards") or []:
        status, reasons = classify_card(card, context)
        row = {
            "id": card.get("id"),
            "type": card.get("type"),
            "question": card.get("question"),
            "evidence_level": card.get("evidence_level"),
            "status": card.get("status"),
            "reasons": reasons,
        }
        if str(card.get("id")) in priors:
            row["measured_prior"] = priors[str(card.get("id"))]
        buckets[status].append(row)
    def rank(row):
        prior = row.get("measured_prior") or {}
        median = float(prior.get("median") or 0)
        measured = 0 if prior and median > 1.0 else 2 if prior else 1
        gain = -median
        support = -int(prior.get("count") or 0)
        absolute = -float(prior.get("absolute_median_vs_frozen_baseline") or 0)
        return TYPE_ORDER.get(row["type"], 99), measured, gain, support, absolute, str(row["id"])

    for values in buckets.values():
        values.sort(key=rank)

    return {
        "schema_version": 1,
        "family": family["id"],
        "context": context,
        "performance_axes": performance_model["axes"],
        "performance_model": performance_model,
        "measured_bundles": compatible_bundles(outcomes, context) if outcomes else [],
        **buckets,
        "contract": (
            "Static match selects applicable cards. Compatible single-ref outcomes may rank "
            "performance candidates but are planner attribution, never a target-box verdict. "
            "Shared performance axes align backend questions and spelling only; they do not make "
            "unlike representations equivalent. Constraints alone may reject impossible candidates."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--operator-family", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--gfx")
    parser.add_argument("--flydsl-version")
    parser.add_argument("--provider")
    parser.add_argument("--metric-kind", choices=("geomean", "time_weighted"))
    parser.add_argument("--aiter-commit")
    parser.add_argument("--rocm-version")
    parser.add_argument("--protocol-fingerprint")
    parser.add_argument("--dtype")
    parser.add_argument("--regime")
    parser.add_argument("--bottleneck")
    parser.add_argument("--m", type=int)
    parser.add_argument("--n", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--outcomes", type=Path,
                        help="optional output of _aggregate_decision_outcomes.py")
    args = parser.parse_args()
    context = {
        "operator_family": args.operator_family,
        "target_language": args.target_language,
        "gfx": args.gfx,
        "flydsl_version": args.flydsl_version,
        "provider": args.provider,
        "metric_kind": args.metric_kind,
        "aiter_commit": args.aiter_commit,
        "rocm_version": args.rocm_version,
        "protocol_fingerprint": args.protocol_fingerprint,
        "dtype": normalize_dtype(args.dtype),
        "regime": args.regime,
        "bottleneck": args.bottleneck,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "traits": sorted(set(args.trait)),
    }
    print(json.dumps(select(args.catalog, context, args.outcomes), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
