#!/usr/bin/env python3
"""Tier-B route report builder for the mega MoE analysis line.

Deterministic, stdlib-only. NEW infra -- NOT part of the SHA-pinned
`moe_bottleneck` skill bundle (SKILL.md / analyze.py / validate_output.py are
untouched). This builder shapes the per-rank/paired latencies the mega
benchmark *already* produces into the minimal `geak-megamoe-analysis-v1` report
that `analyze.py --report` accepts, so `analyze.py` can emit route-causality
findings (which stage-category dominates the fused-vs-baseline delta) at
`analysis_status=awaiting_measurement`, reduced confidence, with NO rocprofv3
measurement tracks. That heavier collection is Tier C, deliberately out of scope.

The output matches `_report(with_tracks=False)` in
tests/test_moe_bottleneck_analysis.py:32-60 (the legacy route form -- no
`metric` key -> analyze.py's legacy branch at analyze.py:219). analyze.py only
requires, per route_comparison: `tokens_per_rank`, `e2e_rank_max_delta_ms`,
`e2e_rank_max_delta_pct`, and a `profile_category_delta.<cat>` carrying
`rank_max_delta_ms` + `rank_max_delta_pct`. It reads `profile_confidence` /
`timing_confidence` off the comparison regardless of branch (analyze.py:145-156).

Granularity limit, surfaced honestly in the report: the bench harness lumps
stage2+combine into a single `stage2_combine` metric unless the candidate's
`component_timings` splits them. When lumped, we emit `stage1` from the split
metric, attribute the lumped delta to `stage2` (combine=0), stamp
`profile_confidence="low"` / `timing_confidence="low"` and
`incomplete_reasons=["stage2_combine lumped by bench harness"]`. Downstream
findings then carry the reduced-confidence label instead of overclaiming a
combine-vs-stage2 split the instrument never resolved.

CLI:
  python3 build_mega_route_report.py \
    --paired <paired_readings.json>            # [{guard,base,cand}, ...]
    --baseline-per-case <baseline.json>        # [{name, latency_ms, ...cats}, ...]
    [--component-timings <cand_components.json>]# optional per-category candidate ms
    [--tokens-per-rank <int>]                  # default: inferred from guard names
    --output <report.json>

Exit 0 on success; prints GEAK_MEGA_ROUTE_REPORT=<path>.
"""

import argparse
import json
import sys
from statistics import median

SCHEMA_VERSION = "geak-megamoe-analysis-v1"

# Canonical analyze.py route categories.
CANON = ("stage1", "stage2", "combine")

# Tolerant key aliases -> canonical category. `component_timings` items are
# free-form (kernel_workflow.js MEGA_ANALYSIS_SCHEMA additionalProperties:true),
# and BASELINE_PER_CASE rows are also open objects, so we probe by alias.
ALIASES = {
    "stage1": "stage1",
    "stage1_dispatch_gemm1": "stage1",
    "dispatch_gemm1": "stage1",
    "gemm1": "stage1",
    "stage2": "stage2",
    "gemm2": "stage2",
    "combine": "combine",
    "scatter": "combine",
    "p2p": "combine",
    # Lumped metric the bench reduces over ranks (build_live_evidence.py:104-109).
    "stage2_combine": "stage2_combine",
    "stage2combine": "stage2_combine",
}


def _num(x):
    """Best-effort float or None (never raises on dirty input)."""
    try:
        if x is None:
            return None
        v = float(x)
        # Reject NaN/inf so analyze.py's _finite() never trips.
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _categories_from(obj):
    """Extract {canonical_or_lumped: ms} from an open dict by alias probing.

    Returns a dict possibly containing 'stage1','stage2','combine','stage2_combine'.
    Only the first alias hit per target wins (deterministic by ALIASES order).
    """
    out = {}
    if not isinstance(obj, dict):
        return out
    lowered = {str(k).strip().lower(): v for k, v in obj.items()}
    for alias, target in ALIASES.items():
        if target in out:
            continue
        if alias in lowered:
            v = _num(lowered[alias])
            if v is not None:
                out[target] = v
    return out


def _reduce_pairs(readings):
    """guard -> (median_base, median_cand) over its {base,cand} readings."""
    by_guard = {}
    for r in readings or []:
        if not isinstance(r, dict):
            continue
        g = str(r.get("guard", "")).strip()
        b = _num(r.get("base"))
        c = _num(r.get("cand"))
        if not g or b is None or c is None:
            continue
        by_guard.setdefault(g, {"base": [], "cand": []})
        by_guard[g]["base"].append(b)
        by_guard[g]["cand"].append(c)
    reduced = {}
    for g, d in by_guard.items():
        if d["base"] and d["cand"]:
            reduced[g] = (median(d["base"]), median(d["cand"]))
    return reduced


def _tokens_from_guard(guard, default):
    """Infer tokens_per_rank from a guard name like '8192_uniform' or 't8192'."""
    import re

    m = re.search(r"(\d{2,6})", str(guard))
    if m:
        return int(m.group(1))
    return default


def _baseline_index(rows):
    """name -> categories dict, from BASELINE_PER_CASE rows."""
    idx = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("case_id") or "").strip()
        if not name:
            continue
        idx[name] = _categories_from(row)
    return idx


def _match_baseline(guard, base_idx):
    """Find the baseline category dict whose case name best matches a guard."""
    if guard in base_idx:
        return base_idx[guard]
    gl = str(guard).lower()
    for name, cats in base_idx.items():
        nl = name.lower()
        if nl in gl or gl in nl:
            return cats
    return {}


def _category_delta(base_cats, cand_cats):
    """Build profile_category_delta + confidence flags.

    Returns (profile_category_delta, lumped_bool). Deltas are candidate-minus-
    baseline per category. When only a lumped stage2_combine is available it is
    folded onto stage2 with combine=0 and lumped=True (low confidence upstream).
    """
    delta = {}
    lumped = False

    def _emit(cat, b, c):
        if b is None or c is None:
            return
        d = c - b
        pct = (d / b * 100.0) if b not in (0, 0.0) else 0.0
        delta[cat] = {"rank_max_delta_ms": d, "rank_max_delta_pct": pct}

    # stage1 is always split when present.
    _emit("stage1", base_cats.get("stage1"), cand_cats.get("stage1"))

    have_split = ("stage2" in base_cats and "stage2" in cand_cats) or (
        "combine" in base_cats and "combine" in cand_cats
    )
    if have_split:
        _emit("stage2", base_cats.get("stage2"), cand_cats.get("stage2"))
        _emit("combine", base_cats.get("combine"), cand_cats.get("combine"))
    else:
        # Fold the lumped stage2_combine onto stage2; combine unresolved -> 0.
        b = base_cats.get("stage2_combine")
        c = cand_cats.get("stage2_combine")
        if b is not None and c is not None:
            _emit("stage2", b, c)
            delta["combine"] = {"rank_max_delta_ms": 0.0, "rank_max_delta_pct": 0.0}
            lumped = True
    return delta, lumped


def build_report(paired, baseline_rows, component_timings, tokens_default):
    reduced = _reduce_pairs(paired)
    base_idx = _baseline_index(baseline_rows)
    cand_idx = _baseline_index(component_timings) if component_timings else {}

    cases = [{"case_id": name} for name in base_idx] or [
        {"case_id": g} for g in reduced
    ]

    route_comparisons = []
    for guard, (b_e2e, c_e2e) in reduced.items():
        e2e_delta = c_e2e - b_e2e
        e2e_pct = (e2e_delta / b_e2e * 100.0) if b_e2e else 0.0
        base_cats = _match_baseline(guard, base_idx)
        cand_cats = _match_baseline(guard, cand_idx)
        pcd, lumped = _category_delta(base_cats, cand_cats)
        rc = {
            "tokens_per_rank": _tokens_from_guard(guard, tokens_default),
            "e2e_rank_max_delta_ms": e2e_delta,
            "e2e_rank_max_delta_pct": e2e_pct,
            "profile_category_delta": pcd,
        }
        # Tier B never has rocprof tracks -> confidence is capped low, and lower
        # still (with an explicit reason) when stage2/combine could not be split.
        rc["timing_confidence"] = "low"
        rc["profile_confidence"] = "low"
        if lumped:
            rc["incomplete_reasons"] = ["stage2_combine lumped by bench harness"]
        route_comparisons.append(rc)

    return {
        "schema_version": SCHEMA_VERSION,
        "cases": cases,
        "route_comparisons": route_comparisons,
    }


def _load(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a Tier-B mega route report.")
    ap.add_argument("--paired", required=True, help="paired_readings JSON [{guard,base,cand}]")
    ap.add_argument("--baseline-per-case", required=True, help="BASELINE_PER_CASE JSON")
    ap.add_argument("--component-timings", default=None, help="optional candidate per-category JSON")
    ap.add_argument("--tokens-per-rank", type=int, default=8192)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    paired = _load(args.paired)
    baseline_rows = _load(args.baseline_per_case)
    component_timings = _load(args.component_timings)

    if not isinstance(paired, list) or not isinstance(baseline_rows, list):
        # Degrade, never fail: emit a schema-valid empty report so analyze.py
        # still runs and reports awaiting_measurement.
        report = {"schema_version": SCHEMA_VERSION, "cases": [], "route_comparisons": []}
    else:
        report = build_report(paired, baseline_rows, component_timings, args.tokens_per_rank)

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"GEAK_MEGA_ROUTE_REPORT={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
