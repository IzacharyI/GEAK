"""Configurable raw-kernel-name -> logical-category bucketing for Chrome/Perfetto trace JSON.

A raw per-rank trace (the kind ``torch.profiler`` / rocprofv3 export via ``export_chrome_trace``)
carries only a ``name`` string per kernel event — there is no category field. Turning "kernel names"
into "logical stages" (e.g. a MoE kernel's dispatch/GEMM1/GEMM2/combine split) is operator-specific
knowledge, so this module takes the category map as a PARAMETER rather than hardcoding one: the
generic framework knows how to bucket by regex, an advisory Skill (or any caller) supplies what the
regexes ARE.

See README.md for why this split exists (mirrors the intent of e2e_workflow's
``parse_profile.py:classify()``, but reusable across operators instead of one hardcoded table).
"""

from __future__ import annotations

import json
import re
from typing import Mapping

_EPS = 1e-9

__all__ = ["load_category_map", "bucket_trace_events"]


def load_category_map(path_or_dict) -> dict:
    """Load a {category_name: regex_pattern} mapping from a dict, a JSON file path, or a JSON string.

    Returns ``{category_name: re.Pattern}``. Never raises on individually-bad patterns — a pattern
    that fails to compile is skipped and recorded under the returned dict's special
    ``"__errors__"`` key (a list of ``{"category":..., "pattern":..., "error":...}``), so one bad
    entry in a hand-edited map does not take down the whole framework.
    """
    if isinstance(path_or_dict, Mapping):
        raw = dict(path_or_dict)
    elif isinstance(path_or_dict, str) and path_or_dict.strip().startswith("{"):
        raw = json.loads(path_or_dict)
    else:
        with open(path_or_dict, "r") as f:
            raw = json.load(f)
    compiled: dict = {}
    errors = []
    for name, pattern in raw.items():
        try:
            compiled[name] = re.compile(pattern)
        except re.error as e:
            errors.append({"category": name, "pattern": pattern, "error": str(e)})
    if errors:
        compiled["__errors__"] = errors
    return compiled


def _categorize(name: str, category_map: Mapping[str, re.Pattern]) -> str:
    for cat, pattern in category_map.items():
        if cat == "__errors__":
            continue
        if pattern.search(name):
            return cat
    return "unclassified"


def bucket_trace_events(trace_json_path: str, category_map: Mapping[str, re.Pattern]) -> dict:
    """Sum ``dur`` (microseconds) of every ``cat:"kernel"`` traceEvent into its matched category.

    Returns ``{"total_ms": float, "per_category_ms": {category: float}, "kernel_names":
    {category: [names...]}}`` for ONE rank's trace file. Callers merge multiple ranks' outputs with
    ``aggregate.merge_rank_records`` (one record per rank, metric path ``"per_category_ms.<cat>"``)
    to get rank_mean/rank_max/rank_tail_spread_pct per category — the same shape the one existing
    MoE analysis artifact this framework generalizes already uses (see this package's README).

    Degrades gracefully: an unreadable/malformed trace file returns
    ``{"total_ms": 0.0, "per_category_ms": {}, "kernel_names": {}, "error": "<message>"}`` rather
    than raising, so one bad rank's trace does not block aggregation across the rest.
    """
    try:
        with open(trace_json_path, "r") as f:
            trace = json.load(f)
        events = trace.get("traceEvents", [])
    except (OSError, json.JSONDecodeError) as e:
        return {"total_ms": 0.0, "per_category_ms": {}, "kernel_names": {}, "error": str(e)}

    per_category_us: dict[str, float] = {}
    kernel_names: dict[str, set] = {}
    for ev in events:
        if ev.get("cat") != "kernel" or "dur" not in ev or "name" not in ev:
            continue
        cat = _categorize(ev["name"], category_map)
        per_category_us[cat] = per_category_us.get(cat, 0.0) + float(ev["dur"])
        kernel_names.setdefault(cat, set()).add(ev["name"])

    per_category_ms = {k: v / 1000.0 for k, v in per_category_us.items()}
    return {
        "total_ms": sum(per_category_ms.values()),
        "per_category_ms": per_category_ms,
        "kernel_names": {k: sorted(v) for k, v in kernel_names.items()},
    }
