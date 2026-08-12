"""Generic per-rank scalar reduction, timing, and merge primitives.

Extracted and generalized from the private helpers written for one distributed MoE UT
(``_reduce_float``, ``_all_ranks_true``, ``_time_graph`` in an AITER op_test file — see this
package's README for the provenance). Nothing here is named after that UT, its operator, or its
backend: every function takes plain values/callables and returns plain dicts, so it can be reused by
any distributed correctness/benchmark harness that GEAK's Kernel Workflow drives.

Two independent halves:
  * ``reduce_scalar`` / ``all_ranks_true`` / ``time_distributed`` require a live
    ``torch.distributed`` process group (torch is imported lazily, only when called) — use these
    INSIDE a running distributed UT/benchmark to produce the per-rank numbers.
  * ``merge_rank_records`` / ``classify_case_speedup`` are pure Python (no torch import at all) —
    use these OFFLINE, after per-rank records have already been gathered (e.g. via
    ``dist.all_gather_object`` in the caller, or read back from a JSON report), including from
    unit tests that have no GPU.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Mapping, Sequence

_EPS = 1e-9

__all__ = [
    "reduce_scalar",
    "all_ranks_true",
    "time_distributed",
    "merge_rank_records",
    "classify_case_speedup",
]


def _require_torch_distributed():
    """Lazy import so this module has zero import-time cost/dependency for offline-only callers."""
    import torch
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "multi_rank_analysis.aggregate: reduce_scalar/all_ranks_true/time_distributed require "
            "an initialized torch.distributed process group. Offline callers should use "
            "merge_rank_records()/classify_case_speedup() instead (no torch dependency)."
        )
    return torch, dist


def reduce_scalar(value: float, op: str = "mean", device=None) -> float:
    """All-reduce one scalar across the current distributed process group.

    ``op`` is one of "mean" | "max" | "min" | "sum". Generalizes the "gather one float per rank via
    all_reduce" pattern common to distributed UTs (previously duplicated per-operator as a private
    ``_reduce_float`` helper).
    """
    torch, dist = _require_torch_distributed()
    ops = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "mean": dist.ReduceOp.SUM,
    }
    if op not in ops:
        raise ValueError(f"reduce_scalar: unsupported op={op!r}, expected one of {sorted(ops)}")
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=ops[op])
    out = float(result.item())
    if op == "mean":
        out /= dist.get_world_size()
    return out


def all_ranks_true(value: bool, device=None) -> bool:
    """True iff every rank in the process group passed True. MIN-reduce over a 0/1 tensor."""
    return reduce_scalar(1.0 if value else 0.0, op="min", device=device) >= 1.0


def time_distributed(fn: Callable[[], Any], iters: int, device=None) -> dict:
    """Time ``fn`` locally with CUDA events over ``iters`` calls, then reduce mean/max across ranks.

    Returns ``{"local_ms", "rank_mean_ms", "rank_max_ms"}`` — the canonical shape every per-rank
    timing metric in this framework is expressed in. Generalizes the "CUDA-graph replay + all-reduce
    mean/max" pattern (previously a private, operator-named ``_time_graph`` helper).
    """
    import torch

    if iters <= 0:
        raise ValueError(f"time_distributed: iters must be positive, got {iters}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    return {
        "local_ms": local_ms,
        "rank_mean_ms": reduce_scalar(local_ms, op="mean", device=device),
        "rank_max_ms": reduce_scalar(local_ms, op="max", device=device),
    }


def _get_path(record: Mapping[str, Any], dotted_path: str):
    node: Any = record
    for part in dotted_path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def merge_rank_records(
    records: Sequence[Mapping[str, Any]],
    metric_paths: Iterable[str],
    repetitions: Sequence[Sequence[Mapping[str, Any]]] | None = None,
) -> dict:
    """Merge N per-rank records (already gathered by the caller) into rank_mean/max/min/tail stats.

    ``records``: one dict per rank for a single run, e.g. ``[{"timing_ms": {"stage1": 0.42}}, ...]``.
    ``metric_paths``: dotted paths into each record, e.g. ``["timing_ms.stage1", "timing_ms.e2e"]``.
    ``repetitions``: OPTIONAL — a list of independent repetitions, each itself a list of per-rank
        records shaped like ``records``. When given, also emits ``*_runs``/``*_span_pct`` (the
        cross-repetition noise band), matching the field names already used by the one existing
        analysis artifact this framework generalizes (see README's "Ground truth" section).

    Returns ``{metric_path: {"rank_mean", "rank_max", "rank_min", "rank_tail_spread_pct",
    ["rank_mean_runs", "rank_max_runs", "rank_mean_span_pct", "rank_max_span_pct"]}}``.

    Never raises on a missing metric path in some ranks — those ranks are simply excluded from that
    metric's aggregate (recorded in ``missing_ranks``), so a partially-populated record set still
    produces a usable (if incomplete) report; callers/skills decide how to react to `missing_ranks`.
    """
    out: dict[str, dict] = {}
    for path in metric_paths:
        values = []
        missing_ranks = []
        for i, rec in enumerate(records):
            v = _get_path(rec, path)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                missing_ranks.append(i)
                continue
            values.append(float(v))
        entry: dict[str, Any] = {"missing_ranks": missing_ranks}
        if values:
            vmax, vmin = max(values), min(values)
            entry["rank_mean"] = sum(values) / len(values)
            entry["rank_max"] = vmax
            entry["rank_min"] = vmin
            entry["rank_tail_spread_pct"] = (vmax - vmin) / max(vmin, _EPS) * 100.0
        else:
            entry["rank_mean"] = None
            entry["rank_max"] = None
            entry["rank_min"] = None
            entry["rank_tail_spread_pct"] = None
        out[path] = entry

    if repetitions:
        for path in metric_paths:
            mean_runs = []
            max_runs = []
            for rep_records in repetitions:
                rep_merged = merge_rank_records(rep_records, [path])
                m = rep_merged[path]
                if m["rank_mean"] is not None:
                    mean_runs.append(m["rank_mean"])
                if m["rank_max"] is not None:
                    max_runs.append(m["rank_max"])
            entry = out[path]
            entry["rank_mean_runs"] = mean_runs
            entry["rank_max_runs"] = max_runs
            for key, runs in (("rank_mean_span_pct", mean_runs), ("rank_max_span_pct", max_runs)):
                if len(runs) >= 2:
                    rmax, rmin = max(runs), min(runs)
                    entry[key] = (rmax - rmin) / max(rmin, _EPS) * 100.0
                else:
                    entry[key] = 0.0 if runs else None
    return out


def classify_case_speedup(
    candidate_rank_max: float,
    baseline_rank_max: float,
    min_pct: float | None = None,
) -> dict:
    """Speedup of a candidate vs. a baseline rank-max latency, with an optional acceptance floor.

    Generalizes the "static per-(shape,route) minimum speedup table" pattern (previously a
    MegaMoE-specific ``PERF_GUARD_MIN_SPEEDUP`` dict keyed by (tokens, route)) into a stateless
    function: the caller supplies its own threshold (or omits it for an unguarded comparison).

    Returns ``{"speedup_pct", "status": "pass"|"fail"|"unguarded"}``.
    """
    if baseline_rank_max <= 0 or not math.isfinite(baseline_rank_max):
        raise ValueError(f"classify_case_speedup: baseline_rank_max must be positive finite, got {baseline_rank_max}")
    speedup_pct = (baseline_rank_max / candidate_rank_max - 1.0) * 100.0
    if min_pct is None:
        status = "unguarded"
    else:
        status = "pass" if speedup_pct >= min_pct else "fail"
    return {"speedup_pct": speedup_pct, "status": status}
