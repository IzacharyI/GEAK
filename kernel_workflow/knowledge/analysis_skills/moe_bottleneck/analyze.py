#!/usr/bin/env python3
"""Deterministic measurement-first MoE analysis generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from multi_rank_analysis import validate_hardware_context  # noqa: E402

SCHEMA_VERSION = "geak-moe-bottleneck-analysis-v3"
REQUIRED_TRACKS = (
    "ep_baseline_decomposition",
    "communication_bytes",
    "wait_padding",
    "publication_granularity",
    "fusion_dag",
    "resource_residency",
)
REQUIRED_EVIDENCE_FIELDS = (
    "artifact_refs",
    "metrics",
    "provenance_refs",
)
CATEGORY_LABELS = {
    "stage1": "stage1_dispatch_gemm1",
    "stage1_dispatch_gemm1": "stage1_dispatch_gemm1",
    "stage2": "stage2",
    "combine": "combine",
    "quantize": "quantize",
}
REFERENCE_PATTERNS = [
    {
        "name": "DeepGEMM Mega MoE",
        "mechanism": "symmetric workspace, arrival counts, expert-wave persistent scheduler",
        "applicability": "dependency and scheduler reference",
        "caveat": "Blackwell SM100 implementation details do not transfer to gfx950",
        "source": "https://github.com/deepseek-ai/DeepGEMM",
    },
    {
        "name": "DeepEP SBO",
        "mechanism": "block-granular producer signals consumed by combine send",
        "applicability": "readiness/publication protocol reference",
        "caveat": "CUDA/NVSHMEM memory ordering must be revalidated on MORI/gfx950",
        "source": "https://github.com/deepseek-ai/DeepEP/pull/483",
    },
    {
        "name": "Comet / Flux",
        "mechanism": "shared-tensor decomposition, work reordering, block specialization",
        "applicability": "candidate pattern catalog",
        "caveat": "CUTLASS/NVSHMEM implementation and speedups are not local evidence",
        "source": "https://arxiv.org/abs/2502.19811",
    },
    {
        "name": "tile-level producer/consumer overlap",
        "mechanism": "persistent GEMM producer, communication consumer, epilogue signals",
        "applicability": "candidate protocol experiment",
        "caveat": "A100 evidence; no dedicated public ROCm implementation identified",
        "source": "https://arxiv.org/abs/2607.19539",
    },
]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _finite(value, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _case_map(report: dict) -> dict:
    cases = report.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("report.cases must be a list")
    result = {}
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case must have a non-empty case_id")
        if case_id in result:
            raise ValueError(f"duplicate case_id={case_id!r}")
        result[case_id] = case
    return result


def _comparison_evidence(
    comparison: dict,
    default_timing_confidence: str,
) -> dict:
    tokens = comparison.get("tokens_per_rank")
    timing_confidence = comparison.get(
        "timing_confidence",
        default_timing_confidence,
    )
    if timing_confidence not in ("high", "medium", "low"):
        timing_confidence = "low"
    profile_confidence = comparison.get(
        "profile_confidence",
        default_timing_confidence,
    )
    if profile_confidence not in ("high", "medium", "low"):
        profile_confidence = "low"
    profile = comparison.get("profile_category_delta") or {}
    categories = {}
    positive_total = 0.0
    for category, values in profile.items():
        delta = _finite(values.get("rank_max_delta_ms"), f"{category}.rank_max_delta_ms")
        relative = _finite(
            values.get("rank_max_delta_pct"),
            f"{category}.rank_max_delta_pct",
        )
        positive_total += max(delta, 0.0)
        categories[category] = {
            "label": CATEGORY_LABELS.get(category, category),
            "absolute_delta_ms": delta,
            "relative_growth_pct": relative,
        }
    for values in categories.values():
        values["positive_absolute_delta_share_pct"] = (
            max(values["absolute_delta_ms"], 0.0) / positive_total * 100.0
            if positive_total
            else 0.0
        )
    return {
        "tokens_per_rank": tokens,
        "e2e_rank_max_delta_ms": _finite(
            comparison.get("e2e_rank_max_delta_ms", 0.0),
            "e2e_rank_max_delta_ms",
        ),
        "e2e_rank_max_delta_pct": _finite(
            comparison.get("e2e_rank_max_delta_pct", 0.0),
            "e2e_rank_max_delta_pct",
        ),
        "categories": categories,
        "timing_confidence": timing_confidence,
        "profile_confidence": profile_confidence,
        "stage_deltas_non_additive": True,
    }


def _measurement_coverage(report: dict) -> tuple[dict, list[str]]:
    raw = report.get("measurement_tracks") or {}
    coverage = {}
    missing = []
    for track in REQUIRED_TRACKS:
        value = raw.get(track)
        status = value.get("status") if isinstance(value, dict) else value
        evidence = value.get("evidence") if isinstance(value, dict) else None
        reason = None
        if status not in ("complete", "partial", "missing"):
            reason = "status must be complete, partial, or missing"
            status = "invalid" if value is not None else "missing"
        if status == "complete" and not evidence:
            reason = "complete requires non-empty evidence"
            status = "invalid"
        if status == "complete":
            if not isinstance(evidence, dict):
                reason = "complete evidence must be a mapping"
                status = "invalid"
            else:
                absent = [
                    field
                    for field in REQUIRED_EVIDENCE_FIELDS
                    if not evidence.get(field)
                ]
                if absent:
                    reason = (
                        "complete evidence is missing required fields: "
                        + ", ".join(absent)
                    )
                    status = "invalid"
                elif track == "fusion_dag" and not evidence.get("bounds"):
                    reason = "complete fusion_dag evidence requires bounds"
                    status = "invalid"
        coverage[track] = {"status": status, "evidence": evidence}
        if reason:
            coverage[track]["reason"] = reason
        if status != "complete":
            missing.append(track)
    return coverage, missing


def _hardware_guidance(context: dict | None) -> dict:
    if context is None:
        return {
            "status": "missing",
            "required": (
                "device/topology/runtime/capability context and measured hardware ceilings"
            ),
            "constraints": [],
        }
    validated = validate_hardware_context(context)
    measured = validated.get("measured", {})
    constraints = [
        (
            f"Interpret evidence for {validated['vendor']} {validated['model']} "
            f"({validated['arch']}), not for a reference system's hardware."
        ),
        (
            f"Report residency bounds with {validated['execution_units_per_device']} execution "
            f"units/device, thread-group width {validated['thread_group_width']}, and "
            f"{validated['local_memory_bytes_per_execution_unit']} bytes local memory/unit."
        ),
        (
            f"Interpret communication against {validated['interconnect']['type']} "
            f"{validated['interconnect']['topology']} topology."
        ),
    ]
    missing_ceilings = [
        name
        for name in (
            "pairwise_interconnect_gbps",
            "all_to_all_interconnect_gbps",
            "device_memory_gbps",
            "launch_overhead_us",
        )
        if measured.get(name) is None
    ]
    provenance = validated.get("provenance", {})
    low_confidence_fields = sorted(
        field
        for field, entry in provenance.items()
        if entry.get("confidence") == "low"
    )
    medium_confidence_fields = sorted(
        field
        for field, entry in provenance.items()
        if entry.get("confidence") == "medium"
    )
    return {
        "status": (
            "ready"
            if not missing_ceilings and not low_confidence_fields
            else "partial"
        ),
        "context": validated,
        "missing_measured_ceilings": missing_ceilings,
        "low_confidence_fields": low_confidence_fields,
        "medium_confidence_fields": medium_confidence_fields,
        "constraints": constraints,
    }


def _measurement_unknowns(missing: list[str]) -> list[dict]:
    descriptions = {
        "ep_baseline_decomposition": (
            "Decompose the frozen scattered EP kernel into quant, dispatch/planning, "
            "GEMM1, activation, GEMM2/publication, combine, launch/materialization, "
            "rank wait, and physical overlap."
        ),
        "communication_bytes": (
            "Collect per-source/destination payload and metadata bytes, XGMI samples, "
            "and full/no-payload/local-loopback controls."
        ),
        "wait_padding": (
            "Instrument per-expert/tile readiness, local wait cycles, useful rows, "
            "padded rows, and the per-rank expert-load matrix."
        ),
        "publication_granularity": (
            "Sweep token, owner-aligned row band, GEMM tile, and chunk publication "
            "granularity with signal/fence and XGMI efficiency."
        ),
        "fusion_dag": (
            "Build the measured dependency DAG and Amdahl ceiling for replacing the "
            "scattered launches with one complete persistent kernel per rank."
        ),
        "resource_residency": (
            "Measure common workgroup shape, LDS/VGPR union, resident CTA count, "
            "role liveness, scheduler-state pressure, and termination for the one-kernel target."
        ),
    }
    return [
        {
            "track": track,
            "question": descriptions[track],
            "required_data": f"measurement_tracks.{track}.evidence",
            "blocks": ["root_cause", "candidate_ranking"],
        }
        for track in missing
    ]


def build_analysis(report: dict) -> dict:
    _case_map(report)
    report_inputs = report.get("input_artifacts")
    upstream_identity_available = (
        isinstance(report_inputs, dict)
        and bool(report_inputs)
        and all(
            isinstance(identity, dict) and bool(identity.get("sha256"))
            for identity in report_inputs.values()
        )
    )
    source_provenance = {
        "status": (
            "upstream_input_identity_available"
            if upstream_identity_available
            else (
                "upstream_input_identity_incomplete"
                if isinstance(report_inputs, dict) and report_inputs
                else "legacy_upstream_identity_missing"
            )
        ),
        "input_artifacts": (
            dict(report_inputs)
            if isinstance(report_inputs, dict)
            else {}
        ),
        "raw_artifact_identities": dict(
            report.get("raw_artifact_identities") or {}
        ),
    }
    default_timing_confidence = (
        "medium"
        if source_provenance["status"] == "upstream_input_identity_available"
        else "low"
    )
    comparisons = [
        _comparison_evidence(comparison, default_timing_confidence)
        for comparison in report.get("route_comparisons", [])
    ]
    coverage, missing = _measurement_coverage(report)
    hardware = _hardware_guidance(report.get("hardware_context"))
    unknowns = _measurement_unknowns(missing)
    if hardware["status"] != "ready":
        unknowns.insert(
            0,
            {
                "track": "hardware_context",
                "question": (
                    "Complete target-hardware provenance and measured ceilings before "
                    "interpreting utilization or hardware-sensitive reference patterns."
                ),
                "required_data": hardware.get("missing_measured_ceilings", []),
                "blocks": ["utilization_interpretation", "candidate_applicability"],
            },
        )
    findings = []
    hypotheses = []
    for comparison in comparisons:
        token_scope = (
            f"{comparison['tokens_per_rank']} tokens/rank"
            if comparison["tokens_per_rank"] is not None
            else "an unspecified token count"
        )
        findings.append(
            {
                "observation": (
                    f"At {token_scope}, the compared case "
                    f"changes E2E rank-max by {comparison['e2e_rank_max_delta_pct']:.3f}%."
                ),
                "evidence": {
                    "tokens_per_rank": comparison["tokens_per_rank"],
                    "e2e_rank_max_delta_ms": comparison["e2e_rank_max_delta_ms"],
                    "e2e_rank_max_delta_pct": comparison["e2e_rank_max_delta_pct"],
                },
                "confidence": comparison["timing_confidence"],
                "scope": "tested route-case pair only",
            }
        )
        categories = comparison["categories"]
        if not categories:
            continue
        largest_absolute = max(
            categories.items(),
            key=lambda item: item[1]["positive_absolute_delta_share_pct"],
        )
        largest_relative = max(
            categories.items(),
            key=lambda item: item[1]["relative_growth_pct"],
        )
        findings.append(
            {
                "observation": (
                    f"At {token_scope}, "
                    f"{largest_absolute[1]['label']} has the largest positive absolute "
                    f"category-delta share, while {largest_relative[1]['label']} has the "
                    "largest relative growth."
                ),
                "evidence": {
                    "largest_absolute_category": largest_absolute[1]["label"],
                    "positive_absolute_delta_share_pct": largest_absolute[1][
                        "positive_absolute_delta_share_pct"
                    ],
                    "relative_growth_leader": largest_relative[1]["label"],
                    "relative_growth_pct": largest_relative[1]["relative_growth_pct"],
                    "e2e_rank_max_delta_pct": comparison["e2e_rank_max_delta_pct"],
                },
                "confidence": comparison["profile_confidence"],
                "scope": "profiler categories; not critical-path contribution",
            }
        )
        hypotheses.append(
            {
                "statement": (
                    f"Internal work or synchronization within "
                    f"{largest_absolute[1]['label']} may contribute materially to the "
                    "observed route sensitivity."
                ),
                "supports": findings[-1]["evidence"],
                "does_not_prove": [
                    "root cause",
                    "dispatch cost when fused with GEMM1",
                    "a preferred fusion or scheduling mechanism",
                ],
                "discriminating_measurement": (
                    "instrument internal readiness, wait, bytes, instruction stalls, "
                    "and physical overlap for this category"
                ),
                "confidence": "low" if missing else "medium",
            }
        )
    constraints = list(hardware.get("constraints", []))
    constraints.extend(
        [
            "Stage1 timing combines dispatch/planning/P2P publication with GEMM1.",
            "Separately captured category deltas are non-additive when execution overlaps.",
            "Public implementation mechanisms are references, not selected solutions.",
        ]
    )
    fusion_track = (report.get("measurement_tracks") or {}).get("fusion_dag")
    bounds = []
    if isinstance(fusion_track, dict):
        evidence = fusion_track.get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("bounds"), list):
            bounds = list(evidence["bounds"])
    degraded = []
    if missing or hardware["status"] != "ready":
        degraded.append(
            {
                "reason": "analysis prerequisites are incomplete",
                "missing_tracks": missing,
                "hardware_context_status": hardware["status"],
                "effect": "no high-confidence fusion or root-cause verdict is allowed",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "framework_status": "ready",
        "analysis_boundary": {
            "workflow_step": 2,
            "emits": [
                "findings",
                "hypotheses",
                "constraints",
                "bounds",
                "unknowns",
                "reference_patterns",
            ],
            "decision_fields_emitted": False,
            "decision_owner": "Step-3 TechLead",
        },
        "analysis_status": (
            "awaiting_measurement"
            if missing or hardware["status"] != "ready"
            else "evidence_complete"
        ),
        "source_schema": report.get("schema_version", "unknown"),
        "source_provenance": source_provenance,
        "hardware_guidance": hardware,
        "measurement_coverage": coverage,
        "route_comparisons": comparisons,
        "findings": findings,
        "hypotheses": hypotheses,
        "constraints": constraints,
        "bounds": bounds,
        "unknowns": unknowns,
        "reference_patterns": list(REFERENCE_PATTERNS),
        "degraded": degraded,
        "claims": {
            "root_cause_proven": (
                bool(report.get("root_cause_proven"))
                if not missing and hardware["status"] == "ready"
                else False
            ),
            "dispatch_independently_measured": bool(
                report.get("dispatch_independent_measurement")
            ),
        },
    }


def _render_markdown(analysis: dict) -> str:
    lines = [
        "# MoE bottleneck analysis",
        "",
        f"Framework status: `{analysis['framework_status']}`",
        f"Analysis status: `{analysis['analysis_status']}`",
        "",
        "## Source provenance",
        "",
        f"- upstream report provenance: {analysis['source_provenance']['status']}",
    ]
    for name, identity in analysis.get("analysis_inputs", {}).items():
        lines.append(
            f"- {name}: `{identity['path']}` / sha256 `{identity['sha256']}`"
        )
    lines.extend(["", "## Hardware context", ""])
    hardware = analysis["hardware_guidance"]
    lines.append(f"- status: {hardware['status']}")
    context = hardware.get("context")
    if context:
        lines.append(
            f"- target: {context['vendor']} {context['model']} / {context['arch']} / "
            f"{context['device_count']} devices"
        )
        for constraint in hardware["constraints"]:
            lines.append(f"- {constraint}")
    if hardware.get("missing_measured_ceilings"):
        lines.append(
            "- missing measured ceilings: "
            + ", ".join(hardware["missing_measured_ceilings"])
        )
    if hardware.get("medium_confidence_fields"):
        lines.append(
            "- medium-confidence provenance: "
            + ", ".join(hardware["medium_confidence_fields"])
        )
    if hardware.get("low_confidence_fields"):
        lines.append(
            "- low-confidence provenance: "
            + ", ".join(hardware["low_confidence_fields"])
        )
    lines.extend(["", "## Measurement coverage", ""])
    for track, details in analysis["measurement_coverage"].items():
        lines.append(f"- `{track}`: {details['status']}")
    lines.extend(["", "## Findings", ""])
    for finding in analysis["findings"]:
        lines.append(
            f"- [{finding['confidence']}] {finding['observation']} "
            f"(scope: {finding['scope']})"
        )
    lines.extend(["", "## Hypotheses", ""])
    for hypothesis in analysis["hypotheses"]:
        lines.append(
            f"- [{hypothesis['confidence']}] {hypothesis['statement']} "
            f"Next measurement: {hypothesis['discriminating_measurement']}"
        )
    lines.extend(["", "## Constraints", ""])
    for constraint in analysis["constraints"]:
        lines.append(f"- {constraint}")
    lines.extend(["", "## Unknowns / collection plan", ""])
    for unknown in analysis["unknowns"]:
        lines.append(
            f"- `{unknown['track']}`: {unknown['question']}"
        )
    lines.extend(["", "## Reference patterns (unranked)", ""])
    for pattern in analysis["reference_patterns"]:
        lines.append(
            f"- {pattern['name']}: {pattern['mechanism']} "
            f"(caveat: {pattern['caveat']}; source: {pattern['source']})"
        )
    if analysis["degraded"]:
        lines.extend(["", "## Degraded", ""])
        for entry in analysis["degraded"]:
            lines.append(f"- {entry['reason']}: {', '.join(entry['missing_tracks'])}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    report = _read_json(args.report)
    if args.hardware_context:
        report["hardware_context"] = _read_json(args.hardware_context)
    analysis = build_analysis(report)
    analysis["analysis_inputs"] = {
        "report": _file_identity(args.report),
        **(
            {"hardware_context": _file_identity(args.hardware_context)}
            if args.hardware_context
            else {}
        ),
    }
    _write_json(args.output, analysis)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            _render_markdown(analysis),
            encoding="utf-8",
        )
    print(f"GEAK_MOE_ANALYSIS_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
