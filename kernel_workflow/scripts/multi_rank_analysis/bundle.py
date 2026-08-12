"""Versioned analysis-bundle contract joining Profile artifacts to analysis runners."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .experiments import validate_experiment_manifest
from .hardware import validate_hardware_context
from .provenance import validate_collection_provenance

ANALYSIS_BUNDLE_SCHEMA_VERSION = "geak-analysis-bundle-v1"

__all__ = [
    "ANALYSIS_BUNDLE_SCHEMA_VERSION",
    "bundle_from_rank_report",
    "validate_analysis_bundle",
    "validate_measurement_tracks",
]


def _case_id(case: Mapping[str, Any], index: int) -> str:
    value = case.get("case_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"case {index} must have a non-empty case_id")
    return value


def bundle_from_rank_report(
    report: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    metric_paths: Sequence[str],
    workload: Mapping[str, Any] | None = None,
) -> dict:
    """Normalize existing UT shapes, including AITER ``cases[].ranks[]``."""
    if isinstance(report, list):
        report_mapping: Mapping[str, Any] = {}
        bare_records = report
    elif isinstance(report, Mapping):
        report_mapping = report
        bare_records = None
    else:
        raise TypeError("rank report must be a mapping or record list")
    metrics = [str(path) for path in metric_paths]
    if not metrics:
        raise ValueError("metric_paths must not be empty")
    if bare_records is not None:
        raw_cases = [
            {
                "case_id": "default",
                "rank_records": bare_records,
                "repetitions": None,
                "metric_paths": metrics,
            }
        ]
    elif isinstance(report_mapping.get("records"), list):
        raw_cases = [
            {
                "case_id": report_mapping.get("case_id", "default"),
                "rank_records": report_mapping["records"],
                "repetitions": report_mapping.get("repetitions"),
                "metric_paths": metrics,
            }
        ]
    else:
        cases = report_mapping.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError("rank report must contain records[] or cases[]")
        raw_cases = []
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                raise TypeError(f"case {index} must be a mapping")
            ranks = case.get("ranks")
            if not isinstance(ranks, list):
                raise ValueError(
                    f"case {_case_id(case, index)!r} must contain ranks[]"
                )
            raw_cases.append(
                {
                    "case_id": _case_id(case, index),
                    "rank_records": ranks,
                    "metric_paths": metrics,
                    "source_case": {
                        key: value
                        for key, value in case.items()
                        if key != "ranks"
                    },
                }
            )
    bundle = {
        "schema_version": ANALYSIS_BUNDLE_SCHEMA_VERSION,
        "workload": dict(
            workload
            or report_mapping.get("metadata")
            or {"source": "rank_report"}
        ),
        "cases": raw_cases,
        "route_comparisons": [],
        "measurement_tracks": {},
        "artifacts": {},
    }
    return validate_analysis_bundle(bundle)


def validate_measurement_tracks(tracks: Mapping[str, Any]) -> dict:
    if not isinstance(tracks, Mapping):
        raise ValueError("measurement_tracks must be a mapping")
    for name, track in tracks.items():
        if not isinstance(track, Mapping):
            raise ValueError(f"measurement track {name!r} must be a mapping")
        if track.get("status") not in ("complete", "partial", "missing"):
            raise ValueError(
                f"measurement track {name!r} status must be complete, partial, or missing"
            )
        if track.get("status") == "complete" and not track.get("evidence"):
            raise ValueError(
                f"measurement track {name!r} complete status requires evidence"
            )
        if track.get("status") == "complete":
            evidence = track["evidence"]
            if not isinstance(evidence, Mapping):
                raise ValueError(
                    f"measurement track {name!r} evidence must be a mapping"
                )
            for field in ("artifact_refs", "metrics", "provenance_refs"):
                if not evidence.get(field):
                    raise ValueError(
                        f"measurement track {name!r} evidence requires {field}"
                    )
    return {str(name): dict(track) for name, track in tracks.items()}


def validate_analysis_bundle(bundle: Mapping[str, Any]) -> dict:
    if not isinstance(bundle, Mapping):
        raise TypeError("analysis bundle must be a mapping")
    if bundle.get("schema_version") != ANALYSIS_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported analysis bundle schema: {bundle.get('schema_version')!r}"
        )
    workload = bundle.get("workload")
    if not isinstance(workload, Mapping) or not workload:
        raise ValueError("analysis bundle workload must be a non-empty mapping")
    cases = bundle.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("analysis bundle cases must be a non-empty list")
    normalized_cases = []
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TypeError(f"case {index} must be a mapping")
        case_id = _case_id(case, index)
        if case_id in seen:
            raise ValueError(f"duplicate case_id={case_id!r}")
        seen.add(case_id)
        records = case.get("rank_records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"case {case_id!r} must contain rank_records[]")
        metric_paths = case.get("metric_paths")
        if not isinstance(metric_paths, list) or not metric_paths:
            raise ValueError(f"case {case_id!r} must contain metric_paths[]")
        if not all(isinstance(path, str) and path for path in metric_paths):
            raise ValueError(
                f"case {case_id!r} metric_paths entries must be non-empty strings"
            )
        repetitions = case.get("repetitions")
        if repetitions is not None and not isinstance(repetitions, list):
            raise ValueError(f"case {case_id!r} repetitions must be a list")
        trace_files = list(case.get("trace_files", []))
        att_files = list(case.get("att_stats_files", []))
        route_summary = dict(case.get("route_summary", {}))
        software_counters = dict(case.get("software_counters", {}))
        provenance_fields = {}
        for artifact_name, present in (
            ("trace", bool(trace_files)),
            ("att", bool(att_files)),
            ("route", bool(route_summary)),
        ):
            field = f"{artifact_name}_provenance"
            value = case.get(field)
            if present:
                provenance_fields[field] = validate_collection_provenance(value)
                files = {
                    "trace": trace_files,
                    "att": att_files,
                    "route": [],
                }[artifact_name]
                if files:
                    raw_paths = set(provenance_fields[field]["raw_artifacts"])
                    unprovenanced = [
                        str(path) for path in files if str(path) not in raw_paths
                    ]
                    if unprovenanced:
                        raise ValueError(
                            f"case {case_id!r} {artifact_name} files missing from "
                            f"provenance raw_artifacts: {unprovenanced}"
                        )
            elif value is not None:
                provenance_fields[field] = validate_collection_provenance(value)
        if software_counters:
            if not isinstance(software_counters.get("records"), list):
                raise ValueError(
                    f"case {case_id!r} software_counters.records must be a list"
                )
            software_counters["provenance"] = validate_collection_provenance(
                software_counters.get("provenance")
            )
        normalized_cases.append(
            {
                **dict(case),
                "case_id": case_id,
                "rank_records": list(records),
                "metric_paths": list(metric_paths),
                "repetitions": repetitions,
                "trace_files": trace_files,
                "att_stats_files": att_files,
                "software_counters": software_counters,
                "route_summary": route_summary,
                **provenance_fields,
            }
        )
    comparisons = bundle.get("route_comparisons", [])
    if not isinstance(comparisons, list):
        raise ValueError("route_comparisons must be a list")
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise TypeError("route comparison must be a mapping")
        for field in ("baseline_case_id", "candidate_case_id"):
            value = comparison.get(field)
            if value not in seen:
                raise ValueError(
                    f"route comparison {field} references unknown case {value!r}"
                )
        metric_path = comparison.get("metric_path")
        if not isinstance(metric_path, str) or not metric_path:
            raise ValueError("route comparison metric_path must be a non-empty string")
        tokens = comparison.get("tokens_per_rank")
        if tokens is not None and (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens <= 0
        ):
            raise ValueError(
                "route comparison tokens_per_rank must be a positive integer or null"
            )
    tracks = validate_measurement_tracks(bundle.get("measurement_tracks", {}))
    artifacts = bundle.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be a mapping")
    hardware_context = (
        validate_hardware_context(bundle["hardware_context"])
        if bundle.get("hardware_context") is not None
        else None
    )
    experiment_manifest = (
        validate_experiment_manifest(bundle["experiment_manifest"])
        if bundle.get("experiment_manifest") is not None
        else None
    )
    return {
        **dict(bundle),
        "workload": dict(workload),
        "cases": normalized_cases,
        "route_comparisons": [dict(item) for item in comparisons],
        "measurement_tracks": tracks,
        "artifacts": dict(artifacts),
        **(
            {"hardware_context": hardware_context}
            if hardware_context is not None
            else {}
        ),
        **(
            {"experiment_manifest": experiment_manifest}
            if experiment_manifest is not None
            else {}
        ),
    }
