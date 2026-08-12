#!/usr/bin/env python3
"""Deterministic CLI for generic multi-rank record and trace aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

if __package__:
    from .aggregate import merge_rank_records
    from .bundle import (
        bundle_from_rank_report,
        validate_analysis_bundle,
        validate_measurement_tracks,
    )
    from .experiments import validate_experiment_manifest
    from .hardware import validate_hardware_context
    from .instruction_analysis import (
        load_instruction_category_map,
        parse_att_stats_csv,
    )
    from .provenance import validate_collection_provenance
    from .intervals import analyze_category_overlap
    from .schema import build_report
    from .trace_categories import bucket_trace_events, load_category_map
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from multi_rank_analysis.aggregate import merge_rank_records
    from multi_rank_analysis.bundle import (
        bundle_from_rank_report,
        validate_analysis_bundle,
        validate_measurement_tracks,
    )
    from multi_rank_analysis.experiments import validate_experiment_manifest
    from multi_rank_analysis.hardware import validate_hardware_context
    from multi_rank_analysis.instruction_analysis import (
        load_instruction_category_map,
        parse_att_stats_csv,
    )
    from multi_rank_analysis.provenance import validate_collection_provenance
    from multi_rank_analysis.intervals import analyze_category_overlap
    from multi_rank_analysis.schema import build_report
    from multi_rank_analysis.trace_categories import (
        bucket_trace_events,
        load_category_map,
    )

_RANK_RE = re.compile(r"(?:^|[_-])rank[_-]?(\d+)(?:\D|$)")


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


def _file_identity_or_error(path: Path) -> dict:
    try:
        return {"status": "available", **_file_identity(path)}
    except OSError as error:
        return {
            "status": "unavailable",
            "path": str(path.resolve()),
            "error": str(error),
        }


def _rank_from_name(path: Path, fallback: int) -> int:
    match = _RANK_RE.search(path.name)
    return int(match.group(1)) if match else fallback


def _aggregate_trace_files(
    trace_files: list[Path],
    category_map,
    expected_ranks: list[int] | None,
) -> dict:
    if not trace_files:
        raise ValueError("no trace files were provided")
    trace_records = []
    per_rank_overlap = {}
    errors = []
    for index, path in enumerate(trace_files):
        rank = _rank_from_name(path, index)
        bucketed = bucket_trace_events(str(path), category_map)
        if bucketed.get("error"):
            errors.append({"rank": rank, "path": str(path), "error": bucketed["error"]})
            continue
        trace_records.append(
            {
                "rank": rank,
                "per_category_ms": bucketed["per_category_ms"],
                "per_category_event_count": bucketed["per_category_event_count"],
            }
        )
        per_rank_overlap[str(rank)] = analyze_category_overlap(
            bucketed["per_category_intervals_us"]
        )
        if bucketed.get("malformed_events"):
            errors.append(
                {
                    "rank": rank,
                    "path": str(path),
                    "malformed_events": bucketed["malformed_events"],
                }
            )
    categories = sorted(
        {
            category
            for record in trace_records
            for category in record["per_category_ms"]
        }
    )
    metrics = [f"per_category_ms.{category}" for category in categories]
    return {
        "trace_files": [str(path) for path in trace_files],
        "categories": merge_rank_records(
            trace_records,
            metrics,
            expected_ranks=expected_ranks,
        ),
        "per_rank_overlap": per_rank_overlap,
        "errors": errors,
    }


def _aggregate_traces(
    trace_dir: Path,
    trace_glob: str,
    category_map,
    expected_ranks: list[int] | None,
) -> dict:
    trace_files = sorted(trace_dir.glob(trace_glob))
    if not trace_files:
        raise ValueError(f"no trace files matched {trace_dir / trace_glob}")
    return _aggregate_trace_files(trace_files, category_map, expected_ranks)


def _aggregate_att(att_files: list[Path], instruction_map) -> dict:
    return {
        "files": [str(path) for path in att_files],
        "reports": [
            parse_att_stats_csv(path, instruction_map)
            for path in att_files
        ],
        "scope_warning": (
            "ATT reports sampled instruction-pipeline evidence; they are not "
            "whole-device E2E time or cross-GPU traffic."
        ),
    }


def _require_provenanced_paths(
    provenance: dict,
    paths: list[Path],
    artifact_kind: str,
) -> None:
    raw_paths = set(provenance["raw_artifacts"])
    missing = [str(path) for path in paths if str(path) not in raw_paths]
    if missing:
        raise ValueError(
            f"{artifact_kind} files missing from provenance raw_artifacts: {missing}"
        )


def _build_route_comparisons(bundle: dict, cases: list[dict]) -> list[dict]:
    by_id = {case["case_id"]: case for case in cases}
    output = []
    for spec in bundle.get("route_comparisons", []):
        baseline = by_id[spec["baseline_case_id"]]
        candidate = by_id[spec["candidate_case_id"]]
        metric_path = spec.get("metric_path")
        if not metric_path:
            raise ValueError("route comparison must define metric_path")
        baseline_metric = baseline["rank_metrics"].get(metric_path)
        candidate_metric = candidate["rank_metrics"].get(metric_path)
        if not baseline_metric or not candidate_metric:
            raise ValueError(
                f"route comparison metric {metric_path!r} missing from cases"
            )
        baseline_max = baseline_metric["rank_max"]
        candidate_max = candidate_metric["rank_max"]
        if not baseline_max or baseline_max <= 0:
            raise ValueError("route comparison baseline rank_max must be positive")
        if candidate_max is None or candidate_max <= 0:
            raise ValueError("route comparison candidate rank_max must be positive")
        comparison = {
            **dict(spec),
            "e2e_rank_max_delta_ms": candidate_max - baseline_max,
            "e2e_rank_max_delta_pct": (
                candidate_max / baseline_max - 1.0
            )
            * 100.0,
        }
        baseline_categories = baseline.get("trace", {}).get("categories", {})
        candidate_categories = candidate.get("trace", {}).get("categories", {})
        category_delta = {}
        for path in sorted(set(baseline_categories) & set(candidate_categories)):
            before = baseline_categories[path]["rank_max"]
            after = candidate_categories[path]["rank_max"]
            if before is None or after is None or before <= 0:
                continue
            category = path.removeprefix("per_category_ms.")
            category_delta[category] = {
                "rank_max_delta_ms": after - before,
                "rank_max_delta_pct": (after / before - 1.0) * 100.0,
            }
        if category_delta:
            comparison["profile_category_delta"] = category_delta
        output.append(comparison)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rank-records", type=Path)
    source.add_argument("--analysis-bundle", type=Path)
    parser.add_argument("--metric", action="append")
    parser.add_argument("--case-id")
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--trace-glob", default="*.json")
    parser.add_argument("--category-map")
    parser.add_argument("--trace-provenance", type=Path)
    parser.add_argument("--att-stats", type=Path, action="append")
    parser.add_argument("--instruction-map")
    parser.add_argument("--att-provenance", type=Path)
    parser.add_argument("--hardware-context", type=Path)
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument("--measurement-tracks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if args.trace_dir and not args.trace_provenance:
        parser.error("--trace-provenance is required with --trace-dir")
    if args.trace_provenance and not args.trace_dir:
        parser.error("--trace-provenance requires --trace-dir")
    if args.att_stats and not args.att_provenance:
        parser.error("--att-provenance is required with --att-stats")
    if args.att_provenance and not args.att_stats:
        parser.error("--att-provenance requires --att-stats")
    if args.analysis_bundle:
        bundle = validate_analysis_bundle(_read_json(args.analysis_bundle))
    else:
        if not args.metric:
            parser.error("--metric is required with --rank-records")
        raw_rank_report = _read_json(args.rank_records)
        bundle = bundle_from_rank_report(
            raw_rank_report,
            args.metric,
        )
        if args.case_id and (
            isinstance(raw_rank_report, list)
            or (isinstance(raw_rank_report, dict) and "records" in raw_rank_report)
        ):
            bundle["cases"][0]["case_id"] = args.case_id
    if args.case_id:
        selected = [
            case for case in bundle["cases"] if case["case_id"] == args.case_id
        ]
        if not selected:
            parser.error(f"--case-id {args.case_id!r} was not found")
        bundle = {**bundle, "cases": selected, "route_comparisons": []}
    if (args.trace_dir or args.att_stats) and len(bundle["cases"]) != 1:
        parser.error(
            "CLI trace/ATT overrides require exactly one selected case; "
            "use per-case bundle fields otherwise"
        )
    expected_ranks = (
        list(range(args.expected_world_size))
        if args.expected_world_size is not None
        else None
    )
    category_map_path = Path(
        args.category_map
        or str(Path(__file__).with_name("default_category_map.json"))
    )
    instruction_map_path = Path(
        args.instruction_map
        or str(Path(__file__).with_name("default_instruction_map.json"))
    )
    category_map = load_category_map(str(category_map_path))
    instruction_map = load_instruction_category_map(str(instruction_map_path))
    cli_trace_files = (
        sorted(args.trace_dir.glob(args.trace_glob))
        if args.trace_dir
        else []
    )
    if args.trace_dir and not cli_trace_files:
        parser.error(f"no trace files matched {args.trace_dir / args.trace_glob}")
    cases = []
    all_trace_files = []
    all_att_files = []
    for index, case_spec in enumerate(bundle["cases"]):
        case = {
            "case_id": case_spec["case_id"],
            "rank_metrics": merge_rank_records(
                case_spec["rank_records"],
                case_spec["metric_paths"],
                repetitions=case_spec.get("repetitions"),
                expected_ranks=expected_ranks,
            ),
            "route_summary": case_spec.get("route_summary", {}),
            "software_counters": case_spec.get("software_counters", {}),
        }
        trace_files = [Path(path) for path in case_spec.get("trace_files", [])]
        if args.trace_dir and index == 0:
            trace_files.extend(cli_trace_files)
        all_trace_files.extend(trace_files)
        if trace_files:
            case["trace"] = _aggregate_trace_files(
                sorted(set(trace_files)),
                category_map,
                expected_ranks,
            )
            trace_provenance = case_spec.get("trace_provenance")
            if args.trace_provenance and index == 0:
                trace_provenance = validate_collection_provenance(
                    _read_json(args.trace_provenance)
                )
            if trace_provenance:
                _require_provenanced_paths(
                    trace_provenance,
                    sorted(set(trace_files)),
                    "trace",
                )
                case["trace"]["provenance"] = trace_provenance
        att_files = [Path(path) for path in case_spec.get("att_stats_files", [])]
        if args.att_stats and index == 0:
            att_files.extend(args.att_stats)
        all_att_files.extend(att_files)
        if att_files:
            case["att"] = _aggregate_att(
                sorted(set(att_files)),
                instruction_map,
            )
            att_provenance = case_spec.get("att_provenance")
            if args.att_provenance and index == 0:
                att_provenance = validate_collection_provenance(
                    _read_json(args.att_provenance)
                )
            if att_provenance:
                _require_provenanced_paths(
                    att_provenance,
                    sorted(set(att_files)),
                    "ATT",
                )
                case["att"]["provenance"] = att_provenance
        if case_spec.get("route_provenance"):
            case["route_provenance"] = case_spec["route_provenance"]
        if case_spec.get("source_case"):
            case["source_case"] = case_spec["source_case"]
        cases.append(case)
    experiment_manifest_payload = (
        _read_json(args.experiment_manifest)
        if args.experiment_manifest
        else bundle.get("experiment_manifest")
    )
    experiment_manifest = (
        validate_experiment_manifest(experiment_manifest_payload)
        if experiment_manifest_payload
        else None
    )
    measurement_tracks_payload = (
        _read_json(args.measurement_tracks)
        if args.measurement_tracks
        else bundle.get("measurement_tracks")
    )
    measurement_tracks = (
        validate_measurement_tracks(measurement_tracks_payload)
        if measurement_tracks_payload
        else None
    )
    hardware_context = (
        validate_hardware_context(_read_json(args.hardware_context))
        if args.hardware_context
        else (
            validate_hardware_context(bundle["hardware_context"])
            if bundle.get("hardware_context")
            else None
        )
    )
    route_comparisons = _build_route_comparisons(bundle, cases)
    report = build_report(
        args.primary_metric,
        cases,
        route_comparisons=route_comparisons,
        hardware_context=hardware_context,
        measurement_tracks=measurement_tracks,
        experiment_manifest=experiment_manifest,
    )
    input_paths = {
        (
            "analysis_bundle" if args.analysis_bundle else "rank_records"
        ): args.analysis_bundle or args.rank_records,
        "category_map": category_map_path,
        "instruction_map": instruction_map_path,
    }
    for name, path in (
        ("hardware_context", args.hardware_context),
        ("experiment_manifest", args.experiment_manifest),
        ("measurement_tracks", args.measurement_tracks),
        ("trace_provenance", args.trace_provenance),
        ("att_provenance", args.att_provenance),
    ):
        if path is not None:
            input_paths[name] = path
    report["input_artifacts"] = {
        name: _file_identity_or_error(Path(path))
        for name, path in input_paths.items()
    }
    report["raw_artifact_identities"] = {
        "trace_files": [
            _file_identity_or_error(path)
            for path in sorted(set(all_trace_files))
        ],
        "att_stats_files": [
            _file_identity_or_error(path)
            for path in sorted(set(all_att_files))
        ],
    }
    _write_json(args.output, report)
    print(f"GEAK_MULTIRANK_ANALYSIS_JSON={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
