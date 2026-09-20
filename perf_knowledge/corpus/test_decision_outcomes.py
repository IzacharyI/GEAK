#!/usr/bin/env python3
"""Tests for the standalone decision-outcome aggregator."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "aggregate_decision_outcomes", HERE / "_aggregate_decision_outcomes.py",
)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)


def write_yaml(run, outcomes, **context):
    run.mkdir(parents=True, exist_ok=True)
    data = {
        "kernel": "test-kernel",
        "captured_at": run.name,
        "workspace_commit": context.get("workspace_commit", "a" * 40),
        "observed_language": context.get("observed_language", "flydsl"),
        "flydsl_version": context.get("flydsl_version", "0.3.0"),
        "provider": context.get("provider", "standalone_flydsl"),
        "gfx": context.get("gfx", "gfx950"),
        "op_spec": context.get("op_spec", {"m": 128, "n": 256, "k": 64}),
        "measurement": {
            "method": "paired events",
            "warmup_iterations": 10,
            "benchmark_iterations": 100,
            "workload_aligned": False,
            "reliable": context.get("reliable", True),
        },
        "validation": {"status": context.get("validation_status", "accepted")},
        "decision_outcomes": outcomes,
    }
    path = run / "validation_environment.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def outcome(decision, **overrides):
    row = {
        "decision": decision,
        "round": 1,
        "direction": "r1_d0",
        "specialty": "compute",
        "cited_then_verified": 1.25,
        "status": "verified",
        "correctness": "pass",
        "became_winner": True,
        "incremental_vs_incumbent": 1.10,
        "incremental_basis": "ratio_of_frozen_baseline_speedups",
    }
    row.update(overrides)
    return row


def test_single_card_direction_is_isolated_and_keeps_context(tmp_path):
    write_yaml(tmp_path / "run-a", [outcome("card-a")], op_spec={"dtype": "fp8"})

    got = AGG.aggregate_decision_outcomes([tmp_path])

    assert got["schema_version"] == 1
    assert got["runs_scanned"] == 1
    assert got["errors"] == []
    card = got["decisions"]["card-a"]
    assert card["attempts"] == card["unique_runs"] == card["winner_count"] == 1
    assert card["status_counts"] == {"verified": 1}
    assert card["correctness_counts"] == {"pass": 1}
    assert card["single_ref_direction_results"] == 1
    assert card["single_ref_seed_results"] == card["bundled_direction_results"] == 0
    assert card["contexts"][0]["observed_language"] == "flydsl"
    assert card["contexts"][0]["metric_kind"] == "geomean"
    assert card["contexts"][0]["measurement_reliable"] is True
    assert card["contexts"][0]["validation_status"] == "accepted"
    assert len(card["contexts"][0]["protocol_fingerprint"]) == 16
    assert card["single_ref_incremental_vs_incumbent"]["median"] == 1.1
    assert card["single_ref_speedup_vs_frozen_baseline"]["median"] == 1.25
    result = got["single_ref_direction_results"][0]
    assert result["decisions"] == ["card-a"]
    assert result["speedup_vs_frozen_baseline"] == 1.25
    assert result["incremental_vs_incumbent"] == 1.1
    assert result["attribution_scope"] == "single_ref_direction"
    assert result["quality_eligible_for_ranking"] is True
    assert "cited_then_verified" not in json.dumps(got)


def test_multi_card_direction_is_bundle_only(tmp_path):
    write_yaml(tmp_path / "run-b", [
        outcome("card-a"),
        outcome("card-b", cited_then_verified=1.25),
    ])

    got = AGG.aggregate([tmp_path])

    assert got["single_ref_direction_results"] == []
    assert len(got["bundled_direction_results"]) == 1
    bundle = got["bundled_direction_results"][0]
    assert bundle["decisions"] == ["card-a", "card-b"]
    assert bundle["decision_count"] == 2
    assert bundle["attribution_scope"] == "bundle_only"
    assert bundle["individual_decision_attribution"] is False
    for decision in ("card-a", "card-b"):
        assert got["decisions"][decision]["attempts"] == 1
        assert got["decisions"][decision]["winner_count"] == 0
        assert got["decisions"][decision]["single_ref_direction_results"] == 0
        assert got["decisions"][decision]["bundled_direction_results"] == 1


def test_bad_yaml_is_skipped_and_counted_as_an_error(tmp_path):
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    write_yaml(good, [outcome("card-a")])
    bad.mkdir()
    (bad / "validation_environment.yaml").write_text(
        "decision_outcomes: [unterminated", encoding="utf-8",
    )

    got = AGG.aggregate([tmp_path])

    assert got["runs_scanned"] == 1
    assert len(got["errors"]) == 1
    assert got["errors"][0]["path"].endswith("bad/validation_environment.yaml")
    assert got["decisions"]["card-a"]["attempts"] == 1


def test_json_manifest_is_accepted(tmp_path):
    run = tmp_path / "json-run"
    run.mkdir()
    (run / "validation_environment.json").write_text(json.dumps({
        "observed_language": "flydsl",
        "flydsl_version": "0.4.0",
        "provider": "aiter_vendored_flydsl",
        "gfx": "gfx942",
        "op_spec": {"shape": [8, 4096, 7168]},
        "decision_outcomes": [
            outcome("json-card", status="regression", correctness="pass",
                    became_winner=False, cited_then_verified=0.98),
        ],
    }), encoding="utf-8")

    got = AGG.aggregate([tmp_path])

    card = got["decisions"]["json-card"]
    assert card["status_counts"] == {"regression": 1}
    assert card["correctness_counts"] == {"pass": 1}
    assert got["single_ref_direction_results"][0]["context"]["gfx"] == "gfx942"
    assert got["single_ref_direction_results"][0]["speedup_vs_frozen_baseline"] == 0.98


def test_duplicate_and_overlapping_runs_paths_are_deduplicated(tmp_path):
    run = tmp_path / "nested" / "run"
    write_yaml(run, [outcome("card-a")])

    got = AGG.aggregate([tmp_path, tmp_path, tmp_path / "nested", run])

    assert got["runs_scanned"] == 1
    assert got["decisions"]["card-a"]["attempts"] == 1
    assert got["decisions"]["card-a"]["unique_runs"] == 1


def test_copied_manifest_is_deduplicated_by_run_identity(tmp_path):
    first = write_yaml(tmp_path / "original", [outcome("card-a")])
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "validation_environment.yaml").write_text(first.read_text())
    got = AGG.aggregate([tmp_path])
    assert got["runs_scanned"] == 1
    assert got["runs_deduplicated"] == 1


def test_cli_json_stdout_and_output_file(tmp_path, capsys):
    write_yaml(tmp_path / "run", [outcome("card-a")])

    assert AGG.main(["--runs", str(tmp_path), "--json"]) == 0
    stdout_payload = json.loads(capsys.readouterr().out)
    assert stdout_payload["runs_scanned"] == 1

    output = tmp_path / "aggregate.yaml"
    assert AGG.main(["--runs", str(tmp_path), "--output", str(output)]) == 0
    file_payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert file_payload["decisions"]["card-a"]["attempts"] == 1


def test_flagged_invalid_numbers_do_not_enter_ranking_statistics(tmp_path):
    write_yaml(
        tmp_path / "flagged",
        [outcome("card-a", incremental_vs_incumbent=float("nan"))],
        validation_status="flagged",
    )
    write_yaml(
        tmp_path / "bool",
        [outcome("card-b", incremental_vs_incumbent=True)],
    )
    got = AGG.aggregate([tmp_path])
    assert "single_ref_incremental_vs_incumbent" not in got["decisions"]["card-a"]
    assert "single_ref_incremental_vs_incumbent" not in got["decisions"]["card-b"]
    assert all(
        row["quality_eligible_for_ranking"] is False
        for row in got["single_ref_direction_results"]
    )


def test_accepted_regression_is_preserved_as_negative_evidence(tmp_path):
    write_yaml(tmp_path / "regression", [
        outcome(
            "card-a", status="regression", became_winner=False,
            cited_then_verified=0.8, incremental_vs_incumbent=0.9,
        ),
    ])
    got = AGG.aggregate([tmp_path])
    assert got["decisions"]["card-a"]["single_ref_incremental_vs_incumbent"]["median"] == 0.9
    assert got["single_ref_direction_results"][0]["quality_eligible_for_ranking"] is True


def test_unscoped_rows_do_not_accidentally_become_one_bundle(tmp_path):
    rows = [
        outcome("card-a", round=None, direction=None),
        outcome("card-b", round=None, direction=None),
    ]
    write_yaml(tmp_path / "unscoped", rows)
    got = AGG.aggregate([tmp_path])
    assert len(got["single_ref_direction_results"]) == 2
    assert got["bundled_direction_results"] == []


def test_single_ref_author_seed_stays_weak_seed_evidence(tmp_path):
    row = outcome(
        "card-a", round=None, direction="author_seed", phase="author_seed",
        attribution="single_ref_seed", incremental_vs_incumbent=None,
    )
    write_yaml(tmp_path / "seed", [row])
    got = AGG.aggregate([tmp_path])
    assert got["single_ref_direction_results"] == []
    assert got["single_ref_seed_results"][0]["attribution_scope"] == "single_ref_seed"
    assert got["decisions"]["card-a"]["single_ref_seed_results"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
