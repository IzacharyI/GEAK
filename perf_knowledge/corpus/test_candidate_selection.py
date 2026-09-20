"""Contracts for the corpus catalog and deterministic card matcher."""

import json
from pathlib import Path

import pytest
import yaml

import _select_candidates as SELECT


HERE = Path(__file__).resolve().parent


def test_catalog_paths_exist_and_family_names_are_unique():
    catalog = yaml.safe_load((HERE / "catalog.yaml").read_text())
    seen = set()
    for family in catalog["families"]:
        names = [family["id"], *(family.get("patterns") or [])]
        assert not (seen & set(names)), f"ambiguous catalog names: {seen & set(names)}"
        seen.update(names)
        for field in (
            "source_evidence", "tuned_evidence", "decisions", "source_document",
            "decision_document", "benchmark_manifest",
        ):
            assert (HERE / family[field]).is_file(), (family["id"], field, family[field])


def test_gfx_constraint_and_eligible_performance_candidates_are_separate():
    result = SELECT.select(HERE / "catalog.yaml", {
        "operator_family": "scaled_quant_gemm",
        "target_language": "flydsl",
        "gfx": "gfx942",
        "dtype": "fp8_e4m3fnuz_blockscale",
        "regime": "prefill",
        "bottleneck": "compute",
        "m": 32768,
        "n": 4096,
        "k": 2048,
        "traits": ["lds_staging"],
    })
    got = {row["id"]: row["type"] for row in result["eligible"]}
    assert got["flydsl-gfx942-path-gates"] == "constraint"
    assert got["flydsl-xor16-lds-layout-consistency"] == "performance_candidate"
    assert "flydsl-small-m-hgemm-family" not in got


def test_small_m_family_is_rejected_on_gfx942_and_selected_on_gfx950():
    base = {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "dtype": "bf16",
        "regime": "decode",
        "bottleneck": "latency",
        "m": 8,
        "n": 4096,
        "k": 7168,
        "traits": [],
    }
    rejected = SELECT.select(HERE / "catalog.yaml", {**base, "gfx": "gfx942"})
    selected = SELECT.select(HERE / "catalog.yaml", {**base, "gfx": "gfx950"})
    assert "flydsl-small-m-hgemm-family" in {row["id"] for row in rejected["rejected"]}
    assert "flydsl-small-m-hgemm-family" in {row["id"] for row in selected["eligible"]}


def test_required_trait_defers_a_card_instead_of_calling_it_slow():
    card = {
        "match": {
            "operator_families": ["gemm"],
            "target_languages": ["flydsl"],
            "requires": ["lds_staging"],
        },
    }
    status, reasons = SELECT.classify_card(card, {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "traits": [],
    })
    assert status == "deferred"
    assert any("lds_staging" in reason for reason in reasons)


@pytest.mark.parametrize(("given", "stored"), [("fp16", "f16"), ("bfloat16", "bf16")])
def test_dtype_aliases_match(given, stored):
    card = {
        "match": {
            "operator_families": ["gemm"],
            "target_languages": ["flydsl"],
            "dtypes": [stored],
        },
    }
    status, _ = SELECT.classify_card(card, {
        "operator_family": "gemm",
        "target_language": "flydsl",
        "dtype": given,
        "traits": [],
    })
    assert status == "eligible"


def test_compatible_single_ref_outcomes_rank_but_bundles_do_not(tmp_path):
    context = {
        "operator_family": "scaled_quant_gemm",
        "target_language": "flydsl",
        "gfx": "gfx942",
        "flydsl_version": "0.3.0",
        "dtype": "fp8_e4m3fnuz_blockscale",
        "regime": "prefill",
        "bottleneck": "compute",
        "m": 32768,
        "n": 4096,
        "k": 1024,
        "traits": ["lds_staging", "split_k_hot_loop"],
    }
    exact = {
        "observed_language": "flydsl",
        "flydsl_version": "0.3.0",
        "gfx": "gfx942",
        "op_spec": {
            "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
            "m": 32768, "n": 4096, "k": 1024,
        },
    }
    outcomes = {
        "single_ref_direction_results": [
            {
                "run": "a", "decisions": ["flydsl-xor16-lds-layout-consistency"],
                "incremental_vs_incumbent": 1.05, "speedup_vs_frozen_baseline": 1.1,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True, "context": exact,
            },
            {
                "run": "b", "decisions": ["flydsl-splitk-hot-loop-schedule-bundle"],
                "incremental_vs_incumbent": 1.20, "speedup_vs_frozen_baseline": 1.3,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True, "context": exact,
            },
            {
                "run": "wrong-gfx", "decisions": ["flydsl-xor16-lds-layout-consistency"],
                "incremental_vs_incumbent": 9.0,
                "attribution_scope": "single_ref_direction",
                "quality_eligible_for_ranking": True,
                "context": {**exact, "gfx": "gfx950"},
            },
        ],
        "bundled_direction_results": [{
            "run": "bundle",
            "decisions": [
                "flydsl-xor16-lds-layout-consistency",
                "flydsl-splitk-hot-loop-schedule-bundle",
            ],
            "incremental_vs_incumbent": 4.0,
            "quality_eligible_for_ranking": True,
            "context": exact,
        }],
    }
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps(outcomes))

    result = SELECT.select(HERE / "catalog.yaml", context, path)
    performance = [
        row for row in result["eligible"] if row["type"] == "performance_candidate"
    ]
    assert [row["id"] for row in performance[:2]] == [
        "flydsl-splitk-hot-loop-schedule-bundle",
        "flydsl-xor16-lds-layout-consistency",
    ]
    assert performance[0]["measured_prior"]["median"] == 1.20
    assert performance[1]["measured_prior"]["median"] == 1.05
    assert performance[1]["measured_prior"]["count"] == 1
    assert len(result["measured_bundles"]) == 1
    assert result["measured_bundles"][0]["incremental_vs_incumbent"] == 4.0


def test_outcome_without_matching_language_or_gfx_is_not_a_prior():
    outcomes = {
        "single_ref_direction_results": [{
            "run": "old",
            "decisions": ["card"],
            "incremental_vs_incumbent": 2.0,
            "attribution_scope": "single_ref_direction",
            "quality_eligible_for_ranking": True,
            "context": {
                "observed_language": "triton",
                "flydsl_version": "0.3.0",
                "gfx": "gfx942",
                "op_spec": {},
            },
        }],
    }
    assert SELECT.measured_priors(outcomes, {
        "target_language": "flydsl", "flydsl_version": "0.3.0", "gfx": "gfx942",
    }) == {}


def test_missing_recorded_shape_cannot_match_a_specific_request():
    row = {
        "run": "missing-shape",
        "decisions": ["card"],
        "incremental_vs_incumbent": 1.5,
        "attribution_scope": "single_ref_direction",
        "quality_eligible_for_ranking": True,
        "context": {
            "observed_language": "flydsl", "flydsl_version": "0.3.0",
            "gfx": "gfx942", "op_spec": {},
        },
    }
    assert SELECT.measured_priors(
        {"single_ref_direction_results": [row]},
        {
            "target_language": "flydsl", "flydsl_version": "0.3.0", "gfx": "gfx942",
            "m": 8, "n": 256, "k": 128,
        },
    ) == {}


def test_measured_regression_ranks_after_unmeasured_candidate(tmp_path):
    context = {
        "operator_family": "scaled_quant_gemm", "target_language": "flydsl",
        "gfx": "gfx942", "flydsl_version": "0.3.0",
        "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
        "bottleneck": "compute", "m": 32768, "n": 4096, "k": 1024,
        "traits": ["lds_staging", "split_k_hot_loop"],
    }
    path = tmp_path / "outcomes.json"
    path.write_text(json.dumps({"single_ref_direction_results": [{
        "run": "loss",
        "decisions": ["flydsl-xor16-lds-layout-consistency"],
        "incremental_vs_incumbent": 0.8,
        "attribution_scope": "single_ref_direction",
        "quality_eligible_for_ranking": True,
        "context": {
            "observed_language": "flydsl", "flydsl_version": "0.3.0",
            "gfx": "gfx942",
            "op_spec": {
                "dtype": "fp8_e4m3fnuz_blockscale", "regime": "prefill",
                "m": 32768, "n": 4096, "k": 1024,
            },
        },
    }]}))
    result = SELECT.select(HERE / "catalog.yaml", context, path)
    performance = [
        row for row in result["eligible"] if row["type"] == "performance_candidate"
    ]
    assert [row["id"] for row in performance[:2]] == [
        "flydsl-splitk-hot-loop-schedule-bundle",
        "flydsl-xor16-lds-layout-consistency",
    ]


def test_missing_gfx_defers_explicit_exclusion():
    card = {
        "match": {
            "operator_families": ["gemm"], "target_languages": ["flydsl"],
            "exclude_gfx": ["gfx942"],
        },
    }
    status, reasons = SELECT.classify_card(card, {
        "operator_family": "gemm", "target_language": "flydsl", "traits": [],
    })
    assert status == "deferred"
    assert any("missing gfx" in reason for reason in reasons)
