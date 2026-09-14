"""Generic declarative Expert Skill contract tests."""

import importlib.util
import os
from pathlib import Path

import pytest
import yaml


TOOL = Path(__file__).resolve().parents[1] / "tools" / "expert_skill_contract.py"
SPEC = importlib.util.spec_from_file_location("expert_skill_contract", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _contract():
    return {
        "schema_version": "expert-skill-contract-v1",
        "skill_id": "test_skill",
        "revision": "v1",
        "source": {
            "include": ["src/*.py"],
            "required": ["src/a.py"],
            "core_identity": ["src/a.py"],
        },
        "provenance": {"require_manifest": True},
        "copy_detection": {
            "exact_similarity": 0.985,
            "suspect_similarity": 0.90,
            "exact_function_ratio": 0.80,
            "suspect_function_ratio": 0.50,
        },
        "plan": {
            "required": True,
            "assertions": [
                {"id": "target", "path": "target_launches", "op": "eq", "value": 2},
                {
                    "id": "threads",
                    "left": {"path": "resource.threads"},
                    "op": "eq",
                    "right": {
                        "op": "multiply",
                        "args": [
                            {"path": "resource.wave"},
                            {"path": "resource.waves"},
                        ],
                    },
                },
            ],
        },
        "checks": [
            {
                "id": "new_logic",
                "kind": "regex",
                "file": "src/a.py",
                "patterns": [r"def new", r"return"],
            }
        ],
    }


def _root(tmp_path: Path, name: str, source: str) -> Path:
    root = tmp_path / name
    code = root / "src"
    code.mkdir(parents=True)
    (code / "a.py").write_text(source)
    return root


def _plan():
    return {
        "target_launches": 2,
        "resource": {"threads": 512, "wave": 64, "waves": 8},
    }


def _manifest(contract, candidate):
    return {
        "sealed_before_reference_comparison": True,
        "reference_exposed_during_authoring": False,
        "candidate_tree_digest": MODULE.tree_digest(
            MODULE.collect_files(candidate, contract)
        ),
    }


def test_contract_loader_rejects_invalid_schema(tmp_path):
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "bad"}))
    with pytest.raises(ValueError):
        MODULE.load_contract(path)


def test_repository_megamoe_contract_is_declarative_and_generic():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    assert contract["skill_id"] == "megamoe_ep_mega_fusion"
    assert any(item["id"] == "host_ready_pointer_identity" for item in contract["checks"])
    assert any(item["id"] == "combine_transport_and_work_domain" for item in contract["checks"])


def test_generic_engine_contains_no_operator_specific_contract():
    source = TOOL.read_text()
    assert "megamoe_ep_mega_fusion" not in source
    assert "M2.5" not in source
    assert "mega_moe_stage1.py" not in source
    assert "AITER_MEGAMOE" not in source


def test_reference_self_is_calibration_only_and_never_hardware_evidence(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    result = MODULE.evaluate(
        contract, baseline, reference, reference,
        _manifest(contract, reference), _plan(),
    )
    assert result["verdict"] == "calibration_only"
    assert result["structural_exact"]
    assert result["reference_copy_detected"]
    assert not result["capability_eligible"]
    assert not result["hardware_verified"]
    assert not result["gpu_executed"]
    assert not result["device_jit_verified"]
    assert not result["path_activation_verified"]
    assert not result["launch_count_verified"]
    assert not result["accuracy_verified"]
    assert not result["graph_liveness_verified"]
    assert not result["residency_verified"]
    assert not result["performance_verified"]


def test_independent_shape_can_pass_with_valid_seal_and_plan(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def new():\n    result = 1\n    return result\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert result["structural_compatible"]
    assert result["plan_consistent"]
    assert not result["reference_copy_detected"]
    assert result["independent_structure_pass"]
    assert result["capability_eligible"]
    assert result["provenance_status"] == "attested_unverified"


def test_feature_in_static_dead_code_or_comment_is_rejected(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        '"""def new(): return 1"""\n# def new(): return 1\nif False:\n'
        "    def new():\n        return 1\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["new_logic"]["pass"]
    assert not result["structural_compatible"]


def test_plan_expression_is_machine_checked(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(tmp_path, "candidate", "def new():\n    return 1\n")
    broken = _plan()
    broken["resource"]["threads"] = 256
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), broken,
    )
    assert not result["plan_consistent"]
    assert not result["structural_compatible"]
    assert any("threads" in error for error in result["plan_consistency_errors"])


def test_plan_may_defer_compile_only_resource_measurement():
    contract = _contract()
    contract["plan"] = {
        "required": True,
        "assertions": [{
            "id": "scratch_unverified_or_zero",
            "path": "resource.scratch_bytes_max",
            "op": "in",
            "value": [None, 0],
        }],
    }
    valid, errors, _ = MODULE.validate_plan(
        contract, {"resource": {"scratch_bytes_max": None}}
    )
    assert valid
    assert not errors


def test_call_keyword_identity_rejects_payload_pointer_as_ready_pointer(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "ready_identity",
            "kind": "call_keywords",
            "file": "src/a.py",
            "scope": "wire",
            "call": "dict",
            "keywords": {
                "local_ready": r"_fx_tok_ready",
                "peer_ready": r"_fx_p2p_tok_ready",
            },
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def wire(op):\n"
        "    return dict(local_ready=op._fx_tok_ready, "
        "peer_ready=op._fx_p2p_comb_inp)\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["ready_identity"]["pass"]
    assert "peer_ready" in result["next_blocker"]


def test_contract_score_denominator_is_stable_when_scope_is_missing(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "three_patterns",
            "kind": "regex",
            "file": "src/a.py",
            "scope": "target",
            "patterns": ["one", "two", "three"],
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    present = _root(
        tmp_path, "present",
        "def target():\n    one = two = three = 1\n    return one + two + three\n",
    )
    missing = _root(tmp_path, "missing", "def other():\n    return 0\n")
    present_result = MODULE.evaluate(
        contract, baseline, present, None,
        _manifest(contract, present), _plan(),
    )
    missing_result = MODULE.evaluate(
        contract, baseline, missing, None,
        _manifest(contract, missing), _plan(),
    )
    assert present_result["semantic_features_total"] == 3
    assert missing_result["semantic_features_total"] == 3


def test_forbidden_host_counter_reset_is_rejected(tmp_path):
    contract = _contract()
    contract["checks"] = [
        {
            "id": "no_reset",
            "kind": "forbid_methods",
            "file": "src/a.py",
            "scope": "launch",
            "methods": ["zero_"],
            "receivers": ["ready"],
        }
    ]
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def launch(token_ready):\n"
        "    token_ready.zero_()\n"
        "    return token_ready\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, None,
        _manifest(contract, candidate), _plan(),
    )
    assert not result["checks"]["no_reset"]["pass"]
    assert not result["structural_compatible"]


def test_missing_required_source_file_fails_closed(tmp_path):
    contract = _contract()
    missing = tmp_path / "missing"
    result = MODULE.evaluate(contract, missing, missing, None, None, _plan())
    assert result["input_errors"]
    assert not result["structural_compatible"]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_reference_links_are_not_independent_evidence(tmp_path, link_kind):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(tmp_path, "candidate", "def placeholder():\n    pass\n")
    target = candidate / "src" / "a.py"
    target.unlink()
    source = reference / "src" / "a.py"
    if link_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    evidence = result["candidate_symlinks"] + result["reference_hardlinks"]
    assert "src/a.py" in evidence
    assert not result["capability_eligible"]
