"""Expert Skill evidence-state and index-eligibility tests."""

import importlib.util
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
VALIDATE_PATH = (
    ROOT / "perf_knowledge" / "expert_skills" / "_contribute" / "validate_skill.py"
)
SCAFFOLD_PATH = (
    ROOT / "perf_knowledge" / "expert_skills" / "_contribute" / "scaffold.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


VALIDATE = _load_module("validate_skill", VALIDATE_PATH)
SCAFFOLD = _load_module("scaffold_skill", SCAFFOLD_PATH)


def _v6_validation():
    path = (
        ROOT
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "validation.yaml"
    )
    return yaml.safe_load(path.read_text())


def test_repository_v6_is_explicit_experimental_transfer():
    skill_path, skill_metadata, body, _ = VALIDATE.load("megamoe_ep_mega_fusion")
    validation = _v6_validation()
    metadata = VALIDATE.validation_metadata(validation)
    assert validation["schema_version"] == "expert-skill-validation-v2"
    assert metadata == {
        "validation_status": "experimental",
        "reference_evidence_status": "measured",
        "constraint_validation_status": "static_validated",
        "auto_apply": False,
        "explicit_pin_modes": ["authoring", "candidate_validation"],
    }
    assert VALIDATE.validation_errors({}, validation) == []
    assert VALIDATE.static_check(skill_path, skill_metadata, body) == []


def test_measured_reference_shape_is_not_mislabeled_as_v6_transfer_rule():
    validation = _v6_validation()
    reference = validation["reference_evidence"]["mechanism_facts"]
    transfer = validation["constraint_validation"]["rules"][
        "g1_owned_mtile_completion"
    ]["transfer_deviation"]
    assert (
        reference["completion_publication"]
        == "per_stripe_system_atomic_in_unified_loop_tail"
    )
    assert transfer["from_reference"] == reference["completion_publication"]
    assert transfer["to_experimental_rule"] == "one_m_tile_owner_system_store"


def test_v2_cannot_claim_validated_with_pending_positive_repair():
    validation = _v6_validation()
    validation["status"] = "validated"
    validation["usage"]["auto_apply"] = True
    errors = VALIDATE.validation_errors({}, validation)
    assert any("hardware_validated" in error for error in errors)
    assert any("hardware_verified" in error for error in errors)
    assert any("pending positive repair" in error for error in errors)


def test_v2_cannot_promote_without_exact_candidate_gate_evidence():
    validation = _v6_validation()
    validation["status"] = "validated"
    validation["constraint_validation"]["status"] = "hardware_validated"
    validation["constraint_validation"]["hardware_verified"] = True
    validation["constraint_validation"]["rules"] = {}
    validation["usage"]["auto_apply"] = True
    errors = VALIDATE.validation_errors({}, validation)
    assert any("non-empty rule evidence" in error for error in errors)
    assert any("exact candidate_evidence" in error for error in errors)
    assert any("candidate gate is not passing" in error for error in errors)
    assert any("manifest verification" in error for error in errors)


def test_v2_validated_missing_auto_apply_defaults_false():
    validation = _v6_validation()
    validation["status"] = "validated"
    validation["usage"].pop("auto_apply")
    assert not VALIDATE.validation_metadata(validation)["auto_apply"]


def test_experimental_v2_cannot_auto_apply():
    validation = _v6_validation()
    validation["usage"]["auto_apply"] = True
    errors = VALIDATE.validation_errors({}, validation)
    assert any("auto_apply" in error for error in errors)
    assert not VALIDATE.validation_metadata(validation)["auto_apply"]


def test_legacy_v1_validated_remains_auto_applicable():
    validation = {
        "schema_version": "expert-skill-validation-v1",
        "status": "validated",
    }
    assert VALIDATE.validation_errors({}, validation) == []
    assert VALIDATE.validation_metadata(validation)["auto_apply"]


def test_scaffold_indexes_experimental_without_auto_apply(tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text("---\nid: sample\n---\n")
    (tmp_path / "validation.yaml").write_text(
        yaml.safe_dump(_v6_validation(), sort_keys=False)
    )
    metadata = SCAFFOLD.validation_metadata(
        str(skill), {"validation_file": "validation.yaml"}
    )
    assert metadata["validation_status"] == "experimental"
    assert metadata["reference_evidence_status"] == "measured"
    assert metadata["constraint_validation_status"] == "static_validated"
    assert not metadata["auto_apply"]


def test_scaffold_rejects_invalid_v2_promotion(tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text("---\nid: sample\n---\n")
    validation = _v6_validation()
    validation["status"] = "validated"
    validation["constraint_validation"]["status"] = "hardware_validated"
    validation["constraint_validation"]["hardware_verified"] = True
    validation["constraint_validation"]["rules"] = {}
    validation["usage"]["auto_apply"] = True
    (tmp_path / "validation.yaml").write_text(
        yaml.safe_dump(validation, sort_keys=False)
    )
    with pytest.raises(ValueError, match="candidate_evidence"):
        SCAFFOLD.validation_metadata(
            str(skill), {"validation_file": "validation.yaml"}
        )


def test_v2_promotion_reads_hashes_and_raw_gate_evidence(tmp_path):
    validation = _v6_validation()
    validation["status"] = "validated"
    validation["constraint_validation"]["status"] = "hardware_validated"
    validation["constraint_validation"]["hardware_verified"] = True
    rule = validation["constraint_validation"]["rules"][
        "g1_owned_mtile_completion"
    ]
    rule["positive_repair"] = {
        "status": "validated",
        "candidate_head": "a" * 40,
        "candidate_tree_sha256": "b" * 64,
    }
    validation["usage"]["auto_apply"] = True
    gates = {
        gate: "pass"
        for gate in validation["candidate_validation_policy"]["required_gates"]
    }
    raw_evidence = {}
    for gate in gates:
        artifact = {
            "schema_version": "expert-skill-gate-evidence-v1",
            "gate": gate,
            "status": "pass",
            "candidate_head": "a" * 40,
            "candidate_tree_sha256": "b" * 64,
        }
        artifact_raw = json.dumps(artifact, sort_keys=True).encode()
        artifact_name = f"{gate}.json"
        (tmp_path / artifact_name).write_bytes(artifact_raw)
        raw_evidence[gate] = {
            "artifact_uri": artifact_name,
            "sha256": hashlib.sha256(artifact_raw).hexdigest(),
        }
    subject = validation["constraint_validation"]["subject"]
    manifest = {
        "schema_version": "expert-skill-candidate-evidence-v1",
        "skill_revision": "mega-ep-fusion-v6",
        "candidate_head": "a" * 40,
        "candidate_tree_sha256": "b" * 64,
        "contract_sha256": subject["contract_sha256"],
        "planner_extension_sha256": subject["planner_extension_sha256"],
        "checker_sha256": subject["checker_sha256"],
        "gates": gates,
        "raw_evidence": raw_evidence,
    }
    raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "candidate_evidence.json").write_bytes(raw)
    validation["candidate_evidence"] = {
        "candidate_head": "a" * 40,
        "candidate_tree_sha256": "b" * 64,
        "contract_sha256": subject["contract_sha256"],
        "planner_extension_sha256": subject["planner_extension_sha256"],
        "checker_sha256": subject["checker_sha256"],
        "manifest_uri": "candidate_evidence.json",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "gates": gates,
    }
    errors = VALIDATE.validation_errors(
        {"revision": "mega-ep-fusion-v6"}, validation, str(tmp_path)
    )
    assert errors == []
    manifest["raw_evidence"] = {gate: "made-up" for gate in gates}
    fake_raw = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / "candidate_evidence.json").write_bytes(fake_raw)
    validation["candidate_evidence"]["manifest_sha256"] = hashlib.sha256(
        fake_raw
    ).hexdigest()
    errors = VALIDATE.validation_errors(
        {"revision": "mega-ep-fusion-v6"}, validation, str(tmp_path)
    )
    assert any("raw evidence is not structured" in error for error in errors)
    validation["candidate_evidence"]["manifest_sha256"] = "c" * 64
    errors = VALIDATE.validation_errors(
        {"revision": "mega-ep-fusion-v6"}, validation, str(tmp_path)
    )
    assert any("manifest_sha256" in error for error in errors)


def test_emit_plan_produces_runnable_gpu_free_experimental_args(capsys):
    path, metadata, _, _ = VALIDATE.load("megamoe_ep_mega_fusion")
    VALIDATE.emit_plan(
        path,
        "megamoe_ep_mega_fusion",
        metadata,
        SimpleNamespace(model=""),
    )
    output = capsys.readouterr().out
    assert "mode=mega" in output
    assert "expert_skill_validation_status=experimental" in output
    assert "expert_skill_usage=authoring" in output
    assert "mega_structural_only=true" in output
    assert "require_expert_skill_contract=true" in output
