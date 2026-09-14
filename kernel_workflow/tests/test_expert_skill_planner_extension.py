"""Generic Expert Skill Planner Extension envelope tests."""

import copy
import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = (
    ROOT / "perf_knowledge" / "expert_skills" / "_contribute" / "validate_skill.py"
)
SPEC = importlib.util.spec_from_file_location("validate_expert_skill", VALIDATOR)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

SKILL_DIR = (
    ROOT
    / "perf_knowledge"
    / "expert_skills"
    / "skills"
    / "megamoe_ep_mega_fusion"
)


def test_repository_planner_extension_passes_generic_envelope():
    skill_path, metadata, _, _ = MODULE.load("megamoe_ep_mega_fusion")
    assert MODULE.planner_extension_errors(skill_path, metadata) == []

    extension = yaml.safe_load((SKILL_DIR / "planner_extension.yaml").read_text())
    assert extension["plan_version"] == "mega-plan-v2"
    assert extension["candidate_templates"][0]["id"] == "full_persistent_pipeline"
    assert any(
        route["id"] == "repair_direct_abi"
        for route in extension["failure_routes"]
    )


def _write_fixture(tmp_path, extension):
    skill_path = tmp_path / "skill.md"
    skill_path.write_text("---\nid: fixture\n---\n")
    (tmp_path / "planner.yaml").write_text(yaml.safe_dump(extension))
    (tmp_path / "contract.yaml").write_text(
        yaml.safe_dump({
            "schema_version": "expert-skill-contract-v1",
            "skill_id": "fixture",
            "revision": "v1",
            "plan": {"assertions": [{"id": "plan_ok"}]},
            "checks": [{"id": "source_ok"}],
        })
    )
    metadata = {
        "id": "fixture",
        "revision": "v1",
        "planner_extension_file": "planner.yaml",
        "contract_file": "contract.yaml",
    }
    return skill_path, metadata


def _minimal_extension():
    return {
        "schema_version": "expert-skill-planner-extension-v1",
        "skill_id": "fixture",
        "revision": "v1",
        "plan_version": "mega-plan-v2",
        "ir_bindings": {"work_domains": [{"id": "items"}]},
        "candidate_templates": [{
            "id": "full",
            "target_topology": {"launch_count": 1},
        }],
        "failure_routes": [{
            "id": "repair",
            "match": {"check_ids": ["source_ok"]},
            "repair_intent": "repair the declared source invariant",
        }],
    }


def test_revision_mismatch_fails_closed(tmp_path):
    extension = _minimal_extension()
    extension["revision"] = "stale"
    skill_path, metadata = _write_fixture(tmp_path, extension)
    errors = MODULE.planner_extension_errors(skill_path, metadata)
    assert any("revision must match" in error for error in errors)


def test_unknown_contract_route_and_environment_identity_are_rejected(tmp_path):
    extension = copy.deepcopy(_minimal_extension())
    extension["failure_routes"][0]["match"]["check_ids"] = ["missing_check"]
    extension["reference_path"] = "/private/reference"
    skill_path, metadata = _write_fixture(tmp_path, extension)
    errors = MODULE.planner_extension_errors(skill_path, metadata)
    assert any("unknown contract checks" in error for error in errors)
    assert any("environment identity key reference_path" in error for error in errors)
    assert any("absolute machine paths" in error for error in errors)


def test_generated_index_exposes_planner_extension():
    index = yaml.safe_load(
        (ROOT / "perf_knowledge" / "expert_skills" / "index.yaml").read_text()
    )
    entry = next(
        item for item in index["skills"]
        if item["id"] == "megamoe_ep_mega_fusion"
    )
    assert entry["planner_extension_file"].endswith("/planner_extension.yaml")
