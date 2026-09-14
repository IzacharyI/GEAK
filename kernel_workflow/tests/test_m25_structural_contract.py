"""Offline M2.5 source-structure contract tests."""

import importlib.util
import os
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "m25_structural_contract.py"
SPEC = importlib.util.spec_from_file_location("m25_structural_contract", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


@pytest.fixture(autouse=True)
def _single_file_contract(monkeypatch):
    monkeypatch.setattr(MODULE, "CORE_IDENTITY_FILES", ("a.py",))
    monkeypatch.setattr(MODULE, "HOST_FILES", ())
    monkeypatch.setattr(MODULE, "SCOPED_FEATURES", {})
    monkeypatch.setattr(MODULE, "ENFORCE_CORRECTED_FACTS", False)
    monkeypatch.setattr(MODULE, "ENFORCE_PINNED_INPUTS", False)


def _root(tmp_path: Path, name: str, source: str) -> Path:
    root = tmp_path / name
    code = root / MODULE.MEGA_REL
    code.mkdir(parents=True)
    (code / "a.py").write_text(source)
    return root


def _plan():
    return {
        "target_launches": 2,
        "resource_contract": {
            "arch": "gfx950",
            "wave_size": 64,
            "threads_per_workgroup": 512,
            "num_waves": 8,
            "lds": {
                "stage1_pool_bytes": 131072,
                "stage2_slab_bytes": 131728,
                "additive_bytes": 28672,
                "group_segment_bytes": 160400,
                "limit_bytes": 163840,
                "aliasing": "max",
                "halves": 1,
            },
            "registers": {"scratch_bytes_max": 0},
            "residency": {"min_workgroups_per_cu": 1},
        },
        "schedule_contract": {
            "unified_gemm_loop": True,
            "carried_scalars": ["consumer_active", "g2_pend", "g2_next"],
            "combine_third_queue": True,
            "g2_chunk_large": 16,
            "g2_chunk_small": 1,
            "preemption_interval": 6,
            "skew_num": 5,
            "skew_den": 4,
            "work_shards": 4,
        },
        "abi": {
            "direct_fused_args": True,
            "stage2_pointer_count": 12,
            "combine_pointer_count": 6,
            "optional_quant_pointer_count": 0,
            "disabled_placeholders": True,
            "argument_order": ["s2", "combine"],
        },
    }


def test_reference_self_is_exact_and_never_claims_hardware(tmp_path):
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    result = MODULE.evaluate(baseline, reference, reference)
    assert result["verdict"] == "calibration_only"
    assert result["structural_exact"]
    assert not result["hardware_verified"]
    assert not result["gpu_executed"]
    assert not result["device_jit_verified"]
    assert not result["path_activation_verified"]
    assert not result["launch_count_verified"]
    assert not result["accuracy_verified"]
    assert not result["numeric_accuracy_verified"]
    assert not result["graph_liveness_verified"]
    assert not result["residency_verified"]
    assert not result["distributed_memory_order_verified"]
    assert not result["performance_verified"]


def test_baseline_cannot_pass_as_reference_structure(tmp_path):
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    result = MODULE.evaluate(baseline, reference, baseline)
    assert result["verdict"] == "incomplete"
    assert result["required_changed_files_missing"] == ["a.py"]
    assert result["missing_new_symbols"] == {"a.py": ["new"]}


def test_exact_oracle_identity_is_not_independent_capability(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new",)),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    result = MODULE.evaluate(baseline, reference, reference)
    assert result["structural_compatible"]
    assert result["reference_copy_detected"]
    assert not result["independent_structure_pass"]
    assert not result["capability_eligible"]


def test_independent_semantic_shape_can_pass_without_exact_source(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new", r"return")),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "def new():\n    value = 1\n    return value\n",
    )
    manifest = {
        "sealed_before_oracle_comparison": True,
        "oracle_exposed_during_authoring": False,
        "candidate_tree_digest": MODULE._tree_digest(MODULE._files(candidate)),
    }
    result = MODULE.evaluate(baseline, reference, candidate, manifest)
    assert result["structural_compatible"]
    assert not result["reference_copy_detected"]
    assert result["independent_structure_pass"]
    assert result["capability_eligible"]
    assert result["provenance_status"] == "attested_unverified"


def test_comment_modified_oracle_copy_is_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new",)),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "# changed comment only\ndef new():\n    return 1\n",
    )
    result = MODULE.evaluate(baseline, reference, candidate)
    assert result["reference_copy_detected"]
    assert not result["capability_eligible"]


def test_feature_present_only_under_if_false_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new",)),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        "if False:\n    def new():\n        return 1\n",
    )
    result = MODULE.evaluate(baseline, reference, candidate)
    assert not result["features"]["new_logic"]["pass"]
    assert not result["structural_compatible"]


def test_feature_present_only_in_comment_or_docstring_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new",)),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(
        tmp_path,
        "candidate",
        '"""def new(): return 1"""\n# def new(): return 1\ndef old():\n    pass\n',
    )
    result = MODULE.evaluate(baseline, reference, candidate)
    assert not result["features"]["new_logic"]["pass"]
    assert not result["structural_compatible"]


def test_missing_roots_cannot_be_exact(tmp_path):
    missing = tmp_path / "missing"
    result = MODULE.evaluate(missing, missing, missing)
    assert result["input_errors"]
    assert not result["structural_exact"]
    assert not result["structural_compatible"]


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_oracle_links_are_not_capability_evidence(tmp_path, monkeypatch, link_kind):
    monkeypatch.setattr(MODULE, "FEATURES", {
        "new_logic": (("a.py", (r"def new",)),),
    })
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(tmp_path, "reference", "def new():\n    return 1\n")
    candidate = _root(tmp_path, "candidate", "def placeholder():\n    pass\n")
    target = candidate / MODULE.MEGA_REL / "a.py"
    target.unlink()
    source = reference / MODULE.MEGA_REL / "a.py"
    if link_kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)
    result = MODULE.evaluate(baseline, reference, candidate)
    evidence = result["candidate_symlinks"] + result["oracle_hardlinks"]
    assert "a.py" in evidence
    assert not result["capability_eligible"]


def test_typed_plan_resource_arithmetic_is_machine_checked():
    valid, errors = MODULE._validate_plan(_plan())
    assert valid and not errors
    broken = _plan()
    broken["resource_contract"]["lds"]["group_segment_bytes"] = 131728
    valid, errors = MODULE._validate_plan(broken)
    assert not valid
    assert any("group_segment_bytes" in error for error in errors)
