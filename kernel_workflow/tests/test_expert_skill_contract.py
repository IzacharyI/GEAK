"""Generic declarative Expert Skill contract tests."""

import ast
import importlib.util
import os
import re
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
                {"id": "target", "path": "target.launch_count", "op": "eq", "value": 2},
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
        "target": {"launch_count": 2},
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
    checks = {item["id"]: item for item in contract["checks"]}
    assert "host_ready_pointer_identity" in checks
    assert "combine_transport_and_work_domain" in checks
    assert "stage2_max_m_blocks_are_block_units" in checks
    assert "stage2_counter_storage_covers_sharded_heads" in checks
    assert "combine_queue_consumes_output_work_domain" in checks
    assert "blockwise_fp8_reduce_is_decoded" in checks
    assert "adaptive_combine_partition_bounds" in checks
    assert "blockwise_fp8_row_layout_matches_producer" in checks
    assert "blockwise_scale_never_uses_readiness_pointer" in checks
    assert "combine_item_has_no_workgroup_barrier" in checks
    assert "combine_generation_published_before_startup_gate" in checks
    assert "g2_completion_and_heads_reset_before_plan" in checks
    assert "fuse_combine_controls_token_publication" in checks
    assert "stage2_emitter_metadata_lds_is_slab_relative" in checks
    assert "fused_stage1_jit_identity_covers_runtime_shape" in checks
    assert checks["stage2_max_m_blocks_are_block_units"]["kind"] == "assignment_value"
    assert checks["combine_generation_published_before_startup_gate"]["kind"] == "regex_sequence"
    assert checks["combine_queue_rejects_routed_row_bound"]["match"] == "none"
    assert len(checks["fixed_slot_group_done_capacity"]["patterns"]) == 1
    consumed = checks["adaptive_combine_values_are_consumed"]["parameters"]
    assert "hdim_per_warp" in consumed
    assert "s3_total_work" in consumed
    dependency = next(
        item for item in contract["checks"]
        if item["id"] == "chunk_all_dependencies"
    )
    assert any("_g2_run_unit" in pattern for pattern in dependency["patterns"])
    assert all(item["id"] != "chunk_last_dependency" for item in contract["checks"])


def test_repository_tuple_patterns_match_ast_normalized_python():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    checks = {
        item["id"]: item
        for item in MODULE.load_contract(path)["checks"]
    }
    emitter_text = MODULE._active_code(ast.parse(
        "def make_combine_reduce_emitter():\n"
        "    return consts, _emit_item\n"
    ))
    return_pattern = checks["adaptive_combine_partition"]["patterns"][-1]
    assert re.search(return_pattern, emitter_text)

    queue_text = MODULE._active_code(ast.parse(
        "def kernel():\n"
        "    _c_consts, _c_emit = make_combine_reduce_emitter()\n"
    ))
    assign_pattern = checks["combine_queue_consumes_output_work_domain"]["patterns"][0]
    assert re.search(assign_pattern, queue_text)


def test_repository_fusion_contract_accepts_operator_neutral_plan_ir_v2():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "contract.yaml"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "expert_skill_id": "megamoe_ep_mega_fusion",
        "expert_skill_revision": "mega-ep-fusion-v3",
        "expert_skill_bundle_sha256": "bundle",
        "expert_skill_planner_extension_sha256": "planner",
        "target": {"launch_count": 2},
        "work_domains": [{"id": "tokens"}],
        "buffers": [{"id": "payload"}],
        "counters": [{"id": "ready"}],
        "resources": {
            "arch": "gfx950",
            "workgroup": {
                "wave_size": 64, "wave_count": 8, "thread_count": 512,
            },
            "local_memory": {
                "total_bytes": 160400,
                "limit_bytes": 163840,
                "allocation_rule": "max",
            },
            "registers": {"scratch_bytes_max": None},
            "occupancy": {"min_workgroups_per_cu": 1},
            "parameters": {
                "stage1_pool_bytes": 131072,
                "stage2_slab_bytes": 131728,
                "role_halves": 1,
                "additive_bytes": 28672,
            },
        },
        "schedule": {
            "primary_loop": {
                "kind": "unified",
                "carried_state": ["consumer_active", "g2_pend", "g2_next"],
            },
            "parameters": {
                "combine_third_queue": True,
                "g2_chunk_large": 16,
                "g2_chunk_small": 1,
                "preemption_interval": 6,
                "skew_num": 5,
                "skew_den": 4,
                "work_shards": 4,
            },
        },
        "abi": {
            "arguments": [{"id": "payload"}],
            "parameters": {
                "direct_fused_args": True,
                "stage2_pointer_count": 12,
                "combine_pointer_count": 7,
                "optional_quant_pointer_count": 0,
                "disabled_placeholders": True,
            },
        },
    }
    valid, errors, _ = MODULE.validate_plan(contract, plan)
    assert valid
    assert not errors


def test_generic_engine_contains_no_operator_specific_contract():
    source = TOOL.read_text()
    assert "megamoe_ep_mega_fusion" not in source
    assert "mega_moe_stage1.py" not in source
    assert "AITER_MEGAMOE" not in source


def test_assignment_value_relates_target_to_its_own_rhs():
    contract = _contract()
    contract["checks"] = [{
        "id": "capacity_units",
        "kind": "assignment_value",
        "file": "src/a.py",
        "scope": "build",
        "target": "capacity",
        "value": r"ceildiv\(rows, block\)",
    }]
    unrelated = ast.parse(
        "def build(rows, block):\n"
        "    blocks = ceildiv(rows, block)\n"
        "    capacity = rows\n"
        "    return capacity\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": unrelated})
    assert not result["capacity_units"]["pass"]

    related = ast.parse(
        "def build(rows, block):\n"
        "    capacity = ceildiv(rows, block)\n"
        "    return capacity\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": related})
    assert result["capacity_units"]["pass"]


def test_assignment_value_none_rejects_forbidden_rhs():
    contract = _contract()
    contract["checks"] = [{
        "id": "domain_separation",
        "kind": "assignment_value",
        "file": "src/a.py",
        "scope": "build",
        "target": "output_extent",
        "value": r"routed_extent",
        "match": "none",
    }]
    tree = ast.parse(
        "def build(output_extent, routed_extent):\n"
        "    output_extent = routed_extent\n"
        "    return output_extent\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": tree})
    assert not result["domain_separation"]["pass"]


def test_regex_sequence_rejects_correct_names_in_wrong_order():
    contract = _contract()
    contract["checks"] = [{
        "id": "publish_order",
        "kind": "regex_sequence",
        "file": "src/a.py",
        "scope": "build",
        "patterns": [r"state\s*=\s*prepare\(\)", r"publish\(state\)"],
    }]
    wrong = ast.parse(
        "def build():\n"
        "    publish(state)\n"
        "    state = prepare()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": wrong})
    assert not result["publish_order"]["pass"]

    right = ast.parse(
        "def build():\n"
        "    state = prepare()\n"
        "    publish(state)\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": right})
    assert result["publish_order"]["pass"]


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
            "category": "abi",
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
    assert result["contract_failures"][0]["category"] == "abi"
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
