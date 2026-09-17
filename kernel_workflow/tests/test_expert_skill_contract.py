"""Generic declarative Expert Skill contract tests."""

import ast
import hashlib
import importlib.util
import json
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

MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS = {
    "complete_host_path",
    "flat_stripe_completion",
    "unified_g1_g2_loop",
    "shared_stage2_body",
    "p2p_visibility",
    "progressive_token_readiness",
    "combine_output_work_item",
    "block_claimed_combine",
    "bucket_512_payload_rows",
}


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
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    assert contract["skill_id"] == "megamoe_ep_mega_fusion"
    assert not contract.get("revision")
    assert {item["id"] for item in contract["checks"]} == {
        "complete_host_path",
        "host_specializes_g2_chunk",
        "flat_stripe_completion",
        "unified_g1_g2_loop",
        "shared_stage2_body",
        "progressive_token_readiness",
        "combine_output_work_item",
        "block_claimed_combine",
        "p2p_visibility",
        "bucket_512_payload_rows",
    }


def test_repository_megamoe_contract_requires_selected_implementation_semantics():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    checks = {item["id"]: item for item in contract["checks"]}
    required = {
        check_id
        for check_id, check in checks.items()
        if check.get("severity", "required") == "required"
    }
    assert required == MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    assert checks["host_specializes_g2_chunk"]["severity"] == "advisory"
    assert contract["activation"] == {
        "scope": "post_selection_claimed_complete_structural_checkpoint",
        "requires_selected_or_pinned_skill": True,
        "required_checks_define_claimed_implementation_completeness": True,
    }
    assert contract["binding_policy"]["applicability_owner"] == "workflow"
    assert (
        contract["binding_policy"]["adapter_selection"]
        == "analyze_discovered_semantic_bindings"
    )


def test_missing_selected_megamoe_implementation_fails_structural_checkpoint(tmp_path):
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    for root in (baseline, candidate):
        source = root / "adapter_moe.py"
        source.parent.mkdir(parents=True)
        source.write_text("def unrelated():\n    return None\n")
    plan = {
        "plan_version": "mega-plan-v2",
        "target": {"launch_count": 2},
        "resources": {"arch": "gfx950"},
        "schedule": {"policies": {
            "combine_transition":
                "after_local_g1_g2_drain_without_grid_barrier",
            "scheduler_semantics":
                "continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16",
            "combine_semantics":
                "runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4",
        }},
    }
    result = MODULE.evaluate(
        contract,
        baseline,
        candidate,
        None,
        _manifest(contract, candidate),
        plan,
    )
    assert not result["input_errors"]
    assert not result["plan_consistency_errors"]
    assert result["failed_required_checks"] == [
        item["id"]
        for item in contract["checks"]
        if item["id"] in MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    ]
    assert not result["structural_compatible"]
    assert result["verdict"] == "incomplete"


def test_void_preflight_fixture_matches_current_contract_identity():
    fixture = json.loads(
        (
            Path(__file__).with_name("fixtures")
            / "megamoe_ep_mega_fusion_void_preflight.json"
        ).read_text()
    )
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    contract_sha = hashlib.sha256(
        yaml.safe_dump(contract, sort_keys=True).encode()
    ).hexdigest()
    assert fixture["candidate_head"] == "4d7806db89328f1fac2e4e24d9251e5917152159"
    assert fixture["contract_sha256"] == contract_sha
    assert fixture["required_check_count"] == len(
        MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    )
    assert (
        set(fixture["failed_required_checks"])
        | set(fixture["passing_required_checks"])
        == MEGAMOE_REQUIRED_IMPLEMENTATION_CHECKS
    )
    assert fixture["required_failure_count"] == len(
        fixture["failed_required_checks"]
    )
    assert fixture["required_failure_count"] == 8
    assert fixture["identity_mismatch_failures"] == []
    assert fixture["structural_compatible"] is False
    assert fixture["verdict"] == "incomplete"


def test_compact_skill_contains_no_private_evidence_identifiers():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    text = path.read_text()
    assert "/sgl-workspace" not in text
    assert "candidate_tree_sha256" not in text
    assert "f33b628" not in text
    assert "mega-ep-fusion-v" not in text


def test_repository_fusion_contract_accepts_operator_neutral_plan_ir_v2():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "expert_skill_id": "megamoe_ep_mega_fusion",
        "expert_skill_revision": "",
        "expert_skill_bundle_sha256": "bundle",
        "expert_skill_planner_extension_sha256": "planner",
        "expert_skill_validation_status": "experimental",
        "expert_skill_auto_apply": False,
        "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
        "target": {"launch_count": 2},
        "work_domains": [{
            "id": "first_stage_m_tiles",
            "unit": "m_tile",
            "extent": "ceildiv(num_valid, stage1_block_m)",
            "index_type": "i32",
            "owned_subdomain": {
                "unit": "n_stripe",
                "extent_per_claim": "stage1_n_tiles",
                "linear_index": "m_tile * stage1_n_tiles + n_stripe",
            },
        }],
        "regions": [{
            "id": "first_compute",
            "role": "producer_consumer",
            "work_domain": "first_stage_m_tiles",
            "engine": "matrix_memory",
        }],
        "queues": [{
            "id": "first_compute_queue",
            "work_domain": "first_stage_m_tiles",
            "claim_unit": "m_tile",
            "chunk_policy": "one_m_tile_all_n_stripes",
        }],
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
            "policies": {
                    "combine_transition":
                        "after_local_g1_g2_drain_without_grid_barrier",
                "completion_publication_granularity": "per_m_tile",
                "first_compute_claim_domain": "m_tile",
                "first_compute_stripe_ownership": "all_n_tiles_per_claim",
                "completion_publication_operation":
                    "waitcnt_barrier_thread0_system_store_n_tiles",
                "completion_rmw_in_unified_loop": "forbidden_transitively",
                "combine_item_jit_frame":
                    "direct_decorated_item_no_extra_wrapper",
                "combine_reducer_vectorization":
                    "pressure_guarded_u1_u2_u4",
                "combine_payload_visibility":
                    "system_scope_loads_without_per_item_acquire",
                "combine_output_cache": "rank_local_slc",
                "scheduler_semantics":
                    "continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16",
                "combine_semantics":
                    "runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4",
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


def test_repository_plan_contract_rejects_missing_combine_transition():
    path = (
        Path(__file__).resolve().parents[2]
        / "perf_knowledge"
        / "expert_skills"
        / "skills"
        / "megamoe_ep_mega_fusion"
        / "skill.md"
    )
    contract = MODULE.load_contract(path)
    plan = {
        "plan_version": "mega-plan-v2",
        "expert_skill_id": "megamoe_ep_mega_fusion",
        "expert_skill_revision": "",
        "expert_skill_bundle_sha256": "bundle",
        "expert_skill_planner_extension_sha256": "planner",
        "expert_skill_validation_status": "experimental",
        "expert_skill_auto_apply": False,
        "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
        "target": {"launch_count": 2},
        "work_domains": [{
            "id": "first_stage_m_tiles",
            "unit": "m_tile",
            "extent": "ceildiv(num_valid, stage1_block_m)",
            "owned_subdomain": {"extent_per_claim": "stage1_n_tiles"},
        }],
        "regions": [{
            "id": "first_compute",
            "work_domain": "first_stage_m_tiles",
        }],
        "queues": [{
            "id": "first_compute_queue",
            "work_domain": "first_stage_m_tiles",
            "claim_unit": "tile",
            "chunk_policy": "one_tile",
        }],
        "buffers": [{}],
        "counters": [{}],
        "resources": {
            "arch": "gfx950",
            "workgroup": {"wave_size": 64, "wave_count": 8, "thread_count": 512},
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
            "policies": {
                "completion_publication_granularity": "per_m_tile",
                "first_compute_claim_domain": "m_tile",
                "first_compute_stripe_ownership": "all_n_tiles_per_claim",
                "completion_publication_operation":
                    "waitcnt_barrier_thread0_system_store_n_tiles",
                "completion_rmw_in_unified_loop": "forbidden_transitively",
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
            "arguments": [{}],
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
    assert not valid
    assert any("combine_in_kernel" in error for error in errors)


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


def test_publication_placement_rejects_rmw_anywhere_in_hot_loop():
    contract = _contract()
    contract["checks"] = [{
        "id": "publish_frame",
        "kind": "publication_placement",
        "file": "src/a.py",
        "scope": "kernel",
        "loop_test": "consumer_active",
        "compute_call": "_compute",
        "address": "ready_addr",
        "rmw_calls": ["atomic_add_system", "atomic_add_agent"],
        "store_calls": ["store_i32_system"],
    }]
    unsafe = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": unsafe})
    assert not result["publish_frame"]["pass"]
    assert any(
        "inside hot loop" in failure
        for failure in result["publish_frame"]["failures"]
    )

    flushed = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "    _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": flushed})
    assert result["publish_frame"]["pass"]

    owned = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.store_i32_system(ready_addr, 0, 1)\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        _publish()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": owned})
    assert result["publish_frame"]["pass"]

    loop_head = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.atomic_add_system(ready_addr, 1)\n"
        "    while consumer_active:\n"
        "        _publish()\n"
        "        _compute()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": loop_head})
    assert not result["publish_frame"]["pass"]

    wrapped = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _raw_publish():\n"
        "        ops.atomic_add_agent(ready_addr, 1)\n"
        "    def renamed_wrapper():\n"
        "        _raw_publish()\n"
        "    while consumer_active:\n"
        "        _compute()\n"
        "        renamed_wrapper()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": wrapped})
    assert not result["publish_frame"]["pass"]

    dead_only = ast.parse(
        "def kernel(consumer_active, ready_addr):\n"
        "    def _publish():\n"
        "        ops.store_i32_system(ready_addr, 0, 1)\n"
        "    if False:\n"
        "        _publish()\n"
        "    while consumer_active:\n"
        "        _compute()\n"
    )
    result = MODULE.evaluate_checks(contract, {"src/a.py": dead_only})
    assert not result["publish_frame"]["pass"]


def _owned_protocol_rule():
    return {
        "id": "owned_completion",
        "kind": "owned_completion_protocol",
        "file": "src/a.py",
        "scope": "kernel",
        "normalize_wrappers": ["Int32", "Int64", "int"],
        "region": {
            "guard": "fuse_all",
            "arm": "body",
            "loop": {"test": "consumer_active", "contains": r"\bg2_pend\b"},
        },
        "claim": {
            "extent_target": "g1_total_mtiles",
            "extent_value": "num_m_tiles",
            "source_extent_target": "num_m_tiles",
            "source_extent_value": r"ceildiv\(num_valid, sort_block_m\)",
            "helper": "_claim",
            "owner_test": "tid == 0",
            "local_target": "local_claim",
            "atomic_call": "atomic_add_agent",
            "head_address": r"a_work_head \+ work_shard \* 64",
            "increment": "1",
            "index_target": "work",
            "index_value": r"work_shard \+ local_claim \* WORK_SHARDS",
            "bound": "work < g1_total_mtiles",
        },
        "handoff": {
            "store_call": "ptr_store",
            "scratch": "scratch",
            "vector_call": "from_elements",
            "values": ["1", "work", "-1"],
            "view_target": "fdrain_view",
            "view_value": r"make_view\(scratch, make_layout\(3, 1\)\)",
            "load_target": "_vals",
            "load_value": r"Vec\(fdrain_view\.load\(\)\)",
            "barrier_call": "barrier",
            "unit_target": "unit",
            "unit_value": r"_vals\[1\]",
        },
        "stripes": {
            "mode_test": "kind == 1",
            "iterator": "stripe",
            "range_call": "range",
            "count": "N_TILES",
            "compute_call": "_compute",
            "compute_index": r"unit \* N_TILES \+ stripe",
        },
        "publication": {
            "helper": "_publish",
            "argument": "unit",
            "wait_call": "s_waitcnt",
            "wait_argument": "0",
            "barrier_call": "barrier",
            "owner_test": "tid == 0",
            "store_call": "store_i32_system",
            "address_call": "_ready_addr",
            "address_argument": "m_tile",
            "direct_address_patterns": [
                r"ready_base \+ .*\bunit\b.* \* 4",
                r"4 \* .*\bunit\b.* \+ ready_base",
            ],
            "address_base_patterns": [r"^ready_base$"],
            "address_index_patterns": [
                r"\bunit\b.* \* 4",
                r"4 \* .*\bunit\b",
            ],
            "store_offset": "0",
            "store_value": "N_TILES",
            "forbidden_rmw_calls": [
                "atomic_add_system", "atomic_add_agent",
                "atomic_add_global_at", "AtomicRMWOp",
            ],
            "forbidden_calls": ["fence_system_release"],
        },
    }


def _owned_protocol_source():
    return (
        "def kernel(fuse_all, consumer_active, g2_pend, num_valid, sort_block_m, "
        "a_work_head, work_shard, tid, peer_ready):\n"
        "    num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))\n"
        "    if fuse_all:\n"
        "        g1_total_mtiles = num_m_tiles\n"
        "        def _ready_addr(m_tile):\n"
        "            return m_tile\n"
        "        def _publish(m_tile):\n"
        "            ops.s_waitcnt(Int32(0))\n"
        "            ops.barrier()\n"
        "            if tid == Int32(0):\n"
        "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
        "Int32(N_TILES))\n"
        "        def _peer_publish(addr):\n"
        "            ops.atomic_add_system(addr, Int32(1))\n"
        "        fdrain_view = make_view(scratch, make_layout(3, 1))\n"
        "        while consumer_active:\n"
        "            g2_pend = g2_pend\n"
        "            def _claim():\n"
        "                local_claim = Int32(ops.atomic_add_agent("
        "a_work_head + Int64(work_shard) * Int64(64), Int32(1)))\n"
        "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n"
        "                if work < g1_total_mtiles:\n"
        "                    ptr_store(Vec.from_elements("
        "[Int32(1), work, Int32(-1)], Int32), scratch)\n"
        "                else:\n"
        "                    ptr_store(Vec.from_elements("
        "[Int32(0), Int32(0), Int32(-1)], Int32), scratch)\n"
        "            if tid == Int32(0):\n"
        "                _claim()\n"
        "            _peer_publish(peer_ready)\n"
        "            ops.barrier()\n"
        "            _vals = Vec(fdrain_view.load())\n"
        "            kind = _vals[0]\n"
        "            unit = _vals[1]\n"
        "            if kind == Int32(1):\n"
        "                for stripe in range(int(N_TILES)):\n"
        "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n"
        "                _publish(unit)\n"
    )


def _owned_protocol_result(source):
    contract = _contract()
    contract["checks"] = [_owned_protocol_rule()]
    return MODULE.evaluate_checks(
        contract, {"src/a.py": ast.parse(source)}
    )["owned_completion"]


def test_owned_completion_protocol_accepts_owner_store_and_unrelated_atomics():
    result = _owned_protocol_result(_owned_protocol_source())
    assert result["pass"], result["failures"]
    assert result["units_passed"] == result["units_total"] == 9


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))",
            "num_m_tiles = num_valid",
        ),
        (
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))",
            "num_m_tiles = ceildiv(num_valid, Int32(sort_block_m))\n"
            "    num_m_tiles = num_valid",
        ),
        ("g1_total_mtiles = num_m_tiles", "g1_total_mtiles = num_m_tiles * N_TILES"),
        (
            "work_shard + local_claim * Int32(WORK_SHARDS)",
            "local_claim",
        ),
        (
            "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n",
            "                work = work_shard + local_claim * Int32(WORK_SHARDS)\n"
            "                work = Int32(0)\n",
        ),
        ("work < g1_total_mtiles", "work < num_m_tiles * N_TILES"),
        ("range(int(N_TILES))", "range(int(N_TILES - 1))"),
        (
            "unit * Int32(N_TILES) + Int32(stripe)",
            "unit * Int32(N_TILES) + Int32(stripe) + Int32(1)",
        ),
        ("ops.s_waitcnt(Int32(0))\n            ops.barrier()",
         "ops.barrier()\n            ops.s_waitcnt(Int32(0))"),
        ("tid == Int32(0)", "tid == Int32(1)"),
        ("Int32(N_TILES))\n", "Int32(1))\n"),
        (
            "ops.store_i32_system(_ready_addr(m_tile), Int32(0), Int32(N_TILES))",
            "ops.store_i32_system(_ready_addr(m_tile + Int32(1)), Int32(0), "
            "Int32(N_TILES))",
        ),
        (
            "ops.store_i32_system(_ready_addr(m_tile), Int32(0), Int32(N_TILES))",
            "ops.atomic_add_system(_ready_addr(m_tile), Int32(1))",
        ),
    ],
)
def test_owned_completion_protocol_rejects_relation_mutations(old, new):
    source = _owned_protocol_source()
    assert old in source
    result = _owned_protocol_result(source.replace(old, new, 1))
    assert not result["pass"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            pass\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            _claim()\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            if tid == Int32(0):\n"
            "                if allow_claim:\n"
            "                    _claim()\n",
        ),
        (
            "[Int32(1), work, Int32(-1)], Int32), scratch)",
            "[Int32(1), work, Int32(-1)], Int32), other_scratch)",
        ),
        (
            "                    ptr_store(Vec.from_elements("
            "[Int32(1), work, Int32(-1)], Int32), scratch)\n",
            "                    if publish_scratch:\n"
            "                        ptr_store(Vec.from_elements("
            "[Int32(1), work, Int32(-1)], Int32), scratch)\n",
        ),
        (
            "            ops.barrier()\n"
            "            _vals = Vec(fdrain_view.load())\n",
            "            _vals = Vec(fdrain_view.load())\n",
        ),
        (
            "            unit = _vals[1]\n",
            "            if load_unit:\n"
            "                unit = _vals[1]\n",
        ),
        (
            "            unit = _vals[1]\n",
            "            unit = _vals[1]\n"
            "            unit = Int32(0)\n",
        ),
        (
            "                _publish(unit)\n",
            "                if unit >= Int32(0):\n"
            "                    _publish(unit)\n",
        ),
        (
            "                _publish(unit)\n",
            "                continue\n"
            "                _publish(unit)\n",
        ),
        (
            "            if tid == Int32(0):\n"
            "                _claim()\n",
            "            continue\n"
            "            if tid == Int32(0):\n"
            "                _claim()\n",
        ),
        (
            "                _publish(unit)\n",
            "                _publish = lambda value: None\n"
            "                _publish(unit)\n",
        ),
        (
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n"
            "                    break\n",
        ),
        (
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
            "                    ops.store_i32_system(_ready_addr(unit), Int32(0), "
            "Int32(N_TILES))\n"
            "                    _compute(unit * Int32(N_TILES) + Int32(stripe))\n",
        ),
        (
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n",
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n"
            "            else:\n"
            "                ops.store_i32_system(_ready_addr(m_tile), Int32(0), "
            "Int32(N_TILES))\n",
        ),
        (
            "                _publish(unit)\n",
            "                ops.fence_system_release()\n"
            "                _publish(unit)\n",
        ),
        (
            "        def _publish(m_tile):\n"
            "            ops.s_waitcnt(Int32(0))\n",
            "        def _publish(m_tile):\n"
            "            m_tile = Int32(0)\n"
            "            ops.s_waitcnt(Int32(0))\n",
        ),
    ],
)
def test_owned_completion_protocol_rejects_cfg_and_handoff_bypasses(old, new):
    source = _owned_protocol_source()
    assert old in source
    result = _owned_protocol_result(source.replace(old, new, 1))
    assert not result["pass"]


@pytest.mark.parametrize(
    "extra",
    [
        (
            "                completion_alias = _ready_addr(unit)\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                completion_alias = _ready_addr(unit)\n"
            "                _peer_publish(completion_alias)\n"
        ),
        (
            "                ops.atomic_add_system("
            "ready_base + unit * Int32(4), Int32(1))\n"
        ),
        (
            "                ops.atomic_add_system("
            "Int32(4) * unit + ready_base, Int32(1))\n"
        ),
        (
            "                ops.atomic_add_global_at("
            "_ready_addr(unit), Int32(1))\n"
        ),
        (
            "                base_alias = ready_base\n"
            "                offset_alias = unit * Int32(4)\n"
            "                completion_alias = base_alias + offset_alias\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                address_factory = _ready_addr\n"
            "                completion_alias = address_factory(unit)\n"
            "                ops.atomic_add_system(completion_alias, Int32(1))\n"
        ),
        (
            "                bad_rmw = ops.atomic_add_system\n"
            "                bad_rmw(_ready_addr(unit), Int32(1))\n"
        ),
    ],
)
def test_owned_completion_protocol_rejects_region_rmw_aliases(extra):
    source = _owned_protocol_source().replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n" + extra,
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_ignores_unselected_standalone_arm():
    source = _owned_protocol_source() + (
        "    else:\n"
        "        ops.atomic_add_system(_ready_addr(Int32(0)), Int32(1))\n"
    )
    result = _owned_protocol_result(source)
    assert result["pass"], result["failures"]


def test_owned_completion_protocol_rejects_transitive_completion_rmw():
    source = _owned_protocol_source().replace(
        "        while consumer_active:\n",
        "        def _raw(addr):\n"
        "            ops.atomic_add_agent(addr, Int32(1))\n"
        "        def _wrapped(addr):\n"
        "            _raw(addr)\n"
        "        while consumer_active:\n",
    ).replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n"
        "                _wrapped(_ready_addr(unit))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_tracks_aliases_and_keyword_forwarding():
    source = _owned_protocol_source().replace(
        "        while consumer_active:\n",
        "        def _address_alias(m_tile):\n"
        "            return _ready_addr(m_tile)\n"
        "        def _raw(addr):\n"
        "            first = addr\n"
        "            second = first\n"
        "            ops.atomic_add_system(second, Int32(1))\n"
        "        def _wrapped(*, target):\n"
        "            _raw(addr=target)\n"
        "        while consumer_active:\n",
    ).replace(
        "                _publish(unit)\n",
        "                _publish(unit)\n"
        "                _wrapped(target=_address_alias(unit))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("RMW" in failure for failure in result["failures"])


def test_owned_completion_protocol_rejects_per_item_fence():
    source = _owned_protocol_source().replace(
        "            ops.s_waitcnt(Int32(0))\n",
        "            ops.fence_system_release()\n"
        "            ops.s_waitcnt(Int32(0))\n",
    )
    result = _owned_protocol_result(source)
    assert not result["pass"]
    assert any("forbidden" in failure for failure in result["failures"])


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


def test_reference_parity_gaps_are_informational_unless_declared(tmp_path):
    contract = _contract()
    baseline = _root(tmp_path, "baseline", "def old():\n    pass\n")
    reference = _root(
        tmp_path, "reference",
        "def new():\n    return 1\n\ndef reference_extra():\n    return 2\n",
    )
    candidate = _root(
        tmp_path, "candidate",
        "def new():\n    value = 1\n    return value\n",
    )
    result = MODULE.evaluate(
        contract, baseline, candidate, reference,
        _manifest(contract, candidate), _plan(),
    )
    assert result["missing_new_symbols"]
    assert result["required_changed_files_missing"] == []
    assert result["contract_failures"] == []
    assert result["independent_structure_pass"]
    assert result["capability_eligible"]


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
