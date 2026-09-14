#!/usr/bin/env python3
"""Offline post-authoring comparison against the pinned M2.5 source oracle.

The author must not receive the oracle. This tool never executes GPU code, and
an exact oracle copy is checker calibration rather than capability evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path


MEGA_REL = Path("aiter/ops/flydsl/kernels/mega_moe")
HOST_REL = Path("aiter/ops/flydsl/kernels")
HOST_FILES = (
    "communication_ops_utils.py",
    "flydsl_dispatch_combine_intranode_kernel.py",
    "flydsl_dispatch_combine_intranode_op.py",
)
SCHEMA_VERSION = "m25-structural-v2"
PINNED_BASELINE_DIGEST = "8dcd670cdae4553ccb9f2497a9ceb6799db0a0537f3b06ad0124acda5aec7f93"
PINNED_ORACLE_DIGEST = "1e66d689c2a76a01889c1af91f816a1dbb43c032271cfde4052ba3fbe695a1ef"
PINNED_ORACLE_REVISION = "494c25a7e016b83ce4fd2bb2de148b602f7a5661"
ENFORCE_PINNED_INPUTS = True
ENFORCE_CORRECTED_FACTS = True
CORRECTED_FACTS = {
    "all_a_lds_calls_forward_nw",
    "token_ready_lds_uses_slab_base",
    "token_publish_requires_write_through",
    "preemption_preserves_g1_cohort",
    "chunk_checks_all_stage1_dependencies",
    "m25_variant_is_sanitized",
}

FEATURES = {
    "host_two_launch_full_fusion": (
        ("mega_moe_v2.py", (
            r"AITER_MEGAMOE_FUSE_ALL", r"num_waves\s*%\s*4",
            r"_run_fused_stage1",
        )),
    ),
    "fixed_group_done_capacity": (
        ("mega_moe_v2.py", (
            r"(?:torch\.zeros\(self\.world_size|_group_done_slots\(self\.world_size\))",
        )),
    ),
    "fused_output_view_contract": (
        ("mega_moe_v2.py", (
            r"shmem_comb_out_tok", r"combine_dtype", r"combine_token_view_dim",
            r"out_tok\[:run_tokens\]\s*if slice_output else out_tok",
        )),
    ),
    "shared_num_valid_bound": (
        ("mega_moe_v2.py", (r"op\.num_valid\.data_ptr\(\)",)),
        ("mega_moe_stage1.py", (r"num_valid", r"s2_cumsum")),
        ("mega_moe_stage2.py", (r"arg_cumsum", r"total_m_blocks")),
    ),
    "direct_fused_kernel_abi": (
        ("mega_moe_stage1.py", (
            r"s2_aq:\s*fx\.Int64", r"s2_ctr:\s*fx\.Int64",
            r"c_tok_ready:\s*fx\.Int64",
            r"(?:c_mtile_ctr|s2_mtile_ctr):\s*fx\.Int64",
        )),
    ),
    "nested_shared_stage2_emitter": (
        ("mega_moe_stage2.py", (
            r"def make_stage2_body_emitter", r"def emit_stage2_body",
            r"@flyc\.jit\s*\n\s*def _emit_stage2_body",
            r"publish_tok_ready", r"arg_mtile_ctr",
        )),
    ),
    "unified_gemm_queue": (
        ("mega_moe_stage1.py", (
            r"while consumer_active", r"\bkind\b", r"\bunit\b",
            r"\bg2_pend\b", r"\bg2_next\b",
            r"(?:total_g2_pairs|g2_total_units)",
            r"""(?:S2\[['"]emit['"]\]|emit_stage2_body)""",
        )),
    ),
    "route_gated_g2_preemption": (
        ("mega_moe_stage1.py", (
            r"fused_g2_pref", r"(?:\bg2_pref\b|\bpreempt_gate\b)", r"\bpeek",
            r"(?:atomic_add_agent\([^)]*fx\.Int32\(0\)|buffer_load\([^)]*g2_head)",
            r"(?:\bgot\b|\blocal_claim\b)",
            r"(?:total_g2_claims|g2_total_claims|g2_total_units)",
        )),
    ),
    "g1_to_g2_publication": (
        ("mega_moe_stage1.py", (
            r"_do_scheduled_tile\(unit\)", r"s_waitcnt\(0\)", r"fx\.barrier\(\)",
            r"atomic_add_system",
            r"(?:unit\s*//\s*n_tiles_i32|s1_m_tile\s*=\s*unit\s*//\s*fx\.Int32\(N_TILES\))",
            r"int32_wait_until_greater_than",
            r"(?:n_tiles_i32\s*-\s*fx\.Int32\(1\)|fx\.Int32\(N_TILES\s*-\s*1\))",
        )),
    ),
    "tile_close_and_token_publication": (
        ("mega_moe_stage2.py", (
            r"(?:def _publish_tok_ready|publish_tok_ready)", r"atomic_add_agent",
            r"num_n_blocks\s*-\s*fx\.Int32\(1\)", r"atomic_add_system",
            r"(?:ready_base|peer_ready)", r"arg_mtile_ctr",
        )),
    ),
    "combine_third_queue": (
        ("mega_moe_stage1.py", (
            r"(?:fused_combine|fuse_all)", r"\bc_ctr\b", r"\bc_epoch\b",
            r"while (?:comb|combine)_active", r"(?:_c_claim|block_claim)",
            r"make_combine_reduce_emitter",
            r"tok_ready_epoch=c_epoch",
        )),
    ),
    "generation_and_arrival_lifetime": (
        ("mega_moe_stage1.py", (
            r"c_ctr", r"c_tok_ready",
            r"fx\.Int32\(fz_k\)\s*\*\s*c_epoch",
        )),
    ),
    "native_nw8_geometry": (
        ("mega_moe_stage2.py", (
            r"NW not in \(4,\s*8\)", r"BN\s*//\s*NW",
            r"BM\s*//\s*NW", r"row_iter\s*\*\s*NW",
        )),
        ("gemm2.py", (r"NW=4", r"BN\s*//\s*NW")),
    ),
    "lds_role_aliasing": (
        ("mega_moe_stage1.py", (
            r"(?:\bS2_SLAB\b|stage2_constants\.slab_bytes)",
            r"lds_pool_bytes\s*=\s*max\(",
        )),
    ),
    "write_through_publication": (
        ("mega_moe_stage1.py", (r"out_cache_modifier",)),
        ("mega_moe_stage2.py", (
            r"(?:p2p_write_through|cache_modifier\s*=\s*17)",
            r"(?:_P2P_CACHE_WT|cache_modifier\s*=\s*17)",
        )),
    ),
}

SCOPED_FEATURES = {
    "reachable_host_fused_path": (
        "mega_moe_v2.py",
        "MegaMoEV2._run_joint",
        (
            r"(?:AITER_MEGAMOE_FUSE_ALL|_m25_variant_enabled)",
            r"_run_fused_stage1",
            r"(?:shmem_comb_out_tok|_combine_output_view)",
            r"return out_tok",
        ),
    ),
    "stage1_single_compiled_pipeline": (
        "mega_moe_stage1.py",
        "compile_mega_moe_stage1",
        (
            r"while consumer_active", r"\bg2_pend\b", r"\bg2_next\b",
            r"""(?:S2\[['"]emit['"]\]|emit_stage2_body)""", r"atomic_add_system",
            r"while (?:comb|combine)_active",
        ),
    ),
    "stage2_nested_compiled_emitter": (
        "mega_moe_stage2.py",
        "make_stage2_body_emitter",
        (
            r"def emit_stage2_body", r"@flyc\.jit\s*\n\s*def _emit_stage2_body",
            r"p2p_scatter_epilog", r"(?:def _publish_tok_ready|publish_tok_ready)",
        ),
    ),
}

CORE_IDENTITY_FILES = (
    "gemm2.py",
    "mega_moe_stage1.py",
    "mega_moe_stage2.py",
    "mega_moe_v2.py",
)

FORBIDDEN_DIAGNOSTICS = (
    "AITER_MEGAMOE_G2_DIAG",
    "AITER_MEGAMOE_SKIP_COMBINE",
    "AITER_MEGAMOE_DISP_DEBUG",
    "AITER_MEGAMOE_FUSE_DRAIN",
)


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _files(root: Path) -> dict[str, Path]:
    base = root / MEGA_REL
    files = {
        path.relative_to(base).as_posix(): path
        for path in sorted(base.glob("*.py"))
    }
    host = root / HOST_REL
    for name in HOST_FILES:
        path = host / name
        if path.is_file():
            files[name] = path
    return files


class _RemoveStaticDeadCode(ast.NodeTransformer):
    def visit_If(self, node: ast.If):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
            return node.body if node.test.value else node.orelse
        return node

    def visit_While(self, node: ast.While):  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            return node.orelse
        return node

    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):  # noqa: N802
        if isinstance(node.value, str):
            if node.value.startswith("AITER_") or node.value in {"emit"}:
                return node
            return ast.copy_location(ast.Constant(value=""), node)
        return node


def _active_code(tree: ast.AST) -> str:
    tree = _RemoveStaticDeadCode().visit(ast.fix_missing_locations(tree))
    return ast.unparse(tree)


def _normalized_tokens(path: Path) -> tuple[str, ...]:
    keep = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        for token in tokens:
            if token.type in {
                tokenize.ENCODING,
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            }:
                continue
            keep.append(token.string)
    except (OSError, UnicodeError, tokenize.TokenError):
        return ()
    return tuple(keep)


def _shingle_similarity(left: tuple[str, ...], right: tuple[str, ...], width: int = 7) -> float:
    def shingles(tokens: tuple[str, ...]) -> set[tuple[str, ...]]:
        if len(tokens) < width:
            return {tokens} if tokens else set()
        return {
            tokens[index:index + width]
            for index in range(len(tokens) - width + 1)
        }

    left_set = shingles(left)
    right_set = shingles(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _tree_digest(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _input_errors(
    baseline: Path,
    reference: Path,
    candidate: Path,
) -> tuple[list[str], list[str], list[str]]:
    errors = []
    symlinks = []
    hardlinks = []
    for label, root in (
        ("baseline", baseline),
        ("reference", reference),
        ("candidate", candidate),
    ):
        if not root.is_dir():
            errors.append(f"{label} root is not a directory: {root}")
            continue
        for name in CORE_IDENTITY_FILES:
            path = root / MEGA_REL / name
            if not path.is_file():
                errors.append(f"{label} missing required core file: {name}")
    if errors:
        return errors, symlinks, hardlinks
    ref_files = _files(reference)
    cand_files = _files(candidate)
    if ENFORCE_PINNED_INPUTS:
        baseline_digest = _tree_digest(_files(baseline))
        reference_digest = _tree_digest(ref_files)
        if baseline_digest != PINNED_BASELINE_DIGEST:
            errors.append(
                f"baseline digest mismatch: {baseline_digest} != {PINNED_BASELINE_DIGEST}"
            )
        if reference_digest != PINNED_ORACLE_DIGEST:
            errors.append(
                f"reference digest mismatch: {reference_digest} != {PINNED_ORACLE_DIGEST}"
            )
    for name, path in cand_files.items():
        if path.is_symlink():
            symlinks.append(name)
        ref = ref_files.get(name)
        if ref and path.resolve() != ref.resolve():
            try:
                if path.stat().st_dev == ref.stat().st_dev and path.stat().st_ino == ref.stat().st_ino:
                    hardlinks.append(name)
            except OSError:
                pass
    return errors, sorted(symlinks), sorted(hardlinks)


def _qualified_functions(tree: ast.AST) -> set[str]:
    found: set[str] = set()

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                found.add(name)
                visit(node.body, f"{name}.")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(getattr(tree, "body", []))
    return found


def _qualified_function_nodes(tree: ast.AST) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}

    def visit(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                found[name] = node
                visit(node.body, f"{name}.")
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(getattr(tree, "body", []))
    return found


def _function_dumps(tree: ast.AST) -> set[str]:
    return {
        ast.dump(node, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _call_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _names_in(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
    }


def _parse_files(files: dict[str, Path]) -> tuple[dict[str, ast.AST], dict[str, str]]:
    trees: dict[str, ast.AST] = {}
    errors: dict[str, str] = {}
    for name, path in files.items():
        try:
            trees[name] = ast.parse(path.read_text(), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors[name] = str(exc)
    return trees, errors


def _validate_plan(plan: dict | None) -> tuple[bool | None, list[str]]:
    if plan is None:
        return None, []
    errors = []
    resource = plan.get("resource_contract") or {}
    lds = resource.get("lds") or {}
    registers = resource.get("registers") or {}
    residency = resource.get("residency") or {}
    schedule = plan.get("schedule_contract") or {}
    abi = plan.get("abi") or {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(plan.get("target_launches") == 2, "target_launches must be 2")
    require(resource.get("arch") == "gfx950", "resource arch must be gfx950")
    require(resource.get("wave_size") == 64, "wave_size must be 64")
    require(resource.get("num_waves") in (4, 8), "num_waves must be 4 or 8")
    require(
        resource.get("threads_per_workgroup")
        == resource.get("wave_size", 0) * resource.get("num_waves", 0),
        "threads_per_workgroup must equal wave_size*num_waves",
    )
    numeric_lds = all(
        isinstance(lds.get(name), (int, float))
        for name in (
            "stage1_pool_bytes", "stage2_slab_bytes", "additive_bytes",
            "group_segment_bytes", "limit_bytes", "halves",
        )
    )
    require(numeric_lds, "LDS byte fields and halves must be numeric")
    if numeric_lds:
        expected_group = max(
            lds["stage1_pool_bytes"],
            lds["stage2_slab_bytes"] * lds["halves"],
        ) + lds["additive_bytes"]
        require(
            lds["group_segment_bytes"] == expected_group,
            f"group_segment_bytes must equal aliased-role max + additive ({expected_group})",
        )
        require(
            lds["group_segment_bytes"] <= lds["limit_bytes"],
            "group segment exceeds declared limit",
        )
    require(lds.get("aliasing") == "max", "Stage1/Stage2 LDS must alias by max")
    require(registers.get("scratch_bytes_max") == 0, "scratch_bytes_max must be 0")
    require(
        isinstance(residency.get("min_workgroups_per_cu"), (int, float))
        and residency["min_workgroups_per_cu"] >= 1,
        "residency must allow at least one workgroup per CU",
    )
    require(schedule.get("unified_gemm_loop") is True, "unified_gemm_loop must be true")
    require(
        schedule.get("carried_scalars")
        == ["consumer_active", "g2_pend", "g2_next"],
        "carried_scalars must be exactly consumer_active,g2_pend,g2_next",
    )
    require(schedule.get("combine_third_queue") is True, "combine third queue is required")
    require(schedule.get("g2_chunk_large") == 16, "large G2 chunk must be 16")
    require(schedule.get("g2_chunk_small") == 1, "small G2 chunk must be 1")
    require(
        schedule.get("preemption_interval") == 0
        or (
            isinstance(schedule.get("preemption_interval"), (int, float))
            and schedule["preemption_interval"] >= 2
        ),
        "preemption interval must be 0 or >=2",
    )
    require(schedule.get("skew_num", 0) > 0, "skew_num must be positive")
    require(schedule.get("skew_den", 0) > 0, "skew_den must be positive")
    require(schedule.get("work_shards") in (1, 2, 4, 8), "work_shards invalid")
    require(abi.get("direct_fused_args") is True, "direct fused ABI is required")
    require(abi.get("stage2_pointer_count") == 12, "Stage2 ABI must carry 12 pointers")
    require(abi.get("combine_pointer_count") in (6, 7), "combine ABI must carry 6 or 7 pointers")
    require(
        abi.get("optional_quant_pointer_count") in (0, 2),
        "optional quant ABI must carry 0 or 2 pointers",
    )
    require(abi.get("disabled_placeholders") is True, "disabled ABI groups need placeholders")
    require(bool(abi.get("argument_order")), "ABI argument_order must be non-empty")
    return not errors, errors


def evaluate(
    baseline: Path,
    reference: Path,
    candidate: Path,
    authoring_manifest: dict | None = None,
    plan_ir: dict | None = None,
) -> dict:
    input_errors, candidate_symlinks, oracle_hardlinks = _input_errors(
        baseline, reference, candidate
    )
    base_files = _files(baseline)
    ref_files = _files(reference)
    cand_files = _files(candidate)
    all_names = sorted(set(base_files) | set(ref_files) | set(cand_files))

    changed_by_reference = [
        name for name in all_names
        if _digest(base_files.get(name, Path())) != _digest(ref_files.get(name, Path()))
    ]
    changed_by_candidate = [
        name for name in all_names
        if _digest(base_files.get(name, Path())) != _digest(cand_files.get(name, Path()))
    ]
    exact_reference_files = [
        name for name in all_names
        if _digest(ref_files.get(name, Path())) == _digest(cand_files.get(name, Path()))
    ]

    base_trees, base_errors = _parse_files(base_files)
    ref_trees, ref_errors = _parse_files(ref_files)
    cand_trees, cand_errors = _parse_files(cand_files)

    required_symbols: dict[str, list[str]] = {}
    missing_symbols: dict[str, list[str]] = {}
    for name, ref_tree in ref_trees.items():
        base_symbols = _qualified_functions(base_trees[name]) if name in base_trees else set()
        new_symbols = sorted(_qualified_functions(ref_tree) - base_symbols)
        if not new_symbols:
            continue
        required_symbols[name] = new_symbols
        candidate_symbols = (
            _qualified_functions(cand_trees[name]) if name in cand_trees else set()
        )
        missing = sorted(set(new_symbols) - candidate_symbols)
        if missing:
            missing_symbols[name] = missing

    feature_results = {}
    for feature, checks in FEATURES.items():
        missing = []
        pattern_total = 0
        for name, patterns in checks:
            text = _active_code(cand_trees[name]) if name in cand_trees else ""
            pattern_total += len(patterns)
            missing.extend(
                f"{name}:{pattern}"
                for pattern in patterns
                if not re.search(pattern, text)
            )
        pattern_passed = pattern_total - len(missing)
        feature_results[feature] = {
            "pass": not missing,
            "patterns_passed": pattern_passed,
            "patterns_total": pattern_total,
            "pattern_ratio": pattern_passed / pattern_total if pattern_total else 0.0,
            "missing_patterns": missing,
        }
    for feature, (name, scope, patterns) in SCOPED_FEATURES.items():
        nodes = (
            _qualified_function_nodes(cand_trees[name])
            if name in cand_trees else {}
        )
        text = _active_code(nodes[scope]) if scope in nodes else ""
        missing = [
            f"{name}:{scope}:{pattern}"
            for pattern in patterns
            if not re.search(pattern, text)
        ]
        pattern_total = len(patterns)
        pattern_passed = pattern_total - len(missing)
        feature_results[feature] = {
            "pass": not missing,
            "patterns_passed": pattern_passed,
            "patterns_total": pattern_total,
            "pattern_ratio": pattern_passed / pattern_total if pattern_total else 0.0,
            "missing_patterns": missing,
        }

    all_candidate_nodes = [
        node
        for tree in cand_trees.values()
        for node in ast.walk(tree)
    ]
    nw_calls = [
        node for node in all_candidate_nodes
        if isinstance(node, ast.Call) and _call_name(node) == "issue_a_load_lds_dt"
    ]
    nw_missing = [
        getattr(node, "lineno", 0)
        for node in nw_calls
        if not any(keyword.arg == "NW" for keyword in node.keywords)
    ]
    feature_results["all_a_lds_calls_forward_nw"] = {
        "pass": bool(nw_calls) and not nw_missing,
        "patterns_passed": len(nw_calls) - len(nw_missing),
        "patterns_total": max(1, len(nw_calls)),
        "pattern_ratio": (
            (len(nw_calls) - len(nw_missing)) / len(nw_calls)
            if nw_calls else 0.0
        ),
        "missing_patterns": [
            f"issue_a_load_lds_dt line {line} omits NW"
            for line in nw_missing
        ] or ([] if nw_calls else ["no issue_a_load_lds_dt call found"]),
    }

    stage2_nodes = (
        _qualified_function_nodes(cand_trees["mega_moe_stage2.py"])
        if "mega_moe_stage2.py" in cand_trees else {}
    )
    publish_nodes = [
        node for name, node in stage2_nodes.items()
        if name.endswith("._publish_tok_ready")
    ]
    if not publish_nodes:
        publish_nodes = [
            node for name, node in stage2_nodes.items()
            if name == "make_stage2_body_emitter"
        ]
    ready_lds_calls = []
    for publish_node in publish_nodes:
        for node in ast.walk(publish_node):
            if isinstance(node, ast.Call) and _call_name(node) == "lds_typed_ptr" and node.args:
                address = ast.unparse(node.args[0])
                if "lds_ready_off" in address or "lds_tok_ready_off" in address:
                    ready_lds_calls.append((getattr(node, "lineno", 0), address))
    unbased_ready_calls = [
        (line, address)
        for line, address in ready_lds_calls
        if "lds_base_i32" not in address
    ]
    ready_rebased = bool(ready_lds_calls) and not unbased_ready_calls
    feature_results["token_ready_lds_uses_slab_base"] = {
        "pass": ready_rebased,
        "patterns_passed": int(ready_rebased),
        "patterns_total": 1,
        "pattern_ratio": float(ready_rebased),
        "missing_patterns": (
            []
            if ready_rebased
            else [
                "_publish_tok_ready LDS lookup line "
                f"{line} is not based on lds_base_i32: {address}"
                for line, address in unbased_ready_calls
            ] or ["no _publish_tok_ready lds_ready_off lookup found"]
        ),
    }

    stage2_tree = cand_trees.get("mega_moe_stage2.py")
    wt_guard = False
    if stage2_tree is not None:
        for node in ast.walk(stage2_tree):
            if isinstance(node, (ast.If, ast.Assert)):
                names = _names_in(node.test)
                if {"publish_tok_ready", "p2p_write_through"} <= names:
                    wt_guard = True
                    break
        if not wt_guard:
            stage2_code = _active_code(stage2_tree)
            wt_guard = (
                bool(re.search(r"cache_modifier\s*=\s*17", stage2_code))
                and "AITER_MEGAMOE_P2P_WT" not in stage2_code
            )
    feature_results["token_publish_requires_write_through"] = {
        "pass": wt_guard,
        "patterns_passed": int(wt_guard),
        "patterns_total": 1,
        "pattern_ratio": float(wt_guard),
        "missing_patterns": [] if wt_guard else [
            "no compile-time publish_tok_ready => p2p_write_through guard"
        ],
    }

    stage1_tree = cand_trees.get("mega_moe_stage1.py")
    pref_guard = False
    if stage1_tree is not None:
        for node in ast.walk(stage1_tree):
            if isinstance(node, (ast.If, ast.Assert)):
                text = ast.unparse(node.test)
                if (
                    "fused_g2_pref" in text
                    and re.search(r"(>=\s*2|==\s*0|not\s+.*fused_g2_pref)", text)
                ):
                    pref_guard = True
                    break
    feature_results["preemption_preserves_g1_cohort"] = {
        "pass": pref_guard,
        "patterns_passed": int(pref_guard),
        "patterns_total": 1,
        "pattern_ratio": float(pref_guard),
        "missing_patterns": [] if pref_guard else [
            "no guard requiring fused_g2_pref == 0 or >= 2"
        ],
    }

    chunk_helpers = []
    if stage1_tree is not None:
        chunk_helpers = [
            node for name, node in _qualified_function_nodes(stage1_tree).items()
            if name.endswith("._g2_chunk_ready_all")
        ]
    chunk_safe = any(
        "int32_wait_until" in _active_code(node)
        for node in chunk_helpers
    )
    feature_results["chunk_checks_all_stage1_dependencies"] = {
        "pass": chunk_safe,
        "patterns_passed": int(chunk_safe),
        "patterns_total": 1,
        "pattern_ratio": float(chunk_safe),
        "missing_patterns": [] if chunk_safe else [
            "missing _g2_chunk_ready_all helper that waits every distinct Stage1 dependency"
        ],
    }

    v2_nodes = (
        _qualified_function_nodes(cand_trees["mega_moe_v2.py"])
        if "mega_moe_v2.py" in cand_trees else {}
    )
    variant_nodes = [
        node for name, node in v2_nodes.items()
        if name.endswith("._m25_variant_enabled")
        or name == "_m25_variant_enabled"
    ]
    variant_text = "\n".join(_active_code(node) for node in variant_nodes)
    variant_markers = (
        "AITER_MEGAMOE_FUSE_ALL",
        "AITER_MEGAMOE_FUSE_COMBINE",
        "AITER_MEGAMOE_FUSE_QUANT",
    )
    variant_missing = [
        marker for marker in variant_markers if marker not in variant_text
    ]
    v2_code = (
        _active_code(cand_trees["mega_moe_v2.py"])
        if "mega_moe_v2.py" in cand_trees else ""
    )
    hardwired_full_variant = (
        "AITER_MEGAMOE_FUSE_ALL" in v2_code
        and "AITER_MEGAMOE_FUSE_COMBINE" not in v2_code
        and "AITER_MEGAMOE_FUSE_QUANT" not in v2_code
        and "AITER_MEGAMOE_P2P_WT" not in v2_code
        and "_run_fused_stage1" in v2_code
        and "shmem_comb_out_tok" in v2_code
    )
    variant_pass = (bool(variant_nodes) and not variant_missing) or hardwired_full_variant
    feature_results["m25_variant_is_sanitized"] = {
        "pass": variant_pass,
        "patterns_passed": (
            len(variant_markers)
            if hardwired_full_variant
            else len(variant_markers) - len(variant_missing)
        ),
        "patterns_total": len(variant_markers),
        "pattern_ratio": (
            1.0 if hardwired_full_variant else
            (len(variant_markers) - len(variant_missing)) / len(variant_markers)
        ),
        "missing_patterns": [] if variant_pass else (
            variant_missing or ["missing sanitized or hardwired full M2.5 variant"]
        ),
    }

    candidate_text = "\n".join(_active_code(tree) for tree in cand_trees.values())
    diagnostics = [
        marker for marker in FORBIDDEN_DIAGNOSTICS if marker in candidate_text
    ]

    required_file_missing = sorted(set(changed_by_reference) - set(changed_by_candidate))
    exact_missing = sorted(set(all_names) - set(exact_reference_files))
    core_identity_matches = sorted(set(CORE_IDENTITY_FILES) & set(exact_reference_files))
    core_ast_identity_matches = sorted(
        name for name in CORE_IDENTITY_FILES
        if name in ref_trees
        and name in cand_trees
        and ast.dump(ref_trees[name], include_attributes=False)
        == ast.dump(cand_trees[name], include_attributes=False)
    )
    token_similarity = {}
    baseline_similarity = {}
    for name in CORE_IDENTITY_FILES:
        if name not in ref_files or name not in cand_files:
            continue
        ref_tokens = _normalized_tokens(ref_files[name])
        cand_tokens = _normalized_tokens(cand_files[name])
        base_tokens = _normalized_tokens(base_files[name]) if name in base_files else ""
        token_similarity[name] = _shingle_similarity(ref_tokens, cand_tokens)
        baseline_similarity[name] = _shingle_similarity(ref_tokens, base_tokens)
    mean_similarity = (
        sum(token_similarity.values()) / len(token_similarity)
        if token_similarity else 0.0
    )
    mean_baseline_similarity = (
        sum(baseline_similarity.values()) / len(baseline_similarity)
        if baseline_similarity else 0.0
    )
    ref_function_dumps = set().union(
        *(_function_dumps(tree) for tree in ref_trees.values())
    ) if ref_trees else set()
    base_function_dumps = set().union(
        *(_function_dumps(tree) for tree in base_trees.values())
    ) if base_trees else set()
    cand_function_dumps = set().union(
        *(_function_dumps(tree) for tree in cand_trees.values())
    ) if cand_trees else set()
    oracle_exclusive_functions = ref_function_dumps - base_function_dumps
    cloned_oracle_functions = oracle_exclusive_functions & cand_function_dumps
    clone_ratio = (
        len(cloned_oracle_functions) / len(oracle_exclusive_functions)
        if oracle_exclusive_functions else 0.0
    )
    reference_copy_detected = (
        (
            set(CORE_IDENTITY_FILES).issubset(exact_reference_files)
            or set(CORE_IDENTITY_FILES).issubset(core_ast_identity_matches)
            or (
                len(token_similarity) == len(CORE_IDENTITY_FILES)
                and mean_similarity >= 0.985
                and mean_similarity >= mean_baseline_similarity + 0.02
            )
            or (
                len(oracle_exclusive_functions) >= 3
                and clone_ratio >= 0.80
            )
        )
        and all(name in cand_files for name in CORE_IDENTITY_FILES)
    )
    reference_copy_suspected = (
        not reference_copy_detected
        and (
            (
                len(token_similarity) == len(CORE_IDENTITY_FILES)
                and mean_similarity >= 0.90
                and mean_similarity >= mean_baseline_similarity + 0.05
            )
            or (
                len(oracle_exclusive_functions) >= 4
                and clone_ratio >= 0.50
            )
        )
    )
    manifest = authoring_manifest if isinstance(authoring_manifest, dict) else {}
    manifest_digest = str(manifest.get("candidate_tree_digest") or "")
    current_tree_digest = _tree_digest(cand_files) if cand_files else ""
    sealed = manifest.get("sealed_before_oracle_comparison") is True
    oracle_hidden = manifest.get("oracle_exposed_during_authoring") is False
    digest_matches = bool(manifest_digest) and manifest_digest == current_tree_digest
    provenance_attestation_valid = (
        sealed and oracle_hidden and digest_matches
        and not candidate_symlinks and not oracle_hardlinks
    )
    structural_compatible = (
        not input_errors
        and not cand_errors
        and not ref_errors
        and all(
            item["pass"]
            for name, item in feature_results.items()
            if ENFORCE_CORRECTED_FACTS or name not in CORRECTED_FACTS
        )
        and not diagnostics
    )
    structural_exact = (
        not input_errors and not exact_missing
        and not base_errors and not ref_errors and not cand_errors
    )
    verdict = (
        "calibration_only"
        if structural_exact and reference_copy_detected
        else "compatible"
        if structural_compatible
        else "incomplete"
    )
    independent_structure_pass = (
        structural_compatible
        and not reference_copy_detected
        and not reference_copy_suspected
        and not candidate_symlinks
        and not oracle_hardlinks
    )
    capability_eligible = independent_structure_pass and provenance_attestation_valid
    plan_consistent, plan_errors = _validate_plan(plan_ir)
    if plan_consistent is False:
        capability_eligible = False
    if reference_copy_detected or reference_copy_suspected:
        provenance_status = "copy_detected" if reference_copy_detected else "copy_suspected"
    elif provenance_attestation_valid:
        provenance_status = "attested_unverified"
    else:
        provenance_status = "unverified"
    feature_passed = sum(item["patterns_passed"] for item in feature_results.values())
    feature_total = sum(item["patterns_total"] for item in feature_results.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "checker_sha256": _digest(Path(__file__)),
        "baseline_tree_digest": _tree_digest(base_files) if base_files else "",
        "reference_tree_digest": _tree_digest(ref_files) if ref_files else "",
        "pinned_oracle_revision": PINNED_ORACLE_REVISION,
        "verdict": verdict,
        "structural_exact": structural_exact,
        "structural_compatible": structural_compatible,
        "independent_structure_pass": independent_structure_pass,
        "reference_copy_detected": reference_copy_detected,
        "reference_copy_suspected": reference_copy_suspected,
        "core_reference_identity_files": core_identity_matches,
        "core_reference_ast_identity_files": core_ast_identity_matches,
        "core_token_similarity": token_similarity,
        "core_token_similarity_mean": mean_similarity,
        "baseline_to_reference_similarity_mean": mean_baseline_similarity,
        "oracle_exclusive_function_count": len(oracle_exclusive_functions),
        "cloned_oracle_function_count": len(cloned_oracle_functions),
        "cloned_oracle_function_ratio": clone_ratio,
        "capability_eligible": capability_eligible,
        "provenance_status": provenance_status,
        "provenance_attestation_valid": provenance_attestation_valid,
        "plan_consistent": plan_consistent,
        "plan_consistency_errors": plan_errors,
        "candidate_tree_digest": current_tree_digest,
        "manifest_tree_digest": manifest_digest,
        "sealed_before_oracle_comparison": sealed,
        "oracle_exposed_during_authoring": manifest.get(
            "oracle_exposed_during_authoring", None
        ),
        "input_errors": input_errors,
        "candidate_symlinks": candidate_symlinks,
        "oracle_hardlinks": oracle_hardlinks,
        "semantic_features_passed": feature_passed,
        "semantic_features_total": feature_total,
        "semantic_feature_ratio": (
            feature_passed / feature_total if feature_total else 0.0
        ),
        "hardware_verified": False,
        "gpu_executed": False,
        "device_jit_verified": False,
        "path_activation_verified": False,
        "launch_count_verified": False,
        "accuracy_verified": False,
        "numeric_accuracy_verified": False,
        "graph_liveness_verified": False,
        "residency_verified": False,
        "distributed_memory_order_verified": False,
        "performance_verified": False,
        "reference_changed_files": changed_by_reference,
        "candidate_changed_files": changed_by_candidate,
        "required_changed_files_missing": required_file_missing,
        "exact_reference_files": exact_reference_files,
        "exact_reference_files_missing": exact_missing,
        "required_new_symbols": required_symbols,
        "missing_new_symbols": missing_symbols,
        "features": feature_results,
        "forbidden_diagnostics_present": diagnostics,
        "parse_errors": {
            "baseline": base_errors,
            "reference": ref_errors,
            "candidate": cand_errors,
        },
        "note": (
            "Offline source-structure result only. GPU accuracy, liveness, launch shape, "
            "and performance remain unverified. Static comparison cannot prove oracle isolation; "
            "a valid pre-comparison seal is reported only as attested_unverified. Exact or "
            "near-exact oracle identity calibrates this checker but is not capability evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--authoring-manifest", type=Path)
    parser.add_argument("--plan-json", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--require",
        choices=("report", "compatible", "independent", "exact"),
        default="report",
    )
    args = parser.parse_args()
    manifest = None
    if args.authoring_manifest:
        manifest = json.loads(args.authoring_manifest.read_text())
    plan_ir = json.loads(args.plan_json.read_text()) if args.plan_json else None
    result = evaluate(
        args.baseline.resolve(),
        args.reference.resolve(),
        args.candidate.resolve(),
        manifest,
        plan_ir,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    if args.require == "exact":
        return 0 if result["structural_exact"] else 1
    if args.require == "compatible":
        return 0 if result["structural_compatible"] else 1
    if args.require == "independent":
        return 0 if result["capability_eligible"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
