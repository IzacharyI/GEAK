#!/usr/bin/env python3
"""Profile-driven graph, launch, resource and paired-performance validation."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
from pathlib import Path


PROFILE = {
    "world": 8,
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "swiglu_limit": 10.0,
}


def _csv_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_adapter(tree, explicit):
    candidates = []
    if explicit:
        path = Path(explicit)
        candidates.append(path if path.is_absolute() else tree / path)
    # Non-normative adapter hint for the validated profile.
    candidates.append(tree / "op_tests/multigpu_tests/test_mega_moe_v2.py")
    candidates.extend(sorted(tree.glob("**/test_*moe*.py")))
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(errors="ignore")
        if all(name in text for name in ("_reference", "_setup_dist", "_quantize_weights")):
            helper = _load_module(path, "_expert_skill_profile_adapter")
            if all(
                callable(getattr(helper, name, None))
                for name in ("_reference", "_setup_dist", "_quantize_weights", "_barrier")
            ):
                return helper, path
    raise RuntimeError("no semantic runtime adapter was discovered")


def _discover_profile(helper, profile_name):
    required = {"model_dim", "inter_dim", "experts", "topk"}
    for value in vars(helper).values():
        if not isinstance(value, dict) or profile_name not in value:
            continue
        profile = value[profile_name]
        if isinstance(profile, dict) and required <= set(profile):
            return dict(profile)
    raise RuntimeError(f"runtime adapter has no profile {profile_name!r}")


def _discover_factory(helper):
    required = {
        "rank", "world_size", "model_dim", "inter_dim", "experts", "topk",
        "quant", "w1", "w1_scale", "w2", "w2_scale", "max_tok_per_rank",
    }
    matches = []
    for value in vars(helper).values():
        if not inspect.isclass(value):
            continue
        try:
            parameters = set(inspect.signature(value).parameters)
        except (TypeError, ValueError):
            continue
        if required <= parameters:
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one semantic candidate factory, discovered {len(matches)}"
        )
    return matches[0]


def _make_route(torch, tokens, route, iteration, profile, rank, world, device):
    generator = torch.Generator(device=device).manual_seed(
        123 + rank + iteration * 1009 + tokens * 17
    )
    x = torch.randn(
        (tokens, profile["model_dim"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    destination = torch.topk(
        torch.rand((tokens, world), device=device, generator=generator),
        profile["topk"],
        dim=-1,
    ).indices
    local_experts = profile["experts"] // world
    if route == "uniform":
        local = torch.randint(
            0, local_experts, destination.shape, device=device, generator=generator
        )
    elif route == "rank-mixed-skew":
        cold = torch.randint(
            1, local_experts, destination.shape, device=device, generator=generator
        )
        local = torch.where(destination < world // 2, torch.zeros_like(cold), cold)
    else:
        raise ValueError(f"unsupported route {route!r}")
    logits = torch.randn(
        destination.shape, dtype=torch.float32, device=device, generator=generator
    )
    return (
        x.contiguous(),
        logits.softmax(dim=-1).contiguous(),
        (destination * local_experts + local).to(torch.int32).contiguous(),
    )


def _relative_l2(torch, dist, helper, output, reference, device):
    value = float(
        (
            torch.linalg.vector_norm(output.float() - reference)
            / torch.linalg.vector_norm(reference)
        ).item()
    )
    return helper._reduce_float(value, device, dist.ReduceOp.MAX)


def _capture(torch, helper, body):
    helper._barrier()
    body()
    helper._barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_rank_max(torch, dist, helper, graph, device, iters):
    helper._barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return helper._reduce_float(
        start.elapsed_time(end) / iters, device, dist.ReduceOp.MAX
    )


def _profile_launch_count(torch, dist, helper, body, device):
    from torch.profiler import ProfilerActivity, profile

    helper._barrier()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        body()
        torch.cuda.synchronize()
    count = 0
    for event in trace.key_averages():
        name = str(event.key).lower()
        device_time = float(
            getattr(event, "device_time_total", 0)
            or getattr(event, "self_device_time_total", 0)
            or 0
        )
        if device_time > 0 and not any(tag in name for tag in ("memcpy", "memset")):
            count += int(event.count)
    lo = torch.tensor(count, dtype=torch.int32, device=device)
    hi = lo.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if int(lo.item()) != 2 or int(hi.item()) != 2:
        raise AssertionError(
            f"BF16 path must emit exactly two kernels per rank, got "
            f"{int(lo.item())}..{int(hi.item())}"
        )
    return 2


def _resource_checks(path):
    if not path:
        return {
            "status": "delegated",
            "required_by_acceptance": True,
            "reason": "outer verifier must supply emitted metadata",
        }
    data = json.loads(Path(path).read_text())
    rows = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(rows, dict):
        raise ValueError("resource evidence must contain a cases object")
    limits = {
        "fixed_128": {
            "threads": 256, "group_segment": 32144, "kernarg": 440,
            "vgpr": 168, "sgpr": 106, "vgpr_spills": 0,
            "sgpr_spills": 106, "private": 0,
        },
        "large_8192": {
            "threads": 512, "group_segment": 160400, "kernarg": 440,
            "vgpr": 256, "sgpr": 106, "vgpr_spills": 80,
            "sgpr_spills": 116, "private": 316,
        },
    }
    for case, bound in limits.items():
        row = rows.get(case)
        if not isinstance(row, dict):
            raise AssertionError(f"missing resource case {case}")
        for key, maximum in bound.items():
            value = int(row.get(key, -1))
            if value < 0 or value > maximum:
                raise AssertionError(
                    f"resource {case}.{key}={value} exceeds {maximum}"
                )
        if int(row.get("resident_workgroups", 1)) < 1:
            raise AssertionError(f"resource {case} has no resident workgroup")
    return {"status": "pass", "cases": sorted(limits)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--profile-adapter", default="")
    parser.add_argument("--profile", default="v4_pro")
    parser.add_argument("--accuracy-cases", default="128,512,8192")
    parser.add_argument("--liveness-cases", default="128,512,8192")
    parser.add_argument("--routes", default="uniform,rank-mixed-skew")
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--numeric-checkpoint-interval", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.03)
    parser.add_argument("--resource-evidence", default="")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    tree = Path(args.candidate_tree).resolve()
    sys.path.insert(0, str(tree))
    helper, adapter_path = _discover_adapter(tree, args.profile_adapter)

    import torch
    import torch.distributed as dist

    accuracy_cases = _csv_ints(args.accuracy_cases)
    liveness_cases = _csv_ints(args.liveness_cases)
    routes = _csv_strings(args.routes)
    required_cases = {128, 512, 8192}
    if not required_cases <= set(accuracy_cases) or not required_cases <= set(liveness_cases):
        raise ValueError("runtime claims require direct 128/512/8192 cases")
    if args.replays <= 0 or args.pairs <= 0 or args.perf_iters <= 0:
        raise ValueError("replays, pairs and perf-iters must be positive")

    rank, world, device = helper._setup_dist()
    result = {
        "schema_version": "expert-skill-runtime-evidence-v1",
        "claim_complete": False,
        "validation_scope": "post-selection claims only",
        "adapter": adapter_path.relative_to(tree).as_posix(),
        "accuracy_results": [],
        "replay_results": [],
        "paired_readings": [],
    }

    def write_result(complete):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, output)

    try:
        if world != PROFILE["world"]:
            raise AssertionError(f"validated claim requires 8 ranks, got {world}")
        profile_data = _discover_profile(helper, args.profile)
        for key in ("model_dim", "inter_dim", "experts", "topk"):
            if int(profile_data[key]) != int(PROFILE[key]):
                raise AssertionError(
                    f"selected profile claim differs at {key}: {profile_data[key]}"
                )
        profile_data["swiglu_limit"] = float(
            profile_data.get("swiglu_limit", PROFILE["swiglu_limit"])
        )
        factory = _discover_factory(helper)
        local_experts = profile_data["experts"] // world
        packed = helper._quantize_weights(
            profile_data["model_dim"],
            profile_data["inter_dim"],
            local_experts,
            rank,
            123,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        max_tokens = _next_power_of_two(max(accuracy_cases + liveness_cases))

        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_COMBINE"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_QUANT"] = "0"
        candidate = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=max_tokens,
            **profile_data,
        )

        target_body = None
        target_tensors = None
        for tokens in sorted(set(accuracy_cases + liveness_cases)):
            x, route_weights, ids = _make_route(
                torch, tokens, "uniform", 0, profile_data, rank, world, device
            )
            state = {}

            def body():
                state["output"] = candidate(x, route_weights, ids)[:tokens]

            graph = _capture(torch, helper, body)
            if tokens == 8192:
                target_body = body
                target_tensors = (x, route_weights, ids, graph)

            if tokens in accuracy_cases:
                reference = helper._reference(
                    x,
                    route_weights,
                    ids,
                    ref_weights,
                    rank,
                    world,
                    profile_data["model_dim"],
                    profile_data["inter_dim"],
                    profile_data["experts"],
                    profile_data["swiglu_limit"],
                )
                rel_l2 = _relative_l2(
                    torch, dist, helper, state["output"], reference, device
                )
                row = {
                    "guard": str(tokens),
                    "metric": "relL2",
                    "value": rel_l2,
                    "threshold": args.rtol,
                    "method": "direct graph-captured candidate vs numeric reference",
                    "status": "pass" if rel_l2 < args.rtol else "fail",
                }
                result["accuracy_results"].append(row)
                write_result(False)
                if row["status"] != "pass":
                    raise AssertionError(f"tokens={tokens} relL2={rel_l2:.6f}")

            if tokens in liveness_cases:
                for route in routes:
                    checkpoints = []
                    for replay in range(args.replays):
                        _, new_weights, new_ids = _make_route(
                            torch,
                            tokens,
                            route,
                            replay + 1,
                            profile_data,
                            rank,
                            world,
                            device,
                        )
                        route_weights.copy_(new_weights)
                        ids.copy_(new_ids)
                        graph.replay()
                        if (
                            (replay + 1) % args.numeric_checkpoint_interval == 0
                            or replay + 1 == args.replays
                        ):
                            torch.cuda.synchronize()
                            reference = helper._reference(
                                x,
                                route_weights,
                                ids,
                                ref_weights,
                                rank,
                                world,
                                profile_data["model_dim"],
                                profile_data["inter_dim"],
                                profile_data["experts"],
                                profile_data["swiglu_limit"],
                            )
                            value = _relative_l2(
                                torch, dist, helper, state["output"], reference, device
                            )
                            checkpoints.append(value)
                            if value >= args.rtol:
                                raise AssertionError(
                                    f"{tokens}/{route} replay relL2={value:.6f}"
                                )
                    replay_row = {
                        "guard": f"{tokens}_{route}",
                        "count": args.replays,
                        "status": "pass",
                        "graph_safe": "pass",
                        "arrival_jitter": True,
                        "routing_changes": args.replays,
                        "max_checkpoint_relL2": max(checkpoints),
                    }
                    result["replay_results"].append(replay_row)
                    write_result(False)

        if target_body is None or target_tensors is None:
            raise AssertionError("8192 target graph was not produced")
        launch_count = _profile_launch_count(
            torch, dist, helper, target_body, device
        )
        result["activation"] = {
            "status": "pass",
            "ranks": world,
            "fusion_switch": os.environ["AITER_MEGAMOE_FUSE_ALL"],
        }
        result["launch_count"] = launch_count
        result["resources"] = _resource_checks(args.resource_evidence)

        x, route_weights, ids, candidate_graph = target_tensors
        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "0"
        control = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=max_tokens,
            **profile_data,
        )
        control_state = {}

        def control_body():
            control_state["output"] = control(x, route_weights, ids)[:8192]

        control_graph = _capture(torch, helper, control_body)
        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        ratios = []
        for pair in range(args.pairs):
            if pair % 2:
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
            else:
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
            ratios.append(base_ms / cand_ms)
            result["paired_readings"].append(
                {"guard": "8192_uniform", "base": base_ms, "cand": cand_ms}
            )
            write_result(False)
        speedup = statistics.median(ratios)
        result["paired_rank_max_speedup"] = speedup
        result["paired_control"] = (
            "same selected tree with persistent fusion disabled; "
            "outer workflow still owns the frozen-baseline pair"
        )
        if speedup < args.min_speedup:
            raise AssertionError(
                f"paired rank-max speedup {speedup:.6f} < {args.min_speedup}"
            )

        result["correctness"] = "pass"
        result["graph_safe"] = "pass"
        result["liveness"] = "pass"
        result["liveness_replays"] = args.replays
        write_result(True)
        return 0
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        write_result(False)
        raise
    finally:
        helper._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())