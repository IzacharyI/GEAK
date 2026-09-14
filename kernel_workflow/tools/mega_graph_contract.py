#!/usr/bin/env python3
"""Direct graph correctness and arrival-jitter liveness for MegaMoE V2 EP8.

The tool is repository-portable: the candidate tree is an argument and runtime
libraries/JIT cache come from the caller's environment. It imports the frozen
task's own reference helpers from the candidate copy, captures the real
MegaMoEV2 call, compares graph output directly with the numeric reference, and
varies routing buffers between replays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--network", default="v4_pro")
    parser.add_argument("--accuracy-cases", default="128,512,8192")
    parser.add_argument("--liveness-cases", default="128,512,8192")
    parser.add_argument("--routes", default="uniform,rank-mixed-skew")
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    tree = Path(args.candidate_tree).resolve()
    test_path = tree / "op_tests/multigpu_tests/test_mega_moe_v2.py"
    if not test_path.is_file():
        raise SystemExit(f"candidate test helper missing: {test_path}")
    sys.path.insert(0, str(tree))

    import torch
    import torch.distributed as dist
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    helper = _load_module(test_path, "_geak_mega_reference")
    accuracy_cases = _csv_ints(args.accuracy_cases)
    liveness_cases = _csv_ints(args.liveness_cases)
    routes = _csv_strings(args.routes)
    if not accuracy_cases or not liveness_cases or not routes:
        raise ValueError("accuracy/liveness cases and routes must be non-empty")
    if args.replays <= 0:
        raise ValueError("--replays must be positive")

    rank, world, device = helper._setup_dist()
    if world != 8:
        raise ValueError(f"MegaMoE contract requires world=8, got {world}")

    rows: list[dict] = []
    replay_rows: list[dict] = []
    result = {
        "schema_version": "geak-megamoe-graph-contract-v1",
        "claim_complete": False,
        "world_size": world,
        "activation": os.environ.get("AITER_MEGAMOE_FUSE_ALL", ""),
        "accuracy_results": rows,
        "replay_results": replay_rows,
    }

    def write_result(complete: bool):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(tmp, output)

    def make_route(
        tokens: int, route: str, iteration: int, network: dict, include_x: bool = True
    ):
        generator = torch.Generator(device=device).manual_seed(
            args.seed + rank + iteration * 1009 + tokens * 17
        )
        x = (
            torch.randn(
                (tokens, network["model_dim"]),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            if include_x
            else None
        )
        destination_scores = torch.rand(
            (tokens, world), device=device, generator=generator
        )
        destination = torch.topk(
            destination_scores, network["topk"], dim=-1
        ).indices
        local_experts = network["experts"] // world
        if route == "uniform":
            local = torch.randint(
                0,
                local_experts,
                (tokens, network["topk"]),
                device=device,
                generator=generator,
            )
        elif route == "rank-mixed-skew":
            hot = destination < world // 2
            cold = torch.randint(
                1,
                local_experts,
                (tokens, network["topk"]),
                device=device,
                generator=generator,
            )
            local = torch.where(hot, torch.zeros_like(cold), cold)
        else:
            raise ValueError(f"unsupported route {route!r}")
        ids = destination * local_experts + local
        logits = torch.randn(
            (tokens, network["topk"]),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return (
            x.contiguous(),
            logits.softmax(dim=-1).contiguous(),
            ids.to(torch.int32).contiguous(),
        )

    def relative_l2(output, reference) -> float:
        numerator = torch.linalg.vector_norm(output.float() - reference)
        denominator = torch.linalg.vector_norm(reference)
        value = float((numerator / denominator).item())
        return helper._reduce_float(value, device, dist.ReduceOp.MAX)

    try:
        network = helper.NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError("expert count must divide world size")
        local_experts = network["experts"] // world
        packed = helper._quantize_weights(
            network["model_dim"],
            network["inter_dim"],
            local_experts,
            rank,
            args.seed,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale

        all_cases = sorted(set(accuracy_cases + liveness_cases))
        for tokens in all_cases:
            max_tok_per_rank = max(16, _next_power_of_two(tokens))
            moe = MegaMoEV2(
                rank=rank,
                world_size=world,
                quant="a8w4",
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=max_tok_per_rank,
                **network,
            )
            x, route_weights, ids = make_route(tokens, "uniform", 0, network)
            state = {}

            def body():
                state["output"] = moe(x, route_weights, ids)[:tokens]

            helper._barrier()
            body()
            helper._barrier()
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=capture_stream):
                body()
            graph.replay()
            torch.cuda.synchronize()

            if tokens in accuracy_cases:
                reference = helper._reference(
                    x,
                    route_weights,
                    ids,
                    ref_weights,
                    rank,
                    world,
                    network["model_dim"],
                    network["inter_dim"],
                    network["experts"],
                    network["swiglu_limit"],
                )
                rel_l2 = relative_l2(state["output"], reference)
                row = {
                    "guard": str(tokens),
                    "route": "uniform",
                    "metric": "relL2",
                    "value": rel_l2,
                    "threshold": args.rtol,
                    "method": "direct graph-captured candidate vs numeric reference",
                    "status": "pass" if rel_l2 < args.rtol else "fail",
                }
                rows.append(row)
                write_result(False)
                if rel_l2 >= args.rtol:
                    raise AssertionError(
                        f"tokens={tokens} direct graph relL2={rel_l2:.6f}"
                    )

            if tokens in liveness_cases:
                for route in routes:
                    for replay in range(args.replays):
                        _unused_x, new_weights, new_ids = make_route(
                            tokens, route, replay + 1, network, include_x=False
                        )
                        route_weights.copy_(new_weights)
                        ids.copy_(new_ids)
                        graph.replay()
                        if (replay + 1) % 16 == 0:
                            torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(state["output"]).all().item())
                    finite_all = torch.tensor(
                        int(finite), dtype=torch.int32, device=device
                    )
                    dist.all_reduce(finite_all, op=dist.ReduceOp.MIN)
                    row = {
                        "guard": f"{tokens}_{route}",
                        "count": args.replays,
                        "status": "pass" if int(finite_all.item()) == 1 else "fail",
                        "graph_safe": "pass",
                        "arrival_jitter": True,
                        "routing_changes": args.replays,
                    }
                    replay_rows.append(row)
                    write_result(False)
                    if row["status"] != "pass":
                        raise AssertionError(
                            f"non-finite graph output after {tokens}/{route}"
                        )

            del moe, graph, state
            torch.cuda.empty_cache()
            helper._barrier()

        result["correctness"] = "pass"
        result["graph_safe"] = "pass"
        result["liveness"] = "pass"
        write_result(True)
        if rank == 0:
            print(
                f"[MEGA-GRAPH-CONTRACT] PASS accuracy={len(rows)} "
                f"replay_cases={len(replay_rows)} replays={args.replays}",
                flush=True,
            )
        return 0
    finally:
        helper._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
