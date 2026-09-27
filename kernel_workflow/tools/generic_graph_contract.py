#!/usr/bin/env python3
"""Operator-neutral graph-contract runner for generic mega runs (Expert Skills off).

Run under ``torchrun`` once per verification. The runner knows nothing about the operator: a
task-owned *contract adapter* (written once at setup from the task's own test/reference code and
kept outside every candidate tree) builds the operator, makes inputs and computes the numeric
reference. The runner owns the measurement:

  * graph-captured accuracy per case, relL2 reduced with rank-MAX;
  * replay liveness: N replays per (case, route) with fresh inputs copied into the captured
    tensors, numeric checkpoints every K replays;
  * per-capacity cases (``--mtpr-cases``): a fresh operator sized for each case, fallback allowed
    only at or below ``--mtpr-fallback-max``;
  * path-marker activation on every rank (fd 1/2 capture of the first eager forward);
  * launches per forward (torch profiler, MIN/MAX over ranks);
  * rank-max graph timing vs the frozen baseline, with the same-tree switch-off arm as a
    diagnostic only;
  * a hang watchdog that writes the stuck phase and exits, instead of waiting for the lease reaper.

Adapter protocol (a plain .py file, loaded by path):

  setup() -> (rank, world, device)          init the process group; called once
  build(capacity) -> op                     fresh operator for ``capacity`` tokens per rank; reads
                                            the activation environment at construction
  make_inputs(tokens, route, iteration)     tuple of tensors, deterministic per (rank, tokens,
                                            route, iteration); same shapes for every iteration
  forward(op, inputs) -> Tensor             one operator call on this rank, no host sync
  reference(inputs) -> Tensor               the task's numeric reference for that output
  cleanup()                                 optional

The command line accepts the argument names of the Expert-Skill runtime (``--runtime-file`` is the
adapter), so a verify role written for that runtime drives this runner unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

SCHEMA = "generic-graph-contract-v1"
FROZEN_BASELINE_MISSING = "not_evaluated: missing --frozen-baseline-ms"
ADAPTER_FUNCTIONS = ("setup", "build", "make_inputs", "forward", "reference")
GUIDE_HINT = "generic mega runs: see MEGA_MEASUREMENT_GUIDE (knowledge/mega_measurement.md) section 4"


def csv_ints(value):
    return [int(item) for item in str(value or "").split(",") if item.strip()]


def csv_strings(value):
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def parse_env(pairs):
    env = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"environment assignment must be KEY=VALUE, got {pair!r}")
        env[key.strip()] = value
    return env


@contextlib.contextmanager
def scoped_env(env):
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_adapter(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"contract adapter {path} not found ({GUIDE_HINT}; without an adapter use the "
            "COMMANDMENT graph-capable correctness command instead)")
    spec = importlib.util.spec_from_file_location("_generic_graph_contract_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import contract adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [name for name in ADAPTER_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(f"contract adapter {path} lacks {', '.join(missing)}")
    return module


def positive_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def activation_summary(marker, world, per_rank_counts):
    """Activation passes only when every rank printed ``marker`` in the window."""
    per_rank = [
        {"rank": rank, "marker_count": int(count), "observed": int(count) > 0}
        for rank, count in enumerate(per_rank_counts)
    ]
    passed = world > 0 and len(per_rank) == world and all(row["observed"] for row in per_rank)
    return {
        "status": "pass" if passed else "fail",
        "path_marker": marker,
        "ranks": world,
        "observed_ranks": sum(row["observed"] for row in per_rank),
        "per_rank": per_rank,
    }


def speedup_summary(frozen_ms, cand_list, control_list, min_speedup, tolerance):
    """Gate on the frozen baseline; the same-tree switch-off pair is diagnostic only."""
    if (not cand_list or len(cand_list) != len(control_list)
            or any(positive_finite(v) is None for v in [*cand_list, *control_list])):
        raise ValueError("paired readings must be matched positive latencies")
    cand = statistics.median(cand_list)
    control = statistics.median(control_list)
    summary = {
        "frozen_baseline_ms": None,
        "candidate_ms_median": cand,
        "switch_off_ms_median": control,
        "incremental_switch_speedup": statistics.median(
            [base / c for base, c in zip(control_list, cand_list)]),
        "absolute_speedup": None,
        "min_speedup": min_speedup,
        "denominator_tolerance": tolerance,
        "denominator_deviation_pct": None,
        "speedup_gate": FROZEN_BASELINE_MISSING,
    }
    frozen = positive_finite(frozen_ms)
    if frozen is None:
        return summary
    deviation = (control - frozen) / frozen
    summary.update(
        frozen_baseline_ms=frozen,
        absolute_speedup=frozen / cand,
        denominator_deviation_pct=100.0 * deviation,
        speedup_gate="pass" if frozen / cand >= min_speedup else "fail",
    )
    if abs(deviation) > tolerance:
        summary["denominator_mismatch"] = {
            "note": "the candidate tree's switch-off path no longer times like the frozen "
                    "baseline; incremental_switch_speedup is not a frozen-baseline speedup",
            "deviation_pct": 100.0 * deviation,
        }
    return summary


def claim_blockers(result, launch_target):
    """Anything missing or failed keeps ``claim_complete`` false."""
    blockers = []
    if "activation" not in result:
        blockers.append("activation_not_run")
    elif result["activation"].get("status") != "pass":
        blockers.append("activation_marker_missing")
    for row in result.get("mtpr_activation") or []:
        if row.get("status") != "pass" and not row.get("fallback_allowed"):
            blockers.append(f"activation_marker_missing:{row.get('case')}")
    launches = result.get("launch_count") or {}
    if launch_target and not launches:
        blockers.append("launch_count_not_run")
    elif launch_target and not (launches.get("max") and launches["max"] <= launch_target
                              and launches.get("min") == launches.get("max")):
        blockers.append("launch_target_not_met")
    if any(row.get("status") != "pass" for row in result.get("accuracy_results") or []):
        blockers.append("accuracy_failed")
    if any(row.get("status") != "pass" for row in result.get("replay_results") or []):
        blockers.append("liveness_failed")
    gate = str(result.get("speedup_gate") or "")
    if not gate:
        blockers.append("timing_not_run")
    elif gate.startswith("not_evaluated"):
        blockers.append("frozen_baseline_missing")
    elif gate != "pass":
        blockers.append("speedup_gate_failed")
    return blockers


class Watchdog:
    """Fail fast with the stuck phase named, instead of an info-less reap at the lease timeout."""

    def __init__(self, seconds, on_expire):
        self.seconds = float(seconds)
        self.on_expire = on_expire
        self.phase = "start"
        self.stamp = time.monotonic()
        self._stop = threading.Event()
        self._thread = None

    def mark(self, phase):
        self.phase = phase
        self.stamp = time.monotonic()

    def start(self):
        if self.seconds <= 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(min(5.0, self.seconds / 4)):
            if time.monotonic() - self.stamp > self.seconds:
                self.on_expire(self.phase, time.monotonic() - self.stamp)
                return


def _flush():
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.flush()
    with contextlib.suppress(AttributeError, OSError, TypeError):
        import ctypes

        ctypes.CDLL(None).fflush(None)


@contextlib.contextmanager
def captured_fds(directory, prefix):
    """Redirect fds 1/2 into files for one window, then restore and re-emit the text."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    captured = {"text": ""}
    redirected = []
    _flush()
    try:
        for fd in (1, 2):
            cap_fd, cap_path = tempfile.mkstemp(prefix=f"{prefix}.fd{fd}.", suffix=".log",
                                                dir=directory)
            redirected.append((fd, os.dup(fd), cap_fd, cap_path))
            os.dup2(cap_fd, fd)
        yield captured
    finally:
        _flush()
        texts = []
        for fd, saved, cap_fd, cap_path in redirected:
            os.dup2(saved, fd)
            os.close(saved)
            os.lseek(cap_fd, 0, os.SEEK_SET)
            chunks = []
            while True:
                chunk = os.read(cap_fd, 1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
            os.close(cap_fd)
            with contextlib.suppress(OSError):
                os.unlink(cap_path)
            data = b"".join(chunks)
            texts.append(data.decode(errors="replace"))
            with contextlib.suppress(OSError):
                os.write(fd, data)
        captured["text"] = "".join(texts)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--candidate-tree", required=True)
    p.add_argument("--adapter", "--runtime-file", dest="adapter", required=True,
                   help="task contract adapter (.py); --runtime-file is an accepted alias")
    p.add_argument("--accuracy-cases", required=True)
    p.add_argument("--liveness-cases", default="")
    p.add_argument("--routes", default="uniform")
    p.add_argument("--replays", type=int, default=30)
    p.add_argument("--numeric-checkpoint-interval", type=int, default=16)
    p.add_argument("--mtpr-cases", default="")
    p.add_argument("--mtpr-fallback-max", type=int, default=0)
    p.add_argument("--capacity", type=int, default=0,
                   help="shared operator capacity; default next pow2 of the largest case")
    p.add_argument("--rtol", type=float, default=0.10)
    p.add_argument("--target-tokens", type=int, default=0)
    p.add_argument("--perf-iters", type=int, default=20)
    p.add_argument("--pairs", type=int, default=5)
    p.add_argument("--min-speedup", type=float, default=1.0)
    p.add_argument("--frozen-baseline-ms", type=float, default=None)
    p.add_argument("--denominator-tolerance", type=float, default=0.05)
    p.add_argument("--path-marker", default="")
    p.add_argument("--activation-env", action="append", default=[],
                   help="KEY=VALUE for the candidate arm (repeat)")
    p.add_argument("--control-env", action="append", default=[],
                   help="KEY=VALUE for the same-tree switch-off arm (repeat)")
    p.add_argument("--launch-target", type=int, default=0)
    p.add_argument("--import-modules", default="",
                   help="modules whose __file__ must resolve inside --candidate-tree")
    p.add_argument("--hang-timeout-s", type=float, default=600.0)
    p.add_argument("--self-test", action="store_true",
                   help="adapter check on the frozen tree: accuracy + liveness only, no claim")
    p.add_argument("--resource-evidence", default="", help="accepted and recorded, not gated")
    p.add_argument("--json-output", required=True)
    return p


def validate_args(args):
    accuracy = csv_ints(args.accuracy_cases)
    if not accuracy:
        raise ValueError("--accuracy-cases must name at least one case")
    if args.replays <= 0 or args.pairs <= 0 or args.perf_iters <= 0:
        raise ValueError("replays, pairs and perf-iters must be positive")
    if args.numeric_checkpoint_interval <= 0:
        raise ValueError("numeric-checkpoint-interval must be positive")
    if not args.self_test and not args.path_marker:
        raise ValueError(f"--path-marker is required for a candidate claim ({GUIDE_HINT})")
    if not math.isfinite(args.denominator_tolerance) or args.denominator_tolerance < 0:
        raise ValueError("denominator-tolerance must be a finite non-negative fraction")
    if not csv_strings(args.routes):
        raise ValueError("--routes must name at least one route")
    parse_env(args.activation_env)
    parse_env(args.control_env)
    return accuracy


def main(argv=None):
    args = build_parser().parse_args(argv)
    accuracy_cases = validate_args(args)
    liveness_cases = csv_ints(args.liveness_cases) or list(accuracy_cases)
    routes = csv_strings(args.routes)
    mtpr_cases = csv_ints(args.mtpr_cases)
    cases = sorted(set(accuracy_cases + liveness_cases))
    target = args.target_tokens or max(cases)
    if target not in cases:
        cases = sorted(set(cases + [target]))
    capacity = args.capacity or next_power_of_two(max(cases))
    cand_env = parse_env(args.activation_env)
    control_env = parse_env(args.control_env)

    tree = Path(args.candidate_tree).resolve()
    sys.path.insert(0, str(tree))
    adapter = load_adapter(args.adapter)

    import torch
    import torch.distributed as dist

    rank, world, device = adapter.setup()
    output_path = Path(args.json_output).resolve()
    result = {
        "schema_version": SCHEMA,
        "mode": "self_test" if args.self_test else "claim",
        "claim_complete": False,
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_sha256": hashlib.sha256(Path(args.adapter).read_bytes()).hexdigest(),
        "world": world,
        "capacity": capacity,
        "target": {"tokens": target, "route": routes[0]},
        "activation_env": cand_env,
        "control_env": control_env,
        "accuracy_results": [],
        "replay_results": [],
        "paired_readings": [],
        "progress": "start",
    }

    def write_result(complete=False):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = output_path.with_suffix(output_path.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, output_path)

    def expire(phase, idle_s):
        result["error"] = f"hang: no progress for {idle_s:.0f}s in phase {phase!r} on rank {rank}"
        result["hang_phase"] = phase
        result["claim_blockers"] = ["hang"]
        write_result(False)
        _flush()
        os._exit(124)

    dog = Watchdog(args.hang_timeout_s, expire).start()

    def progress(phase):
        result["progress"] = phase
        dog.mark(phase)

    def barrier():
        dist.barrier()

    def reduce(value, op):
        t = torch.tensor([float(value)], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=op)
        return float(t.item())

    def gather_int(value):
        local = torch.tensor([int(value)], dtype=torch.int64, device=device)
        out = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(out, local)
        return [int(item.item()) for item in out]

    def rel_l2(output, reference):
        ref = reference.float()
        value = float((torch.linalg.vector_norm(output.float() - ref)
                       / torch.linalg.vector_norm(ref)).item())
        if not math.isfinite(value):
            value = float("inf")
        return reduce(value, dist.ReduceOp.MAX)

    def capture(op, inputs, state, marker_prefix=None):
        """Eager warm-up (marker window), then capture one forward and replay it once."""
        counts = None
        barrier()
        if marker_prefix is not None and args.path_marker:
            with captured_fds(output_path.parent, marker_prefix) as captured:
                state["output"] = adapter.forward(op, inputs)
                torch.cuda.synchronize()
            counts = gather_int(captured["text"].count(args.path_marker))
        else:
            state["output"] = adapter.forward(op, inputs)
        torch.cuda.synchronize()
        barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            state["output"] = adapter.forward(op, inputs)
        graph.replay()
        torch.cuda.synchronize()
        return graph, counts

    def check_accuracy(state, inputs, guard, method):
        value = rel_l2(state["output"], adapter.reference(inputs))
        row = {"guard": guard, "metric": "relL2", "value": value, "threshold": args.rtol,
               "method": method, "status": "pass" if value < args.rtol else "fail"}
        result["accuracy_results"].append(row)
        write_result()
        if row["status"] != "pass":
            raise AssertionError(f"{guard} relL2={value:.6f} >= {args.rtol}")

    def replay_routes(graph, state, inputs, tokens, suffix, replays):
        for route in routes:
            progress(f"replay {tokens}_{route}{suffix}")
            checkpoints = []
            for replay in range(replays):
                fresh = adapter.make_inputs(tokens, route, replay + 1)
                if len(fresh) != len(inputs):
                    raise AssertionError("make_inputs changed its tensor count between iterations")
                for dst, src in zip(inputs, fresh):
                    if dst.shape != src.shape or dst.dtype != src.dtype:
                        raise AssertionError("make_inputs changed a tensor shape/dtype")
                    dst.copy_(src)
                graph.replay()
                if (replay + 1) % args.numeric_checkpoint_interval == 0 or replay + 1 == replays:
                    torch.cuda.synchronize()
                    dog.mark(f"replay {tokens}_{route}{suffix} #{replay + 1}")
                    value = rel_l2(state["output"], adapter.reference(inputs))
                    checkpoints.append(value)
                    if value >= args.rtol:
                        result["replay_results"].append({
                            "guard": f"{tokens}_{route}{suffix}", "count": replay + 1,
                            "status": "fail", "failed_checkpoint_relL2": value})
                        write_result()
                        raise AssertionError(
                            f"{tokens}_{route}{suffix} replay {replay + 1} relL2={value:.6f}")
            result["replay_results"].append({
                "guard": f"{tokens}_{route}{suffix}", "count": replays, "status": "pass",
                "input_changes": replays, "max_checkpoint_relL2": max(checkpoints)})
            write_result()

    def time_rank_max(graph):
        barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.perf_iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return reduce(start.elapsed_time(end) / args.perf_iters, dist.ReduceOp.MAX)

    def launch_count(op, inputs):
        from torch.profiler import ProfilerActivity, profile

        barrier()
        with profile(activities=[ProfilerActivity.CUDA]) as trace:
            adapter.forward(op, inputs)
            torch.cuda.synchronize()
        count = 0
        for event in trace.key_averages():
            name = str(event.key).lower()
            busy = float(getattr(event, "device_time_total", 0)
                         or getattr(event, "self_device_time_total", 0) or 0)
            if busy > 0 and not any(tag in name for tag in ("memcpy", "memset")):
                count += int(event.count)
        per_rank = gather_int(count)
        return {"min": min(per_rank), "max": max(per_rank), "per_rank": per_rank,
                "method": "torch profiler, one eager forward, memcpy/memset excluded"}

    replays = min(args.replays, 8) if args.self_test else args.replays
    try:
        if args.import_modules:
            progress("import identity")
            identity = {}
            for name in csv_strings(args.import_modules):
                module = __import__(name, fromlist=["_"])
                where = str(Path(getattr(module, "__file__", "") or "").resolve())
                identity[name] = where
                if not where.startswith(str(tree) + os.sep):
                    result["import_identity"] = identity
                    raise AssertionError(f"IMPORT_IDENTITY_VOID: {name} resolves to {where}")
            result["import_identity"] = identity

        arm_env = control_env if args.self_test else cand_env
        with scoped_env(arm_env):
            progress(f"build capacity={capacity}")
            op = adapter.build(capacity)
            target_graph = None
            for index, tokens in enumerate(cases):
                progress(f"capture {tokens}")
                inputs = adapter.make_inputs(tokens, routes[0], 0)
                state = {}
                marker = f"{output_path.name}.rank{rank}.activation" if index == 0 else None
                graph, counts = capture(op, inputs, state,
                                        None if args.self_test else marker)
                if counts is not None:
                    result["activation"] = {
                        **activation_summary(args.path_marker, world, counts),
                        "window": f"first eager forward (tokens={tokens}, capacity={capacity})"}
                    write_result()
                if tokens in accuracy_cases:
                    check_accuracy(state, inputs, str(tokens),
                                   "graph-captured candidate vs the adapter's numeric reference")
                if tokens in liveness_cases:
                    replay_routes(graph, state, inputs, tokens, "", replays)
                if tokens == target:
                    target_graph = (graph, inputs, state)
                else:
                    del graph

            result["mtpr_activation"] = []
            for tokens in mtpr_cases:
                cap = next_power_of_two(tokens)
                suffix = f"_mtpr{cap}"
                progress(f"build capacity={cap}")
                small = adapter.build(cap)
                inputs = adapter.make_inputs(tokens, routes[0], 0)
                state = {}
                graph, counts = capture(small, inputs, state,
                                        None if args.self_test
                                        else f"{output_path.name}.rank{rank}.mtpr{tokens}")
                if counts is not None:
                    result["mtpr_activation"].append({
                        **activation_summary(args.path_marker, world, counts),
                        "case": f"{tokens}{suffix}",
                        "fallback_allowed": cap <= args.mtpr_fallback_max})
                    write_result()
                check_accuracy(state, inputs, f"{tokens}_{routes[0]}{suffix}",
                               f"graph-captured candidate at capacity={cap} vs reference")
                replay_routes(graph, state, inputs, tokens, suffix, replays)
                del graph, small

            if args.self_test:
                result["self_test_pass"] = True
                result["progress"] = "done"
                dog.stop()
                write_result(False)
                return 0

            graph, inputs, state = target_graph
            fresh = adapter.make_inputs(target, routes[0], 0)
            for dst, src in zip(inputs, fresh):
                dst.copy_(src)
            progress("launch count")
            result["launch_count"] = launch_count(op, inputs)
            write_result()

        progress("control build")
        with scoped_env(control_env):
            control = adapter.build(capacity)
            control_state = {}
            control_graph, _ = capture(control, inputs, control_state)

        cand_ms, control_ms = [], []
        for pair in range(args.pairs):
            progress(f"timing pair {pair + 1}")
            if pair % 2:
                c = time_rank_max(graph)
                b = time_rank_max(control_graph)
            else:
                b = time_rank_max(control_graph)
                c = time_rank_max(graph)
            cand_ms.append(c)
            control_ms.append(b)
            result["paired_readings"].append({
                "guard": f"{target}_{routes[0]}", "base_arm": "same_tree_switch_off",
                "base": b, "cand": c})
            write_result()
        result.update(speedup_summary(args.frozen_baseline_ms, cand_ms, control_ms,
                                      args.min_speedup, args.denominator_tolerance))
        if args.resource_evidence:
            result["resource_evidence"] = {"path": args.resource_evidence,
                                           "status": "recorded_not_gated"}
        blockers = claim_blockers(result, args.launch_target)
        result["claim_blockers"] = blockers
        result["progress"] = "done"
        dog.stop()
        write_result(not blockers)
        return 1 if blockers else 0
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result.setdefault("claim_blockers", claim_blockers(result, args.launch_target)
                          if not args.self_test else ["self_test_failed"])
        dog.stop()
        write_result(False)
        raise
    finally:
        cleanup = getattr(adapter, "cleanup", None)
        if callable(cleanup):
            with contextlib.suppress(Exception):
                cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
