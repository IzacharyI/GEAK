#!/usr/bin/env python3
"""Immutable oracle for the Triton-to-FlyDSL fp8 block-scale GEMM author run.

Denominator: aiter-Triton gemm_a8w8_blockscale (W=fp8/A=fp8 e4m3fnuz, 128x128 block
scales, bf16 output), with the gfx942 config frozen per shape in baseline_src so
neither the baseline nor the candidate consults a mutable tuning DB. Candidate must
author the block-scale GEMM in FlyDSL and match the frozen baseline within tol.
"""

import argparse
import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
# This file must be named unittest.py for the workflow contract, but PyTorch imports
# stdlib unittest.mock. Preload stdlib unittest while this task directory is absent
# from the import path so this file cannot shadow the package.
_original_path = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
_original_cwd = os.getcwd()
os.chdir("/")
import unittest as _stdlib_unittest  # noqa: F401,E402
os.chdir(_original_cwd)
sys.path[:] = _original_path
sys.path.insert(0, HERE)
os.chdir(HERE)

import torch  # noqa: E402
import harness_lib as h  # noqa: E402
from aiter.ops.triton.utils.types import get_fp8_dtypes  # noqa: E402

_E5M2_DTYPE, FP8_DTYPE = get_fp8_dtypes()
BLOCK_N, BLOCK_K = 128, 128


def _tree_sha256(path):
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in {"__pycache__", "build"})
        for name in sorted(files):
            if name.endswith((".pyc", ".so", ".o")):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path)
            digest.update(rel.encode())
            digest.update(b"\0")
            with open(full, "rb") as fh:
                digest.update(fh.read())
    return digest.hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


with open(os.path.join(HERE, "meta.json"), encoding="utf-8") as fh:
    META = json.load(fh)


def verify_integrity():
    checks = {
        "baseline_src_sha256": _tree_sha256(os.path.join(HERE, "baseline_src")),
        "harness_lib_sha256": _file_sha256(os.path.join(HERE, "harness_lib.py")),
        "unittest_sha256": _file_sha256(__file__),
    }
    wrong = [f"{key}: got {got}, expected {META.get(key)}" for key, got in checks.items()
             if got != META.get(key)]
    if wrong:
        raise RuntimeError("immutable oracle integrity failure: " + "; ".join(wrong))


def _import_callable(spec):
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, attr):
            return getattr(module, attr)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
    if module_name != "kernel_src":
        raise ImportError(f"{module_name} does not export {attr}")
    candidates = []
    for name in sorted(os.listdir(os.path.join(HERE, "kernel_src"))):
        if name.endswith(".py") and name != "__init__.py":
            candidate = importlib.import_module(f"kernel_src.{name[:-3]}")
            if hasattr(candidate, attr):
                candidates.append(candidate)
    if len(candidates) != 1:
        raise ImportError(f"expected exactly one kernel_src module exporting {attr}, got {len(candidates)}")
    return getattr(candidates[0], attr)


def _code_only(source):
    """Blank comments and string contents while preserving token locations."""
    chars = list(source)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type not in {tokenize.COMMENT, tokenize.STRING}:
                continue
            (start_row, start_col), (end_row, end_col) = token.start, token.end
            lines = source.splitlines(keepends=True)
            start = sum(len(line) for line in lines[:start_row - 1]) + start_col
            end = sum(len(line) for line in lines[:end_row - 1]) + end_col
            for index in range(start, min(end, len(chars))):
                if chars[index] not in "\r\n":
                    chars[index] = " "
    except (IndentationError, tokenize.TokenError):
        return source
    return "".join(chars)


def verify_candidate_is_flydsl():
    root = os.path.join(HERE, "kernel_src")
    source = ""
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in sorted(files):
            if name.endswith(".py"):
                source += open(os.path.join(base, name), encoding="utf-8", errors="replace").read() + "\n"
    code = _code_only(source)
    violations = []
    if not re.search(r"^\s*(?:from\s+flydsl\b|import\s+flydsl\b)", code, re.M):
        violations.append("candidate has no FlyDSL import")
    banned = {
        r"^\s*(?:from|import)\s+triton\b": "Triton import",
        r"^\s*(?:from|import)\s+aiter\b": "AITER import",
        r"torch\.(?:matmul|mm|bmm|einsum|addmm)\s*\(": "torch matrix operation",
        r"\s@\s": "Python @ matrix operation",
        r"(?:hipblaslt|rocblas|cublas|load_inline|cpp_extension|torch\.compile)": "non-FlyDSL backend",
    }
    for pattern, label in banned.items():
        if re.search(pattern, code, re.M):
            violations.append(label)
    if violations:
        raise RuntimeError("FlyDSL-only candidate rule failed: " + ", ".join(violations))


def baseline_call(args):
    x, w, x_scale, w_scale = args
    return BASELINE(x, w, x_scale, w_scale)


def current_call(args):
    x, w, x_scale, w_scale = args
    return CURRENT(x, w, x_scale, w_scale)


def build_args(case, seed=None, rng=None):
    if rng is None:
        rng = torch.Generator(device="cuda").manual_seed(int(case["seed"] if seed is None else seed))
    m, n, k = case["m"], case["n"], case["k"]
    scale_k = (k + BLOCK_K - 1) // BLOCK_K
    scale_n = (n + BLOCK_N - 1) // BLOCK_N
    x = (torch.rand((m, k), generator=rng, dtype=torch.float16, device="cuda") / 10).to(FP8_DTYPE)
    w = (torch.rand((n, k), generator=rng, dtype=torch.float16, device="cuda") / 10).to(FP8_DTYPE)
    x_scale = torch.rand((m, scale_k), generator=rng, dtype=torch.float32, device="cuda")
    w_scale = torch.rand((scale_n, scale_k), generator=rng, dtype=torch.float32, device="cuda")
    return x, w, x_scale, w_scale


def shapes():
    entries = []
    for case in META["cases"]:
        entries.append({
            "sig": case["sig"],
            "make_inputs": lambda rng, case=case: build_args(case, rng=rng),
        })
    return entries


def check_correctness():
    seed = META.get("random_seed", 1907)
    draws = META.get("random_draws", 5)
    baseline_outputs = {}
    for shape in shapes():
        for draw in range(draws):
            rng = torch.Generator(device="cuda").manual_seed(seed + draw)
            baseline_outputs[f"{shape['sig']}|{draw}"] = baseline_call(
                shape["make_inputs"](rng)
            ).detach().cpu()
    all_ok, rows = h.check_random_vs_baseline(
        baseline_call,
        current_call,
        shapes(),
        META["tol"],
        draws=draws,
        seed=seed,
        baseline_outputs=baseline_outputs,
    )
    for row in rows:
        print("GEAK_CORRECTNESS: " + json.dumps(row, sort_keys=True))

    case = min(META["cases"], key=lambda c: c["m"] * c["n"] * c["k"])
    independent, reason = h.assert_independent_outputs(
        current_call,
        build_args(case, case["seed"] + 911),
        build_args(case, case["seed"] + 1911),
    )
    print("GEAK_CORRECTNESS: " + json.dumps({
        "case": "output_independence", "correct": independent, "note": reason,
    }, sort_keys=True))
    all_ok = all_ok and independent
    print("CORRECTNESS " + ("PASS" if all_ok else "FAIL"))
    return all_ok


def run_timing(warmup, repeats):
    per_case = []
    receipts = {}
    for case in META["cases"]:
        args = build_args(case)
        base = h.time_op(lambda: baseline_call(args), warmup, repeats, detail=True)
        cand = h.time_op(lambda: current_call(args), warmup, repeats, detail=True)
        if base is None or cand is None or not cand["ms"]:
            print(f"TIMING FAILED case={case['sig']}")
            return False
        speedup = base["ms"] / cand["ms"]
        per_case.append(speedup)
        receipts[case["sig"]] = {"baseline": base, "current": cand}
        print(
            f"GEAK_RESULT_LATENCY_MS={cand['ms']:.6f} case={case['sig']} "
            f"baseline_ms={base['ms']:.6f} optimized_ms={cand['ms']:.6f} speedup={speedup:.6f}"
        )
    geomean = math.exp(sum(math.log(x) for x in per_case) / len(per_case))
    all_primed = all(
        leg.get("primed") is True
        for receipt in receipts.values()
        for leg in receipt.values()
    )
    timer_unprimed = any(
        "primed" not in leg
        for receipt in receipts.values()
        for leg in receipt.values()
    )
    print(f"GEAK_GEOMEAN_SPEEDUP={geomean:.6f}")
    print("GEAK_TIMING_RECEIPT: " + json.dumps({
        "all_primed": all_primed,
        "timer_unprimed": timer_unprimed,
        "cases": receipts,
    }, sort_keys=True))
    print("PERFORMANCE PASS")
    return True


def run_baseline_smoke(warmup, repeats):
    rows = []
    for case in META["cases"]:
        args = build_args(case)
        first = baseline_call(args)
        second = baseline_call(args)
        ok, err = h.correct(first, second, META["tol"])
        detail = h.time_op(lambda: baseline_call(args), warmup, repeats, detail=True)
        rows.append(detail["ms"] if detail else float("nan"))
        print(f"[{'PASS' if ok else 'FAIL'}] {case['sig']}: self_parity={err:.3e} "
              f"baseline_ms={detail['ms'] if detail else None}")
        if not ok or detail is None:
            return False
    print(f"BASELINE_GEOMEAN_MS={math.exp(sum(math.log(x) for x in rows) / len(rows)):.6f}")
    print("BASELINE SMOKE PASS")
    return True


def main():
    global CURRENT
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?",
                        choices=("correctness", "performance", "compile", "baseline-smoke"),
                        default="performance")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    verify_integrity()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU available")
    if args.mode == "baseline-smoke":
        ok = run_baseline_smoke(args.warmup, args.repeats)
        raise SystemExit(0 if ok else 1)

    verify_candidate_is_flydsl()
    CURRENT = _import_callable(META["entry_point"])
    if args.mode == "correctness":
        ok = check_correctness()
    elif args.mode == "compile":
        sample = build_args(META["cases"][0])
        current_call(sample)
        torch.cuda.synchronize()
        print("COMPILE PASS")
        ok = True
    else:
        ok = check_correctness() and run_timing(args.warmup, args.repeats)
    raise SystemExit(0 if ok else 1)


verify_integrity()
BASELINE = _import_callable(META["baseline_callable"])
CURRENT = None

if __name__ == "__main__":
    main()
