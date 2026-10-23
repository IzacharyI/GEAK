#!/usr/bin/env python3
"""KernelFusion first, then the GEAK e2e workflow -- one entry point, two stages.

    run_kf_e2e.py <handoff.json> <result.json> [--timeout-s N] [--kf-budget-s N]
                  [--skip-kf] [--kf-only] [--dry-run]

Stage 1, KernelFusion (``e2e_workflow/kernel_fusion/kf_workflow.js``): a baseline leg on the
deployed stack, sampled traces + shapes after the kernel-extract skill's launch/capture method,
the top-1 GPU-time trace analysed for fusion patterns, observed-workload weighted unit tests, and
one combined apply-back A/B. It runs to completion (or to its own budget, default 12h) before
anything else starts.

Stage 2, the GEAK e2e workflow, is ``interface/run_e2e.py`` UNCHANGED, started in a subprocess on
a derived handoff: when KernelFusion accepted a fused stack, the handoff is schema v2 with the
fused overlay as ``baseline_env_spec.overlay_pythonpath`` (and KernelFusion's accepted flags/env),
so the e2e workflow measures its own baseline on the fused stack and optimizes from there. When
nothing was accepted, the original handoff is passed through byte-for-byte.

The final ``result.json`` is the e2e workflow's, plus a ``kernel_fusion`` section: KernelFusion's
own A/B (baseline vs fused, one measurement protocol), the e2e workflow's speedup over the fused
stack, and their product as the end-to-end gain over the original stack.

Everything about invoking a workflow (Claude SDK / CLI / standalone runtime, env exports for the
bench client and the AgentX workload, process marking and reaping) is reused from run_e2e.py; this
file only sequences the two stages.
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

INTERFACE_DIR = Path(__file__).resolve().parent
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))

import run_e2e as R  # noqa: E402  (the e2e runner; imported, never modified)

KF_SCRIPT = R.E2E_DIR / "kernel_fusion" / "kf_workflow.js"
KF_SCHEMA = "geak.kernel_fusion_stage/1"
DEFAULT_KF_BUDGET_S = 12 * 3600
# Handoff keys forwarded verbatim into the KernelFusion args (all optional, all fusion_*).
KF_ARG_PREFIXES = ("fusion_", "semantics_", "kf_")


def _now_tag() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_argv(argv: list[str]) -> tuple[list[str], dict[str, Any]]:
    pos: list[str] = []
    opts: dict[str, Any] = {"timeout_s": None, "kf_budget_s": None, "skip_kf": False,
                            "kf_only": False, "dry_run": False}
    it = iter(argv)
    for tok in it:
        if tok == "--timeout-s":
            opts["timeout_s"] = int(next(it))
        elif tok.startswith("--timeout-s="):
            opts["timeout_s"] = int(tok.split("=", 1)[1])
        elif tok == "--kf-budget-s":
            opts["kf_budget_s"] = int(next(it))
        elif tok.startswith("--kf-budget-s="):
            opts["kf_budget_s"] = int(tok.split("=", 1)[1])
        elif tok in ("--skip-kf", "--kf-only", "--dry-run"):
            opts[tok[2:].replace("-", "_")] = True
        else:
            pos.append(tok)
    return pos, opts


def total_timeout_s(opts: dict[str, Any]) -> int:
    """The run's hard budget: --timeout-s, else GEAK_E2E_TIMEOUT_S, else run_e2e's default."""
    if opts.get("timeout_s"):
        return int(opts["timeout_s"])
    env = os.environ.get("GEAK_E2E_TIMEOUT_S", "").strip()
    if env.isdigit() and int(env) > 0:
        return int(env)
    return R.DEFAULT_E2E_TIMEOUT_S


def kf_budget_s(h: dict, opts: dict[str, Any], total_s: int) -> int:
    """KernelFusion's share: --kf-budget-s, else handoff.kf_budget_s, else 12h; never more than
    the run leaves the e2e workflow less than a third of the total."""
    raw = opts.get("kf_budget_s") or h.get("kf_budget_s") or os.environ.get("GEAK_KF_BUDGET_S")
    budget = int(raw) if raw else DEFAULT_KF_BUDGET_S
    return max(1800, min(budget, int(total_s * 2 / 3)))


def kf_args(h: dict, ps_args: dict, eval_dir: Path, budget_s: int) -> dict:
    """KernelFusion workflow args: the e2e args' serving/workload identity + fusion_* knobs."""
    keep = ("model_path", "workflow_dir", "backend", "tp", "ep", "gpu_ids", "isl", "osl", "conc",
            "workload_spec", "workload_shape_provenance", "initial_extra_server_args",
            "initial_extra_env", "initial_overlay_pythonpath", "exp_root", "expected_gfx",
            "expected_target", "expected_device_name", "expected_physical_cu_count",
            "agent_timeout_ms")
    args = {k: copy.deepcopy(ps_args[k]) for k in keep if k in ps_args}
    args["eval_dir"] = str(eval_dir)
    args["time_budget_s"] = int(budget_s)
    if ps_args.get("expected_gfx"):
        args["gfx"] = ps_args["expected_gfx"]
    for key, value in h.items():
        if key.startswith(KF_ARG_PREFIXES) and key not in ("kf_budget_s",):
            args[key] = value
    if isinstance(h.get("kernel_fusion"), dict):  # nested form: {"kernel_fusion": {"fusion_top_k": 8}}
        args.update({k: v for k, v in h["kernel_fusion"].items() if k.startswith(KF_ARG_PREFIXES)})
    return args


def build_kf_prompt(args: dict) -> str:
    eval_dir = args["eval_dir"]
    return (
        R.PROCESS_SAFETY
        + "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{KF_SCRIPT}"\n'
        f"  args: {json.dumps(args)}\n"
        "CRITICAL: pass `args` as a real JSON OBJECT (a mapping), NOT as a JSON-encoded string.\n"
        "Run the whole KernelFusion stage (Baseline -> Sample -> Semantics -> Screen -> UnitSide -> "
        "ApplyBack -> Report). The workflow persists its return value to "
        f'"{eval_dir}/kf_return.json" (and the same content as workflow_return.json) as its final '
        "act; that file is the source of truth. When it "
        "finishes, print EXACTLY ONE final line of compact JSON that is the Workflow tool's full "
        "return value (it includes eval_dir, status, baseline_throughput_tok_s, "
        "final_throughput_tok_s, final_overlay). If for ANY reason "
        f'"{eval_dir}/kf_return.json" does not exist when the tool returns, write that exact return '
        "value there (and to workflow_return.json beside it) yourself with the Write tool before "
        "printing. Print nothing after the JSON line."
    )


def read_kf_return(eval_dir: Path, wf: dict | None) -> dict:
    """The stage's return: the workflow's own value, else kf_return.json, else a failure record."""
    if isinstance(wf, dict) and wf.get("schema") == KF_SCHEMA:
        return wf
    path = eval_dir / "kf_return.json"
    try:
        doc = json.loads(path.read_text())
        if isinstance(doc, dict):
            doc.setdefault("recovered_from_disk", True)
            return doc
    except (OSError, ValueError):
        pass
    return {"schema": KF_SCHEMA, "eval_dir": str(eval_dir), "status": "no_return",
            "applied_count": 0, "final_overlay": "", "notes": "KernelFusion returned nothing and "
            f"left no {path}; the e2e workflow starts from the original stack."}


def kf_accepted(kf: dict) -> bool:
    return (int(kf.get("applied_count") or 0) > 0
            and bool(str(kf.get("final_overlay") or "").strip())
            and str(kf.get("status") or "") in ("applied", "applyback_partial_failure"))


def derive_handoff(h: dict, kf: dict) -> dict:
    """The e2e workflow's handoff: the fused stack as its starting point, or `h` unchanged."""
    if not kf_accepted(kf):
        return h
    out = copy.deepcopy(h)
    out["schema_version"] = max(2, int(h.get("schema_version", 1) or 1))
    spec = dict(out.get("baseline_env_spec") or {})
    prior = str(spec.get("overlay_pythonpath") or "").strip()
    fused = str(kf["final_overlay"]).strip()
    # The fused loader already chains whatever overlay KernelFusion started from (its
    # sitecustomize runs CURRENT_OVERLAY first), so it leads; the prior one stays on the path for
    # its modules.
    spec["overlay_pythonpath"] = ":".join(dict.fromkeys(p for p in (fused, prior) if p))
    out["baseline_env_spec"] = spec
    if kf.get("accepted_flags") is not None:
        out["accepted_flags"] = kf["accepted_flags"]
    if kf.get("accepted_env") is not None:
        out["accepted_env"] = kf["accepted_env"]
    out["kernel_fusion_stage"] = {
        "eval_dir": kf.get("eval_dir"), "status": kf.get("status"),
        "baseline_throughput_tok_s": kf.get("baseline_throughput_tok_s"),
        "final_throughput_tok_s": kf.get("final_throughput_tok_s"),
        "final_overlay": fused,
    }
    return out


def kernel_fusion_section(kf: dict, main_result: dict, handoff_used: str) -> dict:
    kf_delta = kf.get("delta_pct")
    main_speedup = main_result.get("throughput_speedup")
    section = {
        "eval_dir": kf.get("eval_dir"),
        "status": kf.get("status"),
        "applied_count": kf.get("applied_count", 0),
        "baseline_throughput_tok_s": kf.get("baseline_throughput_tok_s"),
        "final_throughput_tok_s": kf.get("final_throughput_tok_s"),
        "delta_pct": kf_delta,
        "final_overlay": kf.get("final_overlay") if kf_accepted(kf) else "",
        "analysis_points": kf.get("analysis_points"),
        "observed_workload_json": kf.get("observed_workload_json"),
        "fusion_disposition": kf.get("fusion_disposition"),
        "main_handoff": handoff_used,
        "main_baseline_is_fused_stack": kf_accepted(kf),
    }
    # The two stages measure under their own protocols, so the end-to-end gain is the product of
    # the two same-protocol ratios, not a ratio across protocols.
    try:
        kf_ratio = 1.0 + float(kf_delta) / 100.0 if kf_accepted(kf) else 1.0
        if main_speedup is not None:
            section["combined_speedup_vs_original"] = round(kf_ratio * float(main_speedup), 4)
            section["combined_speedup_basis"] = (
                "(1 + kernel_fusion.delta_pct/100) x e2e throughput_speedup; each ratio is a "
                "same-protocol A/B, KernelFusion's on the original stack, the e2e workflow's on the "
                "fused stack")
    except (TypeError, ValueError):
        pass
    return section


def run_kf_stage(h: dict, ps_args: dict, exp_root: Path, budget_s: int) -> dict:
    eval_dir = exp_root / f"kernel_fusion_{_now_tag()}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    args = kf_args(h, ps_args, eval_dir, budget_s)
    prompt = build_kf_prompt(args)
    (eval_dir / "kf_args.json").write_text(json.dumps(args, indent=2))
    saved_script = R.E2E_SCRIPT
    R.E2E_SCRIPT = KF_SCRIPT          # the standalone runtime path runs whatever this names
    os.environ["GEAK_EVAL_DIR"] = str(eval_dir)
    run = R._mark_run(str(eval_dir))
    wf = None
    try:
        # A margin past the stage's own budget: the workflow paces itself and writes its return.
        wf = R.invoke_workflow(prompt, budget_s + 1800, str(eval_dir), ps_args=args)
    except Exception as exc:  # timeout / crash / parse: fall back to the on-disk return
        sys.stderr.write(f"[run_kf_e2e] KernelFusion stage ended abnormally "
                         f"[{R._classify_error(exc)}]: {exc}\n")
    finally:
        R.E2E_SCRIPT = saved_script
        R._stop_run_processes(run, str(eval_dir))
    kf = read_kf_return(eval_dir, wf)
    (eval_dir / "kf_stage_result.json").write_text(json.dumps(kf, indent=2))
    return kf


def main(argv: list[str]) -> int:
    pos, opts = _parse_argv(argv)
    if len(pos) < 2:
        sys.stderr.write("usage: run_kf_e2e.py <handoff.json> <result.json> [--timeout-s N] "
                         "[--kf-budget-s N] [--skip-kf] [--kf-only] [--dry-run]\n")
        return 2
    handoff_path, result_path = Path(pos[0]).resolve(), Path(pos[1]).resolve()
    h = R._read_json(handoff_path)
    if not h:
        sys.stderr.write(f"empty/invalid handoff: {handoff_path}\n")
        return 2
    started = time.monotonic()
    total_s = total_timeout_s(opts)
    budget_s = kf_budget_s(h, opts, total_s)
    exp_root = Path(h.get("exp_root") or handoff_path.parent)
    env_before = dict(os.environ)

    # The same env exports run_e2e.main performs, so every bench KernelFusion runs uses the
    # run's client, workload and protocol. The e2e subprocess gets the PRE-export environment
    # and performs them itself.
    ps_args = R.map_args(h, budget_s, dry_run=opts["dry_run"])
    R._publish_protected_pgids()
    exports = {
        "bench_client": R.apply_bench_client(h),
        "bench_launcher": R.apply_bench_launcher(h),
        "workload_spec": R.apply_workload_spec(h),
        "agentx_preflight_problems": R.agentx_preflight(h),
        "bench_protocol": R.apply_bench_protocol(h),
        "alignment_flags": R.apply_alignment_flags(h),
    }
    if opts["dry_run"]:
        args = kf_args(h, ps_args, exp_root / "kernel_fusion_<ts>", budget_s)
        print(json.dumps({"kf_script": str(KF_SCRIPT), "kf_args": args, "kf_budget_s": budget_s,
                          "total_timeout_s": total_s, "exports": exports,
                          "kf_prompt": build_kf_prompt(args)}, indent=2, default=str))
        return 0

    if opts["skip_kf"]:
        kf = {"schema": KF_SCHEMA, "status": "skipped", "applied_count": 0, "final_overlay": "",
              "notes": "--skip-kf"}
    else:
        kf = run_kf_stage(h, ps_args, exp_root, budget_s)
    sys.stderr.write(f"[run_kf_e2e] KernelFusion stage: status={kf.get('status')} "
                     f"applied={kf.get('applied_count', 0)} "
                     f"{kf.get('baseline_throughput_tok_s')} -> {kf.get('final_throughput_tok_s')}\n")

    main_handoff = derive_handoff(h, kf)
    if main_handoff is h:
        main_handoff_path = handoff_path
    else:
        main_handoff_path = exp_root / "handoff.after_kernel_fusion.json"
        main_handoff_path.write_text(json.dumps(main_handoff, indent=1))
    if opts["kf_only"]:
        out = {"schema_version": R.SCHEMA_VERSION, "status": "kf_only",
               "kernel_fusion": kernel_fusion_section(kf, {}, str(main_handoff_path))}
        result_path.write_text(json.dumps(out, indent=2))
        print(json.dumps({"status": "kf_only", "result_json": str(result_path)}))
        return 0

    remaining = max(1800, total_s - int(time.monotonic() - started))
    main_result_path = result_path.with_name(result_path.stem + ".e2e.json")
    cmd = [sys.executable, str(INTERFACE_DIR / "run_e2e.py"), str(main_handoff_path),
           str(main_result_path), "--timeout-s", str(remaining)]
    sys.stderr.write(f"[run_kf_e2e] e2e workflow: {' '.join(cmd)}\n")
    rc = subprocess.call(cmd, env=env_before)
    main_result = R._read_json(main_result_path) if main_result_path.exists() else {}
    out = dict(main_result) if main_result else {
        "schema_version": R.SCHEMA_VERSION, "status": "error", "error_class": "e2e_no_result",
        "error": f"run_e2e.py rc={rc} wrote no {main_result_path}"}
    out["kernel_fusion"] = kernel_fusion_section(kf, main_result, str(main_handoff_path))
    out["e2e_result_json"] = str(main_result_path)
    tmp = result_path.with_name(result_path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    os.replace(tmp, result_path)
    print(json.dumps({"status": out.get("status"), "result_json": str(result_path),
                      "kernel_fusion_status": kf.get("status"),
                      "speedup": out.get("throughput_speedup"),
                      "combined_speedup_vs_original":
                          out["kernel_fusion"].get("combined_speedup_vs_original")}))
    return 0 if out.get("status") != "error" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
