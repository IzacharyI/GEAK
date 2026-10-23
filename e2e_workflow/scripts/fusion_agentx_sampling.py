#!/usr/bin/env python3
"""KernelFusion trace evidence for an AgentX trace replay.

A trace replay has no single ISL/OSL/CONC, so a profile window placed inside a replay
leg lands on whatever batch happens to be resident at that moment. This script follows
the kernel-extract skill's launch-and-capture method instead
(knowledge/kernel_extraction_skills/kernel-extract/SKILL.md, references/dynamic-workloads.md,
references/capture.md):

  completed AgentX baseline  ->  fusion_observed_workload.py (the skill's observed_workload.py
                                                              + the hybrid-SWA log dialect)
                             ->  plan_sampling_points.py     (N points, each with the mass it stands for)
                             ->  fusion_serving_probe.py per point (steady state -> one bounded window)
                             ->  assemble                    (top-1 trace + UT weights)

Two things come out, and they are used for different jobs:

* `profile/` + the manifest: the ANALYSIS trace. By default that is the single decode
  point with the largest GPU-time mass -- the point's represented forward instances
  times the per-step device time its own verified trace measured -- linked under
  SGLang's `<stem>-DECODE` name, so trace_capability.py and the semantic mapper consume
  it like a `profile_by_stage` capture. `--analysis-phases prefill,decode` adds the
  heaviest prefill point the same way.
* `fusion_observed_workload.json`: every sampling point as a UT case with the number of
  forward instances it represents (`effective_calls`). The unit-side gate times each
  case and aggregates sum(baseline_ms*calls)/sum(candidate_ms*calls) with the shared
  harness_lib.serving_weighted_speedup -- the same contract as the kernel-extract UTs.

Subcommands
-----------
  plan      log (+ server_info) -> observed_workload.json, sampling_points.json,
            fusion_observed_workload.json.                          (CPU only)
            --synthetic ISL,OSL,CONC replaces the log for a fixed-shape workload.
  run       on a LIVE server (BASE_URL): fetch /get_server_info, plan, probe every
            point, assemble. fusion_serving_session.sh runs it on a server it owns.
  assemble  probe journals -> GPU-time mass -> profile/ links + trace manifest. (CPU only)

SGLang only: the log parser reads the SGLang scheduler log and the probe drives SGLang's
/generate, /metrics and /start_profile. Stdlib only.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DEFAULT_SKILL_ROOT = os.path.join(
    os.path.dirname(HERE), "knowledge", "kernel_extraction_skills", "kernel-extract")

PHASE_REGIME = {"PREFILL": "prefill", "DECODE": "decode"}
# Trace-file tag semantic_kernel_mapping pairs on (`<stem>-EXTEND/-DECODE.trace.json*`).
PHASE_TAG = {"PREFILL": "EXTEND", "DECODE": "DECODE"}
# The rank segments start at `-TP-<r>` and may be followed by `-EP-<e>`, `-DP-<d>`, ...
_RANK_SEGMENTS = r"-TP-\d+(?:-[A-Z]{2,4}-\d+)*"
_TRACE_RE = re.compile(r"^(?P<stem>.*?" + _RANK_SEGMENTS + r")(?:-(?:EXTEND|DECODE))?"
                       r"(?P<suffix>\.trace\.json(?:\.gz)?)$")


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)


def _skill_script(skill_root, name):
    path = os.path.join(skill_root, "scripts", name)
    if not os.path.isfile(path):
        raise SystemExit("fusion_agentx_sampling: kernel-extract script missing: %s" % path)
    return path


def _run(cmd):
    print(">>> " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc


# --------------------------------------------------------------------------- plan
def analysis_points(points):
    """The highest-mass point of each phase: the one semantic/Top-K analyses."""
    best = {}
    for point in points:
        phase = point.get("phase")
        mass = float((point.get("represented") or {}).get("proxy_mass") or 0)
        if phase in PHASE_REGIME and (phase not in best or mass > best[phase][0]):
            best[phase] = (mass, point["point_id"])
    return {PHASE_REGIME[phase]: pid for phase, (_mass, pid) in sorted(best.items())}


def ut_cases(points):
    """Every sampling point as an observed-workload UT case.

    `effective_calls` is the number of forward instances the point stands for
    (prefill batches / decode iterations, from plan_sampling_points `represented`).
    A fusion runs once per layer per forward in either phase, so forwards are one
    consistent basis across both regimes. `m` is the token rows of that forward:
    the batch's new tokens for prefill, the batch size for decode.
    """
    cases = []
    for point in points:
        phase = point.get("phase")
        if phase not in PHASE_REGIME:
            continue
        shape = {k: point.get(k) for k in (
            "new_tokens", "running", "cached_tokens", "batch_size", "padded_to",
            "context_tokens", "accept_len", "represents_batch_sizes") if point.get(k) is not None}
        m = point.get("new_tokens") if phase == "PREFILL" else point.get("batch_size")
        cases.append({
            "sig": point["point_id"],
            "regime": PHASE_REGIME[phase],
            "m": int(m),
            "effective_calls": float((point.get("represented") or {}).get("instances") or 0),
            "phase_mass_share_pct": (point.get("represented") or {}).get(
                "phase_proxy_mass_share_pct"),
            "shape": shape,
        })
    return cases


def build_ut_workload(points_doc, observed_path, points_path):
    points = points_doc.get("points") or []
    return {
        "schema_version": 1,
        "kind": "agentx_trace_replay",
        "weighting": "observed",
        "serving_weight_model": {
            "source": "observed_workload",
            "evidence_path": os.path.abspath(observed_path),
            "sampling_points": os.path.abspath(points_path),
            "basis": "forward instances (prefill batches / decode iterations) each "
                     "sampling point represents in the measured baseline leg",
        },
        "analysis_points": analysis_points(points),
        "cases": ut_cases(points),
        "uncovered_candidates": points_doc.get("uncovered_candidates") or [],
        "aggregate": "sum(baseline_ms*effective_calls) / sum(candidate_ms*effective_calls) "
                     "over the cases of the phases a candidate runs in "
                     "(harness_lib.serving_weighted_speedup)",
    }


def synthetic_points(spec, dp_size=1):
    """Sampling points for a fixed ISL/OSL/CONC workload (no serving log to read).

    One prefill point (one request's ISL) and one decode point (the per-rank resident
    batch at the mean context). Instances are per CONC requests of the workload: one
    prefill forward per request, OSL decode iterations per batch.
    """
    isl, osl, conc = (int(v) for v in str(spec).split(","))
    per_rank = max(1, conc // max(1, int(dp_size)))
    return {
        "source": "synthetic", "isl": isl, "osl": osl, "conc": conc,
        "dp_attention_size": dp_size,
        "points": [
            {"point_id": "pr_syn", "phase": "PREFILL", "new_tokens": isl, "running": 1,
             "cached_tokens": 0,
             "represented": {"instances": float(conc), "proxy_mass": float(conc * isl),
                             "phase_proxy_mass_share_pct": 100.0}},
            {"point_id": "de_syn", "phase": "DECODE", "batch_size": per_rank,
             "context_tokens": isl + osl // 2,
             "represented": {"instances": float(osl), "proxy_mass": float(osl * per_rank),
                             "phase_proxy_mass_share_pct": 100.0}},
        ],
    }


def plan(args):
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    observed = os.path.join(out_dir, "observed_workload.json")
    points_path = os.path.join(out_dir, "sampling_points.json")
    if getattr(args, "synthetic", None):
        _dump(points_path, synthetic_points(args.synthetic, _dp_attention_size(args.server_info)))
        _dump(observed, {"kind": "synthetic", "shape": args.synthetic,
                         "note": "fixed ISL/OSL/CONC workload: no serving log is parsed"})
    else:
        cmd = [sys.executable, os.path.join(HERE, "fusion_observed_workload.py"),
               "--log", *args.log, "--rank", str(args.rank), "--out", observed]
        for flag, value in (("--server-info", args.server_info),
                            ("--decode-log-interval", args.decode_log_interval),
                            ("--bench-log", args.bench_log),
                            ("--completed-output-tokens", args.completed_output_tokens)):
            if value not in (None, ""):
                cmd += [flag, str(value)]
        if _run(cmd).returncode:
            raise SystemExit("fusion_agentx_sampling: observed_workload.py failed")
        cmd = [sys.executable, _skill_script(args.skill_root, "plan_sampling_points.py"),
               "--observed", observed, "-n", str(args.num_points), "--out", points_path]
        if args.server_info:
            cmd += ["--server-info", args.server_info]
        if args.phase_budget:
            cmd += ["--phase-budget", args.phase_budget]
        if _run(cmd).returncode:
            raise SystemExit("fusion_agentx_sampling: plan_sampling_points.py failed")
    workload = build_ut_workload(_load(points_path), observed, points_path)
    if getattr(args, "synthetic", None):
        workload["kind"] = "synthetic_fixed_shape"
        workload["weighting"] = "analytic"
        workload["serving_weight_model"]["source"] = "synthetic_isl_osl_conc"
    missing = [r for r in ("prefill", "decode") if r not in workload["analysis_points"]]
    if missing:
        workload["missing_phases"] = missing
    path = os.path.join(out_dir, "fusion_observed_workload.json")
    _dump(path, workload)
    print("analysis points: %s; %d UT cases -> %s"
          % (workload["analysis_points"], len(workload["cases"]), path))
    return path


# --------------------------------------------------------------------------- assemble
def _rank_traces(point_dir):
    """{rank_stem: largest trace file} for one point's window."""
    best = {}
    for path in glob.glob(os.path.join(point_dir, "**", "*.trace.json*"), recursive=True):
        match = _TRACE_RE.match(os.path.basename(path))
        if not match or not os.path.isfile(path):
            continue
        stem = re.search(_RANK_SEGMENTS + "$", match.group("stem")).group(0)
        size = os.path.getsize(path)
        if stem not in best or size > best[stem][0]:
            best[stem] = (size, path, match.group("suffix"))
    return {stem: (path, suffix) for stem, (_size, path, suffix) in best.items()}


def _journal(journal_path):
    if not os.path.isfile(journal_path):
        return None
    try:
        return _load(journal_path)
    except (OSError, ValueError):
        return None


def _journal_ok(journal_path):
    doc = _journal(journal_path)
    if doc is None:
        return False, "no journal"
    return doc.get("status") == "ok", doc.get("verify_reason") or doc.get("status")


def gpu_time_mass(case, journal):
    """(mass_us, us_per_step, basis): the point's represented forwards x its measured step cost.

    The cost is the probe's own verified trace (fusion_serving_probe.py `step_cost`), so
    the ranking is in device time, not in the log's iteration or token proxy."""
    cost = (journal or {}).get("step_cost") or {}
    per_step = float(cost.get("us_per_step") or 0)
    calls = float(case.get("effective_calls") or 0)
    return calls * per_step, per_step, cost.get("basis")


def select_analysis(cases, points_root, phases):
    """Per analysis phase, the verified point with the largest GPU-time mass.

    Returns ({regime: sig}, [ranking rows], [notes]). A point without a verified trace
    or without a measured step cost cannot be ranked in device time; it stays in the
    table with its reason, and the proxy mass only breaks a tie where no point of the
    phase has a measured cost."""
    used, ranking, notes = {}, [], []
    for regime in phases:
        rows = []
        for case in (c for c in cases if c["regime"] == regime):
            journal = _journal(os.path.join(points_root, case["sig"] + ".journal.json"))
            mass, per_step, basis = gpu_time_mass(case, journal)
            has_trace = bool(_rank_traces(os.path.join(points_root, case["sig"])))
            row = {"regime": regime, "sig": case["sig"], "m": case.get("m"),
                   "effective_calls": case.get("effective_calls"),
                   "us_per_step": per_step, "cost_basis": basis,
                   "gpu_time_mass_us": round(mass, 1),
                   "proxy_share_pct": case.get("phase_mass_share_pct"),
                   "trace_verified": bool(case.get("trace_verified")) and has_trace}
            rows.append(row)
        total = sum(r["gpu_time_mass_us"] for r in rows) or 0.0
        for r in rows:
            r["gpu_time_share_pct"] = round(100.0 * r["gpu_time_mass_us"] / total, 2) if total else None
        rows.sort(key=lambda r: (-r["gpu_time_mass_us"], -float(r["proxy_share_pct"] or 0)))
        ranking.extend(rows)
        pick = next((r for r in rows if r["trace_verified"] and r["gpu_time_mass_us"] > 0), None)
        if pick is None:
            pick = next((r for r in rows if r["trace_verified"]), None)
            if pick is not None:
                notes.append("%s: no verified point has a measured step cost; fell back to the "
                             "proxy-mass order (%s)" % (regime, pick["sig"]))
        if pick is None:
            notes.append("no %s sampling point produced a verified trace" % regime)
            continue
        used[regime] = pick["sig"]
        best = rows[0]
        if best["sig"] != pick["sig"]:
            notes.append("%s: top GPU-time point %s has no verified trace; analysed %s"
                         % (regime, best["sig"], pick["sig"]))
    return used, ranking, notes


def assemble(args):
    plan_dir = os.path.abspath(args.plan_dir)
    workload_path = os.path.join(plan_dir, "fusion_observed_workload.json")
    workload = _load(workload_path)
    points_root = os.path.abspath(args.points_root or os.path.join(plan_dir, "points"))
    profile_dir = os.path.join(os.path.abspath(args.capture_dir), "profile")
    os.makedirs(profile_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(profile_dir, "fusion-agentx*")):
        os.remove(stale)
    phases = [p for p in ("prefill", "decode")
              if p in [x.strip() for x in str(args.analysis_phases).split(",")]]
    if not phases:
        raise SystemExit("fusion_agentx_sampling: --analysis-phases names no phase (prefill, decode)")

    status = {}
    for case in workload["cases"]:
        ok, why = _journal_ok(os.path.join(points_root, case["sig"] + ".journal.json"))
        case["trace_verified"] = ok
        status[case["sig"]] = {"ok": ok, "reason": why}

    # The analysis trace is the top-1 point by GPU-time mass; a point whose window failed
    # hands over to the next one of the same phase, and the substitution is recorded.
    used, ranking, notes = select_analysis(workload["cases"], points_root, phases)

    linked = []
    for regime, sig in sorted(used.items()):
        phase = "PREFILL" if regime == "prefill" else "DECODE"
        for stem, (path, suffix) in sorted(_rank_traces(os.path.join(points_root, sig)).items()):
            link = os.path.join(profile_dir, "fusion-agentx%s-%s%s" % (stem, PHASE_TAG[phase], suffix))
            os.symlink(os.path.abspath(path), link)
            linked.append({"regime": regime, "point_id": sig, "trace": path, "link": link})

    import trace_capability
    manifest = trace_capability.build_manifest(
        profile_dir, auto_select_rank=True, require_phases=phases,
        server_log=args.server_log or None)
    manifest["agentx_sampling"] = {
        "fusion_observed_workload_json": workload_path,
        "analysis_phases": phases,
        "analysis_points": used, "planned_analysis_points": workload["analysis_points"],
        "analysis_selection": "top-1 point per analysis phase by GPU-time mass "
                              "(effective_calls x measured per-step device time)",
        "gpu_time_ranking": ranking,
        "linked_traces": linked, "notes": notes,
    }
    _dump(args.manifest, manifest)
    workload["analysis_phases"] = phases
    workload["analysis_points_used"] = used
    workload["gpu_time_ranking"] = ranking
    workload["probe_status"] = status
    workload["trace_manifest_json"] = os.path.abspath(args.manifest)
    _dump(workload_path, workload)
    summary = {"status": manifest.get("status"), "trace_manifest_json": args.manifest,
               "fusion_observed_workload_json": workload_path,
               "analysis_phases": phases, "analysis_points": used,
               "gpu_time_ranking": ranking, "probe_status": status, "notes": notes,
               "missing_required_phases": manifest.get("missing_required_phases")}
    _dump(os.path.join(plan_dir, "fusion_sampling_summary.json"), summary)
    print(json.dumps(summary, indent=1))
    return 0 if manifest.get("status") == "pass" else 2


# --------------------------------------------------------------------------- run
def _server_info(base_url, out):
    with urllib.request.urlopen(base_url.rstrip("/") + "/get_server_info", timeout=60) as resp:
        doc = json.loads(resp.read().decode())
    _dump(out, doc)
    return out


def _dp_attention_size(server_info_path):
    """DP-attention ranks: each logs and runs its OWN batch (observed_workload --rank)."""
    try:
        info = _load(server_info_path)
    except (OSError, ValueError, TypeError):
        return 1
    if not info.get("enable_dp_attention"):
        return 1
    return max(1, int(info.get("attn_dp_size") or info.get("dp_size") or 1))


def run(args):
    base_url = args.base_url or os.environ.get("BASE_URL")
    if not base_url:
        raise SystemExit("fusion_agentx_sampling run: no --base-url and no BASE_URL")
    plan_dir = os.path.abspath(args.out_dir)
    if not args.server_info:
        args.server_info = _server_info(base_url, os.path.join(plan_dir, "server_info.json"))
    plan(args)
    dp_size = _dp_attention_size(args.server_info)
    if dp_size > 1:
        # The points are rank-`args.rank` batches; tell the UT what M means.
        wl_path = os.path.join(plan_dir, "fusion_observed_workload.json")
        workload = _load(wl_path)
        workload["dp_attention_size"] = dp_size
        workload["m_basis"] = ("per DP-attention rank: attention-side kernels run m rows per "
                               "rank; kernels after the DP gather (MoE / dense TP) see the "
                               "sum over %d ranks" % dp_size)
        _dump(wl_path, workload)
    points_doc = _load(os.path.join(plan_dir, "sampling_points.json"))
    points_root = os.path.join(plan_dir, "points")
    probe = os.path.join(HERE, "fusion_serving_probe.py")
    for point in points_doc.get("points") or []:
        pid = point["point_id"]
        # One recapture per failed point, into its own directory (capture.md), then
        # report the gap and move on.
        for attempt, suffix in enumerate(("", "_retry")):
            trace_dir = os.path.join(points_root, pid + suffix)
            os.makedirs(trace_dir, exist_ok=True)
            journal = os.path.join(points_root, pid + suffix + ".journal.json")
            cmd = [sys.executable, probe, "--points", os.path.join(plan_dir, "sampling_points.json"),
                   "--point-id", pid, "--base-url", base_url, "--trace-dir", trace_dir,
                   "--record-shapes", "--with-stack", "--dp-size", str(dp_size),
                   "--steady-timeout", str(args.steady_timeout),
                   "--arm-timeout", str(args.arm_timeout),
                   "--out", journal]
            if args.server_trace_root:
                cmd += ["--server-trace-dir", os.path.join(args.server_trace_root, pid + suffix)]
            _run(cmd)
            if _journal_ok(journal)[0]:
                if suffix:
                    # The retry is the evidence of record for this point.
                    os.replace(journal, os.path.join(points_root, pid + ".journal.json"))
                    first = os.path.join(points_root, pid)
                    os.rename(first, first + "_failed")
                    os.rename(trace_dir, first)
                break
    return assemble(argparse.Namespace(
        plan_dir=plan_dir, points_root=points_root, capture_dir=args.capture_dir,
        manifest=args.manifest, server_log=args.server_log,
        analysis_phases=args.analysis_phases))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def plan_args(p):
        p.add_argument("--log", nargs="+", default=[],
                       help="baseline SGLang server log(s) of the measured AgentX leg")
        p.add_argument("--synthetic", default=None, metavar="ISL,OSL,CONC",
                       help="fixed-shape workload: build the points from these instead of a log")
        p.add_argument("--rank", type=int, default=0)
        p.add_argument("--server-info", default=None, help="/get_server_info JSON")
        p.add_argument("--decode-log-interval", type=int, default=None)
        p.add_argument("--bench-log", default=None)
        p.add_argument("--completed-output-tokens", type=int, default=None)
        p.add_argument("-n", "--num-points", type=int, default=6)
        p.add_argument("--phase-budget", default=None, help="e.g. PREFILL=3,DECODE=3")
        p.add_argument("--skill-root", default=DEFAULT_SKILL_ROOT)
        p.add_argument("--out-dir", required=True)

    def assemble_args(p, plan_dir=True):
        if plan_dir:
            p.add_argument("--plan-dir", required=True)
            p.add_argument("--points-root", default=None)
        p.add_argument("--capture-dir", required=True, help="CAPTURE_DIR; traces go to profile/")
        p.add_argument("--manifest", required=True, help="TRACE_MANIFEST_JSON to write")
        p.add_argument("--server-log", default=None)
        p.add_argument("--analysis-phases", default="decode",
                       help="phases whose top-1 GPU-time point is the analysis trace "
                            "(default decode; prefill,decode adds the heaviest prefill point)")

    p_plan = sub.add_parser("plan")
    plan_args(p_plan)
    p_run = sub.add_parser("run")
    plan_args(p_run)
    assemble_args(p_run, plan_dir=False)
    p_run.add_argument("--base-url", default=None, help="defaults to $BASE_URL")
    p_run.add_argument("--server-trace-root", default=None,
                       help="points/ as the SERVER sees it, if it differs (container)")
    p_run.add_argument("--steady-timeout", type=float, default=1800,
                       help="per point: seconds to reach the planned resident batch (long "
                            "agentic contexts prefill for minutes before decode is steady)")
    p_run.add_argument("--arm-timeout", type=float, default=600)
    p_asm = sub.add_parser("assemble")
    assemble_args(p_asm)
    args = ap.parse_args(argv)
    if args.cmd in ("plan", "run") and not (args.log or args.synthetic):
        ap.error("plan/run need --log (AgentX baseline) or --synthetic ISL,OSL,CONC")
    if args.cmd == "plan":
        plan(args)
        return 0
    if args.cmd == "run":
        return run(args)
    return assemble(args)


if __name__ == "__main__":
    sys.exit(main())
