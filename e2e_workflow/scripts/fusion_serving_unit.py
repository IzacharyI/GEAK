#!/usr/bin/env python3
"""Serving-side unit test: time a fusion overlay on the LIVE serving stack at every decode
sampling point, for windows a single card cannot reproduce.

Most DeepSeek-V4 windows came back "not isolable" from the single-card validator: their
members read and write the paged-KV / compressed-KV / SWA ring state of the serving attention
backend (flash_c4/c128_prefill, write_c4/c128, _swa_scatter_kernel, fused_norm_rope_flashmla),
which no standalone microbench can rebuild faithfully. The served step is the one place that
state exists. So this measures the overlay where it runs:

  base  = one fusion_serving_session.sh server, no overlay   (measured once, cached)
  cand  = the same server config with the candidate's overlay loader on PYTHONPATH

On each side, every decode sampling point is driven to steady state and profiled with
fusion_serving_probe.py (the same probe the sampling stage used, without --record-shapes), and
gsm8k runs on the same live server. The verdict compares the two sides:

  timing      per point, the per-step device time of every kernel OUTSIDE the expert-parallel
              MoE (megamoe / ep_combine ...), median over steps, mean over the parsed ranks.
              The EP MoE is excluded because EPLB re-places experts per session, which moves
              its time by more than a fusion is worth; the screened windows never contain it
              (fusion_screen.py rejects crosses_ep_moe), so the excluded time is not the
              fusion's. ref_ms / cand_ms per point go to `weighted_cases`; the harness weights
              them by the observed workload's effective_calls.
  accuracy    gsm8k two-proportion z of the base->cand drop < --max-z (fusion_combined_decide).
  engagement  "[overlay-<banner>] ENGAGED" on >= --tp lines of the cand server log.

    fusion_serving_unit.py batch --root DIR --observed-workload fusion_observed_workload.json \
        --tp 8 --verdict-dir VERDICT_DIR --cand CID::LOADER_DIR::BANNER::KERNEL_SOURCE [--cand ...]

--observed-workload supplies the sampling points file, the decode point ids and the DP-attention
size; --points / --point-ids / --dp-size override them.

BACKEND MODEL TP GPU EXTRA_SERVER_ARGS EXTRA_ENV (the run's serving config) come from the
environment, exactly as fusion_combined_ab.sh takes them. Each verdict is written to
VERDICT_DIR/<CID>__serving.json with measure_scope=serving_step. The result is printed as one
JSON line and written to ROOT/result.json (the completion marker a waiter polls for).
"""
import argparse
import glob
import gzip
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Same boundary as fusion_screen._EP_MOE_KERNEL.
EP_MOE_KERNEL = re.compile(r"megamoe|ep_combine|ep_dispatch|deepep|mori_|all_?to_?all|a2a_",
                           re.IGNORECASE)
STEP_ANNOTATION = re.compile(r"^step\[(DECODE|VERIFY|DRAFT|EXTEND|PREFILL)\b")
MEASURE_SCOPE = "serving_step"
TRACE_RANKS = 4          # ranks parsed per point (the probe's step_cost parses the same count)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


def _dump(path, doc):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def trace_step_us(path):
    """(median non-EP kernel us per step, median EP us per step, steps) of one rank's trace.

    A step is one decode cycle: the device time of every kernel inside the step[...] GPU
    annotations between consecutive VERIFY (or DECODE) steps, so a speculative draft step
    counts with the verify step it feeds."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        events = json.load(fh)
    events = events.get("traceEvents", events) if isinstance(events, dict) else events
    windows = []
    for ev in events:
        if ev.get("cat") != "gpu_user_annotation":
            continue
        m = STEP_ANNOTATION.match(str(ev.get("name", "")))
        if m and ev.get("dur") is not None:
            windows.append((float(ev["ts"]), float(ev["ts"]) + float(ev["dur"]), m.group(1)))
    windows.sort()
    if not windows:
        return None
    anchors = [w for w in windows if w[2] in ("VERIFY", "DECODE")] or windows
    # cycle index of each annotation window: the anchor step at or after it
    starts = [w[0] for w in windows]
    cycle_of = []
    a = 0
    for w in windows:
        while a < len(anchors) - 1 and anchors[a][1] <= w[0]:
            a += 1
        cycle_of.append(a)
    non_ep = [0.0] * len(anchors)
    ep = [0.0] * len(anchors)
    import bisect
    for ev in events:
        if ev.get("cat") != "kernel" or ev.get("dur") is None:
            continue
        ts = float(ev["ts"])
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts >= windows[i][1]:
            continue
        bucket = ep if EP_MOE_KERNEL.search(str(ev.get("name", ""))) else non_ep
        bucket[cycle_of[i]] += float(ev["dur"])
    # first and last cycle may be cut by the profile window
    body = list(range(1, len(anchors) - 1)) if len(anchors) > 4 else list(range(len(anchors)))
    return (statistics.median(non_ep[i] for i in body), statistics.median(ep[i] for i in body),
            len(body))


def point_cost(journal):
    """Per-point step cost from a probe journal's traces (non-EP / EP kernel us per step)."""
    doc = _load(journal)
    if doc.get("status") != "ok" or not doc.get("trace_verified"):
        return {"ok": False, "reason": doc.get("verify_reason") or doc.get("status") or "probe failed"}
    paths = sorted(t["path"] for t in doc.get("traces") or [] if t.get("path"))
    per_rank = []
    for path in paths[:TRACE_RANKS]:
        got = trace_step_us(path)
        if got:
            per_rank.append({"file": os.path.basename(path), "non_ep_us": round(got[0], 3),
                             "ep_us": round(got[1], 3), "steps": got[2]})
    if not per_rank:
        return {"ok": False, "reason": "no step annotations in the probe traces"}
    return {"ok": True, "per_rank": per_rank,
            "non_ep_us": statistics.mean(r["non_ep_us"] for r in per_rank),
            "ep_us": statistics.mean(r["ep_us"] for r in per_rank),
            "probe_step_cost": doc.get("step_cost")}


# ---- one side: a session whose command probes every point, then runs gsm8k ----------------
def _healthy(base_url, timeout=20):
    import urllib.request
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - any failure means unhealthy
        return False


def session_body(args):
    """Runs INSIDE fusion_serving_session.sh (BASE_URL/OUT_DIR exported)."""
    base_url = os.environ["BASE_URL"]
    out_dir = os.environ["OUT_DIR"]
    probe = os.path.join(HERE, "fusion_serving_probe.py")
    # A probe that outlives POINT_TIMEOUT_S is killed, and a server that no longer answers
    # /health ends the session: DeepSeek-V4 run 13 hung on rank 0 while re-arming the profiler
    # for its second point (rank 0 idle, ranks 1-7 spinning), and the probe's own timeouts plus
    # a retry let one side sit for ~2 h on a dead server.
    for pid in args.point_ids.split(","):
        journal = os.path.join(out_dir, "points", pid + ".journal.json")
        for attempt in range(2):
            cmd = [sys.executable, probe, "--points", args.points, "--point-id", pid,
                   "--base-url", base_url, "--trace-dir", os.path.join(out_dir, "points", pid),
                   "--dp-size", str(args.dp_size), "--steady-timeout", "300",
                   "--arm-timeout", "300", "--window-timeout", "300",
                   "--request-timeout", "600", "--out", journal]
            print(">>> " + " ".join(shlex.quote(c) for c in cmd), flush=True)
            try:
                subprocess.call(cmd, timeout=args.point_timeout_s)
            except subprocess.TimeoutExpired:
                print("!!! %s: probe exceeded %ss; killed" % (pid, args.point_timeout_s), flush=True)
            try:
                if _load(journal).get("status") == "ok":
                    break
            except (OSError, ValueError):
                pass
            if not _healthy(base_url):
                print("!!! server no longer answers /health after %s; ending this side" % pid,
                      flush=True)
                return 3
    if args.gsm8k_limit <= 0:
        return 0
    served = args.model_name
    try:
        import urllib.request
        with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as resp:
            served = json.load(resp)["data"][0]["id"]
    except Exception:  # noqa: BLE001 - fall back to the model path
        pass
    cmd = [sys.executable, os.path.join(HERE, "fusion_gsm8k_eval.py"), "--base-url",
           base_url + "/v1", "--model", served, "--limit", str(args.gsm8k_limit),
           "--max-tokens", "4096", "--seed", "0", "--concurrency", str(args.gsm8k_concurrency),
           "--no-thinking", "--out", os.path.join(out_dir, "gsm8k.json")]
    print(">>> " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    try:
        subprocess.call(cmd, timeout=args.point_timeout_s)
    except subprocess.TimeoutExpired:
        print("!!! gsm8k exceeded %ss; killed" % args.point_timeout_s, flush=True)
    return 0


def measure_side(side_dir, overlay, args):
    """Measure one side into side_dir (reused when its side.json is complete).

    ONE server session per sampling point (gsm8k rides on the last one). Re-arming the
    profiler on a live server hung a rank on DeepSeek-V4 twice (run 13 and run 14: the first
    point traced, the second left one GPU idle and the other seven spinning in a collective),
    so no server ever opens a second profiling window. A point whose journal is already ok is
    not re-measured."""
    done = os.path.join(side_dir, "side.json")
    if os.path.isfile(done):
        doc = _load(done)
        if doc.get("ok"):
            return doc
    os.makedirs(side_dir, exist_ok=True)
    started = time.time()
    point_ids = args.point_ids.split(",")
    rcs, points, last_log, gsm_path = {}, {}, "", ""
    for i, pid in enumerate(point_ids):
        sess = os.path.join(side_dir, "sess_" + pid)
        journal = os.path.join(sess, "points", pid + ".journal.json")
        last = i == len(point_ids) - 1
        cost = point_cost(journal) if os.path.isfile(journal) else {"ok": False}
        if not cost.get("ok") or (last and not os.path.isfile(os.path.join(sess, "gsm8k.json"))):
            os.makedirs(sess, exist_ok=True)
            body = " ".join(shlex.quote(c) for c in [
                sys.executable, os.path.abspath(__file__), "session-body", "--points", args.points,
                "--point-ids", pid, "--dp-size", str(args.dp_size),
                "--gsm8k-limit", str(args.gsm8k_limit if last else 0),
                "--gsm8k-concurrency", str(args.gsm8k_concurrency),
                "--point-timeout-s", str(args.point_timeout_s),
                "--model-name", os.environ.get("MODEL", "")])
            env = dict(os.environ, OUT_DIR=sess, OVERLAY_PYTHONPATH=overlay or "", SESSION_CMD=body)
            with open(os.path.join(sess, "session.log"), "w") as log:
                rcs[pid] = subprocess.call(["bash", os.path.join(HERE, "fusion_serving_session.sh")],
                                           env=env, stdout=log, stderr=subprocess.STDOUT)
            cost = point_cost(journal) if os.path.isfile(journal) else {
                "ok": False, "reason": "no probe journal"}
        points[pid] = cost
        last_log = os.path.join(sess, "server.log")
        if last:
            gsm_path = os.path.join(sess, "gsm8k.json")
    from fusion_combined_decide import load_gsm8k
    gsm = load_gsm8k(gsm_path) if gsm_path else None
    doc = {"ok": all(p.get("ok") for p in points.values()) and gsm is not None,
           "session_rc": rcs, "overlay": overlay or "", "points": points,
           "gsm8k": list(gsm) if gsm else None, "server_log": last_log, "side_dir": side_dir,
           "wall_s": round(time.time() - started, 1)}
    _dump(done, doc)
    return doc


def verdict_for(cid, banner, kernel_source, base, cand, args):
    from fusion_combined_decide import accuracy_z, engaged_counts
    cases, problems = [], []
    for pid in args.point_ids.split(","):
        b, c = base["points"].get(pid) or {}, cand["points"].get(pid) or {}
        if not (b.get("ok") and c.get("ok")):
            problems.append("%s not measured on %s" % (pid, "base" if not b.get("ok") else "cand"))
            continue
        cases.append({"sig": pid, "ref_ms": round(b["non_ep_us"] / 1000.0, 6),
                      "cand_ms": round(c["non_ep_us"] / 1000.0, 6),
                      "ref_ep_ms": round(b["ep_us"] / 1000.0, 6),
                      "cand_ep_ms": round(c["ep_us"] / 1000.0, 6)})
    engaged_n = engaged_counts(cand["server_log"], [banner]).get(banner, 0) if banner else 0
    engaged = engaged_n >= max(1, args.tp)
    acc = None
    if base.get("gsm8k") and cand.get("gsm8k"):
        z = accuracy_z(tuple(base["gsm8k"]), tuple(cand["gsm8k"]))
        acc = {"base": base["gsm8k"][0], "cand": cand["gsm8k"][0],
               "n": [base["gsm8k"][1], cand["gsm8k"][1]], "z": round(z, 3), "max_z": args.max_z,
               "pass": z < args.max_z}
    else:
        problems.append("gsm8k missing on %s" % ("base" if not base.get("gsm8k") else "cand"))
    ref = sum(x["ref_ms"] for x in cases)
    cnd = sum(x["cand_ms"] for x in cases)
    speedup = (ref / cnd) if cases and cnd > 0 else 0.0
    if problems:
        parity = "not_measured"
    elif not engaged:
        parity = "not_engaged"
    elif not acc["pass"]:
        parity = "fail"
    else:
        parity = "pass"
    doc = {
        "candidate_id": cid,
        "implementation": "self_authored",
        "family": "serving_step",
        "fused_fn": banner,
        "kernel_source": kernel_source,
        "measure_scope": MEASURE_SCOPE,
        "tested_shape": "served decode steps at %s" % args.point_ids,
        "dtypes": "as served",
        "tol": "gsm8k two-proportion z < %s" % args.max_z,
        "parity": parity,
        "ref_ms": round(ref, 6),
        "cand_ms": round(cnd, 6),
        "isolated_speedup": round(speedup, 6),
        "weighted_cases": [{"sig": x["sig"], "ref_ms": x["ref_ms"], "cand_ms": x["cand_ms"]}
                           for x in cases],
        "ref_ops": [],
        "outside_work_removed": [],
        "engaged": engaged,
        "tp": args.tp,
        "integration_checks": {"serving_graph_capture": "pass" if engaged else "not_engaged"},
        "serving_unit": {"base_dir": base.get("side_dir") or os.path.dirname(base["server_log"]),
                         "cand_dir": cand.get("side_dir") or os.path.dirname(cand["server_log"]),
                         "cases": cases, "engaged_banner_lines": engaged_n, "gsm8k": acc,
                         "timing_basis": "non-EP-MoE kernel device time per decode step",
                         "problems": problems},
        "notes": ("serving-side unit test on the live stack: %s; non-EP step time %.3f -> %.3f ms "
                  "summed over %s; gsm8k %s; banner %s x%d"
                  % (parity, ref, cnd, args.point_ids,
                     ("%.3f -> %.3f (z=%.2f)" % (acc["base"], acc["cand"], acc["z"])) if acc else "-",
                     banner, engaged_n)),
    }
    if parity == "fail":
        doc["parity_diagnosis"] = ("gsm8k dropped significantly on the live stack with the overlay "
                                   "engaged; the fused path computes something else")
    return doc


def _from_workload(args):
    """Fill --points / --point-ids / --dp-size from the observed workload when not given."""
    if not args.observed_workload:
        if not args.points:
            raise SystemExit("--points or --observed-workload is required")
        args.point_ids = args.point_ids or "de_5,de_2,de_1"
        args.dp_size = args.dp_size or 1
        return
    wl = _load(args.observed_workload)
    args.points = args.points or (wl.get("serving_weight_model") or {}).get("sampling_points") or \
        os.path.join(os.path.dirname(os.path.abspath(args.observed_workload)), "sampling_points.json")
    regimes = [r.strip() for r in str(getattr(args, "phases", "") or "decode").split(",") if r.strip()]
    if not args.point_ids:
        cases = sorted((c for c in wl.get("cases") or [] if c.get("regime") in regimes),
                       key=lambda c: -float(c.get("effective_calls") or 0))
        args.point_ids = ",".join(c["sig"] for c in cases)
    args.dp_size = args.dp_size or int(wl.get("dp_attention_size") or 1)
    if not args.point_ids:
        raise SystemExit("the observed workload has no %s sampling point" % "/".join(regimes))


def batch(args):
    _from_workload(args)
    cands = []
    for spec in args.cand:
        parts = spec.split("::")
        if len(parts) != 4:
            raise SystemExit("--cand wants CID::LOADER_DIR::BANNER::KERNEL_SOURCE, got %r" % spec)
        cands.append(parts)
    result_path = os.path.join(args.root, "result.json")
    if os.path.exists(result_path):
        os.remove(result_path)
    base = measure_side(os.path.join(args.root, "base"), "", args)
    out = {"ok": True, "base_ok": bool(base.get("ok")), "verdicts": {}}
    if not base.get("ok"):
        out.update(ok=False, reason="base side failed: see %s" % os.path.join(args.root, "base"))
        _dump(result_path, out)
        print(json.dumps(out, separators=(",", ":")))
        return 1
    for cid, loader, banner, kernel_source in cands:
        cand = measure_side(os.path.join(args.root, cid), loader, args)
        doc = verdict_for(cid, banner, kernel_source, base, cand, args)
        path = os.path.join(args.verdict_dir, "%s__serving.json" % cid)
        _dump(path, doc)
        out["verdicts"][cid] = {"verdict_path": path, "parity": doc["parity"],
                                "speedup": doc["isolated_speedup"], "engaged": doc["engaged"]}
    _dump(result_path, out)
    print(json.dumps(out, separators=(",", ":")))
    return 0


def step_gate(base, cand, workload_path, point_ids):
    """Predicted e2e GPU-time gain of the cand stack over base from per-step device time.

    Each sampling point stands for `effective_calls` forwards of the measured window, so
    sum(calls x step time) is that window's GPU time. The fusions never touch the EP MoE
    (fusion_screen rejects crosses_ep_moe), whose time EPLB moves between sessions, so the
    saving counts only kernels outside it while the denominator is the whole step:
        gain = sum_p calls_p (base_nonEP_p - cand_nonEP_p) / sum_p calls_p (base_nonEP_p + base_EP_p)
    """
    calls = {c["sig"]: float(c.get("effective_calls") or 0)
             for c in (_load(workload_path).get("cases") or [])}
    rows, saved, total, problems = [], 0.0, 0.0, []
    for pid in point_ids.split(","):
        b, c = base["points"].get(pid) or {}, cand["points"].get(pid) or {}
        if not (b.get("ok") and c.get("ok")):
            problems.append("%s not measured on %s" % (pid, "base" if not b.get("ok") else "cand"))
            continue
        n = calls.get(pid, 0.0)
        delta = b["non_ep_us"] - c["non_ep_us"]
        saved += n * delta
        total += n * (b["non_ep_us"] + b["ep_us"])
        rows.append({"sig": pid, "calls": n, "base_non_ep_us": round(b["non_ep_us"], 1),
                     "cand_non_ep_us": round(c["non_ep_us"], 1), "base_ep_us": round(b["ep_us"], 1),
                     "step_saving_pct": round(100.0 * delta / (b["non_ep_us"] + b["ep_us"]), 3)})
    gain = 100.0 * saved / total if total > 0 and not problems else None
    return {"ok": gain is not None, "predicted_e2e_gpu_time_gain_pct": None if gain is None else round(gain, 3),
            "points": rows, "problems": problems,
            "basis": "per-step device time at every sampling point (non-EP-MoE saving / whole step), "
                     "weighted by the observed workload's effective_calls",
            "gsm8k": {"base": base.get("gsm8k"), "cand": cand.get("gsm8k")}}


def stepgate(args):
    args.phases = args.phases or "prefill,decode"
    _from_workload(args)
    def side(name, overlay):
        # A side whose server hung or died is measured once more on a fresh server: a
        # measurement failure must not read as "the fusion has no gain".
        path = os.path.join(args.root, name)
        doc = measure_side(path, overlay, args)
        if not doc.get("ok") and os.path.isdir(path):
            os.replace(path, path + ".failed1")
            doc = measure_side(path, overlay, args)
        return doc
    base = side("base", args.base_overlay)
    cand = side("cand", args.cand_overlay)
    out = step_gate(base, cand, args.observed_workload, args.point_ids)
    out.update(base_ok=bool(base.get("ok")), cand_ok=bool(cand.get("ok")), point_ids=args.point_ids)
    _dump(args.out, out)
    print(json.dumps(out, separators=(",", ":")))
    return 0 if out["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("batch", "session-body", "stepgate"):
        p = sub.add_parser(name)
        p.add_argument("--points", default="", help="sampling_points.json")
        p.add_argument("--point-ids", default="")
        p.add_argument("--dp-size", type=int, default=0)
        p.add_argument("--gsm8k-limit", type=int, default=200)
        p.add_argument("--gsm8k-concurrency", type=int, default=64)
        p.add_argument("--point-timeout-s", type=int, default=900,
                       help="kill a probe (or gsm8k) that runs longer than this")
        if name in ("batch", "stepgate"):
            p.add_argument("--phases", default="",
                           help="regimes whose sampling points are measured (batch: decode; "
                                "stepgate: prefill,decode)")
        if name == "stepgate":
            p.add_argument("--root", required=True)
            p.add_argument("--observed-workload", required=True)
            p.add_argument("--base-overlay", default="")
            p.add_argument("--cand-overlay", required=True)
            p.add_argument("--out", required=True)
        elif name == "batch":
            p.add_argument("--root", required=True, help="measurement root (base/ is cached)")
            p.add_argument("--observed-workload", default="")
            p.add_argument("--verdict-dir", required=True)
            p.add_argument("--tp", type=int, default=1)
            p.add_argument("--max-z", type=float, default=2.0)
            p.add_argument("--cand", action="append", default=[],
                           help="CID::LOADER_DIR::BANNER::KERNEL_SOURCE")
        else:
            p.add_argument("--model-name", default="")
    args = ap.parse_args(argv)
    if args.cmd == "session-body":
        return session_body(args)
    if args.cmd == "stepgate":
        return stepgate(args)
    return batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
