#!/usr/bin/env python3
"""Execute ONE sampling point: arm a bounded profile window on the live service,
drive the intended shape through it, and verify the trace contains what was asked for.

KernelFusion's copy of the kernel-extract skill's serving_probe.py
(knowledge/kernel_extraction_skills/kernel-extract/scripts/serving_probe.py). The arming
rules below are the skill's and are unchanged; KernelFusion adds what its analysis needs:
  * --record-shapes / --with-stack: Input Dims and module spans for the semantic tables;
  * --dp-size: under DP attention a point is ONE rank's batch, so every rank gets it;
  * --vocab-size: random prompt ids stay inside the served vocabulary (an id past it
    indexes past the embedding table and takes the server down with an HSA fault);
  * the verified trace's per-step device time (`step_cost`), which turns the point's
    represented instances into GPU-time mass when the analysis trace is chosen.

Step three of log -> sampling points -> capture. Reads a point from
`plan_sampling_points.py` output and produces a trace plus a journal recording what
actually happened, which `coverage_ledger.py` consumes.

This is a DIAGNOSTIC probe. Its synthetic requests are chosen to hit an observed shape;
they do not replace the measured replay's weights. Per-kernel costs measured here are
combined with the replay's observed instance counts to get whole-workload mass.

Four rules, each of which was learned by a window failing
---------------------------------------------------------
1. REACH STEADY STATE BEFORE ARMING, AND CONFIRM IT.
   A window armed as load starts captures the ramp: batch 1, no steady-state expert or
   attention mix. "I sent N requests" is not confirmation -- the scheduler decides what
   is resident. This probe scrapes the server's own occupancy counter and refuses to arm
   until it reads the intended batch with non-zero generation throughput.

2. ARM ONCE. NEVER profile_by_stage.
   With a shape-discovery hook installed (TraceLens patches torch.profiler.profile.start),
   arming is SYNCHRONOUS and costs minutes -- 110-134 s measured on an 8-rank MXFP4
   deployment. profile_by_stage arms twice, so a two-stage window needs ~4 minutes of
   arming before any budget is spent on steps. A window budgeted at 150 s wall-clock
   that arms twice captures zero steps and reports no error. Budget the arming cost
   explicitly (--arm-timeout) and let the step counter close the window.

3. CLOSE THE WINDOW EXPLICITLY, INCLUDING ON EXCEPTION.
   An unclosed window rolls into the next request and silently attributes someone
   else's work to this point. /stop_profile runs in a finally block.

4. VERIFY THE TRACE, DO NOT ASSUME IT.
   A window can close successfully and contain the wrong phase, or no device time at
   all. (A graph-capture trace, for instance, records dispatch structure and shapes but
   essentially no kernel events -- it cannot carry a census.) After the window closes,
   this probe parses the trace and asserts the intended phase and shape are actually in
   it, and reports what it found either way.

Stdlib only.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent))


def post(base, path, obj=None, timeout=1800):
    data = json.dumps(obj if obj is not None else {}).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return fh.read().decode()


def scrape_metrics(base, timeout=30, per_rank=False):
    """(num_running_reqs, gen_throughput) from the server's own counters.

    The scheduler, not the client, decides what is resident. This is the authoritative
    occupancy signal; a request count sent from outside is not.

    per_rank=True (DP attention): every rank exports its own series, and the batch a
    rank's kernels run is that rank's count. Returns the list of per-series running
    counts and the summed throughput instead of the last series seen.
    """
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=timeout) as fh:
            text = fh.read().decode()
    except Exception as exc:
        return None, None, str(exc)
    out, series = {}, {"num_running_reqs": [], "gen_throughput": []}
    for line in text.splitlines():
        for key in ("sglang:num_running_reqs", "sglang:gen_throughput",
                    "num_running_reqs", "gen_throughput"):
            if line.startswith(key + "{") or line.startswith(key + " "):
                try:
                    value = float(line.rsplit(" ", 1)[1])
                except ValueError:
                    continue
                out[key.split(":")[-1]] = value
                series[key.split(":")[-1]].append(value)
    if per_rank:
        if not series["num_running_reqs"]:
            return None, None, None
        return series["num_running_reqs"], sum(series["gen_throughput"]), None
    return out.get("num_running_reqs"), out.get("gen_throughput"), None


def model_vocab_size(base, timeout=30):
    """vocab_size of the served model, from its config.json (path via /get_model_info)."""
    try:
        with urllib.request.urlopen(base + "/get_model_info", timeout=timeout) as fh:
            model_path = json.loads(fh.read().decode()).get("model_path") or ""
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        return int(cfg.get("vocab_size") or (cfg.get("text_config") or {}).get("vocab_size") or 0) or None
    except Exception:
        return None


def rand_ids(n, seed, vocab_size):
    # An id at or past vocab_size indexes past the embedding table: the GPU faults
    # (HSA hardware exception) and takes the whole server down.
    rnd = random.Random(seed)
    hi = min(150000, int(vocab_size) - 1)
    return [rnd.randint(1000, hi) for _ in range(n)]


def generate(base, ids, max_new_tokens, journal, label, timeout=1800, dp_rank=None):
    body = {"input_ids": ids,
            "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0,
                                "ignore_eos": True}}
    if dp_rank is not None:
        # DP attention: pin the request to one rank. The server's own balancer spread a probe
        # batch 1..14 per rank on DeepSeek-V4-Pro, so "every rank holds bs" never happened.
        body["routed_dp_rank"] = int(dp_rank)
    t0 = time.time()
    try:
        txt = post(base, "/generate", body, timeout=timeout)
        rec = {"event": "request_done", "label": label, "dp_rank": dp_rank,
               "prompt_tokens": len(ids), "max_new_tokens": max_new_tokens,
               "wall_s": round(time.time() - t0, 2)}
        # What the server actually did: a request that "finished" in seconds was aborted or
        # stopped early, and that must be visible next to an occupancy that never built up.
        try:
            meta = (json.loads(txt) or {}).get("meta_info") or {}
            rec.update({k: meta.get(k) for k in ("finish_reason", "prompt_tokens",
                                                 "completion_tokens", "cached_tokens")
                        if meta.get(k) is not None})
            if "prompt_tokens" in meta:
                rec["prompt_tokens"] = meta["prompt_tokens"]
                rec["sent_tokens"] = len(ids)
        except (ValueError, AttributeError):
            pass
        journal.append(rec)
    except Exception as exc:
        journal.append({"event": "request_failed", "label": label, "error": str(exc),
                        "wall_s": round(time.time() - t0, 2)})


def wait_for_occupancy(base, want_bs, journal, timeout, poll=2.0, tolerance=0, dp_size=1):
    """Block until the scheduler reports the intended batch, actively generating.

    With dp_size > 1 the intended batch is PER RANK and every rank must hold it.
    Returns (ok, observed_bs, observed_tps). Never arms on a guess.
    """
    t0 = time.time()
    last = (None, None)
    while time.time() - t0 < timeout:
        bs, tps, err = scrape_metrics(base, per_rank=dp_size > 1)
        last = (bs, tps)
        if err is None and bs is not None:
            at_target = (all(abs(b - want_bs) <= tolerance for b in bs) if dp_size > 1
                         else abs(bs - want_bs) <= tolerance)
            if at_target and (tps or 0) > 0:
                journal.append({"event": "steady_state_confirmed", "want_bs": want_bs,
                                "observed_bs": bs, "gen_throughput": tps,
                                "waited_s": round(time.time() - t0, 1)})
                return True, bs, tps
        time.sleep(poll)
    journal.append({"event": "steady_state_NOT_confirmed", "want_bs": want_bs,
                    "observed_bs": last[0], "gen_throughput": last[1],
                    "waited_s": round(time.time() - t0, 1),
                    "consequence": "window NOT armed; arming here would capture the ramp "
                                   "or an unrelated batch and label it as this point"})
    return False, last[0], last[1]


def start_profile(base, out_dir, num_steps, journal, timeout, record_shapes=False,
                  with_stack=False):
    """Arm once. profile_by_stage is deliberately not offered -- see rule 2.

    record_shapes/with_stack are opt-in: a consumer that maps kernels back to
    layers and operator shapes (KernelFusion's semantic tables) needs Input Dims
    and module spans; a census does not, and both make the trace larger."""
    body = {"output_dir": out_dir, "num_steps": int(num_steps),
            "activities": ["CPU", "GPU"], "profile_by_stage": False}
    if record_shapes:
        body["record_shapes"] = True
    if with_stack:
        body["with_stack"] = True
    t0 = time.time()
    txt = post(base, "/start_profile", body, timeout=timeout)
    arm_s = round(time.time() - t0, 1)
    journal.append({"event": "armed", "arm_cost_s": arm_s, "num_steps": int(num_steps),
                    "profile_by_stage": False, "record_shapes": bool(record_shapes),
                    "with_stack": bool(with_stack), "response": txt[:200]})
    if arm_s > 60:
        journal.append({"event": "note",
                        "text": "arming took %.0f s -- a shape-discovery hook is making "
                                "profiler start synchronous. Budget for this; it is not "
                                "a hang." % arm_s})
    return arm_s


def stop_profile(base, journal, timeout=1800):
    try:
        txt = post(base, "/stop_profile", {}, timeout=timeout)
        journal.append({"event": "stopped", "response": txt[:200]})
        return True
    except Exception as exc:
        journal.append({"event": "stop_FAILED", "error": str(exc),
                        "consequence": "the window may still be open and will attribute "
                                       "subsequent work to this point. Stop it by hand."})
        return False


def traces_written(trace_dir, since_ts):
    out = []
    for p in glob.glob(os.path.join(trace_dir, "**", "*"), recursive=True):
        if os.path.isfile(p) and os.path.getmtime(p) >= since_ts:
            out.append({"path": p, "bytes": os.path.getsize(p)})
    return sorted(out, key=lambda d: -d["bytes"])


def _classify(name):
    """(PREFILL|DECODE, shape) for a forward step span, else None.

    SGLang speculative decoding names the target model's generation step TARGET_VERIFY
    (VERIFY on some builds); KernelFusion's sglang_step_modes knows every mode, and its
    draft-model steps are not a target forward, so they are skipped. Other dialects fall
    back to parse_profile's registry."""
    try:
        from sglang_step_modes import parse_step
    except Exception:
        parse_step = None
    step = parse_step(name) if parse_step else None
    if step is not None:
        if step["phase"] == "prefill":
            return "PREFILL", step["tokens"]
        if step["phase"] in ("decode", "verify"):
            return "DECODE", step["batch_size"]
        return None
    from parse_profile import _classify_step
    c = _classify_step(name)
    if c is None:
        return None
    return ("PREFILL", c[1]) if c[0] else ("DECODE", c[2])


def verify_trace(paths, want_phase, want_shape_key, want_shape, journal):
    """Rule 4: assert the trace contains the phase and shape that were asked for."""
    try:
        from parse_profile import _step_span_candidates
    except Exception as exc:
        journal.append({"event": "verify_SKIPPED", "error": str(exc)})
        return {"verified": None, "reason": "parse_profile unavailable: %s" % exc}

    import gzip
    found, kernels, scanned = {}, 0, 0
    per_file_cost = []
    for rec in paths[:4]:
        p = rec["path"]
        if not (p.endswith(".json") or p.endswith(".json.gz") or p.endswith(".pt.trace.json")
                or p.endswith(".pt.trace.json.gz")):
            continue
        try:
            op = gzip.open if p.endswith(".gz") else open
            with op(p, "rt", errors="replace") as fh:
                data = json.load(fh)
        except Exception:
            continue
        scanned += 1
        events = data.get("traceEvents", data if isinstance(data, list) else [])
        kern = [e for e in events
                if isinstance(e, dict) and e.get("cat") in ("kernel", "Kernel")]
        kernels += len(kern)
        spans = []
        for e, nm in _step_span_candidates(events):
            c = _classify(nm)
            if c is None:
                continue
            phase, shape = c
            found.setdefault(phase, {}).setdefault(shape, 0)
            found[phase][shape] += 1
            if phase == want_phase:
                spans.append(float(e.get("dur") or 0))
        if spans:
            # The window is phase-pure by construction (the probe drives one phase), so the
            # rank's device time over its steps of that phase is the per-forward cost.
            # Step-span wall time is kept beside it for a trace whose graph replays carry
            # no per-kernel events.
            spans.sort()
            per_file_cost.append({
                "file": os.path.basename(p), "steps": len(spans),
                "kernel_us_per_step": sum(float(k.get("dur") or 0) for k in kern) / len(spans),
                "span_us_median": spans[len(spans) // 2],
            })

    step_cost = None
    if per_file_cost:
        def _median(vals):
            vals = sorted(vals)
            return vals[len(vals) // 2]
        kus = _median([c["kernel_us_per_step"] for c in per_file_cost])
        sus = _median([c["span_us_median"] for c in per_file_cost])
        step_cost = {"phase": want_phase, "per_file": per_file_cost,
                     "kernel_us_per_step": round(kus, 3), "span_us_per_step": round(sus, 3),
                     "us_per_step": round(kus if kus > 0 else sus, 3),
                     "basis": "kernel_device_time" if kus > 0 else "step_span_wall_time"}
    res = {"files_scanned": scanned, "kernel_events": kernels, "step_cost": step_cost,
           "step_spans_by_phase_shape": {k: dict(sorted(v.items())) for k, v in found.items()},
           "want": {"phase": want_phase, want_shape_key: want_shape}}
    got = found.get(want_phase, {})
    res["phase_present"] = bool(got)
    res["exact_shape_present"] = want_shape in got
    if kernels == 0 and scanned:
        res["verified"] = False
        res["reason"] = ("trace has step spans but ~no kernel events -- it records "
                         "dispatch structure only and cannot carry a census. A "
                         "graph-capture trace looks exactly like this.")
    elif not got:
        res["verified"] = False
        res["reason"] = "no %s step spans in the trace; the window captured another phase" % want_phase
    elif not res["exact_shape_present"]:
        res["verified"] = "partial"
        res["reason"] = ("%s captured, but not at %s=%s. Present: %s. The measurement is "
                         "real; it just stands for a different shape than planned, and "
                         "the ledger must say so."
                         % (want_phase, want_shape_key, want_shape, sorted(got)[:12]))
    else:
        res["verified"] = True
        res["reason"] = "%s at %s=%s present, %d kernel events" % (
            want_phase, want_shape_key, want_shape, kernels)
    journal.append({"event": "verify", **res})
    return res


# --------------------------------------------------------------------------- #
def prefill_prefix(point, args):
    """The point's cached context, as a shared prefix (empty when the point has none)."""
    cached = int(point.get("cached_tokens") or 0)
    if not args.shared_prefix or cached <= 0:
        return []
    return rand_ids(cached, args.seed + 98, args.vocab_size)


def drive_prefill(base, point, args, journal, prefix=()):
    """One sequence per rank: the point's cached prefix (already in the cache) plus exactly
    `new_tokens` uncached tokens, so the extend step is the planned shape.

    The point is ONE batch of the log: `new_tokens` is that batch's uncached work and
    `cached_tokens` its prefix hits. `running` is the decode population resident beside it,
    not a count of prefill sequences; sending `running` sequences of `new_tokens` each made
    the scheduler pack them into full chunks (pr_3333 came back as 8192-token steps).
    """
    toks = int(point["new_tokens"])
    dp = max(1, int(getattr(args, "dp_size", 1) or 1))
    prefix = list(prefix)
    # max_new_tokens=1 keeps the request in prefill: we are measuring the context
    # forward, and a long generation would mix decode into the window.
    with ThreadPoolExecutor(max_workers=dp) as ex:
        for i in range(dp):
            ex.submit(generate, base, prefix + rand_ids(toks, args.seed + i, args.vocab_size), 1,
                      journal, "prefill_seq%d" % i, args.request_timeout, i if dp > 1 else None)


def prepare_cached_prefix(base, prefix, args, journal):
    """Put the shared context prefix in the prefix cache on every DP rank before the batch.

    The skill's decode recipe: "prepare target context and cache state". An agentic replay
    serves most of each request's context from the prefix cache; a probe that prefills
    bs x dp distinct ~100k-token contexts from scratch spends tens of minutes in prefill
    (DeepSeek-V4-Pro, 8 ranks: 12 x 8 x 103k tokens never reached steady state in 300 s).
    Each rank's cache is its own, so the prefix goes through every rank (`dp` requests,
    balanced by the server like the batch that follows)."""
    n = max(1, int(getattr(args, "dp_size", 1) or 1))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        for i in range(n):
            ex.submit(generate, base, prefix, 1, journal, "prefix_warm%d" % i, args.request_timeout,
                      i if n > 1 else None)
    journal.append({"event": "prefix_cached", "prefix_tokens": len(prefix), "requests": n,
                    "wall_s": round(time.time() - t0, 1)})


def drive_decode(base, point, args, journal):
    """Hold `bs` requests resident and generating, then arm. Returns the futures so the
    caller can keep them alive across the window.

    Each request is the shared cached prefix plus its own short suffix: the resident
    context per request is the point's context (decode attention reads all of it), while
    the prefill to get there is one prefix per rank plus the suffixes."""
    bs = int(point["batch_size"]) * max(1, int(getattr(args, "dp_size", 1) or 1))
    ctx = int(point.get("context_tokens") or 1024)
    suffix = max(1, min(int(args.unique_suffix_tokens), ctx))
    prefix = []
    if args.shared_prefix and ctx > suffix:
        prefix = rand_ids(ctx - suffix, args.seed + 99, args.vocab_size)
        prepare_cached_prefix(base, prefix, args, journal)
    else:
        suffix = ctx
    ex = ThreadPoolExecutor(max_workers=bs)
    dp = max(1, int(getattr(args, "dp_size", 1) or 1))
    for i in range(bs):
        ex.submit(generate, base, prefix + rand_ids(suffix, args.seed + 100 + i, args.vocab_size),
                  args.decode_tokens, journal, "decode_req%d" % i, args.request_timeout,
                  (i % dp) if dp > 1 else None)
    return ex


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--points", required=True, help="plan_sampling_points.py output")
    ap.add_argument("--point-id", required=True)
    ap.add_argument("--base-url", default=os.environ.get("PROBE_BASE_URL",
                                                         "http://127.0.0.1:30000"))
    ap.add_argument("--trace-dir", required=True,
                    help="host-visible directory the server writes traces to "
                         "(SGLANG_TORCH_PROFILER_DIR)")
    ap.add_argument("--server-trace-dir", default=None,
                    help="same directory as the SERVER sees it, if it differs (container)")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="forwards to capture; the scheduler closes the window on this")
    ap.add_argument("--arm-timeout", type=float, default=600,
                    help="budget for the synchronous arm; 110-134 s is normal with a "
                         "shape hook installed")
    ap.add_argument("--steady-timeout", type=float, default=300)
    ap.add_argument("--window-timeout", type=float, default=900)
    ap.add_argument("--decode-tokens", type=int, default=4096,
                    help="decode: tokens per request; with speculation (accept ~6) 512 tokens "
                         "lived ~5 s, too short to confirm occupancy and arm")
    ap.add_argument("--request-timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--record-shapes", action="store_true",
                    help="ask the profiler for operator Input Dims (off by default)")
    ap.add_argument("--with-stack", action="store_true",
                    help="ask the profiler for Python/module stacks (off by default)")
    ap.add_argument("--dp-size", type=int, default=1,
                    help="DP-attention ranks. Points from observed_workload.py --rank R are "
                         "ONE rank's batches: drive dp-size times the requests and require "
                         "every rank to hold the point's batch before arming")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="served model's vocab_size; default: read from the server's model "
                         "config.json, else 32000. Random prompt ids stay below it")
    ap.add_argument("--shared-prefix", type=int, default=1,
                    help="decode: build each context as a shared, pre-cached prefix plus a "
                         "unique suffix (1, default) instead of a fully distinct prompt (0)")
    ap.add_argument("--unique-suffix-tokens", type=int, default=512,
                    help="decode: per-request unique tokens after the shared prefix")
    ap.add_argument("--occupancy-tolerance", type=int, default=1,
                    help="decode: accepted |running - batch_size| per rank before arming")
    ap.add_argument("--out", required=True, help="journal JSON for this point")
    args = ap.parse_args()
    args.vocab_size = args.vocab_size or model_vocab_size(args.base_url) or 32000

    plan = json.load(open(args.points))
    pts = [p for p in plan["points"] if p["point_id"] == args.point_id]
    if not pts:
        raise SystemExit("serving_probe: no point %r in %s. Have: %s"
                         % (args.point_id, args.points,
                            ", ".join(p["point_id"] for p in plan["points"])))
    point = pts[0]
    phase = point["phase"]

    journal = [{"event": "begin", "point": point, "base_url": args.base_url,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}]
    t_start = time.time()
    server_dir = args.server_trace_dir or args.trace_dir
    num_steps = args.num_steps or (32 if phase == "DECODE" else 8)
    ok = False
    verify = {}
    ex = None

    try:
        if phase == "DECODE":
            # Steady state FIRST (rule 1): put the batch on, confirm the scheduler has
            # admitted it and is generating, and only then pay for the arm.
            ex = drive_decode(args.base_url, point, args, journal)
            want = int(point["batch_size"])
            confirmed, obs_bs, _ = wait_for_occupancy(
                args.base_url, want, journal, args.steady_timeout,
                tolerance=args.occupancy_tolerance, dp_size=args.dp_size)
            if not confirmed:
                raise RuntimeError(
                    "steady state not reached: wanted bs=%s, scheduler reported %s. Not "
                    "arming." % (want, obs_bs))
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout,
                          record_shapes=args.record_shapes, with_stack=args.with_stack)
            # The scheduler closes the window itself once num_steps forwards have run;
            # the batch is already resident, so the steps are the intended shape.
            deadline = time.time() + args.window_timeout
            while time.time() < deadline:
                time.sleep(5)
                bs, tps, _ = scrape_metrics(args.base_url, per_rank=args.dp_size > 1)
                if bs == 0 or (isinstance(bs, list) and not any(bs)):
                    journal.append({"event": "batch_drained", "note": "requests finished "
                                    "before the window closed; shorten --decode-tokens "
                                    "or lower --num-steps"})
                    break
        else:
            # Prefill is a single forward per request: put the cached prefix in place,
            # arm, then send, because there is no resident state to confirm. The window
            # is still bounded by num_steps and still closed explicitly.
            prefix = prefill_prefix(point, args)
            if prefix:
                prepare_cached_prefix(args.base_url, prefix, args, journal)
            start_profile(args.base_url, server_dir, num_steps, journal, args.arm_timeout,
                          record_shapes=args.record_shapes, with_stack=args.with_stack)
            drive_prefill(args.base_url, point, args, journal, prefix)
        ok = True
    except Exception as exc:
        journal.append({"event": "probe_failed", "error": "%s: %s"
                        % (type(exc).__name__, exc)})
    finally:
        # Rule 3: always, even on the failure path.
        stop_profile(args.base_url, journal)
        if ex is not None:
            ex.shutdown(wait=False)

    written = traces_written(args.trace_dir, t_start)
    journal.append({"event": "traces_written", "count": len(written),
                    "total_bytes": sum(w["bytes"] for w in written),
                    "files": written[:12]})
    if written:
        verify = verify_trace(
            written, phase,
            "new_tokens" if phase == "PREFILL" else "batch_size",
            point.get("new_tokens") if phase == "PREFILL" else point.get("batch_size"),
            journal)
    else:
        journal.append({"event": "verify", "verified": False,
                        "reason": "no trace files appeared under %s" % args.trace_dir})
        verify = {"verified": False}

    result = {
        "point_id": args.point_id,
        "point": point,
        "status": "ok" if (ok and verify.get("verified") is True) else
                  "partial" if (ok and verify.get("verified") == "partial") else "failed",
        "trace_verified": verify.get("verified"),
        "verify_reason": verify.get("reason"),
        "step_cost": verify.get("step_cost"),
        "wall_s": round(time.time() - t_start, 1),
        "traces": written[:12],
        "journal": journal,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print("%s: %s (%s) in %.0f s -> %s"
          % (args.point_id, result["status"], result["verify_reason"],
             result["wall_s"], args.out))
    # A failed point is evidence, not an error to hide; preserve the journal and let the
    # caller decide whether to spend its one recapture.
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
