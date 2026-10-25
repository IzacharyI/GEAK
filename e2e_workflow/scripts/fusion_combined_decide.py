#!/usr/bin/env python3
"""Decide the combined apply-back A/B: one base server vs one server with every fusion stacked.

Combined apply-back measures two servers, each with the same lifecycle (bench_e2e.sh
warm_server, REPEATS timed rounds, then gsm8k on the same live server):

  base  = the current accepted stack (CURRENT_OVERLAY/FLAGS/ENV)
  cand  = base + every authored fusion overlay, through one combined loader

  throughput  accept iff min(cand) > max(base) AND delta > noise band
              delta = (median(cand) / median(base) - 1) * 100
  accuracy    reject iff the gsm8k drop is significant: two-proportion z >= --max-z
  engagement  every --expect-banner tag must print "[overlay-<tag>] ENGAGED" on all
              TP ranks of the cand server log; a tag that does not is reported in
              not_engaged (it ran nothing, so it is blocked, not credited)

decision = accept when throughput and accuracy pass, else reject. Engagement does not
change the decision; it only decides which fusions the accept covers.

With --step-gate (fusion_serving_unit.py stepgate) the throughput test changes, because one
AgentX leg is too noisy to judge a fusion: on DeepSeek-V4 five identical no-overlay legs gave
512-602 tok/s, and an A/A pair "accepted" +7.3%. Then
  step        the stepgate's predicted e2e GPU-time gain (per-step device time at every
              sampling point, weighted by the observed workload) must exceed --step-min-gain-pct
  direction   the interleaved e2e legs (several --base-summary / --cand-summary) must not put
              the cand median below the base median by more than --e2e-tolerance-pct
and the reported gain is the step-time one; the e2e medians are evidence of direction only.

Prints one JSON object and a final line ``COMBINED_DECISION=<accept|reject>``.
"""
import argparse
import json
import math
import re
import statistics
import sys


def load_throughputs(summary_path):
    """Per-round tok/s from a bench_e2e.sh bench_summary.json."""
    with open(summary_path) as fh:
        doc = json.load(fh)
    values = doc.get("all_throughput") or []
    return [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]


def load_gsm8k(path):
    """(exact_match, n) from a gsm8k_eval.py --out file, or None."""
    if not path:
        return None
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    summary = doc.get("summary", doc)
    try:
        return float(summary["exact_match"]), int(summary["n"])
    except (KeyError, TypeError, ValueError):
        return None


def engaged_counts(server_log, tags):
    """Number of "[overlay-<tag>] ENGAGED" lines per tag in the server log."""
    counts = {tag: 0 for tag in tags}
    if not tags:
        return counts
    pattern = re.compile(r"\[overlay-([^\]]+)\] ENGAGED")
    try:
        with open(server_log, errors="replace") as fh:
            for line in fh:
                for tag in pattern.findall(line):
                    if tag in counts:
                        counts[tag] += 1
    except OSError:
        pass
    return counts


def accuracy_z(base, cand):
    """Two-proportion z of the base->cand drop (positive = cand is worse)."""
    (p1, n1), (p2, n2) = base, cand
    pooled = (p1 * n1 + p2 * n2) / float(n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0 if p2 >= p1 else float("inf")
    return (p1 - p2) / se


def decide(base_tps, cand_tps, base_acc=None, cand_acc=None, noise_band_pct=0.5,
           max_z=2.0, engaged=None, tp=1, step=None, step_min_gain_pct=0.5,
           e2e_tolerance_pct=5.0):
    out = {"noise_band_pct": noise_band_pct, "max_z": max_z,
           "base_tok_s": [round(v, 4) for v in base_tps],
           "cand_tok_s": [round(v, 4) for v in cand_tps]}
    reasons = []
    perf_ok = False
    if not base_tps or not cand_tps:
        reasons.append("a side has no timed round")
    elif any(v <= 0 for v in base_tps + cand_tps):
        reasons.append("a round reported a non-positive throughput")
    else:
        base_med = statistics.median(base_tps)
        cand_med = statistics.median(cand_tps)
        delta = (cand_med / base_med - 1.0) * 100.0
        nonoverlap = min(cand_tps) > max(base_tps)
        out.update(base_median=round(base_med, 4), cand_median=round(cand_med, 4),
                   delta_pct=round(delta, 4), nonoverlap=nonoverlap)
        if step is not None:
            gain = step.get("predicted_e2e_gpu_time_gain_pct") if step.get("ok") else None
            step_ok = gain is not None and gain > step_min_gain_pct
            direction_ok = delta >= -e2e_tolerance_pct
            perf_ok = step_ok and direction_ok
            out.update(gate="step_time", step_gain_pct=gain, step_min_gain_pct=step_min_gain_pct,
                       e2e_tolerance_pct=e2e_tolerance_pct, step_points=step.get("points"))
            if not step_ok:
                reasons.append("step time: predicted gain %s%% <= %s%%%s" % (
                    gain, step_min_gain_pct,
                    " (%s)" % "; ".join(step.get("problems") or []) if step.get("problems") else ""))
            if not direction_ok:
                reasons.append("e2e direction: cand median %.3f%% below base, beyond the %s%% "
                               "tolerance of %d+%d interleaved legs"
                               % (-delta, e2e_tolerance_pct, len(base_tps), len(cand_tps)))
        else:
            perf_ok = nonoverlap and delta > noise_band_pct
            if not perf_ok:
                reasons.append("throughput: nonoverlap=%s, delta %.3f%% vs noise band %s%%"
                               % (nonoverlap, delta, noise_band_pct))

    acc_ok = True
    if base_acc is None or cand_acc is None:
        acc_ok = False
        reasons.append("accuracy: gsm8k missing on %s"
                       % ("both sides" if base_acc is None and cand_acc is None
                          else ("base" if base_acc is None else "cand")))
    else:
        z = accuracy_z(base_acc, cand_acc)
        acc_ok = z < max_z
        out.update(gsm8k_base=base_acc[0], gsm8k_cand=cand_acc[0],
                   gsm8k_n=[base_acc[1], cand_acc[1]], gsm8k_z=round(z, 3))
        if not acc_ok:
            reasons.append("accuracy: gsm8k %.3f -> %.3f, z=%.2f >= %s"
                           % (base_acc[0], cand_acc[0], z, max_z))

    engaged = engaged or {}
    out["engaged"] = engaged
    out["not_engaged"] = sorted(tag for tag, n in engaged.items() if n < max(1, tp))

    out["perf_pass"] = perf_ok
    out["accuracy_pass"] = acc_ok
    out["decision"] = "accept" if perf_ok and acc_ok else "reject"
    if reasons:
        out["reason"] = "; ".join(reasons)
    elif step is not None:
        out["reason"] = ("step time predicts +%.3f%% e2e GPU time; e2e legs %+.3f%% (direction ok); "
                         "gsm8k within noise" % (out["step_gain_pct"], out["delta_pct"]))
    else:
        out["reason"] = "throughput +%.3f%% non-overlapping, gsm8k within noise" % out["delta_pct"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-summary", action="append", required=True,
                    help="base leg bench_summary.json (repeat for interleaved legs)")
    ap.add_argument("--cand-summary", action="append", required=True,
                    help="cand leg bench_summary.json (repeat for interleaved legs)")
    ap.add_argument("--step-gate", default="", help="fusion_serving_unit.py stepgate JSON")
    ap.add_argument("--step-min-gain-pct", type=float, default=0.5)
    ap.add_argument("--e2e-tolerance-pct", type=float, default=5.0)
    ap.add_argument("--base-gsm8k", default="", help="base leg gsm8k_eval.py --out json")
    ap.add_argument("--cand-gsm8k", default="", help="cand leg gsm8k_eval.py --out json")
    ap.add_argument("--cand-server-log", default="", help="cand leg server.log (engagement)")
    ap.add_argument("--expect-banner", action="append", default=[],
                    help="overlay tag expected to print [overlay-<tag>] ENGAGED (repeatable)")
    ap.add_argument("--tp", type=int, default=1, help="ENGAGED lines required per tag")
    ap.add_argument("--noise-band-pct", type=float, default=0.5)
    ap.add_argument("--max-z", type=float, default=2.0)
    ap.add_argument("--out", default="", help="also write the JSON here")
    a = ap.parse_args(argv)
    step = None
    if a.step_gate:
        try:
            with open(a.step_gate) as fh:
                step = json.load(fh)
        except (OSError, ValueError):
            step = {"ok": False, "problems": ["step gate %s unreadable" % a.step_gate]}
    out = decide(sum((load_throughputs(p) for p in a.base_summary), []),
                 sum((load_throughputs(p) for p in a.cand_summary), []),
                 load_gsm8k(a.base_gsm8k), load_gsm8k(a.cand_gsm8k),
                 a.noise_band_pct, a.max_z,
                 engaged_counts(a.cand_server_log, a.expect_banner), a.tp,
                 step, a.step_min_gain_pct, a.e2e_tolerance_pct)
    text = json.dumps(out, sort_keys=True)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    print(f"COMBINED_DECISION={out['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
