#!/usr/bin/env python3
"""Seed this round's VERDICT_DIR with the best PASSING unit-side verdicts of earlier rounds.

A unit-side verdict is an agent-authored kernel plus its measurement, and authoring is not
repeatable: on DeepSeek-V4 the same rung (e01.r4, shared experts + router/topk) measured
1.43-1.50x in one round and 0.97x in a later one. Re-running KernelFusion on the same
deployment and the same Top-K therefore loses wins at random. This copies, for every
candidate id and side, the earlier verdict with the highest speedup among those with parity
pass and speedup > 1 (its kernel stays where it was authored; the verdict points there).
Candidates with no such verdict are left for this round's validators.

    fusion_reuse_verdicts.py --to VERDICT_DIR --from EVAL_DIR [--from EVAL_DIR ...]

Prints ONE JSON line: {"ok": true, "reused": {cid: {"side", "speedup", "from"}}}.
Only earlier rounds of the SAME Top-K should be passed (same candidate ids for the same
windows); the workflow passes args.fusion_reuse_unitside_dirs for that.
"""
import argparse
import glob
import json
import os
import shutil


def _speedup(doc):
    for key in ("weighted_speedup", "observed_weighted_speedup", "isolated_speedup"):
        try:
            value = float(doc.get(key))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def candidates(eval_dirs):
    """{(cid, side): (speedup, path)} for passing verdicts across `eval_dirs`."""
    best = {}
    for eval_dir in eval_dirs:
        for path in sorted(glob.glob(os.path.join(eval_dir, "verdict", "**", "*__*.json"),
                                     recursive=True)):
            name = os.path.basename(path)[:-len(".json")]
            cid, _, side = name.partition("__")
            try:
                with open(path) as fh:
                    doc = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(doc, dict) or str(doc.get("parity", "")).lower() != "pass":
                continue
            cid = str(doc.get("candidate_id") or cid)
            if "undefined" in cid:
                continue  # a misnamed rung verdict cannot be attributed
            speedup = _speedup(doc)
            if speedup <= 1.0:
                continue
            key = (cid, side)
            if key not in best or speedup > best[key][0]:
                best[key] = (speedup, path)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--to", required=True, help="this round's VERDICT_DIR")
    ap.add_argument("--from", dest="sources", action="append", default=[],
                    help="an earlier eval dir of the same deployment and Top-K")
    args = ap.parse_args(argv)
    os.makedirs(args.to, exist_ok=True)
    reused = {}
    for (cid, side), (speedup, path) in sorted(candidates(args.sources).items()):
        target = os.path.join(args.to, "%s__%s.json" % (cid, side))
        if os.path.exists(target):
            continue  # this round already has its own verdict for it
        with open(path) as fh:
            doc = json.load(fh)
        doc["reused_from"] = os.path.abspath(path)
        with open(target, "w") as fh:
            json.dump(doc, fh, indent=1)
        reused[cid] = {"side": side, "speedup": round(speedup, 4), "from": os.path.abspath(path)}
    print(json.dumps({"ok": True, "reused": reused}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
