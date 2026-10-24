#!/usr/bin/env python3
"""Print the alternative ladders of a fusion_topk.json as ONE compact JSON line.

The KernelFusion workflow has no filesystem, and the screen agent's structured return is an
echo that may drop `alt_ladder` (run 7 on DeepSeek-V4 did, and ladder descent then tested
nothing). The workflow asks a script-runner agent to run this and reads the ladders back,
keeping only what schedules and names a rung's unit test, so the echo stays small.

    fusion_ladders.py <fusion_topk.json> [--exec e01 --exec e02 ...]
"""
import argparse
import json

# Enough to schedule and name a rung's unit test; the validator reads the full rung (row ids,
# member shapes) from FUSION_TOPK_JSON by exec_id + rung.
RUNG_KEYS = ("rung", "kind", "kernel_names", "phases", "entry_id", "expected_e2e_gain_pct",
             "op_tags", "candidate_id")


def ladders(topk, wanted=None):
    out = {}
    for entry in topk.get("execution_list") or []:
        exec_id = entry.get("exec_id")
        if not exec_id or (wanted and exec_id not in wanted):
            continue
        rungs = []
        for item in entry.get("alt_ladder") or []:
            if not isinstance(item, dict):
                continue
            rung = {k: item[k] for k in RUNG_KEYS if item.get(k) is not None}
            # Mangled kernel names run to hundreds of characters; the workflow only shows them
            # and tests them for collectives, and the validator reads the full ones.
            if rung.get("kernel_names"):
                rung["kernel_names"] = [str(n)[:48] for n in rung["kernel_names"]]
            rungs.append(rung)
        out[exec_id] = rungs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("topk")
    ap.add_argument("--exec", action="append", default=[], dest="execs")
    args = ap.parse_args(argv)
    with open(args.topk) as fh:
        topk = json.load(fh)
    print(json.dumps({"ok": True, "ladders": ladders(topk, set(args.execs) or None)},
                     separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
