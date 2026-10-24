#!/usr/bin/env python3
"""Seed this round's VERDICT_DIR with the best PASSING unit-side verdicts of earlier rounds.

A unit-side verdict is an agent-authored kernel plus its measurement, and authoring is not
repeatable: on DeepSeek-V4 the same rung (e01.r4, shared experts + router/topk) measured
1.43-1.50x in one round and 0.97x in a later one. Re-running KernelFusion on the same
deployment and the same Top-K therefore loses wins at random. This copies, for every
candidate id and side, the earlier verdict with the highest speedup among those with parity
pass and speedup > 1 (its kernel stays where it was authored; the verdict points there).
Candidates with no such verdict are left for this round's validators.

    fusion_reuse_verdicts.py --to VERDICT_DIR --from EVAL_DIR [--from EVAL_DIR ...] [--prefer-best]

--prefer-best is the best-of-N step inside one round: the --from dirs are this round's extra
authoring attempts, and an attempt replaces this round's own verdict when it is faster (or the
own verdict is not a parity pass). The replaced verdict is kept next to it as `<name>.json.prev`.

Prints ONE JSON line: {"ok": true, "reused": {cid: {"side", "speedup", "from"}}}.
Candidate ids are board-local: a changed screen renumbers them (c23 of one board is another
window on the next). With --to-topk THIS_ROUND_TOPK every earlier verdict is matched by WINDOW
instead -- the captured kernel instances (semantic-table row_ids) its candidate covered on its own
board (EVAL_DIR/fusion_capture/fusion/fusion_topk.json) -- and lands under the id this round's
board gives that same window; a verdict whose window this board does not have is dropped. Without
--to-topk ids are taken as they are, so only earlier rounds of the SAME Top-K may be passed.
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


def window_ids(topk):
    """{candidate_id: frozenset(row_ids)} of every candidate a Top-K board names."""
    out = {}
    for entry in (topk.get("execution_list") or []) + (topk.get("candidate_pool") or []):
        for member in entry.get("merged_members") or []:
            if member.get("candidate_id") and member.get("row_ids"):
                out[str(member["candidate_id"])] = frozenset(member["row_ids"])
        cids = entry.get("candidate_ids") or []
        if len(cids) == 1 and entry.get("row_ids"):
            out.setdefault(str(cids[0]), frozenset(entry["row_ids"]))
        for rung in entry.get("alt_ladder") or []:
            if not isinstance(rung, dict) or int(rung.get("rung") or 0) == 0 or not rung.get("row_ids"):
                continue
            cid = rung.get("candidate_id") or "%s.r%s" % (entry.get("exec_id"), rung.get("rung"))
            out[str(cid)] = frozenset(rung["row_ids"])
    return out


def _board(eval_dir):
    path = os.path.join(eval_dir, "fusion_capture", "fusion", "fusion_topk.json")
    try:
        with open(path) as fh:
            return window_ids(json.load(fh))
    except (OSError, ValueError):
        return None


def candidates(eval_dirs, to_windows=None):
    """{(cid, side): (speedup, path, source_cid)} for passing verdicts across `eval_dirs`.

    With `to_windows` ({cid: rows} of this round's board) a verdict is keyed by the id this
    board gives its window, and dropped when the window is not on this board."""
    by_rows = {}
    for cid, rows in (to_windows or {}).items():
        by_rows.setdefault(rows, []).append(cid)
    best = {}
    for eval_dir in eval_dirs:
        source = _board(eval_dir) if to_windows is not None else None
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
            targets = [cid]
            if to_windows is not None:
                rows = (source or {}).get(cid)
                targets = by_rows.get(rows, []) if rows else []
            for target in targets:
                key = (target, side)
                if key not in best or speedup > best[key][0]:
                    best[key] = (speedup, path, cid)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--to", required=True, help="this round's VERDICT_DIR")
    ap.add_argument("--from", dest="sources", action="append", default=[],
                    help="an earlier eval dir of the same deployment and Top-K")
    ap.add_argument("--to-topk", default="",
                    help="this round's fusion_topk.json: match earlier verdicts by window, not id")
    ap.add_argument("--prefer-best", action="store_true",
                    help="replace this round's verdict when a --from verdict is faster")
    args = ap.parse_args(argv)
    os.makedirs(args.to, exist_ok=True)
    to_windows = None
    if args.to_topk:
        with open(args.to_topk) as fh:
            to_windows = window_ids(json.load(fh))
    reused = {}
    for (cid, side), (speedup, path, source_cid) in sorted(candidates(args.sources, to_windows).items()):
        target = os.path.join(args.to, "%s__%s.json" % (cid, side))
        if os.path.exists(target):
            if not args.prefer_best:
                continue  # this round already has its own verdict for it
            try:
                with open(target) as fh:
                    own = json.load(fh)
            except (OSError, ValueError):
                own = {}
            if str(own.get("parity", "")).lower() == "pass" and _speedup(own) >= speedup:
                continue
            shutil.copyfile(target, target + ".prev")
        with open(path) as fh:
            doc = json.load(fh)
        doc["reused_from"] = os.path.abspath(path)
        if source_cid != cid:
            doc["reused_candidate_id"] = source_cid
            doc["candidate_id"] = cid
        with open(target, "w") as fh:
            json.dump(doc, fh, indent=1)
        reused[cid] = {"side": side, "speedup": round(speedup, 4), "from": os.path.abspath(path)}
    print(json.dumps({"ok": True, "reused": reused}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
