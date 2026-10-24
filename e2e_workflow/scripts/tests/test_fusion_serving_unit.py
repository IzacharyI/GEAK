"""fusion_serving_unit.py + the harness's serving_step branch: a window a single card cannot
isolate is timed on the live stack (base vs overlay server) and gated like any other verdict."""
import argparse
import gzip
import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_serving_unit as fsu  # noqa: E402
import fusion_unitside_harness as uh  # noqa: E402


def _trace(path, cycles, non_ep_us, ep_us):
    """A rank trace: per cycle a DRAFT step then a VERIFY step; non_ep_us is the cycle total
    outside the EP MoE (draft 100us + verify the rest), ep_us the verify step's megamoe time."""
    events, ts = [], 0.0
    for _ in range(cycles):
        for step, kernels in (("DRAFT", [("draft_gemm", 100.0)]),
                              ("VERIFY", [("quant", non_ep_us - 100.0 - 50.0), ("rmsnorm", 50.0),
                                          ("megamoe_stage1_compact", ep_us)])):
            dur = sum(d for _, d in kernels) + 10.0
            events.append({"cat": "gpu_user_annotation", "name": "step[%s bs=5]" % step,
                           "ts": ts, "dur": dur})
            t = ts + 1.0
            for name, d in kernels:
                events.append({"cat": "kernel", "name": name, "ts": t, "dur": d})
                t += d
            ts += dur + 5.0
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)


def _side(root, name, non_ep_us, gsm8k, banners=0, points=("de_5", "de_2")):
    side = os.path.join(root, name)
    os.makedirs(os.path.join(side, "points"), exist_ok=True)
    out = {}
    for pid in points:
        traces = []
        for rank in range(2):
            path = os.path.join(side, "points", "%s-TP-%d.trace.json.gz" % (pid, rank))
            _trace(path, 8, non_ep_us[pid], 1000.0 + 50 * rank)
            traces.append({"path": path})
        journal = os.path.join(side, "points", pid + ".journal.json")
        with open(journal, "w") as fh:
            json.dump({"status": "ok", "trace_verified": True, "traces": traces}, fh)
        out[pid] = fsu.point_cost(journal)
    with open(os.path.join(side, "server.log"), "w") as fh:
        fh.write("[overlay-su_e04_r3] ENGAGED\n" * banners)
    doc = {"ok": True, "points": out, "gsm8k": list(gsm8k),
           "server_log": os.path.join(side, "server.log")}
    with open(os.path.join(side, "side.json"), "w") as fh:
        json.dump(doc, fh)
    return doc


def _args(**over):
    ns = argparse.Namespace(point_ids="de_5,de_2", tp=2, max_z=2.0)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


class StepTimeTest(unittest.TestCase):
    def test_a_cycle_is_draft_plus_verify_and_the_ep_moe_is_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "r.trace.json.gz")
            _trace(path, 8, 2000.0, 700.0)
            non_ep, ep, steps = fsu.trace_step_us(path)
            self.assertAlmostEqual(non_ep, 2000.0)   # the draft step counts with its verify step
            self.assertAlmostEqual(ep, 700.0)
            self.assertEqual(steps, 6)                       # first/last cycle dropped

    def test_the_observed_workload_supplies_points_and_dp(self):
        with tempfile.TemporaryDirectory() as tmp:
            wl = os.path.join(tmp, "fusion_observed_workload.json")
            with open(wl, "w") as fh:
                json.dump({"dp_attention_size": 8, "serving_weight_model": {
                    "sampling_points": "/s/sampling_points.json"}, "cases": [
                        {"sig": "pr_8192", "regime": "prefill", "effective_calls": 9},
                        {"sig": "de_2", "regime": "decode", "effective_calls": 7800},
                        {"sig": "de_5", "regime": "decode", "effective_calls": 16120}]}, fh)
            ns = argparse.Namespace(observed_workload=wl, points="", point_ids="", dp_size=0)
            fsu._from_workload(ns)
            self.assertEqual((ns.points, ns.point_ids, ns.dp_size),
                             ("/s/sampling_points.json", "de_5,de_2", 8))


class ServingVerdictTest(unittest.TestCase):
    WORKLOAD = {"kind": "agentx_trace_replay", "weighting": "observed",
                "serving_weight_model": {"source": "observed_workload",
                                         "evidence_path": "/eval/observed_workload.json"},
                "cases": [{"sig": "de_5", "regime": "decode", "m": 5, "effective_calls": 16120},
                          {"sig": "de_2", "regime": "decode", "m": 2, "effective_calls": 7800}]}

    def _harness(self, tmp, verdict):
        table = {"tables": [{"phase": "verify", "rows": [
            {"row_id": "r1", "raw_name": "write_c4_prefill", "parent_operator": "self_attn",
             "phase": "verify", "step_batch_size": 5,
             "shape": {"source": "unresolved", "input_dims": []}},
            {"row_id": "r2", "raw_name": "_swa_scatter_kernel", "parent_operator": "self_attn",
             "phase": "verify", "step_batch_size": 5,
             "shape": {"source": "unresolved", "input_dims": []}}]}]}
        cands = {"candidates": [{"candidate_id": "e04.r3", "phase": "verify", "kind": "serial",
                                 "row_ids": ["r1", "r2"], "op_tags": ["kv_cache"],
                                 "kernel_names": ["write_c4_prefill", "_swa_scatter_kernel"],
                                 "matched_existing_kernels": [], "self_author": True}]}
        topk = {"execution_list": [{"exec_id": "e04", "candidate_ids": ["e04.r3"]}]}
        paths = {}
        for name, doc in (("c.json", cands), ("t.json", table), ("k.json", topk)):
            paths[name] = os.path.join(tmp, name)
            with open(paths[name], "w") as fh:
                json.dump(doc, fh)
        vdir = os.path.join(tmp, "verdict")
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "e04.r3__serving.json"), "w") as fh:
            json.dump(verdict, fh)
        res = uh.validate(paths["c.json"], vdir, topk_path=paths["k.json"],
                          semantic_table_path=paths["t.json"], observed_workload=self.WORKLOAD)
        return res, next((r for r in res["results"] if r["candidate_id"] == "e04.r3"), None)

    def _verdict(self, tmp, cand_us, banners=2, cand_gsm=(0.95, 200)):
        base = _side(tmp, "base", {"de_5": 24000.0, "de_2": 21000.0}, (0.95, 200))
        cand = _side(tmp, "e04.r3", cand_us, cand_gsm, banners=banners)
        kernel = os.path.join(tmp, "k.py")
        open(kernel, "w").close()
        return fsu.verdict_for("e04.r3", "su_e04_r3", kernel, base, cand, _args())

    def test_a_faster_engaged_overlay_with_gsm8k_intact_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._verdict(tmp, {"de_5": 23700.0, "de_2": 20800.0})
            self.assertEqual(doc["parity"], "pass")
            self.assertEqual(doc["measure_scope"], "serving_step")
            res, row = self._harness(tmp, doc)
            self.assertFalse(res["errors"], res["errors"])
            self.assertEqual(row["unit_side_status"], "pass")
            calls = {"de_5": 16120, "de_2": 7800}
            ref = {"de_5": 24.1, "de_2": 21.1}
            cnd = {"de_5": 23.8, "de_2": 20.9}
            expected = sum(ref[k] * calls[k] for k in calls) / sum(cnd[k] * calls[k] for k in calls)
            self.assertAlmostEqual(row["weighted_speedup"], round(expected, 4), places=3)

    def test_a_gain_inside_the_noise_floor_is_no_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._verdict(tmp, {"de_5": 23950.0, "de_2": 20990.0})
            _, row = self._harness(tmp, doc)
            self.assertEqual(row["unit_side_status"], "fail")
            self.assertEqual(row["failure_kind"], "performance")

    def test_no_banner_is_not_engaged_and_a_gsm8k_drop_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._verdict(tmp, {"de_5": 23000.0, "de_2": 20000.0}, banners=1)
            self.assertEqual(doc["parity"], "not_engaged")
            _, row = self._harness(tmp, doc)
            self.assertEqual(row["unit_side_status"], "blocked")
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._verdict(tmp, {"de_5": 23000.0, "de_2": 20000.0}, cand_gsm=(0.80, 200))
            self.assertEqual(doc["parity"], "fail")
            self.assertIn("parity_diagnosis", doc)
            _, row = self._harness(tmp, doc)
            self.assertEqual(row["unit_side_status"], "fail")

    def test_a_verdict_without_its_measured_sides_is_untrusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = self._verdict(tmp, {"de_5": 23000.0, "de_2": 20000.0})
            doc["serving_unit"]["cand_dir"] = os.path.join(tmp, "nowhere")
            res, row = self._harness(tmp, doc)
            self.assertTrue(any("serving_step verdict names no measured" in e for e in res["errors"]))
            self.assertNotEqual((row or {}).get("unit_side_status"), "pass")

if __name__ == "__main__":
    unittest.main()
