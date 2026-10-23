"""fusion_agentx_sampling.py: AgentX baseline log -> sampling points -> fusion trace
manifest + observed-workload UT weights. No service, no GPU."""
import gzip
import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_agentx_sampling as fas  # noqa: E402
import harness_lib  # noqa: E402

LOG = (
    '[2026-10-01 00:00:10 TP0] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, #running-req: 2, #queue-req: 0\n'
    '[2026-10-01 00:00:11 TP1] Prefill batch, #new-seq: 1, #new-token: 32, #cached-token: 96, #running-req: 2, #queue-req: 0\n'
    '[2026-10-01 00:00:12 TP0] Prefill batch, #new-seq: 2, #new-token: 4096, #cached-token: 768, #running-req: 4, #queue-req: 0\n'
    '[2026-10-01 00:00:12 TP0] Prefill batch, #new-seq: 2, #new-token: 4000, #cached-token: 768, #running-req: 4, #queue-req: 0\n'
    '[2026-10-01 00:00:13 TP0] Decode batch, #running-req: 2, #token: 256, accept len: 1.0, cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n'
    '[2026-10-01 00:00:14 TP0] Decode batch, #running-req: 2, #token: 300, accept len: 1.0, cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n'
    '[2026-10-01 00:00:15 TP0] Decode batch, #running-req: 2, #token: 320, accept len: 1.0, cuda graph: True, gen throughput (token/s): 100.0, #queue-req: 0\n'
    '[2026-10-01 00:00:16 TP0] Decode batch, #running-req: 16, #token: 9000, accept len: 1.0, cuda graph: True, gen throughput (token/s): 900.0, #queue-req: 0\n')


def _trace(path, steps):
    events = []
    for index, step in enumerate(steps):
        start = index * 100
        events.append({"cat": "gpu_user_annotation", "name": step, "ts": start, "dur": 50})
        events.append({"cat": "kernel", "name": "k", "ts": start + 10, "dur": 5})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt") as fh:
        json.dump({"traceEvents": events}, fh)


class FusionAgentxSamplingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fusion agentx ")
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.log = os.path.join(self.root, "server.log")
        with open(self.log, "w") as fh:
            fh.write(LOG)
        self.info = os.path.join(self.root, "server_info.json")
        with open(self.info, "w") as fh:
            json.dump({"decode_log_interval": 10, "cuda_graph_bs": [1, 2, 4, 8, 16, 32]}, fh)
        self.plan_dir = os.path.join(self.root, "sampling")

    def plan(self):
        fas.main(["plan", "--log", self.log, "--server-info", self.info, "-n", "4",
                  "--out-dir", self.plan_dir])
        with open(os.path.join(self.plan_dir, "fusion_observed_workload.json")) as fh:
            return json.load(fh)

    def probe(self, workload, sig, steps, status="ok", us_per_step=None):
        _trace(os.path.join(self.plan_dir, "points", sig, "1730000000.1-TP-0.trace.json.gz"), steps)
        doc = {"point_id": sig, "status": status, "verify_reason": status}
        if us_per_step is not None:
            doc["step_cost"] = {"us_per_step": us_per_step, "basis": "kernel_device_time"}
        with open(os.path.join(self.plan_dir, "points", sig + ".journal.json"), "w") as fh:
            json.dump(doc, fh)

    def assemble(self, phases=None):
        manifest = os.path.join(self.root, "capture", "profile_trace_manifest.json")
        argv = ["assemble", "--plan-dir", self.plan_dir,
                "--capture-dir", os.path.join(self.root, "capture"),
                "--manifest", manifest]
        if phases:
            argv += ["--analysis-phases", phases]
        rc = fas.main(argv)
        with open(manifest) as fh:
            return rc, json.load(fh)

    def test_plan_weights_every_point_by_the_forwards_it_represents(self):
        wl = self.plan()
        self.assertEqual(wl["kind"], "agentx_trace_replay")
        self.assertEqual(wl["serving_weight_model"]["source"], "observed_workload")
        decode = {c["sig"]: c for c in wl["cases"] if c["regime"] == "decode"}
        # 3 logged rows at bs=2 and 1 at bs=16, each standing for 10 iterations.
        self.assertEqual(decode["de_2"]["effective_calls"], 30.0)
        self.assertEqual(decode["de_16"]["effective_calls"], 10.0)
        self.assertEqual(decode["de_2"]["m"], 2)
        self.assertEqual(wl["analysis_points"]["decode"], "de_2")
        # prefill mass is new tokens: the 4K bucket outweighs the 32-token one.
        self.assertTrue(wl["analysis_points"]["prefill"].startswith("pr_4"))
        prefill_calls = sum(c["effective_calls"] for c in wl["cases"] if c["regime"] == "prefill")
        self.assertEqual(prefill_calls, 3.0)
        # The UT contract the shared harness enforces accepts these cases as-is.
        meta = {"workload": wl}
        self.assertEqual(len(harness_lib._observed_workload_cases(meta)), len(wl["cases"]))

    def test_default_analysis_is_the_top1_decode_point_by_gpu_time(self):
        wl = self.plan()
        # The log proxy prefers de_2 (30 iterations vs 10), but de_16's measured step is
        # 4x as expensive: 10 x 400us > 30 x 100us, so de_16 carries the GPU time.
        self.assertEqual(wl["analysis_points"]["decode"], "de_2")
        self.probe(wl, "de_2", ["step[DECODE bs=2]", "step[DECODE bs=2]"], us_per_step=100.0)
        self.probe(wl, "de_16", ["step[DECODE bs=16]"], us_per_step=400.0)
        self.probe(wl, wl["analysis_points"]["prefill"], ["step[EXTEND bs=2 toks=4096]"],
                   us_per_step=900.0)
        rc, manifest = self.assemble()
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["status"], "pass")
        sampling = manifest["agentx_sampling"]
        self.assertEqual(sampling["analysis_phases"], ["decode"])
        self.assertEqual(sampling["analysis_points"], {"decode": "de_16"})
        names = sorted(os.path.basename(f["path"]) for f in manifest["trace_files"])
        self.assertEqual(names, ["fusion-agentx-TP-0-DECODE.trace.json.gz"])
        ranking = {r["sig"]: r for r in sampling["gpu_time_ranking"]}
        self.assertEqual(ranking["de_16"]["gpu_time_mass_us"], 4000.0)
        self.assertEqual(ranking["de_2"]["gpu_time_mass_us"], 3000.0)
        self.assertAlmostEqual(ranking["de_16"]["gpu_time_share_pct"], 57.14)
        with open(os.path.join(self.plan_dir, "fusion_observed_workload.json")) as fh:
            updated = json.load(fh)
        # Every point stays a UT case: the analysis choice does not drop weights.
        self.assertEqual(len(updated["cases"]), len(wl["cases"]))
        self.assertEqual(updated["analysis_points_used"], {"decode": "de_16"})
        self.assertEqual(updated["trace_manifest_json"],
                         os.path.abspath(os.path.join(self.root, "capture",
                                                      "profile_trace_manifest.json")))

    def test_prefill_and_decode_pair_as_extend_and_decode_on_request(self):
        wl = self.plan()
        pre, dec = wl["analysis_points"]["prefill"], wl["analysis_points"]["decode"]
        self.probe(wl, pre, ["step[EXTEND bs=2 toks=4096]"], us_per_step=900.0)
        self.probe(wl, dec, ["step[DECODE bs=2]", "step[DECODE bs=2]"], us_per_step=100.0)
        rc, manifest = self.assemble(phases="prefill,decode")
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["status"], "pass")
        names = sorted(os.path.basename(f["path"]) for f in manifest["trace_files"])
        self.assertEqual(names, ["fusion-agentx-TP-0-DECODE.trace.json.gz",
                                 "fusion-agentx-TP-0-EXTEND.trace.json.gz"])
        self.assertEqual(manifest["agentx_sampling"]["analysis_points"],
                         {"prefill": pre, "decode": dec})
        # The semantic mapper reads analysis_rank_trace and adopts its phase sibling.
        import semantic_kernel_mapping
        siblings = semantic_kernel_mapping._sibling_phase_traces(manifest["analysis_rank_trace"])
        self.assertEqual(sorted(siblings), ["DECODE", "EXTEND"])

    def test_a_failed_top_point_hands_over_to_the_next_and_says_so(self):
        wl = self.plan()
        self.probe(wl, "de_2", ["step[DECODE bs=2]"], status="failed", us_per_step=500.0)
        self.probe(wl, "de_16", ["step[DECODE bs=16]"], us_per_step=100.0)
        rc, manifest = self.assemble()
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["agentx_sampling"]["analysis_points"]["decode"], "de_16")
        self.assertTrue(any("de_2" in n and "de_16" in n
                            for n in manifest["agentx_sampling"]["notes"]))

    def test_unmeasured_costs_fall_back_to_the_proxy_order_and_say_so(self):
        wl = self.plan()
        self.probe(wl, "de_2", ["step[DECODE bs=2]"])
        self.probe(wl, "de_16", ["step[DECODE bs=16]"])
        rc, manifest = self.assemble()
        self.assertEqual(rc, 0)
        self.assertEqual(manifest["agentx_sampling"]["analysis_points"]["decode"], "de_2")
        self.assertTrue(any("proxy" in n for n in manifest["agentx_sampling"]["notes"]))

    def test_no_decode_trace_fails_the_manifest(self):
        wl = self.plan()
        self.probe(wl, wl["analysis_points"]["prefill"], ["step[EXTEND bs=2 toks=4096]"])
        rc, manifest = self.assemble()
        self.assertEqual(rc, 2)
        self.assertEqual(manifest["status"], "failed")
        self.assertIn("decode", manifest["missing_required_phases"])

    def test_hybrid_swa_dsv4_log_plans_through_the_kernelfusion_front_end(self):
        with open(self.log, "w") as fh:
            fh.write(
                '[2026-10-01 00:00:10 DP0 TP0 EP0] Prefill batch, #new-seq: 1, #new-token: 8192, #cached-token: 65536, full token usage: 0.30, swa token usage: 0.05, #running-req: 3, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 1000.00\n'
                '[2026-10-01 00:00:11 DP1 TP1 EP1] Prefill batch, #new-seq: 1, #new-token: 64, #cached-token: 0, full token usage: 0.30, swa token usage: 0.05, #running-req: 3, #queue-req: 0, #pending-token: 0, cuda graph: False, input throughput (token/s): 10.00\n'
                '[2026-10-01 00:00:12 DP0 TP0 EP0] Decode batch, #running-req: 16, #full token: 160000, full token usage: 0.40, #swa token: 4096, swa token usage: 0.10, accept len: 3.40, accept rate: 0.40, block accept len: 3.10, cuda graph: True, gen throughput (token/s): 800.00, #queue-req: 2\n')
        wl = self.plan()
        decode = [c for c in wl["cases"] if c["regime"] == "decode"]
        self.assertEqual([c["m"] for c in decode], [16])
        self.assertEqual(decode[0]["effective_calls"], 10.0)
        self.assertEqual([c["m"] for c in wl["cases"] if c["regime"] == "prefill"], [8192])

    def test_synthetic_shape_needs_no_log(self):
        with open(self.info, "w") as fh:
            json.dump({"enable_dp_attention": True, "dp_size": 8}, fh)
        fas.main(["plan", "--synthetic", "8192,1024,128", "--server-info", self.info,
                  "--out-dir", self.plan_dir])
        with open(os.path.join(self.plan_dir, "fusion_observed_workload.json")) as fh:
            wl = json.load(fh)
        self.assertEqual(wl["kind"], "synthetic_fixed_shape")
        cases = {c["regime"]: c for c in wl["cases"]}
        self.assertEqual(cases["decode"]["m"], 16)      # 128 requests over 8 DP ranks
        self.assertEqual(cases["prefill"]["m"], 8192)


if __name__ == "__main__":
    unittest.main()
