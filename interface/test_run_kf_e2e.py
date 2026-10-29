"""interface/run_kf_e2e.py: KernelFusion stage -> derived handoff -> unchanged run_e2e.py."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e2e as R  # noqa: E402
import run_kf_e2e as K  # noqa: E402

FLAGS = "--served-model-name m --tensor-parallel-size 8 --page-size 256"
ENV = "PYTHONNOUSERSITE=1 SGLANG_X=1"


def _handoff(**extra):
    h = {"model_path": "/data/models/M", "framework": "sglang", "tp": 8,
         "gpu_ids": "0,1,2,3,4,5,6,7", "expected_gfx": "gfx950", "expected_target": "unknown",
         "workload": {"isl": 1024, "osl": 1024, "conc": 128},
         "workload_spec": {"kind": "agentx_trace_replay", "concurrency": 128},
         "accepted_flags": FLAGS, "accepted_env": ENV, "exp_root": "/tmp/x"}
    h.update(extra)
    return h


ACCEPTED = {"schema": K.KF_SCHEMA, "status": "applied", "applied_count": 2,
            "final_overlay": "/e/fusion/fusion_overlays/M/combined",
            "accepted_flags": FLAGS, "accepted_env": ENV, "delta_pct": 4.0,
            "baseline_throughput_tok_s": 100.0, "final_throughput_tok_s": 104.0, "eval_dir": "/e"}


class DeriveHandoffTest(unittest.TestCase):
    def test_nothing_accepted_passes_the_handoff_through_untouched(self):
        h = _handoff()
        for kf in ({"status": "no_win", "applied_count": 0, "final_overlay": ""},
                   {"status": "combined_ab_failed", "applied_count": 0, "final_overlay": "/x"},
                   {"status": "no_return"}):
            self.assertIs(K.derive_handoff(h, kf), h)

    def test_accepted_stack_is_the_e2e_workflows_starting_overlay(self):
        h = _handoff()
        d = K.derive_handoff(h, ACCEPTED)
        self.assertEqual(d["schema_version"], 2)
        self.assertEqual(d["baseline_env_spec"]["overlay_pythonpath"], ACCEPTED["final_overlay"])
        self.assertNotIn("baseline_env_spec", h)          # the caller's handoff is not mutated
        args = R.map_args(d, 3600, dry_run=True)
        self.assertEqual(args["initial_overlay_pythonpath"], ACCEPTED["final_overlay"])
        for flag in FLAGS.split():
            self.assertIn(flag, args["initial_extra_server_args"])
        self.assertIn("SGLANG_X=1", args["initial_extra_env"])
        self.assertIn("PYTHONNOUSERSITE=1", args["initial_extra_env"])

    def test_a_prior_overlay_stays_behind_the_fused_one(self):
        h = _handoff(schema_version=2, baseline_env_spec={"overlay_pythonpath": "/prior"})
        d = K.derive_handoff(h, ACCEPTED)
        self.assertEqual(d["baseline_env_spec"]["overlay_pythonpath"],
                         ACCEPTED["final_overlay"] + ":/prior")
        args = R.map_args(d, 3600, dry_run=True)
        self.assertTrue(args["initial_overlay_pythonpath"].startswith(ACCEPTED["final_overlay"]))


class StageReturnTest(unittest.TestCase):
    def test_disk_return_is_used_when_the_workflow_reply_is_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "kf_return.json").write_text(json.dumps(ACCEPTED))
            kf = K.read_kf_return(Path(tmp), None)
            self.assertEqual(kf["status"], "applied")
            self.assertTrue(kf["recovered_from_disk"])
            self.assertEqual(K.read_kf_return(Path(tmp) / "missing", None)["status"], "no_return")

    def test_combined_gain_is_the_product_of_the_two_same_protocol_ratios(self):
        sec = K.kernel_fusion_section(ACCEPTED, {"throughput_speedup": 1.10}, "/h.json")
        self.assertAlmostEqual(sec["combined_speedup_vs_original"], 1.144)
        self.assertTrue(sec["main_baseline_is_fused_stack"])
        sec = K.kernel_fusion_section({"status": "no_win", "delta_pct": 0}, {"throughput_speedup": 1.1}, "/h")
        self.assertAlmostEqual(sec["combined_speedup_vs_original"], 1.1)
        self.assertEqual(sec["final_overlay"], "")

    def test_the_l3_outcome_travels_only_when_stage_fusion_ran(self):
        self.assertNotIn("l3", K.kernel_fusion_section(ACCEPTED, {}, "/h"))
        l3 = {"stages": [{"stage_id": "l3_01", "status": "packaged"}]}
        self.assertEqual(K.kernel_fusion_section(dict(ACCEPTED, l3=l3), {}, "/h")["l3"], l3)

    def test_budget_is_capped_to_leave_the_e2e_workflow_its_share(self):
        self.assertEqual(K.kf_budget_s({}, {}, 129600), 12 * 3600)
        self.assertEqual(K.kf_budget_s({}, {"kf_budget_s": 99999}, 36000), 24000)
        self.assertEqual(K.kf_budget_s({"kf_budget_s": 7200}, {}, 129600), 7200)

    def test_fusion_knobs_reach_the_stage(self):
        args = K.kf_args({"fusion_top_k": 4, "semantics_shape_capture": "false", "kf_budget_s": 1,
                          "accepted_flags": FLAGS},
                         {"model_path": "/m", "tp": 8, "gpu_ids": "0", "expected_gfx": "gfx950"},
                         "/e", 3600)
        self.assertEqual(args["fusion_top_k"], 4)
        self.assertEqual(args["semantics_shape_capture"], "false")
        self.assertNotIn("kf_budget_s", args)
        self.assertNotIn("accepted_flags", args)
        self.assertEqual(args["gfx"], "gfx950")
        self.assertEqual(args["time_budget_s"], 3600)
        self.assertEqual(args["eval_dir"], "/e")

    def test_prompt_points_the_workflow_tool_at_the_stage_script(self):
        prompt = K.build_kf_prompt({"eval_dir": "/e", "model_path": "/m"})
        self.assertIn(str(K.KF_SCRIPT), prompt)
        self.assertIn("/e/kf_return.json", prompt)
        self.assertTrue(K.KF_SCRIPT.is_file())


if __name__ == "__main__":
    unittest.main()
