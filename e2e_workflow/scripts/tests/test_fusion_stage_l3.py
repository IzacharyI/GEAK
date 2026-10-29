#!/usr/bin/env python3
"""StageFusion (L3) in kernel_fusion/kf_workflow.js: EP MoE stages handed to GEAK mode=mega.

Runs the workflow under the mock runtime (tests/kf_workflow_mockrun.mjs, workflow() stubbed as
kernel_mega.js); skipped without a Node.js binary (GEAK_NODE_BIN, else `node` on PATH)."""

import json
import os
import shutil
import subprocess
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
WORKFLOW = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(WORKFLOW)
KF_SCRIPT = os.path.join(WORKFLOW, "kernel_fusion", "kf_workflow.js")
MOCKRUN = os.path.join(HERE, "kf_workflow_mockrun.mjs")
NODE = os.environ.get("GEAK_NODE_BIN") or shutil.which("node")


def _inputs(prompt):
    """The `- KEY: value` input lines of a role prompt."""
    out = {}
    for line in prompt.split("## Inputs", 1)[-1].splitlines():
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            out[key] = value
    return out


class StageFusionStaticTest(unittest.TestCase):
    def test_role_and_mega_worker_exist(self):
        with open(os.path.join(WORKFLOW, "roles", "fusion_stage_l3.md")) as fh:
            role = fh.read()
        for phase in ("## PHASE=prepare", "## PHASE=package"):
            self.assertIn(phase, role)
        self.assertTrue(os.path.isfile(os.path.join(REPO, "kernel_workflow", "kernel_mega.js")))

    def test_new_files_are_in_the_manifest(self):
        with open(os.path.join(WORKFLOW, "kernel_fusion", "MANIFEST.txt")) as fh:
            listed = set(fh.read().split())
        for path in ("e2e_workflow/roles/fusion_stage_l3.md",
                     "e2e_workflow/scripts/tests/test_fusion_stage_l3.py"):
            self.assertIn(path, listed)


@unittest.skipUnless(NODE, "no Node.js binary (set GEAK_NODE_BIN)")
class StageFusionBehaviourTest(unittest.TestCase):
    def run_stage(self, scenario):
        proc = subprocess.run([NODE, MOCKRUN, KF_SCRIPT, scenario], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["bad"], [], "a prompt rendered undefined/NaN/[object Object]")
        return out

    def test_off_by_default(self):
        out = self.run_stage("accept")
        self.assertNotIn("StageFusion", out["phases"])
        self.assertNotIn("l3", out["ret"])
        self.assertEqual(out["workflowCalls"], [])
        self.assertNotIn("stage-l3", out["prompts"]["fusion-screen"])
        self.assertNotIn("L3", out["prompts"]["fusion_integrator:combined_ab"])

    def test_a_stage_runs_mega_and_joins_the_combined_ab(self):
        out = self.run_stage("l3")
        self.assertEqual(out["phases"], ["Baseline", "Sample", "Semantics", "Screen", "UnitSide",
                                         "StageFusion", "ApplyBack", "Report"])
        self.assertIn("--stage-l3", out["prompts"]["fusion-screen"])
        # prepare -> mega -> package, each under the serving lease, before apply-back authoring
        labels = out["labels"]
        order = ["fusion-lease:acquire:l3_prepare_l3_01", "fusion_stage_l3:prepare:l3_01",
                 "fusion-lease:acquire:l3_mega_l3_01", "fusion-lease:wait:l3_mega_l3_01",
                 "fusion_stage_l3:package:l3_01", "fusion_integrator:author_one:e1",
                 "fusion_integrator:combined_ab"]
        self.assertEqual([labels.index(x) for x in order], sorted(labels.index(x) for x in order))
        # l3_02 is below the share floor: never prepared
        self.assertNotIn("fusion_stage_l3:prepare:l3_02", labels)

        (call,) = out["workflowCalls"]
        self.assertEqual(call["ref"], {"scriptPath": "/w/kernel_workflow/kernel_mega.js"})
        a = call["args"]
        self.assertEqual((a["mode"], a["workflow_dir"], a["gpu_ids"], a["gpus_per_job"]),
                         ("mega", "/w/kernel_workflow", "0,1,2,3,4,5,6,7", 8))
        self.assertEqual(a["kernel_path"], "/e/fusion/l3/l3_01/baseline_tree")
        self.assertEqual(a["state_dir"], "/e/fusion/l3/l3_01/mega_state")
        self.assertEqual(a["budget"], 12)                  # the workflow's, not prepare's
        self.assertEqual(a["target_guards"], ["64_uniform"])
        self.assertEqual(a["use_expert_skills"], "true")   # caller override wins

        ab = _inputs(out["prompts"]["fusion_integrator:combined_ab"])
        authored = json.loads(ab["AUTHORED"])
        self.assertEqual([r["exec_id"] for r in authored], ["e1", "l3_01"])
        self.assertEqual(authored[1]["level"], "L3")
        self.assertEqual(json.loads(ab["L3_SUBSUMED"]), {"e2": "l3_01"})

        ret = out["ret"]
        self.assertEqual(ret["status"], "applied")
        stages = {s["stage_id"]: s for s in ret["l3"]["stages"]}
        self.assertEqual(stages["l3_01"]["status"], "applied")
        self.assertEqual(stages["l3_01"]["mega"]["selected"]["final_tree"], "/m/f3")
        self.assertEqual(stages["l3_02"]["status"], "not_picked")

    def test_no_deliverable_leaves_the_l1_l2_stack_unchanged(self):
        out = self.run_stage("l3_nodeliver")
        self.assertNotIn("fusion_stage_l3:package:l3_01", out["labels"])
        ab = _inputs(out["prompts"]["fusion_integrator:combined_ab"])
        self.assertEqual([r["exec_id"] for r in json.loads(ab["AUTHORED"])], ["e1", "e2"])
        self.assertNotIn("L3_SUBSUMED", ab)
        self.assertEqual(out["ret"]["l3"]["stages"][0]["status"], "mega_no_deliverable")

    def test_a_mega_exception_does_not_stop_kernelfusion(self):
        out = self.run_stage("l3_throw")
        self.assertIn("fusion_integrator:combined_ab", out["labels"])
        stage = out["ret"]["l3"]["stages"][0]
        self.assertEqual(stage["status"], "mega_error")
        self.assertIn("Setup failed", stage["reason"])

    def test_an_l3_only_board_still_reaches_apply_back(self):
        out = self.run_stage("l3_only")
        self.assertNotIn("UnitSide", out["phases"])
        self.assertIn("ApplyBack", out["phases"])
        self.assertFalse(any(l.startswith("fusion_integrator:author_one") for l in out["labels"]))
        ab = _inputs(out["prompts"]["fusion_integrator:combined_ab"])
        self.assertEqual([r["exec_id"] for r in json.loads(ab["AUTHORED"])], ["l3_01"])

    def test_an_earlier_mega_result_is_packaged_without_running_mega(self):
        out = self.run_stage("l3_reuse")
        self.assertEqual(out["workflowCalls"], [])
        self.assertNotIn("fusion_stage_l3:prepare:l3_01", out["labels"])
        pkg = _inputs(out["prompts"]["fusion_stage_l3:package:l3_01"])
        self.assertEqual(pkg["MEGA_RESULT_JSON"], "/old/fusion/l3/l3_01/mega_result.json")
        self.assertEqual(out["ret"]["l3"]["stages"][0]["status"], "applied")

    def test_an_l3_overlay_that_loses_end_to_end_does_not_cost_the_l1_l2_stack(self):
        out = self.run_stage("l3_abreject")
        labels = out["labels"]
        self.assertLess(labels.index("fusion_integrator:combined_ab"),
                        labels.index("fusion_integrator:combined_ab:without_l3"))
        retry = out["prompts"]["fusion_integrator:combined_ab:without_l3"]
        self.assertIn("RETRY WITHOUT L3", retry)
        ab = _inputs(retry)
        # the L3 row leaves; the L1/L2 entry it had subsumed comes back
        self.assertEqual([r["exec_id"] for r in json.loads(ab["AUTHORED"])], ["e1", "e2"])
        self.assertEqual([r["exec_id"] for r in json.loads(ab["L3_REJECTED"])], ["l3_01"])
        self.assertNotIn("L3_SUBSUMED", ab)
        ret = out["ret"]
        self.assertEqual((ret["status"], ret["final_overlay"]), ("applied", "/e/stack2"))
        stage = ret["l3"]["stages"][0]
        self.assertEqual(stage["status"], "ab_rejected")
        self.assertIn("gsm8k dropped", stage["reason"])

    def test_an_accepted_l3_overlay_is_marked_applied(self):
        self.assertEqual(self.run_stage("l3")["ret"]["l3"]["stages"][0]["status"], "applied")

    def test_a_blocked_prepare_is_retried_once_with_its_reason(self):
        out = self.run_stage("l3_prepretry")
        retry = out["prompts"]["fusion_stage_l3:prepare:l3_01:retry"]
        self.assertIn("RETRY: the previous prepare", retry)
        self.assertIn("UT failed: wrong call site", retry)
        (call,) = out["workflowCalls"]
        self.assertEqual(call["args"]["gpus_per_job"], 4)   # the run's ep when prepare names none

    def test_operator_hint_and_extra_ep_regex_reach_their_agents(self):
        out = self.run_stage("l3")
        self.assertEqual(_inputs(out["prompts"]["fusion-screen"])["EP_KERNEL_REGEX"], "uccl_")
        hint = json.loads(_inputs(out["prompts"]["fusion_stage_l3:prepare:l3_01"])["OPERATOR_HINT"])
        self.assertEqual(hint["call_site"], "aiter.megamoe:mega_moe")


if __name__ == "__main__":
    unittest.main()
