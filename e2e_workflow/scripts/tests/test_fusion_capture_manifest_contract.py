#!/usr/bin/env python3
"""Contract checks for the standalone KernelFusion stage (kernel_fusion/kf_workflow.js).

Static checks read the workflow source; the behavioural checks run it under a mock Workflow
runtime (tests/kf_workflow_mockrun.mjs) and are skipped when no Node.js binary is available
(GEAK_NODE_BIN, else `node` on PATH)."""

import json
import os
import shutil
import subprocess
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
WORKFLOW = os.path.dirname(SCRIPTS)
ROLES = os.path.join(WORKFLOW, "roles")
KF_SCRIPT = os.path.join(WORKFLOW, "kernel_fusion", "kf_workflow.js")
MOCKRUN = os.path.join(HERE, "kf_workflow_mockrun.mjs")
NODE = os.environ.get("GEAK_NODE_BIN") or shutil.which("node")


def _source():
    with open(KF_SCRIPT) as fh:
        return fh.read()


def _role(name):
    with open(os.path.join(ROLES, name)) as fh:
        return fh.read()


class KernelFusionStageContractTest(unittest.TestCase):
    def test_stages_run_in_order_and_hand_back_to_the_e2e_workflow(self):
        source = _source()
        order = [source.index("phase('%s')" % p) for p in
                 ("Baseline", "Sample", "Semantics", "Screen", "UnitSide", "ApplyBack", "Report")]
        self.assertEqual(order, sorted(order))
        self.assertNotIn("phase('Strategize')", source)
        self.assertNotIn("label: 'profiler:baseline'", source)
        self.assertIn("return kfReturn;", source)
        self.assertIn("schema: 'geak.kernel_fusion_stage/1'", source)

    def test_sampling_follows_the_kernel_extract_launch_and_capture(self):
        source = _source()
        capture = source[source.index("const captureInputs = {"):source.index("let samplingStarted")]
        for key in ("SAMPLING_DIR: fusionSamplingDir", "SAMPLING_POINTS: FUSION_SAMPLING_POINTS",
                    "ANALYSIS_PHASES: FUSION_ANALYSIS_PHASES",
                    "OBSERVED_WORKLOAD_JSON: `${fusionSamplingDir}/fusion_observed_workload.json`",
                    "BASELINE_LOG: baseline.server_log", "SYNTHETIC_SHAPE: `${ISL},${OSL},${CONC}`",
                    "TRACE_MANIFEST_JSON: expectedFusionManifest"):
            self.assertIn(key, capture)
        self.assertIn("A.fusion_analysis_phases || 'decode'", source)
        role = _role("fusion_trace_collector.md")
        for text in ("fusion_serving_session.sh", "fusion_agentx_sampling.py run",
                     "--analysis-phases", "gpu_time_ranking", "TOP-1 point by that mass",
                     "kernel-extract", "fusion_leg_wait.sh"):
            self.assertIn(text, role)
        # The old in-replay capture path is gone: no bench_e2e fusion window, no POST_BENCH_HOOK.
        self.assertNotIn("POST_BENCH_HOOK", role)
        self.assertNotIn("GEAK_FUSION_TRACE", role)

    def test_the_manifest_path_is_deterministic_when_the_agent_returns_nothing(self):
        source = _source()
        self.assertIn("const fusionCaptureDir = `${EVAL_DIR}/${fusionRound}`;", source)
        self.assertIn(
            "const expectedFusionManifest = `${fusionCaptureDir}/profile_trace_manifest.json`;", source)
        self.assertIn("? fusionCapture.trace_manifest_json : expectedFusionManifest;", source)
        self.assertIn("TRACE_MANIFEST_JSON: fusionTraceManifest", source)

    def test_semantics_is_told_its_analysis_phases(self):
        source = _source()
        self.assertIn("SEMANTICS_REQUIRE_PHASES: FUSION_ANALYSIS_PHASES", source)
        self.assertEqual(source.count("...SEMANTICS_PHASE_INPUTS"), 2)
        self.assertIn('export GEAK_SEMANTICS_REQUIRE_PHASES="$SEMANTICS_REQUIRE_PHASES"',
                      _role("semantics_mapper.md"))

    def test_unit_gate_is_weighted_by_the_observed_workload(self):
        source = _source()
        self.assertIn("OBSERVED_WORKLOAD_JSON: fusionObservedWorkload", source)
        self.assertIn("--observed-workload", _role("fusion_unit_validator.md"))

    def test_agentx_ab_is_one_replay_per_side(self):
        source = _source()
        self.assertIn("A.fusion_ab_repeat_mode || (IS_AGENTX ? 'replay_once' : 'warm_server')", source)
        self.assertEqual(source.count("AB_REPEAT_MODE: FUSION_AB_REPEAT_MODE"), 2)
        self.assertIn('AB_REPEAT_MODE="$AB_REPEAT_MODE"', _role("fusion_integrator.md"))
        self.assertIn("fusion_gsm8k_eval.py", source)

    def test_external_fusion_artifacts_cannot_bypass_run_local_discovery(self):
        source = _source()
        self.assertNotIn("(A.fusion && typeof A.fusion === 'object')", source)
        self.assertIn("let FUSION_INPUTS = { FUSION_TOPK_JSON: '', FUSION_CANDIDATES_JSON: ''", source)

    def test_unexpected_exception_restores_the_pre_fusion_stack(self):
        source = _source()
        start = source.index("const fusionRecoveryState = {")
        block = source[source.index("} catch (e) {", start):source.index("phase('Report')")]
        for text in ("curOverlay = fusionRecoveryState.overlay;", "curFlags = fusionRecoveryState.flags;",
                     "curEnv = fusionRecoveryState.env;", "curTput = fusionRecoveryState.throughput;",
                     "status: 'unexpected_exception'", "failed_stage: fusionFailureStage"):
            self.assertIn(text, block)

    def test_only_an_accepted_stack_leaves_the_stage(self):
        source = _source()
        self.assertIn("final_overlay: kfApplied.length ? (curOverlay || '') : INIT_OVERLAY,", source)
        self.assertIn("accepted_flags: kfApplied.length ? curFlags : INIT_FLAGS,", source)
        self.assertIn("accepted_env: kfApplied.length ? curEnv : INIT_ENV,", source)

    def test_no_forbidden_runtime_apis(self):
        source = _source()
        self.assertNotIn("import(", source)
        self.assertNotIn("Date.now(", source)
        self.assertNotIn("require(", source)


@unittest.skipUnless(NODE, "no Node.js binary (set GEAK_NODE_BIN)")
class KernelFusionStageBehaviourTest(unittest.TestCase):
    def run_stage(self, scenario):
        proc = subprocess.run([NODE, MOCKRUN, KF_SCRIPT, scenario], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["bad"], [], "a prompt rendered undefined/NaN/[object Object]")
        return out

    def test_accepted_stack_is_returned(self):
        out = self.run_stage("accept")
        self.assertEqual(out["phases"], ["Baseline", "Sample", "Semantics", "Screen", "UnitSide",
                                         "ApplyBack", "Report"])
        ret = out["ret"]
        self.assertEqual(ret["status"], "applied")
        self.assertEqual(ret["final_overlay"], "/e/stack")
        self.assertEqual(ret["applied_count"], 1)
        self.assertEqual(ret["analysis_points"], {"decode": "de_16"})
        capture = out["prompts"]["fusion-trace-collector:capture"]
        self.assertIn("- ANALYSIS_PHASES: decode", capture)
        self.assertIn("- BASELINE_LOG: /e/baseline/server.log", capture)
        ab = out["prompts"]["fusion_integrator:combined_ab"]
        self.assertIn("- AB_REPEAT_MODE: replay_once", ab)
        self.assertIn("- AB_REPEATS: 1", ab)
        self.assertIn("kf_return.json", out["prompts"]["kernel-fusion:report"])

    def test_rejected_stack_hands_back_the_original(self):
        ret = self.run_stage("reject")["ret"]
        self.assertEqual(ret["status"], "combined_ab_failed")
        self.assertEqual(ret["final_overlay"], "")
        self.assertEqual(ret["accepted_env"], "E=1")

    def test_failed_baseline_stops_before_sampling(self):
        out = self.run_stage("baseline_fail")
        self.assertEqual(out["phases"], ["Baseline", "Report"])
        self.assertEqual(out["ret"]["status"], "baseline_failed")

    def test_a_stage_that_cannot_finish_is_not_started(self):
        out = self.run_stage("tight")
        self.assertNotIn("fusion-trace-collector:capture", out["labels"])
        self.assertTrue(any("not starting it" in line for line in out["logs"]))

    def test_fixed_shape_samples_without_a_log(self):
        out = self.run_stage("fixed_shape")
        capture = out["prompts"]["fusion-trace-collector:capture"]
        self.assertIn("- SYNTHETIC_SHAPE: 1024,1024,64", capture)
        self.assertNotIn("BASELINE_LOG", capture)
        self.assertIn("- AB_REPEAT_MODE: warm_server", out["prompts"]["fusion_integrator:combined_ab"])

    def test_an_earlier_eval_dir_resumes_at_semantics(self):
        out = self.run_stage("reuse")
        labels = out["labels"]
        self.assertIn("fusion-trace-collector:reuse_baseline", labels)
        self.assertIn("fusion-trace-collector:reuse_capture", labels)
        self.assertNotIn("fusion-trace-collector:baseline", labels)
        self.assertNotIn("fusion-trace-collector:capture", labels)
        self.assertEqual(out["ret"]["baseline_throughput_tok_s"], 799.42)
        self.assertEqual(out["ret"]["analysis_points"], {"decode": "de_5"})
        self.assertIn("- REUSE_EVAL_DIR: /old", out["prompts"]["fusion-trace-collector:reuse_baseline"])
        self.assertIn("/old/fusion_capture/profile_trace_manifest.json",
                      out["prompts"]["semantics-mapper:fusion"])
        self.assertEqual(out["ret"]["status"], "applied")

    def test_unsupported_backend_is_skipped_with_a_reason(self):
        out = self.run_stage("vllm")
        self.assertEqual(out["ret"]["status"], "skipped_backend")
        self.assertNotIn("fusion-trace-collector:capture", out["labels"])


if __name__ == "__main__":
    unittest.main()
