# KernelFusion — standalone stage before the GEAK e2e workflow

KernelFusion runs to completion first; the unchanged GEAK e2e workflow then starts from whatever
fused stack it accepted.

```
interface/run_kf_e2e.py <handoff.json> <result.json> [--timeout-s N] [--kf-budget-s N]
                        [--skip-kf] [--kf-only] [--dry-run]
```

1. **KernelFusion stage** — `kernel_fusion/kf_workflow.js`, run through the same Workflow
   machinery as `run_e2e.py` (Claude SDK / CLI / standalone runtime), budget
   `--kf-budget-s` (default 12 h, capped at 2/3 of the run):

   | Phase | What happens | Owner |
   |---|---|---|
   | Baseline | ONE measured leg on the deployed stack. AgentX: the CI-faithful replay (parity window, the client's own warm-up, no outer warm-up round). | `roles/fusion_trace_collector.md` PHASE=baseline |
   | Sample | The kernel-extract skill's **launch + capture** method only (`knowledge/kernel_extraction_skills/kernel-extract/SKILL.md` §2, `references/dynamic-workloads.md`, `references/capture.md`): baseline server log → `fusion_observed_workload.py` → `plan_sampling_points.py` → one owned probe server (`fusion_serving_session.sh`) → `fusion_serving_probe.py` per point (steady state, arm once, `record_shapes` + `with_stack`, verify) → assemble. | PHASE=capture, `scripts/fusion_agentx_sampling.py` |
   | (analysis trace) | Per analysis phase (default **decode only**), the **top-1 point by GPU-time mass** = represented forwards × the per-step device time its own verified trace measured. Only that trace is analysed. | `fusion_agentx_sampling.py assemble` |
   | Semantics | Pattern/Phase/Layer/Kernel tables from the analysis trace; decode boundaries/shapes from the graph-construction replay. `GEAK_SEMANTICS_REQUIRE_PHASES` = the analysis phases. | `roles/semantics_mapper.md` |
   | Screen | `fusion_screen.py` → Top-K execution list (+ candidate pool). | `roles/kernel_fusion_analyst.md` |
   | UnitSide | Every Top-K entry / rung / pool item unit-tested. **Weighted**: the gate is `sum(ref_ms·calls)/sum(cand_ms·calls)` over EVERY sampling point of the phases a candidate runs in (`fusion_observed_workload.json`, `effective_calls` from the baseline log). | `roles/fusion_unit_validator.md`, `fusion_unitside_harness.py --observed-workload` |
   | ApplyBack | Author every unit-side pass as an overlay, stack them, gate the stack with ONE A/B (`fusion_combined_ab.sh`: one base server vs one stacked server, each a `fusion_serving_session.sh` server measured by `bench_e2e.sh REUSE_SERVER=1`, then gsm8k on the same server). AgentX: one replay leg per side (`AB_REPEAT_MODE=replay_once`). | `roles/fusion_integrator.md` |
   | Report | `kf_return.json` (+ `workflow_return.json`), `KF_REPORT.md`, `00_INDEX.md`. | reporter agent |

2. **GEAK e2e workflow** — `interface/run_e2e.py`, unmodified, in a subprocess. When the stage
   accepted a stack, it gets `handoff.after_kernel_fusion.json`: schema v2 with the fused loader as
   `baseline_env_spec.overlay_pythonpath` (+ the stage's accepted flags/env), so the e2e workflow
   measures its own baseline on the fused stack. Otherwise it gets the original handoff verbatim.

3. **result.json** — the e2e workflow's result plus a `kernel_fusion` section: the stage's own A/B
   (`delta_pct`), and `combined_speedup_vs_original = (1 + delta_pct/100) × throughput_speedup`
   (each factor a same-protocol A/B).

## Scope and separation

* Everything outside the files listed in `MANIFEST.txt` is byte-identical to upstream
  `AMD-AGI/GEAK` main; `check_main_parity.sh <upstream-ref>` verifies it. KernelFusion never edits a
  shared script: where it needed different behaviour it owns a copy or a wrapper
  (`fusion_serving_probe.py`, `fusion_observed_workload.py`, `fusion_gsm8k_eval.py`,
  `fusion_serving_session.sh`, `fusion_leg_wait.sh`) and its own card index
  (`knowledge/learned/FUSION_INDEX.md`).
* SGLang only. The probe drives SGLang's `/generate`, `/metrics`, `/start_profile`, and the log
  parser reads the SGLang scheduler log; any other backend is skipped with `status: skipped_backend`.
* A fixed ISL/OSL/CONC workload uses the same path: `--synthetic ISL,OSL,CONC` replaces the log.

## Handoff knobs

Any top-level `fusion_*`, `semantics_*` or `kf_*` key (or the same keys under `"kernel_fusion": {}`)
is forwarded to the stage. The useful ones:

| key | default | meaning |
|---|---|---|
| `kf_budget_s` | 43200 | stage wall-clock budget (also `--kf-budget-s`, `GEAK_KF_BUDGET_S`) |
| `fusion_analysis_phases` | `decode` | phases whose top-1 GPU-time point is analysed (`prefill,decode` adds the heaviest prefill point) |
| `fusion_sampling_points` | 6 | sampling points planned from the log |
| `fusion_ab_repeat_mode` | `replay_once` (AgentX) / `warm_server` | A/B lifecycle per side |
| `fusion_noise_band_pct` | 1.0 (AgentX) / 0.5 | combined A/B noise band |
| `fusion_applyback_mode` | `combined` | `serial` = one A/B per execution-list entry |
| `fusion_top_k`, `fusion_budget`, `fusion_pool_max`, `fusion_unit_parallel` | 8, 10, 8, 8 | screen / apply-back sizing |
| `semantics_shape_capture` | `true` | graph-construction Shape replay for decode boundaries |
| `semantics_shape_capture_setup` | host default | overrides for that replay's setup, key by key |

## Tests

```
python3 -m pytest e2e_workflow/scripts/tests/test_fusion_*.py e2e_workflow/scripts/tests/test_semantic_*.py \
    interface/test_run_kf_e2e.py
GEAK_NODE_BIN=/path/to/node python3 -m pytest e2e_workflow/scripts/tests/test_fusion_capture_manifest_contract.py
```
The last one also runs `kf_workflow.js` under a mock Workflow runtime (`tests/kf_workflow_mockrun.mjs`)
for the accept / reject / failed-baseline / out-of-time / fixed-shape / unsupported-backend paths.
