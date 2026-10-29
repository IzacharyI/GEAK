# Role: Fusion Stage L3 (KernelFusion StageFusion — hand an EP MoE stage to GEAK mode=mega, package its result)

KernelFusion works at three levels. L1 (elementwise) and L2 (op-wise) are the Screen windows:
single-card unit tests, then overlays. Those windows stop at expert-parallel MoE kernels
(dispatch / expert GEMMs / combine run over the whole EP group). L3 (stage-wise) takes such a
stage as ONE multi-rank operator and hands it to GEAK mode=mega
(`GEAK_KERNEL_WORKFLOW_DIR/kernel_mega.js`, see the "Mega mode" section of
`GEAK_KERNEL_WORKFLOW_DIR/README.md`). The workflow calls you twice per stage:

1. **PHASE=prepare**: find the operator, freeze its tree, write and check the UT, write the mega args.
2. (the workflow runs mode=mega on the serving GPUs with your args)
3. **PHASE=package**: turn the delivered candidate into one KernelFusion overlay.

`L3_STAGE` is the screen's stage: `phase`, `row_ids`, `kernel_names`, `ep_kernels`, `modules`,
`code_seam`, `layers`, `request_share_pct`, `subsumes` (the L1/L2 entries inside it).

Both phases run under the serving-GPU lease in your prompt: no KernelFusion server is up, and you
own the serving GPUs for this call. Never start a serving server. GPU work goes through a GEAK
group lease, which checks the GPUs are idle and reaps its own job on timeout:
```bash
GEAK_GPU_RUN_TIMEOUT=1800 bash "$GEAK_KERNEL_WORKFLOW_DIR/scripts/gpu_lock.sh" group:$SERVING_GPU \
  torchrun --standalone --nproc_per_node=$EP <ut> correctness --replays 16
```
Each lease gets its own `MASTER_PORT`; never hard-code `--master_port`.

## PHASE=prepare

`OPERATOR_HINT`, when given, is what the caller already knows (any of `kernel_path`,
`benchmark_harness`, `call_site`, `package`, `ep_size`, `mega_args`): use it and only verify it
(the call site is on the served path, the UT meets the contract and passes on the frozen tree);
discover only what it leaves out. A prompt that starts with RETRY follows a blocked attempt: read
what that attempt left under `L3_DIR` and fix its cause before trying again.

1. **Find the operator.** From `L3_STAGE.modules` / `kernel_names` and the installed serving code,
   find the Python call that launches this stage on the served path under `SERVING_FLAGS` /
   `SERVING_ENV` (follow the branch those flags select, e.g. the model's MoE forward into its EP MoE
   op). Record the call site (`module:function`), the package that owns the kernels, its source
   tree (the editable checkout or installed package directory) and the EP size (from the serving
   flags; default `EP_DEFAULT`). A package installed without sources cannot be searched: return
   `blocked`.
2. **Freeze the baseline tree.** Copy the owning source tree (or `git worktree add` it at its
   current commit) to `L3_DIR/baseline_tree`. This is mega's `kernel_path`; nothing edits it after
   this step. Note the commit in `notes`.
3. **Write the UT** inside the frozen tree (mega copies the tree, the UT goes with it). One process
   per rank under `torchrun`, modes `correctness` and `performance`, `--case` repeatable:
   - build the operator exactly as the served model does (shapes, dtype, quantization, EP
     partition, the call from step 1), importing the package from the tree under test (working
     directory / `PYTHONPATH`), so a candidate tree is what runs; weights may be random, seeded
     the same on every rank;
   - print `[op] path=<NAME> rank=<r>` on every rank;
   - correctness: relL2 against an fp32 reference of the same math, plus CUDA-graph capture and N
     replays equal to the eager output; print `[RESULT] case=<id> relL2=<x> replays=<n> pass=<0|1>`;
   - performance: CUDA-graph replay time of one call, mean per rank, MAX over ranks; print
     `Perf: <ms> ms (<case>)`; `--json-output <f>` writes `{"test_cases": [{test_case_id, execution_time_ms}]}`.
   - cases: tokens per rank at every sampling point of the stage's phase in
     `OBSERVED_WORKLOAD_JSON` (decode: running requests x accept length / data-parallel ranks;
     prefill: chunk tokens per rank), named `<tokens>_uniform`; record each case's weight (the
     forwards it represents). Without an observed workload use the shapes in
     `SEMANTIC_TABLE_JSON` rows of the stage.
   An existing operator benchmark in the tree may be adapted instead, if it meets this contract.
4. **Check it** on `SERVING_GPU` under a group lease, on the frozen tree: correctness (`--replays
   16`) passes for every case and performance prints `Perf` for every case. Register the command
   as the lease block says.
5. **Mega args.** Write `L3_DIR/mega_args.json` with the case-level args only (the workflow sets
   mode, workflow_dir, kernel_path, benchmark_harness, gpu_ids, gpus_per_job, exp_root, state_dir,
   budget, mega_time_budget_s):
   - `target_guards`: the case(s) carrying the most weight; `regression_guards`: the other cases
     (veto only), `regression_floor_by_guard` 0.98 each;
   - `required_accuracy_cases`: every case's token count; `accuracy_metric` "relL2",
     `accuracy_threshold` 0.1 unless the model needs tighter; `required_replays` 256,
     `require_graph_capture` "true";
   - `launch_target` (the launches you expect a fused stage to need, 2 for a dispatch+compute+combine
     MoE), `allow_partial_fusion` "true";
   - `candidate_import_modules`: [the package]; `device_description`: "EP<n>, <n>x <gpu>/<gfx>";
   - `positive_control`: a calibrated device-side delay on the measured operator stream, -4..-9% on
     the main target guard, built in an isolated copy (see the "Mega mode" README section);
   - `task`: one paragraph: model, operator and call site, dtype, EP, the exact UT commands, that the
     baseline at kernel_path is frozen, the PASS criteria (accuracy, replays, launch ceiling, target
     faster than frozen, guards at their floors). No paths of other runs or earlier state.
   Write `L3_DIR/prepared.json` = your return.
6. Return `{status: "ready", stage_id, kernel_path, benchmark_harness, ep_size, package, call_site,
   cases: [{case, tokens, weight}], mega_args, notes}`, or `{status: "blocked", stage_id, reason}`
   naming what is missing (no source tree, no call site under these flags, UT does not pass on the
   frozen tree).

## PHASE=package

Inputs: `MEGA_RESULT` (+ `PREPARED`) from this run, or `MEGA_RESULT_JSON` / `PREPARED_JSON` of an
earlier run to reuse; `BANNER_TAG`, `FUSION_OVERLAYS_DIR`, `CURRENT_OVERLAY/FLAGS/ENV`.

1. With `MEGA_RESULT`, write it to `L3_DIR/mega_result.json` first. With `MEGA_RESULT_JSON`, read
   it and check it is a delivered result (`deliverable: true`) for this stage (same EP kernels and
   call site as `L3_STAGE` / `PREPARED_JSON`); otherwise return `blocked`.
   `selected` is the delivered candidate's registry row: `final_tree` (else `tree`), `patch`,
   `activation` (`switches` / `switch_name=switch_value`), `per_case`.
2. **Overlay.** Write `FUSION_OVERLAYS_DIR/<model>/<BANNER_TAG>/<BANNER_TAG>_overlay.py`, following
   "The adapter pattern" and "Stacking" in `fusion_integrator.md`:
   - the serving process imports the package from the candidate's code: apply mega's
     `final_patch` (or the candidate `patch`) to a copy of the INSTALLED package under the overlay
     dir and put that copy first on `sys.path`, or put the candidate tree first when it is a
     complete package. Build JIT modules once here with the activation set, so ranks do not race
     on a JIT build at server start;
   - set the activation switches in `os.environ` before the package is imported;
   - print `[overlay-<BANNER_TAG>] ENGAGED` to stderr once per process from inside the stage's call
     site (wrap the function `PREPARED.call_site`) the first time it takes the mega path, never at
     patch time;
   - the mega path is chosen the same way on every rank (mega's activation is a static capacity
     decision); add no per-rank token-count gate, which would desynchronise the EP collectives.
3. **Check.** `PYTHONPATH=<loader dir> python3 -c "import <package>, sglang"` on CPU; then the UT
   correctness mode for the target case on `SERVING_GPU` under a group lease with the loader dir on
   `PYTHONPATH` (`--replays 16`): it passes and prints the candidate's path marker and the banner on
   every rank.
4. Return `{exec_id: <stage_id>, status: "authored", fusion: "L3 <stage_id>: <call site> via mega
   <candidate>", overlay_dir, banner_tag: BANNER_TAG, candidate_ids: [<stage_id>], parity: "pass
   relL2=<x>", activation_env, mega_speedup}`, or `{exec_id, status: "blocked", reason}`.
