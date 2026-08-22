# kernel_workflow — Dynamic Workflow for Kernel / Model Inference Optimization

A deterministic **Workflow** (JS-orchestrated multi-agent pipeline) that optimizes the inference
speed of a GPU kernel directory — a single kernel, several kernels fused together, or an end-to-end
vLLM / SGLang model — on AMD Instinct MI-series accelerators (MI300X / MI300A / MI308X / MI325X on
CDNA3 gfx942, and MI350X / MI355X on CDNA4 gfx950 — the card is detected on-box, not assumed). The
budget loop, round fan-out, and verification are **JS control flow**, while every judgement call is made
by an agent returning **structured JSON**.

## Key properties
1. **Deterministic orchestration** — the budget loop / parallelism / verification live in
   `kernel_workflow.js`, not in LLM-interpreted prose. The TechLead returns structured decisions.
2. **Independent verification of every claimed speedup** — each engineer's patch is re-benchmarked
   by a separate `verify_engineer` in a clean workspace *as soon as it finishes* (pipelined). The
   script trusts only verified, absolute-latency numbers → the winner is genuinely the fastest.
3. **Specialist engineers (A)** — `algorithm | memory | compute | host_runtime`; each loads only its
   relevant knowledge → focused context, sharper results, naturally orthogonal & mergeable. Plus a
   fifth **`deep_explore`** track: an open-ended deep optimizer the TechLead hands a high target (Nx
   and/or ~90% roofline) with minimal steering — broad authority (kernel + wrapper + binding), its own
   long measure→self-profile→rewrite loop. It costs `deep_cost` budget (default 2) and always runs in a
   dedicated round on its own (its ground-up rewrite isn't expected to merge with specialist patches).
4. **Host/Runtime as a first-class track (B)** — attacks the wall-clock floor (dispatch collapse,
   native layouts, CUDA graph, wrapper overhead). This is where the last 1.5–3x of geomean lives.
5. **Cross-round memory (C)** — an insight blackboard + hypothesis ledger threads what was learned
   into the next round's engineer prompts; dead-ends are not retried.
6. **Integrator (E)** — combines the round's winning ideas (stack compatible patches OR hand-merge
   conflicting ones into a coherent best implementation). Does not consume budget.
7. **Director arbitration (H)** — independently validates the final patch against the TRUE original
   baseline and can flag / request a corrective round.

## Roles → workflow mapping
- **Director** = the script's orchestration + a setup agent + a final validation/arbitration agent.
- **TechLead** = agent for analyze/roadmap, per-round planning (orthogonal directions + stop), the
  cross-round memory, and the final report.
- **Engineers** = parallel specialist agents (optimize), plus `benchmark_engineer`, `profile_engineer`,
  `verify_engineer`, and `integrator`.

## Pipeline
`Setup → Analyze+Roadmap → Benchmark(COMMANDMENT+baseline) → Baseline Profile →`
`LOOP[ Plan round → (Optimize ‖ Verify, pipelined) → Integrate → Commit winner → Re-profile → Update memory ] →`
`Final Report → Director Validation`.

Each round's winner is committed into the canonical workspace, so the next round builds on the
cumulative best. Speedup is always measured in **absolute latency vs the true baseline**:
`geomean( baseline_ms / optimized_ms )`.

## Budget
`budget` = the **total number of optimization directions** the TechLead may dispatch to engineers
across all rounds. Only optimization-direction engineers count; benchmark / profile / verify /
integrate / commit / validate do **not** consume budget. The script hard-caps each round to the
remaining budget; the TechLead may also stop early (`stop=true`) when further directions won't pay.
Example (budget=6): round 1 = 3 directions, round 2 = 3; or 4 then 2; or stop after 4.

## Invocation

### Packaged tasks — start here if one fits
`tasks/<name>/` holds a task the workflow already knows how to stand up: the task statement, and a
`launch_args.json` recipe with a positive control that does not presuppose the answer. Both are
templates; `scripts/bootstrap_task.sh` resolves them against the local machine.

```
bash scripts/bootstrap_task.sh --check            # is this box even capable of the task?
bash scripts/bootstrap_task.sh \
  --baseline /path/to/frozen/baseline-checkout \
  --out      /path/to/new/workspace
```

It probes for the hardware and runtime libraries the task needs, refuses to assemble a workspace the
machine cannot run (exit 2), distinguishes that from "fit, but a co-tenant holds the VRAM right now"
(exit 3 — assemble and wait), copies the baseline into a workspace an engineer may freely edit, and
writes a `launch_args.json` with every path resolved. What it deliberately does **not** take is a
previous run's artifacts: analyses, logs and accumulated patches are the workflow's *output*, and a
task that needs them as input is a task that has been solved elsewhere and is now being replayed.

Available: `tasks/megamoe_v2_ep8` (MegaMoE V2, 8-rank expert-parallel fusion + compute/comm overlap).

### Validating a change to the workflow itself
`scripts/replay_runs.js` re-decides finished runs with today's decision logic, on no GPU:

```
node scripts/replay_runs.js --runs <exp_root> --snapshot before.txt   # record
#   ... edit kernel_workflow.js ...
node scripts/replay_runs.js --runs <exp_root> --check before.txt      # exit 4 = a verdict flipped
```

It reads the raw per-invocation records in each run's `setup_ab*.json` (arm, guard, rep, rank-max,
exit code, path markers), recomputes the paired effect under the measurement discipline (within-rep,
rank-max, VOID arms dropping their whole pair, control arms matched by workspace rather than to the
run's declared base), and pushes the result through the gate arithmetic **lifted verbatim** from
`kernel_workflow.js` between its `<<REPLAY:pc_gate>>` markers — so the replay cannot drift from the
code it replays. It does not replay agent judgement; that layer is recorded, not reproducible.

Use it before committing any change to the gates. A `--check` that stays clean means the change is
inert or aimed outside the corpus, and it is worth knowing which.

### Direct invocation
This is a Workflow, run via the `Workflow` tool with `scriptPath` and `args`. **No paths are
hard-coded in the script** — it is portable to any install location. Set `scriptPath` to wherever
this folder lives and pass that same folder as `args.workflow_dir`:

> **IMPORTANT:** pass `args` as a real JSON **object** (a mapping), **not** as a JSON-encoded
> string. Do not wrap it in quotes or `json.dumps()` it. If `args` arrives as a string the
> workflow cannot read `args.workflow_dir` / `args.kernel_path` and aborts immediately.

```
Workflow({
  scriptPath: "<WF_DIR>/kernel_workflow.js",   // <WF_DIR> = absolute path to THIS kernel_workflow/ folder
  args: {
    kernel_path: "/abs/path/to/kernel_or_model_dir",  // REQUIRED
    workflow_dir: "<WF_DIR>",  // REQUIRED: same folder as scriptPath (holds roles/ knowledge/ scripts/);
                               //           a JS workflow can't read its own path, so the caller passes it
    budget: 6,                 // optional, default 6
    min_improve: 0.02,         // optional, default 0.02 (2%): min verified geomean gain over the
                               //           cumulative best for a round winner to be committed
    deep_cost: 2,              // optional, default 2: budget cost of one deep_explore direction
                               //           (heavyweight; always runs in its own dedicated round)
    gpu_ids: "0",              // optional GPU pool, comma-separated, default "0"
    gpus_per_job: 1,           // optional, default 1; >1 atomically leases a group from gpu_ids
    job_gpu_ids: "",           // optional fixed group; when set, its size must match gpus_per_job
    task: "focus on ...",      // optional natural-language steer
    exp_root: "",              // optional, output root; default = sibling "exp/" next to workflow_dir
    eval_dir: "",              // optional, override the output dir for this single run
    apply_to_original: "false",// optional; if "true", write the validated patch back to kernel_path
    // --- author mode (write a fresh implementation from scratch, then optimize it) ---
    mode: "optimize",          // optional: "optimize" (default, edit an existing kernel) | "author"
    target_language: "triton", // author mode: triton (always) | flydsl | hip | ck — the language to write
    op_spec: {},               // author mode: op contract; may include resource.gpus_per_job/job_gpu_ids
    perf_knowledge_dir: "",  // optional: AMD authoring knowledge base the author_engineer reads
    analysis_skill: "none",  // optional, default none; e.g. "moe_bottleneck" dispatches a separate
                              //   analysis_engineer after each successful generic Profile
    // --- workload alignment (optional; aligns the PERF harness with the real workload) ---
    workload_spec_path: "",    // optional: path to a workload-v1 json (parse_profile.py --workload-out).
                               //   The benchmark harness then times the EXACT (shape,dtype) cases the
                               //   workload hits, weighted by each case's total-time contribution, and
                               //   the PRIMARY metric becomes the time-weighted ratio-of-sums (the
                               //   unweighted geomean is kept as a secondary diagnostic). Correctness is
                               //   unaffected (it stays on the frozen reference_io.pt oracle).
                               //   Also accepted as op_spec.workload_path, or op_spec.workload (inline).
  }
})
```

### Fixed multi-GPU command lease

`gpu_lock.sh` preserves its historical single-GPU interface:

```bash
bash scripts/gpu_lock.sh 0 <command...>
```

For a command whose own harness launches a fixed multi-GPU job (for example an 8-rank MegaMoE
`torchrun`), use the opt-in group interface:

```bash
bash scripts/gpu_lock.sh \
  --group 0,1,2,3,4,5,6,7 \
  --wait-timeout 1200 \
  --run-timeout 900 \
  -- \
  torchrun --standalone --nproc_per_node=8 unittest.py
```

Dynamic N-GPU selection from a pool uses:

```bash
bash scripts/gpu_lock.sh \
  --pool 0,1,2,3,4,5,6,7 \
  --count 2 \
  -- \
  python3 unittest.py
```

Workflow phases pass the same requests in compact form:

```text
group:0,1,2,3,4,5,6,7
pool:2:0,1,2,3,4,5,6,7
```

The wrapper atomically locks every per-GPU lock, checks the selected devices are idle when sysfs
telemetry is available, exposes the full group through `HIP_VISIBLE_DEVICES` /
`CUDA_VISIBLE_DEVICES`, and terminates the command's process group on timeout or signal. It never
holds a partial GPU group while waiting. Logical GPU IDs are mapped through `amd-smi list --json`
to PCI sysfs before the idle check; group mode fails closed when that mapping or the busy metric is
unavailable.

Fixed, dynamic, and historical numeric single-GPU requests all use the same lease manager. Pending
requests are FIFO-ordered when their pools overlap, so repeated one-card backfill cannot starve an
older N-card request. The launched command inherits the lock descriptors, keeping the reservation
held even if the lease-manager process is killed.

Configuration:

- `GEAK_GPU_LOCK_DIR` — shared lock directory (default `/tmp/team_gpu_locks`; containers must share it).
- `GEAK_GPU_WAIT_TIMEOUT` — default allocation wait timeout (default `1200` seconds).
- `GEAK_GPU_RUN_TIMEOUT` — command timeout; numeric legacy mode defaults to no timeout.
- `GEAK_GPU_REQUIRE_IDLE=0` — disable the best-effort sysfs idle gate (group mode defaults on).
- `GEAK_GPU_MAX_BUSY_PCT` — maximum accepted busy percentage (default `5`).
- `GEAK_GPU_MAX_VRAM_MB` — optional VRAM-used ceiling; negative disables it (default `-1` because
  physical MI350 VRAM accounting can include partition reservations).
- `GEAK_GPU_SYSFS_ROOT` — override DRM sysfs root for testing or non-standard systems.

When multiple containers share the lock directory, they must use the same logical GPU enumeration.
The fixed EP8 group locks every logical GPU and is therefore safe for the current MegaMoE scope;
future partial/dynamic groups should key allocation by stable BDF/UUID.

This is the execution primitive only. Passing one stable GPU group through Author, Benchmark,
Profile, Engineer, Verify, Integrator, and Validate is enabled by the workflow resource arguments:

```text
# Fixed EP8 group (recommended for MegaMoE)
gpu_ids="0,1,2,3,4,5,6,7"
job_gpu_ids="0,1,2,3,4,5,6,7"

# Dynamic two-GPU group selected from a pool
gpu_ids="0,1,2,3,4,5,6,7"
gpus_per_job=2
```

With neither new argument set, `gpus_per_job=1` and the historical round-robin single-GPU behavior
is unchanged. A fixed/dynamic group request is propagated through Author, Benchmark, Profile,
Engineer, Verify, Integrator, re-profile, and final Validate.

An immutable task can declare the same requirement in `op_spec.resource`:

```json
{
  "resource": {
    "gpus_per_job": 8,
    "job_gpu_ids": "0,1,2,3,4,5,6,7"
  }
}
```

Direct workflow args override task metadata.

Dynamic pool selection assumes candidate GPUs are performance/topology-equivalent. Use
`job_gpu_ids` for topology-sensitive benchmarks such as the current fixed EP8 MegaMoE target.

### Workload alignment (NEW)
By default the harness benchmarks small/medium/large cases unweighted. Pass a **workload spec** to
instead benchmark the shapes/dtypes the kernel actually sees in production, weighted by how much
wall-clock each contributes (`weight = call_count × baseline_latency`). Generate one from a profiler
trace with `python3 e2e_workflow/scripts/parse_profile.py --torch-trace <trace> --workload-out
workload.json [--target <kernel_name>]`, then pass `workload_spec_path: ".../workload.json"`. The
optimization target becomes the **time-weighted ratio-of-sums**
`Σ count·baseline / Σ count·optimized` (true wall-clock speedup of the kernel's total workload
contribution); the unweighted geomean is still reported. The perf **baseline is the original/extracted
implementation**, never an LLM naive reimplementation. When invoked from the e2e layer this is wired
automatically (profiler → extractor → `op_spec.workload_path`).

### Author mode (NEW)
`mode="author"` is for when there is **no existing source to optimize** — a hot op (e.g. a library
GEMM/attention) needs a fresh implementation. Here `kernel_path` is an **op task dir** holding the
IMMUTABLE oracle (`meta.json` + `unittest.py` + optional `reference_io.pt`). The `author_engineer`
writes the simplest correct implementation in `target_language` (correctness-judged against the
oracle), commits it as the baseline, and then the **same optimize loop** improves it. Returns
`authored:false` / `validation_status:"author_failed"` if no correct baseline can be produced (the
caller drops that language). `mode="optimize"` (default) is unchanged and fully backward compatible.

`<WF_DIR>` is the only location-specific value and it is supplied at call time (it is just the
dirname of `scriptPath`). Everything else is derived: `exp_root` defaults to `<parent of WF_DIR>/exp`.

The user-facing prompt stays minimal & generic, e.g.:
- `optimize /xxx/xxx/knn`
- `optimize /xxx/xxx/knn, budget 6, focus on wrapper overhead`
These map to `kernel_path` (+ optional `budget` / `task`). No repo URL needed.

## Output
Everything lands under `<exp_root>/team_<kernel>_<timestamp>/<kernel>/` (default `exp_root` =
the `exp/` folder sibling to `workflow_dir`):
- `COMMANDMENT.md`, `baseline_timing.json`, `analysis.json`, `codebase_context.md`, `roadmap.md`
- `baseline_metrics.json`, `profiling_summary.md`
- `round_N/engineer_i/{worker_result.json, report.md, best_patch.diff}` — each engineer's mini-report
- `round_N/integrate/`, `insight_log.md`, `current_best.diff`
- `tech_lead_report.md` — round-by-round narrative + final per-case table (the TechLead summary)
- `final_patch.diff`, `optimized/`, `director_validation.json` — the official verified result

## Generality (single kernel ↔ e2e model)
The script never branches on kernel type or single-vs-e2e. Everything flows through the
**COMMANDMENT** discovered/built at the Benchmark phase (setup / correctness / benchmark / profile
commands + a parse hint). For a vLLM/SGLang model the only difference is what those commands contain
(launch the server, run a throughput/latency benchmark, define output-parity correctness); the
Director/TechLead/Engineer orchestration is identical.

## Files
```
kernel_workflow.js     orchestration (deterministic)
roles/               director, tech_lead, engineer, deep_engineer (deep_explore),
                     author_engineer, benchmark_engineer, profile_engineer,
                     verify_engineer, integrator
knowledge/           optimization_strategies, hip/triton/wrapper, profiling_guide,
                     amd_instinct (multi-card: gfx942/gfx950), self_monitoring, geomean_levers,
                     distributed_fusion (multi-launch multi-rank -> one persistent kernel),
                     jit_arm_isolation (making an env-gated A/B compile to two binaries)
tasks/               packaged tasks: GEAK_TASK.md + launch_args.json, both templated
scripts/             gpu_lock.sh, gpu_group_lock.sh, gpu_lease.py, profile_kernel.sh,
                     bootstrap_task.sh (stand a packaged task up on a new machine),
                     replay_runs.js (re-decide finished runs with today's logic, no GPU),
                     reference_leak_sweep.sh, skill_address_scan.sh (capability-eval containment:
                     git addresses AND filesystem paths that open on this machine)
tests/               node tests/*.js -- run from THIS directory, not from the repo root
```
