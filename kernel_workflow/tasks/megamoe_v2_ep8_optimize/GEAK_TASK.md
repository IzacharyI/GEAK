# GEAK task: MegaMoE V2, EP8 intra-node — fuse the operator and overlap compute with communication

This directory is a self-contained snapshot of public AITER at `8775229e0`. It runs entirely from
its own path — every command below sets `PYTHONPATH="$PWD"` so the copy you are editing IS the
`aiter` that gets imported. Verified: a detached copy of this tree passes the EP8 UT unmodified.

> **This file is a template.** `scripts/bootstrap_task.sh` materialises it next to a baseline AITER
> checkout, resolving the two machine-local paths (the MORI checkout and the AITER JIT cache) as it
> copies. If you are reading unsubstituted `${...}` placeholders below, the workspace was assembled
> by hand and those are the values you must supply — see "Environment prerequisites".

## Environment prerequisites

This task cannot run on an arbitrary box, and the failure mode of a wrong environment is a hang or an
abort inside the communication runtime rather than a clear error. Required:

| requirement | why | how to check |
|---|---|---|
| 8 × gfx950 (MI355X) on one node, XGMI | the op is an 8-rank intra-node collective; EP8 is baked into the guards | `rocm-smi --showproductname` |
| **MORI** checkout at `${MORI_ROOT}` | the symmetric heap and `mori_shmem` primitives the kernels call | `python -c "import mori"` with it on `PYTHONPATH` |
| ≥150 GiB free VRAM per card at launch | the heap needs ~62 GiB/card at t=0, but that is a floor, not a window — an arm that starts with just enough dies if a co-tenant grows | `/sys/class/drm/card*/device/mem_info_vram_{total,used}` |
| a prebuilt AITER JIT cache at `${AITER_JIT_DIR}` | pointing this at a fresh directory triggers a full C++ rebuild inside your first lease | directory exists and is populated |

`bootstrap_task.sh --check` verifies all four and refuses to assemble a workspace that cannot run.

## The op

`MegaMoEV2` — a fused Mixture-of-Experts forward for an 8-rank expert-parallel group on one node
(8× MI355X, gfx950, XGMI). One iteration is currently **four kernel launches per rank**:

```
quant  →  Stage1 (dispatch + GEMM1)  →  Stage2 (GEMM2 + P2P publish)  →  Combine
```

Entry point / launch orchestration: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py`
Kernel bodies:
- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`  (dispatch + GEMM1; the largest stage)
- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py`  (GEMM2 + `p2p_scatter_epilog`)
- `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py`  (combine)
- `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py`  (combine **host-op wrapper**:
  combine arg-tuple assembly + `flyc.compile`, the producer→consumer arg-threading boundary. This is
  on the MegaMoEV2 path and MUST be modifiable — without it a fused `path=MEGA` is unauthorable
  because the switched path's args cannot be threaded from host to the fused device entry.)
- `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`  (`DispatchSlot` symmetric-heap state, publish/wait helpers)
- `aiter/ops/flydsl/kernels/communication_ops_utils.py`  (fences, system/agent atomics). NOTE: this
  snapshot does **not** define a `read_memrealtime` helper — an earlier revision of this file claimed
  it did, and a run lost time discovering otherwise. If you want a cycle counter for in-kernel
  timing, you must add the `s_memrealtime` inline-asm helper yourself.

These are FlyDSL kernels (Python-authored, JIT-compiled), not Triton or HIP source.

## DIRECTIVE — D1 whole-fusion edit lane (authoritative; overrides any inferred file set)

The whole-fusion dimension (D1: "author whole-fused persistent megakernel", `path=MEGA`) MUST use
EXACTLY this file set as both its `focus_files` and its MODIFIABLE set. Every file below is on the
MegaMoEV2 path and is modifiable:

- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py`            (entry / launch orchestration)
- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py`       (dispatch + GEMM1)
- `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py`       (GEMM2 + p2p publish)
- `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py`  (combine device kernel)
- `aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py`      (combine host-op wrapper / arg-threading boundary)
- `aiter/ops/flydsl/kernels/communication_ops_utils.py`        (fences, atomics)

DO NOT target `moe_kernels.py` (or any `moe_kernels*`): it is a DEAD LANE, off the MegaMoEV2 path,
and produces no in-lane patch. Prior rounds repeatedly failed with `apply_failed` / "no candidate
patch exists to measure" because the inferred `focus_files` were disjoint from the MODIFIABLE set
(focus pointed at `moe_kernels.py` + a frozen wrapper, while the real fusion files were unfocused).
The intersection {focus} ∩ {modifiable} ∩ {needed} MUST be non-empty; the set above guarantees it.

## The optimization target

The four stages are **strictly serialized** — measured kernel-level overlap is zero.

That is the observation. *Why* they are serialized, which of the orderings between them are required
by the data and which are artifacts of how the code is written today, at what granularity a consumer
could begin, what enforces each ordering now, and where the critical path actually runs — the Analyze
phase must still work this out and defend each fused edge against it. This is an **optimize** run, not
a blind-derivation proof: you MAY use GEAK's accumulated MegaMoE knowledge — the
`megamoe_ep_persistent_fusion` expert skill (the distilled full-fusion production recipe) and the
`moe_bottleneck` analysis skill — and apply the known fusion recipe. A candidate still must be
justified by the task-graph analysis (a proposal that is a bare restatement of the launch count is
not), but it does not have to be reinvented from scratch.

The end state the acceptance bar requires is a **two-launch** shape: the `quant` ingress stays its
own launch, and the remaining three stages become **one persistent kernel per rank** with genuine
compute/communication overlap. That is the *goal*. The mechanism that gets there is the work.

Start from `SKILL_DIR/knowledge/tile_task_graph.md` — the Analyze phase must emit the tile-level
dependency graph as an artifact (nodes, edges, edge scope, what enforces each edge today, critical
path, slack) before any candidate is generated. Then `SKILL_DIR/knowledge/fusion_preconditions.md`,
which gives you the test each candidate edge has to pass and the cheaper levers you must rule out
first — including the conditions under which the answer is legitimately "this edge does not pay".
Then `SKILL_DIR/knowledge/resource_partition.md` for who gets the CUs once anything overlaps.

**Read `SKILL_DIR/knowledge/distributed_fusion.md` before proposing anything.** It carries the
correctness invariants (acquire fence pairing, residency, reset-free counters, acyclicity), the
measured anti-patterns, and the measurement discipline for this exact operator. Two things that
bound this task:

- **Launch count is not the objective.** Two launches is the acceptance shape, not the goal; a
  fused kernel that is slower than four launches has failed. If your graph says an edge does not
  pay, say so and defend it — that is a result, not a non-result.
- Any fused path you add behind an env var / config predicate **must print a one-line path marker**
  (e.g. `[megamoe] path=MEGA` vs `path=SCATTERED`) once per process. An opt-in path that fails its
  predicate falls back silently and produces a plausible wrong number. A benchmark log without the
  marker is void, not zero.
  Bind each A/B arm to its expected marker value (`expect_paths` in `tools/ab_retry.py`), not only
  the count: eight `SCATTERED` markers on an intended fused arm are eight proofs of fallback.

## Hard constraints

1. **Correctness**: `relL2 < 0.10` on BOTH the fixed-slot and compact routing paths. The frozen
   baseline's own `relL2` is **not a constant** — it is recomputed on freshly drawn inputs each run
   and has been observed anywhere in `0.053 .. 0.069` across both routing paths on this box, with
   `path=compact` (bs=512) usually reading lower than `path=fixed` (bs=128). Earlier revisions of
   this file quoted a single figure (`0.0691`); treat any such number as stale, and do not use one
   as a regression target. **Measure the baseline in the same run as the candidate** and compare
   those two. A candidate's `relL2` that lands inside the range above is indistinguishable from the
   baseline; the bound that matters is the hard `< 0.10`.
2. **Liveness**: the operator is captured into a CUDA Graph and replayed. A fused kernel with
   cross-rank waits can deadlock on replay N without failing on replay 1. Stress ≥1000 replays per
   route; a timeout is a FAILURE, never a skip.
3. **Residency**: any grid-wide wait requires all blocks co-resident. Blocks must not exceed the CU
   budget; `_check_block_num_resident` is the existing check.
4. Never modify anything outside this workspace.

## Acceptance (optimize run)

This task's launch template sets `capability_eval=false` and `strict_autonomy=false`. The frozen
public AITER tree is the optimization **starting point and the denominator**, not an answer to fence
off — read it freely. The workflow is expected to use its accumulated MegaMoE knowledge (the
`megamoe_ep_persistent_fusion` expert skill and the `moe_bottleneck` analysis skill). There is **no
derived-not-copied / byte-identity (criterion 5) requirement** in this mode.

The hand-authored **M2.5** full-fusion implementation is kept **out of the task tree** and is used
only as an offline oracle / performance bar — it is not fed into any engineer prompt, and the goal is
to **beat** it, not reproduce it byte-for-byte. Per the `megamoe_ep_persistent_fusion` skill, M2.5 is
treated as a **floor**.

A terminal candidate is accepted only when the same independently verified candidate has: at most two
launches per rank; a positive `8192_uniform` operator rank-max result versus the baseline (and, per
the persistent-fusion skill, at or above M2.5); no regression on the other three guards; numeric
`relL2 < 0.10` evidence; distinct JIT artifact hashes for the A/B arms (measurement integrity, not
containment); graph-safe liveness over at least 1000 replays; every mandatory arm; controlled,
non-zero on-edge overlap; and launch-change attribution. A working but slower/intermediate artifact is
**preserved for the next round** (objective=working_kernel) so the staged fusion chain can reach its
terminal step — it is not a failure, it is a rung.

Run `bootstrap_task.sh --task megamoe_v2_ep8_optimize` to assemble the workspace. No `MARKER_FILE`,
`--known-reference`, or containment preflight is needed in this mode.

## How to run it

Every GPU command takes the **whole 8-GPU group** — this is a collective, one job at a time. Use the
group form of the lease wrapper (not the numeric single-GPU form):

```bash
cd <workspace> && bash $SKILL_DIR/scripts/gpu_lock.sh \
  --group 0,1,2,3,4,5,6,7 --wait-timeout 1800 --run-timeout 3600 -- \
  env PYTHONPATH="$PWD:${MORI_ROOT}:${MORI_ROOT}/python" \
      AITER_JIT_DIR=${AITER_JIT_DIR} \
      MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G \
  timeout 50m torchrun --standalone --nproc_per_node=8 <script> <args>
```

`AITER_JIT_DIR` points at a **shared, prebuilt, read-only** module cache. Leave it as-is; do not
point it at a fresh directory (that triggers a full C++ rebuild) and do not write into it.

**Correctness** (`<script> <args>`):

```
op_tests/multigpu_tests/test_mega_moe_v2.py --network v4_pro --bs-list 128,512,8192 \
  --iters 10 --accuracy-max-bs 8192 --rtol 0.10
```
Prints one line per batch size:
`[MEGA-V2] bs=128 relL2=0.069100 path=fixed ... e2e=0.3821/0.3832ms mean/max`
`path=fixed` (bs=128), the small compact path (bs=512), and the large 8192 configuration used by
the target benchmark must all appear and have `relL2 < 0.10`. For a terminal fusion, force its
candidate path on the 8192 case and require the candidate path marker; a correctness pass obtained
through SCATTERED fallback is activation failure, not fusion correctness.

**Performance**:

```
op_tests/multigpu_tests/bench_mega_moe_v2.py --tokens <T> --route <R> --iters 10 --mega-only
```
Prints `[RESULT] route=... tokens=... mega_e2e=<rank-mean>/<rank-max>ms speedup=...`

**The metric is the rank-MAX** (the second number). A collective is gated by its slowest rank;
rank-mean can improve while the operator gets slower. Rank-mean is diagnostic only.

## Target and regression guards

| tokens | route            | role                                                   |
|--------|------------------|-------------------------------------------------------|
| 8192   | `uniform`        | **target** — the only route allowed to create credit   |
| 8192   | `rank-mixed-skew`| regression guard — preserve skew capability            |
| 512    | `uniform`        | regression guard — small-shape safety                  |
| 512    | `rank-mixed-skew`| regression guard — small + skew safety                 |

A final candidate must beat baseline rank-max on `8192_uniform` and be **at or above baseline on
all three regression guards**. Regression guards are vetoes, never positive weight: a skew-only win
cannot compensate for a uniform loss or become the current-best score.

## Profiling / multi-rank evidence — what this frozen tree can and cannot give you

`bench_mega_moe_v2.py --profile-dir <dir>` works and is the way to collect per-rank traces.

What is **not** available here, and must not be faked:

- The bench has **no `--json-output`**. That flag does not exist in this tree.
- The `[RESULT]` line prints **rank-mean/rank-max aggregates only** — `mega_e2e`, `stage1`,
  `stage2_combine` each as `mean/max`. There is **no per-rank breakdown on stdout**, so a
  `rank_report.json` with `cases[].ranks[]` cannot be produced by parsing it.

Consequence for the MoE bottleneck analysis: several of its measurement tracks will legitimately be
`awaiting_measurement` rather than `complete`, and the correct output is to **name the missing
collection experiment**, not to synthesize per-rank numbers from an aggregate. Naming what is
missing is a first-class result of that analysis; inventing a value to fill the contract is not.

Per-rank evidence that IS obtainable: the `--profile-dir` traces (one per rank), and any counter you
add to the kernel yourself and read back. A rank-max that is far above the rank-mean is itself
evidence of straggler gating, but it does not tell you WHICH rank or why.

## Measurement discipline (this operator specifically)

Batch-to-batch drift on the small guards is larger than the whole available gain, and on the 512
guards it is not even drift. Measured 2026-08-23, unmodified tree against itself, 10 runs per guard:
`8192_uniform` is unimodal with a worst pair of 1.09%; `512_uniform` worst pair 6.21%; `512_rank-
mixed-skew` worst pair 9.30%, with 2 of 10 runs sitting ~7-8% above an otherwise tight cluster. That
slow state is **bimodal, not noisy** — repeat readings land on the same value to four digits — and it
is invisible to both kernel timers, showing up only in the residual (end-to-end minus stage1 minus
stage2_combine). **Do not read that as proof of a host cost.** All three timers are independently
rank-reduced, so the residual is `max-of-sums minus sum-of-maxes` and is signed: its median over 68
records on this box is −9.6 µs. A large residual is therefore uninformative on its own; what carries
weight is a residual that moves *between paired arms* while both kernel timers hold still. The slow
state is real and currently unattributed. A 5-pair sample of `512_skew` the day before had read its
worst pair as 1.93% and missed the tail entirely. Therefore:

- Run baseline and candidate **alternately** (A,B,A,B,A,B) and report the **per-pair** delta plus the
  median. Never compare two independently collected medians. **At least 5 pairs on the 8192 guards
  and at least 10 on the 512 guards** — fewer than 10 there does not sample the slow state.
- Report the raw per-run readings for the 512 guards, not only the pair deltas. A ratio against a
  fat-tailed worst pair understates a real effect badly — but **do not reach for complete separation
  of the two arms here.** That is the right escape hatch on a unimodal guard and it is unreachable on
  a bimodal one: both arms draw slow runs, the ranges interleave, and no amount of extra sampling
  separates them. Condition on the state instead. Pool the readings from both arms *arm-blind*, split
  at the widest relative gap (a gap under ~3% means one state, so say so and analyse all pairs), then
  make the primary statistic the paired delta over pairs where **both** arms are fast, with `n_fast`
  and the sign agreement reported. Report how often each arm landed in the slow state as a result in
  its own right — the slow state costs 9-13% of a 512-token iteration, so a candidate that enters it
  less often has a bigger effect than the fast-mode delta will ever show, and dividing that out
  throws away the win. Disclose the mixed-mode and both-slow pairs you dropped.
- Size the 512 collection for the *usable* pairs. At the measured ~20% slow rate a pair survives with
  probability 0.64, so "10 pairs" has been yielding about six: collect **16** when you need 10.
- Report the per-rep spread alongside the median.
- The speedup denominator is this frozen tree, run under the identical command. Nothing else.
- **`stage1` and `stage2_combine` stop being comparable the moment anything overlaps, and they move
  the wrong way.** They are separate rank-reduced timers over stages that are serialized today. Once
  a stage is allowed to start early it charges its own waiting to itself, and concurrent stages
  contend, so **both stage timers can rise while the operator gets faster** — a successful overlap
  reads as a per-stage regression under a naive comparison, and `stage1 + stage2_combine` is not a
  cost at all once they run together. The number that stays comparable across this change is
  `mega_e2e` rank-max, which is what the guards are written against; keep the stage timers as
  *evidence of overlap* (a stage timer that inflates while `mega_e2e` falls is a positive signal),
  never as an objective, and never sum them. This also changes the residual's meaning — see above —
  so re-derive it rather than carrying an interpretation across the fusion.
- **The promotion metric is operator `mega_e2e` rank-max on the target guard.** Do not replace it
  with `stage1_max + stage2_combine_max`: those timers are captured in separate graphs and reduced
  independently across ranks, so their sum is not the time of either the scattered operator or the
  fused kernel. When a same-timeline changed-kernel measurement exists, report it in `attribution`
  to explain the mechanism; it does not override a real operator result in this task.
- A latency win with no measured change in overlap is **suspicious, not accepted** — find the
  mechanism or discard the result. That sentence was unenforceable for several waves because there
  was no instrument behind it: once the stages fuse there is one kernel record and latency is
  precisely the quantity that cannot tell overlap from removed launch overhead. **Read
  `SKILL_DIR/knowledge/overlap_instrument.md` before claiming overlap either way** — it says what to
  build, the two controls that decide whether the reading may be believed (the scattered path must
  read ~0 or the meter is void; forced concurrency must read high or the meter is dead), and the
  traps that have already produced plausible wrong numbers. `measured: "unknown"` is a first-class
  answer there; a synthesised fraction is not.
