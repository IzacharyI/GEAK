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
- `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`  (`DispatchSlot` symmetric-heap state, publish/wait helpers)
- `aiter/ops/flydsl/kernels/communication_ops_utils.py`  (fences, system/agent atomics). NOTE: this
  snapshot does **not** define a `read_memrealtime` helper — an earlier revision of this file claimed
  it did, and a run lost time discovering otherwise. If you want a cycle counter for in-kernel
  timing, you must add the `s_memrealtime` inline-asm helper yourself.

These are FlyDSL kernels (Python-authored, JIT-compiled), not Triton or HIP source.

## The optimization target

The four stages are **strictly serialized** — measured kernel-level overlap is zero.

That is the observation. It is not a diagnosis, and it is deliberately all you are given. *Why*
they are serialized, which of the orderings between them are required by the data and which are
artifacts of how the code is written today, at what granularity a consumer could begin, what
enforces each ordering now, and where the critical path actually runs — **you have to derive.**
Nobody has written it down for you, and an implementation proposal that is not derived from that
derivation will not be ranked.

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

## Hard constraints

1. **Correctness**: `relL2 < 0.10` on BOTH the fixed-slot and compact routing paths. The frozen
   baseline on this box measures `relL2 ≈ 0.060–0.062` at bs=128 (0.0604 fixed-slot / 0.0617
   compact, re-measured). An older revision of this file claimed `0.0691`; if you read that number
   anywhere, it is stale. Re-measure the baseline yourself in the same run as the candidate —
   a correctness bound compared against a number from a different build is not a comparison.
2. **Liveness**: the operator is captured into a CUDA Graph and replayed. A fused kernel with
   cross-rank waits can deadlock on replay N without failing on replay 1. Stress ≥1000 replays per
   route; a timeout is a FAILURE, never a skip.
3. **Residency**: any grid-wide wait requires all blocks co-resident. Blocks must not exceed the CU
   budget; `_check_block_num_resident` is the existing check.
4. Never modify anything outside this workspace.

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
op_tests/multigpu_tests/test_mega_moe_v2.py --network v4_pro --bs-list 128,512 \
  --iters 10 --accuracy-max-bs 512 --rtol 0.10
```
Prints one line per batch size:
`[MEGA-V2] bs=128 relL2=0.069100 path=fixed ... e2e=0.3821/0.3832ms mean/max`
Both `path=fixed` (bs=128) and `path=compact` (bs=512) must appear and both `relL2` must be < 0.10.

**Performance**:

```
op_tests/multigpu_tests/bench_mega_moe_v2.py --tokens <T> --route <R> --iters 10 --mega-only
```
Prints `[RESULT] route=... tokens=... mega_e2e=<rank-mean>/<rank-max>ms speedup=...`

**The metric is the rank-MAX** (the second number). A collective is gated by its slowest rank;
rank-mean can improve while the operator gets slower. Rank-mean is diagnostic only.

## The four route guards — all four must hold

| tokens | route            | why it is in the set                                  |
|--------|------------------|-------------------------------------------------------|
| 8192   | `uniform`        | large, balanced — the compute-bound case               |
| 8192   | `rank-mixed-skew`| large, hot-expert skew — where the barrier cost lives  |
| 512    | `uniform`        | small — launch/latency dominated                       |
| 512    | `rank-mixed-skew`| small + skew — the most drift-prone, guard against regression |

A candidate must be **at or above baseline rank-max on all four**, not on average.

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

Batch-to-batch drift on the small guards is ~4%, which is larger than the whole available gain.
Two arms timed minutes apart will disagree by more than the effect, **in either direction** — a real
win reads as a regression as often as not. Therefore:

- Run baseline and candidate **alternately** (A,B,A,B,A,B — at least 3 pairs per guard) and report
  the **per-pair** delta plus the median. Never compare two independently collected medians.
- Report the per-rep spread alongside the median.
- The speedup denominator is this frozen tree, run under the identical command. Nothing else.
- A latency win with no measured change in overlap is **suspicious, not accepted** — find the
  mechanism or discard the result.
