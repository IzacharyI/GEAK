---
id: megamoe_ep_mega_fusion
title: 'MegaMoE EP8 persistent tile-pipeline playbook'
kind: expert_skill
mode: mega
revision: mega-ep-fusion-v6
validation_schema: expert-skill-validation-v2
constraint_profile: semantic_and_compiler_shape
authors:
  - GEAK Team
scope: kernel
match:
  operator: moe_dispatch_combine
  arch_class:
    - '*'
  gens:
    - gfx950
  dtypes:
    - mxfp8_e4m3
    - mxfp4
  regimes:
    - prefill
    - decode
  from_backend: ''
  to_backend: flydsl
  profile_signature:
    op_name_regex: mega_moe|dispatch_combine|p2p_scatter
    min_pct_gpu: 20.0
  config:
    framework: MegaMoE_v2
    parallel: EP8
    arch: gfx950
    precision: a8w4
    graph: cuda_graph_captured
expects:
  isolated_speedup_min: 1.01
  e2e_delta_min_pct: 1.0
  parity: required
provenance:
  source: measured_reference
  origin: deconstructed_capability
  transfer_status: static_validated
  reuse_mode: common_mega_candidate_lifecycle
incumbent:
  label: measured_reference_persistent_megakernel
  is_ceiling: false
  measured_gain_vs_baseline_pct:
    tokens512_uniform: 1.49
    tokens8192_uniform: 4.71
role: experimental_authoring_prior
normative: true
normative_scope: explicitly_pinned_authoring
supersedes:
  - mega-ep-fusion-v4
  - mega-ep-fusion-v5
embedded_components: [planner_extension, contract, runtime_validation]
validation:
  schema_version: expert-skill-validation-v2
  skill_id: megamoe_ep_mega_fusion
  revision: mega-ep-fusion-v6
  status: experimental

  reference_evidence:
    status: measured
    subject:
      kind: source_reference
      revision: m25-measured-reference-v1
      tree_sha256: 53cc0eca0c814e8344600dc2064b6b4ba57683286e51dd6453e015de44dfe7ec
      baseline_tree_sha256: 54a1711b081b17af7a9744113d707a32f68b90a5a0272f4bf6d03ab14b9fa6ef
    recorded_at: "2026-09-15"
    gpu: gfx950/MI355X
    model: MegaMoE_v2
    measured:
      isolated: 1.0448
      e2e_pct: 4.48
      parity: pass
    mechanism_facts:
      first_compute_claim_domain: flat_m_tile_n_stripe
      completion_publication: per_stripe_system_atomic_in_unified_loop_tail
      note: This is measured reference evidence, not the v6 transfer constraint.
    artifact:
      kind: recorded_measurement_only
      uri: ""
      sha256: ""
      reproducible: false

  constraint_validation:
    status: static_validated
    hardware_verified: false
    subject:
      skill_revision: mega-ep-fusion-v6
      contract_sha256: ceb77f405eca1c609941f84e1a32a92deca816e093ca168f34055cb82d7c5a2c
      planner_extension_sha256: 535f60ded23fff7bc1346bebc11c9b5d0935671b252e35d7fc2616415d65c8b3
      checker_sha256: c1d0c023e7444f70259c67a8c48a7dfb2e7baa9e6e929b6e754ebd69fb9bb2bc
    rules:
      g1_owned_mtile_completion:
        transfer_deviation:
          from_reference: per_stripe_system_atomic_in_unified_loop_tail
          to_experimental_rule: one_m_tile_owner_system_store
          reason: the reference atomic shape is not compiler-equivalent in the evolved candidate frame
        negative_observations:
          - evidence_id: g1-completion-rmw-tail-fault
            placement: unified_loop_tail
            result: all_rank_null_base_device_fault
          - evidence_id: g1-completion-rmw-loop-head-fault
            placement: unified_loop_head
            result: all_rank_null_base_device_fault
        positive_repair:
          status: pending
          candidate_evidence_id: owner-mtile-system-store-v1
          required:
            - exact_head_independent_structure
            - on_card_jit
            - direct_ep8_accuracy

  candidate_validation_policy:
    evidence_scope: exact_candidate_head_and_tree_digest
    required_gates:
      - independent_structure
      - device_jit
      - path_activation_all_ranks
      - launch_shape
      - direct_accuracy
      - graph_liveness
      - residency
      - distributed_memory_order
      - artifact_distinctness
      - paired_performance
      - do_no_harm

  usage:
    auto_apply: false
    explicit_pin_modes: [authoring, candidate_validation]

  harness: kernel_workflow
  efficacy:
    model_path: ""
    gpu: gfx950/MI355X
    shapes:
      framework: MegaMoE_v2
      parallel: EP8
      precision: a8w4
      tokens_per_rank: [512, 8192]
      routes: [uniform, rank-mixed-skew]
    expect_min: auto

  do_no_harm:
    noise_band_pct: 1.45
    boundary_checks:
      - name: activation_required
        assert: every candidate leg reports path=MEGA on all eight ranks
      - name: direct_accuracy
        assert: relL2 is below 0.10 for 128, 512, and 8192 tokens
      - name: replay_stress
        assert: arrival-jittered graph replay passes before finalist promotion
      - name: compiler_frame_publication
        assert: no completion-address system or agent atomic RMW is reachable from the fused region; the owner-store replacement requires an on-card retry
      - name: frozen_denominator
        assert: paired rank-max performance is measured against frozen SCATTERED MegaMoE V2
---

# MegaMoE EP8 persistent tile-pipeline playbook

Use this experimental transfer of measured reference knowledge inside an
explicitly pinned ordinary `mode=mega`
Analyze → Plan → Author → Verify loop. It does not create a reproduction mode,
reserved lane, source label, or scheduler. When the match and baseline revision
apply, the detailed playbook's `MUST`/`MUST NOT` source-shape constraints override
generic optimization heuristics and Engineer improvisation; correctness and
current hardware evidence still override the playbook. Revision v6 is not
hardware-validated until an exact generated candidate passes the declared
on-card gates.

## Required reading

Read the detailed transfer playbook and embedded machine contract below. The
reference reached the recorded performance band; the v6 transfer rules remain
experimental until candidate validation.
Re-derive applicability from the current frozen source/task graph. Do not
replace the direct fused ABI, nested emitter, unified queue loop, publication
scope or generation protocol unless current evidence proves the replacement
and the deviation is recorded.

## When to use

Use for `mode=mega`, EP8 MegaMoE V2 on gfx950. With
`use_expert_skills=false`, this skill package must be absent and the
same roles derive candidates without this fusion knowledge.

## Mechanism

The target is not wrapper-level launch fusion. Build a tiled/instruction-level
persistent pipeline whose task graph, resource ownership and readiness edges are
explicit:

- CTAs move from GEMM1 to ready GEMM2 work when their local shard drains,
  without a global stage barrier;
- GEMM2/P2P and combine can likewise be phase-staggered across CTAs;
- LDS/VGPR/CU residency, publication/acquire scope, monotone generations and
  graph replay safety are correctness constraints;
- the fused region and local call closure contain no completion-address atomic
  RMW; one CTA claims one m-tile, computes all N stripes, then performs the
  ordered thread-0 system store;
- full fusion is the measured reference topology; the v6 owner-store transfer
  remains experimental until exact-candidate hardware validation.

For the pinned baseline, the following are closed source shapes, not equivalent
implementations: separate GEMM1-then-GEMM2 loops, fused pointers hidden in
trailing dispatch-table slots, a module-level Stage2 device body replacing the
nested closure/JIT emitter, and an agent-scope G1 completion publish. The
  playbook specifies the pinned transfer shape.

## Procedure

Run the normal Mega analysis and candidate loop:

1. Build the tile task graph and identify missing readiness edges.
2. Use the playbook to prioritize a credible persistent pipeline; do not copy
   or read an external implementation tree.
3. Author resumable production-source checkpoints in the ordinary candidate
   lane.
4. Debug JIT, direct correctness and graph liveness on the actual candidate.
5. Use paired rank-max measurement against frozen MegaMoE V2. Continue
   optimizing beyond the reference when evidence supports it.

When GPUs are unavailable, run the skill contract's independent structural tier after
authoring. An optional source reference is visible only to the read-only verifier, never
to the author. A byte-exact reference copy is checker calibration, not capability
evidence.

## Completion

A validated full-fusion candidate confirms:

- `path=MEGA` on all eight ranks;
- exactly two launches per rank;
- direct graph-path `relL2 < 0.10`;
- rank-max paired performance against the frozen scattered baseline;
- the required liveness and finalist guard contract.

## Do-no-harm notes

The playbook never overrides measurement. A skill-enabled candidate uses the
same correctness, artifact, graph, liveness, regression and frozen-baseline
speed gates as a skill-disabled candidate.

## Sources

- This `skill.md`: measured-reference guide, PlannerIR bindings, structural
  contract, validation evidence, and runtime validator in one versioned entry.
- Frozen baseline identity is supplied and verified by the workflow; no
  resolvable repository address is exposed to the authoring Skill.

## Detailed transfer playbook

The following is the embedded detailed transfer playbook for this Expert Skill.

# MegaMoE EP persistent-fusion playbook

This file is the independent implementation contract distilled from one
validated tiled/instruction-pipelined design produced from the frozen public
MegaMoE V2 baseline. When the baseline revision and match fields apply, its
`MUST`, `MUST NOT`, queue, memory-order, and compiler-shape rules are normative
inside the ordinary Mega planner/Engineer lifecycle. Current hardware evidence
may override a rule only when the candidate records the contradiction and the
replacement invariant explicitly. It does not create a reproduction mode,
candidate source, or reserved lane.

The measured reference and the v6 transfer are deliberately not conflated.
The reference used a per-stripe system atomic in its unified-loop tail. That
exact source frame ran successfully, but the same abstract operation in the
evolved independent candidate produced a repeatable compiler frame fault.
Revision v6 therefore marks its one-m-tile owner-store rule as an experimental
transfer deviation rather than claiming it was the measured reference shape.

## 0. Oracle use and known defects

The pinned oracle revision is visible only to the post-authoring verifier. It
calibrates feature coverage; it is not source to copy and is not blindly
normative. Independent implementations MUST correct these oracle defects:

1. the recurring A-to-LDS issue inside `gemm2_compute_v2` omits `NW=NW` and
   falls back to NW4;
2. tile-close token publication reloads the peer-ready base without adding the
   caller-supplied LDS slab base.

The oracle also waits only on the maximum Stage1 tile touched by a G2 chunk.
That is valid only if producer completion is prefix-monotone, which concurrent
sharded producers do not establish. An independent implementation MUST either
check every distinct Stage1 counter touched by a chunk or prevent a chunk from
crossing a Stage1 dependency boundary.

These corrections mean a byte-exact oracle materialization is expected to pass
identity calibration but fail the corrected independent semantic contract.

## 1. Result contract

The completed operator has exactly two launches per rank:

1. the existing BF16 to MXFP8 1x32 quantization launch;
2. one persistent FlyDSL megakernel containing dispatch, GEMM1, GEMM2, P2P
   publication, and combine.

The only campaign activation switch is:

```text
AITER_MEGAMOE_FUSE_ALL=1
```

With that switch set, the complete validated topology is selected whenever
`stage1.num_waves % 4 == 0`. `AITER_MEGAMOE_FUSE_COMBINE` may remain as a
diagnostic opt-out, but its default is `"1"`. Quant fusion is not part of this
playbook and remains off.

For a scored BF16 full-fusion arm, require all ranks to resolve:

```text
FUSE_ALL=1
FUSE_COMBINE=1
FUSE_QUANT=0
publish_tok_ready=true
p2p_write_through=true
all analysis/no-publication/geometry overrides unset
```

`path=MEGA` alone does not prove this variant: combine-off is M2 and
quant-on is M3. Require the marker plus the BF16 two-launch observation.
`forward_prequant` legitimately has no quant launch and is therefore a
one-launch API; it is not the BF16 launch-count contract.

Expose one host helper that resolves and validates this complete environment
before launch. Do not let independent
call sites infer “MEGA” from `FUSE_ALL` alone.

The candidate is complete only when all implementation checkpoints in section
13 are true. Intermediate commits are source checkpoints, not alternative
topologies and not performance candidates.

## 2. Source boundary

Start from the workflow-supplied frozen baseline tree. The validated runtime
structure is carried primarily by `mega_moe_v2.py`, `mega_moe_stage1.py`,
`mega_moe_stage2.py`, and `gemm2.py`. Author only these production files when
the corresponding call-chain change is required:

```text
aiter/ops/flydsl/kernels/communication_ops_utils.py
aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py
aiter/ops/flydsl/kernels/mega_moe/gemm1.py
aiter/ops/flydsl/kernels/mega_moe/gemm2.py
aiter/ops/flydsl/kernels/mega_moe/gemm_util.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_fused_s2c.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
aiter/ops/flydsl/kernels/mega_moe/quant.py
```

`mega_moe_fused_s2c.py` is an earlier Stage2+combine infrastructure/control,
not a requirement for the final Stage1-hosted third queue. The quant
emitter is optional infrastructure; the validated target keeps quant as its own launch.
Neither file may be required merely because it differs in a later oracle tree.

Do not edit tests or benchmarks and do not commit scripts, logs, caches, dumps,
or evidence to the candidate tree. Store all such artifacts in `OUTPUT_DIR`.
Do not read or diff an external implementation tree. The implementation must be
authored from this playbook and the frozen baseline.

## 3. Fixed 8192-uniform configuration

For EP8, 384 experts, 48 experts/rank, top-k 6, model dimension 7168,
intermediate dimension 3072, MXFP8 activation and MXFP4 weights:

### Stage1

```text
sort_block_m       = 128
tile_n             = 512
tile_k             = 256
num_waves          = 8
threads/block      = 512
grid_mult          = 1
num_dispatch_cu    = 96
work_shards        = 4
payload_chunk_rows = 384
payload_tile_ready = true
```

`grid_mult=1` is the selected 8192-large value. Other token buckets retain their
selected `grid_mult`; the fused protocol must remain correct at those launch
sizes and must not force the 8192 value into fixed/compact paths.

### Fused Stage2

Start from the selected Stage2 values:

```text
BM = 64
BN = 256
BK = 256
NW = 4
```

For the fused native-eight-wave form, set `NW=num_waves=8` and widen
`BN = 256 * (num_waves / 4) = 512`. All eight waves cooperate on one BN512
tile; each wave owns 64 N columns. This is one tile, not two lockstep BN256
halves.

Set:

```text
persist          = false
g2_spart         = 0
cu_num           = 0
fused_g2_pref    = 6
fused_g2_chunk   = 16
fused_g2_skew    = (5, 4)
g2_bhoist        = true
g2_ascale_pf     = true
p2p_write_through = true
publish_tok_ready  = true when combine is fused
```

Compile-time validation MUST enforce
`publish_tok_ready implies p2p_write_through` and require nonzero peer-ready
and tile-close pointers. If a different implementation uses cached P2P stores,
it must add a system release before the ready atomic; the validated path
uses write-through stores instead.

Do not apply a generic 64 KiB LDS cutoff to this gfx950 design. The target
supports a workgroup allocation up to 160 KiB; the validated NW8/BN512 emitter
uses more than 64 KiB. Compute the actual combined group-segment size, keep
Stage1/Stage2 storage aliased, and feed that size into the residency check.
Retile only if the compiled group segment exceeds the target limit or breaks
the one-resident-workgroup requirement.

The source-level accounting is:

```text
stage1_pool  = max(2 * SBM * tile_k, SBM * (tile_n / 2) * 4)
stage2_core  = max(BM * BN * accum_bytes,
                   (kStages + 1) * BM * KH_TILE_A)
stage2_slab  = stage2_core + packed/weight metadata
              + peer payload bases + optional peer-ready bases/broadcast
group_static = max(stage1_pool, stage2_slab * halves)
               + SBM * (model_dim / 32)  # Stage1 A_scale is additive
```

For the pinned 8192 geometry this is 131,728 B of aliased role storage plus
28,672 B of Stage1 scale LDS, or 160,400 B before compiler-added alignment,
against a 163,840 B gfx950 limit. Static arithmetic is necessary but not a
residency proof; emitted group-segment size and VGPR occupancy remain hardware
verification.

`persist_cu` and `skew_cu` belong to standalone Stage2 paths. They do not
partition the megakernel.

## 4. Host orchestration

In `MegaMoEV2._run_joint`:

- select the normal per-token-bucket config first;
- set `mega = (config.stage1.num_waves % 4 == 0 and
  AITER_MEGAMOE_FUSE_ALL == "1")`;
- print exactly one `[megamoe] path=MEGA` or `[megamoe] path=SCATTERED`
  marker per process (no rank suffix is required);
- leave quant as a separate launch;
- on `mega`, pass the complete fused Stage2 and fused-combine specifications and
  arguments into Stage1;
- default fused combine to on;
- when combine is fused, do not call trailing `combine_no_stage1`; return the
  shared-memory combine output produced by the megakernel.

The fused Stage2 specification is derived from the ordinary selected Stage2
config, with `persist=false`, `g2_spart=0`, `cu_num=0`,
`publish_tok_ready=fuse_combine`, and the nw8 override from section 3.

One `MegaMoEV2` instance supports one stream-serialized launch in flight.
Every EP rank must select the same token bucket, fused-combine setting, and
compatible topology for each combine generation. Concurrent launches on one
instance can mix ticket generations and are unsupported.

Pass Stage2 pointers for activation, scales, weights, expert metadata, sorted
ids/weights, tile-row base, P2P bases, per-m-tile readiness counters, and
`max_m_blocks`. Pass combine pointers for input, output, local token readiness,
claim/generation counters, peer token-readiness bases, and tile-close counters.
These are direct typed arguments of the compiled Stage1 launcher. The unfused
launcher supplies zero placeholder tuples for the unused argument groups so
both host paths retain one explicit ABI.

For the pinned oracle-compatible ABI this is 12 Stage2 pointers plus one
Int32 `max_m_blocks`, seven combine pointers, and two optional quant pointers.
Zero placeholders are valid only when the corresponding role is compiled out;
an active fused Stage2 or combine role rejects missing/zero required pointers.
Host tuple order, kernel parameter order and launcher forwarding must match
exactly. A clean full-fusion implementation may omit later wait-stat/quant-ingress fields,
but must declare that ABI deviation explicitly.

The fused kernel returns no Python value. With combine enabled, the host uses a
view of `shmem_comb_out_tok` with `comb_cfg.combine_dtype` and unsliced shape
`(mtpr, combine_token_view_dim)`, returning `[:run_tokens]` only when
`slice_output=true`. It must not return Stage1 intermediates or launch a
trailing combine.

Quant and the megakernel must execute on the same effective stream, or an
explicit event must order quant completion before the megakernel. Returned
shared-memory views inherit that stream dependency.

## 5. Stage2 emitter contract

Do not hand-inline a second, approximate GEMM2/P2P body in Stage1.
`mega_moe_stage2.py` exposes `derive_stage2_emit_constants` and one
`make_stage2_body_emitter` closure used by both the standalone Stage2 kernel
and the fused Stage1 work loop. Stage1 consumes the derived BM/BN/NW, slab,
P2P, and publication constants; it must not recompute an approximate subset.

FlyDSL outlining rules are part of this reference:

- `make_stage2_body_emitter` is an ordinary host-side factory. Compile-time
  constants are captured in its closure.
- It returns `emit_stage2_body`, which accepts the complete dynamic tile
  arguments as keywords. Inside that closure, a zero-argument nested
  `_emit_stage2_body` is decorated with `@flyc.jit`; this is the source shape
  on which FlyDSL rewrites the dynamic branches.
- Do not replace that shape with module-level device functions carrying a
  constants `dict`, `SimpleNamespace`, or opaque config object. That alternate
  outlining compiled in isolation but changed argument binding and faulted on
  device.
- Nested publication helpers that contain dynamic control flow are also
  `@flyc.jit`; simple helpers that are already lexically inside the rewritten
  body need not be moved to module scope.
- validate the exact shared body first through the standalone NW4 caller, then
  through the fused caller. Do not leave an unused outlined body beside a
  restored inline implementation and call that checkpoint complete.

The emitted unit accepts:

```text
tx_i32, bx_i32, lane, wave
arg_aq, arg_ascale, arg_bq, arg_bscale
arg_eids, arg_cumsum, arg_max_expert_tiles
arg_stids, arg_sweights, arg_trb
arg_p2p_comb_inp, i32_max_m_blocks
i32_inter, i32_hidden, i32_kpad, i32_npad
lds_slab, lds_byte_off
arg_p2p_tok_ready, arg_mtile_ctr
```

Stage1's `num_valid` queue bound and the Stage2 emitter's cumsum bound MUST use
the same value, preferably the same `op.num_valid` pointer. For this exact
shape, require `i32_hidden == N_OUT`, `i32_inter == INTER_MAX`, and zero
padding. A disagreement can turn a claimed G2 unit into a no-op and leave
combine permanently short of arrivals.

Within one call it:

1. loads peer payload bases into the emitter LDS slab;
2. when token publication is enabled, loads peer token-ready bases into LDS;
3. maps the claimed pair to `(m_block, n_block)`;
4. stages routed activation rows and metadata;
5. invokes the existing `gemm2_compute_v2` once;
6. invokes `p2p_scatter_epilog` with write-through P2P stores;
7. invokes the tile-close publication in section 8.

The unified outer loop begins every iteration with a block-uniform LDS hazard
barrier separating the previous unit's LDS reads from the current claim
scratch. A carried `g2_pend` continuation skips the claim atomic and
kind/unit broadcast, not this top-of-loop hazard barrier. A fresh claim is
written by thread 0 and followed by one block barrier before all threads read
kind/unit.

`gemm2_compute_v2` and its helpers must accept the emitter's `NW` so the BN512
tile maps eight waves to distinct 64-column bands. Preserve the existing
four-wave behavior when `NW=4`; do not duplicate waves 0-3. The 64-column
band is specific to the 8192 base-BN256 case. Other selected buckets may
legally produce a 32-column band after NW8 widening; require `BN % NW == 0`
and `(BN / NW) % 16 == 0`, not a universal 64-column assertion. Also enforce
`BN <= 512`, which the epilogue lane mask assumes.

Generalize the row-side epilogue partition as well as the column band:

```text
for row_iter in range(BM / NW):
    row = wave + row_iter * NW
```

Retaining the standalone NW4 constants (`BM / 4`, `row_iter * 4`) under NW8
makes waves overlap rows and lets later iterations address `row >= BM`, which
can surface as a peer-store memory fault only after the LDS-base bug is fixed.
Pass `NW` to every `issue_a_load_lds_dt` call, including the initial A-stage
priming helper and the recurring prefetch issued inside
`gemm2_compute_v2`'s K loop. Leaving that internal call at its default NW4
corrupts the LDS row partition for waves 4–7 even when the outer shared emitter
was generalized correctly.

Every LDS reference is relative to the supplied slab base. This includes
`p2p_scatter_epilog` lookups for packed route, weight, and peer payload base,
plus `_publish_tok_ready`'s peer token-ready lookup. Either pass absolute
`lds_slab + offset` values or add `lds_base_i32` inside each helper. No
relocatable helper may retain a zero-based `lds_typed_ptr(offset + ...)`
reference. The defect can be latent while `lds_byte_off==0`, so test a nonzero
slab base statically even when the scored NW8 path uses offset zero.

The fused Stage2 slab aliases the Stage1 pool because a block executes one work
unit at a time. Set:

```text
lds_pool_bytes = max(stage1_pool_bytes, stage2_slab_bytes * halves)
```

For fused combine, `halves` must be one.

## 6. Persistent entry and startup roles

At kernel entry, thread 0 of each block atomically obtains a 64-bit ticket from
the entry-counter slot selected by `GRID_MULT_VALUES.index(grid_mult)` and
broadcasts it through LDS:

```text
generation = ticket64 // launch_grid_x
ticket     = ticket64 - generation * launch_grid_x
gate_epoch = generation + 1
```

Preserve the inherited startup arithmetic:

```text
launch_grid_x = num_cu * grid_mult
next_parity   = old_parity XOR 1
next_expected = expected[next_parity] + npes
launch_epoch  = (next_expected / npes) * 2 - next_parity
```

Exactly `launch_grid_x` blocks acquire one ticket from the 64-bit entry counter
slot selected by `grid_mult`. The owner publishes rotated peer launch-ready
state with system release, waits for every peer and acquires, resets work and
combine state, then publishes the gate. Plan-ready and each payload tile/expert
wait retain their baseline release/acquire ordering.

Roles:

```text
ticket == 0                    owner/planner
1 <= ticket <= dispatch_blocks dispatch producer
ticket > dispatch_blocks       consumer pool
```

The owner performs parity/epoch handoff, peer launch-ready exchange, and
per-launch state initialization before publishing the epoch gate. Other blocks
wait for that exact gate and acquire it.

The dispatch owner/producers use the existing dispatch planning, grouping, and
payload emitters. After their startup responsibilities, owner and dispatch
producer CTAs also reach the shared fine-grained work pool; entry ticket is not
a permanent compute-role partition. There is no static GEMM1/GEMM2 percentage
partition.

State initialization is ordered at two boundaries:

- work heads and combine claim/generation state are initialized before the
  epoch gate is published;
- token-window padding, GEMM1-to-GEMM2 completion counters, and sharded GEMM2
  heads are initialized after the epoch gate but before `PLAN_READY`.

## 7. Per-launch state lifetimes

Keep these state classes distinct:

- dispatch epoch/parity: existing double-buffered protocol;
- fixed-slot `GROUP_DONE`: at least `world_size` Int32 slots, with every
  destination-indexed slot cleared before gate publication;
- GEMM1-to-GEMM2 completion counters: one Int32 per Stage1 m-tile, cleared by
  the owner before plan-ready publication;
- sharded GEMM2 claim heads: one Int32 on each 64-byte line, cleared each
  launch;
- Stage2 tile-close counters: one per GEMM2 m-block, self-cleared by the block
  that closes the final N stripe;
- combine claim head `c_ctr[0]`: rank-local, reset to zero each fused-combine
  launch;
- combine generation `c_ctr[1]`: rank-local monotone counter, incremented once
  each fused-combine launch;
- destination token readiness: cross-rank monotone counters, never cleared.

The validated ABI stores the first two classes in one direct `s2_ctr` buffer:

```text
max_m_blocks = ceil(s1_row_capacity / BM)
max_s1_tiles = ceil(max_m_blocks * BM / SBM)
completion counters = Int32 indices [0, max_s1_tiles)
G2 head(shard s)    = Int32 index max_m_blocks + 16*s
s2_ctr_count >= max(max_s1_tiles,
                    max_m_blocks + 16*(WORK_SHARDS-1) + 1)
```

The host may over-allocate the backing tensor. Tile-close storage needs at
least `max_m_blocks` Int32 values, and `c_ctr` needs at least two. Do not move
these addresses into ad-hoc trailing dispatch-table slots.

Combine generation is not dispatch generation. A Stage1 launch compiled without
combine publishes no combine arrivals and must not advance the combine target.

Before plan-ready, the owner pads every destination token outside
`[0, cur_tok)` to `topk * combine_generation`. Without this, a token omitted by
a small batch is one generation behind when a later larger batch includes it.

The single P2P payload buffer remains replay-safe because the next launch's
cross-rank entry handshake occurs only after every rank has completed the
current local kernel/combined reduction. Preserve that ordering if host launch
or stream orchestration changes. `c_ctr[1]` and token-ready counters are Int32;
“never clear” means monotone within the supported pre-wrap lifetime, not
unbounded process lifetime.

## 8. Publication protocols

Scope matrix for the pinned gfx950 source:

- agent atomic: entry ticket, GEMM1/GEMM2 queue heads, Stage2 tile-close
  counter, and combine claim head;
- system atomic: remote peer token-ready publication;
- owner-only system store: GEMM1 m-tile completion publication;
- owner-only ordinary load/store: combine generation;
- workgroup release plus block barrier: Stage2 tile-close ordering.

Preserve the visibility domain, not a simplified “rank-local means agent”
rule. Relaxed MORI waits across XCDs are the reason the G1 completion uses
system scope.

### GEMM1 to GEMM2

GEMM1 activation and scale stores use the gfx95x system-visible write-through
cache modifier. Completion publication preserves this ordering:

1. all waves execute `s_waitcnt(0)`;
2. execute a block barrier;
3. thread 0 publishes `n_tiles` to the completion counter for the claimed
   m-tile with `store_i32_system`.

The counter threshold is `n_tiles`: all N-column tiles for that Stage1 m-tile.
The validated gfx950 source uses system scope even though the logical edge is
rank-local: its consumer is a relaxed MORI wait across XCDs. Do not substitute
agent scope based only on the abstract edge classification. The write-through
stores plus `s_waitcnt(0)` and the block barrier are the producer ordering; do
not add a separate per-tile `fence_system_release()`. The consumer waits with
`int32_wait_until_greater_than(counter, n_tiles - 1)`. There is no whole-grid
`ready1 == NUM_G1_BLOCKS` barrier.

The fused region and every transitively called local helper MUST contain no
system- or agent-scope atomic RMW whose address derives from the GEMM1
completion table. Moving that RMW from the loop tail to the loop head still
leaves it in the same gfx950/flyc register-allocation frame and has reproduced
the same target-independent null-base device fault with a valid `s2_ctr`.

The selected v6 shape is exact: one CTA claims one m-tile, computes all
`n_tiles` column stripes, executes `s_waitcnt(0)` and a block barrier, and then
thread 0 performs one system-visible store of `n_tiles` to that m-tile's direct
`s2_ctr` counter. Do not substitute a loop-head or post-loop completion atomic,
split stripe ownership across CTAs, tunnel the counter through an extended
dispatch-pointer table, add a whole-grid barrier, or add a separate GEMM2
drain.

Any change to this placement requires an on-card JIT/correctness retry; a
structural pass alone is insufficient evidence.

Do not add and repeatedly benchmark a producer-only readiness activation
switch while Stage2 still launches separately. The counter has no consumer in
that shape, changes the Stage1 code-generation frame, and cannot validate the
pipeline. Keep publication under the complete fused-mode compile-time path,
wire its GEMM2 consumer in the same authoring checkpoint, then validate the
producer/wait pair together with `AITER_MEGAMOE_FUSE_ALL`.

### GEMM2/P2P to combine

P2P payload stores are write-through. Publication occurs once per destination
row only after every N stripe of that GEMM2 m-block has stored:

1. all waves execute the emitter's workgroup release and block barrier;
2. thread 0 increments `arg_mtile_ctr[m_block]` with `atomic_add_agent`;
3. broadcast the previous count through LDS;
4. only the block observing `previous == num_n_blocks - 1` closes the tile;
5. that closer subtracts `num_n_blocks` from the tile-close counter with
   `atomic_add_agent`;
6. one thread per valid BM row decodes destination PE/token and performs one
   `atomic_add_system(peer_tok_ready[token], 1)`.

Do not publish once per N stripe and do not surround every stripe with a
separate Stage1-side system fence/barrier sequence.

After the relaxed token-ready wait, combine payload loads MUST be
system-visible (`sc0|sc1` in the validated shared reducer) or be paired with an
explicit system acquire. Ordinary cached payload loads are not legal merely
because the readiness count advanced.

## 9. Unified GEMM work loop

The consumer pool runs one `while consumer_active` loop containing both GEMM1
and GEMM2 dispatch. A first `while` for all GEMM1 work followed by a second
GEMM2 drain is structurally wrong even without a grid barrier: it omits
opportunistic G2 preemption, changes the live ranges, and does not reproduce
the validated compiler shape.

Stage2/combine buffers are direct typed arguments of the compiled Stage1
kernel (`s2_*`, `c_*`, and optional `q_*` groups). Tuning values remain closure
constants/JIT-key fields; do not encode the fused data pointers as extra slots
in Stage1's dispatch table merely to preserve the old signature.

GEMM2 claims are linear Stage2 work units, not whole m-blocks. At 8192 both
GEMM1 and GEMM2 heads use `work_shards=4` on distinct 64-byte lines; other
buckets use the selected Stage1 `work_shards`.
For every CTA:

```text
work_shard = ticket & (WORK_SHARDS - 1)
```

Multiple CTAs share each shard head. Exhausting one CTA's shard claim does not
mean the grid has drained GEMM1 or GEMM2.

Exactly these scalars are loop-carried:

```text
consumer_active
g2_pend
g2_next
```

`kind` and `unit` are reinitialized every iteration. Config values are closure
constants and are encoded in the JIT key, not added as device arguments.

The priority order is:

1. finish a pair already carried in `g2_pend/g2_next`, without an atomic or LDS
   broadcast;
2. on the route-gated preemption path, a `ticket % 6 == 0` block may
   non-destructively peek and claim an already-ready GEMM2 chunk only when the
   `(5,4)` expert-skew predicate is true;
3. otherwise claim one GEMM1 m-tile from this block's sharded GEMM1 head,
   compute every N stripe owned by that claim, then publish once;
4. once GEMM1 is drained, claim one sharded GEMM2 chunk and wait only for the
   complete set of in-range m-tile dependencies;
5. execute one unit and repeat.

The GEMM2 queue claim unit is a chunk of 16 consecutive linear Stage2 units for
the 8192 bucket. With native nw8:

```text
num_n_blocks = hidden_dim / fused_BN
total_units  = ceildiv(num_valid, BM) * num_n_blocks
total_claims = ceildiv(total_units, g2_chunk)
```

A local shard claim maps to work as:

```text
global_claim = work_shard + local_claim * work_shards
unit_base    = global_claim * g2_chunk
m_block      = unit / num_n_blocks
ready_s1_tile = (m_block * BM) / SBM
```

The last unit in a chunk is clamped to `total_units - 1`. A non-destructive
peek and the following claim race: another CTA may consume the peeked claim, so
all readiness checks use the actual returned claim.

Provide a lexically visible `_g2_chunk_ready_all` helper. It enumerates every
distinct Stage1 m-tile touched by the returned chunk and waits for each
counter, or the claim allocator truncates the chunk at the current Stage1
dependency boundary. Waiting only for the numerically greatest tile is not
sufficient: sharded producers do not complete tiles in prefix order.

The opportunistic predicate is:

```text
max_expert_tiles * experts_per_rank * SBM * skew_den
    > num_valid * skew_num
```

with `(skew_num, skew_den) = (5, 4)`. It is observed false on the scored
8192-uniform route, but must not be assumed false for every nominally uniform
small shape because tile-ceiling effects can trigger it. Chunk size remains a
host bucket decision (`16` at `cur_tok >= 4096`, otherwise `1`), not a result of
this predicate.

`fused_g2_pref` MUST be zero or at least two, so a non-preempting cohort always
continues producing GEMM1 dependencies. Both skew numerator and denominator
must be positive. The value six is target tuning, not the progress invariant.

When preemption is false, each CTA continues claiming GEMM1 only from its own
shared shard. As soon as that CTA's shard claim is out of range it starts
claiming its GEMM2 shard, without waiting for other shards or in-flight GEMM1
units. When its G2 shard is exhausted it may enter combine without a grid-wide
G2 drain. Thus a CTA's local order is G1 then G2, while the grid still executes
G1 and G2 concurrently across CTAs, followed by G2 and combine concurrently.
Do not turn this into a global drain or static role split.

`g2_pend/g2_next` carry remaining linear units in the outer work loop. Never wrap the
GEMM2 body in a nested dynamic `while`: that makes the full GEMM2 live range
loop-carried and causes register spilling.

## 10. Combine third queue

Combine executes inside the same persistent kernel after a block has drained
both local GEMM queues. A block waiting for cross-rank arrivals must never still
hold an unexecuted GEMM unit.

Refactor the baseline combine kernel's Stage3 reduction into
`make_combine_reduce_emitter` and call the same emitter from both the standalone
combine kernel and this persistent role. The host wrapper must expose the
combine input/output, token-ready, peer-ready, claim/generation, and tile-close
buffers needed by the fused call. Do not duplicate or simplify the arithmetic
inside Stage1.

Build `_combine_transport_spec` with `enable_weights=false`,
`fp8_direct_cast=false`, the selected transport mode, and the effective
`max_recv`; then configure the shared combine-reduce emitter with:

```text
zero_copy                  = false
skip_stage1                = true
blockwise_fp8_transport    = true for the 8192 large config
tok_ready_expected         = topk
tok_ready_epoch            = c_epoch
nominal_warp_num           = launch_grid_x * num_waves
```

Keep these capacities distinct:

```text
row_bytes(BF16)      = 2 * hidden_dim
row_bytes(block-FP8) = hidden_dim + hidden_dim / 32
slot                 = destination_local_token * topk + topk_slot
comb_inp_nbytes      = mtpr * topk * row_bytes
physical input rows  = max(world_size * mtpr, mtpr * topk)
recv_cap             = world_size * mtpr
effective_max_recv   = world_size
                       * min(ceil(max_total_recv_tokens / world_size), mtpr)
```

Do not conflate routing validity capacity, physical allocation capacity and
the reducer's effective maximum.

The shared reducer partitions work as:

```text
nominal_warps   = launch_grid_x * num_waves
warps_per_token = ceil(nominal_warps / max(cur_tok, 1))
s3_total_work   = cur_tok * warps_per_token
```

Blockwise FP8 clamps `warps_per_token` to `hidden_dim / 32`; its
`hdim_per_warp` is a rounded multiple of eight Int32 elements.

Combine work items are per wave, but claims are per block:

1. block barrier;
2. thread 0 performs an agent-scope atomic increment of `c_ctr[0]` once and broadcasts the block
   claim;
3. block barrier after the LDS claim write, then
   `base = block_claim * num_waves`;
4. wave `w` processes `base + w` when in range;
5. repeat until the block base is out of range.

The combine emitter may block on
`tok_ready[token] > topk * c_epoch - 1` because this block has already drained
both local GEMM queues. Do not turn this into per-wave claim atomics and do not
insert combine claims into the middle of a block's GEMM loop.
The fused combine path does not execute the standalone all-rank combine
barrier; it runs only the shared Stage3 reducer with per-token arrival waits.

Routing must guarantee exactly `topk` accepted contributions per output token,
or publish an explicit zero contribution/readiness arrival for every rejected
slot. Expert-ID validation, capacity overflow, and route drops must not leave a
token waiting for an arrival the producer is allowed to omit.

## 11. Small-shape and bucket rules

Never force 8192 tuning constants through campaign activation switches.
Derive them in the host configuration:

- `fused_g2_chunk = 16` only when `cur_tok >= 4096`, otherwise `1`;
- large-path `work_shards = 1` for buckets up to 32, `8` at 2048, otherwise
  `4`;
- large-path `payload_chunk_rows = 256` at bucket 512, otherwise `384`;
- preserve the complete selected Stage1 configuration, including `grid_mult`,
  `work_shards`, payload mode, `b_nt`, resource flags, `sort_block_m`,
  `num_waves`, and `num_dispatch_cu`; fixed/compact defaults may use
  `work_shards=8`;
- set fused `NW = selected_stage1.num_waves` and
  `BN = selected_stage2.BN * (NW / 4)`, so NW4 keeps BN and NW8 doubles it;
- gate fused execution on `%4==0`, then enforce the emitter's actual NW4/NW8,
  BM/BN divisibility, LDS, and single-half constraints;
- derive `blockwise_fp8_transport` from `config.p2p_quant`; a 512-token run in a
  large-MTPR config may still use blockwise transport, while fixed/compact
  configurations may not;
- derive the Stage2 slab from the running BM/BN and grow the shared pool with
  `max`, rather than asserting that 8192's slab fits a smaller shape's pool.

The same authored source must support fixed, compact, and large paths without
an 8192-only environment override.

Route encoding and bounded-buffer invariants:

- routed token/PE occupies 24 bits and the top-k slot occupies 8 bits;
- require `npes * max_tok <= 2^24`, `topk <= 2^8`, and power-of-two
  `max_tok`;
- P2P resource byte size is positive and below `2^31`;
- invalid lanes use the bounded resource's OOB sentinel, never an arbitrary
  address;
- Stage1 non-tile resources retain their separate `2^32` capacity check.

## 12. FlyDSL invariants

- Keep all tuning/config fields in the cached Python closure and JIT kernel
  name. Do not grow the device signature with role percentages or tuning
  scalars. The signature does grow with the direct `s2_*`, `c_*`, and optional
  `q_*` runtime pointer groups; that is intentional and distinct from passing
  tuning values as device arguments.
- Initialize every dynamic-control-flow local to a typed FlyDSL value before
  its `if`/`while`.
- Do not rebind a kernel argument name inside device control flow.
- Preserve the validated nested-emitter and unified-loop lexical shape. A
  semantically similar module-level emitter, extended dispatch table, or
  separate post-GEMM1 drain is not compiler-equivalent on this FlyDSL revision.
- Preserve `gemm2_compute_v2` arithmetic; the playbook changes scheduling and
  publication, not GEMM math.
- Every barrier is block-uniform. A carried continuation path must be
  block-uniform before it bypasses the claim broadcast.
- Keep Stage1 and fused Stage2 LDS as aliased role storage, not additive
  allocations.
- Validate Stage1 guards: gfx95x, `num_waves>1`,
  `1<=waves_per_eu_hint<=4`, `tile_n%num_waves==0`,
  `(2*inter_dim)%tile_n==0`, allowed grid multipliers,
  `0<num_dispatch_cu<num_cu` divisible by `npes`, `tile_k==256`,
  `WORK_SHARDS in {1,2,4,8}`, and payload-tile readiness only with valid
  chunking.
- Validate fused guards: `num_waves%4==0`, `num_n_blocks%halves==0`,
  `SBM%BM==0`, power-of-two `max_tok`, supported BM/dtype/transport,
  `INTER_MAX%BK==0`, `NW in {4,8}`, `BN%NW==0`,
  `(BN/NW)%16==0`, and `BM%NW==0`.

Every IR-affecting value must participate in the actual memoization/compiler
cache identity. It need not appear literally in the human-readable kernel
symbol when `functools.cache`, sorted hashable specs and schema versions make
collisions impossible. Resolve environment-derived values before cache lookup;
changing an environment variable after the first compile must not silently
reuse the old kernel.

Preserve hidden barriers inside the shared emitters: GEMM1 epilogue's final
LDS barrier, Stage2 CShuffle barriers, tile-close release/barrier, and the
outer loop's top-of-iteration LDS hazard barrier are cumulative requirements,
not substitutes for one another.

## 13. Reference implementation decomposition

These checkpoints describe a proven way to decompose implementation work. The
common Mega lifecycle may combine checkpoint boundaries, but it must not
replace the nested emitter, direct fused ABI, unified `kind/unit` loop, queue
priority, publication scopes, or state-generation rules with merely
semantically similar source:

1. `host_wiring`: fused specs/arguments, path marker, default fused combine,
   two-launch return path.
2. `shared_emitters`: write-through support and the shared Stage2 emitter with
   tile-close publication.
3. `startup_state`: ticket roles, generation, reset/pad rules, and all required
   pointers.
4. `unified_gemm_drain`: sharded/chunked pair claims, route-gated preemption,
   outer-loop carry, per-m-tile publication.
5. `combine_queue`: block-granular third-queue claims and monotone arrival wait.
6. `bucket_safety`: fixed/compact/large configuration derivation.
7. full-path validation: all required mechanisms are present in one runnable
   candidate.

Commit whenever a stage builds or materially advances source. Do not run a
paired performance campaign on a half-implemented path. A compile or short
on-device correctness/liveness smoke may localize a fault.

## 14. Final verification

For a candidate implementing the complete v6 transfer design:

- candidate activation is exactly `AITER_MEGAMOE_FUSE_ALL=1`;
- baseline activation is exactly `AITER_MEGAMOE_FUSE_ALL=0`;
- every rank reports `[megamoe] path=MEGA`;
- every rank reports two launches;
- direct graph-captured candidate-vs-reference correctness has
  `relL2 < 0.10` for fixed, compact, and 8192 paths;
- score tier runs paired rank-max `8192_uniform` measurements and at least 30
  graph replays;
- finalist tier runs all target/regression guards and the configured jittered
  replay contract.

The recorded validation band is a target, not permission to skip any
correctness or stability gate.

For the per-token cross-rank arrival protocol, finalist liveness means at
least 256 bounded-timeout graph replays with routing mutated between replays,
covering fixed and large shapes on uniform and skew routes; compact 512 is
additionally required for direct correctness. At declared checkpoints compare
replayed output numerically, not only for finiteness, so a stale finite graph
result cannot pass.

## 15. GPU-free independent structural validation

When the EP8 pool is unavailable, authoring may continue without claiming
hardware success. The author receives only the frozen baseline, this Skill,
generic knowledge, and its own lane state. It must not read the source oracle.
After the authoring turn ends, a separate read-only verifier may compare the
candidate with the pinned oracle:

```text
python kernel_workflow/tools/expert_skill_contract.py \
  --contract perf_knowledge/expert_skills/skills/megamoe_ep_mega_fusion/skill.md \
  --baseline <frozen-baseline> \
  --reference <read-only-reference-tree> \
  --candidate <independently-authored-tree> \
  --authoring-manifest <pre-comparison-generation-manifest.json> \
  --require independent
```

The independent structural tier requires the two-launch host path, direct
fused ABI, nested shared Stage2 emitter, one unified GEMM `kind/unit` loop,
route-gated G2 preemption, system-scope G1 publication, tile-close/token
publication, combine third queue/generation, NW8 mapping, LDS role aliasing,
and write-through stores. Candidate-only diagnostic switches are forbidden.

Byte-exact identity of the core oracle files is useful only to calibrate the
checker. It sets `reference_copy_detected=true` and
`capability_eligible=false`; copied source is not evidence that Mega authored
the structure. A valid independent pass must be structurally compatible while
remaining source-distinct. Before revealing the oracle, seal the authored tree
digest in the manifest with `sealed_before_reference_comparison=true` and
`reference_exposed_during_authoring=false`. This remains an orchestration
attestation (`provenance_status=attested_unverified`), not cryptographic proof
of what the author could access.

Static validation can establish syntax, call-graph and source-structure
coverage. It cannot establish device compilation, actual launch count,
correctness, deadlock freedom, graph replay safety, or performance. Until a
later EP8 run passes those gates, record all of `hardware_verified`,
`accuracy_verified`, and `performance_verified` as false.

## Embedded planner extension

```expert-skill-planner-extension
schema_version: expert-skill-planner-extension-v1
skill_id: megamoe_ep_mega_fusion
revision: mega-ep-fusion-v6
plan_version: mega-plan-v2

applicability:
  mode: mega
  operator: moe_dispatch_combine
  architecture: gfx950
  parallelism: EP8
  framework: MegaMoE_v2
  required_baseline_capabilities:
    - separate_quantization_launch
    - dispatch_stage
    - first_expert_compute
    - second_expert_compute
    - distributed_combine

source_hints:
  operator_entry: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
  focus_files:
    host:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
    first_compute:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
      - aiter/ops/flydsl/kernels/mega_moe/gemm1.py
    second_compute:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
      - aiter/ops/flydsl/kernels/mega_moe/gemm2.py
      - aiter/ops/flydsl/kernels/mega_moe/gemm_util.py
    transport_and_reduce:
      - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
      - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py
      - aiter/ops/flydsl/kernels/communication_ops_utils.py

selection_policy:
  continue_existing_lane_first: true
  preferred_template: full_persistent_pipeline
  partial_template_is_fallback: true
  required_failure_order:
    - plan
    - correctness
    - abi
    - lifecycle
    - resource
    - compiler
    - schedule
    - performance
  do_not_measure_half_implemented_topology: true

ir_bindings:
  target:
    launch_count: 2
    required_regions:
      - input_quantize
      - dispatch_plan
      - first_compute
      - second_compute_transport
      - output_reduce
    required_queues:
      - first_compute_queue
      - second_compute_queue
      - output_reduce_queue
    required_capabilities:
      - item_level_overlap
      - direct_fused_abi
      - monotone_cross_rank_readiness
      - graph_replay_safe

  work_domains:
    - id: input_tokens
      unit: token
      extent: run_tokens
      index_type: i32
    - id: routed_rows
      unit: routed_row
      extent: num_valid
      index_type: i32
    - id: first_stage_m_tiles
      unit: m_tile
      extent: ceildiv(num_valid, stage1_block_m)
      index_type: i32
      owned_subdomain:
        unit: n_stripe
        extent_per_claim: stage1_n_tiles
        linear_index: m_tile * stage1_n_tiles + n_stripe
    - id: second_stage_units
      unit: tile_pair
      extent: ceildiv(num_valid, stage2_block_m) * stage2_n_tiles
      index_type: i32
    - id: output_tokens
      unit: output_token
      extent: run_tokens
      index_type: i32

  regions:
    - id: input_quantize
      role: producer
      work_domain: input_tokens
      engine: vector_memory
    - id: dispatch_plan
      role: producer
      work_domain: routed_rows
      engine: scalar_memory
    - id: first_compute
      role: producer_consumer
      work_domain: first_stage_m_tiles
      engine: matrix_memory
    - id: second_compute_transport
      role: producer_consumer
      work_domain: second_stage_units
      engine: matrix_interconnect
    - id: output_reduce
      role: consumer
      work_domain: output_tokens
      engine: vector_interconnect

  buffers:
    - id: quantized_input
      role: input
      element_type: mxfp8_e4m3
      capacity: run_tokens * hidden_dim
      address_space: global
      format: block_scaled
      producers: [input_quantize]
      consumers: [dispatch_plan, first_compute]
      lifetime: stream_ordered_launch
      alias_group: ""
    - id: routed_payload
      role: intermediate
      element_type: packed_route_and_activation
      capacity: max_routed_rows
      address_space: symmetric_global
      format: fixed_slot
      producers: [dispatch_plan]
      consumers: [first_compute]
      lifetime: launch_generation
      alias_group: ""
    - id: first_stage_output
      role: intermediate
      element_type: mxfp8_e4m3
      capacity: num_valid * intermediate_dim
      address_space: global
      format: write_through
      producers: [first_compute]
      consumers: [second_compute_transport]
      lifetime: launch_generation
      alias_group: ""
    - id: peer_reduce_payload
      role: transport
      element_type: bf16_or_block_fp8
      capacity: max_tokens_per_rank * topk * row_bytes
      address_space: symmetric_global
      format: write_through
      producers: [second_compute_transport]
      consumers: [output_reduce]
      lifetime: combine_generation
      alias_group: ""
    - id: shared_role_storage
      role: scratch
      element_type: byte
      capacity: max(stage1_pool_bytes, stage2_slab_bytes * role_halves)
      address_space: workgroup
      format: aliased_roles
      producers: [first_compute, second_compute_transport]
      consumers: [first_compute, second_compute_transport]
      lifetime: work_unit
      alias_group: compute_role_pool
    - id: combined_output
      role: output
      element_type: bf16
      capacity: run_tokens * hidden_dim
      address_space: symmetric_global
      format: token_major
      producers: [output_reduce]
      consumers: []
      lifetime: returned_view
      alias_group: ""

  counters:
    - id: entry_ticket
      element_type: i64
      capacity: one_per_grid_multiplier
      scope: agent
      lifecycle: monotone
      reset_owner: none
      generation: ticket // launch_grid_x
      producers: [dispatch_plan]
      consumers: [dispatch_plan, first_compute, second_compute_transport, output_reduce]
      publish: atomic_add_relaxed
      wait: none
    - id: launch_gate
      element_type: i32
      capacity: one
      scope: system
      lifecycle: monotone
      reset_owner: none
      generation: launch_epoch
      producers: [dispatch_plan]
      consumers: [first_compute, second_compute_transport, output_reduce]
      publish: release_after_launch_state_initialization
      wait: acquire_exact_epoch
    - id: first_tile_ready
      element_type: i32
      capacity: max_first_stage_m_tiles
      scope: system
      lifecycle: per_launch
      reset_owner: dispatch_plan
      generation: launch_epoch
      producers: [first_compute]
      consumers: [second_compute_transport]
      publish: waitcnt_then_block_barrier_then_thread0_system_store_stage1_n_tiles
      wait: greater_than(stage1_n_tiles - 1)
    - id: second_claim_heads
      element_type: i32
      capacity: max_second_stage_m_blocks + 16 * (work_shards - 1) + 1
      scope: agent
      lifecycle: per_launch
      reset_owner: dispatch_plan
      generation: launch_epoch
      producers: [second_compute_transport]
      consumers: [second_compute_transport]
      publish: sharded_atomic_claim
      wait: none
    - id: second_tile_close
      element_type: i32
      capacity: max_second_stage_m_blocks
      scope: agent
      lifecycle: self_clearing
      reset_owner: closing_workgroup
      generation: combine_generation
      producers: [second_compute_transport]
      consumers: [second_compute_transport]
      publish: workgroup_release_barrier_atomic_add
      wait: previous_count_equals_last_n_stripe
    - id: peer_token_ready
      element_type: i32
      capacity: max_tokens_per_rank
      scope: system
      lifecycle: monotone
      reset_owner: none
      generation: topk * combine_generation
      producers: [second_compute_transport]
      consumers: [output_reduce]
      publish: atomic_add_system_once_per_accepted_route
      wait: greater_than(topk * combine_generation - 1)
    - id: combine_claim_generation
      element_type: i32
      capacity: two
      scope: agent
      lifecycle: mixed
      reset_owner: dispatch_plan
      generation: monotone_second_slot
      producers: [dispatch_plan, output_reduce]
      consumers: [output_reduce]
      publish: block_claim_plus_owner_generation_increment
      wait: none

  queues:
    - id: first_compute_queue
      work_domain: first_stage_m_tiles
      claim_unit: m_tile
      owner: workgroup
      shards: 4
      stride_bytes: 64
      chunk_policy: one_m_tile_all_n_stripes
      ready_when: dispatch plan and payload tile are ready
      priority: 1
    - id: second_compute_queue
      work_domain: second_stage_units
      claim_unit: linear_tile_pair
      owner: workgroup
      shards: 4
      stride_bytes: 64
      chunk_policy: 16 units for large bucket, otherwise 1
      ready_when: every distinct first-stage dependency touched by the claim is ready
      priority: 0
    - id: output_reduce_queue
      work_domain: output_tokens
      claim_unit: num_waves work items per block claim
      owner: workgroup
      shards: 1
      stride_bytes: 4
      chunk_policy: one block claim maps one item per wave
      ready_when: local workgroup has drained both compute queues
      priority: 2

  events:
    - id: quant_before_persistent_kernel
      producer: input_quantize
      consumer: dispatch_plan
      scope: stream
      counter: ""
      publish: same_stream_launch_or_event_record
      wait: stream_order_or_event_wait
    - id: plan_ready
      producer: dispatch_plan
      consumer: first_compute
      scope: system
      counter: launch_gate
      publish: release_after_counter_and_padding_initialization
      wait: acquire_exact_epoch
    - id: first_to_second_item_ready
      producer: first_compute
      consumer: second_compute_transport
      scope: system
      counter: first_tile_ready
      publish: write_through_then_waitcnt_barrier_thread0_system_store_stage1_n_tiles
      wait: each_distinct_dependency
    - id: second_to_reduce_item_ready
      producer: second_compute_transport
      consumer: output_reduce
      scope: system
      counter: peer_token_ready
      publish: tile_close_then_remote_token_atomic
      wait: generation_qualified_token_threshold

  abi:
    entry_point: stage1_hosted_persistent_kernel
    arguments:
      - id: stage1_runtime
        type: pointer_group
        role: first_stage_inputs_and_outputs
        source: operator_state
        consumers: [dispatch_plan, first_compute]
        optional: false
      - id: stage2_runtime
        type: pointer_group
        role: second_stage_inputs_outputs_and_readiness
        source: operator_state
        consumers: [second_compute_transport]
        optional: false
      - id: combine_runtime
        type: pointer_group
        role: reduce_inputs_outputs_claim_and_generation
        source: operator_state
        consumers: [output_reduce]
        optional: false
      - id: max_m_blocks
        type: i32
        role: active_capacity
        source: selected_runtime_shape
        consumers: [second_compute_transport]
        optional: false
    parameters:
      direct_fused_args: true
      stage2_pointer_count: 12
      combine_pointer_count: 7
      optional_quant_pointer_count: 0
      disabled_placeholders: true
      preserve_argument_order: true

  resources:
    arch: gfx950
    workgroup:
      wave_size: 64
      wave_count: 8
      thread_count: 512
    local_memory:
      total_bytes: 160400
      limit_bytes: 163840
      allocation_rule: max
      allocations:
        - id: aliased_compute_role_storage
          bytes: 131728
          alias_group: compute_role_pool
        - id: first_stage_scale_storage
          bytes: 28672
          alias_group: ""
    registers:
      vgpr_max: null
      sgpr_max: null
      scratch_bytes_max: null
      source: emitted_artifact_required
    occupancy:
      compute_units: 256
      grid_blocks: 256
      min_workgroups_per_cu: 1
      requires_full_grid_residency: false
    engines:
      - region: input_quantize
        pipe: vector_memory
      - region: dispatch_plan
        pipe: scalar_memory
      - region: first_compute
        pipe: matrix
      - region: second_compute_transport
        pipe: matrix_interconnect
      - region: output_reduce
        pipe: vector_interconnect
    parameters:
      stage1_pool_bytes: 131072
      stage2_slab_bytes: 131728
      role_halves: 1
      additive_bytes: 28672
      compiled_group_segment_required: true

  schedule:
    primary_loop:
      kind: unified
      carried_state: [consumer_active, g2_pend, g2_next]
      queue_priority:
        - second_compute_queue
        - first_compute_queue
        - output_reduce_queue
      progress_invariants:
        - a non-preempting cohort continues producing first-stage dependencies
        - every actual second-stage claim waits on all distinct dependencies it touches
        - a workgroup enters output reduction only after draining local compute work
        - no workgroup-wide barrier is reached divergently
    policies:
      second_stage_preemption: route_gated_non_destructive_peek_then_actual_claim
      second_stage_continuation: carried_in_outer_loop_without_nested_dynamic_loop
      output_reduce_claim: one_agent_atomic_per_workgroup_then_wave_partition
      role_transition: dynamic_after_local_queue_drain
      completion_publication_granularity: per_m_tile
      first_compute_claim_domain: m_tile
      first_compute_stripe_ownership: all_n_tiles_per_claim
      completion_publication_operation: waitcnt_barrier_thread0_system_store_n_tiles
      completion_rmw_in_unified_loop: forbidden_transitively
    parameters:
      combine_third_queue: true
      g2_chunk_large: 16
      g2_chunk_small: 1
      preemption_interval: 6
      skew_num: 5
      skew_den: 4
      work_shards: 4

  compiler_constraints:
    - id: nested_shared_stage2_emitter
      rule: use one host factory with a nested JIT body shared by standalone and fused callers
    - id: no_opaque_dynamic_config
      rule: compile-time constants remain closure values and every IR-affecting value enters cache identity
    - id: wave_partition
      rule: every activation-to-local-memory load forwards the selected wave count
    - id: relocatable_local_memory
      rule: every helper local-memory address is relative to the caller supplied slab base
    - id: bounded_loop_state
      rule: only consumer_active, g2_pend and g2_next are loop-carried
    - id: block_uniform_barriers
      rule: every workgroup barrier is reached uniformly, including continuation paths
    - id: owned_mtile_completion
      rule: one workgroup claims one m-tile, computes every N stripe, then thread 0 publishes stage1_n_tiles with an ordered system store; no completion-address RMW is reachable from the fused region

  evidence_requirements:
    activation: fused path marker from all eight ranks
    accuracy: direct graph-captured relL2 below 0.10 for fixed, compact and large shapes
    launch_count: two launches per BF16 rank, quantization plus persistent kernel
    liveness: at least 256 bounded graph replays with route mutation and arrival jitter
    artifact_identity: baseline and candidate use distinct JIT and ISA identities
    resources: emitted group-segment, VGPR, SGPR, scratch and occupancy evidence
    performance: paired rank-max operator timing on target and regression guards

  known_unknowns:
    - id: emitted_register_pressure
      why: source arithmetic cannot prove VGPR occupancy or scratch use
      what_would_settle_it: compiled artifact metadata and occupancy report
    - id: live_route_distribution
      why: nominal token shape does not determine expert skew
      what_would_settle_it: current-run route histogram

candidate_templates:
  - id: full_persistent_pipeline
    priority: 100
    candidate_source: search
    target_topology:
      launch_count: 2
      included_regions:
        - input_quantize
        - dispatch_plan
        - first_compute
        - second_compute_transport
        - output_reduce
      included_queues:
        - first_compute_queue
        - second_compute_queue
        - output_reduce_queue
      capabilities:
        - item_level_overlap
        - direct_fused_abi
        - monotone_cross_rank_readiness
        - graph_replay_safe
      require_overlap: true
      parameters:
        quantization_remains_separate: true
        combine_fused_by_default: true
    checkpoints:
      - host_wiring
      - shared_emitters
      - startup_state
      - unified_compute_drain
      - output_reduce_queue
      - bucket_safety
      - full_path_validation
    completion_rule: all checkpoints coexist in one reachable and current-revision candidate

  - id: measured_partial_fallback
    priority: 20
    candidate_source: search
    rung_deviation: full target remains open; this is a complete measurable fallback
    target_topology:
      launch_count: 3
      included_regions:
        - input_quantize
        - dispatch_plan
        - first_compute
        - second_compute_transport
        - output_reduce
      included_queues:
        - first_compute_queue
        - second_compute_queue
      capabilities:
        - item_level_overlap
        - direct_fused_abi
      require_overlap: true
      parameters:
        output_reduce_separate_launch: true
    admissible_only_when:
      - full template has a recorded correctness or resource blocker
      - external operator remains complete and faster than frozen baseline

failure_routes:
  - id: repair_plan_binding
    match:
      check_ids:
        - plan_ir_v2
        - matched_skill_revision
        - matched_skill_id
        - skill_bundle_identity
        - planner_extension_identity
        - target_two_launches
        - work_domains_declared
        - buffers_declared
        - counters_declared
        - experimental_validation_status
        - experimental_never_auto_applies
        - experimental_pin_modes
        - completion_publication_granularity
        - first_compute_claim_domain
        - first_compute_stripe_ownership
        - completion_publication_operation
        - completion_rmw_in_unified_loop
        - first_stage_domain_is_mtile
        - first_stage_domain_extent
        - first_stage_owned_stripes
        - first_compute_region_domain
        - first_compute_queue_domain
        - first_compute_queue_claim
        - first_compute_queue_chunk
    repair_intent: rebuild MegaPlanIR from this extension and current frozen-source evidence before authoring
    checkpoint: host_wiring
    focus_files: []
    proof: current revision PlanIR passes generic relation checks and the Skill contract

  - id: repair_host_activation
    match:
      check_ids:
        - host_full_fusion_path
        - host_variant_controls_are_live
    repair_intent: wire one complete activation decision through host configuration, direct arguments and return path
    checkpoint: host_wiring
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
    proof: selected full path reaches the persistent launcher and omits the trailing reduce launch

  - id: repair_direct_abi
    match:
      check_ids:
        - no_private_token_ready_shadow
        - host_ready_pointer_identity
        - host_fuse_combine_direct_abi
        - direct_fused_kernel_abi
        - active_capacity_arguments_are_consumed
        - stage2_max_m_blocks_direct_abi_value
        - ordered_abi
    repair_intent: make every active buffer, readiness pointer and capacity a typed direct launcher argument in identical host-kernel order
    checkpoint: host_wiring
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: call-keyword identity and active runtime-value consumption checks pass

  - id: repair_counter_abi
    match:
      check_ids:
        - stage2_max_m_blocks_are_block_units
        - stage2_counter_storage_covers_sharded_heads
        - zero_based_first_tile_ready_layout
        - zero_based_first_tile_ready_all_reads
        - tile_close_counter_covers_routed_rows
        - fixed_slot_group_done_capacity
    repair_intent: express max_m_blocks in Stage2 block units, allocate every sharded head and routed-row tile-close slot, and keep completion counters disjoint from claim heads
    checkpoint: startup_state
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: host allocation bounds every device index and the completion/head regions do not overlap

  - id: repair_state_lifecycle
    match:
      check_ids:
        - no_host_counter_reset_in_launch_builder
        - no_host_counter_reset_on_hot_path
        - device_generation_and_token_padding
        - combine_generation_published_before_startup_gate
        - g2_completion_and_heads_reset_before_plan
        - monotone_token_ready_wait
    repair_intent: separate per-launch reset state from monotone generations and initialize each state before its publication gate
    checkpoint: startup_state
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: repeated graph launches advance generations without stale finite output or host hot-path reset

  - id: repair_compiler_shape
    match:
      check_ids:
        - nested_stage2_emitter
        - every_a_lds_load_forwards_nw
        - stage2_metadata_lds_is_slab_relative
        - stage2_emitter_metadata_lds_is_slab_relative
        - stage1_peer_lds_is_slab_relative
        - unified_loop_starts_with_lds_hazard_barrier
        - fused_stage1_jit_identity_covers_runtime_shape
    repair_intent: restore the shared nested emitter, explicit wave partition and relocatable local-memory addressing before adding schedule logic
    checkpoint: shared_emitters
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
      - aiter/ops/flydsl/kernels/mega_moe/gemm2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: standalone and fused callers compile the same body at a nonzero local-memory slab base

  - id: repair_unified_schedule
    match:
      check_ids:
        - unified_gemm_queue
        - expert_skew_preemption
        - sharded_g2_heads
        - g2_peek_bounds_dominate_completion_read
    repair_intent: use one outer kind-unit loop with carried continuation, sharded claims and a producer-preserving preemption cohort
    checkpoint: unified_compute_drain
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: source contract passes and a trace shows interleaved first- and second-stage units

  - id: repair_dependency_publication
    match:
      check_ids:
        - chunk_all_dependencies
        - g1_write_through_publication
        - g1_owned_mtile_completion
        - no_per_tile_system_flush
        - stage2_tile_close_publication
        - combine_transport_and_work_domain
        - blockwise_transport_selector_is_exact
        - fuse_combine_controls_token_publication
        - token_ready_payload_loads_are_system_scope
    repair_intent: make one workgroup own every N stripe of an m-tile, then publish stage1_n_tiles with waitcnt, barrier and one thread-0 system store; reject any completion-address RMW reachable from the fused region
    checkpoint: unified_compute_drain
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
      - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    proof: direct correctness and route-mutating replay liveness pass without global drains

  - id: repair_resource_geometry
    match:
      check_ids:
        - selected_stage2_geometry
        - aliased_lds_pool
        - aliased_lds_arithmetic
        - lds_within_limit
        - lds_alias_by_max
        - scratch_unverified_or_zero
        - resident_workgroup
    repair_intent: derive selected bucket geometry, alias role storage by max and use emitted resource evidence rather than a generic local-memory cutoff
    checkpoint: bucket_safety
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    proof: emitted group segment stays within the architecture limit with no scratch and viable occupancy

  - id: repair_reduce_partition
    match:
      check_ids:
        - adaptive_combine_partition
        - adaptive_combine_partition_bounds
        - adaptive_combine_values_are_consumed
        - blockwise_fp8_reduce_is_decoded
        - blockwise_fp8_row_layout_matches_producer
        - blockwise_scale_never_uses_readiness_pointer
        - combine_item_has_no_workgroup_barrier
        - standalone_and_fused_share_combine_item
        - standalone_combine_reaches_shared_reduce_helper
        - combine_queue_consumes_output_work_domain
        - combine_queue_rejects_routed_row_bound
        - combine_third_queue
        - combine_pointer_count
        - device_fuse_combine_guard
    repair_intent: share the production reducer and claim output work once per block before wave-level partition
    checkpoint: output_reduce_queue
    focus_files:
      - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    proof: every valid token receives exactly topk arrivals and the persistent path has no trailing reduce launch

runtime_feedback:
  - metric: compiled_group_segment_bytes
    plan_path: resources.local_memory.total_bytes
    failure_category: resource
    response: retile or reduce metadata only when emitted bytes exceed the target limit
  - metric: scratch_bytes
    plan_path: resources.registers.scratch_bytes_max
    failure_category: compiler
    response: reduce live range across the unified outer loop before changing topology
  - metric: resident_workgroups_per_cu
    plan_path: resources.occupancy.min_workgroups_per_cu
    failure_category: resource
    response: preserve at least one resident workgroup and recompute grid progress assumptions
  - metric: first_to_second_wait_cycles
    plan_path: schedule.parameters
    failure_category: schedule
    response: inspect claim chunk boundaries, readiness coverage and producer cohort before tuning preemption
  - metric: gemm1_publish_atomic_frame_fault
    plan_path: schedule.primary_loop
    failure_category: compiler
    response: remove the completion RMW, claim one m-tile per workgroup, compute every N stripe, and publish stage1_n_tiles with one ordered owner system store; require an on-card retry
  - metric: peer_token_wait_cycles
    plan_path: schedule.parameters
    failure_category: schedule
    response: inspect tile-close publication and route arrival completeness before tuning reducer partition
  - metric: paired_rank_max_speedup
    plan_path: evidence_requirements.performance
    failure_category: performance
    response: keep any verified above-baseline fallback and tune only after correctness and liveness are closed

anti_patterns:
  - separate first-stage and second-stage drain loops
  - fused runtime pointers hidden in a dispatch table
  - module-level replacement for the validated nested dynamic emitter
  - agent-scope publication consumed by a cross-domain relaxed wait
  - readiness inferred from only the numerically greatest dependency in a sharded producer graph
  - completion-address system or agent atomic RMW reachable anywhere from the fused unified region
  - per-stripe remote token publication
  - host reset of monotone cross-rank readiness
  - fixed large-shape tuning forced through every token bucket
  - performance measurement of an incomplete source checkpoint
```

## Embedded contract

```expert-skill-contract
schema_version: expert-skill-contract-v1
skill_id: megamoe_ep_mega_fusion
revision: mega-ep-fusion-v6
description: Declarative source and MegaPlanIR preflight for the experimental v6 transfer target.

source:
  include:
    - aiter/ops/flydsl/kernels/mega_moe/*.py
    - aiter/ops/flydsl/kernels/communication_ops_utils.py
    - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py
  required:
    - aiter/ops/flydsl/kernels/mega_moe/gemm2.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
  core_identity:
    - aiter/ops/flydsl/kernels/mega_moe/gemm2.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py

pins:
  enforce: true
  baseline_tree_sha256: 54a1711b081b17af7a9744113d707a32f68b90a5a0272f4bf6d03ab14b9fa6ef
  reference_tree_sha256: 53cc0eca0c814e8344600dc2064b6b4ba57683286e51dd6453e015de44dfe7ec
  reference_revision: m25-measured-reference-v1

provenance:
  require_manifest: true

copy_detection:
  exact_similarity: 0.985
  suspect_similarity: 0.90
  exact_function_ratio: 0.80
  suspect_function_ratio: 0.50

forbidden_markers:
  - AITER_MEGAMOE_G2_DIAG
  - AITER_MEGAMOE_DIAGCUT
  - AITER_MEGA_CCUT
  - AITER_MEGA_RCUT
  - AITER_MEGAMOE_SKIP_COMBINE
  - AITER_MEGAMOE_DISP_DEBUG
  - AITER_MEGAMOE_FUSE_DRAIN

plan:
  required: true
  assertions:
    - id: plan_ir_v2
      path: plan_version
      op: eq
      value: mega-plan-v2
    - id: matched_skill_revision
      path: expert_skill_revision
      op: eq
      value: mega-ep-fusion-v6
    - id: matched_skill_id
      path: expert_skill_id
      op: eq
      value: megamoe_ep_mega_fusion
    - id: skill_bundle_identity
      path: expert_skill_bundle_sha256
      op: nonempty
    - id: planner_extension_identity
      path: expert_skill_planner_extension_sha256
      op: nonempty
    - id: experimental_validation_status
      path: expert_skill_validation_status
      op: eq
      value: experimental
    - id: experimental_never_auto_applies
      path: expert_skill_auto_apply
      op: eq
      value: false
    - id: experimental_pin_modes
      path: expert_skill_explicit_pin_modes
      op: eq
      value: [authoring, candidate_validation]
    - id: target_two_launches
      path: target.launch_count
      op: eq
      value: 2
    - id: work_domains_declared
      path: work_domains
      op: nonempty
    - id: buffers_declared
      path: buffers
      op: nonempty
    - id: counters_declared
      path: counters
      op: nonempty
    - id: gfx950_arch
      path: resources.arch
      op: eq
      value: gfx950
    - id: wave64
      path: resources.workgroup.wave_size
      op: eq
      value: 64
    - id: native_wave_count
      path: resources.workgroup.wave_count
      op: in
      value: [4, 8]
    - id: threads_match_waves
      left: {path: resources.workgroup.thread_count}
      op: eq
      right:
        op: multiply
        args:
          - {path: resources.workgroup.wave_size}
          - {path: resources.workgroup.wave_count}
    - id: aliased_lds_arithmetic
      left: {path: resources.local_memory.total_bytes}
      op: eq
      right:
        op: add
        args:
          - op: max
            args:
              - {path: resources.parameters.stage1_pool_bytes}
              - op: multiply
                args:
                  - {path: resources.parameters.stage2_slab_bytes}
                  - {path: resources.parameters.role_halves}
          - {path: resources.parameters.additive_bytes}
    - id: lds_within_limit
      left: {path: resources.local_memory.total_bytes}
      op: lte
      right: {path: resources.local_memory.limit_bytes}
    - id: lds_alias_by_max
      path: resources.local_memory.allocation_rule
      op: eq
      value: max
    - id: scratch_unverified_or_zero
      path: resources.registers.scratch_bytes_max
      op: in
      value: [null, 0]
    - id: resident_workgroup
      path: resources.occupancy.min_workgroups_per_cu
      op: gte
      value: 1
    - id: unified_primary_loop
      path: schedule.primary_loop.kind
      op: eq
      value: unified
    - id: minimal_loop_carry
      path: schedule.primary_loop.carried_state
      op: eq
      value: [consumer_active, g2_pend, g2_next]
    - id: completion_publication_granularity
      path: schedule.policies.completion_publication_granularity
      op: eq
      value: per_m_tile
    - id: first_compute_claim_domain
      path: schedule.policies.first_compute_claim_domain
      op: eq
      value: m_tile
    - id: first_compute_stripe_ownership
      path: schedule.policies.first_compute_stripe_ownership
      op: eq
      value: all_n_tiles_per_claim
    - id: completion_publication_operation
      path: schedule.policies.completion_publication_operation
      op: eq
      value: waitcnt_barrier_thread0_system_store_n_tiles
    - id: completion_rmw_in_unified_loop
      path: schedule.policies.completion_rmw_in_unified_loop
      op: eq
      value: forbidden_transitively
    - id: first_stage_domain_is_mtile
      path: work_domains[id=first_stage_m_tiles].unit
      op: eq
      value: m_tile
    - id: first_stage_domain_extent
      path: work_domains[id=first_stage_m_tiles].extent
      op: eq
      value: ceildiv(num_valid, stage1_block_m)
    - id: first_stage_owned_stripes
      path: work_domains[id=first_stage_m_tiles].owned_subdomain.extent_per_claim
      op: eq
      value: stage1_n_tiles
    - id: first_compute_region_domain
      path: regions[id=first_compute].work_domain
      op: eq
      value: first_stage_m_tiles
    - id: first_compute_queue_domain
      path: queues[id=first_compute_queue].work_domain
      op: eq
      value: first_stage_m_tiles
    - id: first_compute_queue_claim
      path: queues[id=first_compute_queue].claim_unit
      op: eq
      value: m_tile
    - id: first_compute_queue_chunk
      path: queues[id=first_compute_queue].chunk_policy
      op: eq
      value: one_m_tile_all_n_stripes
    - id: combine_third_queue
      path: schedule.parameters.combine_third_queue
      op: eq
      value: true
    - id: large_chunk
      path: schedule.parameters.g2_chunk_large
      op: eq
      value: 16
    - id: small_chunk
      path: schedule.parameters.g2_chunk_small
      op: eq
      value: 1
    - id: preemption_keeps_producer_cohort
      path: schedule.parameters.preemption_interval
      op: gte
      value: 2
    - id: skew_numerator_positive
      path: schedule.parameters.skew_num
      op: gt
      value: 0
    - id: skew_denominator_positive
      path: schedule.parameters.skew_den
      op: gt
      value: 0
    - id: supported_work_shards
      path: schedule.parameters.work_shards
      op: in
      value: [1, 2, 4, 8]
    - id: direct_fused_abi
      path: abi.parameters.direct_fused_args
      op: eq
      value: true
    - id: stage2_pointer_count
      path: abi.parameters.stage2_pointer_count
      op: eq
      value: 12
    - id: combine_pointer_count
      path: abi.parameters.combine_pointer_count
      op: eq
      value: 7
    - id: optional_quant_pointer_count
      path: abi.parameters.optional_quant_pointer_count
      op: in
      value: [0, 2]
    - id: disabled_abi_placeholders
      path: abi.parameters.disabled_placeholders
      op: eq
      value: true
    - id: ordered_abi
      path: abi.arguments
      op: nonempty

checks:
  - id: host_full_fusion_path
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._run_joint
    patterns:
      - 'AITER_MEGAMOE_FUSE_ALL'
      - '_run_fused_stage1'
      - 'fuse_comb'
      - 'return out_tok'

  - id: host_variant_controls_are_live
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._run_joint
    patterns:
      - 'FUSE_COMBINE'
      - 'fuse_combine\s*=\s*fuse_comb'
      - 'if fuse_comb'

  - id: host_fuse_combine_direct_abi
    category: abi
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    call: dict
    keywords:
      fuse_combine: 'variant\.fuse_combine'

  - id: fixed_slot_group_done_capacity
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._allocate_dispatch_workspace
    patterns:
      - '[''"]group_done[''"]\s*:\s*torch\.zeros\(\s*(?:_group_done_slots\(self\.world_size\)|self\.world_size)\s*,'

  - id: no_private_token_ready_shadow
    category: abi
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    match: none
    patterns:
      - '_s1_c_tok_ready\s*=\s*torch\.zeros'

  - id: host_ready_pointer_identity
    category: abi
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    patterns:
      - 'comb_op\._fx_comb_inp'
      - 'comb_op\._fx_comb_out'
      - 'comb_op\._fx_tok_ready'
      - 'comb_op\._fx_p2p_tok_ready'
      - '_fx_fused_comb_ctr'
      - '(?:_fx_fused_comb_mtile_ctr|_s1_c_mtile_ctr\.data_ptr)'

  - id: no_host_counter_reset_in_launch_builder
    category: lifecycle
    kind: forbid_methods
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    methods: [zero_, fill_]
    receivers: ['_s1_s2ctr', '_s1_c_ctr', '_s1_c_tok_ready', '_s1_c_mtile_ctr']

  - id: no_host_counter_reset_on_hot_path
    category: lifecycle
    kind: forbid_methods
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._run_fused_stage1
    methods: [zero_, fill_]
    receivers: ['ctr', 'ready', 'epoch']

  - id: stage2_counter_storage_covers_sharded_heads
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._build_fused_stage1
    patterns:
      - '_s1_s2ctr\s*=\s*torch\.zeros\(\s*max\(\s*1\s*,\s*self\._s1_nvm\s*\)'

  - id: tile_close_counter_covers_routed_rows
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._build_fused_stage1
    patterns:
      - '_s1_c_mtile_ctr\s*=\s*torch\.zeros\(\s*max\(\s*1\s*,\s*self\._s1_nvm\s*\)'

  - id: selected_stage2_geometry
    category: resource
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    patterns:
      - 'BM\s*=\s*stage2\.block_m'
      - 'BN\s*=\s*stage2\.block_n'
      - '_g2_active_block_m\s*=\s*stage2\.block_m'
      - '_g2_sbm\s*=\s*(?:int\()?full_config\.stage1\.sort_block_m'

  - id: stage2_max_m_blocks_are_block_units
    category: correctness
    kind: assignment_value
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    target: '(?:max_m_blocks|_g2_max_m_blocks)'
    value: '(?:ceildiv\(self\._s1_nvm,\s*(?:BM|stage2\.block_m)\)|\(self\._s1_nvm\s*\+\s*(?:BM|stage2\.block_m)\s*-\s*1\)\s*//\s*(?:BM|stage2\.block_m))'

  - id: stage2_max_m_blocks_direct_abi_value
    category: abi
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._fused_all_kwargs
    call: dict
    keywords:
      i32_max_m_blocks: 'fx\.Int32\((?:max_m_blocks|_g2_max_m_blocks)\)'

  - id: direct_fused_kernel_abi
    category: abi
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1
    patterns:
      - 's2_aq'
      - 's2_ctr'
      - 'i32_s2_maxmb'
      - 'c_tok_ready'
      - 'c_p2p_ready'
      - 'c_mtile_ctr'

  - id: nested_stage2_emitter
    category: compiler
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    patterns:
      - 'def emit_stage2_body'
      - 'def _emit_stage2_body'
      - 'p2p_scatter_epilog'
      - 'publish_tok_ready'

  - id: every_a_lds_load_forwards_nw
    category: compiler
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    call: 'issue_a_load_lds_dt'
    policy: every
    keywords:
      NW: '.+'

  - id: unified_gemm_queue
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'while consumer_active'
      - 'g2_pend'
      - 'g2_next'
      - 'S2\[[''"]emit[''"]\]'

  - id: sharded_g2_heads
    category: performance
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'g2_head\s*=\s*s2_ctr'
      - 'i32_s2_maxmb'
      - 'work_shard.*fx\.Int64\(64\)'
      - 'work_shard\s*\+\s*local\s*\*\s*fx\.Int32\(WORK_SHARDS\)'

  - id: zero_based_first_tile_ready_layout
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'def _g2_comp_addr\(tile_index\):[\s\S]{0,180}return s2_ctr\s*\+\s*fx\.Int64\(tile_index\)\s*\*\s*fx\.Int64\(4\)'

  - id: zero_based_first_tile_ready_all_reads
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    match: none
    patterns:
      - '_buffer_load\(\s*g2_s2ctr_rsrc,\s*fx\.Int32\(1\)\s*\+\s*peek_tile'

  - id: g2_peek_bounds_dominate_completion_read
    category: correctness
    kind: regex_sequence
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '(?:peek_head|peek)\s*='
      - 'if\s+(?:\(?\s*peek_head|_g2_cid\(\s*peek\s*\))\s*<\s*(?:g2_total_units|total_g2_claims)'
      - '(?:peek_done|done)\s*='

  - id: unified_loop_starts_with_lds_hazard_barrier
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'while consumer_active:\s*fx\.barrier\(\)'

  - id: device_fuse_combine_guard
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1
    patterns:
      - 'fuse_combine'
      - 'if const_expr\(fuse_combine\)'

  - id: fuse_combine_controls_token_publication
    category: lifecycle
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1
    call: make_stage2_body_emitter
    keywords:
      publish_tok_ready: 'fuse_combine'
      p2p_write_through: 'fuse_combine'

  - id: expert_skew_preemption
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_g2_metiles'
      - '_g2_skewed'
      - 'fz_epr\s*\*\s*fz_tile_m'
      - 'num_valid\s*\*\s*fx\.Int32\(_g2_bal_num\)'
      - 'and _g2_skewed'

  - id: chunk_all_dependencies
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'def _g2_chunk_ready_all\(unit\)'
      - 'def _g2_run_unit\(unit\):[\s\S]{0,320}_g2_chunk_ready_all\(unit\)'
      - '_g2_chunk_ready_all\(unit\)[\s\S]{0,1600}emit_stage2_body'
      - 'g2_next\s*=\s*did_g2\.select\(g2_next_c\s*\+\s*fx\.Int32\(1\)'
      - 'g2_pend\s*=\s*did_g2\.select\(g2_pend_c\s*-\s*fx\.Int32\(1\)'

  - id: g1_write_through_publication
    category: correctness
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    call: build_fused_gemm1
    keywords:
      out_cache_modifier: '_G1_OUT_WT'

  - id: g1_owned_mtile_completion
    category: compiler
    kind: owned_completion_protocol
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    normalize_wrappers: [Int32, Int64, int]
    region:
      guard: 'fuse_all'
      arm: body
      loop:
        test: 'consumer_active'
        contains: '\bg2_pend\b'
    claim:
      extent_target: 'g1_total_mtiles'
      extent_value: 'num_m_tiles'
      source_extent_target: 'num_m_tiles'
      source_extent_value: 'ceildiv\(num_valid, sort_block_m\)'
      helper: '_claim_gemm1_or_drain'
      owner_test: 'tid == 0'
      local_target: 'local_claim'
      atomic_call: 'atomic_add_agent'
      head_address: 'a_work_head \+ work_shard \* 64'
      increment: '1'
      index_target: 'work'
      index_value: 'work_shard \+ local_claim \* WORK_SHARDS'
      bound: 'work < g1_total_mtiles'
    handoff:
      store_call: 'ptr_store'
      scratch: 'fdrain_scratch'
      vector_call: 'from_elements'
      values: ['1', 'work', '-1']
      view_target: 'fdrain_view'
      view_value: 'fx\.make_view\(fdrain_scratch, fx\.make_layout\(3, 1\)\)'
      load_target: '_vals'
      load_value: 'Vec\(fdrain_view\.load\(\)\)'
      barrier_call: 'barrier'
      unit_target: 'unit'
      unit_value: '_vals\[1\]'
    stripes:
      mode_test: 'kind == 1'
      iterator: '_n_stripe'
      range_call: 'range'
      count: 'N_TILES'
      compute_call: '_do_scheduled_tile'
      compute_index: 'unit \* N_TILES \+ _n_stripe'
    publication:
      helper: '_g2_publish_mtile_done'
      argument: 'unit'
      wait_call: 's_waitcnt'
      wait_argument: '0'
      barrier_call: 'barrier'
      owner_test: 'tid == 0'
      store_call: 'store_i32_system'
      address_call: '_g2_comp_addr'
      address_argument: 'm_tile'
      direct_address_patterns:
        - '(?:s2_ctr|g2_ctr_i64) \+ .*\b(?:unit|m_tile|tile_index)\b.* \* 4'
        - '4 \* .*\b(?:unit|m_tile|tile_index)\b.* \+ (?:s2_ctr|g2_ctr_i64)'
      address_base_patterns: ['^(?:s2_ctr|g2_ctr_i64)$']
      address_index_patterns:
        - '\b(?:unit|m_tile|tile_index)\b.* \* 4'
        - '4 \* .*\b(?:unit|m_tile|tile_index)\b'
      store_offset: '0'
      store_value: 'N_TILES'
      forbidden_rmw_calls:
        - atomic_add_system
        - atomic_add_agent
        - atomic_add_global_at
        - AtomicRMWOp
      forbidden_calls: [fence_system_release]

  - id: no_per_tile_system_flush
    category: performance
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    match: none
    patterns:
      - 'fence_system_release\(\)[\s\S]{0,220}(?:atomic_add_system|store_i32_system)\([\s\S]{0,120}(?:g2_ctr|_g2_comp)'

  - id: device_generation_and_token_padding
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_c_rsrc\s*=\s*_make_buffer_from_addr\(c_ctr,\s*fx\.Int32\)'
      - '_buffer_store\(\s*_c_rsrc,\s*fx\.Int32\(1\),[\s\S]{0,240}_buffer_load\(\s*_c_rsrc,\s*fx\.Int32\(1\)[\s\S]{0,120}\+\s*fx\.Int32\(1\)'
      - 'if compact_owner:[\s\S]{0,1000}_c_pad\s*=\s*fx\.Int32\(fz_k\)\s*\*\s*c_epoch'
      - '_c_t\s*>=\s*i32_cur_tok'

  - id: combine_generation_published_before_startup_gate
    category: lifecycle
    kind: regex_sequence
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_c_rsrc\s*=\s*_make_buffer_from_addr\(c_ctr,\s*fx\.Int32\)'
      - '_buffer_store\(\s*_c_rsrc,\s*fx\.Int32\(0\),\s*fx\.Int32\(0\)'
      - '_buffer_store\(\s*_c_rsrc,\s*fx\.Int32\(1\),[\s\S]{0,240}_buffer_load\(\s*_c_rsrc,\s*fx\.Int32\(1\)[\s\S]{0,120}\+\s*fx\.Int32\(1\)'
      - 'store_i32_system\(\s*gate_addr,\s*fx\.Int32\(0\),\s*gate_epoch'

  - id: g2_completion_and_heads_reset_before_plan
    category: lifecycle
    kind: regex_sequence
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_ctr_rsrc\s*=\s*_make_buffer_from_addr\(s2_ctr,\s*fx\.Int32\)'
      - '_buffer_store\(\s*_ctr_rsrc,\s*fx\.Int32\(_c\),\s*fx\.Int32\(0\)'
      - '_buffer_store\(\s*_ctr_rsrc,\s*i32_s2_maxmb\s*\+\s*tid\s*\*\s*fx\.Int32\(16\),\s*fx\.Int32\(0\)'
      - 'emit_dispatch_plan\('

  - id: combine_transport_and_work_domain
    category: correctness
    kind: call_keywords
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    call: make_combine_reduce_emitter
    keywords:
      blockwise_fp8_transport: '.+'
      cur_rank_num_token: 'i32_cur_tok'
      nominal_warp_num: 'launch_grid_x\s*\*\s*NUM_WAVES'
      tok_ready_addr: 'c_tok_ready'
      tok_ready_epoch: 'c_epoch'

  - id: blockwise_transport_selector_is_exact
    category: correctness
    kind: assignment_value
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    target: 'g2_blockwise_fp8'
    value: 'g2_p2p_quant_type\s*==\s*[''"]fp8_blockwise_1x32[''"]'

  - id: adaptive_combine_partition
    category: performance
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'warps_per_tok'
      - 'safe_token_count'
      - 's3_total_work\s*=\s*cur_rank_num_token\s*\*\s*warps_per_tok'
      - 'blockwise_fp8_transport'
      - 'hdim_per_warp'
      - 'def _emit_item\(s3_work_idx\)'
      - 'tok_id\s*=\s*s3_work_idx\s*//\s*warps_per_tok'
      - 'part_id\s*=\s*s3_work_idx\s*%\s*warps_per_tok'
      - 'return\s*\(\s*consts,\s*_emit_item\s*\)'

  - id: adaptive_combine_partition_bounds
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'n_elems\s*=\s*n_i32'
      - 'hdim_per_warp\s*=\s*\(\s*n_elems\s*\+\s*warps_per_tok\s*-\s*1\s*\)\s*//\s*warps_per_tok'
      - 'rem_hdim\s*=\s*n_elems\s*-\s*hdim_off'
      - 'eff_end\s*=\s*\(\s*rem_hdim\s*<\s*hdim_per_warp\s*\)\.select\(\s*rem_hdim,\s*hdim_per_warp\s*\)'
      - '_accum_step\(\s*hdim_off\s*\+\s*ec'
      - '_accum_loop\(\s*eff_end'

  - id: adaptive_combine_values_are_consumed
    category: correctness
    kind: parameter_loads
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    parameters:
      - hdim_per_warp
      - s3_total_work
    min_loads: 1

  - id: blockwise_fp8_reduce_is_decoded
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'expert_scale_rsrcs'
      - 'if const_expr\(blockwise_fp8_transport\)'
      - 'expert_tok_addr\s*\+\s*fx\.Int64\((?:hidden_dim|N_OUT)\)'
      - 'dtype=T\.i32'
      - 'scale_idx\s*=\s*\(\s*ec_abs\s*\+\s*u\s*\*\s*64\s*\)\s*//\s*8'
      - 'dtype=T\.i8'
      - '(?:cvt_pk_f32_fp8|_to_accum\s*=\s*spec\.to_accum)'
      - 'ds_bpermute'
      - 'sc_i32\s*<<\s*fx\.Int32\(23\)'
      - '_to_accum\(\s*vals\[u\]\[k_slot\]\s*\)\s*\*\s*scales\[u\]\[k_slot\]'

  - id: token_ready_payload_loads_are_system_scope
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'IN_CACHE\s*=\s*_SLC_CACHE\s*\|\s*1\s*\|\s*16\s*if\s*tok_ready_addr\s+is\s+not\s+None\s+else\s+_SLC_CACHE'
      - 'cache_modifier=IN_CACHE'

  - id: blockwise_fp8_row_layout_matches_producer
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'nbytes\s*=\s*(?:spec\.nbytes|(?:N_OUT|hidden_dim)\s*\+\s*(?:N_OUT|hidden_dim)\s*//\s*32)'
      - 'slot_idx\s*=\s*tok_id\s*\*\s*(?:TOPK|experts_per_token)\s*\+\s*k_slot'
      - 'expert_tok_off\s*=\s*fx\.Int64\(slot_idx\)\s*\*\s*nbytes'
      - 'expert_tok_addr\s*=\s*(?:c_inp|addr_shmem_tok)\s*\+\s*expert_tok_off'

  - id: blockwise_scale_never_uses_readiness_pointer
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    match: none
    patterns:
      - 'create_buffer_resource_from_addr\(\s*c_peer_ready\s*\)'

  - id: combine_item_has_no_workgroup_barrier
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter._emit_item
    match: none
    patterns:
      - 'fx\.barrier\('

  - id: standalone_and_fused_share_combine_item
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: emit_combine_barrier_and_reduce
    patterns:
      - 'make_combine_reduce_emitter\('

  - id: standalone_combine_reaches_shared_reduce_helper
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_kernel.ep_combine_intranode
    patterns:
      - 'emit_combine_barrier_and_reduce\('

  - id: combine_queue_consumes_output_work_domain
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '\(\s*_c_consts,\s*_c_emit\s*\)\s*=\s*make_combine_reduce_emitter'
      - '_c_total\s*=\s*_c_consts\[[''"]s3_total_work[''"]\]'
      - '_c_item\s*=\s*_c_base\s*\+\s*_c_wave'
      - 'if _c_item\s*<\s*_c_total'
      - '_c_emit\(\s*_c_item\s*\)'
      - 'comb_active\s*=\s*_c_base\s*<\s*_c_total'

  - id: combine_queue_rejects_routed_row_bound
    category: correctness
    kind: assignment_value
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    target: '(?:c_total_tokens|_c_total)'
    value: 'g2_num_valid'
    match: none

  - id: monotone_token_ready_wait
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'tok_ready_expected'
      - 'tok_ready_epoch'
      - 'int32_wait_until_greater_than'

  - id: stage2_tile_close_publication
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    patterns:
      - 'arg_mtile_ctr'
      - 'num_n_blocks\s*-\s*fx\.Int32\(1\)'
      - 'atomic_add_system'
      - 'arg_p2p_tok_ready'

  - id: stage2_metadata_lds_is_slab_relative
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: p2p_scatter_epilog
    patterns:
      - 'lds_typed_ptr\(\s*lds_acc_base\s*\+\s*lds_packed_off'
      - 'lds_typed_ptr\(\s*lds_acc_base\s*\+\s*lds_weight_off'
      - 'lds_typed_ptr\(\s*lds_acc_base\s*\+\s*lds_peer_off'

  - id: stage2_emitter_metadata_lds_is_slab_relative
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    patterns:
      - 'lds_typed_ptr\(\s*_?lds_base_i32\s*\+\s*fx\.Int32\(lds_peer_off\)'
      - 'lds_typed_ptr\(\s*_?lds_base_i32\s*\+\s*fx\.Int32\(lds_packed_off\)'
      - 'lds_typed_ptr\(\s*_?lds_base_i32\s*\+\s*fx\.Int32\(lds_weight_off\)'

  - id: stage1_peer_lds_is_slab_relative
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    match: none
    patterns:
      - '_s2_lds_typed_ptr\(\s*fx\.Int32\(g2_lds_peer_off\)'

  - id: aliased_lds_pool
    category: resource
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1
    patterns:
      - 'lds_pool_bytes\s*=\s*max\('
      - 'S2\[[''"]slab[''"]\]'

  - id: active_capacity_arguments_are_consumed
    category: abi
    kind: parameter_loads
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    parameters:
      - [s2_metiles, s2_max_expert_tiles]
      - [i32_s2_maxmb, i32_max_m_blocks]
    min_loads: 1

  - id: fused_stage1_jit_identity_covers_runtime_shape
    category: compiler
    kind: assignment_value
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1
    target: kernel_name
    value: '(?=[\s\S]*fused_g2_pref)(?=[\s\S]*fuse_combine)(?=[\s\S]*G2_CHUNK)'
```

## Embedded runtime validation

```expert-skill-runtime-python
#!/usr/bin/env python3
"""Direct graph correctness and arrival-jitter liveness for MegaMoE V2 EP8.

The tool is repository-portable: the candidate tree is an argument and runtime
libraries/JIT cache come from the caller's environment. It imports the frozen
task's own reference helpers from the candidate copy, captures the real
MegaMoEV2 call, compares graph output directly with the numeric reference, and
varies routing buffers between replays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--network", default="v4_pro")
    parser.add_argument("--accuracy-cases", default="128,512,8192")
    parser.add_argument("--liveness-cases", default="128,512,8192")
    parser.add_argument("--routes", default="uniform,rank-mixed-skew")
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--numeric-checkpoint-interval", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    tree = Path(args.candidate_tree).resolve()
    test_path = tree / "op_tests/multigpu_tests/test_mega_moe_v2.py"
    if not test_path.is_file():
        raise SystemExit(f"candidate test helper missing: {test_path}")
    sys.path.insert(0, str(tree))

    import torch
    import torch.distributed as dist
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    helper = _load_module(test_path, "_geak_mega_reference")
    accuracy_cases = _csv_ints(args.accuracy_cases)
    liveness_cases = _csv_ints(args.liveness_cases)
    routes = _csv_strings(args.routes)
    if not accuracy_cases or not liveness_cases or not routes:
        raise ValueError("accuracy/liveness cases and routes must be non-empty")
    if args.replays <= 0:
        raise ValueError("--replays must be positive")
    if args.numeric_checkpoint_interval <= 0:
        raise ValueError("--numeric-checkpoint-interval must be positive")

    rank, world, device = helper._setup_dist()
    if world != 8:
        raise ValueError(f"MegaMoE contract requires world=8, got {world}")

    rows: list[dict] = []
    replay_rows: list[dict] = []
    result = {
        "schema_version": "geak-megamoe-graph-contract-v1",
        "claim_complete": False,
        "world_size": world,
        "activation": os.environ.get("AITER_MEGAMOE_FUSE_ALL", ""),
        "accuracy_results": rows,
        "replay_results": replay_rows,
    }

    def write_result(complete: bool):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2) + "\n")
            os.replace(tmp, output)

    def make_route(
        tokens: int, route: str, iteration: int, network: dict, include_x: bool = True
    ):
        generator = torch.Generator(device=device).manual_seed(
            args.seed + rank + iteration * 1009 + tokens * 17
        )
        x = (
            torch.randn(
                (tokens, network["model_dim"]),
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            if include_x
            else None
        )
        destination_scores = torch.rand(
            (tokens, world), device=device, generator=generator
        )
        destination = torch.topk(
            destination_scores, network["topk"], dim=-1
        ).indices
        local_experts = network["experts"] // world
        if route == "uniform":
            local = torch.randint(
                0,
                local_experts,
                (tokens, network["topk"]),
                device=device,
                generator=generator,
            )
        elif route == "rank-mixed-skew":
            hot = destination < world // 2
            cold = torch.randint(
                1,
                local_experts,
                (tokens, network["topk"]),
                device=device,
                generator=generator,
            )
            local = torch.where(hot, torch.zeros_like(cold), cold)
        else:
            raise ValueError(f"unsupported route {route!r}")
        ids = destination * local_experts + local
        logits = torch.randn(
            (tokens, network["topk"]),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return (
            x.contiguous(),
            logits.softmax(dim=-1).contiguous(),
            ids.to(torch.int32).contiguous(),
        )

    def relative_l2(output, reference) -> float:
        numerator = torch.linalg.vector_norm(output.float() - reference)
        denominator = torch.linalg.vector_norm(reference)
        value = float((numerator / denominator).item())
        return helper._reduce_float(value, device, dist.ReduceOp.MAX)

    try:
        network = helper.NETWORKS[args.network]
        if network["experts"] % world:
            raise ValueError("expert count must divide world size")
        local_experts = network["experts"] // world
        packed = helper._quantize_weights(
            network["model_dim"],
            network["inter_dim"],
            local_experts,
            rank,
            args.seed,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale

        all_cases = sorted(set(accuracy_cases + liveness_cases))
        for tokens in all_cases:
            max_tok_per_rank = max(16, _next_power_of_two(tokens))
            moe = MegaMoEV2(
                rank=rank,
                world_size=world,
                quant="a8w4",
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=max_tok_per_rank,
                **network,
            )
            x, route_weights, ids = make_route(tokens, "uniform", 0, network)
            state = {}

            def body():
                state["output"] = moe(x, route_weights, ids)[:tokens]

            helper._barrier()
            body()
            helper._barrier()
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            with torch.cuda.graph(graph, stream=capture_stream):
                body()
            graph.replay()
            torch.cuda.synchronize()

            if tokens in accuracy_cases:
                reference = helper._reference(
                    x,
                    route_weights,
                    ids,
                    ref_weights,
                    rank,
                    world,
                    network["model_dim"],
                    network["inter_dim"],
                    network["experts"],
                    network["swiglu_limit"],
                )
                rel_l2 = relative_l2(state["output"], reference)
                row = {
                    "guard": str(tokens),
                    "route": "uniform",
                    "metric": "relL2",
                    "value": rel_l2,
                    "threshold": args.rtol,
                    "method": "direct graph-captured candidate vs numeric reference",
                    "status": "pass" if rel_l2 < args.rtol else "fail",
                }
                rows.append(row)
                write_result(False)
                if rel_l2 >= args.rtol:
                    raise AssertionError(
                        f"tokens={tokens} direct graph relL2={rel_l2:.6f}"
                    )

            if tokens in liveness_cases:
                for route in routes:
                    checkpoint_rel_l2: list[float] = []
                    for replay in range(args.replays):
                        _unused_x, new_weights, new_ids = make_route(
                            tokens, route, replay + 1, network, include_x=False
                        )
                        route_weights.copy_(new_weights)
                        ids.copy_(new_ids)
                        graph.replay()
                        checkpoint = (
                            (replay + 1) % args.numeric_checkpoint_interval == 0
                            or replay + 1 == args.replays
                        )
                        if checkpoint:
                            torch.cuda.synchronize()
                            reference = helper._reference(
                                x,
                                route_weights,
                                ids,
                                ref_weights,
                                rank,
                                world,
                                network["model_dim"],
                                network["inter_dim"],
                                network["experts"],
                                network["swiglu_limit"],
                            )
                            replay_rel_l2 = relative_l2(state["output"], reference)
                            checkpoint_rel_l2.append(replay_rel_l2)
                            if replay_rel_l2 >= args.rtol:
                                raise AssertionError(
                                    f"tokens={tokens} route={route} replay={replay + 1} "
                                    f"relL2={replay_rel_l2:.6f}"
                                )
                    torch.cuda.synchronize()
                    finite = bool(torch.isfinite(state["output"]).all().item())
                    finite_all = torch.tensor(
                        int(finite), dtype=torch.int32, device=device
                    )
                    dist.all_reduce(finite_all, op=dist.ReduceOp.MIN)
                    row = {
                        "guard": f"{tokens}_{route}",
                        "count": args.replays,
                        "status": "pass" if int(finite_all.item()) == 1 else "fail",
                        "graph_safe": "pass",
                        "arrival_jitter": True,
                        "routing_changes": args.replays,
                        "numeric_checkpoints": len(checkpoint_rel_l2),
                        "max_checkpoint_relL2": max(checkpoint_rel_l2),
                        "threshold": args.rtol,
                    }
                    replay_rows.append(row)
                    write_result(False)
                    if row["status"] != "pass":
                        raise AssertionError(
                            f"non-finite graph output after {tokens}/{route}"
                        )

            del moe, graph, state
            torch.cuda.empty_cache()
            helper._barrier()

        result["correctness"] = "pass"
        result["graph_safe"] = "pass"
        result["liveness"] = "pass"
        write_result(True)
        if rank == 0:
            print(
                f"[MEGA-GRAPH-CONTRACT] PASS accuracy={len(rows)} "
                f"replay_cases={len(replay_rows)} replays={args.replays}",
                flush=True,
            )
        return 0
    finally:
        helper._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
```
