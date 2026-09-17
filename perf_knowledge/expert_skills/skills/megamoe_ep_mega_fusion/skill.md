---
id: megamoe_ep_mega_fusion
title: MegaMoE EP8 two-launch persistent fusion
kind: expert_skill
mode: mega
authors: [GEAK Team]
scope: kernel
match:
  operator: moe_dispatch_combine
  arch_class: ['*']
  gens: [gfx950]
  dtypes: [mxfp8_e4m3, mxfp4]
  regimes: [prefill, decode]
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
expects:
  isolated_speedup_min: 1.03
  e2e_delta_min_pct: 1.0
  parity: required
role: experimental_authoring_prior
normative: true
normative_scope: explicitly_pinned_authoring
embedded_components: [planner_extension, contract]
validation:
  schema_version: expert-skill-validation-v1
  skill_id: megamoe_ep_mega_fusion
  status: experimental
  reference_evidence:
    status: measured
  constraint_validation:
    status: static_validated
  usage:
    auto_apply: false
    explicit_pin_modes: [authoring, candidate_validation]
  measured:
    target: 8192_uniform_rank_max
    reference_speedup: 1.0448
    parity: pass
---

## When to use

Use only for MegaMoE V2 on one EP8 gfx950 node when the requested result is
the complete two-launch persistent path. Quantization remains a separate launch; dispatch,
GEMM1, GEMM2, P2P publication and combine must execute in one persistent
megakernel.

This Skill is parsed from the final measured persistent implementation. The reference
may be read during offline Skill maintenance and post-authoring verification,
but it is not an Engineer input. Analyze, Plan and Engineer receive this Skill
and the frozen MegaMoEV2 baseline only.

## Mechanism

The target is exactly two BF16-path launches per rank:

1. BF16 to MXFP8 quantization.
2. One persistent dispatch/GEMM1/GEMM2/P2P/combine kernel.

Implement one coherent final target, not a sequence of intermediate stages:

- Dispatch initializes the current generation and publishes routed work.
- The measured GEMM1 queue is flat `(m_tile, n_stripe)` work. One CTA computes
  one stripe, retires its stores, reaches a block barrier, then one lane
  system-atomically increments the owning m-tile completion counter.
- GEMM2 consumes a tile only after the corresponding GEMM1 counter reaches the
  number of N stripes. GEMM1 and ready GEMM2 work share one CTA-local loop.
  G1 stays first by default: only `ticket % 6 == 0` may preempt it with ready
  G2 work, and only when
  `max_expert_tiles * experts_per_rank * tile_m > valid_rows * 5 / 4`.
  No grid-wide GEMM1 drain is allowed.
- Keep Stage2 constants under one derivation authority and keep route, weight,
  peer and readiness buffer resources local to the shared Stage2 body. Do not
  carry those descriptors through the outer persistent Stage1 frame.
- Specialize G2 claim size before compilation: chunk 1 below 4096 tokens and
  chunk 16 at 4096 or above. Claims are contiguous:
  `claim_id = shard + local_claim * shards`, `base = claim_id * chunk`.
  Check readiness once at the last in-range unit and carry only `next` and
  `remaining`; an unready head is pending work, never a drained queue.
- P2P payload and scale stores become system-visible before publication.
  Token readiness exists only when combine is in-kernel. Each G2 m-block closes
  locally with one agent add per N stripe; its closing CTA emits one remote
  arrival per accepted routed row.
- Token readiness is monotone and generation-qualified. A token is ready after
  exactly top-k arrivals; omitted tokens are advanced so later graph replays
  cannot wait on a stale generation.
- After a CTA has drained its local GEMM work, it irreversibly enters a
  block-claimed combine loop. There is no grid barrier. One block claim feeds
  one output item per wave.
- Derive combine partitions as
  `P = min(ceildiv(grid_waves, run_tokens), hidden_i32 / 8)`. A combine item
  maps `(output_token, partition)`, waits independently for that token
  generation, reads every top-k payload slot, accumulates in FP32, bounds its
  tail, and stores BF16 with a rank-local cache policy. At 8192 tokens,
  `grid_waves=2048`, `hidden_i32=1792`, therefore `P=1`, reduction width
  `U=4`, total work is 8192 items, and an NW8 block handles eight distinct
  tokens rather than eight partitions of one token.
- Put every compile-affecting topology or bucket choice in the JIT identity.

The active target excludes optional M3 quant ingress, wait instrumentation,
profiling code and diagnostic cuts.

## Function blueprint

Keep these responsibilities separate even if local symbol names differ:

- `MegaMoEV2._run_joint` owns the one-time path decision. It launches
  quantization, builds the selected fused configuration, launches the persistent
  kernel once, and returns the in-kernel combine output without a Stage2 or
  combine tail launch.
- The host fused-argument builder derives one Stage2 specification and one
  combine specification from the selected bucket. It passes raw tensor
  addresses and scalar capacities; it does not construct device buffer
  resources.
- `compile_mega_moe_stage1` owns the persistent kernel, shared-storage layout,
  startup roles, route-gated unified G1/G2 work loop and the irreversible
  transition into combine.
- `build_fused_gemm1` emits exactly one claimed GEMM1 output stripe. It does not
  own global queue progression or the completion threshold.
- `make_stage2_body_emitter` derives and validates Stage2 constants once, then
  emits exactly one claimed `(m_block,n_block)` unit. Route, weight, peer and
  readiness buffer resources are constructed inside this body so their SGPR
  lifetime ends with the Stage2 unit.
- `make_combine_reduce_emitter` emits exactly one
  `(output_token,partition)` item. The caller derives `P` and `U`, owns
  block-level claiming, and assigns distinct tokens to waves when `P=1`; the
  item owns its token wait, all top-k payload reads, FP32 reduction and bounded
  BF16 output store.

Standalone and fused callers must reuse the same Stage2 arithmetic and combine
arithmetic authority. They may use different work distributors, but must not
rederive row layouts, scale offsets or output partitioning independently.

## Baseline math, ABI and state layout

Reuse MegaMoEV2's existing numerical bodies:

- keep the standalone BF16-to-MXFP8 `per_1x32` quantization launch;
- reuse the existing GEMM1 MFMA, activation/scale layout and weight shuffle;
- reuse the existing GEMM2 MFMA and weighted P2P scatter epilogue;
- reuse combine's BF16 and block-FP8 decode/FP32-accumulate rules.

The optimization is relocation and scheduling, not a new GEMM approximation.
Do not change quant formats, scale grouping, expert indexing, top-k weights or
accumulation dtype to obtain the launch reduction.

The fused host ABI carries semantic groups rather than hidden table slots:

- GEMM1 tensors and dispatch-plan state;
- raw GEMM2 activation/scale/weight and route-table addresses;
- selected BM/BN/BK, maximum m-block count and exact table capacities;
- combine payload/output, token-ready, block-claim and tile-close state.

All backing tensors remain strongly owned by the MegaMoEV2/operator instance
through graph replay. Disabled optional roles may be zero only when compiled
out; every active pointer must be nonzero and bound to the emitted argument
with matching size/alignment.

Minimum state capacities are derived from device indices:

```text
g1_ready entries       >= ceildiv(max_routed_rows, stage1_block_m)
g2 tile-close entries  >= ceildiv(max_routed_rows, stage2_block_m)
token_ready entries    >= max_tokens_per_rank
peer pointer entries   =  world_size
combine input rows     >= max(world_size * max_tokens_per_rank,
                              max_tokens_per_rank * topk)
combine output rows    >= max_tokens_per_rank
```

If G2 claim heads share an allocation with G1 completion counters, each shard
starts on its own 64-byte cache line and the regions are disjoint. Route,
weight, tile-row and peer tables carry their real byte capacities into local
Stage2 resources so a malformed index cannot become an arbitrary symmetric-heap
address.

## Scheduling blueprint

The persistent grid is a CTA-local state machine:

1. Startup elects owner/dispatch roles and publishes the current launch
   generation only after counters, padding and route state are initialized.
2. Remaining CTAs enter one G1/G2 loop.
3. A G1 claim is a flat output-stripe index:
   `unit = m_tile * n_stripes + n_stripe`.
4. After the stripe's activation/scale stores retire, all waves execute
   `s_waitcnt(0)` and a block barrier; one lane system-atomically increments
   `g1_ready[m_tile]` by one.
5. A G2 unit may execute only when its required G1 completion counter has
   reached `n_stripes`. Bounds must dominate every readiness read.
6. Route-gated non-destructive peeking may pull ready G2 work between G1
   stripes only for the `ticket % 6 == 0` cohort and only when the 5/4
   expert-tile skew predicate is true. Every other CTA remains G1-first.
7. A CTA exits the G1/G2 loop only when its local G1 work is exhausted and
   every valid G2 chunk is claimed or owned. An unready G2 head is pending, not
   drained. It never returns to GEMM work after entering combine.
8. The combine loop begins with a block barrier, lets one lane claim a block of
   output items, publishes the claim through LDS, executes another block
   barrier, then assigns one item to each wave.

There is no grid-wide barrier between G1, G2 and combine. Progress comes from
per-item readiness and monotone generations, allowing different CTAs to occupy
different phases concurrently.

A G2 work item is the linear pair
`unit = m_block * stage2_n_blocks + n_block`. Sharded heads return
`claim_id = shard + local_claim * shards`; each claim owns the contiguous range
starting at `claim_id * chunk`. The last in-range unit is bounds-checked and
readiness-tested once before the chunk executes. A CTA never blocks on G2 while
it owns an unexecuted G1 stripe. Small-shape c1 has no carried chunk; c16 carries
only the two block-uniform scalars `next` and `remaining`, never the GEMM2 live
range.

## Memory-order blueprint

- G1 activation and scale stores use the gfx950 system-visible write-through
  policy. Publication order is write-through stores, `s_waitcnt(0)`, a uniform
  block barrier, then the system completion atomic. Do not add a post-atomic
  full-L2 release.
- G1 completion is a per-stripe system atomic; GEMM2 waits for the exact
  `n_stripes` threshold.
- Compile token readiness only when combine is in-kernel. A G2 m-block closes
  in rank-local state with one agent add per N stripe. The closing CTA
  self-clears that generation and publishes one remote token arrival for each
  accepted route.
- P2P payload and block-FP8 scale stores are write-through before token-ready
  publication. Combine uses system-scope streaming payload loads; do not replace
  this with a per-token full-L2 invalidate.
- Token-ready counters are monotone across graph replays. The expected value is
  `topk * generation`; do not clear a counter while peers may already publish
  the next generation.
- Tokens omitted from the current launch are advanced to the current target
  before a later, larger launch can wait on them.
- Combine output is rank-local and uses an ordinary rank-local cache policy.

## LDS and register budget

Shared memory is reused by lifetime, not added by role:

```text
role_pool = max(gemm1_pool, stage2_slab * stage2_halves)
group_segment = role_pool + gemm1_scale_storage
```

For the large NW8/BN512 path:

```text
gemm1_pool          = 131072 B
stage2_slab         = 131728 B
stage2_halves       = 1
gemm1_scale_storage =  28672 B
group_segment       = 160400 B
gfx950 limit        = 163840 B
headroom            =   3440 B
```

The allocation must therefore remain `max(role lifetimes) + scale storage`.
Allocating G1 and G2 slabs additively exceeds the hardware limit. Every helper
LDS address is relative to the caller-provided slab base; no helper may assume
offset zero.

Measured bs=128 final-path compiler anchors are:

```text
workgroup threads   = 256 (4 waves)
group segment       = 32144 B
kernarg segment     = 440 B
VGPR                = 168
SGPR                = 106
VGPR spills         = 0
SGPR spills         = 106
private/scratch     = 0 B
```

Measured 8192 final-path compiler anchors are:

```text
workgroup threads   = 512 (8 waves)
group segment       = 160400 B
kernarg segment     = 440 B
VGPR                = 256
SGPR                = 106
VGPR spills         = 80
SGPR spills         = 116
private segment     = 316 B
resident workgroups = 1 per CU
```

These are emitted-resource envelopes, not exact source-contract values. The
large measured path has controlled spill/private storage, so a blanket
zero-spill rule is wrong. For every bucket, collect group-segment, VGPR, SGPR,
spill, private-segment and occupancy metadata. Require LDS within the gfx950
limit, at least one resident workgroup per CU, and no unexplained growth beyond
the corresponding measured envelope. Constructing Stage2 descriptors in the
outer Stage1 frame unnecessarily extends SGPR live ranges into combine and must
be avoided.

## Geometry and compile-time specialization

- Small shapes compile G2 chunk 1; shapes at or above 4096 compile chunk 16.
  Do not compile c16 and select c1 dynamically on the device.
- The large path uses native NW8 Stage2 geometry. Widen BN proportionally so
  each wave retains the intended N-column share; NW8 uses one Stage2 slab half.
- Fixed small buckets use four-wave Stage1 geometry. Large buckets use eight
  waves and one persistent workgroup per CU.
- Large-path G1 claim shards are 1 for buckets up to 32, 8 at bucket 2048 and
  4 otherwise. Fixed-slot tuning may use 8.
- Route-gated G2 preemption uses the measured interval 6 and skew ratio 5/4 as
  exact scheduling gates, then requires same-machine measurement.
- Derive combine partitions from nominal grid waves. At 8192 tokens the
  required large-shape geometry is `P=1`, `U=4`, 8192 total items and eight
  distinct tokens per NW8 block claim.
- Bucket 512 uses `payload_chunk_rows=256`; other large buckets use 384.
- Every value affecting NW, BN/BM/BK, chunking, preemption, transport format,
  combine presence or LDS layout is part of the JIT/cache identity.

## Compiler pitfalls

- A source-level two-launch declaration is not emitted-artifact evidence.
- Do not move device buffer-resource construction out of its owning role merely
  to simplify Python call signatures.
- FlyDSL rewrites dynamic branches and loops; values created only inside a
  rewritten branch must not be consumed outside it without explicit carried
  state.
- A workgroup barrier must be reached uniformly. Per-wave combine items contain
  no workgroup barrier; barriers belong to the surrounding block-claim loop.
- Do not infer safety from VGPR/SGPR counts alone. Validate the exact emitted
  artifact and execute the all-six-slot bs=128 path before broader testing.

## Procedure

Run one normal `mode=mega` lane:

1. Confirm the supplied baseline exposes MegaMoEV2 EP8 dispatch, both GEMMs,
   P2P transport and combine.
2. Plan the complete two-launch topology once.
3. Continue the same candidate until all regions coexist. Intermediate
   checkpoints are not separate candidates and are not benchmarked.
4. Run the GPU-free contract. An incomplete candidate cannot request GPUs.
5. Validate JIT and bs=128 first, then 512 and 8192.
6. Verify all eight `path=MEGA` markers, runtime launch count, graph liveness
   and paired rank-max performance.
7. If hardware fails, return the concrete failure to the same lane. Do not
   replace the complete target with an open-ended local optimization search.

## Do-no-harm notes

- Skill-off must preserve the ordinary Mega lifecycle and receive none of this
  knowledge.
- Engineer must not receive a reference path, reference source, local run IDs
  or private candidate identities.
- A structural pass means only that the complete source is ready for hardware.
- A silent scattered fallback, three-launch partial, timeout, device fault or
  missing path marker is not a valid result.
- Measure against the frozen scattered baseline on the same machine. Historical
  latency is context, never the denominator.

## Acceptance

The Skill succeeds only when its independently authored candidate has:

- `path=MEGA` from all eight ranks;
- exactly two launches on the BF16 path;
- no JIT or device fault;
- `relL2 < 0.10` at 128, 512 and 8192 tokens;
- graph replay and route-mutation liveness;
- no regression on fixed, compact or skew guards;
- paired 8192-uniform rank-max speedup close to the measured 1.0448x reference
  and at least 1.03x after the established noise allowance.

## Sources

- Final measured persistent implementation, used only to distill and calibrate this
  Skill.
- Public MegaMoEV2 baseline supplied by the invoking task.
- Hardware verification produced by the current independent candidate.

## Embedded planner extension

```expert-skill-planner-extension
schema_version: expert-skill-planner-extension-v1
skill_id: megamoe_ep_mega_fusion
plan_version: mega-plan-v2

applicability:
  mode: mega
  operator: moe_dispatch_combine
  architecture: gfx950
  parallelism: EP8
  framework: MegaMoE_v2

selection_policy:
  continue_existing_lane_first: true
  preferred_template: full_persistent_pipeline
  do_not_measure_partial_source: true

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
      - monotone_cross_rank_readiness
      - graph_replay_safe
  work_domains:
    - {id: input_tokens, unit: token, extent: run_tokens, index_type: i32}
    - {id: routed_rows, unit: routed_row, extent: num_valid, index_type: i32}
    - {id: g1_output_stripes, unit: m_tile_n_stripe, extent: g1_m_tiles * g1_n_stripes, index_type: i32}
    - {id: g2_output_tiles, unit: m_block_n_block, extent: g2_m_blocks * g2_n_blocks, index_type: i32}
    - {id: combine_items, unit: output_token_partition, extent: run_tokens * combine_partitions, index_type: i32}
  regions:
    - {id: input_quantize, role: producer, work_domain: input_tokens}
    - {id: dispatch_plan, role: producer, work_domain: routed_rows}
    - {id: first_compute, role: producer_consumer, work_domain: g1_output_stripes}
    - {id: second_compute_transport, role: producer_consumer, work_domain: g2_output_tiles}
    - {id: output_reduce, role: consumer, work_domain: combine_items}
  queues:
    - {id: first_compute_queue, work_domain: g1_output_stripes, claim_unit: stripe}
    - {id: second_compute_queue, work_domain: g2_output_tiles, claim_unit: tile}
    - {id: output_reduce_queue, work_domain: combine_items, claim_unit: block_wave_items}
  schedule:
    primary_loop:
      kind: unified_g1_g2_then_combine
      carried_state: [g2_next, g2_remaining]
      queue_selection: route_gated_g2_preempt_else_g1
    policies:
      g1_completion: stripe_system_atomic_threshold
      g1_store_visibility: write_through_then_wait_barrier_atomic
      g2_preempt_cohort: ticket_mod_6
      g2_preempt_skew: expert_tile_capacity_gt_valid_rows_times_5_over_4
      g2_chunk: contiguous_host_specialized_1_or_16
      stage2_resources: local_to_shared_body
      combine_transition: after_local_g1_g2_drain_without_grid_barrier
      combine_claim: one_block_claim_then_distinct_token_per_wave_at_p1
      combine_partitions: min_ceildiv_grid_waves_tokens_hidden_i32_div_8
      token_readiness: combine_only_local_mblock_close_remote_row_arrival
  evidence_requirements:
    activation: path=MEGA on all eight ranks
    accuracy: relL2 below 0.10 at 128, 512 and 8192
    launch_count: exactly two BF16-path launches
    liveness: graph replay with route mutation
    performance: paired 8192-uniform rank-max speedup at least 1.03x

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
        - monotone_cross_rank_readiness
        - graph_replay_safe
      require_overlap: true
      parameters:
        quantization_remains_separate: true
        combine_is_in_kernel: true
        g2_preempt_modulus: 6
        g2_preempt_skew_ratio: 5/4
        g2_chunks_are_contiguous: true
        combine_8192_partitions: 1
        combine_8192_reduce_u: 4
    checkpoints:
      - complete_host_path
      - complete_persistent_pipeline
      - static_contract
      - hardware_acceptance
    completion_rule: one candidate contains every required region and passes every acceptance gate

failure_routes:
  - id: repair_complete_persistent_pipeline
    match:
      categories: [plan, correctness, lifecycle, compiler, resource, performance]
    repair_intent: repair the same complete two-launch candidate using this Skill; do not open local optimization rungs or expose the reference
    checkpoint: complete_persistent_pipeline
    focus_files:
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
      - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
      - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
```

## Embedded contract

```expert-skill-contract
schema_version: expert-skill-contract-v1
skill_id: megamoe_ep_mega_fusion

source:
  include:
    - aiter/ops/flydsl/kernels/mega_moe/*.py
    - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
  required:
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    - aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    - aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py

provenance:
  require_manifest: true

plan:
  required: true
  assertions:
    - {id: plan_v2, path: plan_version, op: eq, value: mega-plan-v2}
    - {id: two_launch_target, path: target.launch_count, op: eq, value: 2}
    - {id: target_arch, path: resources.arch, op: eq, value: gfx950}
    - {id: combine_in_kernel, path: schedule.policies.combine_transition, op: eq, value: after_local_g1_g2_drain_without_grid_barrier}

checks:
  - id: complete_host_path
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._run_joint
    patterns:
      - 'FUSE_ALL'
      - '_run_fused_stage1'
      - 'FUSE_COMBINE'
      - 'return'

  - id: host_specializes_g2_chunk
    category: performance
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    patterns:
      - '[''"]16[''"]\s+if\s+cur_tok\s*>=\s*4096\s+else\s+[''"]1[''"]'

  - id: flat_stripe_completion
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_do_scheduled_tile'
      - 'atomic_add_system'
      - 'unit\s*//\s*n_tiles'
      - 's_waitcnt\(0\)'
      - 'barrier\(\)'

  - id: unified_g1_g2_loop
    category: schedule
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'while consumer_active'
      - '_do_scheduled_tile'
      - '(?:emit_stage2_body|S2\[[''"]emit[''"]\])'
      - '(?:ticket|cta_ticket)\s*%\s*(?:fx\.Int32\()?6'
      - '(?:max_expert_tiles|expert_tile_capacity).*(?:5\s*/\s*4|5.*valid.*4)'
      - '(?:claim_id|chunk_claim).*(?:local_claim|claim_local).*(?:shards|WORK_SHARDS)'
      - '(?:base|chunk_base).*(?:claim_id|chunk_claim).*(?:chunk|G2_CHUNK)'
      - '(?:unready|not_ready|pending).*(?:drain|drained|pending)'

  - id: shared_stage2_body
    category: compiler
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    patterns:
      - 'def emit_stage2_body'
      - 'def _emit_stage2_body'
      - 'p2p_scatter_epilog'

  - id: progressive_token_readiness
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'tok_ready'
      - 'epoch'
      - 'topk|fuse_topk|fz_k'

  - id: combine_output_work_item
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    patterns:
      - '(?:grid_waves|nominal_grid_waves)'
      - '(?:partitions_per_token|combine_partitions).*(?:min|minimum)'
      - '(?:hidden_i32|n_i32).*(?:/\s*8|//\s*8)'
      - '(?:reduce_u|reduction_width|combine_u).*(?:=\s*4|8192)'
      - '(?:total_items|total_combine_work).*(?:run_tokens|tokens).*(?:partitions_per_token|combine_partitions)'
      - 'def _emit_item\(s3_work_idx\)'
      - 'for k_slot in range_constexpr\((?:experts_per_token|TOPK)\)'
      - 'tok_id\s*\*\s*(?:experts_per_token|TOPK)\s*\+\s*k_slot'
      - 'buffer_store'

  - id: block_claimed_combine
    category: lifecycle
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'while comb_active'
      - 'atomic_add_agent'
      - '_c_emit'

  - id: p2p_visibility
    category: correctness
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    patterns:
      - 'write.through|WRITE_THROUGH|_P2P_CACHE_WT|cache_modifier'
      - 'publish_tok_ready|tok_ready'

  - id: bucket_512_payload_rows
    category: performance
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
    patterns:
      - 'payload_chunk_rows\s*=\s*256\s+if\s+bucket\s*==\s*512\s+else\s+384'
```
