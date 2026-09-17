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
embedded_components: [planner_extension, contract, runtime_validation]
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
    bf16_launches: 2
    quantization_separate: true
    persistent_ctas_8192: 256
    swiglu_limit: 10
    stage1_8192: {sbm: 128, tile_n: 512, tile_k: 256, waves: 8, dispatch_ctas: 96, shards: 4}
    stage2_8192: {bm: 64, bn: 512, bk: 256, waves: 8, halves: 1}
    g2_schedule: {preempt_modulus: 6, skew_numerator: 5, skew_denominator: 4, chunk: 16}
    cache_bits: {g1_store: 17, p2p_store: 19, combine_load: 19, combine_output: 2}
    combine_8192: {partitions_per_token: 1, reduction_width: 4, total_items: 8192}
    timed_diagnostics: false
---

## When to use

The validated evidence covers MegaMoE V2 on one EP8 gfx950 node and the
complete two-launch persistent target. In that target quantization remains a
separate launch; dispatch, GEMM1, GEMM2, P2P publication and Combine execute
in one persistent grid.

This Skill declares measured Pattern evidence; it does not own applicability
control flow. Analyze reports the current semantic fingerprint and bindings.
Planner owns whether to apply, adapt, partially use, request re-analysis, or
decline the guidance. Source names and layout are not part of the Pattern.

## Validated profile and semantic fingerprint

The measured evidence context is one EP8 gfx950 node, wave size 64, A8W4,
BF16 external input/output, per-1x32 MXFP8 activation quantization with E8M0
scales, MXFP4 expert weights with the supplied shuffle/scale contracts,
384 experts (`48/rank`), top-k 6, hidden/intermediate dimensions 7168/3072,
and bounded SwiGLU with limit 10.

Analyze should expose this checklist to Planner before using profile-scoped
constants:

- stage graph: separate input quantization, routed first expert GEMM,
  activation plus output-scale production, dependent second expert GEMM,
  route-weight multiplication, cross-rank publication, and top-k output
  reduction;
- topology: one node, eight ranks, experts divisible by ranks, rank-local
  expert weights, global expert IDs, and exactly top-k routed contributions per
  output token;
- numerical fingerprint: BF16 external rows, MXFP8 first/second activations,
  MXFP4 weights, 1x32 E8M0 scale groups, SwiGLU semantics, route weights
  applied before publication, and FP32 final accumulation;
- transport fingerprint: large-MTPR direct slots indexed by
  `(destination_token,route_slot)`, block-FP8 payload plus one E8M0 byte per
  32 values, and BF16 final output;
- lifecycle fingerprint: symmetric peer-visible payload/readiness ownership,
  one in-flight invocation per operator instance, persistent backing storage,
  stream ordering, and graph replay across changing routes and token counts;
- architecture fingerprint: gfx950 cache-scope semantics, 64-lane waves,
  163840-byte LDS limit, and enough CUs for the profile grid.

A mismatch is evidence, not a Skill-side applicability decision. Analyze
reports it, re-discovers the current role bindings, re-derives every affected
formula or profile constant, and exposes the result to Planner for
workflow-owned selection. Profile-scoped measured constants are not
extrapolated silently.

## Non-normative discovery and mapping

Analyze discovers current bindings from dataflow, calls, types, shapes,
address spaces and launch sites:

1. Follow BF16 rows into the 1x32 quantizer and identify its MXFP8 payload and
   E8M0 scale outputs.
2. Follow route IDs/weights into the rank-local first expert contraction and
   identify the activation-plus-scale rows consumed by the dependent
   contraction.
3. Follow the dependent contraction through route weighting into peer-visible
   direct slots and their readiness publication.
4. Follow every top-k slot into the FP32 reducer and BF16 output.
5. Find the host path that owns buffers, selects bucket parameters, constructs
   the JIT identity and issues launches.
6. Record discovered files and symbols in the ordinary Analyze modifiable-file
   set, task graph, PlanIR regions, ABI arguments and compiler constraints.

When the baseline evolves, repeat this mapping. Current class, function and
module names are neither requirements nor evidence of the Pattern.

## Mechanism

The canonical BF16 path has exactly two launches per rank:

1. the existing BF16-to-MXFP8 `per_1x32` quantizer;
2. one persistent grid containing dispatch, GEMM1, GEMM2, weighted P2P
   publication, readiness, and final Combine.

The full path is opt-in and must be selected only when the fusion switch is the
exact string `1` and the selected Stage1 wave count is divisible by four.
Combine is present in the persistent grid. Quantization is not. The host
returns the persistent kernel's output view and launches no Stage2 or Combine
tail.

The second launch is one CTA-local state machine:

- One ticket identifies the launch generation and the CTA's startup role.
  Ticket zero is the plan owner; tickets `1..dispatch_blocks` are payload
  producers. All roles join the compute queues after startup.
- G1 work is the flat stripe domain
  `unit = m_tile * g1_n_stripes + n_stripe`.
- G1 activation and scale stores use cache bits `sc0|sc1` (`17`). Every wave
  retires its stores, all waves reach one block barrier, and lane zero performs
  one system-scope completion add for the owning m-tile. There is no
  post-publication L2 writeback.
- G1 and G2 are chosen inside the same loop. G1 is the default. A CTA may peek
  and preempt with ready G2 only when both `ticket % 6 == 0` and
  `max_expert_tiles * experts_per_rank * stage1_tile_m * 4 >
  valid_rows * 5` hold.
- After G1 exhaustion, every CTA may claim G2 and wait for its dependency. An
  unready G2 claim is pending work, not an empty queue. A CTA reaches Combine
  only after it owns no continuation and its G2 shard head is out of range;
  consequently every valid G2 chunk has already been claimed by some CTA.
- G2 claims are contiguous. For shard `s`, local claim `q`, shard count `S`,
  and chunk `C`: `claim_id = s + q*S`, `base = claim_id*C`, and the claim owns
  `[base, min(base+C,total_g2_units))`. `C=1` for `run_tokens<4096`; `C=16`
  otherwise. Readiness of the last in-range unit dominates the contiguous
  claim. The continuation carries only two block-uniform scalars,
  `next_unit` and `units_remaining`; it performs no further head atomic,
  readiness poll, or LDS handoff.
- The G2 single-unit body owns all route, weight, peer, and readiness
  descriptors. No descriptor, GEMM accumulator, or nested G2 loop crosses the
  outer scheduler backedge.
- Stage2 writes payload and block scales with `sc0|sc1|nt` (`19`). After a
  workgroup release and uniform barrier, lane zero adds one to the rank-local
  m-block close counter. The previous value is broadcast through LDS. The
  closing CTA resets the counter by an agent-scope `-g2_n_blocks` add, then the
  first `BM` threads publish exactly one system-scope arrival for their
  accepted routed row.
- Token-arrival counters are never cleared after initialization. A
  Combine-specific generation increases only on launches that contain both
  arrival publication and Combine. The wait target is `topk * generation`;
  tokens outside `[0,run_tokens)` are advanced to that target before startup
  publication.
- Combine is an irreversible third queue with no grid barrier. One agent-scope
  block claim advances by one; claim `b` gives wave `w` item
  `b*cta_waves+w`. Each item waits independently in lane zero, reads all top-k
  direct slots with cache bits `19`, reduces in FP32, and stores BF16 with
  rank-local `nt` cache bits `2`.

Timed builds fix native Stage2 geometry, G2 cadence, cache policies, readiness,
and Combine geometry to the rules below. Every non-target launch topology,
publication cut, fused-quant path, wait timer, profiler, and diagnostic is
disabled and has a distinct JIT identity.

## Semantic role boundaries

Keep these responsibilities separate regardless of source organization:

- The host-selection role validates the semantic fingerprint, selects one
  bucket, keeps quantization separate, builds semantic argument groups, launches
  the persistent grid once, and returns its output without a tail launch.
- The first-compute item role emits exactly one flat output stripe. Queue
  progression and completion publication belong to the persistent scheduler.
- One derivation authority validates every dependent-GEMM tile, scale,
  transport, LDS and bounds constant. One single-unit role emits exactly one
  `(m_block,n_block)` unit. Standalone and persistent work distributors call
  that same numerical body and P2P epilogue.
- The dependent-GEMM role constructs route, weight, expert, peer and readiness
  resources inside the single-unit frame. Every LDS address is relative to a
  caller-supplied slab base.
- One output-reduction item role derives runtime `P`, partition width and `U`,
  owns direct-slot resources, the per-wave wait, all top-k loads, FP32
  accumulation and bounded BF16 stores. Its U-step and resource collections
  remain inside that decorated item frame. The scheduler owns only block
  claims.

Reuse the supplied baseline's semantically equivalent numerical bodies after
verifying these contracts. Scheduling relocation must preserve their
quantization, scale layout, expert indexing, activation, weighting and
accumulation semantics.

## Baseline math, ABI and state layout

Reuse the supplied baseline's semantically equivalent numerical bodies after
verifying their dataflow and type contracts:

- keep the standalone BF16-to-MXFP8 `per_1x32` quantization launch;
- reuse the existing GEMM1 MFMA, activation/scale layout and weight shuffle;
- reuse the existing GEMM2 MFMA and weighted P2P scatter epilogue;
- reuse combine's BF16 and block-FP8 decode/FP32-accumulate rules.

The optimization is relocation and scheduling, not a new GEMM approximation.
Do not change quant formats, scale grouping, expert indexing, top-k weights or
accumulation dtype to obtain the launch reduction.

### Canonical configuration

Token buckets are fixed to
`[1,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768]`.
Choose the nearest bucket, resolving a tie upward. `max_tokens_per_rank` is a
power of two. Values at or below 1024 retain their own MTPR class; larger
values use class 32768. Fixed-slot dispatch is used only for MTPR at or below
255. The expert-count class is `ceildiv(experts_per_rank,64)*64`; dispatch CTA
counts below are multiplied by that class divided by 64 and capped at 224.

Stage1 defaults are fixed: `tile_k=256`, weight pipelining on, A swizzle on,
`waves_per_eu=2`, and eight work shards unless a rule below overrides them.
Grid multipliers are drawn from `[1,2,3,4,6,8,12,16,24,32]`; the launch
generation state has one slot for each value.

For fixed-slot MTPR:

- `SBM=32`, `waves=4`, `tile_n=256` through bucket 8 and 128 thereafter.
- `grid_mult=max(1,bucket//4)` through bucket 16 and 3 thereafter.
- Dispatch CTA bases are 64 at bucket 1, 128 through bucket 8, 96 through
  bucket 16, 128 through bucket 32, then
  `min(224,16*(bucket.bit_length()+7))`.
- MFMA A-major and asynchronous A copy are off. Tile resources are on through
  bucket 16. B cache mode is 0 at bucket 1 and 3 otherwise.
  `waves_per_eu=1` only at bucket 16.

For non-fixed MTPR at or below 1024:

- Buckets through 4 use `(SBM,tile_n,waves,grid_mult)=(32,256,4,1)` with
  A-major and asynchronous A copy off.
- Buckets 8 through 128 use `(32,512,8,1)` for `inter_dim>=2048`
  (`tile_n=256` otherwise), with A-major and asynchronous A copy on.
- Buckets 256 through 1024 use `SBM=64`, the same tile-N rule and eight waves;
  `grid_mult=1` at 256 and 2 otherwise.
- Dispatch CTA bases are the compact sequence
  `224,128,192,64,128,192,128` for bucket ceilings
  `1,4,8,16,32,64,>=128`, then 160 at 256 and 128 above it.
  When MTPR exceeds the bucket, override bases at 32/64/128 to 64/160/192;
  bucket 512 additionally fixes `grid_mult=1`, dispatch CTAs 64, tile
  resources on and B cache mode 0.
- Tile resources are otherwise on only at bucket 256. B cache mode is 0 at
  bucket 1 and at or above 1024, and 3 otherwise. Work shards remain eight.

For fixed-slot and non-fixed MTPR at or below 1024, Stage2 uses `BK=256`,
B hoisting and A-scale prefetch on, spatial code 402, and BF16 LDS off. If the
path is non-fixed and MTPR exceeds the selected bucket, fix `BM=32`,
`BN=128` only at bucket 256 and 256 otherwise, persistence on with 240 CUs,
non-temporal B loads through bucket 128, and persistent striding from bucket
512 through 2048. Otherwise use `BM=32`; choose `BN=256` for buckets
1/4/64, at or above 1024, or non-fixed buckets below 128, and 128 for the
remaining buckets; force BN=128 when `model_dim<4096`. Persistence begins at
bucket 128, with 128 CUs at bucket 256 and 240 CUs otherwise; non-temporal B
loads are used through bucket 128 and persistent striding from bucket 512
through 2048.

For MTPR above 1024, including the measured 8192 class:

```text
bucket       1   4   8  16  32  64 128 256 512 1024 2048 4096 8192 16384 32768
SBM         32  32  32  32  32  32  32  64  64   64   64  128  128   128   128
tile_n     256 256 512 512 512 512 512 512 512  512  512  512  512   512   512
waves        4   4   8   8   8   8   8   8   8    8    8    8    8     8     8
dispatch   224 128 192  64  64 160 192 160  64   64   64   64   96    32    32
G1 shards    1   1   1   1   1   4   4   4   4    4    8    4    4     4     4
payload rows384 384 384 384 384 384 384 384 256  384  384  384  384   384   384
G2 BM       32  32  32  32  32  32  32  32  32   32   32   64   64    64    64
fused G2 BN256 256 512 512 512 512 512 256 512  512  512  512  512   512   512
G2 persist 240 240 240 240 240 240 240 240 240  224  256  240  240   192   240
G2 skew CUs  0   0   0   0   0   0   0   0  96   96   96   96   96    96    96
G2 chunk     1   1   1   1   1   1   1   1   1    1    1   16   16    16    16
```

In this class `grid_mult=1`, tile resources and payload-tile readiness are on,
and transport is block-FP8 `1x32`. A-major and asynchronous A copy are off
through bucket 4 and on thereafter. B cache mode is 3 only for buckets
`4..256`, otherwise 0. External grouping is on at bucket 4 and at or above
256; external counting is on at or above 256. Stage2 B loads are non-temporal
through bucket 128 only; persistent striding is on for buckets 512 through
2048. Stage2 `BK=256`, B hoisting and A-scale prefetch are on, BF16 LDS is off,
and the configured spatial code is 402.

The persistent body, not standalone Stage2, owns work distribution. It
therefore forces Stage2 persistence off and spatial remapping to zero, uses
one slab, sets `NW=Stage1 waves`, and widens the configured Stage2 BN by
`Stage1 waves/4`. GEMM2 uses FP8 A, FP4 B, scaled 16x16x128 MFMA, `BK=256`,
two 128-K halves, three A LDS stages, `wave_n=BN/NW`, and
`num_acc_n=wave_n/16`. B payload loads are four i32 per lane. The
BM64/BN256 case alone uses the one-stage B path; all other canonical fused
shapes use the two-stage B pipeline with B hoisting and A-scale prefetch. At
the measured 8192 shape this fixes:

```text
EP=8, experts/rank=48, topk=6, hidden=7168, inter=3072, MTPR=8192
Stage1: SBM=128, tile_n=512, tile_k=256, waves=8, grid=256 CTAs
         dispatch=96, shards=4, payload_rows=384, B cache=0
Stage2: BM=64, BN=512, BK=256, NW=8, one slab, 14 N blocks
         B non-temporal=off, B-hoist=on, A-scale-prefetch=on
Scheduler: preemption modulus=6, skew ratio=5/4, G2 chunk=16
```

### Host ABI and persistent state

The host ABI carries semantic groups: G1 tensors and dispatch state; raw G2
activation, scale, weight and route addresses; selected BM/BN/BK and maximum
m-block count; Combine payload/output and peer-ready addresses; two local
counter arenas; and the Combine control words. Device buffer resources are
constructed only inside the role that consumes them.

For compact dispatch:
`num_valid_max = world_size*MTPR*topk + experts_per_rank*128`.
For fixed-slot dispatch:
`num_valid_max = experts_per_rank*roundup(world_size*MTPR,32) + 256`.
Metadata reserves `ceildiv(num_valid_max,32)` entries.

The canonical host owns these allocations for the object's whole graph
lifetime:

```text
dispatch work heads       = 8 * 16 i32; one 64-byte line per shard
fixed-slot group_done     = world_size i32
G1/G2 counter arena       = max(1,num_valid_max) i32
G2-close counter arena    = max(1,num_valid_max) i32
Combine control           = 4 i32: block head at 0, generation at 1
token arrivals            = MTPR + 8 symmetric i32
peer-pointer tables       = world_size i64
Combine backing rows      >= max(world_size*MTPR, MTPR*topk)
Combine output rows       = MTPR BF16 rows
```

Within the G1/G2 arena, G1 completion entries cover
`ceildiv(max_g2_m_blocks*G2_BM,SBM)`. G2 claim head `s` is at
`max_g2_m_blocks + 16*s`, so completion and head regions are disjoint and
every head has its own 64-byte line. The second arena is indexed by G2
m-block and self-clears on close. The Stage2 payload resource is bounded to
`MTPR*topk*row_bytes`, where `row_bytes=hidden+hidden/32` for block-FP8 and
`2*hidden` for BF16, even when the backing allocation is larger.

All active addresses are nonzero and match emitted size/alignment. Route,
weight, expert, tile-row and peer resources use their real byte capacities.
Source-token encoding requires `world_size*MTPR <= 2^24`; route-slot encoding
requires `topk <= 2^8`; the Stage2 payload resource must remain below `2^31`
bytes. Disabled diagnostic roles may use zero placeholders only when
compile-time eliminated.

## Scheduling blueprint

Launch tickets are monotone per grid-multiplier slot. The owner toggles the
dispatch parity bank, advances its expected cross-rank value, performs the
launch handshake, resets G1 heads and active completion entries, resets the
Combine block head, increments the Combine generation, pads omitted tokens,
then publishes the epoch gate. No producer or consumer passes that gate first.

G1 heads are sharded exactly like this:
`shard = ticket & (S-1)`, `local = atomic_add(head[shard],1)`,
`unit = shard + local*S`. Each head begins on its own 64-byte line.
`g1_n_stripes=(2*inter_dim)/tile_n`; the owning completion index is
`unit/g1_n_stripes`.

The fused loop follows this order on every iteration:

1. A carried c16 continuation executes immediately. `next_unit` and
   `units_remaining` are block-uniform SSA values outside the lane-zero
   selection branch; no six-word LDS scheduler frame is permitted.
2. An eligible skew CTA non-destructively peeks its G2 shard head with an
   agent-scope add of zero. It tests the last in-range unit's owning G1
   completion. If ready, it atomically claims one chunk; because another CTA
   may win between peek and claim, it still performs the monotone blocking wait
   on the actual claim.
3. If no preemption was selected, claim one G1 stripe. Compact dispatch waits
   for that stripe's payload before executing it.
4. If G1 is out of range, atomically claim one G2 chunk regardless of current
   readiness and wait on the claim's last in-range dependency.
5. Broadcast only `kind` and `unit` through the shared scratch. Execute exactly
   one G1 stripe or one G2 unit. Loop while `kind!=none`.

The loop-top block barrier is mandatory: Stage2's epilogue has no trailing
barrier, so it separates the previous unit's LDS reads from the next scratch
write. Bounds dominate every G1/G2 readiness address. A CTA never waits for G2
while holding an unexecuted G1 claim.

When a CTA observes both heads out of range and has no continuation, it enters
Combine permanently. Other CTAs may still finish already-owned G2 chunks.
This is safe because no unclaimed G2 unit remains. Combine uses:

1. one unconditional block barrier;
2. lane-zero `atomic_add(block_head,1)` and one LDS claim store;
3. one unconditional block barrier;
4. `_base=claim*cta_waves`, `_item=_base+wave`;
5. one direct per-wave item call when `_item<total_items`.

The next loop-top barrier is the post-item barrier. The item contains no
workgroup barrier and no CTA-wide readiness mask or retry frame.

## Memory-order blueprint

- G1 activation and scale stores use `sc0|sc1=17`; order is stores,
  `s_waitcnt(0)`, uniform block barrier, then one system-scope add. A cached
  store, cache value 2, a release after the add, or an RMW before the barrier
  is not equivalent.
- G2 waits for `g1_ready[owner_tile] >= g1_n_stripes`.
- P2P payload and block-scale stores use `sc0|sc1|nt=19`. Each active lane
  writes eight contiguous payload bytes. For block-FP8, one lane per four-lane
  group writes the E8M0 byte at
  `hidden + n_block*(BN/32) + lane/4`.
- Readiness publication is compiled only with in-kernel Combine. All waves
  execute the workgroup release and barrier before lane zero closes the local
  m-block. The close counter is monotone only within that m-block operation and
  is self-cleared by the unique closer; it is not a per-route or cross-rank
  close table.
- The closer broadcasts through a dedicated LDS word. Threads `0..BM-1`, one
  row per thread, decode the route metadata already in LDS and issue at most
  one remote system arrival each. Lane zero must not serially loop over BM
  rows.
- Combine waits with `ready >= topk*combine_generation`; lane zero uses the
  greater-than primitive with target minus one. There is no acquire fence after
  the wait. Payload and scale loads themselves use cache value 19, avoiding a
  per-item L2 invalidate. BF16 output stores use rank-local cache value 2.
- Arrival counters are initialized once and never cleared. The Combine
  generation is independent of dispatch generations that did not publish
  arrivals. Omitted tokens are written to the current target before the epoch
  gate, so graph replay may safely grow a later batch. The cross-rank launch
  gate prevents the next launch's single-buffered P2P writes until every rank
  has finished the prior launch's Combine and entered the next generation.

## Combine work geometry

The shared Combine item emitter derives geometry from runtime
`run_tokens` and `nominal_waves=launch_grid_ctas*cta_waves`; `run_tokens` is not
a compile-time specialization added solely for Combine.

```text
safe_tokens = 1 if run_tokens == 0 else run_tokens
P = ceildiv(nominal_waves, safe_tokens)
if block_fp8:
    scale_blocks = hidden_i32 / 8
    P = min(P, scale_blocks)
    hidden_i32_per_wave = ceildiv(scale_blocks, P) * 8
else:
    hidden_i32_per_wave = ceildiv(hidden_i32, P)
total_items = run_tokens * P
```

For block-FP8, `hidden_i32=hidden/4`; for BF16 it is `hidden/2`.
Item `i` maps to `token=i/P`, `partition=i%P`, starting at
`partition*hidden_i32_per_wave`. Tail length is
`min(hidden_i32-start,hidden_i32_per_wave)`.

Reduction width is fixed by the item emitter:

```text
U=1                                  when hidden_i32_per_wave <= 895
U=1                                  when hidden_i32 % 256 != 0
U=4                                  when width > 895 and width % 256 == 0
U=2                                  otherwise when width > 895
```

The U-wide step first issues all top-k payload and scale loads for every `u`,
then broadcasts scale bytes, performs top-k FP32 sums, and emits independent
stores. Per `(expert,u)`, each lane loads one i32 payload; one lane per
eight-lane group loads one i8 scale and broadcasts it within that group.
`u>0` payload offsets advance 256 bytes and block-FP8 BF16 output offsets
advance 512 bytes. A U1 tail handles the remainder.

For the measured MTPR-8192 class, all three acceptance shapes launch 256 CTAs:

```text
tokens  nominal waves  P   hidden_i32/wave  U  total items
128          2048      16        112         1      2048
512          2048       4        448         1      2048
8192         2048       1       1792         4      8192
```

At 8192 each eight-wave block claim therefore covers eight distinct output
tokens, not eight partitions of one token.

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
offset zero. The 131728-byte large slab is exactly:
131072 compute bytes, `BM*4` packed-route bytes, `BM*4` weight bytes,
`world_size*8` payload-peer bytes, `world_size*8` readiness-peer bytes, and a
16-byte readiness broadcast tail.

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

- G2 chunk, Stage1/G2 tile geometry, dispatch CTAs, grid multiplier, work
  shards, payload rows, transport, preemption modulus and skew ratio are the
  fixed bucket values above.
- Native Stage2 uses all Stage1 waves on one tile and one slab. Widen BN by the
  wave ratio; never run two lockstep four-wave halves in the timed target.
- The complete Python compile key contains the full hashable Stage2 and Combine
  specifications plus every Stage1 argument. The emitted identity includes
  dispatch mode, Stage1 tiles/waves/grid/dispatch/cache/shards/payload mode,
  Stage2 halves/BM/BN/preemption/chunk/skew, Combine presence, wait-timer
  presence, fused-quant presence, and every LDS-affecting option.
- Runtime Combine `P` and `U` do not create a host-specialized kernel identity.
  Diagnostic no-publication, timer, and fused-quant variants must have distinct
  identities and must not populate the timed cache entry.

## Compiler pitfalls

- A source-level two-launch declaration is not emitted-artifact evidence.
- Do not move Stage2 descriptors out of the single-unit body or Combine
  resource lists/U-step out of the decorated item frame.
- FlyDSL rewrites dynamic branches and loops; values created only inside a
  rewritten branch must not be consumed outside it without explicit carried
  state.
- A workgroup barrier must be reached uniformly. Per-wave combine items contain
  no workgroup barrier; barriers belong to the surrounding block-claim loop.
- Names, comments, metadata attributes, host-only work specifications, and a
  separately reimplemented fused body do not satisfy a device semantic.
- Do not infer safety from VGPR/SGPR counts alone. Validate the exact emitted
  artifact and execute the all-six-slot bs=128 path before broader testing.

## Procedure

Run one normal `mode=mega` lane:

1. Analyze the supplied baseline's semantic fingerprint and discover current
   bindings for every stage and state edge. Report mismatches and derived
   replacements to Planner; the workflow owns applicability selection.
2. Lower the one topology and the exact configuration/state-machine rules
   above. Do not create alternative schedules or tuning branches.
3. Intermediate WIP commits are permitted and required in the same lane. Each
   authoring turn closes a coherent subset of required failures, commits that
   checkpoint, and continues the same candidate/head lineage. For that
   committed-turn artifact set `candidate_status=authoring`,
   `claim_complete=true`, and every runtime/correctness/performance flag false
   or pending. Never benchmark or promote it until all required implementation
   checks pass; a WIP checkpoint is not a separate candidate or topology.
4. Run the GPU-free contract after the workflow selects this guidance. Its
   required implementation checks reject any claimed-complete checkpoint
   missing the complete host path, flat stripe publication, unified G1/G2
   loop, shared Stage2 body, P2P visibility, progressive readiness, Combine
   item, block-claimed Combine, or bucket rule. The contract validates claims;
   it does not decide applicability. A structural pass makes no runtime claim.
5. In the later hardware phase, compile the exact artifacts, inspect resources,
   validate bs=128, then 512 and 8192, and use the same operator instance for a
   smaller-to-larger graph-generation check with route mutation.
6. Complete the same chain with eight path markers, exactly two BF16-path
   launches, direct numerical parity, liveness, the four established route
   guards, and paired 8192-uniform rank-max performance.
7. Return any concrete failure to this one candidate. Do not replace the
   canonical target with diagnostics, partial fusion, or open-ended search.

## Do-no-harm notes

- Skill-off must preserve the ordinary Mega lifecycle and receive none of this
  knowledge.
- This guidance does not create a reproduction mode, reserved candidate source,
  or alternate lifecycle; it is consumed by the ordinary workflow roles.
- Engineer must not receive a reference path, reference source, local run IDs
  or private candidate identities.
- A structural pass means only that the complete source is ready for hardware.
- A silent scattered fallback, three-launch partial, timeout, device fault or
  missing path marker is not a valid result.
- Timed runs disable fused quantization, no-publication cuts, wait timing,
  profiling, readiness reports and stage-isolation controls.
- Measure against the frozen scattered baseline on the same machine. No
  previously reported latency is a denominator.

## Acceptance

The Skill succeeds only when its independently authored candidate has:

- `path=MEGA` from all eight ranks;
- exactly two launches on the BF16 path;
- no JIT or device fault;
- `relL2 < 0.10` at 128, 512 and 8192 tokens;
- same-instance graph replay with route mutation and a smaller-to-larger token
  transition, proving generation padding;
- emitted 128 and 8192 resource envelopes within the values above and at least
  one resident workgroup per CU;
- no regression on fixed, compact or skew guards;
- paired 8192-uniform rank-max speedup close to the measured 1.0448x reference
  and at least 1.03x after the established noise allowance.

## Sources

- Final measured persistent implementation, used only to distill and calibrate
  this Pattern.
- Public baseline supplied by the invoking task.
- Independent candidate validation produced by the one acceptance chain.

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
  evidence_profile:
    ranks: 8
    experts_per_rank: 48
    topk: 6
    hidden_dim: 7168
    inter_dim: 3072
    swiglu_limit: 10
    external_dtype: bf16
    activation_quant: mxfp8_per_1x32_e8m0
    weight_quant: mxfp4_e8m0
    large_transport: block_fp8_per_1x32

binding_policy:
  authority: analyze
  names_are_normative: false
  discovery: call_dataflow_type_shape_address_space_and_launch_sites
  report_mismatches_to: planner
  roles:
    - input_quantization
    - route_plan_and_payload
    - first_expert_compute
    - dependent_expert_compute
    - weighted_peer_publication
    - output_reduction
    - host_selection_and_state
  record_in:
    - modifiable_files
    - task_graph
    - mega_plan_ir.regions
    - mega_plan_ir.abi
    - mega_plan_ir.compiler_constraints

selection_policy:
  continue_existing_lane_first: true
  preferred_template: full_persistent_pipeline
  do_not_measure_partial_source: true
  intermediate_wip:
    lane: same_candidate_required
    commit_each_turn: true
    candidate_status: authoring
    claim_complete_for_committed_turn_artifact: true
    runtime_flags: false
    benchmark_or_promote_before_required_checks_pass: false
    continue_head_lineage: true
  required_failure_order:
    - plan
    - correctness
    - abi
    - lifecycle
    - resource
    - compiler
    - schedule
    - performance

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
      carried_state: [g2_next_unit, g2_units_remaining]
      queue_selection: continuation_then_gated_g2_then_g1_then_blocking_g2
    policies:
      g1_completion: stripe_system_atomic_threshold
      g1_store_visibility: sc0_sc1_then_waitcnt_barrier_system_add
      g2_preempt_cohort: ticket_mod_6_eq_0
      g2_preempt_skew: max_expert_tiles_times_epr_times_tile_m_times_4_gt_valid_rows_times_5
      g2_chunk: contiguous_compile_time_1_below_4096_else_16
      g2_continuation: two_block_uniform_scalars_no_atomic_poll_or_lds_handoff
      scheduler_semantics: continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16
      stage2_geometry: one_native_stage1_wave_cohort_one_slab
      stage2_resources: single_shared_body_local_descriptors
      p2p_visibility: sc0_sc1_nt_19_then_workgroup_release
      tile_close: local_mblock_agent_add_broadcast_parallel_remote_rows
      combine_transition: after_local_g1_g2_drain_without_grid_barrier
      combine_claim: agent_add_one_then_claim_times_waves_plus_wave
      combine_partitions: runtime_ceildiv_nominal_waves_tokens_capped_only_for_block_fp8
      combine_item_jit_frame: direct_decorated_item_no_extra_wrapper
      combine_reducer_vectorization: pressure_guarded_u1_u2_u4
      combine_payload_visibility: system_scope_loads_without_per_item_acquire
      combine_output_cache: rank_local_nt_2
      combine_semantics: runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4
      token_readiness: combine_generation_monotone_topk_with_omitted_padding
      jit_identity: complete_compile_specs_and_all_codegen_switches
  evidence_requirements:
    activation: path=MEGA on all eight ranks
    accuracy: relL2 below 0.10 at 128, 512 and 8192
    launch_count: exactly two BF16-path launches
    liveness: same-instance smaller-to-larger graph replay with route mutation
    resources: emitted envelopes and one resident workgroup per CU
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
        timed_diagnostics: false
        persistent_grid_ctas_8192: 256
        swiglu_limit: 10
        g2_preempt_modulus: 6
        g2_preempt_skew_ratio: 5/4
        g2_chunk_small: 1
        g2_chunk_large: 16
        g2_chunks_are_contiguous: true
        g2_continuation_scalars: 2
        g1_store_cache: 17
        p2p_store_cache: 19
        combine_load_cache: 19
        combine_output_cache: 2
        stage2_8192_bm_bn_bk_nw: 64x512x256x8
        stage2_8192_halves: 1
        combine_8192_partitions: 1
        combine_8192_reduce_u: 4
        combine_8192_total_items: 8192
    checkpoints:
      - complete_host_path
      - complete_persistent_pipeline
      - static_contract
      - hardware_acceptance
    completion_rule: one candidate contains every required region and passes every acceptance gate

failure_routes:
  - id: repair_complete_persistent_pipeline
    match:
      categories: [plan, correctness, abi, lifecycle, resource, compiler, schedule, performance]
      check_ids:
        - complete_host_path
        - flat_stripe_completion
        - unified_g1_g2_loop
        - shared_stage2_body
        - p2p_visibility
        - progressive_token_readiness
        - combine_output_work_item
        - block_claimed_combine
        - bucket_512_payload_rows
    repair_intent: return required failures in category order to the same candidate lane, commit one coherent subset per turn, and continue its head lineage using Analyze-discovered bindings; never classify remaining work as dead_end or too-large-for-one-turn; applicability remains workflow-owned
    checkpoint: complete_persistent_pipeline
    topology: full_persistent_pipeline
    terminal_when: all_required_checks_pass
    focus_roles:
      - host_selection_and_state
      - first_expert_compute
      - dependent_expert_compute
      - weighted_peer_publication
      - output_reduction
```

## Embedded contract

```expert-skill-contract
schema_version: expert-skill-contract-v1
skill_id: megamoe_ep_mega_fusion

source:
  include:
    - '**/*moe*.py'
    - '**/*dispatch*combine*.py'
    - '**/*quant*.py'

binding_policy:
  authority: analyze
  names_are_normative: false
  source_checks_are_profile_adapter_hints: true
  adapter_hint_failures_block_applicability: false
  adapter_scope: validated_megamoe_ep8_gfx950_profile_only
  adapter_selection: analyze_discovered_semantic_bindings
  applicability_owner: workflow
  applicability_failures_block_compatibility: false

activation:
  scope: post_selection_claimed_complete_structural_checkpoint
  requires_selected_or_pinned_skill: true
  required_checks_define_claimed_implementation_completeness: true

provenance:
  require_manifest: true

plan:
  required: true
  assertions:
    - {id: plan_v2, path: plan_version, op: eq, value: mega-plan-v2}
    - {id: two_launch_target, path: target.launch_count, op: eq, value: 2}
    - {id: target_arch, path: resources.arch, op: eq, value: gfx950}
    - {id: combine_in_kernel, path: schedule.policies.combine_transition, op: eq, value: after_local_g1_g2_drain_without_grid_barrier}
    - {id: unified_semantic_loop, path: schedule.policies.scheduler_semantics, op: eq, value: continuation_then_skew_mod6_ready_g2_else_g1_then_blocking_g2_contiguous_c1_c16}
    - {id: direct_runtime_combine_item, path: schedule.policies.combine_semantics, op: eq, value: runtime_p_direct_per_wave_wait_block_claim_one_system19_u1_u2_u4}

# These probes describe the validated profile's present adapter only. Paths
# and symbols are not applicability gates. After Analyze selects this adapter
# by semantic bindings and the workflow selects/pins the Skill, the nine
# implementation checks are required evidence for a claimed-complete
# structural checkpoint.
checks:
  - id: complete_host_path
    category: plan
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    scope: MegaMoEV2._run_joint
    patterns:
      - 'AITER_MEGAMOE_FUSE_ALL[''"]\)\s*==\s*[''"]1'
      - 'self\.quantize'
      - '_run_fused_stage1'
      - 'AITER_MEGAMOE_FUSE_COMBINE[''"],\s*[''"]1[''"]\)\s*==\s*[''"]1'
      - 'out_tok'
      - 'return\s+out_tok'

  - id: host_specializes_g2_chunk
    category: profile_adapter_hint
    severity: advisory
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
    patterns:
      - '[''"]16[''"]\s+if\s+cur_tok\s*>=\s*4096\s+else\s+[''"]1[''"]'

  - id: flat_stripe_completion
    category: correctness
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '_do_scheduled_tile'
      - 'out_cache_modifier\s*=\s*0\s+if\s+S2\s+is\s+None\s+else\s+_G1_OUT_WT'
      - 'unit\s*//\s*n_tiles_i32'
      - 's_waitcnt\(0\)'
      - 'barrier\(\)'
      - 'atomic_add_system'

  - id: unified_g1_g2_loop
    category: schedule
    severity: required
    kind: regex_sequence
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - '\A(?![\s\S]*make_layout\(6,\s*1\))'
      - '_g2_skewed\s*=\s*_g2_metiles\s*\*\s*fx\.Int32\(fz_epr\s*\*\s*fz_tile_m\s*\*\s*_g2_bal_den\)\s*>\s*num_valid\s*\*\s*fx\.Int32\(_g2_bal_num\)'
      - 'ticket\s*%\s*fx\.Int32\(max\(1,\s*fused_g2_pref\)\)'
      - 'g2_pend\s*=\s*fx\.Int32\(0\)'
      - 'g2_next\s*=\s*fx\.Int32\(0\)'
      - 'while\s+consumer_active:'
      - 'fx\.barrier\(\)'
      - 'kind\s*=\s*fx\.Int32\(0\)'
      - 'unit\s*=\s*fx\.Int32\(0\)'
      - 'if\s+g2_pend\s*>\s*fx\.Int32\(0\):'
      - 'kind\s*=\s*fx\.Int32\(2\)'
      - 'atomic_add_agent\(g2_head,\s*fx\.Int32\(0\)\)'
      - 'atomic_add_agent\(g2_head,\s*fx\.Int32\(1\)\)'
      - 'atomic_add_agent\(a_work_head\s*\+\s*fx\.Int64\(work_shard\)\s*\*\s*fx\.Int64\(64\),\s*fx\.Int32\(1\)\)'
      - 'if\s+kind\s*==\s*fx\.Int32\(1\):'
      - '_do_scheduled_tile\(unit\)'
      - 'if\s+kind\s*==\s*fx\.Int32\(2\):'
      - 'S2\[[''"]emit[''"]\]\('
      - 'consumer_active\s*=\s*kind\s*!=\s*fx\.Int32\(0\)'

  - id: shared_stage2_body
    category: resource
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    scope: make_stage2_body_emitter
    patterns:
      - 'def emit_stage2_body'
      - 'def _emit_stage2_body'
      - 'p2p_scatter_epilog\('
      - 'lds_slab'
      - 'lds_byte_off'

  - id: progressive_token_readiness
    category: lifecycle
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'c_epoch\s*=\s*_buffer_load'
      - '_c_pad\s*=\s*fx\.Int32\(fz_k\)\s*\*\s*c_epoch'
      - 'if\s+_c_t\s*>=\s*i32_cur_tok'
      - 'tok_ready_epoch\s*=\s*c_epoch'
      - 'tok_ready_expected\s*=\s*fuse_topk'

  - id: combine_output_work_item
    category: schedule
    severity: required
    kind: regex_sequence
    file: aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
    scope: make_combine_reduce_emitter
    patterns:
      - 'global_warp_num\s*=\s*nominal_warp_num'
      - 'IN_CACHE\s*=\s*_SLC_CACHE\s*\|\s*1\s*\|\s*16\s*if\s+tok_ready_addr\s+is\s+not\s+None\s+else\s+_SLC_CACHE'
      - 'safe_token_count\s*=\s*\(cur_rank_num_token\s*==\s*0\)\.select\(1,\s*cur_rank_num_token\)'
      - 'warps_per_tok\s*=\s*\(global_warp_num\s*\+\s*safe_token_count\s*-\s*1\)\s*//\s*safe_token_count'
      - 'scale_blocks\s*=\s*n_elems\s*//\s*8'
      - 'warps_per_tok\s*=\s*\(warps_per_tok\s*>\s*scale_blocks\)\.select\(scale_blocks,\s*warps_per_tok\)'
      - 'hdim_per_warp\s*=\s*\(scale_blocks\s*\+\s*warps_per_tok\s*-\s*1\)\s*//\s*warps_per_tok\s*\*\s*8'
      - 's3_total_work\s*=\s*cur_rank_num_token\s*\*\s*warps_per_tok'
      - 'def _emit_item\(s3_work_idx\)'
      - 'tok_id\s*=\s*s3_work_idx\s*//\s*warps_per_tok'
      - 'int32_wait_until_greater_than'
      - 'for k_slot in range_constexpr\((?:experts_per_token|TOPK)\)'
      - '[''"]cache_modifier[''"]:\s*IN_CACHE'
      - 'buffer_store'
      - '_accum_loop\(eff_end,\s*4\)'
      - '_accum_loop\(eff_end,\s*2\)'
      - '_accum_loop\(eff_end,\s*1\)'

  - id: block_claimed_combine
    category: schedule
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
    scope: compile_mega_moe_stage1.kernel
    patterns:
      - 'while comb_active'
      - 'atomic_add_agent\(c_ctr,\s*fx\.Int32\(1\)\)'
      - '_c_base\s*=\s*Vec\(work_scratch_view\.load\(\)\)\[0\]\s*\*\s*fx\.Int32\(NUM_WAVES\)'
      - '_c_item\s*=\s*_c_base\s*\+\s*_c_wave'
      - '_c_emit\(_c_item\)'

  - id: p2p_visibility
    category: correctness
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
    patterns:
      - '_P2P_CACHE_WT\s*=\s*2\s*\|\s*1\s*\|\s*16'
      - 'p2p_write_through'
      - 'fence_release\(fx\.rocdl\.SyncScope\.WorkgroupOneAs\)'
      - 'atomic_add_agent\(\s*arg_mtile_ctr'
      - 'atomic_add_agent\(\s*arg_mtile_ctr[\s\S]*fx\.Int32\(0\)\s*-\s*num_n_blocks'
      - 'if\s+tx_i32\s*<\s*fx\.Int32\(BM\)'
      - 'atomic_add_system\(\s*ready_base'

  - id: bucket_512_payload_rows
    category: plan
    severity: required
    kind: regex
    file: aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
    patterns:
      - 'payload_chunk_rows\s*=\s*256\s+if\s+bucket\s*==\s*512\s+else\s+384'
```

## Embedded runtime validation

This is the runtime half of the same acceptance chain. Kernel Workflow invokes
it only after selecting and applying the guidance. It validates the resulting
claims; it does not make an applicability decision. The default helper path is
a non-normative adapter hint, and the candidate entry class is discovered from
its semantic constructor signature.

```expert-skill-runtime-python
#!/usr/bin/env python3
"""Profile-driven graph, launch, resource and paired-performance validation."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import statistics
import sys
from pathlib import Path


PROFILE = {
    "world": 8,
    "model_dim": 7168,
    "inter_dim": 3072,
    "experts": 384,
    "topk": 6,
    "swiglu_limit": 10.0,
}


def _csv_ints(value):
    return [int(item) for item in value.split(",") if item.strip()]


def _csv_strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime adapter {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _discover_adapter(tree, explicit):
    candidates = []
    if explicit:
        path = Path(explicit)
        candidates.append(path if path.is_absolute() else tree / path)
    # Non-normative adapter hint for the validated profile.
    candidates.append(tree / "op_tests/multigpu_tests/test_mega_moe_v2.py")
    candidates.extend(sorted(tree.glob("**/test_*moe*.py")))
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = path.read_text(errors="ignore")
        if all(name in text for name in ("_reference", "_setup_dist", "_quantize_weights")):
            helper = _load_module(path, "_expert_skill_profile_adapter")
            if all(
                callable(getattr(helper, name, None))
                for name in ("_reference", "_setup_dist", "_quantize_weights", "_barrier")
            ):
                return helper, path
    raise RuntimeError("no semantic runtime adapter was discovered")


def _discover_profile(helper, profile_name):
    required = {"model_dim", "inter_dim", "experts", "topk"}
    for value in vars(helper).values():
        if not isinstance(value, dict) or profile_name not in value:
            continue
        profile = value[profile_name]
        if isinstance(profile, dict) and required <= set(profile):
            return dict(profile)
    raise RuntimeError(f"runtime adapter has no profile {profile_name!r}")


def _discover_factory(helper):
    required = {
        "rank", "world_size", "model_dim", "inter_dim", "experts", "topk",
        "quant", "w1", "w1_scale", "w2", "w2_scale", "max_tok_per_rank",
    }
    matches = []
    for value in vars(helper).values():
        if not inspect.isclass(value):
            continue
        try:
            parameters = set(inspect.signature(value).parameters)
        except (TypeError, ValueError):
            continue
        if required <= parameters:
            matches.append(value)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one semantic candidate factory, discovered {len(matches)}"
        )
    return matches[0]


def _make_route(torch, tokens, route, iteration, profile, rank, world, device):
    generator = torch.Generator(device=device).manual_seed(
        123 + rank + iteration * 1009 + tokens * 17
    )
    x = torch.randn(
        (tokens, profile["model_dim"]),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    destination = torch.topk(
        torch.rand((tokens, world), device=device, generator=generator),
        profile["topk"],
        dim=-1,
    ).indices
    local_experts = profile["experts"] // world
    if route == "uniform":
        local = torch.randint(
            0, local_experts, destination.shape, device=device, generator=generator
        )
    elif route == "rank-mixed-skew":
        cold = torch.randint(
            1, local_experts, destination.shape, device=device, generator=generator
        )
        local = torch.where(destination < world // 2, torch.zeros_like(cold), cold)
    else:
        raise ValueError(f"unsupported route {route!r}")
    logits = torch.randn(
        destination.shape, dtype=torch.float32, device=device, generator=generator
    )
    return (
        x.contiguous(),
        logits.softmax(dim=-1).contiguous(),
        (destination * local_experts + local).to(torch.int32).contiguous(),
    )


def _relative_l2(torch, dist, helper, output, reference, device):
    value = float(
        (
            torch.linalg.vector_norm(output.float() - reference)
            / torch.linalg.vector_norm(reference)
        ).item()
    )
    return helper._reduce_float(value, device, dist.ReduceOp.MAX)


def _capture(torch, helper, body):
    helper._barrier()
    body()
    helper._barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _time_rank_max(torch, dist, helper, graph, device, iters):
    helper._barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return helper._reduce_float(
        start.elapsed_time(end) / iters, device, dist.ReduceOp.MAX
    )


def _profile_launch_count(torch, dist, helper, body, device):
    from torch.profiler import ProfilerActivity, profile

    helper._barrier()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        body()
        torch.cuda.synchronize()
    count = 0
    for event in trace.key_averages():
        name = str(event.key).lower()
        device_time = float(
            getattr(event, "device_time_total", 0)
            or getattr(event, "self_device_time_total", 0)
            or 0
        )
        if device_time > 0 and not any(tag in name for tag in ("memcpy", "memset")):
            count += int(event.count)
    lo = torch.tensor(count, dtype=torch.int32, device=device)
    hi = lo.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    if int(lo.item()) != 2 or int(hi.item()) != 2:
        raise AssertionError(
            f"BF16 path must emit exactly two kernels per rank, got "
            f"{int(lo.item())}..{int(hi.item())}"
        )
    return 2


def _resource_checks(path):
    if not path:
        return {
            "status": "delegated",
            "required_by_acceptance": True,
            "reason": "outer verifier must supply emitted metadata",
        }
    data = json.loads(Path(path).read_text())
    rows = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(rows, dict):
        raise ValueError("resource evidence must contain a cases object")
    limits = {
        "fixed_128": {
            "threads": 256, "group_segment": 32144, "kernarg": 440,
            "vgpr": 168, "sgpr": 106, "vgpr_spills": 0,
            "sgpr_spills": 106, "private": 0,
        },
        "large_8192": {
            "threads": 512, "group_segment": 160400, "kernarg": 440,
            "vgpr": 256, "sgpr": 106, "vgpr_spills": 80,
            "sgpr_spills": 116, "private": 316,
        },
    }
    for case, bound in limits.items():
        row = rows.get(case)
        if not isinstance(row, dict):
            raise AssertionError(f"missing resource case {case}")
        for key, maximum in bound.items():
            value = int(row.get(key, -1))
            if value < 0 or value > maximum:
                raise AssertionError(
                    f"resource {case}.{key}={value} exceeds {maximum}"
                )
        if int(row.get("resident_workgroups", 1)) < 1:
            raise AssertionError(f"resource {case} has no resident workgroup")
    return {"status": "pass", "cases": sorted(limits)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--profile-adapter", default="")
    parser.add_argument("--profile", default="v4_pro")
    parser.add_argument("--accuracy-cases", default="128,512,8192")
    parser.add_argument("--liveness-cases", default="128,512,8192")
    parser.add_argument("--routes", default="uniform,rank-mixed-skew")
    parser.add_argument("--replays", type=int, default=256)
    parser.add_argument("--numeric-checkpoint-interval", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=0.10)
    parser.add_argument("--perf-iters", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=1.03)
    parser.add_argument("--resource-evidence", default="")
    parser.add_argument("--json-output", required=True)
    args = parser.parse_args()

    tree = Path(args.candidate_tree).resolve()
    sys.path.insert(0, str(tree))
    helper, adapter_path = _discover_adapter(tree, args.profile_adapter)

    import torch
    import torch.distributed as dist

    accuracy_cases = _csv_ints(args.accuracy_cases)
    liveness_cases = _csv_ints(args.liveness_cases)
    routes = _csv_strings(args.routes)
    required_cases = {128, 512, 8192}
    if not required_cases <= set(accuracy_cases) or not required_cases <= set(liveness_cases):
        raise ValueError("runtime claims require direct 128/512/8192 cases")
    if args.replays <= 0 or args.pairs <= 0 or args.perf_iters <= 0:
        raise ValueError("replays, pairs and perf-iters must be positive")

    rank, world, device = helper._setup_dist()
    result = {
        "schema_version": "expert-skill-runtime-evidence-v1",
        "claim_complete": False,
        "validation_scope": "post-selection claims only",
        "adapter": adapter_path.relative_to(tree).as_posix(),
        "accuracy_results": [],
        "replay_results": [],
        "paired_readings": [],
    }

    def write_result(complete):
        result["claim_complete"] = bool(complete)
        if rank == 0:
            output = Path(args.json_output)
            output.parent.mkdir(parents=True, exist_ok=True)
            tmp = output.with_suffix(output.suffix + ".tmp")
            tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, output)

    try:
        if world != PROFILE["world"]:
            raise AssertionError(f"validated claim requires 8 ranks, got {world}")
        profile_data = _discover_profile(helper, args.profile)
        for key in ("model_dim", "inter_dim", "experts", "topk"):
            if int(profile_data[key]) != int(PROFILE[key]):
                raise AssertionError(
                    f"selected profile claim differs at {key}: {profile_data[key]}"
                )
        profile_data["swiglu_limit"] = float(
            profile_data.get("swiglu_limit", PROFILE["swiglu_limit"])
        )
        factory = _discover_factory(helper)
        local_experts = profile_data["experts"] // world
        packed = helper._quantize_weights(
            profile_data["model_dim"],
            profile_data["inter_dim"],
            local_experts,
            rank,
            123,
            device,
        )
        w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
        ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
        max_tokens = _next_power_of_two(max(accuracy_cases + liveness_cases))

        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_COMBINE"] = "1"
        os.environ["AITER_MEGAMOE_FUSE_QUANT"] = "0"
        candidate = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=max_tokens,
            **profile_data,
        )

        target_body = None
        target_tensors = None
        for tokens in sorted(set(accuracy_cases + liveness_cases)):
            x, route_weights, ids = _make_route(
                torch, tokens, "uniform", 0, profile_data, rank, world, device
            )
            state = {}

            def body():
                state["output"] = candidate(x, route_weights, ids)[:tokens]

            graph = _capture(torch, helper, body)
            if tokens == 8192:
                target_body = body
                target_tensors = (x, route_weights, ids, graph)

            if tokens in accuracy_cases:
                reference = helper._reference(
                    x,
                    route_weights,
                    ids,
                    ref_weights,
                    rank,
                    world,
                    profile_data["model_dim"],
                    profile_data["inter_dim"],
                    profile_data["experts"],
                    profile_data["swiglu_limit"],
                )
                rel_l2 = _relative_l2(
                    torch, dist, helper, state["output"], reference, device
                )
                row = {
                    "guard": str(tokens),
                    "metric": "relL2",
                    "value": rel_l2,
                    "threshold": args.rtol,
                    "method": "direct graph-captured candidate vs numeric reference",
                    "status": "pass" if rel_l2 < args.rtol else "fail",
                }
                result["accuracy_results"].append(row)
                write_result(False)
                if row["status"] != "pass":
                    raise AssertionError(f"tokens={tokens} relL2={rel_l2:.6f}")

            if tokens in liveness_cases:
                for route in routes:
                    checkpoints = []
                    for replay in range(args.replays):
                        _, new_weights, new_ids = _make_route(
                            torch,
                            tokens,
                            route,
                            replay + 1,
                            profile_data,
                            rank,
                            world,
                            device,
                        )
                        route_weights.copy_(new_weights)
                        ids.copy_(new_ids)
                        graph.replay()
                        if (
                            (replay + 1) % args.numeric_checkpoint_interval == 0
                            or replay + 1 == args.replays
                        ):
                            torch.cuda.synchronize()
                            reference = helper._reference(
                                x,
                                route_weights,
                                ids,
                                ref_weights,
                                rank,
                                world,
                                profile_data["model_dim"],
                                profile_data["inter_dim"],
                                profile_data["experts"],
                                profile_data["swiglu_limit"],
                            )
                            value = _relative_l2(
                                torch, dist, helper, state["output"], reference, device
                            )
                            checkpoints.append(value)
                            if value >= args.rtol:
                                raise AssertionError(
                                    f"{tokens}/{route} replay relL2={value:.6f}"
                                )
                    replay_row = {
                        "guard": f"{tokens}_{route}",
                        "count": args.replays,
                        "status": "pass",
                        "graph_safe": "pass",
                        "arrival_jitter": True,
                        "routing_changes": args.replays,
                        "max_checkpoint_relL2": max(checkpoints),
                    }
                    result["replay_results"].append(replay_row)
                    write_result(False)

        if target_body is None or target_tensors is None:
            raise AssertionError("8192 target graph was not produced")
        launch_count = _profile_launch_count(
            torch, dist, helper, target_body, device
        )
        result["activation"] = {
            "status": "pass",
            "ranks": world,
            "fusion_switch": os.environ["AITER_MEGAMOE_FUSE_ALL"],
        }
        result["launch_count"] = launch_count
        result["resources"] = _resource_checks(args.resource_evidence)

        x, route_weights, ids, candidate_graph = target_tensors
        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "0"
        control = factory(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            max_tok_per_rank=max_tokens,
            **profile_data,
        )
        control_state = {}

        def control_body():
            control_state["output"] = control(x, route_weights, ids)[:8192]

        control_graph = _capture(torch, helper, control_body)
        os.environ["AITER_MEGAMOE_FUSE_ALL"] = "1"
        ratios = []
        for pair in range(args.pairs):
            if pair % 2:
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
            else:
                base_ms = _time_rank_max(
                    torch, dist, helper, control_graph, device, args.perf_iters
                )
                cand_ms = _time_rank_max(
                    torch, dist, helper, candidate_graph, device, args.perf_iters
                )
            ratios.append(base_ms / cand_ms)
            result["paired_readings"].append(
                {"guard": "8192_uniform", "base": base_ms, "cand": cand_ms}
            )
            write_result(False)
        speedup = statistics.median(ratios)
        result["paired_rank_max_speedup"] = speedup
        result["paired_control"] = (
            "same selected tree with persistent fusion disabled; "
            "outer workflow still owns the frozen-baseline pair"
        )
        if speedup < args.min_speedup:
            raise AssertionError(
                f"paired rank-max speedup {speedup:.6f} < {args.min_speedup}"
            )

        result["correctness"] = "pass"
        result["graph_safe"] = "pass"
        result["liveness"] = "pass"
        result["liveness_replays"] = args.replays
        write_result(True)
        return 0
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        write_result(False)
        raise
    finally:
        helper._cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
```
