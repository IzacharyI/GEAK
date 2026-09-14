---
reference_id: megamoe_m25_tile_pipeline
reference_revision: m25-repro-v2
baseline_identity: workflow_supplied_frozen_tree
mode: mega
normative: true
normative_scope: semantic_and_compiler_shape
source: validated_knowledge
oracle_role: post_authoring_comparison_only
---

# MegaMoE M2.5 independent implementation contract v2

This file is the independent implementation contract distilled from one
validated tiled/instruction-pipelined design produced from the frozen public
MegaMoE V2 baseline. When the baseline revision and match fields apply, its
`MUST`, `MUST NOT`, queue, memory-order, and compiler-shape rules are normative
inside the ordinary Mega planner/Engineer lifecycle. Current hardware evidence
may override a rule only when the candidate records the contradiction and the
replacement invariant explicitly. It does not create a reproduction mode,
candidate source, or reserved lane.

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

With that switch set, the complete M2.5 topology is selected whenever
`stage1.num_waves % 4 == 0`. `AITER_MEGAMOE_FUSE_COMBINE` may remain as a
diagnostic opt-out, but its default is `"1"`. Quant fusion is not part of this
recipe and remains off.

For a scored BF16 M2.5 arm, require all ranks to resolve:

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

Expose one host helper named `_m25_variant_enabled(config)` that resolves and
validates this complete environment before launch. Do not let independent
call sites infer “MEGA” from `FUSE_ALL` alone.

The candidate is complete only when all implementation checkpoints in section
13 are true. Intermediate commits are source checkpoints, not alternative
topologies and not performance candidates.

## 2. Source boundary

Start from the workflow-supplied frozen baseline tree. The M2.5 runtime
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
not a requirement for the final M2.5 Stage1-hosted third queue. The quant
emitter is optional M3 infrastructure; M2.5 keeps quant as its own launch.
Neither file may be required merely because it differs in a later oracle tree.

Do not edit tests or benchmarks and do not commit scripts, logs, caches, dumps,
or evidence to the candidate tree. Store all such artifacts in `OUTPUT_DIR`.
Do not read or diff an external M2.5 source tree. The implementation must be
authored from this recipe and the frozen baseline.

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
it must add a system release before the ready atomic; the validated M2.5 path
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
exactly. A clean M2.5-only implementation may omit later wait-stat/M3 fields,
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
- system atomic: GEMM1 completion publication and remote peer token-ready
  publication;
- owner-only ordinary load/store: combine generation;
- workgroup release plus block barrier: Stage2 tile-close ordering.

Preserve the visibility domain, not a simplified “rank-local means agent”
rule. Relaxed MORI waits across XCDs are the reason the G1 completion uses
system scope.

### GEMM1 to GEMM2

GEMM1 activation and scale stores use the gfx95x system-visible write-through
cache modifier. After one GEMM1 tile:

1. all waves execute `s_waitcnt(0)`;
2. execute a block barrier;
3. thread 0 performs `atomic_add_system` on the completion counter for
   `unit // n_tiles`.

The counter threshold is `n_tiles`: all N-column tiles for that Stage1 m-tile.
The validated gfx950 source uses system scope even though the logical edge is
rank-local: its consumer is a relaxed MORI wait across XCDs. Do not substitute
agent scope based only on the abstract edge classification. The write-through
stores plus `s_waitcnt(0)` and the block barrier are the producer ordering; do
not add a separate per-tile `fence_system_release()`. The consumer waits with
`int32_wait_until_greater_than(counter, n_tiles - 1)`. There is no whole-grid
`ready1 == NUM_G1_BLOCKS` barrier.

The publication lives in the same unified `kind/unit` work loop immediately
after `_do_scheduled_tile(unit)`. It uses the direct `s2_ctr` kernel argument.
Do not tunnel the counter through an extended dispatch-pointer table or move
GEMM2 into a second post-loop drain: both change the compiler frame and are not
the validated M2.5 source shape.

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
3. otherwise claim one GEMM1 tile from this block's sharded GEMM1 head;
4. once GEMM1 is drained, claim one sharded GEMM2 chunk and wait only for the
   last in-range pair's own m-tile counter;
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
- Preserve `gemm2_compute_v2` arithmetic; the recipe changes scheduling and
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

For a candidate implementing the complete reference design:

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

The recorded M2.5 band is a reproduction target, not permission to skip any
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
python kernel_workflow/tools/m25_structural_contract.py \
  --baseline <frozen-baseline> \
  --reference <read-only-M2.5-oracle> \
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
digest in the manifest with `sealed_before_oracle_comparison=true` and
`oracle_exposed_during_authoring=false`. This remains an orchestration
attestation (`provenance_status=attested_unverified`), not cryptographic proof
of what the author could access.

Static validation can establish syntax, call-graph and source-structure
coverage. It cannot establish device compilation, actual launch count,
correctness, deadlock freedom, graph replay safety, or performance. Until a
later EP8 run passes those gates, record all of `hardware_verified`,
`accuracy_verified`, and `performance_verified` as false.
