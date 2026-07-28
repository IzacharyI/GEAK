---
id: flydsl_decode_moe_stage1_blkmap
title: Halve grouped-MoE stage-1 weight HBM traffic at decode by fusing paired same-expert sort-blocks into one double-height compute tile (expert weight streamed once, reused across both blocks)
kind: expert_skill
authors:
- yz
scope: kernel
match:
  operator: grouped_gemm_moe
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp4
  - fp4_e2m1
  - fp8_e4m3
  - fp8_e4m3_fnuz
  regimes:
  - decode
  from_backend: ''
  to_backend: ''
  profile_signature:
    op_name_regex: ''
    min_pct_gpu: 0.0
expects:
  isolated_speedup_min: 1.05
  isolated_scope: 'stage-1 SEGMENT = the gate+up GEMM kernel PLUS the descriptor-producer kernel this
    recipe adds. The producer is a separate dispatch whose name does not contain the GEMM name, so a
    GEMM-only timing filter drops it and overstates the win by ~1.9pp (1.103x GEMM-only vs 1.080x with
    the producer counted). Count the producer.'
  e2e_delta_min_pct: 1.0
  parity: required
validation:
  status: validated
  last_verified: '2026-07-27'
  gpu: 'gfx950 / MI355X'
  model: 'fp8-act / fp4-wt grouped-MoE decode (provenance only; selection is by shape/bottleneck, not model)'
  measured:
    isolated: 'stage-1 gate+up GEMM 185.7 -> 168.4 us = 1.103x (-9.3%) on a base tile_m=64 kernel with 86
      of 347 sort-blocks pairable; 3-trial average 179.5+-0.3 -> 162.5+-0.5 us (-9.5%), independent repeat
      -9.1%. Counting the +3.62us descriptor producer the segment is 185.7 -> 172.0 us = 1.080x (-7.4%).
      Weight-HBM read units 347 -> 261 (analytic count over the real routing distribution, floor 259 --
      NOT a hardware counter). Device-time sum over the whole fused decode path 313.9 -> 301.3 us (-4.0%).
      INDEPENDENT REPRODUCTION on a second tree, 4 interleaved trials, selection driven purely by the
      tuner config (no env override): stage-1 segment 192.69 -> 174.59 us = 1.1037x (-9.4%), arm ranges
      191.38-195.32 vs 173.20-174.76 (non-overlapping); GEMM-only 192.69 -> 170.92 = 1.1274x; descriptor
      producer 3.68 us; whole fused decode path 352.10 -> 333.25 us = 1.0566x.'
    e2e_pct: ''
    parity: 'pass - numerically equivalent to the unpaired tile (logits_diff delta <4e-6 vs baseline, cos_sim 0.99893); differs only by reduction order'
  artifact: ''
role: advisory_prior
supersedes: []
---

## When to use
Trigger on the **problem signature, not a specific model**: a **grouped-GEMM MoE stage-1 (gate+up)** kernel
at **decode / small-M** that is **weight-HBM-traffic-bound**, where the MoE token-sort has padded each expert
to a fixed sort-block granularity `B` (commonly 64 rows) and the routing gives many experts **>=2 sort-blocks**.
In that regime a per-sort-block compute tile re-reads the expert's (low-precision, e.g. fp4) weight tile from
HBM once per block, so a `k`-block expert streams its identical weight `k` times — redundant traffic that pins
the kernel below its shared-weight floor. Inert for compute-bound stage-1 (prefill / large-M) or routing where
almost every expert owns a single block.

**Two hard preconditions — check BOTH before attempting, and report them either way:**
1. **The baseline must select a `tile_m = 64` compute tile.** If the baseline already picks a 128-row tile,
   doubling it needs ~263 KB of LDS against a ~160 KB limit and **will not build**. The recipe is
   inapplicable on that config, not merely unprofitable.
2. **A substantial fraction of sort-blocks must be pairable** (adjacent same-expert pairs). Count them from
   the sorted block->expert table *first*: the reference win had **86 pairs / 347 blocks (~25%)**. Routing
   with only ~2% pairable caps the achievable speedup near **1.01x**, below this skill's own `expects`.

A re-verification run measured exactly **1.0000x** because both preconditions failed at once (2.17%
pairable, and a baseline already on a 128-row tile). That is an **out-of-applicability shape, not a
refutation** — report it that way instead of as a failed reproduction.

## Mechanism
Why the redundancy exists: MoE align/sort rounds each expert's token count **up** to the sort-block `B` so the
grouped GEMM can index fixed-size row-blocks; the kernel streams the expert's full weight tile per `B`-row
block. An expert holding `k` sort-blocks therefore reads its weight from HBM `k` times even though the weight
is identical across those blocks. At decode each block is a thin sliver of useful rows, so this redundant
weight read dominates stage-1 HBM traffic.

The lever: **fuse two adjacent same-expert sort-blocks into one `2B`-row compute tile.** The expert weight is
loaded to registers/LDS **once** and MFMA'd against both `B`-row halves, so weight HBM reads for paired experts
are halved (approaching the floor where each expert's weight is read exactly once). The decisive property is
that the **sort-block granularity `B` is left unchanged** — only the stage-1 compute tile's row-height doubles
— so the downstream stage-2 (down) padding and work are identical; the reuse is free of any stage-2 cost. It is
a pure **data-reuse / traffic** optimization: the arithmetic is the same accumulation over a larger `BLOCK_M`,
so the result is numerically equivalent modulo reduction order (not a new numeric scheme).

## Procedure
1. **Confirm the bottleneck by ablation.** Stage-1 must be weight-traffic-bound (weight-bytes streamed per
   token >> activation bytes; measured HBM read ~= sum over experts of `k_e x weight_bytes`). If most experts
   are single-block, there is no reuse to win — skip. **Also verify the two hard preconditions in *When to
   use* (base `tile_m=64`; pairable-block fraction) and record the pairable count — it bounds the best
   achievable speedup, so compute that bound before spending any effort.**
2. **Build a compact leader-block descriptor.** Scan the sorted-block -> expert table; for each run of
   same-expert blocks emit `ceil(k/2)` "leader" tiles, each covering two `B`-row sub-blocks (a trailing odd
   block becomes a solo leader covering one). Compute this **device-side and CUDA-graph-safe** — preallocated
   outputs, no host sync, deterministic shapes — so it works under graph capture; an in-kernel parity scan or a
   host-side build is a slower fallback.
3. **Launch stage-1 over the halved leader grid** (~`ceil(total_blocks/2)` CTAs); excess CTAs early-exit.
4. **In the kernel, set `BLOCK_M = 2B`:** load the expert weight tile once and MFMA it against both `B`-row
   halves; accumulate / run the epilogue exactly as the unpaired kernel does per half.
5. **Store masking (the one correctness-critical step).** A solo leader (odd tail) must mask the store of its
   absent upper half, else it double-writes / goes out of bounds. Carry a per-leader "store-second-half"
   predicate and honor it at the store.
6. **Leave stage-2 (down) on the original `B`-row blocks** — do not widen the sort-block.
7. **Validate parity at the SAME tile granularity as the baseline** (the paired kernel is numerically
   equivalent, so logits must match within reduction-order noise), then measure same-session A/B and confirm
   the weight-HBM-read drop with hardware counters (rocprof).

## Knobs & pitfalls
- **Pairing (merge factor 2) is the validated setting.** Merging `>2` blocks widens `BLOCK_M` further and risks
  LDS / register / occupancy limits — measure before assuming it helps.
- **Keep the base sort-block granularity as the baseline uses it.** Forcing the paired (double-height) path on
  top of an already-large base compute tile exceeds the compile-time LDS/register budget and fails to build
  (measured: a 128-row base tile would need ~263 KB LDS against a ~160 KB limit).
- **The descriptor producer must be ONE kernel.** The reference first built it with eager tensor ops: ~13 tiny
  dispatches, **~47 us/iter**, which turned the whole optimization into a net **loss**. A single fused
  producer costs **~3.6 us**. An in-kernel parity scan instead of a producer measured *worse than baseline*
  (202 us vs 162 us). Budget the producer explicitly against the GEMM saving (~17 us) before starting.
- **Time the producer together with the GEMM.** The producer is a separate dispatch with an unrelated kernel
  name, so a GEMM-name timing filter silently drops it and reports 1.103x where the honest segment number is
  1.080x. See `expects.isolated_scope`.
- **The win scales with the fraction of `>=2`-block experts.** Synthetic / uniform routing overstates it vs
  skewed real decode routing — report real-routing numbers.
- **A `tile_m = 128` tuning entry, where one exists, beats this recipe — compare against it before claiming a
  win.** Pairing reaches a `2B`-row compute tile from `B`-row sort blocks; a config that simply *sorts* at
  `2B` reaches the same tile with no descriptor kernel and fewer weight reads (measured on the reference
  shape: paired-`t64` segment 174.6 us vs `t128` 168.6 us, because only 89 of 379 leaders actually pair, so
  290 solo leaders still read a weight tile for just `B` rows). This recipe's value is therefore confined to
  baselines that are **pinned** to `tile_m = B` — precondition 1 is about buildability, this is about
  profitability. State which baseline the speedup is measured against, and never compare a paired arm to a
  differently-tuned arm.
- **Make the merge factor selectable from the tuner's config, never from an environment variable alone.** In
  the reference the config tag was parsed but then dropped on the way to the kernel, so the tuner could not
  actually select pairing and the path silently never activated — a config-driven arm and an env-forced arm
  must be verified to emit the identical kernel before any A/B is trusted.
- **The descriptor producer must be graph-capturable;** a host-side scan defeats CUDA-graph use.

## Do-no-harm notes
- **Numerically equivalent** to the unpaired kernel (same math, larger tile) — but only if the odd-tail store
  mask is correct; a missing mask double-counts. Parity-gate every build.
- **Compute-bound stage-1 (prefill / large-M) sees little or no win** and the larger tile can cost occupancy —
  keep this to weight-traffic-bound decode; the workflow's on-box A/B picks the winner (advisory prior, never
  overrides measurement).
- **The paired mode is a pure add-on** — when not selected the kernel is byte-identical to the generic path, so
  a non-matching shape regresses nothing.

## Sources
Evidence is external and cited for **provenance only** — GEAK does not depend on any tree, and exact
commits / file paths are intentionally omitted. The portable knowledge is the signature, mechanism, and
measured numbers below.

- **Reference measurement** (same-session interleaved A/B; decode shape with **86 of 347 sort-blocks pairable**
  on a **base `tile_m=64`** kernel; fp8 activations / fp4 weights on gfx950 / MI355X):
  - stage-1 gate+up GEMM **185.7 -> 168.4 us = 1.103x (-9.3%)**; 3-trial average **179.5 -> 162.5 us (-9.5%)**,
    independent repeat **-9.1%**;
  - **counting the +3.62us descriptor producer: 185.7 -> 172.0 us = 1.080x (-7.4%)** — this is the honest
    segment number;
  - sorting / align kernels unchanged (18.30 -> 18.26 us, i.e. noise), confirming the sort-block granularity
    was untouched;
  - weight-HBM read units **347 -> 261** (floor 259), an **analytic count over the real routing
    distribution — not a hardware counter reading**;
  - **parity numerically equivalent** (logits_diff delta `<4e-6` vs the unpaired baseline, cos_sim `0.99893`)
    — the output differs only by reduction order.
- Implemented as an optional double-height-tile mode in a FlyDSL grouped-MoE stage-1 core with a device-side
  leader-block descriptor producer; the default (unpaired) path is unchanged. Any staged-rollout gating is an
  implementation detail of the reference, **not** part of this recipe — the transferable content is the
  pair-fuse-and-reuse technique above.
