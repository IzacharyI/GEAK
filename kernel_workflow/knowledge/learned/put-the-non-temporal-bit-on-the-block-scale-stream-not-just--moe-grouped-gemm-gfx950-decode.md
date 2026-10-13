---
key: cache policy on the per-block scale metadata stream of an mxfp4 a4w4 grouped GEMM, which the kernel's own _nt flag does not cover
type: lever
confidence: ★★
effect: One non-temporal bit on stage-1's w1 SCALE stream, 8 functional lines, moved weighted 1.1432 -> 1.2858; per-case it is what finally unstuck the decode bucket holding 22.45% of the metric (0.9902 -> 1.1728) after three rounds of fill levers had left it inside its own MDE.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: flydsl 0.3.4.1 / aiter@4faf8713 / ROCm 7.2
last_seen: 2026-10-12
name: put-the-non-temporal-bit-on-the-block-scale-stream-not-just--moe-grouped-gemm-gfx950-decode
description: mxfp4 MoE: the scale stream is 1/16 of the bytes but evicts the other 15/16 from a 256 MiB LLC; one NT bit on it is worth +12.5pp weighted.
keywords: ['moe-grouped-gemm', 'mxfp4', 'block-scale', 'cache-modifier', 'non-temporal-loads', 'l2-residency', 'llc-capacity', 'infinity-cache', 'decode', 'flydsl', 'gfx950']
kernels: ['gemm1_a4w4_port']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: decode
language: flydsl
decision_refs: []
layer: learned
lifecycle: active
origin_kernels: ['glm52_mxfp4_fused_moe']
cost: L1
verified_on: 2026-10-12
levers: ['mem.load-cache-policy']
---
# Put the non-temporal bit on the block-scale stream, not just the weight data
- lever: Treat every stream the kernel loads as its own cache-policy decision -- weight data, block scales, activations, output -- and A/B the scale stream separately; it is small enough to look irrelevant and large enough to evict everything else.
- apply: One cache_modifier argument on the scale-tile load. Scales are loads only, so parity is free and the output stays bit-identical.
- stack: Independent of tile geometry and of split-K; compose it with a fill lever rather than choosing between them.
- verify: Prove the LLC-capacity premise with a working-set sweep rather than a profiler: here 129 MiB read -7.5%, 258 MiB -12.6%, and past overflow it plateaued at -13%. A warm profiler cannot see this win at all and will report it as a regression.
- pitfall: A kernel-name tag like _nt names ONE operand, not the kernel -> the flag drove only the B data stream while the scale stream stayed allocating -> an axis written off as 'cache policy is closed' cost a full campaign.
A non-temporal hint on a small stream pays only where a large ALLOCATING neighbour is being evicted -> on a stream with no such neighbour it measures as a plausible null.
- caution: Verify the hint actually reached the ISA; a policy edit that flips one bit with no ISA delta is the silent no-op failure mode, and it reads exactly like a real null.
- source: run wwd7t6625 / team_volck_A_20261012_082749 round 2, 2026-10-12, gfx950/MI355X; director weighted 1.3205 correctness pass, re-verified 1.3212 and 1.3139 in a kernel-only interleaved A/B
