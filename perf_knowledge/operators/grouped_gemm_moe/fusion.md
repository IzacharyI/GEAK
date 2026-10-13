---
title: grouped_gemm_moe — fusion
kind: technique
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp8_e4m3_fnuz, fp4_e2m1]
regimes: [prefill, decode]
updated: 2026-10-13
sources:
  - https://github.com/ROCm/aiter
  - https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
---

# grouped_gemm_moe — fusion

## TL;DR
> The big fused unit is the whole **fused_moe** op: sort/align → grouped gate/up GEMM → act-and-mul →
> grouped down GEMM → weighted combine, ideally as few launches as possible (AITER asm path fuses most of it).

## Fusion neighbors
- **act-and-mul (silu/gelu) between the two grouped GEMMs**: fuse as an epilogue of gate/up to avoid a
  round trip → [../act_and_mul_silu_gelu/overview.md](../act_and_mul_silu_gelu/overview.md).
- **Per-token/per-block quant epilogue**: quantize gate/up output to fp8/fp4 before the down GEMM, fused
  with the activation → [../fused_norm_quant/overview.md](../fused_norm_quant/overview.md),
  [../scaled_quant_gemm/fusion.md](../scaled_quant_gemm/fusion.md).
- **Routing + align&sort upstream**: [../moe_routing_topk/overview.md](../moe_routing_topk/overview.md),
  [../moe_dispatch_combine/overview.md](../moe_dispatch_combine/overview.md).
- **Weighted combine / scatter** as the down-GEMM epilogue (multiply by gate weight, scatter to token
  order) avoids a separate combine kernel.
- **Shared expert**: fuse the always-on shared expert with the routed experts →
  [../shared_expert_fusion/overview.md](../shared_expert_fusion/overview.md).
- **EP communication**: with expert parallelism, dispatch/combine all-to-all overlaps with the grouped
  GEMM → [../all_to_all_dispatch_combine/overview.md](../all_to_all_dispatch_combine/overview.md).

## Why it pays (Amdahl) — and why it pays differently than for dense GEMM
Dense GEMM fuses a memory-bound elementwise pass into a compute-bound matmul, so the epilogue is
nearly free. A grouped MoE has that case too (act-and-mul between the stages), but its **larger** win
is a different one: **launch count**. The unfused stack is sort → GEMM → act → GEMM → combine, five
dispatches whose intermediates each make a full round trip through HBM, and at decode batch sizes the
GEMMs themselves are small enough that per-launch overhead is a first-order term rather than a
rounding error.

Two consequences worth holding separately:
- **Prefill**: the Amdahl argument is dense GEMM's — the activation/quant epilogue removes a
  `[M, inter×2]` pass and that is most of the win.
- **Decode**: the win is dominated by dispatch and by the intermediate traffic, not by the epilogue
  FLOPs. ⚠ But a launch *saved* is not automatically time saved: under a captured CUDA graph the
  floor is **per-replay, not per-node**, so deleting one node can recover ~0 — measure the whole
  replay, never the node count. See [`./tuning.md`](./tuning.md).

## How it's expressed per backend
- **aiter asm**: the strongest form — a *single* kernel (`fmoe*`) that takes the sorted-expert layout
  and does gate+up, activation, down and the weighted reduce in one launch. You select it, you cannot
  author it: [`./backends/asm.md`](./backends/asm.md).
- **CK**: fusion lives in the CShuffle epilogue of each stage (`moe_ck2stages_*`) — activation and the
  routed-weight multiply as `CDEElementwise`; the two stages stay two kernels.
- **FlyDSL**: stage-1 can fuse activation **and requantization** in-kernel (silu_and_mul + fp4/fp8
  quant + tiled scale write), so stage-2's A operand is produced ready to consume — the fusion that
  makes the two-stage mxfp4 path competitive: [`./backends/flydsl.md`](./backends/flydsl.md).
- **triton/tilelang**: epilogue appended inside the kernel after the MFMA accumulation; the sort stays
  a separate launch.
- **hipBLASLt**: per-entry `GemmEpilogue` (bias/act/aux) only — SwiGLU's gate⊙up is **not** expressible
  as an epilogue there, so the act stays its own kernel: [`./backends/hipblaslt.md`](./backends/hipblaslt.md).

## Pitfalls
- ⚠ **The combine's fusion choice is a numerics choice.** Folding the weighted combine into the
  down-GEMM epilogue means an atomic add, which costs run-to-run determinism; the unfused `reduce`
  epilogue keeps it and pays a launch. Decide it as numerics, not as fusion —
  [`./numerics.md`](./numerics.md).
- ⚠ **Fusing changes the tuned-DB lookup key** (activation / quant / `doweight_stage1` flags), so a
  fused shape that was never captured misses the DB and silently falls to a default config — the same
  failure mode as dense's bias mismatch.
- ⚠ **BLOCK_M must match the align&sort pad.** Sort and both stages are coupled through it; fusing the
  sort prologue into a stage without carrying the pad corrupts real tokens' partial sums.
- ⚠ **A fused sort prologue inherits the routing draw's variance.** Recomputing routing per CTA makes
  the kernel's own runtime depend on the random routing of that process, which widens cross-process
  spread and makes small A/B deltas unreadable.
- Over-fusing rare paths multiplies kernel variants and tuning time for little Amdahl gain.

## Sources
- AITER fused_moe / asm MoE: https://github.com/ROCm/aiter , https://rocm.blogs.amd.com/artificial-intelligence/aiter-intergration-s/README.html
- Single-launch asm family and its binary-only distribution: [`./backends/asm.md`](./backends/asm.md),
  cited there to `ROCm/aiter@4faf8713:csrc/py_itfs_cu/asm_fmoe.cu` and `:csrc/py_itfs_cu/asm_moe_2stage.cu`.
  (The `a6bb49937` pin and `aiter/ops/moe_op.py` belong to the sibling family's card,
  [`../fused_moe_grouped_gemm/backends/asm.md`](../fused_moe_grouped_gemm/backends/asm.md) — the two
  are pinned at different commits on purpose, so do not merge their source lists.)
- Stage-1 in-kernel activation + requantization (FlyDSL): [`./backends/flydsl.md`](./backends/flydsl.md).
- Atomic-vs-reduce combine as a kernel choice, and weight-before-convert ordering: corpus decision cards
  `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag`,
  `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic`.
- BLOCK_M coupling sort↔both stages: corpus decision card `grouped-moe-sort-block-m-couples-both-stages`.
