---
title: fused_moe_grouped_gemm on tilelang — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: tilelang
gens: [gfx942, gfx950]
dtypes: [fp16]
regimes: [prefill]
status: experimental
updated: 2026-10-13
sources:
  - https://github.com/tile-ai/tilelang
  - https://github.com/tile-ai/tilelang/tree/main/examples/fusedmoe
---

# fused_moe_grouped_gemm × tilelang

## TL;DR (status: experimental — a real routed grouped GEMM, but an example, not an op)
> TileLang **can** express this operator, and on this box it already does: the bundled example
> `examples/fusedmoe/example_fusedmoe_tilelang.py` is a genuine grouped GEMM over sorted/padded
> expert tiles — it takes `group_sizes` / `group_offsets` / `group_padded_offsets` /
> `group_idx_for_bx` and selects the expert's weight slice per tile block, the same problem shape as
> aiter's `sorted_expert_ids` path. What it is **not** is a backend you can select at serving time:
> there is **no MoE operator anywhere in the `tilelang` library itself** — the example is a
> KernelBench-style benchmark harness, **fp16 only**, with no quantized dtype. Treat it as a
> *prototyping language for this operator*, not a shipping path. For serving use
> [`aiter.md`](aiter.md); for an authorable kernel on gfx950 MXFP4 use [`flydsl.md`](flydsl.md).

## What exists on this box
Checkout `/opt/tilelang` at `a55a8230`, installed as `tilelang 0.1.7.post3+cuda.gita55a8230`
(loaded from the dev root `/opt/tilelang/build`).

| thing | path | what it is |
|---|---|---|
| `moe_forward_tilelang_routed` | `examples/fusedmoe/example_fusedmoe_tilelang.py:102` | the routed grouped GEMM — **two** kernel launches |
| `moe_forward_tilelang_shared` | same file `:12` | the dense shared-expert path (un-routed) |
| torch reference | `examples/fusedmoe/example_fusedmoe_torch.py` | the oracle the example compares against |
| regression + test | `regression_example_fusedmoe.py`, `test_example_fusedmoe.py` | CI harness |
| library op | — | **absent**: no file under `/opt/tilelang/tilelang/` matches `moe`, and no module there mentions it |

### Structure: 2-stage, not a mega-kernel
The example splits exactly where aiter's 2-stage path splits:
- **Step 1** (`:153-200`): `T.gemm` gate + `T.gemm` up → SiLU → writes `up_logits` **to global memory**.
- **Step 2** (`:203-235`): `T.gemm` down → routed-weight-scaled write to `output`.

Per-tile expert routing is `cur_group_idx = group_idx_for_bx[bx]` (`:165`, `:212`); the sort/pad
bookkeeping tensors are built host-side (`:350-394`). So it is comparable to aiter's
asm-stage1 + CK-stage2 shape, **not** to the single-launch `fmoe*` mega-kernel
([`asm.md`](asm.md)) — it pays the same intermediate HBM round-trip the 1-stage path exists to avoid.

## Config space / knobs
The usual TileLang levers (`T.Kernel` grid, block/tile sizes, pipeline stages, shared-memory
staging) apply, because this is source you own. What the example does **not** parameterize:
dtype (`dtype_str = T.float16`, `:459`), quantization (none), or activation (SiLU hardcoded).
Adding fp8/MXFP4 is new work, not a flag.

## Numerics / parity
fp16 in, fp16 out, fp32 accumulate in `T.gemm`. The example ships its own torch oracle, which is the
right parity gate for a prototype; it is **not** the operator-level gate — see
[`../numerics.md`](../numerics.md) before concluding anything about serving parity.

## Pitfalls & anti-patterns
- ⚠ **Example ≠ op.** `import tilelang` gives you no MoE entry point. Anything built on this file is
  built on a benchmark script whose API (`custom_kernel` / `generate_input` / `ref_kernel`) is
  KernelBench's, not a stable library contract.
- ⚠ **fp16 only.** Every serving-relevant MoE path on this box is fp8 or fp4. A fp16 comparison
  against aiter's fp8 fused path measures two different problems — see
  [`../tuning.md`](../tuning.md) on comparing like with like.
- ⚠ **CUDA-flavored build tag.** The installed version string is `+cuda.gita55a8230`; confirm the
  HIP/ROCm target actually compiles for your gfx before planning around it.

## How to verify
Run the example's own regression (`regression_example_fusedmoe.py`) first — it proves the routed path
compiles and matches the torch reference on this box. Only then is a timing comparison meaningful,
and it must be against an **fp16** aiter configuration, not the fp8/fp4 serving default.

## Alternatives / cross-links
[`aiter.md`](aiter.md) (serving) · [`asm.md`](asm.md) (the 1-stage mega-kernel this example is not) ·
[`ck.md`](ck.md) · [`flydsl.md`](flydsl.md) · [`triton.md`](triton.md) ·
[`../overview.md`](../overview.md) · [`../tuning.md`](../tuning.md)

## Sources
- On-box checkout `/opt/tilelang` @ `a55a8230` (2026-01-29), package
  `tilelang.__version__ == "0.1.7.post3+cuda.gita55a8230"`. Cited by path rather than
  `repo@commit:path` because `/opt` is outside the citation resolver's search roots, so a pinned
  citation could not be verified by `index/resolve_citation.py` on this box.
- Routed grouped GEMM: `/opt/tilelang/examples/fusedmoe/example_fusedmoe_tilelang.py:102`
  (`moe_forward_tilelang_routed`), `:12` (`moe_forward_tilelang_shared`), `:153-200` (stage 1),
  `:203-235` (stage 2), `:165`/`:212` (`group_idx_for_bx[bx]`), `:350-394` (host-side sort/pad),
  `:459` (`dtype_str = T.float16`). 625 lines total.
- No library op: `find /opt/tilelang/tilelang -iname '*moe*'` and
  `grep -rl moe /opt/tilelang/tilelang --include=*.py` both return nothing (2026-10-13).
- Upstream: https://github.com/tile-ai/tilelang
