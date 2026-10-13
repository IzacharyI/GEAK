---
title: grouped_gemm_moe on rocwmma — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: rocwmma
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz]
regimes: [prefill, decode]
status: na
updated: 2026-10-13
sources:
  - https://github.com/ROCm/rocWMMA
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# grouped_gemm_moe × rocwmma

## TL;DR (status: na — rocWMMA has no grouping layer to use)
rocWMMA is a **header-only, warp-tile fragment API** — `fragment` plus `fill_fragment`,
`load_matrix_sync`, `store_matrix_sync`, `mma_sync`, `synchronize_workgroup`, and nothing above that.
Grouped GEMM is entirely a *scheduling* problem: map tiles to variable-M groups through an
`expert_ids`/offset table, in one launch. rocWMMA supplies no host API, no driver, no batching
abstraction and no scheduler, so "grouped GEMM on rocWMMA" is not a thing you select — it is a kernel
you write yourself, in which rocWMMA replaces the four lines that issue MFMA. **Use
[`./aiter.md`](./aiter.md) (serving) or [`./flydsl.md`](./flydsl.md) / [`./triton.md`](./triton.md)
(authored and tunable).**

## Why na (not just "slower")
- The complete public surface is `rocwmma.hpp:75-354` (tags `row_major`/`col_major`/`matrix_a`/
  `matrix_b`/`accumulator`, the `fragment` class, and the five functions above). There is no entry
  point that takes more than one problem, and no host-side type at all.
- The package is headers only — `dpkg -L rocwmma-dev` lists `include/`, CMake config, and a LICENSE;
  there is no `.so` to link, confirming there is no runtime/driver component where a grouping layer
  could live.
- The grouped-GEMM levers this operator is actually tuned on — tile→group mapping from the offset
  table, per-expert M padding to the tile, CU-count fitting, persistent/stream-K scheduling (see
  [`../tuning.md`](../tuning.md)) — all sit above the fragment level. rocWMMA neither helps nor
  hinders them; it is orthogonal to the thing being optimized.

## What to do instead
- **Serving / default**: [`./aiter.md`](./aiter.md) — fused sort+GEMM+act+combine, DB-driven.
- **Authored and tunable**: [`./flydsl.md`](./flydsl.md), [`./triton.md`](./triton.md),
  [`./ck.md`](./ck.md).
- **If you genuinely want rocWMMA's fragments** (portability from a CUDA `wmma` kernel, or teaching),
  write the grouped kernel yourself and use rocWMMA only for the MMA inner loop — then expect the
  dense card's conclusion to carry over: it is a portability/teaching API, not a SOTA GEMM path
  ([`../../dense_gemm/backends/rocwmma.md`](../../dense_gemm/backends/rocwmma.md), status `legacy`).

## Sources
- Full public API enumerated on box, rocWMMA 2.2.0 (`/opt/rocm/include/rocwmma/rocwmma-version.hpp`,
  dpkg `rocwmma-dev 2.2.0.70200-43~22.04`): `/opt/rocm/include/rocwmma/rocwmma.hpp:75-354` —
  `fragment` `:203`, `fill_fragment` `:293`, `load_matrix_sync` `:303`/`:315`, `store_matrix_sync`
  `:325`/`:337`, `mma_sync` `:351`, `synchronize_workgroup` `:354`. No multi-problem or host-side
  entry point exists.
- Header-only, no runtime component: `dpkg -L rocwmma-dev` (include/, lib/cmake/, share/doc/LICENSE.md
  only). No samples ship on this box (`/opt/rocm/share` holds only the LICENSE), checked 2026-10-13.
- rocWMMA upstream: https://github.com/ROCm/rocWMMA
- MFMA/matrix-core background: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
