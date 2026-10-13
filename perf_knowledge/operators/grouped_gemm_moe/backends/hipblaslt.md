---
title: grouped_gemm_moe on hipBLASLt — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: hipblaslt
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3]
regimes: [prefill, decode]
status: competitive
updated: 2026-10-13
sources:
  - https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
  - https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
  - ROCm/aiter@a6bb4993:aiter/tuned_gemm.py
---

# grouped_gemm_moe × hipBLASLt

## TL;DR (one-line decision)
> hipBLASLt is the **only vendor library on this box with a first-class grouped-GEMM API**
> (`hipblaslt_ext::GroupedGemm`, with a device-resident argument path that lets per-expert shapes
> change without a host round-trip), and it is the right choice when you need many differently-shaped
> GEMMs in one launch *outside* an MoE layer. Inside an MoE layer it is **not** the serving path:
> it solves only the GEMM, leaving sort/pad, activation and routed-weight combine as separate
> launches, and the stacks that serve MoE on AMD call aiter, which bypasses hipBLASLt entirely.
> Use it as a **grouped-GEMM baseline and a fallback**, not as the MoE kernel — see [aiter.md](aiter.md).

## SOTA implementation(s)

| impl | source | gens / dtypes / shapes | measured perf | when it's best |
|---|---|---|---|---|
| `hipblaslt_ext::GroupedGemm` (host args) | `/opt/rocm/include/hipblaslt/hipblaslt-ext.hpp:786`, ROCm 7.2.0 / hipblaslt 1.2.1 | gfx908–gfx950; bf16/fp16/fp8; per-entry `m,n,k,batch_count` | not measured here (see Pitfalls — no on-box sample to run) | a grouped GEMM whose group shapes are known host-side and change rarely |
| same + `run(deviceUserArgs, stream)` | same header `:317` (`UserArguments`), `:987` (`getDefaultValueForDeviceUserArguments`), `:1015` | as above | not measured here | shapes change every step (the MoE case) — avoids re-running `setProblem`/algo selection per call |
| aiter fused MoE | [aiter.md](aiter.md) | gfx942/gfx950, incl. fp8 block-scale and A4W4 | see that card | **the serving path**: sort+GEMM+act+combine in one kernel |

The device-argument path is the one that matters for MoE-shaped work and it is really shipped, not
just declared: `nm -D /opt/rocm/lib/libhipblaslt.so` exports
`_ZN13hipblaslt_ext11GroupedGemm37getDefaultValueForDeviceUserArgumentsEPv`. You fill an array of the
packed `UserArguments` POD (per-entry `m/n/batch/k`, `a/b/c/d` pointers, strides, alpha/beta, scales,
bias, activation), upload it once, and vary its contents between launches.

## Config space / knobs (backend-specific)
- **Problem list** — `setProblem(m[], n[], k[], batch_count[], epilogue[], inputs[])`: the per-group
  shapes. A second overload adds `lda/ldb/ldc/ldd` and `strideA..D` when the per-expert weight slices
  are not contiguous.
- **Algorithm** — `algoGetHeuristic` on the `GemmInstance` base, then `initialize`. Selection is per
  problem *list*, so a changed list means re-selection unless you stay on the device-args path.
- **Epilogue** — `GemmEpilogue` per entry (bias, activation, aux). SwiGLU's gate⊙up is **not** an
  epilogue here; it stays a separate kernel.
- **Offline tuning** — the usual hipBLASLt tuning flow applies to the solutions it picks from, but see
  the first pitfall before spending time on it.

## Numerics / parity
- fp32 accumulate; same Tensile/CK solution pool as dense hipBLASLt, so parity vs. a dense reference
  per group is the dense card's story ([../../dense_gemm/backends/hipblaslt.md](../../dense_gemm/backends/hipblaslt.md)).
- Per-entry `scaleA/scaleB/scaleC/scaleD/scaleAlphaVec` exist in `UserArguments`, so fp8 per-tensor
  scaling is expressible per expert; MX block scaling at the granularity MoE A4W4 uses is not.

## Integration (how it gets used at serving time)
- C++ only — there is no Python binding for `GroupedGemm` in the installed stack, so wiring it into a
  PyTorch MoE layer means writing an extension. Compare: aiter is already on the sglang/vLLM path.
- **Verify it engages** by checking your own extension's call site; there is no library-side log
  marker equivalent to aiter's `is tuned on cu_num`.

## Pitfalls & anti-patterns
- ⚠ **Tuning hipBLASLt does nothing for a serving MoE layer.** aiter bypasses the PyTorch BLAS
  dispatch and calls by solution index; the MoE path does not reach hipBLASLt at all. See
  [`../../../backends/aiter/tuned_gemm.md`](../../../backends/aiter/tuned_gemm.md).
- ⚠ **It is a GEMM, not an MoE kernel.** You still own `moe_align_block_size`-style sort/pad, the
  act-and-mul between the two GEMMs, and the routed-weight combine — three extra launches and two
  extra round-trips through HBM that the fused path does not pay.
- ⚠ **No sample ships on this box.** `/opt/rocm/share/doc/hipblaslt/` holds only `LICENSE.md`, and no
  hipBLASLt source checkout exists here, so the API above is read from the installed header and the
  exported symbol table. Budget time for first-use trial and error.
- ⚠ `GroupedGemm` is move-only (copy ctor and copy assignment are deleted) — holding one in a
  container needs `std::move`.

## How to verify (bench + oracle)
- Oracle: loop `torch.matmul` per expert slice and compare; tolerance per the dense card.
- Bench: time `run(deviceUserArgs, stream)` against the same group list driven through aiter's fused
  path, and against a per-expert loop of dense `hipblasLtMatmul`. The number that decides anything is
  the whole-MoE-layer time, not the GEMM time — the sort/act/combine launches belong in the
  comparison, because the fused alternative absorbs them.

## Alternatives / cross-links
[`./aiter.md`](./aiter.md) (serving path) · [`./flydsl.md`](./flydsl.md) · [`./triton.md`](./triton.md) ·
[`./ck.md`](./ck.md) · [`./asm.md`](./asm.md) · [`./hip.md`](./hip.md) ·
[`../../fused_moe_grouped_gemm/overview.md`](../../fused_moe_grouped_gemm/overview.md) (the fused MoE specialization) ·
[`../../dense_gemm/backends/hipblaslt.md`](../../dense_gemm/backends/hipblaslt.md)

## Sources
- On-box header, ROCm 7.2.0 (`/opt/rocm/.info/version`), hipblaslt 1.2.1 (`hipblaslt-version.h`,
  tweak `5b515cf1bc`): `/opt/rocm/include/hipblaslt/hipblaslt-ext.hpp` — `GemmInstance` base `:359`,
  `Gemm` `:576`, `GroupedGemm` `:786`, `UserArguments` `:317`,
  `getDefaultValueForDeviceUserArguments` `:987`, `run(void*, hipStream_t)` `:1015`.
- Shipped, not just declared: `nm -D /opt/rocm/lib/libhipblaslt.so` exports
  `_ZN13hipblaslt_ext11GroupedGemm37getDefaultValueForDeviceUserArgumentsEPv`
  (`libhipblaslt.so.1.2.70200`).
- No samples and no source checkout on this box: `/opt/rocm/share/doc/hipblaslt/` contains only
  `LICENSE.md`; no hipBLASLt git checkout exists (searched 2026-10-13).
- aiter bypasses the hipBLASLt tuning override on the serving path:
  `ROCm/aiter@a6bb4993:aiter/tuned_gemm.py` and
  [`../../../backends/aiter/tuned_gemm.md`](../../../backends/aiter/tuned_gemm.md).
- hipBLASLt docs: https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/
- Offline tuning flow: https://rocm.blogs.amd.com/software-tools-optimization/hipblaslt-offline-tuning-part1/README.html
