---
title: grouped_gemm_moe on Gluon — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: gluon
gens: [gfx1250, gfx950, gfx942]
dtypes: [fp4_e2m1, mxfp4, fp8_e4m3, mxfp8, bf16, int8]
regimes: [prefill, decode]
status: sota
updated: 2026-10-11
sources:
  - ROCm/aiter@4faf8713:aiter/ops/triton/_gluon_kernels/gfx1250/moe/
  - ROCm/aiter@4faf8713:aiter/ops/triton/_gluon_kernels/gfx950/moe/moe_op_gemm_a16w4.py
  - ROCm/aiter@4faf8713:aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py
  - ROCm/aiter@4faf8713:aiter/ops/triton/moe/moe_op_gemm_a16w4.py
  - https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
---

# grouped_gemm_moe × Gluon

## TL;DR
> Gluon is the **default MoE GEMM backend on gfx1250** (a4w4, a8w4, a16w4 — chosen with no env var and no
> opt-in) and an opt-in alternative on gfx950 (a16w4) / gfx942 (int8 SmoothQuant). It is the same Triton
> front-end with the *scheduling decisions moved into the kernel source*: explicit WMMA/MFMA layouts,
> hand-allocated shared memory, a hand-written multi-buffer pipeline, and `scaled_upcast` + `wmma` as two
> separately scheduled ops instead of one `tl.dot_scaled`. Read this card before assuming "Triton MoE" —
> on gfx1250 the kernel that actually runs is the Gluon one.

## SOTA implementation
Six gluon MoE kernels exist, under `aiter/ops/triton/_gluon_kernels/<arch>/moe/`:

| arch | file | scheme | kernels |
|---|---|---|---|
| gfx1250 | `moe_op_gemm_a4w4.py` | MXFP4 × MXFP4 | `_moe_gemm_a4w4_prefill:551`, `_moe_gemm_a4w4_decode:1402` |
| gfx1250 | `moe_op_gemm_a8w4.py` | fp8e4m3 **or** MXFP8 act × MXFP4 w | `_decode_persistent:135`, `_decode:897`, `_prefill:1922` |
| gfx1250 | `moe_op_gemm_a16w4.py` | bf16 act × MXFP4 w | `_moe_gemm_a16w4:104` |
| gfx1250 | `reduce.py` | grouped combine | `reduce_grouped_gluon:16` |
| gfx950 | `moe_op_gemm_a16w4.py` | bf16 × MXFP4 | `_swizzle:80`, `_swizzle_pipelined:430` (**dead**, see pitfalls) |
| gfx942 | `moe_op_gemm_int8_smoothquant.py` | int8 per-token × per-channel | `_moe_gemm_int8_smoothquant:28` |

gfx950 and gfx1250 a16w4 are **structurally different kernels**, not one source: gfx1250 uses TDM
descriptors + `gl.amd.gfx1250.wmma`, gfx950 uses `gl.amd.cdna4.async_copy.buffer_load_to_shared` +
`gl.amd.cdna4.mfma`. There is **no a4w4/a8w4 gluon kernel outside gfx1250**, and **no gluon MoE kernel for
a8w8 or a8w8_blockscale on any arch**.

## Config space / knobs
No env var gates any of this. Selection is an arch check plus an optional `backend=` kwarg, and the three
wrappers disagree on the failure mode — that difference is the thing to know:

| wrapper | default | explicit `backend="gluon"` on wrong arch |
|---|---|---|
| `moe/moe_op_gemm_a16w4.py:391-403` | gluon if arch ∈ `("gfx1250","gfx950")` (`:30`) | **silently falls back to triton** + one-shot `lru_cache` warning (`:33-38`) |
| `moe/moe_op_gemm_a8w4.py:397-404` | gluon iff gfx1250 | **raises** |
| `moe/moe_op_gemm_a4w4.py:318-324` | gluon iff gfx1250 | **raises** |
| `moe/moe_op_gemm_int8_smoothquant.py:229-256` | pure heuristic, no `backend=` arg | n/a |

a16w4 has a second, runtime fallback: `K % 256 != 0 or swizzle_mx_scale is None → backend = "triton"`
(`:447-460`). `expert_map` is triton-only (`:498-501`); `preshuffled` (a8w4) and `out_mx_quant` (a4w4) are
gluon-only.

Representative constexpr set (`_gluon_kernels/gfx1250/moe/moe_op_gemm_a16w4.py:136-156`):
`BLOCK_M/N/K`, `GROUP_M`, `XCD_SWIZZLE`, `NUM_BUFFERS`, `SWIZZLE_MX_SCALE`, `EVEN_K`, `SPLIT_K`,
`W_CACHE_MODIFIER`, `num_warps`, `APPLY_SWIGLU`, `ACTIVATION_REDUCTION_N`, `N_EXPTS_ACT`, plus in-body
`MX_PACK_DIVISOR=32` (`:158`) and `NUM_TDM_OPS=3` (`:159`, the async_wait accounting unit).
Arch-specific deltas: gfx950 has no `EVEN_K` but adds `MASK_K_LIMIT`/`NUM_FULL_K` (`:128-129`) and
`TILE_PER_WARP_0/1` (`:133-134`); a8w4 adds `X_SCALE_TDM`, `PRESHUFFLED`, `CLAMP_BOUNDS`, `HAS_MX_OUT`,
`EP_SCATTER` and a dynamic `NUM_TDM_OPS = 4 if (is_x_microscaled and X_SCALE_TDM) else 3` (`:968-971`);
gfx942 int8 has `UNROLL_TIMES` from a `@triton.heuristics` (`:23-27`) and **no** `NUM_BUFFERS`/`SPLIT_K`.
`SPLIT_K` exists only in a16w4.

Tuned tables live at `configs/<arch>/<backend>/moe/<dtype>/DEFAULT.json`, resolved by
`utils/config_utils.py:126-166` — **no cross-backend search**, so a family tuned only for triton is never
silently borrowed. Present: `gfx1250/gluon/moe/a4w4` (379 entries), `gfx1250/gluon/moe/a8w4` (135).
Key format `bm{block_m}_n{n}_k{k}_{bucket}` with fallback `bm{block_m}_any`.

## Numerics / parity
fp32 accumulate everywhere except **gfx942 int8, which accumulates in int32**
(`moe_op_gemm_int8_smoothquant.py:216-218`) and folds a rank-1 per-token × per-channel scale after the K
loop (`:251-255`). MX block size is 32 throughout; scales are e8m0, weights e2m1 nibble-packed along K.
Scales are consumed **inside** the matrix instruction or by a dedicated upcast, never as a post-loop
multiply — `wmma_scaled(x, x_scales, "e4m3", w, w_scales, "e2m1", acc)` for a8w4 (`:704-710`; with plain
fp8 activations the x-scale operand is literally `0`), `scaled_upcast(...)` + `wmma` for a16w4 (`:464,:467`).
SwiGLU is fused in the epilogue of all four gfx1250/gfx950 GEMM kernels via the shared
`_triton_kernels/moe/activations.py:41-59` (`ACTIVATION_REDUCTION_N=2`); a4w4/a8w4 additionally quantize
the output (`_mxfp4_quant_op` / `_mxfp8_quant_op`).

Test gate is `op_tests/triton_tests/moe/moe_test_utils.py:10-55` — note it is **not** the dense elementwise
gate: 1-byte dtypes compare byte-exact (`:11-16`), otherwise error is normalized by the reference RMS
(`rel_err = |refn-trin| / max(ref_rms, |refn|)`, `:51`). Defaults 2e-2 / 4e-3; a8w4 loosens to 4e-1 / 4e-2
(1.5e-1 / 3e-2 when activations are MXFP8), a16w4 uses 4e-1 / 4e-2.

## Integration (rebind seam)
Call the public wrapper and pass `backend="gluon"` — `aiter.ops.triton.moe.moe_op_gemm_{a4w4,a8w4,a16w4}`.
Benchmarks expose it as a flag (`bench_moe_gemm_a4w4_cudagraph.py:952`, `..._a8w4_cudagraph.py:694`).
There is no env-overlay seam and no `.so` to rebuild.

## Pitfalls & anti-patterns
- **a16w4 fails open, a4w4/a8w4 fail closed.** Asking for gluon a16w4 on gfx942 gives you a *triton* kernel
  and one cached warning — a benchmark that reports "gluon" there is measuring triton.
- **`_moe_gemm_a16w4_swizzle_pipelined` (gfx950) is unreachable** — `use_pipelined_gluon = False` is
  hardcoded at `moe/moe_op_gemm_a16w4.py:443`. Don't tune it; it never runs.
- **The CGA / multi-CTA path is inert.** `num_ctas` is pinned to 1 in a4w4 (`:232`) and a8w4 (`:280`),
  so `make_cga_layout` and `cluster.arrive()/wait()` are present but never exercised.
- **The gfx942 int8 gluon kernel has zero test coverage.** Its gate needs `K<=192 and M>=4096 and N>=1024`
  but no shape in `test_moe_gemm_int8_smoothquant.py` (`:104-124`, `k>=256`, `m<=2048`) or the default
  benchmark can satisfy it. It also declares `APPLY_ACTIVATION`/`alpha`/`limit` and **never uses them** —
  consistent with its own `not apply_activation` gate.
- **No A16W4 gluon JSON table on any arch**; configs come from hardcoded Python heuristics
  (`get_kernel_config_gluon_gfx950:234`, `..._gfx1250:298`). `configs/gfx950/gluon/moe/` is a `.gitkeep`.
- **`is_gluon_avail()` (`utils/_triton/arch_info.py:30-31`) is dead for MoE** — no wrapper calls it; a16w4
  reimplements the check locally and a4w4/a8w4 hardcode `== "gfx1250"`. Reading it will mislead you.
- **No warp specialization anywhere** — no `warp_specialize`/`setprio`/`sched_barrier`/`named_barrier` in
  any of the six files. The pipelining is multi-buffering + explicit async waits only; don't look for a
  ping-pong schedule that isn't there.
- Gluon's `EVEN_K` does **not** produce a masked tail loop like the Triton kernel's (`:327-350`); it only
  sets `CLAMP_BOUNDS` (`:172`), and the TDM descriptor does the clamping.

## How to verify (worked example)
```bash
# 1. which backend actually ran (the a16w4 silent-fallback trap)
python -c "from aiter.ops.triton.utils._triton import arch_info; print(arch_info.get_arch())"
pytest op_tests/triton_tests/moe/test_moe_gemm_a4w4.py -k gluon   # parametrized ["gluon","triton"]
# 2. A/B the two backends on one shape
python op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py --backend gluon
python op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4_cudagraph.py --backend triton
# 3. confirm a tuned row exists before blaming the kernel
jq 'keys[]' aiter/configs/gfx1250/gluon/moe/a8w4/DEFAULT.json | grep bm16_n7168
```

## Alternatives / cross-links
[[operators/grouped_gemm_moe/backends/triton]] (same wrappers, autoscheduled) ·
[[operators/grouped_gemm_moe/backends/flydsl]] (the fp4 competitor on gfx950/1250) ·
[[operators/grouped_gemm_moe/backends/asm]] (gfx942/950 only, stage-1 only) ·
[[operators/dense_gemm/backends/gluon]] · [[operators/grouped_gemm_moe/numerics]] ·
[[operators/moe_dispatch_combine/overview]] (the gluon `reduce_grouped_gluon` combine).

## Sources
- Kernel inventory, layouts, pipeline, epilogues: `aiter/ops/triton/_gluon_kernels/{gfx1250,gfx950,gfx942}/moe/` (`ROCm/aiter@4faf8713`).
- Backend gates, runtime fallbacks, config resolution: `aiter/ops/triton/moe/moe_op_gemm_{a4w4,a8w4,a16w4,int8_smoothquant}.py`, `aiter/ops/triton/utils/config_utils.py` (same commit).
- Parity gate: `op_tests/triton_tests/moe/moe_test_utils.py` (same commit).
- Gluon authoring model (explicit layout/pipeline/MFMA scheduling): https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
