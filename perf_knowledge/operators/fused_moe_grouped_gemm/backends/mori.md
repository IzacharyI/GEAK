---
title: fused_moe_grouped_gemm on mori — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: mori
gens: [gfx942, gfx950]
dtypes: []
regimes: [prefill, decode]
status: na
updated: 2026-10-13
sources:
  - ROCm/mori@96ffa169:README.md
  - https://github.com/ROCm/mori
  - https://www.lmsys.org/blog/2026-05-28-mori/
---

# fused_moe_grouped_gemm × mori

## TL;DR (status: na — mori moves the tokens, it does not multiply them)
> MORI is a **communication** framework: MORI-EP (MoE dispatch & combine all-to-all), MORI-IO (p2p
> KVCache transfer), MORI-CCL (collectives). It contains **zero GEMM kernels** — an exhaustive search
> for `*gemm*` / `*matmul*` across the checkout returns no files, and the only mentions of "GEMM" in
> the tree are a comment about what the *caller* may overlap, and a benchmark that **imports aiter**
> to supply the GEMM half. So "fused MoE grouped GEMM on mori" does not exist as a thing to select.
> mori is not an alternative to [`aiter.md`](aiter.md) — it is the layer that **pairs with** it:
> mori dispatches tokens to the expert's rank, aiter's fused kernel does the grouped GEMM there,
> mori combines the results back. See [`../../moe_dispatch_combine/backends/mori.md`](../../moe_dispatch_combine/backends/mori.md),
> which is where mori's real card for MoE lives.

## Why na (and what exactly was checked)
- **Scope, from mori's own README** (`ROCm/mori@96ffa169:README.md:9-23`): *"MORI (Modular RDMA
  Interface) is a bottom-up, modular, and composable framework for building high-performance
  communication applications … MORI-EP (MoE dispatch & combine kernels), MORI-IO (p2p communication
  for KVCache transfer), and MORI-CCL (collective communication)."* Communication, stated outright.
- **Source layout has no compute directory**: `src/{ops/{dispatch_combine,kernels}, io/{rdma,xgmi,
  kernels}, collective, shmem, application, umbp, pybind}` and
  `include/mori/{core,ops,io,collective,shmem,application}`. No `gemm/`, no `matmul/`, no MoE-compute
  directory anywhere.
- **Exhaustive filename search**: `find /sgl-workspace/mori -iname '*gemm*' -o -iname '*matmul*'` →
  **zero files**.
- **Every textual "gemm" hit is a non-kernel**, checked one by one:
  - `examples/ops/dispatch_combine/test_low_latency.py:424` — `simulated_gemm_x = packed_recv_x.clone()`,
    a literal stand-in used to model compute/communication overlap.
  - `src/collective/core/twoshot_allreduce_sdma_class.cpp:295` — `// The wait is deferred to
    wait_async so the user can run GEMM on CU`, a comment about the caller.
  - `tests/python/ccl/test_allgather_overlap.py:41,52,130-131` — imports **aiter** for the GEMM half
    of an overlap benchmark, and warns `"aiter not available, gemm timing will be disabled"`.
- **Python surface confirms it**: `python/mori/ops/` exposes only `dispatch_combine.py` and
  `local_expert_count.py` — routing bookkeeping, no GEMM op.

## What to do instead
- **The grouped GEMM**: [`aiter.md`](aiter.md) (serving), [`asm.md`](asm.md) / [`ck.md`](ck.md) (the
  two stages), [`flydsl.md`](flydsl.md) / [`triton.md`](triton.md) (authorable).
- **The thing mori is actually for**: [`../../moe_dispatch_combine/backends/mori.md`](../../moe_dispatch_combine/backends/mori.md).
- **Together**: aiter's fused MoE integrates mori's in-kernel EP dispatch/combine, which is the
  configuration that matters at EP scale — that integration is described from the GEMM side in
  [`aiter.md`](aiter.md) and from the communication side in the dispatch/combine card. Choosing
  "mori" for this operator is a category error; choosing it *alongside* aiter is the real decision.

## Sources
- Scope statement: `ROCm/mori@96ffa169:README.md:9-23`.
- Routing-only Python surface: `ROCm/mori@96ffa169:python/mori/ops/dispatch_combine.py`,
  `ROCm/mori@96ffa169:python/mori/ops/local_expert_count.py`.
- No GEMM kernel anywhere: filename and content search over `/sgl-workspace/mori` @ `96ffa169`
  (2026-10-13) — `find -iname '*gemm*' -o -iname '*matmul*'` returns nothing; the three textual hits
  are enumerated above (`examples/ops/dispatch_combine/test_low_latency.py:424`,
  `src/collective/core/twoshot_allreduce_sdma_class.cpp:295`,
  `tests/python/ccl/test_allgather_overlap.py:41`).
- Upstream: https://github.com/ROCm/mori
- MORI-EP in-kernel dispatch/combine fusion with the MoE path: https://www.lmsys.org/blog/2026-05-28-mori/
