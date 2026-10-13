---
title: grouped_gemm_moe on HipKittens — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: hipkittens
gens: [gfx942, gfx950]
dtypes: [bf16, fp16]
regimes: [prefill]
status: na
updated: 2026-10-13
sources:
  - https://arxiv.org/abs/2511.08083
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk
  - https://github.com/HazyResearch/HipKittens
---

# grouped_gemm_moe × HipKittens

## TL;DR (status: na — primitives, no grouping layer, and nothing on this box to check)
HipKittens is a **tile-primitive C++ library** (register/shared tiles + `mma`/`load`/`store`), not a
kernel catalog: it gives you the inner loop, and the grouped-GEMM problem is everything *around* the
inner loop — tile→expert mapping from an offset table, per-group M padding, one launch over variable
M. HK's published kernel set is dense GEMM, attention fwd/bwd and memory-bound kernels; grouped/MoE
GEMM is not among them, and **no HipKittens checkout exists on this box**, so no grouped path could
be verified even in principle here. **Use [`./aiter.md`](./aiter.md) for serving and
[`./flydsl.md`](./flydsl.md) / [`./triton.md`](./triton.md) when you want to author and tune one.**

## Why na (and what exactly was and was not checked)
- **Not checked upstream.** This box carries no HipKittens source (searched the filesystem
  2026-10-13: no checkout under `/sgl-workspace`, `/root` or `/opt`, no installed package). The only
  HK material here is this tree's own reading notes. So this card does **not** assert that upstream
  lacks a grouped kernel — it asserts that none is documented in the paper's benchmarked set and that
  nothing on this machine can confirm or deny one. Check
  `github.com/HazyResearch/HipKittens` directly before concluding otherwise.
- **Structurally the same gap as rocWMMA.** A tile-primitive library is orthogonal to grouping: it
  would speed up the MFMA inner loop of a grouped kernel you wrote yourself, and change nothing about
  the scheduling levers this operator is tuned on (see [`../tuning.md`](../tuning.md)).
- **Research artifact, no support contract.** HK is arXiv 2511.08083v1 (Nov 2025) with unstable APIs
  and a "pin a commit" caveat — see [`../../../languages/hipkittens/overview.md`](../../../languages/hipkittens/overview.md).
  That is tolerable for a dense GEMM you can A/B in an afternoon
  ([`../../dense_gemm/backends/hipkittens.md`](../../dense_gemm/backends/hipkittens.md), status
  `sota` on author-reported numbers) and a poor trade for an MoE path where the fused alternative
  already absorbs sort, activation and combine.

## What to do instead
- **Serving / default**: [`./aiter.md`](./aiter.md).
- **Authored and tunable**: [`./flydsl.md`](./flydsl.md), [`./triton.md`](./triton.md),
  [`./ck.md`](./ck.md), [`./gluon.md`](./gluon.md).
- **If you want HK's scheduling ideas anyway**, take them as ideas, not as a backend: the 8-wave
  ping-pong and XCD/L2-aware grid from the dense card transfer to a grouped kernel you write in
  FlyDSL or CK. That is a technique import, not a backend selection.

## Sources
- No HipKittens checkout or installed package on this box (filesystem search, 2026-10-13); the only
  HK material present is this tree's notes at
  [`../../../languages/hipkittens/overview.md`](../../../languages/hipkittens/overview.md), which
  itself states HK is "a research artifact ... not a shipping production dependency" and names the
  benchmarked set as GEMM, attention fwd/bwd and memory-bound kernels.
- HipKittens paper (kernel set, tile primitives, AMD-native scheduling): https://arxiv.org/abs/2511.08083
- Author blog: https://hazyresearch.stanford.edu/blog/2025-11-09-hk
- Upstream repo, to check directly rather than from this box: https://github.com/HazyResearch/HipKittens
