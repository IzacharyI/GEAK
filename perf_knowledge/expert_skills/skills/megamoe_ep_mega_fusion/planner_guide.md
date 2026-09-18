# MegaMoE EP fusion — Planner guide

Use this guide only after Analyze matches the pinned EP8/gfx950 profile and
validates the canonical Skill bundle.

## Planner job

Select exactly one direction: continue `full_persistent_pipeline` on its current
lane and exact working HEAD.

- Never open a duplicate lane.
- Never restart from the frozen baseline when a WIP lane exists.
- Never propose an alternate schedule or partial-fusion fallback.
- Never restate the full implementation contract in the Author prompt.
- Choose the earliest unresolved evidence boundary, not a broad rewrite.

## Blocker order

1. Host construction and argument allocation.
2. FlyDSL trace/JIT construction.
3. `path=MEGA` on all ranks and exactly two launches.
4. relL2 correctness at 128, 512, then 8192.
5. Same-instance graph replay and distributed liveness.
6. Paired 8192-uniform rank-max performance.

When Verify returns a traceback or hang classification, route only that concrete
failure to Author. Preserve all already-passing source and runtime evidence.

## Fixed target

- Launch 1: existing BF16→MXFP8 per-1x32 quantization.
- Launch 2: one persistent grid containing dispatch, GEMM1, GEMM2, weighted
  P2P publication, readiness, and Combine.
- No grid barrier between G1/G2 drain and Combine.
- 8192 profile: 256 CTAs, 8 waves, G2 `64x512x256`, chunk 16, skew 5/4,
  Combine P=1/U=4.
- Acceptance: relL2 < 0.10 and paired rank-max speedup >=1.03x.

The complete declarative contract belongs to independent Verify. Planner should
reference failed check IDs or GPU evidence paths instead of copying its contents.
