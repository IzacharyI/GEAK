Use ${SKILL_DIR} to optimize the MegaMoE V2 megakernel (EP8, 8× MI355X)
in `mode=mega`.

Goal: fuse the operator into one persistent kernel per rank
(`launches=2`: quant plus the persistent megakernel), running GEMM1, GEMM2
and combine with real overlap, and beat the frozen MegaMoE V2 baseline on
`8192_uniform` using rank-max `mega_e2e`.

Correctness UT (all batch sizes must have `relL2 < 0.10` and the fused arm
must report `path=MEGA`; a SCATTERED fallback does not count):

```bash
op_tests/multigpu_tests/test_mega_moe_v2.py \
  --network v4_pro \
  --bs-list 128,512,8192 \
  --iters 10 \
  --accuracy-max-bs 8192 \
  --rtol 0.10
```

Performance UT (the score is rank-max, the second `mega_e2e` value):

```bash
op_tests/multigpu_tests/bench_mega_moe_v2.py \
  --tokens 8192 \
  --route uniform \
  --iters 10 \
  --mega-only
```

Output `exp_root`: `${EXP_ROOT}`
