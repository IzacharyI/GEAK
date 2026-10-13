---
title: grouped_gemm_moe — numerics
kind: quant
operator: grouped_gemm_moe
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1, mxfp4]
regimes: [prefill, decode]
updated: 2026-10-13
sources:
  - ROCm/aiter@4faf87137cfb5e4d5538e010bb4fefa6c8c5df8d:op_tests/flydsl_tests/test_flydsl_grouped_gemm.py
  - ROCm/aiter@4faf87137cfb5e4d5538e010bb4fefa6c8c5df8d:op_tests/test_moe_2stage.py
  - https://github.com/ROCm/aiter
---

# grouped_gemm_moe — numerics

## TL;DR
Accumulation is dense GEMM's story (fp32 accumulate, cast at the epilogue). Everything *else* is not:
a grouped MoE adds three seams dense GEMM does not have — **per-expert quant scales**, the
**routing-weight multiply**, and **padded rows that must not contaminate output** — and its shipped
parity gate is a **cosine distance, not an elementwise bound**. Once activations are quantized,
elementwise closeness is the wrong instrument: no threshold both passes a correct kernel and catches a
real bug. Gate on `logits_diff < 0.01`; keep rel-L2 as an informational print.

## Accumulation & dtype contract
- In bf16/fp16 → **fp32 accumulate** (MFMA accumulators are fp32) → cast to out dtype at the epilogue.
- **Quantized operands** carry scales *per expert* on top of dense's per-tensor/per-token/per-block
  choice: fp8 (E4M3FNUZ on gfx942, OCP E4M3 on gfx950), int8, and mxfp4/mxfp6 with per-32-element
  **E8M0** block scales applied after the MFMA dot, before accumulation. The representational trap —
  arbitrary fp32 scales do **not** fold into an E8M0 operand — is dense's and applies unchanged here:
  see [`../dense_gemm/numerics.md`](../dense_gemm/numerics.md) and
  [`../scaled_quant_gemm/numerics.md`](../scaled_quant_gemm/numerics.md).
- **Routing weight multiply in fp32**, before the convert. A token routed to k experts sums k
  contributions; weighting after the cast changes the rounding of every one of them.
- **Padding must not contaminate.** Align&sort pads each expert's token block to the tile; padded rows
  must be masked or routed to a sink. A bug here corrupts *real* tokens' partial sums, and it shows up
  as a plausible-looking small error rather than as garbage.

## Parity bands
| change | comparable? | band |
|---|---|---|
| same-dtype solution/tile swap | yes — same math | cosine `logits_diff < 0.01`, and expect ≈0 |
| split-K / stream-K added | yes, with ULP drift | same gate; reduction order changed, not the math |
| fp8 / int8 quant | **no** | task-level eval; the kernel gate only says "not broken" |
| mxfp4 (per-32 E8M0) | **no** | task-level eval; naive MXFP4 loses real accuracy |
| atomic vs reduce combine | yes, but **non-deterministic** | see determinism below |

## The gate is COSINE, not elementwise — and the shipped assert is weaker than it looks
AITER's own MoE tests moved off the dense elementwise gate deliberately. The criterion is

> `logits_diff = ||x−y||² / (||x||² + ||y||²) < 0.01`

declared as `LOGITS_DIFF_TOL = 0.01` (`op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:90`, whose
own comment says rel-L2 "is kept as an informational print only"). **Why**: once activations are fp4/fp8
the reference and the kernel disagree elementwise *by construction*, so an elementwise bound tight
enough to catch a bug fails a correct kernel, and one loose enough to pass it catches nothing. A
direction-of-vector metric is insensitive to per-element quantization noise while still catching the
failure modes a grouped MoE actually has — a whole expert slice wrong, a scale row misread, padding
leaking in — because those move the vector, not one element.

⚠ **Read the two-stage assert before you trust it.** `op_tests/test_moe_2stage.py:570` asserts
`not (err != 0 and logits_diff > 0.01)` — a conjunction. It fails only when the elementwise check
*also* fires, so a result that is cosine-wrong but elementwise-clean passes. Treat the shipped assert
as a regression tripwire, not as your gate; run `logits_diff` on its own.

⚠ **And it is AITER's gate for AITER's kernels.** An authored kernel is judged by the task oracle's
tolerance, not by this card. Use the card to tell known quantization drift from a bug and to pick
*which* comparison to run — never to overrule a task that demands something stricter. (The GLM-5.2
MXFP4 oracle, for instance, sets `tol: 0.02` over 3 random draws and gates correctness, parity and
graph separately.)

## Tie-break / determinism
- **The atomic combine is not run-to-run deterministic.** The stage-2 epilogue weights in f32, converts
  to bf16, then issues a packed-bf16×2 `atomic fadd` (monotonic, agent scope). Atomic arrival order
  varies between runs, so fp32→bf16 summation order varies, so the output varies in the last ULPs.
  If you need bit-reproducibility, select the **reduce** epilogue (store unsummed partials, sum in a
  second kernel) and pay the extra launch — it is a kernel choice, not a flag.
- **Split-K / stream-K** change the fp32 reduction order the same way dense GEMM does; pin the
  configuration if exact reproducibility across runs is required.
- **Routing tie-breaks are upstream**, in top-k, not in this GEMM:
  [`../moe_routing_topk/numerics.md`](../moe_routing_topk/numerics.md). A changed top-k tie-break
  reroutes a token and changes the output far more than any arithmetic choice here.
- **Cross-stage scale layout is a contract, and a mismatch is silent.** On the two-stage mxfp4 path
  stage-1's fp4 output and its E8M0 scales are stage-2's A operand; the layout is *derived* from the
  tile, not tuned. If the two stages disagree, stage 2 reads a real scale from the wrong row — parity
  degrades with no out-of-bounds signal. See [`./tuning.md`](./tuning.md).

## How to gate
1. **Oracle**: per-expert dense reference `X_e @ W_e` in fp32, gathered back and combined with fp32
   gate weights. Build it from the *same* routing decision — never re-run top-k for the reference, or
   you are comparing two different routings (see [`./overview.md`](./overview.md)).
2. **Metric**: `logits_diff < 0.01`. Print rel-L2 and max-abs beside it for diagnosis only.
3. **Negative control**: run the unmodified kernel against the same oracle first and record the band.
   A grouped MoE with an atomic combine does not return 0, and not knowing its floor is how a real
   regression gets accepted.
4. **Quant changes**: the kernel gate is necessary, not sufficient — add a task eval (accuracy on the
   serving workload) and accept only on no regression.
5. **Representative inputs**: all-zero or uniform tensors under-detect quant error *and* collapse the
   routing distribution, so every expert sees the same work and the padding bug above stays hidden.

## Sources
- Cosine gate, declared: `ROCm/aiter@4faf8713:op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:88-90`
  (`LOGITS_DIFF_TOL = 0.01`, with the comment that rel-L2 is informational only).
- The weaker two-stage assert: `ROCm/aiter@4faf8713:op_tests/test_moe_2stage.py:568-571`
  (`assert not (err != 0 and logits_diff > 0.01)`).
- Weight-then-convert-then-atomic ordering, and the reduce alternative: corpus decision cards
  `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic`,
  `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` (both `evidence_level: source_observed`,
  pinned at `4faf8713`) — [`../../corpus/gemm_decisions.md`](../../corpus/gemm_decisions.md).
- Cross-stage scale-layout contract: corpus decision card
  `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract`.
- Gate choice and its rationale: corpus decision card
  `grouped-moe-parity-gate-is-cosine-not-elementwise`.
- AITER repo (scaled grouped-GEMM paths): https://github.com/ROCm/aiter
