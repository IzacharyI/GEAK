---
key: filling the CUs on a decode-regime mxfp4 a4w4 two-stage MoE stage-1 GEMM by narrowing the N tile, where split-K overshoots into a second wave
type: lever
confidence: ★★
effect: BN 256->128 (num_n_blocks 2->4) took weighted 1.1294 -> 1.1502 as a standalone round-1 edit and anchored a 1.3205 stack (director 3-run median, span 2.5% -- ~1.32x); per-case the gain lands only where the grid was quantisation-starved -- at 1.69 waves the grid costs 17% more time per CTA than at 1.27 or 2.02 waves -- so decode buckets sitting mid-wave move and already-saturated ones do not.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: flydsl 0.3.4.1 / aiter@4faf8713 / ROCm 7.2
last_seen: 2026-10-12
name: narrow-bn-to-multiply-the-decode-grid-on-an-mxfp4-two-stage--moe-grouped-gemm-gfx950-decode
description: Decode mxfp4 MoE stage-1: halve BN to double num_n_blocks and land the grid on a wave boundary; a fill lever with no cross-CTA handoff.
keywords: ['moe-grouped-gemm', 'mxfp4', 'a4w4', 'tile-geometry', 'grid-occupancy', 'cu-underfill', 'wave-quantisation', 'split-k', 'decode', 'flydsl', 'gfx950']
kernels: ['gemm1_a4w4_port', 'gemm1_a4w4_bnflex']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: decode
language: flydsl
decision_refs: []
layer: learned
lifecycle: active
origin_kernels: ['glm52_mxfp4_fused_moe']
cost: L1
verified_on: 2026-10-12
levers: ['geom.tile-shape', 'geom.grid-fill']
---
# Narrow BN to multiply the decode grid on an mxfp4 two-stage MoE stage-1
- lever: When a grouped-GEMM decode grid lands mid-wave, halve the N tile instead of splitting K: num_n_blocks = n_out/BN multiplies the CTA count directly, with no partial buffer, no arrival counters and no XCD co-residency requirement.
- apply: Gate the narrow tile on the COMPUTED GRID, not on the token/expert tier -- the same tier produces different grids as expert count varies -- and make narrow-BN and split-K mutually exclusive so two fill levers never compound into a third wave.
- stack: Pairs with a per-tier dispatch that keeps the wide-BN variant in the same file, so each comparison stays an A/B on identical inputs; it does not compose with split-K.
- verify: Plot us/CTA against grid size across the tiers before and after: a real fill win flattens that curve. A bucket whose grid was already >= 2 full waves must stay on the null floor -- that is the in-run negative control.
- pitfall: The first run of a fresh workspace reads ~6% low because the new tile variant pays JIT compilation -> it looks like a failed reproduction -> clear __pycache__ and discard run 1.
The fill ceiling priced from a cost model fitted to the WIDE-tile implementation said ~7% weighted; the campaign delivered +16.9% (1.1294 -> 1.3205), so the ceiling was wrong by ~10pp -> a model fitted to one tile shape has no authority over another. (The error is 10pp, not 17 -- 17% is the total gain; do not quote the outcome as the prediction error.)
- caution: Upstream may accept tile widths the vendored copy hard-asserts away; confirm the assert is a simplification and not a layout contract (the scale-layout contract genuinely does bind both stages) before widening the whitelist.
- source: run wwd7t6625 / team_volck_A_20261012_082749, 2026-10-12, gfx950/MI355X, 3 rounds, director weighted 1.3205 correctness pass; re-verified 1.3212 and 1.3139 in a kernel-only interleaved A/B
