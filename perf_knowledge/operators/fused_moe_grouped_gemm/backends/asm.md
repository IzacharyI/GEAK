---
title: fused_moe_grouped_gemm on asm — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: asm
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, fp8_e4m3, int8, fp4_e2m1]
regimes: [prefill, decode]
status: sota
updated: 2026-10-13
sources:
  - ROCm/aiter@a6bb49937:csrc/py_itfs_cu/asm_fmoe.cu
  - ROCm/aiter@a6bb49937:csrc/py_itfs_cu/asm_moe_2stage.cu
  - ROCm/aiter@a6bb49937:aiter/ops/moe_op.py
  - ROCm/aiter@a6bb49937:aiter/fused_moe.py
  - ROCm/aiter@a6bb49937:csrc/cpp_itfs/moe/asm_moe.py
---

# fused_moe_grouped_gemm × asm

## TL;DR (status: sota — fastest path, and the one you cannot edit)
> Hand-tuned MFMA assembly is where aiter's fused MoE actually spends its time, in two different
> shapes: the **1-stage mega-kernel** (`fmoe*` — gate+up GEMM, activation, down GEMM and routed-weight
> reduce in a *single* launch over sorted-expert tiles) and **stage-1 of the 2-stage path**
> (`moe_stage1_g1u1`). The decisive constraint for any optimization campaign: aiter vendors only
> compiled **`.co` HSA code objects** — there is **not one `.s` file** in either on-box checkout — so
> this backend is a *catalog you select from*, never a seam you author. And the catalog is
> **asymmetric**: 190 stage-1 objects on gfx942 / 186 on gfx950, and **zero** stage-2 objects on
> either. Stage-2 (down-proj) always lands on CK. Own stage-1 here, own stage-2 in
> [`ck.md`](ck.md) — or move both into [`flydsl.md`](flydsl.md) when you need source-level control.

## SOTA implementation(s)
| impl | source | gens/dtypes | shape | when best |
|---|---|---|---|---|
| `fmoe` / `fmoe_g1u1` / `fmoe_g1u1_tkw1` / `fmoe_int8_g1u0` | `aiter/ops/moe_op.py:62`, `:95`, `:115`, `:76` | gfx942/950; bf16, fp16, int8, fp8 | **1 launch**: gate+up + act + down + reduce | small/medium token counts where one launch beats two + an HBM round-trip |
| `fmoe_fp8_blockscale_g1u1` | `aiter/ops/moe_op.py:173` | gfx942/950; fp8 128×128 block scale, bf16 out | 1 launch, quant inside the kernel | the fp8 block-scale serving path (DeepSeek-style) |
| `moe_stage1_g1u1` | `aiter/ops/moe_op.py:196`, C++ entry `csrc/py_itfs_cu/asm_moe_2stage.cu:149` | gfx942/950; fp8/int8 per-token + blockscale, A4W4 | **stage-1 only** of the 2-stage path | large M, where stage-2 wants its own CK tiling |

All of them are launched from one C++ host class, `FMoeKernel`
(`csrc/py_itfs_cu/asm_fmoe.cu:71`, `launch_kernel` at `:97`), which takes
`sorted_token_ids` / `sorted_weights` / `sorted_expert_ids` / `num_valid_ids` — i.e. it consumes the
align&sort layout directly, which is what makes it a *grouped* GEMM rather than a batched one.

The Python entry points are ctypes bindings (`@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")`),
linked from exactly two translation units — `asm_fmoe.cu` + `asm_moe_2stage.cu` → `module_moe_fmoe_asm.so`.

### Who chooses the kernel
You do not. Two layers pick for you:
- **Dispatch**: `fused_moe_()` reads `metadata.run_1stage` (`aiter/fused_moe.py:310`) and, if set, calls
  `fused_moe_1stage()` (`:367`), whose quant-signature table routes to one of the `fmoe*` APIs. The
  2-stage path goes through `asm_stage1()` (`:1389` → `aiter.moe_stage1_g1u1` at `:1427`).
- **Object selection**: `select_tile()` / `select_hsaco()` (`csrc/cpp_itfs/moe/asm_moe.py:42`, `:132`)
  map (dtype, quant, inter_dim) to a `.co` filename such as
  `fmoe/silu/fmoe_fp8_g1u1_subGU_{128..512}.co`.

So "use asm" is not a call you write — it is a consequence of the quant signature and the tuned DB.
See [`aiter.md`](aiter.md) for the DB keying.

## Config space / knobs (what is actually tunable)
| knob | where | effect |
|---|---|---|
| `subGU` tile (128/256/320/512) | `.co` filename, chosen by `select_tile()` | the only "tiling" choice, and it is a *selection*, not a parameter |
| `block_size_M` | call arg + `moe_sorting` | grouped tile M; **must equal** the align&sort pad |
| activation | separate `.co` subdirectory (`fmoe/silu/`, `fmoe/gelu/`) | silu vs gelu are different binaries |
| `doweight_stage1` | call arg | selects `fmoe_g1u1_tkw1` (routed weight in stage-1) vs combine-time |
| smooth-quant variant | `_smf` / `_multix` in the object name | int8/fp8 smooth-quant flavors |

That table is the whole config space. There is no register-allocation knob, no scheduling knob, no
LDS-budget knob — those decisions are baked into the binary.

## Numerics / parity
fp32 accumulate inside MFMA. Coverage by quant: bf16/fp16 unquantized, int8 g1u0 and g1u1
(± smooth-quant), fp8 g1u1 (plain / smooth-quant / 128×128 block-scale), and int4-weight+fp8
(`fmoe_int4fp8_g1u1_subGU_{128,256,512}_gelu.co` — **gelu only**). Parity gating is the operator's,
not the backend's: see [`../numerics.md`](../numerics.md).

## Pitfalls & anti-patterns
- ⚠ **There is no stage-2 asm kernel.** `hsa/gfx942/fmoe_2stages/` holds 190 objects and
  `hsa/gfx950/fmoe_2stages/` holds 186; `grep -c stage2` over either is **0**. Any plan that reads
  "replace both stages with asm" is unimplementable on this box — stage-2 goes to
  `ck_moe_stage2_fwd` (`aiter/fused_moe.py:1130`, binding `aiter/ops/moe_op.py:320`) or CK-Tile.
- ⚠ **No assembly source is vendored.** `find -iname '*.s'` over both `/sgl-workspace/aiter` and
  `/sgl-workspace/geak_flydsl/aiter_main/aiter` returns **zero** hits. The `.co` files are built from
  a private repo that is not here. Treat asm as a **closed axis** when scoping a campaign: it bounds
  the implementation, not the problem.
- ⚠ **Beating asm means not using asm.** Since you cannot edit the binary, the only way a campaign
  improves on this path is to re-author the stage in a language with a source seam
  ([`flydsl.md`](flydsl.md), [`triton.md`](triton.md), [`ck.md`](ck.md)) and *win on measurement*.
  That is a legitimate goal — just never report it as "tuned the asm kernel".
- ⚠ **Per-gen coverage differs.** gfx942 carries the larger `.co` set (int4fp8, the `smf`/`multix`
  smooth-quant families); gfx950's set is smaller and MXFP4-centric. A config that dispatches to asm
  on MI300X may fall to CK on MI350.

## How to verify
`AITER_LOG_MORE=1` and confirm an `fmoe*` (1-stage) or `fmoe_stage1_*` (2-stage) name appears. If you
see only `moe_ck2stages_*`, the asm path did not engage and your A/B is measuring CK against CK.
Cross-check the selected object against `select_hsaco()`'s mapping for your quant signature.

## Alternatives / cross-links
[`aiter.md`](aiter.md) (the dispatcher that selects this) · [`ck.md`](ck.md) (stage-2, always) ·
[`hip.md`](hip.md) (the HIP-level view of the same stage-1) · [`flydsl.md`](flydsl.md) (source seam
for gfx950 MXFP4) · [`triton.md`](triton.md) (portable fallback) · [`../overview.md`](../overview.md) ·
[`../tuning.md`](../tuning.md)

## Sources
- Host launch class over the sorted-expert layout: `ROCm/aiter@a6bb49937:csrc/py_itfs_cu/asm_fmoe.cu:71`
  (`class FMoeKernel`), `:97` (`launch_kernel`).
- 1-stage Python entry points: `ROCm/aiter@a6bb49937:aiter/ops/moe_op.py:62` (`fmoe`), `:76`
  (`fmoe_int8_g1u0`), `:95` (`fmoe_g1u1`), `:115` (`fmoe_g1u1_tkw1`), `:135`, `:154`, `:173`
  (`fmoe_fp8_blockscale_g1u1`).
- 2-stage stage-1: `ROCm/aiter@a6bb49937:csrc/py_itfs_cu/asm_moe_2stage.cu:149`
  (`AITER_CTYPES_DEFINE_ENTRYPOINT_VOID(moe_stage1_g1u1, ...)`), Python binding
  `ROCm/aiter@a6bb49937:aiter/ops/moe_op.py:196`.
- Dispatch: `ROCm/aiter@a6bb49937:aiter/fused_moe.py:310` (`metadata.run_1stage`), `:367`
  (`fused_moe_1stage`), `:1389` (`asm_stage1`), `:1427` (`aiter.moe_stage1_g1u1`), `:1130`
  (`ck_moe_stage2_fwd` — the stage-2 fallback).
- Object selection: `ROCm/aiter@a6bb49937:csrc/cpp_itfs/moe/asm_moe.py:42` (`select_tile`), `:132`
  (`select_hsaco`).
- Binary-only, asymmetric catalog (counted on box 2026-10-13, `/sgl-workspace/aiter` @ `a6bb49937`):
  `hsa/gfx942/fmoe_2stages/` 190 entries / 0 matching `stage2`; `hsa/gfx950/fmoe_2stages/` 186 / 0
  (same counts in `/sgl-workspace/geak_flydsl/aiter_main/aiter` @ `4faf87137`). `find -iname '*.s'`
  over both trees: 0 hits.
