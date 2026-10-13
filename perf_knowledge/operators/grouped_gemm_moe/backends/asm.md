---
title: grouped_gemm_moe on asm — SOTA card
kind: sota_card
operator: grouped_gemm_moe
backend: asm
gens: [gfx942, gfx950]
dtypes: [fp8_e4m3_fnuz, int8, bf16, fp16, fp4_e2m1]
regimes: [prefill, decode]
status: competitive
updated: 2026-10-11
sources:
  - ROCm/aiter@4faf8713:csrc/py_itfs_cu/asm_moe_2stage.cu
  - ROCm/aiter@4faf8713:csrc/py_itfs_cu/asm_fmoe.cu
  - ROCm/aiter@4faf8713:aiter/fused_moe.py
  - ROCm/aiter@4faf8713:hsa/codegen.py
  - ROCm/aiter@4faf8713:csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
  - https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
---

# grouped_gemm_moe × asm

## TL;DR
> aiter ships **two unrelated asm MoE families**, and the grouped-GEMM one covers **stage 1 only**:
> `moe_stage1_g1u1` (gate+up GEMM + fused activation over sorted-expert blocks), paired with a **CK**
> stage 2. There is no asm stage-2 kernel. The kernels are pre-compiled `.co` blobs selected per shape
> by an offline race, so "using asm" here means *winning a tuned-DB row*, not authoring anything.
> Hard limit for low-bit work: the 2-stage asm path accepts **fp8 and int8 only** — fp4/mxfp4 lives
> exclusively in the *other* (1-stage fused) family.

## SOTA implementation
Stage 1 enters through `asm_stage1` (`aiter/fused_moe.py:4530`), a thin Python wrapper that allocates the
split-K scratch and calls the ctypes entrypoint at `:4568`:

```python
# aiter/ops/moe_op.py:299  @compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
moe_stage1_g1u1(...)        # -> csrc/py_itfs_cu/asm_moe_2stage.cu:149
```

Stage 2 is chosen independently by `kernelName2`'s string prefix (`aiter/fused_moe.py:3963-3992`):
`"flydsl_"` → FlyDSL, `"cktile_"` → CK-Tile, else `ck_moe_stage2_fwd`. In sampled `tuned_fmoe.csv`
rows asm-stage1 is almost always partnered with `moe_ck2stages_gemm2_*`.

| impl | source | gens/dtypes | role | when best |
|---|---|---|---|---|
| asm stage-1 g1u1 | `asm_moe_2stage.cu:149` | gfx942/950; fp8, int8 | grouped gate+up GEMM + fused act | wins the per-shape race in `tuned_fmoe.csv` |
| asm stage-2 | **does not exist** | — | — | stage 2 is always CK / CK-Tile / FlyDSL |
| 1-stage fused `fmoe_g1u1` | `asm_fmoe.cu:559` | gfx942/950; fp8/int8/**mxfp4**/int4 | gate+up+act+down in one launch | small-M decode; the only asm fp4 path |
| 1-stage `fmoe_g1u1_tkw1` | `asm_fmoe.cu:737` | same | applies topk routed weight *inside* stage 1 | saves a later scale pass |

`g1u1` vs `g1u0` is defined literally at `aiter/fused_moe.py:4819` — `use_g1u1 = w1.shape[1] == (2*inter_dim)`,
i.e. g1u1 = `w1` is the fused `[gate;up]` matrix. The 2-stage asm file is **g1u1-only**.

## Config space / knobs
Only 4 `(in, weight, out, quant, do_weight)` tuples exist; anything else hits
`AITER_CHECK(false, "Unsupported ...")` at `asm_moe_2stage.cu:98-101`:

| in | weight | out | quant_type | do_weight | config map | line |
|---|---|---|---|---|---|---|
| fp8 | fp8 | bf16 | `per_Token` | true | `cfg_fmoe_stage1_bf16_pertokenFp8_doweight_g1u1` | 64 |
| fp8 | fp8 | bf16 | `per_Token` | false | `cfg_fmoe_stage1_bf16_pertokenFp8_g1u1` | 72 |
| i8 | i8 | bf16 | `per_Token` | false | `cfg_fmoe_stage1_bf16_pertokenInt8_g1u1` | 80 |
| fp8 | fp8 | fp8 | `per_1x128` | false | `cfg_fmoe_stage1_bf16_pertokenFp8_blockscale_g1u1` | 88 |

| param | range / values | effect | enforced at |
|---|---|---|---|
| `block_m` | = selected kernel's `tile_m` (16/32/…) | also the `moe_sorting` block size | hard assert `:215` |
| `tile_n` (= `sub_GU`) | per-CSV set | sub-group width over `inter_dim`; `gdx = ceil(hidden/sub_GU)` | `:196-198`, `:214`, `:272` |
| `ksplit` | int, `k_num = 1 << ksplit` | split-K; `gdz = k_num`, fp32 scratch then combine | `:265-266`, `:274` |
| `activation` | `No/Silu/Gelu/Swiglu/Situv2/GeluTanh` | fused into stage 1 | `aiter_enum.h:7-15`, `:262` |
| `do_weight` | implicit `sorted_weights != nullptr` | picks the doweight config map | `:176` |
| `kernelName` | explicit, else heuristic | occupancy heuristic minimizes `ceil(tg_num/num_cu)` rounds | `:106-147` |

Blobs live at `hsa/<arch>/fmoe_2stages/*.co` with CSV manifests (`tile_m,tile_n,knl_name,co_name`);
`hsa/codegen.py:116-230` turns each CSV into one `static CFG cfg_<name>` map, keyed by **`arch + knl_name`
string concatenation** (`codegen.py:191-194`, looked up at `asm_moe_2stage.cu:182`). Naming grammar:
`fmoe_stage1_bf16_pertoken{Int8|Fp8}[_blockscale|_doweight]_g1u1_<tile_m>x<tile_n>[_<N>tg][_pf{2,3}]`
— confirmed by the tuner's path template at `gemm_moe_tune.py:3019-3023`. At load time
`AiterAsmKernel::load_hsaco_file` (`csrc/include/aiter_hip_common.h:468-505`) reads
`$AITER_ASM_DIR/<arch>/<path>` if set, else an embedded blob map, and validates LDS usage first (`:395-467`).

## Numerics / parity
fp32 accumulate. Scale tensors **must be fp32** — `AITER_CHECK(w1_scale->dtype() == AITER_DTYPE_fp32)`
at `:224-227`, because the stride math assumes 4-byte scale elements. That rules out e8m0/mxfp4-style
scales structurally, not just by dispatch. Parity gate is the grouped cosine gate, not elementwise —
see [../numerics.md](../numerics.md).

## Integration (rebind seam)
No standalone env seam. The only seam is the tuned DB: `aiter/configs/tuned_fmoe.csv` rows whose
`kernelName1` is a mangled asm symbol (e.g. `_ZN5aiter47fmoe_stage1_bf16_pertokenFp8_g1u1_32x64_4tg_pf3E`)
fall through to the default branch → `asm_stage1` (`fused_moe.py:4039-4051`). With **no** tuned row the
dispatcher routes most weight dtypes to CK (`:3949-3962`); **int8 weight is the one combo that reaches asm
by default**. The race itself is offline: `gen_2stages_asm1_task` (`gemm_moe_tune.py:3000-3070`) enumerates
every `(tile_m, tile_n)` from the manifest CSV and times it via `FmoeTuner.run_asm_stage1` (`:1346-1367`).

## Pitfalls & anti-patterns
- **No fp4/mxfp4 in the 2-stage asm path.** Routing an fp4x2 shape there raises
  `Unsupported input_type:fp4x2, weight_type:fp4x2, out_type:bf16, quant_type:3` at `asm_moe_2stage.cu:101`.
  This is a live failure mode, not a hypothetical — it is exactly what aborts the GLM-5.2 mxfp4 oracle's
  M=1 edge case when tuned-config resolution names a `flydsl_moe1_*` stage 1 but metadata resolves to
  `asm_stage1`. Use the 1-stage `fmoe_g1u1` family for fp4 (`asm_fmoe.cu:632-665`).
- **Don't look for an asm stage 2.** No `fmoe_stage2_*.co` exists anywhere; assume CK.
- **`tuned_grouped_fmoe.csv` is not this table.** It is gfx1250-only FlyDSL fp4 (`grouped_gemm1_fp4_gugu`
  etc.) and is never read by the asm/CK 2-stage dispatcher. Conflating it with `tuned_fmoe.csv` is a
  silent wrong-config bug.
- **gfx1250 has zero asm MoE kernels** (no `fmoe/` or `fmoe_2stages/` under `hsa/gfx1250/`).
- Tile sets are **not** arch-identical: gfx942 carries four int8 tiles gfx950 lacks — `(16,192)`,
  `(16,384)`, `(32,192)`, `(32,384)`. A config copied across arches can fail to resolve.
- No int8-doweight and no blockscale-doweight config exists; combine doweight with fp8 per-token only.
- fp8 block-scale: aiter itself notes `# for fp8 blockscale, ck has better performance so disable
  assembly kernel` (`fused_moe.py:3396-3397`), and `fused_moe_bf16_asm.py:142-152` restricts it to bf16
  hidden states, block shape (128,128), fp8_e4m3fnuz weights.
- No `.s` sources ship — only `.co` blobs. The `_pf2`/`_pf3`/`_<N>tg` suffixes are undocumented in-repo;
  do not reason about them as if their meaning were established.

## How to verify (worked example)
```bash
# 1. confirm the asm kernel actually ran (not a silent CK fallback)
rocprofv3 --stats -- python op_tests/test_moe_2stage.py   # expect fmoe_stage1_bf16_pertoken*_g1u1_*
# 2. confirm which row the dispatcher picked
grep '^<M>,<N>,<K>' aiter/configs/tuned_fmoe.csv          # kernelName1 / kernelName2 / block_m / ksplit
# 3. confirm the blob exists for your arch before blaming perf
ls hsa/gfx950/fmoe_2stages/ | grep 32x64
```
Gate correctness with the grouped cosine criterion, not the dense elementwise one.

## Alternatives / cross-links
[[operators/grouped_gemm_moe/backends/aiter]] (dispatches asm stage 1) ·
[[operators/grouped_gemm_moe/backends/ck]] (the stage-2 partner) ·
[[operators/grouped_gemm_moe/backends/flydsl]] (fp4 stage 1/2 on gfx950/1250) ·
[[operators/dense_gemm/backends/asm]] · [[operators/moe_routing_topk/overview]] ·
[[operators/grouped_gemm_moe/tuning]] · language ref `languages/asm_mfma/`.

## Sources
- asm 2-stage entrypoint, `get_cfg` table, grid/knob asserts: `csrc/py_itfs_cu/asm_moe_2stage.cu` (`ROCm/aiter@4faf8713`).
- 1-stage fused family incl. mxfp4 dispatch: `csrc/py_itfs_cu/asm_fmoe.cu`, `aiter/ops/moe_op.py` (same commit).
- Python dispatch, kernelName prefixes, default fallthrough: `aiter/fused_moe.py` (same commit).
- Config-map codegen and `arch+knl_name` key: `hsa/codegen.py` (same commit).
- Offline race / manifest path template: `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py` (same commit).
- CDNA3/4 matrix-core background: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
