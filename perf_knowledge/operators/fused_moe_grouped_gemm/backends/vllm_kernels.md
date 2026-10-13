---
title: fused_moe_grouped_gemm on vllm_kernels — SOTA card
kind: sota_card
operator: fused_moe_grouped_gemm
backend: vllm_kernels
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [prefill, decode]
status: competitive
updated: 2026-10-13
sources:
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py
  - https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
  - https://github.com/vllm-project/vllm/issues/34641
---

# fused_moe_grouped_gemm × vllm_kernels

## TL;DR (status: competitive — vLLM owns the *dispatch*, aiter owns the kernel on ROCm)
> vLLM does author a fused-MoE grouped GEMM: its Triton `fused_moe_kernel` plus per-shape tuned JSON
> configs, and the C++/HIP helpers around it (`moe_align_block_size`, `topk_softmax`). But on ROCm
> the recommended configuration **does not run it** — `VLLM_ROCM_USE_AITER=1` +
> `VLLM_ROCM_USE_AITER_MOE=1` re-registers the layer onto aiter's fused kernel. So this card is the
> **dispatch-and-registration view**: which env hierarchy decides whether vLLM's own kernel or
> [`aiter.md`](aiter.md) runs, and when the Triton one is still the right answer (it is the
> correctness fallback). The kernel itself is covered by [`triton.md`](triton.md) — that card and
> this one describe the same Triton code from two sides; do not count them as two implementations.

## ⚠ Evidence basis: off-box
**No vLLM source, wheel or installed package exists on this machine** (checked 2026-10-13: `import
vllm` → `ModuleNotFoundError`; a filesystem-wide search for `*vllm*` finds only shell scripts that
invoke a vLLM server elsewhere, aiter CI workflow files, and this knowledge base's own notes). Every
claim below is therefore cited to **public upstream URLs**, never to a `repo@commit:path` on this
box, and cannot be line-verified here. Treat specifics as *upstream-as-documented*; if a campaign
depends on one, clone vLLM first. The rest of this tree's `vllm_kernels` cards share this basis —
see [`../../../backends/vllm_kernels/overview.md`](../../../backends/vllm_kernels/overview.md).

## SOTA implementation(s)
| impl | source | gens/dtypes | when best |
|---|---|---|---|
| aiter fused MoE via `rocm_aiter_fused_moe` | `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py` | gfx942/950; bf16, fp8 (per-token / block-scale), int8 | **the recommended ROCm path** — see [`aiter.md`](aiter.md) |
| vLLM Triton `fused_moe_kernel` + tuned JSON | `vllm/model_executor/layers/fused_moe/fused_moe.py`, `configs/E=…,N=…,device_name=MI300X.json` | gfx942/950; bf16/fp16/fp8/int8 | `VLLM_ROCM_USE_AITER=0`, a shape aiter lacks, or when the aiter path crashes |
| `moe_align_block_size`, `topk_softmax` (C++/HIP) | `csrc/moe/` | both | the sort/pad and routing around the GEMM — used by the Triton path |

## Config space / knobs (the env hierarchy *is* the config)
| env | default | effect on this operator |
|---|---|---|
| `VLLM_ROCM_USE_AITER` | **0** | master gate; `0` means every sub-flag below is inert and the Triton kernel runs |
| `VLLM_ROCM_USE_AITER_MOE` | 1 | once the master is on, routes the FusedMoE layer to aiter |
| `VLLM_ROCM_USE_AITER_FP4BMM` | 1 | **crashes on gfx942** (no FP4 HW) — set `0` on MI300X (vLLM #34641) |
| tuned JSON (`E=…,N=…,device_name=…`) | shipped | the Triton path's per-shape BLOCK_M/N/K, stages, warps |

The default being `0` is the trap worth remembering: an un-configured ROCm vLLM runs the **Triton**
kernel, so a "vLLM MoE baseline" measured without the env block is not the aiter path.

## Numerics / parity
fp32 accumulate in both paths. The Triton reference is the one to trust when the aiter path is under
suspicion — that is its main value here. Quant parity is the operator's concern, not the
dispatcher's: see [`../numerics.md`](../numerics.md).

## Integration (rebind seam)
The seam is `direct_register_custom_op` — aiter wrappers are registered as `torch.ops.vllm.*` with
fake/meta implementations so they survive `torch.compile` (Inductor fuses around them rather than
decomposing them). The FusedMoE layer then selects between the registered aiter op and vLLM's own
Triton path. Same mechanism as the router: [`../../moe_routing_topk/backends/vllm_kernels.md`](../../moe_routing_topk/backends/vllm_kernels.md).

## Pitfalls & anti-patterns
- ⚠ **Benchmarking "vLLM" without declaring the env block measures a different kernel.** State
  `VLLM_ROCM_USE_AITER` in any result, or the number is uninterpretable.
- ⚠ **FP4BMM defaults on and crashes on MI300X** — set `VLLM_ROCM_USE_AITER_FP4BMM=0` on gfx942
  (vLLM #34641).
- ⚠ **Image mismatch**: `VLLM_ROCM_USE_AITER=1` with no aiter wheel in the image → ImportError at
  layer construction, not a graceful fallback.
- ⚠ **Do not cite this card as on-box evidence.** See the evidence-basis note above.

## How to verify
Confirm *which* kernel ran before comparing anything: `AITER_LOG_MORE=1` should show `fmoe_g1u1*` /
`moe_ck2stages_*` when the aiter path is live; if you instead see Triton JIT activity, the env gate
did not take. Then A/B tok/s with `VLLM_ROCM_USE_AITER_MOE=1` vs `=0` on the same shapes.

## Alternatives / cross-links
[`aiter.md`](aiter.md) (what this dispatches to) · [`triton.md`](triton.md) (the same Triton kernel,
kernel-side) · [`asm.md`](asm.md) · [`ck.md`](ck.md) ·
[`../../../backends/vllm_kernels/overview.md`](../../../backends/vllm_kernels/overview.md) ·
[`../../../backends/vllm_kernels/aiter_integration.md`](../../../backends/vllm_kernels/aiter_integration.md) ·
[`../overview.md`](../overview.md)

## Sources
- **Not verifiable on this box**: no vLLM checkout, wheel or site-packages install exists here
  (`import vllm` → `ModuleNotFoundError`; filesystem-wide `*vllm*` search, 2026-10-13, finds only
  server-launch scripts, aiter CI workflow files, and this tree's own notes). All citations below are
  upstream URLs.
- Triton fused-MoE kernel and tuned configs: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py
- aiter MoE registration on ROCm: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py
- Env hierarchy and defaults: https://github.com/vllm-project/vllm/blob/main/vllm/envs.py
- FP4BMM default-on crash on gfx942: https://github.com/vllm-project/vllm/issues/34641
