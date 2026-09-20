# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Frozen fp8 block-scale GEMM baseline (aiter-Triton gemm_a8w8_blockscale denominator).

Denominator for the Triton-to-FlyDSL fp8 author run. W=fp8/A=fp8 (e4m3fnuz), 128x128
block scales, bf16 output. The gfx942 config aiter selects for each exact (M,N,K) via
GEMM-A8W8_BLOCKSCALE is embedded below so neither the baseline nor the candidate
consults a mutable tuning DB. This weight (N,K) serves both decode (M=8) and prefill
(M=32768); each M keeps its own frozen config.
"""

import torch
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale


CONFIGS = {
    (8, 4096, 1024): {
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 1,
        "cache_modifier": '.cg',
        "kpack": 2,
        "matrix_instr_nonkdim": 16,
        "num_stages": 2,
        "num_warps": 4,
        "waves_per_eu": 2,
    },
    (32768, 4096, 1024): {
        "BLOCK_SIZE_K": 128,
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 1,
        "cache_modifier": '.cg',
        "kpack": 2,
        "matrix_instr_nonkdim": 16,
        "num_stages": 2,
        "num_warps": 4,
        "waves_per_eu": 2,
    },
}


def gemm(x, w, x_scale, w_scale, out=None):
    """Y = X @ W^T with 128x128 fp8 block-scale dequant (frozen aiter denominator)."""
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("expected x[M,K] and w[N,K]")
    m, k = x.shape
    n = w.shape[0]
    key = (m, n, k)
    if key not in CONFIGS:
        raise ValueError(f"shape {key} is outside the frozen case manifest")
    return gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, y=out, config=dict(CONFIGS[key])
    )
