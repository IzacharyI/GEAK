# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Frozen Triton GEMM baseline for the Triton-to-FlyDSL author A/B (Llama-3-70B layer GEMM).

Kernel body/launch adapted from AITER gemm_a16w16. The gfx942 config AITER selects for this
exact shape (M-bucket heuristic over the default gfx942-GEMM-A16W16.json; no per-(M,N,K) tuned
table exists upstream) is embedded below so neither arm consults a mutable tuning DB.
"""

import torch
import triton
import triton.language as tl


CONFIGS = {
    (2048, 10240, 8192): {
        "BLOCK_SIZE_M": 256,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": "",
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 8192,
    },
}


@triton.jit
def _remap_xcd(pid, grid_mn, num_xcds: tl.constexpr = 8):
    pids_per_xcd = (grid_mn + num_xcds - 1) // num_xcds
    tall_xcds = grid_mn % num_xcds
    tall_xcds = num_xcds if tall_xcds == 0 else tall_xcds
    xcd = pid % num_xcds
    local_pid = pid // num_xcds
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    return pid


@triton.jit
def _pid_grid(pid, num_pid_m, num_pid_n, group_size_m: tl.constexpr = 1):
    if group_size_m == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = group_size_m * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * group_size_m
        group_m = min(num_pid_m - first_pid_m, group_size_m)
        tl.assume(group_m >= 0)
        pid_m = first_pid_m + (pid % group_m)
        pid_n = (pid % num_pid_in_group) // group_m
    return pid_m, pid_n


@triton.heuristics({
    "even_k": lambda args: args["k"] % args["block_k"] == 0,
    "even_mn": lambda args: (args["m"] % args["block_m"] == 0)
    and (args["n"] % args["block_n"] == 0),
})
@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    even_k: tl.constexpr,
    even_mn: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, block_m)
    num_pid_n = tl.cdiv(n, block_n)
    pid = _remap_xcd(pid, num_pid_m * num_pid_n)
    pid_m, pid_n = _pid_grid(pid, num_pid_m, num_pid_n, group_m)

    offs_k = tl.arange(0, block_k)
    if even_mn:
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
    else:
        offs_m = (pid_m * block_m + tl.arange(0, block_m)) % m
        offs_n = (pid_n * block_n + tl.arange(0, block_n)) % n
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for kk in range(0, k, block_k):
        if even_k:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs, cache_modifier=cache_modifier)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] + kk < k, other=0.0)
            b = tl.load(
                b_ptrs,
                mask=offs_k[:, None] + kk < k,
                other=0.0,
                cache_modifier=cache_modifier,
            )
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += block_k * stride_ak
        b_ptrs += block_k * stride_bk

    c = acc.to(c_ptr.type.element_ty)
    offs_cm = pid_m.to(tl.int64) * block_m + tl.arange(0, block_m)
    offs_cn = pid_n.to(tl.int64) * block_n + tl.arange(0, block_n)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    if even_mn:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < m) & (offs_cn[None, :] < n))


def gemm(a, b, out=None, *, stream=None):
    """Compute C = A @ B.T for contiguous bf16 matrices."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K] and B[N,K]")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("this frozen baseline supports bf16 inputs")
    m, k = a.shape
    n = b.shape[0]
    key = (m, n, k)
    if key not in CONFIGS:
        raise ValueError(f"shape {key} is outside the frozen case manifest")
    cfg = CONFIGS[key]
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    elif out.shape != (m, n) or out.dtype != torch.bfloat16 or out.device != a.device:
        raise ValueError("out must be bf16 with shape (M,N) on A's device")

    b_t = b.T
    grid = (triton.cdiv(m, cfg["BLOCK_SIZE_M"]) * triton.cdiv(n, cfg["BLOCK_SIZE_N"]),)
    _gemm_kernel[grid](
        a,
        b_t,
        out,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_t.stride(0),
        b_t.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=cfg["BLOCK_SIZE_M"],
        block_n=cfg["BLOCK_SIZE_N"],
        block_k=cfg["BLOCK_SIZE_K"],
        group_m=cfg["GROUP_SIZE_M"],
        cache_modifier=cfg["cache_modifier"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
    )
    return out
