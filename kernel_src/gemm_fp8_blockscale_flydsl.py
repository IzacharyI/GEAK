# fp8 e4m3fnuz block-scale GEMM authored in FlyDSL (aiter's Python kernel DSL).
#
# Math contract (from meta.json):
#   Y[M,N] = (X[M,K]*x_scale) @ (W[N,K]*w_scale).T
#   with 128x128 fp8 block-scale dequant, fp32 accumulation, bf16 output.
#
#   X : fp8_e4m3fnuz [M, K]         x_scale : fp32 [M,  ceil(K/128)]  (per-row, per-K-block)
#   W : fp8_e4m3fnuz [N, K]         w_scale : fp32 [ceil(N/128), ceil(K/128)]  (per-128x128 block)
#
# Correctness-first seed for the optimize loop. It computes the product with the
# plain (NON block-scaled) MFMA `mfma_f32_16x16x32_fp8_fp8` and applies the
# arbitrary fp32 block scales in SOFTWARE *after* the MFMA (promote-then-scale).
# This deliberately avoids the native block-scaled MFMA (mfma_scale_*_f8f6f4),
# whose scale is E8M0 (power-of-two only) and would silently round the arbitrary
# fp32 block scales -> parity fail. See perf_knowledge dense_gemm/backends/flydsl.md
# "fp8 a8w8 block-scale parity trap".
#
# One 16x16 output tile per workgroup (64 threads = 1 wave, one 16x16x32 MFMA
# lane group). Naive but correct; the optimize loop widens tiles / stages / splits K.

import functools

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, vector
from flydsl.expr.typing import T

BLOCK_K = 128
BLOCK_N = 128


@functools.lru_cache(maxsize=None)
def _build(M, N, K, SK):
    """Build (and cache) the JIT launcher for one (M,N,K,scale_k) shape."""

    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, Y: fx.Tensor):
        vec4f = T.vec(4, T.f32)
        t = fx.thread_idx.x
        lane16 = t % 16          # output column within the 16-wide tile / A,B row
        laned = t // 16          # 0..3 : which 8-fp8 K-chunk this lane feeds
        by = fx.block_idx.x      # n tile index (16 wide)
        bm = fx.block_idx.y      # m tile index (16 tall)
        m0 = bm * 16
        n0 = by * 16
        nb = n0 // BLOCK_N       # which 128-wide N block (for w_scale)

        # Buffer resources with hardware bounds checking: OOB loads read 0,
        # OOB stores are dropped -> ragged M (e.g. M=8) needs no masking.
        Xr = buffer_ops.create_buffer_resource(X, max_size=True)
        Wr = buffer_ops.create_buffer_resource(W, max_size=True)
        XSr = buffer_ops.create_buffer_resource(XS, max_size=True)
        WSr = buffer_ops.create_buffer_resource(WS, max_size=True)
        Yr = buffer_ops.create_buffer_resource(Y, max_size=True)

        acc0 = arith.constant(0.0, type=T.f32)
        accs_init = [acc0, acc0, acc0, acc0]
        NKB = K // BLOCK_K
        K8 = K // 8

        # Loop over 128-wide K blocks; each block carries one fp32 (x_scale,w_scale).
        for kb, state in range(0, NKB, 1, init=accs_init):
            main = list(state)
            kbi = arith.index_cast(T.i32, kb)
            bacc = arith.constant_vector(0.0, vec4f)
            # 4 MFMA calls * 32 K each = 128 K in this block. Lanes load 8 fp8
            # (one i64) at a time; laned selects the 8-chunk inside each 32.
            for ks in range_constexpr(4):
                koff = kbi * (BLOCK_K // 8) + ks * 4 + laned
                ai = buffer_ops.buffer_load(Xr, (m0 + lane16) * K8 + koff, vec_width=1, dtype=T.i64)
                bi = buffer_ops.buffer_load(Wr, (n0 + lane16) * K8 + koff, vec_width=1, dtype=T.i64)
                bacc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(vec4f, [ai, bi, bacc])
            # Software fp32 post-MFMA scale: this K block contributes bacc * (x_scale*w_scale).
            ws = buffer_ops.buffer_load(WSr, nb * SK + kbi, vec_width=1, dtype=T.f32)
            for ii in range_constexpr(4):
                row = m0 + laned * 4 + ii
                xs = buffer_ops.buffer_load(XSr, row * SK + kbi, vec_width=1, dtype=T.f32)
                sc = arith.mulf(xs, ws)
                part = arith.mulf(vector.extract(bacc, static_position=[ii], dynamic_position=[]), sc)
                main[ii] = arith.addf(main[ii], part)
            res = yield main

        # Epilogue: cast fp32 accumulator -> bf16 and store. MFMA 16x16 output
        # layout: lane holds 4 rows (laned*4+ii), column = lane16.
        main = list(res)
        for ii in range_constexpr(4):
            row = m0 + laned * 4 + ii
            col = n0 + lane16
            yb = arith.truncf(T.bf16, main[ii])
            buffer_ops.buffer_store(yb, Yr, row * N + col)

    @flyc.jit
    def launch(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, Y: fx.Tensor,
               stream: fx.Stream = fx.Stream(None)):
        kern(X, W, XS, WS, Y).launch(
            grid=(N // 16, (M + 15) // 16, 1), block=(64, 1, 1), stream=stream
        )

    return launch


def gemm(x, w, x_scale, w_scale, out=None):
    """Y = X @ W^T with 128x128 fp8 block-scale dequant (FlyDSL seed)."""
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("expected x[M,K] and w[N,K]")
    M, K = x.shape
    N = w.shape[0]
    SK = x_scale.shape[1]
    x = x.contiguous()
    w = w.contiguous()
    x_scale = x_scale.contiguous()
    w_scale = w_scale.contiguous()
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    launch = _build(M, N, K, SK)
    stream = torch.cuda.current_stream()
    launch(
        x.view(-1), w.view(-1), x_scale.view(-1), w_scale.view(-1), out.view(-1),
        stream=stream,
    )
    return out
