# SPDX-License-Identifier: MIT
"""FlyDSL fp8 e4m3fnuz 128x128 block-scale GEMM — register-tiled MFMA seed (gfx942).

Simplest correct-first baseline for the Triton->FlyDSL fp8 block-scale author run.

    Y[m, n] = sum_kb ( x_scale[m, kb] * w_scale[n // 128, kb]
                       * sum_{k in block kb} X[m, k] * W[n, k] )

X [M, K] fp8 (e4m3fnuz), W [N, K] fp8 (row-major).  Y = X @ W.T after per-128-K-block
dequant, fp32 accumulate, bf16 output.  The (x_scale * w_scale) block factor is an
arbitrary fp32 value, so it is applied in SOFTWARE fp32 AFTER each 128-K partial dot
(native block-scaled MFMA would need E8M0 and does not exist on gfx942 anyway).

The K reduction runs on the matrix core via rocdl.mfma_f32_16x16x32_fp8_fp8 (fp8 in,
fp32 accumulate).  Each wave owns a (M_REP*16) x (N_REP*16) output tile and issues
M_REP*N_REP independent MFMAs per 32-K step.  Buffer resources use hardware OOB
guarding (max_size=False) so a partial M tile (decode M=8 in a 16-row tile) reads 0 for
the missing rows and drops the corresponding stores.  Only ``flydsl`` / ``torch``
imported.  This is the optimize loop's correct starting point, not a tuned kernel.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, range_constexpr, rocdl, vector
from flydsl.expr.typing import T

BLOCK_K = 128            # dequant block along K (one scale per 128-K)
FP8_PER_WORD = 4         # 4 fp8 bytes packed per i32 word
MFMA_M = 16
MFMA_N = 16
MFMA_K = 32              # rocdl.mfma_f32_16x16x32_fp8_fp8
WAVE = 64

M_REP = 4               # prefill wave tile: 64 x 64
N_REP = 4
KS_PER_BLOCK = BLOCK_K // MFMA_K          # 4 MFMA k-steps per 128-K dequant block
WORDS_PER_LANE = 8 // FP8_PER_WORD        # 2 i32 words each lane feeds one MFMA (8 fp8)
WORDS_PER_KSTEP = MFMA_K // FP8_PER_WORD  # 8 i32 words spanning one 32-K step


def _compile(m, n, k, m_rep, n_rep):
    scale_k = k // BLOCK_K
    k_words = k // FP8_PER_WORD
    tile_m = m_rep * MFMA_M
    tile_n = n_rep * MFMA_N
    n_tiles = n // tile_n
    m_tiles = (m + tile_m - 1) // tile_m
    grid = m_tiles * n_tiles
    kb_words = BLOCK_K // FP8_PER_WORD        # 32 i32 words per 128-K block per row

    kernel_name = f"flygemm_fp8blk_mfma_{m}x{n}x{k}_{m_rep}x{n_rep}"

    @flyc.kernel(name=kernel_name, known_block_size=[WAVE, 1, 1])
    def gemm_kernel(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        XS: fx.Tensor,
        WS: fx.Tensor,
    ):
        # OOB-guarded buffer resources: partial M tiles read 0 / drop stores.
        x_rsrc = buffer_ops.create_buffer_resource(X, max_size=False)
        w_rsrc = buffer_ops.create_buffer_resource(W, max_size=False)
        xs_rsrc = buffer_ops.create_buffer_resource(XS, max_size=False)
        ws_rsrc = buffer_ops.create_buffer_resource(WS, max_size=False)
        out_rsrc = buffer_ops.create_buffer_resource(Out, max_size=False)

        lane = fx.Int32(fx.thread_idx.x)
        pid = fx.Int32(fx.block_idx.x)

        tile_m_idx = pid // n_tiles
        tile_n_idx = pid % n_tiles
        m_base = tile_m_idx * tile_m
        n_base = tile_n_idx * tile_n
        nb = n_base // BLOCK_K            # weight-scale block index

        m_in = lane % MFMA_M             # A input row within a 16-block
        kg = lane // MFMA_M              # K-group for A/B fragment (0..3)
        n_in = lane % MFMA_N            # B input col / C output col within a 16-block
        kg2 = kg * WORDS_PER_LANE        # 0,2,4,6

        a_wrow = [((m_base + mm * MFMA_M) + m_in) * k_words for mm in range_constexpr(m_rep)]
        w_wrow = [((n_base + nn * MFMA_N) + n_in) * k_words for nn in range_constexpr(n_rep)]
        out_row0 = [(m_base + mm * MFMA_M) + kg * 4 for mm in range_constexpr(m_rep)]
        xs_row0 = [((m_base + mm * MFMA_M) + kg * 4) * scale_k for mm in range_constexpr(m_rep)]

        n_acc = m_rep * n_rep
        zero4 = arith.constant_vector(0.0, T.f32x4)
        init_state = [zero4 for _ in range_constexpr(n_acc)]
        init_state.append(arith.constant(0, type=T.i32))

        for _, state in range(0, scale_k, init=init_state):
            masters = [state[i] for i in range_constexpr(n_acc)]
            kb = state[n_acc]
            k_word_base = kb * kb_words

            partials = [arith.constant_vector(0.0, T.f32x4) for _ in range_constexpr(n_acc)]
            a_all = []
            b_all = []
            for ks in range_constexpr(KS_PER_BLOCK):
                base_off = k_word_base + ks * WORDS_PER_KSTEP + kg2
                for mm in range_constexpr(m_rep):
                    a_v2 = buffer_ops.buffer_load(
                        x_rsrc, a_wrow[mm] + base_off, vec_width=2, dtype=T.i32
                    )
                    a_all.append(vector.extract(
                        vector.bitcast(T.vec(1, T.i64), a_v2),
                        static_position=[0], dynamic_position=[],
                    ))
                for nn in range_constexpr(n_rep):
                    b_v2 = buffer_ops.buffer_load(
                        w_rsrc, w_wrow[nn] + base_off, vec_width=2, dtype=T.i32
                    )
                    b_all.append(vector.extract(
                        vector.bitcast(T.vec(1, T.i64), b_v2),
                        static_position=[0], dynamic_position=[],
                    ))
            for ks in range_constexpr(KS_PER_BLOCK):
                a_frag = a_all[ks * m_rep:(ks + 1) * m_rep]
                b_frag = b_all[ks * n_rep:(ks + 1) * n_rep]
                for mm in range_constexpr(m_rep):
                    for nn in range_constexpr(n_rep):
                        idx = mm * n_rep + nn
                        partials[idx] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [a_frag[mm], b_frag[nn], partials[idx], 0, 0, 0]
                        )

            # fp32 block scale applied after the 128-K dot
            ws = buffer_ops.buffer_load(ws_rsrc, nb * scale_k + kb, vec_width=1, dtype=T.f32)
            ws_vec = vector.from_elements(T.f32x4, [ws, ws, ws, ws])
            scale_vec = []
            for mm in range_constexpr(m_rep):
                base = xs_row0[mm] + kb
                xs0 = buffer_ops.buffer_load(xs_rsrc, base + 0 * scale_k, vec_width=1, dtype=T.f32)
                xs1 = buffer_ops.buffer_load(xs_rsrc, base + 1 * scale_k, vec_width=1, dtype=T.f32)
                xs2 = buffer_ops.buffer_load(xs_rsrc, base + 2 * scale_k, vec_width=1, dtype=T.f32)
                xs3 = buffer_ops.buffer_load(xs_rsrc, base + 3 * scale_k, vec_width=1, dtype=T.f32)
                xs_vec = vector.from_elements(T.f32x4, [xs0, xs1, xs2, xs3])
                scale_vec.append(arith.ArithValue(xs_vec) * arith.ArithValue(ws_vec))

            new_state = []
            for mm in range_constexpr(m_rep):
                for nn in range_constexpr(n_rep):
                    idx = mm * n_rep + nn
                    scaled = arith.ArithValue(partials[idx]) * arith.ArithValue(scale_vec[mm])
                    new_state.append(arith.ArithValue(masters[idx]) + arith.ArithValue(scaled))
            new_state.append(kb + 1)
            state = yield new_state

        masters = [state[i] for i in range_constexpr(n_acc)]
        for mm in range_constexpr(m_rep):
            for nn in range_constexpr(n_rep):
                idx = mm * n_rep + nn
                acc = masters[idx]
                col_base = n_base + nn * MFMA_N + n_in
                for i in range_constexpr(4):
                    val = vector.extract(acc, static_position=[i], dynamic_position=[])
                    out_val = arith.ArithValue(val).truncf(T.bf16)
                    out_row = out_row0[mm] + i
                    buffer_ops.buffer_store(out_val, out_rsrc, out_row * n + col_base)

    @flyc.jit
    def launch(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        XS: fx.Tensor,
        WS: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(Out, X, W, XS, WS).launch(
            grid=(grid, 1, 1),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=64)
def _compile_for(m, n, k):
    m = int(m); n = int(n); k = int(k)
    if m <= MFMA_M:
        m_rep, n_rep = 1, 1
    else:
        m_rep, n_rep = M_REP, N_REP
    while (n % (n_rep * MFMA_N)) != 0:
        n_rep -= 1
    return _compile(m, n, k, m_rep, n_rep)


_DEFAULT_STREAM = fx.Stream(None)


def gemm(x, w, x_scale, w_scale, out=None):
    """Y = X @ W.T with 128x128 fp8 block-scale dequant, fp32 accumulate, bf16 out."""
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("expected x[M,K] and w[N,K] with matching K")
    m, k = x.shape
    n = w.shape[0]

    if not x.is_contiguous():
        x = x.contiguous()
    if not w.is_contiguous():
        w = w.contiguous()
    if not x_scale.is_contiguous():
        x_scale = x_scale.contiguous()
    if not w_scale.is_contiguous():
        w_scale = w_scale.contiguous()

    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

    launch = _compile_for(int(m), int(n), int(k))
    launch(out, x, w, x_scale, w_scale, _DEFAULT_STREAM)
    return out
