"""FlyDSL fp8 block-scale GEMM -- fused MFMA + LDS multi-wave staging.

Math contract (matches the frozen aiter-Triton denominator gemm_a8w8_blockscale):

    Y[M,N] = (X[M,K] * x_scale) @ (W[N,K] * w_scale)^T

with 128x128 fp8 (e4m3fnuz) block-scale dequant, fp32 accumulation, bf16 output.
x_scale is [M, ceil(K/128)] (per-row-token, per-128-K-block); w_scale is
[ceil(N/128), ceil(K/128)] (per-128-N-block, per-128-K-block).

Two device kernels:
  * decode (M small): 1 wave/block, register-only 1x1..BMxBN tiles. Untouched.
  * prefill (M large): MEMORY lane. W waves per workgroup share the B (weight)
    K-tile through LDS. B[N-tile, K-tile] is cooperatively read from HBM ONCE per
    workgroup and reused across the workgroup's whole M range (W*16*BM rows),
    cutting redundant B HBM traffic ~W-fold. A stays wave-private (global loads,
    OOB-masked). Optional XOR16 LDS swizzle (applied identically at B store and B
    fragment load). Software fp32 per-128-K-block scale applied AFTER the MFMA
    accumulate (gfx942 parity gate -- no native scaled-MFMA).

The MFMA 16x16x32 fp8 fragment layout (lane l in 0..63): g=l//16, idx=l%16.
  A/B frag: 8 fp8 for row idx, K = kblk + kstep*32 + g*8 + [0..8).
  C/D frag: f32x4, value r -> C[row = tile_row + g*4 + r, col = tile_col + idx].
"""

import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref as _memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector, range_constexpr, const_expr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import get_rocm_arch as _get_arch
except Exception:  # pragma: no cover
    _get_arch = None

import functools as _functools

_DECODE_BLOCK_K = 128
_DECODE_BLOCK_N = 128


# ---------------------------------------------------------------------------
# decode kernel (M<=64): the proven seed kernel — 1 wave, one 16x16 tile/block,
# fp8 read 8-at-a-time as i64 straight to MFMA, HW OOB masking (max_size=True).
# ---------------------------------------------------------------------------
@_functools.lru_cache(maxsize=None)
def _build_decode(M, N, K, SK):
    @flyc.kernel(known_block_size=[64, 1, 1])
    def kern(X: fx.Tensor, W: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor, Y: fx.Tensor):
        vec4f = T.vec(4, T.f32)
        t = fx.thread_idx.x
        lane16 = t % 16
        laned = t // 16
        by = fx.block_idx.x
        bm = fx.block_idx.y
        m0 = bm * 16
        n0 = by * 16
        nb = n0 // _DECODE_BLOCK_N

        Xr = buffer_ops.create_buffer_resource(X, max_size=True)
        Wr = buffer_ops.create_buffer_resource(W, max_size=True)
        XSr = buffer_ops.create_buffer_resource(XS, max_size=True)
        WSr = buffer_ops.create_buffer_resource(WS, max_size=True)
        Yr = buffer_ops.create_buffer_resource(Y, max_size=True)

        acc0 = arith.constant(0.0, type=T.f32)
        accs_init = [acc0, acc0, acc0, acc0]
        NKB = K // _DECODE_BLOCK_K
        K8 = K // 8
        for kb, state in range(0, NKB, 1, init=accs_init):
            main = list(state)
            kbi = arith.index_cast(T.i32, kb)
            bacc = arith.constant_vector(0.0, vec4f)
            for ks in range_constexpr(4):
                koff = kbi * (_DECODE_BLOCK_K // 8) + ks * 4 + laned
                ai = buffer_ops.buffer_load(Xr, (m0 + lane16) * K8 + koff, vec_width=1, dtype=T.i64)
                bi = buffer_ops.buffer_load(Wr, (n0 + lane16) * K8 + koff, vec_width=1, dtype=T.i64)
                bacc = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(vec4f, [ai, bi, bacc])
            ws = buffer_ops.buffer_load(WSr, nb * SK + kbi, vec_width=1, dtype=T.f32)
            for ii in range_constexpr(4):
                row = m0 + laned * 4 + ii
                xs = buffer_ops.buffer_load(XSr, row * SK + kbi, vec_width=1, dtype=T.f32)
                sc = arith.mulf(xs, ws)
                part = arith.mulf(vector.extract(bacc, static_position=[ii], dynamic_position=[]), sc)
                main[ii] = arith.addf(main[ii], part)
            res = yield main
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
            grid=(N // 16, (M + 15) // 16, 1), block=(64, 1, 1), stream=stream)
    return launch


# ---------------------------------------------------------------------------
# decode kernel: 1 wave/block, register-only (unchanged from the MFMA base)
# ---------------------------------------------------------------------------
@flyc.kernel
def _gemm_fp8_mfma(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor,
    M: fx.Int32, N: fx.Int32, K: fx.Int32, KB: fx.Int32, NB: fx.Int32,
    BM: fx.Constexpr[int], BN: fx.Constexpr[int],
):
    tx = fx.thread_idx.x
    bn = fx.block_idx.x
    bm = fx.block_idx.y

    idx = tx % 16
    g = tx // 16

    m_tile = bm * (16 * BM)
    n_tile = bn * (16 * BN)

    ra = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=(M * K))
    rb = buffer_ops.create_buffer_resource(B, max_size=False, num_records_bytes=(N * K))
    rc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=(M * N * 2))
    rxs = buffer_ops.create_buffer_resource(XS, max_size=False, num_records_bytes=(M * KB * 4))
    rws = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=(NB * KB * 4))

    K8 = K // 8

    aoff8 = [(m_tile + mi * 16 + idx) * K8 for mi in range(BM)]
    boff8 = [(n_tile + ni * 16 + idx) * K8 for ni in range(BN)]
    nblk = [(n_tile + ni * 16) // 128 for ni in range(BN)]

    z = arith.constant(0.0, type=T.f32)
    init = [z] * (BM * BN * 4)

    for kb, carried in range(0, KB, 1, init=init):
        rr = [carried[i] for i in range(BM * BN * 4)]
        kb_i = arith.index_cast(T.i32, kb)
        kb16 = kb_i * 16

        acc = [arith.constant_vector(0.0, T.f32x4) for _ in range(BM * BN)]
        for kk in range_constexpr(4):
            koff = kb16 + (kk * 4) + g
            af = [buffer_ops.buffer_load(ra, aoff8[mi] + koff, vec_width=1, dtype=T.i64)
                  for mi in range(BM)]
            bf = [buffer_ops.buffer_load(rb, boff8[ni] + koff, vec_width=1, dtype=T.i64)
                  for ni in range(BN)]
            for mi in range_constexpr(BM):
                for ni in range_constexpr(BN):
                    ai = mi * BN + ni
                    acc[ai] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [af[mi], bf[ni], acc[ai], 0, 0, 0])

        sB = [buffer_ops.buffer_load(rws, nblk[ni] * KB + kb_i, vec_width=1, dtype=T.f32)
              for ni in range(BN)]
        new = list(rr)
        for mi in range_constexpr(BM):
            crow = m_tile + mi * 16 + g * 4
            for r in range_constexpr(4):
                sA = buffer_ops.buffer_load(rxs, (crow + r) * KB + kb_i, vec_width=1, dtype=T.f32)
                for ni in range_constexpr(BN):
                    ai = mi * BN + ni
                    ri = ai * 4 + r
                    acci = vector.extract(acc[ai], static_position=[r])
                    scaled = arith.mulf(arith.mulf(acci, sA), sB[ni])
                    new[ri] = arith.addf(rr[ri], scaled)
        res = yield new

    for mi in range_constexpr(BM):
        crow = m_tile + mi * 16 + g * 4
        for ni in range_constexpr(BN):
            ncol = n_tile + ni * 16 + idx
            for r in range_constexpr(4):
                ri = (mi * BN + ni) * 4 + r
                cbf = arith.trunc_f(T.bf16, res[ri])
                buffer_ops.buffer_store(cbf, rc, (crow + r) * N + ncol)


@flyc.jit
def _gemm_launch(
    A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor,
    M: fx.Int32, N: fx.Int32, K: fx.Int32, KB: fx.Int32, NB: fx.Int32,
    grid_m: fx.Int32, grid_n: fx.Int32,
    BM: fx.Constexpr[int], BN: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    _gemm_fp8_mfma(A, B, C, XS, WS, M, N, K, KB, NB, BM, BN).launch(
        grid=(grid_n, grid_m, 1), block=(64, 1, 1), stream=stream
    )


# ---------------------------------------------------------------------------
# prefill kernel builder: W waves/workgroup, B staged through LDS (per config)
# ---------------------------------------------------------------------------
_PREFILL_CACHE = {}


def _swz(row_i, col_i, k16_c, do_swz):
    # XOR16 byte-column transform in i64 units.  col is in i64 units; XOR16 acts
    # on 16-byte columns => 2 i64 per 16B => (row % k16)*2 in i64 units.
    if not do_swz:
        return col_i
    return arith.xori(col_i, arith.andi(row_i, k16_c) * 2)


def _build_prefill(BM, BN, W, KBpT, do_swz, arch, stage=1, do_vec=False):
    key = (BM, BN, W, KBpT, do_swz, arch, stage, do_vec)
    if key in _PREFILL_CACHE:
        return _PREFILL_CACHE[key]

    TK8 = KBpT * 16            # i64 per row per K-tile
    BROWS = BN * 16            # B rows staged per tile
    NLDS = BROWS * TK8         # logical i64 elements per LDS B buffer
    NTHREADS = 64 * W
    PER_THREAD = (NLDS + NTHREADS - 1) // NTHREADS
    # Pad the allocation so every cooperative store (tx + it*NTHREADS, it<PER_THREAD)
    # lands in-bounds -> no per-store mask needed. Padding rows are written (OOB
    # global reads return 0) but never read.
    NLDS_ALLOC = ((PER_THREAD * NTHREADS + TK8 - 1) // TK8) * TK8
    # k16 (# of 16B columns) for XOR16 on the LDS tile row = TK bytes / 16.
    K16 = (KBpT * 128) // 16   # = KBpT * 8

    allocator = SmemAllocator(None, arch=arch, global_sym_name="smemB")
    b_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = b_off + stage * NLDS_ALLOC * 8   # i64 = 8 bytes

    @flyc.kernel
    def _prefill_kernel(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor,
        M: fx.Int32, N: fx.Int32, K: fx.Int32, KB: fx.Int32, NB: fx.Int32,
    ):
        tx = fx.thread_idx.x            # 0 .. 64*W-1
        bn = fx.block_idx.x
        bm = fx.block_idx.y

        wave = tx // 64
        lane = tx % 64
        idx = lane % 16
        g = lane // 16

        n_tile = bn * (16 * BN)
        m_tile = bm * (W * 16 * BM) + wave * (16 * BM)

        ra = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=(M * K))
        rb = buffer_ops.create_buffer_resource(B, max_size=False, num_records_bytes=(N * K))
        rc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=(M * N * 2))
        rxs = buffer_ops.create_buffer_resource(XS, max_size=False, num_records_bytes=(M * KB * 4))
        rws = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=(NB * KB * 4))

        K8 = K // 8

        aoff8 = [(m_tile + mi * 16 + idx) * K8 for mi in range(BM)]
        nblk = [(n_tile + ni * 16) // 128 for ni in range(BN)]

        # LDS B buffer (i64 elements); stage buffers packed back-to-back.
        base_ptr = allocator.get_base()
        lds_b = SmemPtr(base_ptr, b_off, T.i64, shape=(stage * NLDS_ALLOC,))

        k16_c = arith.constant(K16 - 1, type=T.i32)  # (row % K16) via AND (K16 pow2)

        KTILES = KB // KBpT

        z = arith.constant(0.0, type=T.f32)
        init = [z] * (BM * BN * 4)

        def _coop_load(ktile8_i, buf_off_i):
            # cooperative HBM -> LDS load of B[n_tile:n_tile+BROWS, K-tile] into
            # the LDS buffer starting at i64 element buf_off_i.  OOB global reads
            # return 0 and OOB tile rows land in padding never read -> no mask.
            if const_expr(do_vec and (TK8 % 2 == 0)):
                # Vectorized 16B g->LDS: one buffer_load of vec2xi64 (16 bytes)
                # per thread halves the VMEM issue count.  Global base offset is
                # even (K8 even, col0 even) so 16B alignment holds; the two i64
                # lanes are stored to consecutive (swizzled) LDS slots.
                TK8H = TK8 // 2
                NPAIR = NLDS // 2
                PT2 = (NPAIR + NTHREADS - 1) // NTHREADS
                for it in range_constexpr(PT2):
                    pe = tx + it * NTHREADS
                    row = pe // TK8H
                    cp = pe % TK8H
                    col0 = cp * 2
                    gidx = (n_tile + row) * K8 + ktile8_i + col0
                    gv = buffer_ops.buffer_load(rb, gidx, vec_width=2, dtype=T.i64)
                    g0 = vector.extract(gv, static_position=[0], dynamic_position=[])
                    g1 = vector.extract(gv, static_position=[1], dynamic_position=[])
                    swz0 = _swz(row, col0, k16_c, do_swz)
                    swz1 = _swz(row, col0 + 1, k16_c, do_swz)
                    base = buf_off_i + row * TK8
                    lds_b.store(g0, [arith.index_cast(T.index, base + swz0)])
                    lds_b.store(g1, [arith.index_cast(T.index, base + swz1)])
            else:
                for it in range_constexpr(PER_THREAD):
                    ee = tx + it * NTHREADS
                    row = ee // TK8
                    col = ee % TK8                       # i64 col within tile
                    gidx = (n_tile + row) * K8 + ktile8_i + col
                    gval = buffer_ops.buffer_load(rb, gidx, vec_width=1, dtype=T.i64)
                    swz = _swz(row, col, k16_c, do_swz)
                    lidx = buf_off_i + row * TK8 + swz
                    lds_b.store(gval, [arith.index_cast(T.index, lidx)])

        def _compute(cur_off_i, kt_i, ktile8, rr):
            new = list(rr)
            for kbb in range_constexpr(KBpT):
                kb_i = kt_i * KBpT + kbb
                acc = [arith.constant_vector(0.0, T.f32x4) for _ in range(BM * BN)]
                for kk in range_constexpr(4):
                    a_koff = ktile8 + kbb * 16 + kk * 4 + g
                    af = [buffer_ops.buffer_load(ra, aoff8[mi] + a_koff, vec_width=1, dtype=T.i64)
                          for mi in range(BM)]
                    b_col = kbb * 16 + kk * 4 + g   # i32 (g runtime)
                    bf = []
                    for ni in range_constexpr(BN):
                        brow = ni * 16 + idx           # i32 (idx runtime)
                        swz = _swz(brow, b_col, k16_c, do_swz)
                        lidx = cur_off_i + brow * TK8 + swz
                        bval = lds_b.load([arith.index_cast(T.index, lidx)])
                        bf.append(bval)
                    for mi in range_constexpr(BM):
                        for ni in range_constexpr(BN):
                            ai = mi * BN + ni
                            acc[ai] = rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [af[mi], bf[ni], acc[ai], 0, 0, 0])

                sB = [buffer_ops.buffer_load(rws, nblk[ni] * KB + kb_i, vec_width=1, dtype=T.f32)
                      for ni in range(BN)]
                for mi in range_constexpr(BM):
                    crow = m_tile + mi * 16 + g * 4
                    for r in range_constexpr(4):
                        sA = buffer_ops.buffer_load(rxs, (crow + r) * KB + kb_i, vec_width=1, dtype=T.f32)
                        for ni in range_constexpr(BN):
                            ai = mi * BN + ni
                            ri = ai * 4 + r
                            acci = vector.extract(acc[ai], static_position=[r])
                            scaled = arith.mulf(arith.mulf(acci, sA), sB[ni])
                            new[ri] = arith.addf(new[ri], scaled)
            return new

        if const_expr(stage == 1):
            for kt, carried in range(0, KTILES, 1, init=init):
                rr = [carried[i] for i in range(BM * BN * 4)]
                kt_i = arith.index_cast(T.i32, kt)
                ktile8 = kt_i * TK8
                _coop_load(ktile8, arith.constant(0, type=T.i32))
                gpu.barrier()
                new = _compute(arith.constant(0, type=T.i32), kt_i, ktile8, rr)
                gpu.barrier()
                res = yield new
        else:
            # ping-pong double buffer: prologue loads tile 0 into buffer 0, then
            # each iter prefetches tile kt+1 into the OTHER buffer while computing
            # on the current one.  cur/nxt buffer chosen at runtime (kt is a
            # runtime scf.for var, cannot index a Python list).
            NA = arith.constant(NLDS_ALLOC, type=T.i32)
            one = arith.constant(1, type=T.i32)
            _coop_load(arith.constant(0, type=T.i32), arith.constant(0, type=T.i32))
            gpu.barrier()
            for kt, carried in range(0, KTILES, 1, init=init):
                rr = [carried[i] for i in range(BM * BN * 4)]
                kt_i = arith.index_cast(T.i32, kt)
                ktile8 = kt_i * TK8
                cur_off = arith.andi(kt_i, one) * NA
                nxt_off = arith.andi(arith.addi(kt_i, one), one) * NA
                # prefetch next K-tile (OOB-safe; padding rows for the last tile)
                _coop_load(arith.addi(ktile8, arith.constant(TK8, type=T.i32)), nxt_off)
                new = _compute(cur_off, kt_i, ktile8, rr)
                gpu.barrier()
                res = yield new

        for mi in range_constexpr(BM):
            crow = m_tile + mi * 16 + g * 4
            for ni in range_constexpr(BN):
                ncol = n_tile + ni * 16 + idx
                for r in range_constexpr(4):
                    ri = (mi * BN + ni) * 4 + r
                    cbf = arith.trunc_f(T.bf16, res[ri])
                    buffer_ops.buffer_store(cbf, rc, (crow + r) * N + ncol)

    @flyc.jit
    def _prefill_launch(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, XS: fx.Tensor, WS: fx.Tensor,
        M: fx.Int32, N: fx.Int32, K: fx.Int32, KB: fx.Int32, NB: fx.Int32,
        grid_m: fx.Int32, grid_n: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        _prefill_kernel(A, B, C, XS, WS, M, N, K, KB, NB).launch(
            grid=(grid_n, grid_m, 1), block=(64 * W, 1, 1), stream=stream
        )

    _PREFILL_CACHE[key] = _prefill_launch
    return _prefill_launch


def _detect_arch():
    if _get_arch is not None:
        try:
            return str(_get_arch())
        except Exception:
            pass
    return "gfx942"


def gemm(x, w, x_scale, w_scale, out=None):
    """Drop-in for the frozen baseline: Y = (X*x_scale) @ (W*w_scale)^T, bf16."""
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("expected x[M,K] and w[N,K]")
    M, K = x.shape
    N = w.shape[0]
    KB = (K + 127) // 128
    NB = (N + 127) // 128

    x = x.contiguous()
    w = w.contiguous()
    x_scale = x_scale.contiguous().to(torch.float32)
    w_scale = w_scale.contiguous().to(torch.float32)

    if out is None:
        C = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    else:
        C = out

    if M <= 64:
        # decode: proven seed kernel (1 wave, 16x16 tile, HW OOB masking).
        SK = x_scale.shape[1]
        _build_decode(M, N, K, SK)(
            x.view(-1), w.view(-1), x_scale.view(-1), w_scale.view(-1), C.view(-1),
            stream=torch.cuda.current_stream(),
        )
        return C
    else:
        # prefill: LDS multi-wave staging.
        BM, BN, W = _PREFILL_BM, _PREFILL_BN, _PREFILL_W
        KBpT = _PREFILL_KBPT
        do_swz = _PREFILL_SWZ
        arch = _detect_arch()
        launch = _build_prefill(BM, BN, W, KBpT, do_swz, arch, _PREFILL_STAGE, _PREFILL_VEC)
        tm = W * 16 * BM
        tn = 16 * BN
        grid_n = (N + tn - 1) // tn
        grid_m = (M + tm - 1) // tm
        launch(
            x, w, C, x_scale, w_scale, M, N, K, KB, NB, grid_m, grid_n,
            stream=torch.cuda.current_stream(),
        )

    return C


# tunables (overridable via env for sweeps; defaults are the shipped config)
import os as _os
_PREFILL_KBPT = int(_os.environ.get("R2D1_KBPT", "1"))
_PREFILL_SWZ = _os.environ.get("R2D1_SWZ", "1") == "1"
_PREFILL_W = int(_os.environ.get("R2D1_W", "4"))
_PREFILL_BM = int(_os.environ.get("R2D1_BM", "2"))
_PREFILL_BN = int(_os.environ.get("R2D1_BN", "8"))
_PREFILL_STAGE = int(_os.environ.get("R2D1_STAGE", "2"))
_PREFILL_VEC = _os.environ.get("R2D1_VEC", "1") == "1"
