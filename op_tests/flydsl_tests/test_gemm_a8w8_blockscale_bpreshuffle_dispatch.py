import torch
import pytest
from packaging.version import Version

import flydsl

pytestmark = pytest.mark.skipif(
    Version(flydsl.__version__.split("+")[0]) < Version("0.2.4"),
    reason="aiter.ops.flydsl currently requires flydsl>=0.2.4",
)


def _gemm_kernels():
    from aiter.ops.flydsl import gemm_kernels

    return gemm_kernels


def test_gemm_a8w8_blockscale_bpreshuffle_supports_prefill_shapes():
    # One op (A8W8 block-scale bpreshuffle GEMM); shapes are supported or not.
    # We check support by shape, not which kernel is selected underneath.
    gemm_kernels = _gemm_kernels()

    get_config = gemm_kernels.get_flydsl_gemm_a8w8_blockscale_bpreshuffle_config
    assert get_config(256, 2048, 7168) is not None
    assert get_config(256, 65536, 1536) is not None
    assert get_config(1024, 7168, 768) is not None


def test_gemm_a8w8_blockscale_bpreshuffle_dispatch_is_shape_and_m_aware():
    # The implementation is chosen purely from (M, N, K). The same shape can
    # resolve to different implementations across M -- assert the plan changes,
    # without caring which kernel it names.
    gemm_kernels = _gemm_kernels()

    plan_small = gemm_kernels._select_gemm_plan(64, 2048, 7168)
    plan_mid = gemm_kernels._select_gemm_plan(256, 2048, 7168)
    assert plan_small is not None and plan_mid is not None
    assert plan_small != plan_mid


def test_gemm_a8w8_blockscale_bpreshuffle_rejects_uncovered_shapes():
    gemm_kernels = _gemm_kernels()

    get_config = gemm_kernels.get_flydsl_gemm_a8w8_blockscale_bpreshuffle_config
    assert get_config(256, 4096, 7168) is None


def test_gemm_a8w8_blockscale_bpreshuffle_validates_output_dtype():
    gemm_kernels = _gemm_kernels()

    q = torch.empty((1, 7168), dtype=torch.float8_e4m3fn)
    w = torch.empty((2048, 7168), dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((56, 1), dtype=torch.float32)
    w_scale = torch.empty((16, 56), dtype=torch.float32)

    try:
        gemm_kernels.flydsl_gemm_a8w8_blockscale_bpreshuffle(
            q, w, x_scale, w_scale, dtype=torch.float32
        )
    except ValueError as exc:
        assert "unsupported output dtype" in str(exc)
    else:
        raise AssertionError("expected unsupported dtype to raise")
