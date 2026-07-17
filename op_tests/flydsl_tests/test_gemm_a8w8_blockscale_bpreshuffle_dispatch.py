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


def test_gemm_a8w8_blockscale_bpreshuffle_configs_cover_prefill_shapes():
    gemm_kernels = _gemm_kernels()

    get_config = gemm_kernels.get_flydsl_gemm_a8w8_blockscale_bpreshuffle_config
    assert get_config(256, 2048, 7168)["kind"] == "8wave_blockscale"
    assert get_config(256, 65536, 1536)["kind"] == "blockscale_preshuffle"
    assert get_config(1024, 7168, 768)["kind"] == "blockscale_preshuffle"


def test_gemm_a8w8_blockscale_bpreshuffle_dispatch_is_m_aware():
    # qkv / q_up / mlp are the same GEMM; the implementation is picked per
    # (M, N, K), not fixed per shape. The qkv shape (N=2048, K=7168) is the
    # clearest case: small mid-M dispatches to the small-tile preshuffle kernel
    # while larger M uses the 8-wave (split-K) kernel.
    gemm_kernels = _gemm_kernels()

    get_config = gemm_kernels.get_flydsl_gemm_a8w8_blockscale_bpreshuffle_config
    assert get_config(64, 2048, 7168)["kind"] == "blockscale_preshuffle"
    assert get_config(256, 2048, 7168)["kind"] == "8wave_blockscale"

    # The same selection is exposed through the internal planner.
    assert gemm_kernels._select_gemm_plan(64, 2048, 7168)[0] == "preshuffle"
    assert gemm_kernels._select_gemm_plan(256, 2048, 7168)[0] == "8wave"


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
