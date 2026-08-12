"""Generic multi-GPU-rank analysis primitives for GEAK Kernel Workflow.

This package is deliberately operator-agnostic: it must never import or reference a specific
kernel, model, or backend (no MegaMoE/AITER/MORI/FlyDSL names). It exists so that any distributed
UT/benchmark's per-rank timing and profiling output can be merged into a stable, versioned report
shape, and so that operator-specific advisory Skills (see
``kernel_workflow/knowledge/analysis_skills/``) have a shared, tested foundation to build on instead
of re-deriving all_gather/mean/max plumbing per operator.

See README.md in this directory for the full design rationale and the contract each module honors.
"""

from .aggregate import (
    all_ranks_true,
    classify_case_speedup,
    merge_rank_records,
    reduce_scalar,
    time_distributed,
)
from .schema import SCHEMA_VERSION, build_report
from .trace_categories import bucket_trace_events, load_category_map

__all__ = [
    "SCHEMA_VERSION",
    "all_ranks_true",
    "bucket_trace_events",
    "build_report",
    "classify_case_speedup",
    "load_category_map",
    "merge_rank_records",
    "reduce_scalar",
    "time_distributed",
]
