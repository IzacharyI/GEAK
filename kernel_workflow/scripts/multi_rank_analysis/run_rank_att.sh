#!/bin/bash
set -euo pipefail

target_rank="${GEAK_ATT_RANK:-0}"
local_rank="${LOCAL_RANK:-${RANK:-}}"
if [ -z "$local_rank" ]; then
    echo "GEAK ATT wrapper requires LOCAL_RANK or RANK" >&2
    exit 2
fi

if [ "$local_rank" != "$target_rank" ]; then
    exec "$@"
fi

: "${GEAK_ATT_OUTPUT_DIR:?GEAK_ATT_OUTPUT_DIR is required}"
: "${GEAK_ATT_KERNEL_REGEX:?GEAK_ATT_KERNEL_REGEX is required}"

decoder_args=()
if [ -n "${GEAK_ATT_LIBRARY_PATH:-}" ]; then
    decoder_args=(--att-library-path "$GEAK_ATT_LIBRARY_PATH")
fi

exec rocprofv3 \
    --att \
    "${decoder_args[@]}" \
    --att-gpu-index "${GEAK_ATT_GPU_INDEX:-0}" \
    --att-target-cu "${GEAK_ATT_TARGET_CU:-1}" \
    --att-simd-select "${GEAK_ATT_SIMD_SELECT:-0xF}" \
    --kernel-include-regex "$GEAK_ATT_KERNEL_REGEX" \
    --output-directory "$GEAK_ATT_OUTPUT_DIR" \
    --output-file "${GEAK_ATT_OUTPUT_NAME:-att}" \
    --output-format csv \
    -- "$@"
