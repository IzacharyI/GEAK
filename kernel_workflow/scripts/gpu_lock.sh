#!/bin/bash
# GPU lock + per-workspace build isolation wrapper.
# Usage:  cd <workspace> && bash gpu_lock.sh <gpu_id> <command...>
#
# Run EVERY kernel command (compile / correctness / benchmark / profile) through this wrapper,
# invoked from inside the workspace directory. It does three generic things — none kernel-specific:
#
#  1. flock per GPU id  -> multiple engineers can share GPUs safely (exclusive during the command).
#  2. TORCH_EXTENSIONS_DIR = <workspace>/.torch_ext  -> isolates the torch cpp_extension build cache
#     PER WORKSPACE. Without this, torch.utils.cpp_extension.load(name=...) compiles every engineer's
#     DIFFERENT source into ONE global cache (~/.cache/torch_extensions/...), which both serializes
#     all parallel compiles on a single global lock AND lets one engineer benchmark another's .so.
#     Deriving it from $PWD makes each isolated workspace get its own cache. (Honors a caller-set
#     TORCH_EXTENSIONS_DIR if already exported.)
#  3. PYTORCH_ROCM_ARCH = the local GPU's gfx arch only -> avoids compiling for ~9 architectures
#     (huge compile speedup). Runtime perf and correctness are unaffected (the kernel runs on the
#     local arch either way). Honors a caller-set PYTORCH_ROCM_ARCH if already exported.

set -euo pipefail

if [ "${GEAK_GPU_LEASE_ACTIVE:-0}" = "1" ]; then
    echo "ERROR: nested GPU lease request via gpu_lock.sh is not supported" >&2
    exit 2
fi

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "--group" ] || [ "${1:-}" = "--pool" ]; then
    exec bash "$_SCRIPT_DIR/gpu_group_lock.sh" "$@"
fi

case "${1:-}" in
    group:*)
        _GROUP_IDS="${1#group:}"
        shift
        [ "${1:-}" = "--" ] && shift
        exec bash "$_SCRIPT_DIR/gpu_group_lock.sh" --group "$_GROUP_IDS" -- "$@"
        ;;
    pool:*)
        _POOL_SPEC="${1#pool:}"
        _POOL_COUNT="${_POOL_SPEC%%:*}"
        _POOL_IDS="${_POOL_SPEC#*:}"
        shift
        [ "${1:-}" = "--" ] && shift
        exec bash "$_SCRIPT_DIR/gpu_group_lock.sh" \
            --pool "$_POOL_IDS" --count "$_POOL_COUNT" -- "$@"
        ;;
esac

GPU_ID="${1:?Usage: gpu_lock.sh <gpu_id> <command...>}"
shift

# Keep the historical numeric interface, but route it through the same lease manager as
# multi-GPU requests so stale process-group metadata cannot be bypassed. Single-GPU mode
# preserves the old no-idle-gate and no-run-timeout defaults unless callers opt in.
export GEAK_GPU_REQUIRE_IDLE="${GEAK_GPU_REQUIRE_IDLE:-0}"
exec bash "$_SCRIPT_DIR/gpu_group_lock.sh" \
    --group "$GPU_ID" \
    --run-timeout "${GEAK_GPU_RUN_TIMEOUT:--1}" \
    -- "$@"
