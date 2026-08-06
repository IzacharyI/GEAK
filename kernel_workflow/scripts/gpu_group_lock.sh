#!/bin/bash
# Fixed or dynamic GPU-group lease wrapper. Single-GPU callers use gpu_lock.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_IDS=""
POOL_IDS=""
GPU_COUNT=""
WAIT_TIMEOUT="${GEAK_GPU_WAIT_TIMEOUT:-1200}"
RUN_TIMEOUT="${GEAK_GPU_RUN_TIMEOUT:-900}"
TERM_GRACE="${GEAK_GPU_TERM_GRACE:-5}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --group)
            GROUP_IDS="${2:?--group requires a comma-separated GPU list}"
            shift 2
            ;;
        --pool)
            POOL_IDS="${2:?--pool requires a comma-separated GPU list}"
            shift 2
            ;;
        --count)
            GPU_COUNT="${2:?--count requires a positive integer}"
            shift 2
            ;;
        --wait-timeout)
            WAIT_TIMEOUT="${2:?--wait-timeout requires seconds}"
            shift 2
            ;;
        --run-timeout)
            RUN_TIMEOUT="${2:?--run-timeout requires seconds}"
            shift 2
            ;;
        --term-grace)
            TERM_GRACE="${2:?--term-grace requires seconds}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "ERROR: unknown gpu_group_lock.sh option: $1" >&2
            exit 2
            ;;
    esac
done

if [ -n "$GROUP_IDS" ] && [ -n "$POOL_IDS" ]; then
    echo "ERROR: use exactly one of --group or --pool" >&2
    exit 2
fi
if [ -z "$GROUP_IDS" ] && [ -z "$POOL_IDS" ]; then
    echo "ERROR: one of --group or --pool is required" >&2
    exit 2
fi
if [ -n "$POOL_IDS" ] && [ -z "$GPU_COUNT" ]; then
    echo "ERROR: --count is required with --pool" >&2
    exit 2
fi
[ "$#" -gt 0 ] || { echo "ERROR: command is required after --" >&2; exit 2; }

# Preserve gpu_lock.sh's orphan-enumerator backstop for the new group path.
if [ "${KERNEL_ENV_SKIP_ENUM_REAP:-0}" != "1" ]; then
    for _p in $(pgrep -f rocm_agent_enumerator 2>/dev/null || true); do
        _pp="$(ps -o ppid= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        _et="$(ps -o etimes= -p "$_p" 2>/dev/null | tr -d ' ' || true)"
        if [ "${_pp:-0}" = "1" ] && [ -n "${_et:-}" ] && [ "${_et:-0}" -gt 60 ] 2>/dev/null; then
            kill -9 "$_p" 2>/dev/null || true
        fi
    done
fi

# Preserve per-workspace torch extension cache isolation.
: "${TORCH_EXTENSIONS_DIR:=$PWD/.torch_ext}"
export TORCH_EXTENSIONS_DIR
mkdir -p "$TORCH_EXTENSIONS_DIR" 2>/dev/null || true

# Preserve the existing local-architecture compile optimization.
if [ "${KERNEL_ENV_KEEP_ARCH:-0}" != "1" ]; then
    _ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
    [ -n "${_ARCH:-}" ] && export PYTORCH_ROCM_ARCH="$_ARCH"
    [ -n "${_ARCH:-}" ] && export GPU_ARCHS="${GPU_ARCHS:-$_ARCH}"
fi

IDLE_ARGS=()
if [ "${GEAK_GPU_REQUIRE_IDLE:-1}" = "1" ]; then
    IDLE_ARGS+=(
        --require-idle
        --sysfs-root "${GEAK_GPU_SYSFS_ROOT:-/sys/class/drm}"
        --max-busy-pct "${GEAK_GPU_MAX_BUSY_PCT:-5}"
        --max-vram-mb "${GEAK_GPU_MAX_VRAM_MB:--1}"
    )
fi

REQUEST_ARGS=()
if [ -n "$GROUP_IDS" ]; then
    REQUEST_ARGS+=(--fixed-ids "$GROUP_IDS")
    [ -n "$GPU_COUNT" ] && REQUEST_ARGS+=(--count "$GPU_COUNT")
else
    REQUEST_ARGS+=(--pool "$POOL_IDS" --count "$GPU_COUNT")
fi

exec python3 "$SCRIPT_DIR/gpu_lease.py" run \
    "${REQUEST_ARGS[@]}" \
    --lock-dir "${GEAK_GPU_LOCK_DIR:-/tmp/team_gpu_locks}" \
    --wait-timeout "$WAIT_TIMEOUT" \
    --run-timeout "$RUN_TIMEOUT" \
    --term-grace "$TERM_GRACE" \
    "${IDLE_ARGS[@]}" \
    -- "$@"
