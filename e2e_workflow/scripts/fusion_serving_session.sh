#!/usr/bin/env bash
# KernelFusion serving session: ONE server on the run's serving config, ONE command against it.
#
# The kernel-extract skill (knowledge/kernel_extraction_skills/kernel-extract/references/capture.md)
# has no universal server-start command: a probe writes a task-local launcher adapted to the
# user's actual serving launch, owns the server for the whole client lifecycle in the
# foreground, and stops exactly what it started. This is that launcher for KernelFusion. It does
# not re-implement the launch: it sources the SAME adapter bench_e2e.sh uses
# (adapters/<BACKEND>.sh, plus adapters/launchers/<BENCH_LAUNCHER>.sh when one is set) and the
# shared identity-verified teardown (server_teardown.sh), so the served stack is the one every
# GEAK measurement runs.
#
#   OUT_DIR=<dir> SESSION_CMD='<shell command>' \
#   BACKEND=sglang MODEL=<path> TP=8 GPU=0,..,7 EXTRA_SERVER_ARGS=.. EXTRA_ENV=.. \
#   [OVERLAY_PYTHONPATH=..] [MEM_FRACTION=..] [PORT=..] bash fusion_serving_session.sh
#
# SESSION_CMD runs while the server is up, with BASE_URL HOST PORT MODEL OUT_DIR LOG PROFILE_DIR
# exported. A bench_e2e.sh call inside it must pass REUSE_SERVER=1 (it then measures this server
# and leaves its lifecycle alone). The serving-GPU lock is held for the whole session, exactly as
# bench_e2e.sh holds it around its own server.
#
# Writes $OUT_DIR/server.log, $OUT_DIR/server_start.json and $OUT_DIR/session_cmd.rc. Exit code:
# SESSION_CMD's, or 2 when the server never became healthy (SESSION_CMD then does not run).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT_DIR:?OUT_DIR is required}"
: "${SESSION_CMD:?SESSION_CMD is required}"
: "${MODEL:?MODEL is required}"
mkdir -p "$OUT_DIR"

BACKEND=${BACKEND:-sglang}
HOST=${HOST:-127.0.0.1}
TP=${TP:-1}
GPU=${GPU:-0}
MEM_FRACTION=${MEM_FRACTION:-0.9}
EXTRA_SERVER_ARGS=${EXTRA_SERVER_ARGS:-}
EXTRA_ENV=${EXTRA_ENV:-}
OVERLAY_PYTHONPATH=${OVERLAY_PYTHONPATH:-}
LOG=${LOG:-$OUT_DIR/server.log}
PROFILE=${PROFILE:-0}
PROFILE_DIR=${PROFILE_DIR:-$OUT_DIR/profile}
PROFILE_PYTHONPATH=""
mkdir -p "$PROFILE_DIR"

ADAPTER="${ADAPTER:-$HERE/adapters/${BACKEND}.sh}"
[ -f "$ADAPTER" ] || { echo "!!! no adapter for BACKEND=$BACKEND at $ADAPTER" >&2; exit 3; }
# shellcheck disable=SC1090
source "$ADAPTER"
copy_function() { declare -F "$1" >/dev/null || return 1; eval "${2}() $(declare -f "$1" | sed '1d')"; }
BENCH_LAUNCHER=${BENCH_LAUNCHER:-native}
if [ "$BENCH_LAUNCHER" != "native" ]; then
  LAUNCHER_ADAPTER="${LAUNCHER_ADAPTER:-$HERE/adapters/launchers/${BENCH_LAUNCHER}.sh}"
  [ -f "$LAUNCHER_ADAPTER" ] || { echo "!!! no server launcher '$BENCH_LAUNCHER' at $LAUNCHER_ADAPTER" >&2; exit 3; }
  copy_function adapter_launch adapter_launch_native
  copy_function adapter_health adapter_health_native
  # shellcheck disable=SC1090
  source "$LAUNCHER_ADAPTER"
fi

if [ -n "${ALLOWED_GPUS:-}" ]; then
  _allow=",$(echo "$ALLOWED_GPUS" | tr -d ' '),"
  for _g in $(echo "$GPU" | tr ',' ' '); do
    case "$_allow" in *",$_g,"*) : ;; *) echo "!!! GPU '$_g' not in ALLOWED_GPUS='$ALLOWED_GPUS'" >&2; exit 5 ;; esac
  done
fi

if [ -z "${PORT:-}" ]; then
  PORT=$(PORT_BASE="${PORT_BASE:-30000}" PORT_SPAN="${PORT_SPAN:-1000}" python3 - <<'PY' 2>/dev/null || true
import os, random, socket
base = int(os.environ["PORT_BASE"]); span = int(os.environ["PORT_SPAN"])
order = list(range(span)); random.shuffle(order)
for off in order:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", base + off)); s.close(); print(base + off); break
    except OSError:
        s.close()
PY
)
fi
[ -n "$PORT" ] || { echo "!!! no free port" >&2; exit 4; }
BASE_URL="http://${HOST}:${PORT}"
export MODEL HOST PORT TP GPU MEM_FRACTION EXTRA_SERVER_ARGS EXTRA_ENV OVERLAY_PYTHONPATH
export PROFILE PROFILE_DIR PROFILE_PYTHONPATH BASE_URL LOG OUT_DIR

TEARDOWN_LIB="$HERE/server_teardown.sh"
[ -f "$TEARDOWN_LIB" ] || { echo "!!! $TEARDOWN_LIB missing: refusing to start a server nothing can stop" >&2; exit 3; }
# shellcheck disable=SC1090
source "$TEARDOWN_LIB"
trap server_teardown EXIT

# Same serving-GPU lock bench_e2e.sh takes, held across the whole session.
if [ "${SERVING_GPU_LOCK_DISABLE:-0}" != "1" ]; then
  _gpu_key="${GPU//,/_}"
  SERVING_LOCK="${SERVING_GPU_LOCK:-/tmp/geak_serving_gpu_${_gpu_key}.lock}"
  exec {SERVING_LOCK_FD}>"$SERVING_LOCK"
  echo ">>> session: acquiring serving-GPU lock ($SERVING_LOCK) for GPU=$GPU ..."
  flock -w "${SERVING_LOCK_WAIT:-7200}" "$SERVING_LOCK_FD" || { echo "!!! serving-GPU lock timeout" >&2; exit 4; }
fi

echo ">>> session: launching $BACKEND (TP=$TP GPU=$GPU port=$PORT overlay=${OVERLAY_PYTHONPATH:-<none>}) log=$LOG"
adapter_launch
[ -n "${SERVER_PID:-}" ] || { echo "!!! adapter_launch did not set SERVER_PID" >&2; exit 2; }
server_record_identity "$SERVER_PID"

STALL_WINDOW_SEC=${STALL_WINDOW_SEC:-600}
case "${SERVER_STARTUP_TIMEOUT_SEC:-}" in ''|*[!0-9]*) CEILING=7200 ;; *) CEILING="$SERVER_STARTUP_TIMEOUT_SEC" ;; esac
_up=0; _reason=""; _t0=$SECONDS; _last_size=""; _last_change=$SECONDS
while :; do
  if adapter_health >/dev/null 2>&1; then _up=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then _reason="died_early"; break; fi
  if grep -Eq 'CUDA out of memory|HIP out of memory' "$LOG" 2>/dev/null; then _reason="oom"; break; fi
  if grep -Eq 'watchdog timeout|Capturing cuda graph failed|\[FATAL\]|(^|[[:space:]])FATAL:' "$LOG" 2>/dev/null; then
    _reason="fatal_marker"; break
  fi
  _size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$_size" != "$_last_size" ]; then _last_size="$_size"; _last_change=$SECONDS
  elif [ $((SECONDS - _last_change)) -ge "$STALL_WINDOW_SEC" ]; then _reason="stalled"; break; fi
  [ $((SECONDS - _t0)) -ge "$CEILING" ] && { _reason="ceiling_exceeded"; break; }
  sleep 5
done
_waited=$((SECONDS - _t0))
printf '{"status":"%s","reason":"%s","wait_sec":%d,"port":"%s","backend":"%s","log":"%s","base_url":"%s"}\n' \
  "$([ "$_up" = 1 ] && echo ok || echo failed)" "${_reason:-none}" "$_waited" "$PORT" "$BACKEND" "$LOG" "$BASE_URL" \
  > "$OUT_DIR/server_start.json"
if [ "$_up" != 1 ]; then
  echo "!!! session: server did not come up (reason=$_reason) after ~${_waited}s" >&2
  tail -n 40 "$LOG" >&2
  exit 2
fi
echo ">>> session: server up after ~${_waited}s at $BASE_URL; running SESSION_CMD"

# SERVER_PID is NOT handed down: a bench_e2e.sh inside the command sources the same teardown
# library, whose EXIT trap would otherwise stop this session's server.
_session_pid="$SERVER_PID"
SERVER_PID= SESSION_SERVER_PID="$_session_pid" REUSE_SERVER=1 bash -c "$SESSION_CMD"
_rc=$?
echo "$_rc" > "$OUT_DIR/session_cmd.rc"
echo ">>> session: SESSION_CMD rc=$_rc; tearing the server down"
exit "$_rc"
