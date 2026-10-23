#!/usr/bin/env bash
# Block (inside ONE tool call) until a detached measurement leg finishes or a time slice elapses.
#
#   fusion_leg_wait.sh <done_file> <pid> <log> [slice_s=540]
#
# Returns when <done_file> is non-empty (exit 0), <pid> is gone (exit 3), or slice_s elapsed while the
# leg is still running (exit 2 -- call it again). Prints the leg's log tail either way. Agent harnesses
# that refuse a bare foreground `sleep` still allow this, and it never ends the agent's turn the way
# waiting on a background-notification (Monitor) tool does.
set -u
done_file=$1; pid=$2; log=$3; slice=${4:-540}
end=$(( $(date +%s) + slice ))
rc=2
while :; do
  if [ -s "$done_file" ]; then rc=0; break; fi
  if ! kill -0 "$pid" 2>/dev/null; then
    [ -s "$done_file" ] && rc=0 || rc=3
    break
  fi
  [ "$(date +%s)" -ge "$end" ] && break
  sleep 15
done
case $rc in
  0) echo "LEG_DONE: $done_file" ;;
  2) echo "LEG_RUNNING: pid $pid alive after ${slice}s slice -- call fusion_leg_wait.sh again" ;;
  3) echo "LEG_EXITED: pid $pid gone and $done_file missing/empty" ;;
esac
echo "--- tail $log ---"; tail -5 "$log" 2>/dev/null
exit $rc
