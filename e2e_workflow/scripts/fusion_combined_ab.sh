#!/usr/bin/env bash
# Combined apply-back A/B: ONE base server vs ONE server with every authored fusion stacked.
#
# Each side is one fusion_serving_session.sh server. On it, bench_e2e.sh measures with
# REUSE_SERVER=1 under the warm_server lifecycle (one full discarded warm-up round, then
# AB_REPEATS timed rounds), and gsm8k then runs on that same live server before teardown.
# fusion_combined_decide.py compares the two sides.
#
# AB_REPEAT_MODE=replay_once is for a trace-replay client that owns its own warm-up (AgentX:
# aiperf replays per-lane cache warm-ups inside every leg): no outer warm-up round, ONE timed
# leg per side (bench_e2e.sh legacy lifecycle with GEAK_ISOLATED_REPLICA=1). An outer warm-up
# would be a second full replay per side for nothing the client does not already do.
#
#   OUT_DIR=<dir> CAND_OVERLAY=<combined loader dir> \
#   [BASE_OVERLAY=<current accepted overlay>] [EXPECT_BANNERS="tag1 tag2 ..."] \
#   bash fusion_combined_ab.sh
#
# Everything else bench_e2e.sh reads (BACKEND, TP, GPU, MODEL, ISL, OSL, CONC, EXTRA_SERVER_ARGS,
# EXTRA_ENV, BENCH_CLIENT and the AgentX/bench protocol exports, SERVING_GPU_LOCK, ...) is passed
# through from the caller's environment unchanged, so both sides use the run's serving config.
# Writes $OUT_DIR/{base,cand}/ and $OUT_DIR/combined_decision.json; exits 0 once a decision
# is written, 2 when a side produced no throughput summary.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${OUT_DIR:?OUT_DIR is required}"
: "${CAND_OVERLAY:?CAND_OVERLAY is required}"
BENCH="${BENCH_SCRIPT:-$HERE/bench_e2e.sh}"
SESSION="$HERE/fusion_serving_session.sh"
GSM8K="${GSM8K_EVAL_SCRIPT:-$HERE/fusion_gsm8k_eval.py}"
DECIDE="$HERE/fusion_combined_decide.py"
AB_REPEATS="${AB_REPEATS:-3}"
AB_REPEAT_MODE="${AB_REPEAT_MODE:-warm_server}"
case "$AB_REPEAT_MODE" in
  warm_server) _lifecycle="GEAK_REPEAT_MODE=warm_server REPEATS=$AB_REPEATS" ;;
  replay_once) _lifecycle="GEAK_REPEAT_MODE=legacy GEAK_ISOLATED_REPLICA=1 REPEATS=1" ;;
  *) echo "!!! AB_REPEAT_MODE must be warm_server or replay_once (got '$AB_REPEAT_MODE')" >&2; exit 4 ;;
esac
GSM8K_LIMIT="${GSM8K_LIMIT:-200}"
GSM8K_CONCURRENCY="${GSM8K_CONCURRENCY:-${CONC:-4}}"
GSM8K_THINKING="${GSM8K_THINKING:-false}"
mkdir -p "$OUT_DIR"

_think_flag=""
[ "$GSM8K_THINKING" = "true" ] || _think_flag="--no-thinking"

# Run inside the session (BASE_URL/PORT/OUT_DIR exported by fusion_serving_session.sh): the
# timed rounds on this server, then gsm8k on the same server. gsm8k asks for the name the server
# actually serves (--served-model-name), not the weights path.
SIDE_CMD="$_lifecycle \
MEASUREMENT_PURPOSE=\${MEASUREMENT_PURPOSE:-search} bash $(printf %q "$BENCH") > \"\$OUT_DIR/bench.log\" 2>&1; \
_bench_rc=\$?; echo \"\$_bench_rc\" > \"\$OUT_DIR/bench.rc\"; \
_served=\$(curl -sf \"\$BASE_URL/v1/models\" | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])' 2>/dev/null || echo \"\$MODEL\"); \
python3 $(printf %q "$GSM8K") --base-url \"\$BASE_URL/v1\" --model \"\$_served\" \
--limit $GSM8K_LIMIT --max-tokens 4096 --seed 0 --concurrency $GSM8K_CONCURRENCY $_think_flag \
--out \"\$OUT_DIR/gsm8k.json\" > \"\$OUT_DIR/gsm8k.log\" 2>&1; exit \$_bench_rc"

run_side() {
  local side="$1" overlay="$2"
  echo ">>> combined A/B: $side (overlay='${overlay:-<none>}') $(date -Is)"
  OUT_DIR="$OUT_DIR/$side" OVERLAY_PYTHONPATH="$overlay" SESSION_CMD="$SIDE_CMD" \
    bash "$SESSION" > "$OUT_DIR/$side.log" 2>&1
  echo ">>> combined A/B: $side rc=$? $(grep -m1 '^E2E_SUMMARY' "$OUT_DIR/$side/bench.log" 2>/dev/null) $(date -Is)"
}

run_side base "${BASE_OVERLAY:-}"
run_side cand "$CAND_OVERLAY"

for side in base cand; do
  if [ ! -s "$OUT_DIR/$side/bench_summary.json" ]; then
    echo "!!! combined A/B: $side produced no bench_summary.json (see $OUT_DIR/$side.log, $OUT_DIR/$side/bench.log)" >&2
    exit 2
  fi
done

_banners=()
for tag in ${EXPECT_BANNERS:-}; do _banners+=(--expect-banner "$tag"); done
python3 "$DECIDE" \
  --base-summary "$OUT_DIR/base/bench_summary.json" \
  --cand-summary "$OUT_DIR/cand/bench_summary.json" \
  --base-gsm8k "$OUT_DIR/base/gsm8k.json" \
  --cand-gsm8k "$OUT_DIR/cand/gsm8k.json" \
  --cand-server-log "$OUT_DIR/cand/server.log" \
  --tp "${TP:-1}" --noise-band-pct "${NOISE_BAND_PCT:-0.5}" \
  "${_banners[@]}" \
  --out "$OUT_DIR/combined_decision.json"
