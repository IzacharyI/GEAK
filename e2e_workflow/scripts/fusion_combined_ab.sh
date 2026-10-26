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
# With OBSERVED_WORKLOAD_JSON (AgentX) the gate is AB_GATE=step: per-step device time at every
# sampling point on a base and a cand server (fusion_serving_unit.py stepgate), then
# E2E_CONFIRM_ROUNDS (default 2) interleaved base/cand legs as a direction check
# (STEP_MIN_GAIN_PCT=0.5, E2E_TOLERANCE_PCT=5). AB_GATE=legs keeps the one-leg-per-side rule.
#
# Everything else bench_e2e.sh reads (BACKEND, TP, GPU, MODEL, ISL, OSL, CONC, EXTRA_SERVER_ARGS,
# EXTRA_ENV, BENCH_CLIENT and the AgentX/bench protocol exports, SERVING_GPU_LOCK, ...) is passed
# through from the caller's environment unchanged, so both sides use the run's serving config.
# Writes $OUT_DIR/{base,cand}/ and $OUT_DIR/combined_decision.json; exits 0 once a decision
# is written, 2 when a side produced no throughput summary, 5 when the engagement preflight
# failed (cand server did not start, or an EXPECT_BANNERS overlay never engaged), 7 when a side's
# server died or hung under load (combined_decision.json is then a reject naming why).
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
  # REUSE_COMPLETED_SIDES=1: a side that already has its throughput summary and gsm8k (e.g. the
  # base leg of an A/B whose cand server failed to start) is kept instead of re-measured.
  if [ "${REUSE_COMPLETED_SIDES:-0}" = 1 ] && [ -s "$OUT_DIR/$side/bench_summary.json" ] \
      && [ -s "$OUT_DIR/$side/gsm8k.json" ]; then
    echo ">>> combined A/B: $side reused (complete in $OUT_DIR/$side) $(date -Is)"
    return 0
  fi
  echo ">>> combined A/B: $side (overlay='${overlay:-<none>}') $(date -Is)"
  OUT_DIR="$OUT_DIR/$side" OVERLAY_PYTHONPATH="$overlay" SESSION_CMD="$SIDE_CMD" \
    bash "$SESSION" > "$OUT_DIR/$side.log" 2>&1
  local rc=$?
  echo ">>> combined A/B: $side rc=$rc $(grep -m1 '^E2E_SUMMARY' "$OUT_DIR/$side/bench.log" 2>/dev/null) $(date -Is)"
  # rc 6 = the session watchdog found the server dead or hung (server_dead.json says why). A cand
  # server that dies under the AgentX load is itself the verdict on the stack: stop here.
  if [ "$rc" = 6 ]; then
    echo "!!! combined A/B: the $side server died or hung under load: $(cat "$OUT_DIR/$side/server_dead.json" 2>/dev/null)" >&2
    printf '{"decision":"reject","reason":"%s server unhealthy under load: %s","server_dead":%s}\n' \
      "$side" "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reason",""))' "$OUT_DIR/$side/server_dead.json" 2>/dev/null)" \
      "$(cat "$OUT_DIR/$side/server_dead.json" 2>/dev/null || echo null)" > "$OUT_DIR/combined_decision.json"
    echo "COMBINED_DECISION=reject"
    exit 7
  fi
}

# Engagement preflight (PREFLIGHT_ENGAGEMENT=1, the default when EXPECT_BANNERS is set): boot the
# cand server, push a few gsm8k prompts through it, and require every EXPECT_BANNERS tag to print
# "[overlay-<tag>] ENGAGED" on all TP ranks before the two full legs are spent. DeepSeek-V4 run 9
# spent ~2h of A/B on a stack whose loader crashed the server, then on one where three of five
# overlays never engaged. Exit 5 names what failed; the base leg has not run yet.
# 64 prompts by default: under DP attention each rank must receive a prefill, or a
# prefill-only seam reads 7/8 (DeepSeek-V4 run 14 with 16 prompts).
if [ -n "${EXPECT_BANNERS:-}" ] && [ "${PREFLIGHT_ENGAGEMENT:-1}" = 1 ] \
    && ! { [ "${REUSE_COMPLETED_SIDES:-0}" = 1 ] && [ -s "$OUT_DIR/cand/bench_summary.json" ]; }; then
  _pf="$OUT_DIR/preflight"
  mkdir -p "$_pf"
  echo ">>> combined A/B: engagement preflight (cand overlay, ${PREFLIGHT_PROMPTS:-64} gsm8k prompts) $(date -Is)"
  # Short gsm8k prompts AND long-context requests at once (PREFLIGHT_LONG_PROMPTS x
  # PREFLIGHT_LONG_TOKENS): ranks then run very different batches, the condition under which a
  # fused path gated on the local token count can desynchronise the ranks' collectives.
  _pf_cmd="_served=\$(curl -sf \"\$BASE_URL/v1/models\" | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"id\"])' 2>/dev/null || echo \"\$MODEL\"); \
python3 $(printf %q "$HERE/fusion_preflight_load.py") --base-url \"\$BASE_URL\" --model \"\$_served\" \
--n ${PREFLIGHT_LONG_PROMPTS:-16} --tokens ${PREFLIGHT_LONG_TOKENS:-32768} > \"\$OUT_DIR/long_load.json\" 2>&1 & \
python3 $(printf %q "$GSM8K") --base-url \"\$BASE_URL/v1\" --model \"\$_served\" --limit ${PREFLIGHT_PROMPTS:-64} \
--max-tokens 512 --seed 0 --concurrency ${PREFLIGHT_PROMPTS:-64} --no-thinking --out \"\$OUT_DIR/gsm8k.json\" > \"\$OUT_DIR/gsm8k.log\" 2>&1; wait"
  OUT_DIR="$_pf" OVERLAY_PYTHONPATH="$CAND_OVERLAY" SESSION_CMD="$_pf_cmd" bash "$SESSION" > "$_pf.log" 2>&1
  # engagement.json: {server_ok, tp, engaged: {tag: ENGAGED lines}, missing: [tags], server_log}
  _missing=$(python3 - "$_pf" "${TP:-1}" $EXPECT_BANNERS <<'PY'
import json, os, re, sys
pf, tp, tags = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
log = os.path.join(pf, "server.log")
text = open(log, errors="replace").read() if os.path.isfile(log) else ""
engaged = {t: len(re.findall(r"\[overlay-%s\] ENGAGED" % re.escape(t), text)) for t in tags}
try:
    ok = json.load(open(os.path.join(pf, "server_start.json"))).get("status") == "ok"
except (OSError, ValueError):
    ok = False
missing = [t for t in tags if engaged[t] < tp]
json.dump({"server_ok": ok, "tp": tp, "engaged": engaged, "missing": missing, "server_log": log},
          open(os.path.join(pf, "engagement.json"), "w"), indent=1)
print(" ".join("%s(%d/%d)" % (t, engaged[t], tp) for t in missing))
PY
)
  if ! grep -q '"status": *"ok"' "$_pf/server_start.json" 2>/dev/null; then
    echo "!!! combined A/B preflight: the cand server did not start ($_pf/server_start.json, $_pf/server.log)" >&2
    exit 5
  fi
  if [ -s "$_pf/server_dead.json" ]; then
    echo "!!! combined A/B preflight: the cand server died or hung under the mixed preflight load: $(cat "$_pf/server_dead.json")" >&2
    exit 5
  fi
  if [ -n "$_missing" ]; then
    echo "!!! combined A/B preflight: overlay(s) never ENGAGED on every rank:$_missing (see $_pf/server.log)." >&2
    echo "!!! Fix their seam or drop them from the stack, then rerun; the full legs were not started." >&2
    exit 5
  fi
  echo ">>> combined A/B: preflight ok, every overlay engaged on ${TP:-1} rank(s) $(date -Is)"
fi
# PREFLIGHT_ONLY=1: the engagement check is the whole job (the apply-back repair loop runs it
# before deciding which overlays go into the A/B).
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && exit 0

# Step-time gate (AB_GATE=step, the default when OBSERVED_WORKLOAD_JSON is set): one AgentX leg
# per side cannot judge a fusion (DeepSeek-V4: five identical no-overlay legs spanned 512-602
# tok/s; an A/A pair "accepted" +7.3%). The judge is the per-step device time at every sampling
# point (fusion_serving_unit.py stepgate, base and cand server, gsm8k on both), weighted by the
# observed workload; E2E_CONFIRM_ROUNDS interleaved base/cand AgentX legs then only check that
# the e2e direction is not reversed.
AB_GATE="${AB_GATE:-$([ -n "${OBSERVED_WORKLOAD_JSON:-}" ] && echo step || echo legs)}"
_legs_base=(base); _legs_cand=(cand)
if [ "$AB_GATE" = step ]; then
  echo ">>> combined A/B: step-time gate (prefill+decode sampling points) $(date -Is)"
  python3 "${STEP_GATE_SCRIPT:-$HERE/fusion_serving_unit.py}" stepgate --root "$OUT_DIR/step" \
    --observed-workload "$OBSERVED_WORKLOAD_JSON" --base-overlay "${BASE_OVERLAY:-}" \
    --cand-overlay "$CAND_OVERLAY" --gsm8k-concurrency "$GSM8K_CONCURRENCY" \
    --out "$OUT_DIR/step_gate.json" > "$OUT_DIR/step_gate.log" 2>&1
  echo ">>> combined A/B: step gate rc=$? $(tail -c 300 "$OUT_DIR/step_gate.json" 2>/dev/null | tr '\n' ' ') $(date -Is)"
  for r in $(seq 2 "${E2E_CONFIRM_ROUNDS:-2}"); do _legs_base+=("base$r"); _legs_cand+=("cand$r"); done
fi

for i in "${!_legs_base[@]}"; do
  run_side "${_legs_base[$i]}" "${BASE_OVERLAY:-}"
  run_side "${_legs_cand[$i]}" "$CAND_OVERLAY"
done

for side in "${_legs_base[@]}" "${_legs_cand[@]}"; do
  if [ ! -s "$OUT_DIR/$side/bench_summary.json" ]; then
    echo "!!! combined A/B: $side produced no bench_summary.json (see $OUT_DIR/$side.log, $OUT_DIR/$side/bench.log)" >&2
    exit 2
  fi
done
_sum=()
for side in "${_legs_base[@]}"; do _sum+=(--base-summary "$OUT_DIR/$side/bench_summary.json"); done
for side in "${_legs_cand[@]}"; do _sum+=(--cand-summary "$OUT_DIR/$side/bench_summary.json"); done
_step=()
[ "$AB_GATE" = step ] && _step=(--step-gate "$OUT_DIR/step_gate.json"
  --step-min-gain-pct "${STEP_MIN_GAIN_PCT:-0.5}" --e2e-tolerance-pct "${E2E_TOLERANCE_PCT:-5}")

_banners=()
for tag in ${EXPECT_BANNERS:-}; do _banners+=(--expect-banner "$tag"); done
python3 "$DECIDE" \
  "${_sum[@]}" "${_step[@]}" \
  --base-gsm8k "$OUT_DIR/base/gsm8k.json" \
  --cand-gsm8k "$OUT_DIR/cand/gsm8k.json" \
  --cand-server-log "$OUT_DIR/cand/server.log" \
  --tp "${TP:-1}" --noise-band-pct "${NOISE_BAND_PCT:-0.5}" \
  "${_banners[@]}" \
  --out "$OUT_DIR/combined_decision.json"
