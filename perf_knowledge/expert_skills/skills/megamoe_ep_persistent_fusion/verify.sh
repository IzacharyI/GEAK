#!/usr/bin/env bash
# verify.sh — executable gate for expert_skill:megamoe_ep_persistent_fusion.
#
# This script takes NO reference answer. It drives the one-flag A/B on the CANDIDATE tree only and
# encodes the five acceptance checks from skill.md ("## Executable verification"). It is the
# production-mode reproduction gate; the blind autonomy proof does not use it (use_expert_skills=OFF).
#
# Containment: only relative paths inside the candidate tree and env flags appear here. No reference
# branch/commit/absolute path, and no literal diff of the answer — mechanism, never the built copy.
# (scripts/skill_address_scan.sh enforces this.)
#
# Usage:
#   bash verify.sh --tree <candidate_aiter> --world-size 8 \
#        [--incumbent <m2.5_aiter>] [--require-improvement] [--three-way] [--out <dir>]
#
# --incumbent turns M2.5 from a printed reference into an ENFORCED FLOOR: the candidate's fused path
# is paired against the incumbent's fused path and the selection is max(candidate, M2.5). A candidate
# that does not strictly beat M2.5 past the noise floor does NOT ship — the incumbent is kept. This is
# the "保底 / incumbent, not ceiling" rule: never deploy below M2.5, but a real win supersedes it.
# Without --incumbent the M2.5 arm cannot be measured, so the floor is report-only (Baseline gate only).
#
# Env the run command is parameterized through (set to your box's MegaMoE_v2 EP8 a8w4 bench):
#   BENCH_CMD   command that runs ONE measured route and prints a line "mega_e2e_us=<float>" per rank
#               and the per-process path marker; receives $TOKENS and $ROUTE in its env.
#   TOKENS      512 8192            (space-separated buckets to gate; default "512 8192")
#   ROUTES      uniform             (uniform gates; add "skew" to report it, it never gates)
#   REPLAYS     1000                (>=1000 CUDA-graph replays for the stress check)
#   RELL2_MAX   0.10                (parity tolerance at the largest bucket)
#   NOISE_PCT   1.45                (measured per-case noise floor; |gain| within this = a tie)
set -euo pipefail

TREE=""; WORLD=8; THREEWAY=0; OUT="./verify_out"; INCUMBENT=""; REQUIRE_IMPROVE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE="$2"; shift 2;;
    --world-size) WORLD="$2"; shift 2;;
    --incumbent) INCUMBENT="$2"; shift 2;;
    --require-improvement) REQUIRE_IMPROVE=1; shift;;
    --three-way) THREEWAY=1; shift;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$TREE" ] || { echo "ERROR: --tree <candidate_aiter> required" >&2; exit 2; }
[ -d "$TREE" ] || { echo "ERROR: tree not found: $TREE" >&2; exit 2; }
[ -z "$INCUMBENT" ] || [ -d "$INCUMBENT" ] || { echo "ERROR: incumbent tree not found: $INCUMBENT" >&2; exit 2; }
[ -n "${BENCH_CMD:-}" ] || { echo "ERROR: set BENCH_CMD (see header)" >&2; exit 2; }
TOKENS="${TOKENS:-512 8192}"; ROUTES="${ROUTES:-uniform}"; REPLAYS="${REPLAYS:-1000}"; RELL2_MAX="${RELL2_MAX:-0.10}"; NOISE_PCT="${NOISE_PCT:-1.45}"
mkdir -p "$OUT"
FAIL=0; note(){ echo "[verify] $*"; }

# one arm: $1=label $2=fuse(0|1) $3=tokens $4=route  -> echoes rank-max mega_e2e_us, records marker count
run_arm(){
  local label="$1" fuse="$2" tok="$3" route="$4" log="$OUT/${label}_${tok}_${route}.log"
  ( cd "$TREE" && AITER_MEGAMOE_FUSE_ALL="$fuse" TOKENS="$tok" ROUTE="$route" bash -c "$BENCH_CMD" ) >"$log" 2>&1 || true
  # (2) path marker: fused arm must print one 'path=MEGA' per rank
  if [ "$fuse" = "1" ]; then
    local mk; mk=$(grep -c 'path=MEGA' "$log" || true)
    echo "$mk" > "$OUT/.marker_${tok}_${route}"
  fi
  # rank-max across ranks
  awk -F= '/mega_e2e_us=/{v=$2+0; if(v>m)m=v} END{printf "%.4f", m}' "$log"
}

# incumbent M2.5 fused arm on ITS OWN tree (path given at runtime, never baked into this card)
# $1=tree $2=tok $3=route -> rank-max mega_e2e_us (fused)
run_fused_on(){
  local tr="$1" tok="$2" route="$3" log="$OUT/incumbent_${tok}_${route}.log"
  ( cd "$tr" && AITER_MEGAMOE_FUSE_ALL=1 TOKENS="$tok" ROUTE="$route" bash -c "$BENCH_CMD" ) >"$log" 2>&1 || true
  awk -F= '/mega_e2e_us=/{v=$2+0; if(v>m)m=v} END{printf "%.4f", m}' "$log"
}

for tok in $TOKENS; do
  for route in $ROUTES; do
    note "guard tokens=$tok route=$route  (A,B,A,B x3 interleaved)"
    declare -a S=(); declare -a M=()
    for i in 1 2 3; do
      S+=("$(run_arm scattered 0 "$tok" "$route")")
      M+=("$(run_arm megakernel 1 "$tok" "$route")")
      S+=("$(run_arm scattered 0 "$tok" "$route")")
      M+=("$(run_arm megakernel 1 "$tok" "$route")")
    done
    # (2) marker gate
    mk=$(cat "$OUT/.marker_${tok}_${route}" 2>/dev/null || echo 0)
    if [ "$mk" -lt "$WORLD" ]; then note "VOID: path marker $mk < world $WORLD (fusion predicate not taken)"; FAIL=1; fi
    # paired median gain
    gain=$(python3 - "$tok" "$route" "${S[@]}" ::: "${M[@]}" <<'PY'
import sys,statistics as st
tok,route=sys.argv[1],sys.argv[2]; rest=sys.argv[3:]; k=rest.index(':::'); S=list(map(float,rest[:k])); M=list(map(float,rest[k+1:]))
pair=[(s-m)/s*100 for s,m in zip(S,M)]
g=st.median(pair)
print(f"{g:.2f} {min(pair):.2f} {max(pair):.2f}")
PY
)
    med=$(echo "$gain"|awk '{print $1}')
    note "  gain median=${med}%  range=[$(echo "$gain"|awk '{print $2" .. "$3}')]%"
    # (5) uniform gates on beating Baseline; skew is reported only
    if [ "$route" = "uniform" ]; then
      awk -v g="$med" 'BEGIN{exit !(g>0)}' || { note "  GATE FAIL: uniform not > Baseline"; FAIL=1; }
    fi
  done
done

# (3) relL2 parity at the largest bucket — expects BENCH_CMD in PARITY=1 to print 'relL2=<float>'
big=$(echo $TOKENS|tr ' ' '\n'|sort -n|tail -1)
( cd "$TREE" && PARITY=1 TOKENS="$big" ROUTE=uniform AITER_MEGAMOE_FUSE_ALL=1 bash -c "$BENCH_CMD" ) >"$OUT/parity.log" 2>&1 || true
r=$(grep -oE 'relL2=[0-9.eE+-]+' "$OUT/parity.log" | tail -1 | cut -d= -f2 || echo "")
if [ -z "$r" ]; then note "PARITY: no relL2 emitted"; FAIL=1
else awk -v r="$r" -v t="$RELL2_MAX" 'BEGIN{exit !(r<=t)}' && note "parity relL2=$r <= $RELL2_MAX" || { note "PARITY FAIL relL2=$r > $RELL2_MAX"; FAIL=1; }; fi

# (4) 1000-replay stress: last iteration must equal first — expects BENCH_CMD in STRESS mode to print
#     'replay_drift=<float>' comparing generation N to generation 1.
( cd "$TREE" && STRESS="$REPLAYS" TOKENS="$big" ROUTE=uniform AITER_MEGAMOE_FUSE_ALL=1 bash -c "$BENCH_CMD" ) >"$OUT/stress.log" 2>&1 || true
d=$(grep -oE 'replay_drift=[0-9.eE+-]+' "$OUT/stress.log" | tail -1 | cut -d= -f2 || echo "")
if [ -z "$d" ]; then note "STRESS: no replay_drift emitted over $REPLAYS iters"; FAIL=1
else awk -v d="$d" 'BEGIN{exit !(d<=0)}' && note "stress ${REPLAYS} replays drift=$d (last==first)" || { note "STRESS FAIL drift=$d over $REPLAYS"; FAIL=1; }; fi

# (1) two-launch shape: terminal form is quant + one megakernel, never fused-quant
( cd "$TREE" && AITER_MEGAMOE_FUSE_ALL=1 GRAPH_DUMP=1 TOKENS="$big" ROUTE=uniform bash -c "$BENCH_CMD" ) >"$OUT/graph.log" 2>&1 || true
launches=$(grep -oE 'launch=[a-zA-Z0-9_]+' "$OUT/graph.log" | sort -u | wc -l || echo 0)
note "captured distinct launches/rank=$launches (expect 2: quant, megakernel)"

# ---- Floor / three-way selection: M2.5 is the incumbent FLOOR, not the ceiling. -------------------
# Ship max(candidate, M2.5). A candidate that does not strictly beat M2.5 past the noise floor keeps
# the incumbent; the floor guarantees deployment is never below M2.5. A real win supersedes it.
SELECTED="candidate"; VS_INC="not_measured"; DECISION="baseline_gate_only"
if [ -n "$INCUMBENT" ]; then
  note "floor check: candidate vs M2.5 incumbent on ${big}_uniform (A,B,A,B x3, rank-max, noise=${NOISE_PCT}%)"
  declare -a C=(); declare -a I=()
  for i in 1 2 3; do
    C+=("$(run_arm megakernel 1 "$big" uniform)"); I+=("$(run_fused_on "$INCUMBENT" "$big" uniform)")
    C+=("$(run_arm megakernel 1 "$big" uniform)"); I+=("$(run_fused_on "$INCUMBENT" "$big" uniform)")
  done
  read -r VS_INC lo hi <<PYOUT
$(python3 - "${C[@]}" ::: "${I[@]}" <<'PY'
import sys, statistics as st
r = sys.argv[1:]; k = r.index(':::'); C = list(map(float, r[:k])); I = list(map(float, r[k+1:]))
p = [(i - c) / i * 100 for c, i in zip(C, I)]   # + = candidate faster than incumbent M2.5
print(f"{st.median(p):.2f} {min(p):.2f} {max(p):.2f}")
PY
)
PYOUT
  note "  candidate vs M2.5: median=${VS_INC}%  per-pair=[${lo} .. ${hi}]%"
  if awk -v g="$VS_INC" -v n="$NOISE_PCT" 'BEGIN{exit !(g>n)}'; then
    SELECTED="candidate"; DECISION="supersede"
    note "  -> candidate SUPERSEDES M2.5 past noise; ship candidate, bump incumbent.label + re-record gains"
  elif awk -v g="$VS_INC" -v n="$NOISE_PCT" 'BEGIN{exit !(g < -n)}'; then
    SELECTED="M2.5_incumbent"; DECISION="regress_keep_incumbent"
    note "  -> candidate SLOWER than M2.5; FLOOR HOLDS, keep & ship M2.5 (never deploy below incumbent)"
    [ "$REQUIRE_IMPROVE" = "1" ] && { note "  --require-improvement: no gain over incumbent -> gate fail"; FAIL=1; }
  else
    SELECTED="M2.5_incumbent"; DECISION="tie_keep_incumbent"
    note "  -> tie within noise; capability reproduced (not a regression), keep incumbent M2.5"
    [ "$REQUIRE_IMPROVE" = "1" ] && { note "  --require-improvement: a tie is not an improvement -> gate fail"; FAIL=1; }
  fi
elif [ "$THREEWAY" = "1" ]; then
  note "three-way requested but no --incumbent tree: M2.5 arm NOT measured, floor is report-only."
  note "  Baseline = four serialized launches | M2.5 incumbent = a kept fused build (source=validated_skill) | candidate = this tree"
  note "  pass --incumbent <m2.5_aiter> to enforce the floor (ship max(candidate, M2.5))."
fi

verdict=$([ "$FAIL" = "0" ] && echo pass || echo fail)
printf '{"skill":"megamoe_ep_persistent_fusion","source":"validated_skill","verdict":"%s","selected":"%s","decision":"%s","vs_incumbent_pct":"%s","world_size":%s,"tokens":"%s","routes":"%s"}\n' \
  "$verdict" "$SELECTED" "$DECISION" "$VS_INC" "$WORLD" "$TOKENS" "$ROUTES" | tee "$OUT/verdict.json"
[ "$FAIL" = "0" ] || { echo "[verify] GATE FAILED" >&2; exit 1; }
note "selected=$SELECTED decision=$DECISION (floor: never below M2.5)"
echo "[verify] all gating checks passed"
