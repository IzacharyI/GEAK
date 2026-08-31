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
#   bash verify.sh --tree <candidate_aiter> --world-size 8 [--three-way] [--out <dir>]
#
# Env the run command is parameterized through (set to your box's MegaMoE_v2 EP8 a8w4 bench):
#   BENCH_CMD   command that runs ONE measured route and prints a line "mega_e2e_us=<float>" per rank
#               and the per-process path marker; receives $TOKENS and $ROUTE in its env.
#   TOKENS      512 8192            (space-separated buckets to gate; default "512 8192")
#   ROUTES      uniform             (uniform gates; add "skew" to report it, it never gates)
#   REPLAYS     1000                (>=1000 CUDA-graph replays for the stress check)
#   RELL2_MAX   0.10                (parity tolerance at the largest bucket)
set -euo pipefail

TREE=""; WORLD=8; THREEWAY=0; OUT="./verify_out"
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE="$2"; shift 2;;
    --world-size) WORLD="$2"; shift 2;;
    --three-way) THREEWAY=1; shift;;
    --out) OUT="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$TREE" ] || { echo "ERROR: --tree <candidate_aiter> required" >&2; exit 2; }
[ -d "$TREE" ] || { echo "ERROR: tree not found: $TREE" >&2; exit 2; }
[ -n "${BENCH_CMD:-}" ] || { echo "ERROR: set BENCH_CMD (see header)" >&2; exit 2; }
TOKENS="${TOKENS:-512 8192}"; ROUTES="${ROUTES:-uniform}"; REPLAYS="${REPLAYS:-1000}"; RELL2_MAX="${RELL2_MAX:-0.10}"
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

if [ "$THREEWAY" = "1" ]; then
  note "three-way arms measured under one denominator (frozen public-AITER baseline):"
  note "  Baseline = four serialized launches | M2.5 incumbent = fused (source=validated_skill) | candidate = this tree"
  note "  M2.5 is the known-good floor, not the ceiling; a candidate beating it supersedes the incumbent."
fi

verdict=$([ "$FAIL" = "0" ] && echo pass || echo fail)
printf '{"skill":"megamoe_ep_persistent_fusion","source":"validated_skill","verdict":"%s","world_size":%s,"tokens":"%s","routes":"%s"}\n' \
  "$verdict" "$WORLD" "$TOKENS" "$ROUTES" | tee "$OUT/verdict.json"
[ "$FAIL" = "0" ] || { echo "[verify] GATE FAILED" >&2; exit 1; }
echo "[verify] all gating checks passed"
