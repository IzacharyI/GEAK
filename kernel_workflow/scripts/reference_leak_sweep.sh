#!/usr/bin/env bash
# reference_leak_sweep.sh -- find every reachable copy of the answer inside a capability_eval run tree.
#
# WHY THIS EXISTS, AND WHY THE PATH CHECK IN kernel_workflow.js IS NOT ENOUGH.
#
# kernel_workflow.js asserts that each `known_reference_paths` entry resolves OUTSIDE the run's own
# tree. That closes exactly one hole -- the sibling checkout an engineer reaches with `ls ..`. It was
# written after two incidents and it does stop those two incidents. It does not stop the general case,
# because the reference is not only a checkout. On this project the same answer was ALSO reachable as:
#
#   1. artifacts/control/m25_fusion_reference.patch  -- the reference diff itself, handed to the
#      benchmark engineer on purpose for the positive control, sitting in the run tree
#   2. kernelworkflow-out/<wave1>/                   -- a previous wave's full output, workspaces and all
#   3. artifacts/geak_runs/<older waves>/            -- SIBLINGS OF THE CURRENT EVAL DIR, i.e. inside the
#      one directory every engineer is given by name
#   4. artifacts/geak_state/<task>/patches/*.diff + best/  -- the durable ledger, handed over by design
#   5. portable/megamoe-v2-candidate.bundle          -- a git bundle of the candidate branch
#   6. a branch in a DIFFERENT repo the run can read (here /sgl-workspace/aiter_main/aiter via
#      AITER_JIT_DIR) -- outside the tree, so no path check can see it at all
#
# Five of those six are inside the tree and none of them is a `known_reference_paths` entry. A run can
# therefore pass the containment assertion and still be one `grep -r` away from the answer. So this
# sweep looks for CONTENT, not for configured paths.
#
# It is deliberately a shell script and not part of kernel_workflow.js: workflow scripts have no
# filesystem access (and no Node API -- `require` throws), so the tree can only be walked from outside
# the workflow. Run it before launching, and after any wave that writes into the tree.
#
# Usage:
#   reference_leak_sweep.sh --tree <run tree root> [--allow <path>]... [--markers <file>] [--derive <ref> <base>]
#
# Exit 0 = clean. Exit 1 = leaks found (listed on stdout). Exit 2 = bad invocation.
#
# A "leak" is a file inside <tree> that contains a marker string that exists ONLY in the reference
# implementation. Markers are identifiers the reference introduces and the frozen baseline never uses,
# so a hit means reference-derived bytes, not a coincidental word. --derive regenerates the list from
# the two trees: added identifiers in `git diff base..ref` minus every identifier the baseline already
# contains. Regenerate it whenever the reference moves; a stale marker list silently under-reports.

set -uo pipefail

TREE=""; MARKER_FILE=""; ALLOW=()
DERIVE_REF=""; DERIVE_BASE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree)    TREE="$2"; shift 2 ;;
    --allow)   ALLOW+=("$2"); shift 2 ;;
    --markers) MARKER_FILE="$2"; shift 2 ;;
    --derive)  DERIVE_REF="$2"; DERIVE_BASE="$3"; shift 3 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MARKER_FILE:=$HERE/reference_leak_markers.txt}"

# --- derive mode: regenerate the marker list -------------------------------------------------------
if [[ -n "$DERIVE_REF" ]]; then
  [[ -d "$DERIVE_REF" && -d "$DERIVE_BASE" ]] || { echo "--derive needs two existing trees" >&2; exit 2; }
  base_rev="$(cd "$DERIVE_BASE" && git rev-parse --short HEAD 2>/dev/null)"
  ref_rev="$(cd "$DERIVE_REF" && git rev-parse --short HEAD 2>/dev/null)"
  [[ -n "$base_rev" && -n "$ref_rev" ]] || { echo "both trees must be git checkouts" >&2; exit 2; }
  added=$(mktemp); baseline=$(mktemp)
  (cd "$DERIVE_REF" && git diff "$base_rev".."$ref_rev" -- . ) \
    | grep '^+' | grep -oE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' | sort -u > "$added"
  grep -rhoE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$DERIVE_BASE" 2>/dev/null | sort -u > "$baseline"
  comm -23 "$added" "$baseline"
  rm -f "$added" "$baseline"
  exit 0
fi

[[ -n "$TREE" && -d "$TREE" ]] || { echo "--tree <existing dir> required" >&2; exit 2; }
[[ -f "$MARKER_FILE" ]] || { echo "marker file not found: $MARKER_FILE" >&2; exit 2; }

TREE="$(cd "$TREE" && pwd)"
mapfile -t MARKERS < <(grep -vE '^\s*(#|$)' "$MARKER_FILE")
[[ ${#MARKERS[@]} -gt 0 ]] || { echo "marker file is empty: $MARKER_FILE" >&2; exit 2; }

# MARKERS ARE IDENTIFIERS, SO THEY MATCH AS IDENTIFIERS -- not as substrings.
#
# --derive harvests whole `\b`-delimited tokens, so `self._out_cache_modifier` enters the list as the
# bare tail `_out_cache_modifier`. Under plain -F that marker also fires on `g1_out_cache_modifier`,
# which is a DIFFERENT name. That is not hypothetical: on 2026-08-21 it flagged r2_d2's independently
# authored megakernel, whose only overlap with the reference was the forced coinage of an
# `out_cache_modifier=` kwarg alongside the baseline's existing `b_cache_modifier`. Four tool calls to
# establish a non-finding is how a checker earns the reputation that gets it switched off.
#
# So each marker must be preceded and followed by a non-identifier character. A marker appearing as the
# tail or head of a longer identifier IS a different identifier and is not evidence of copied bytes.
#
# ONE alternation, not `-f patternfile`: GNU grep's -P accepts only a single pattern, and with the
# error swallowed it matches NOTHING and the sweep reports clean in a quarter of a second. That fail-
# open cost one debugging round here and is the same shape as the glob bug in skill_address_scan.sh --
# hence the self-test below, which refuses to run the sweep if the engine cannot catch a known marker.
pattern="(?<![A-Za-z0-9_])(?:$(printf '%s|' "${MARKERS[@]}" | sed 's/|$//'))(?![A-Za-z0-9_])"

# SELF-TEST: a checker that silently matches nothing is worse than no checker, because it converts
# "unchecked" into a green line in a report. Prove the engine fires on a synthetic marker occurrence
# and does NOT fire on the same marker embedded in a longer identifier, before trusting a clean result.
probe="${MARKERS[0]}"
if ! printf ' %s \n' "$probe" | grep -q -P "$pattern" 2>/dev/null; then
  echo "reference_leak_sweep: grep -P cannot evaluate the marker pattern on this system; the sweep" >&2
  echo "  would report clean by matching nothing. Refusing to run." >&2
  exit 2
fi
if printf 'zz%szz\n' "$probe" | grep -q -P "$pattern" 2>/dev/null; then
  echo "reference_leak_sweep: identifier boundaries are not being enforced. Refusing to run." >&2
  exit 2
fi

# SCOPE: files a leak can be COPIED OUT OF -- source, patches, bundles. Not measurement JSON, not
# reports, not __pycache__. This distinction is load-bearing and was learned by running the sweep
# unscoped: it returned 373 hits, most of them Step-2 analysis JSON whose only crime is naming an
# instrumentation counter (`_analysis_combine_wait_ticks`) in a field key. Those are EVIDENCE the run
# is entitled to and cannot be applied to a tree; folding them in makes the sweep unreadable and
# therefore ignorable. A sweep nobody reads catches nothing.
# Override with --ext when the reference lives in some other form.
: "${EXTS:=py diff patch bundle h hpp cpp cc cu hip s asm sh}"
find_args=(); first=1
for e in $EXTS; do
  if [[ $first -eq 1 ]]; then find_args+=(-name "*.$e"); first=0; else find_args+=(-o -name "*.$e"); fi
done

# The scanner and its marker list quote markers by construction; a checker flagging itself trains the
# reader to skim the output, which is how a real hit gets missed.
hits=$(find "$TREE" -type f \( "${find_args[@]}" \) -not -path '*/__pycache__/*' \
         -not -name 'reference_leak_sweep.sh' -not -name 'reference_leak_markers.txt' -print0 2>/dev/null \
       | xargs -0 -r grep -l -P "$pattern" 2>/dev/null | sort -u)

# --allow entries are trees the run is SUPPOSED to be able to read (the frozen baseline, the reference
# itself if it is somehow inside, an isolated control workspace). Everything else is a leak.
leaks=()
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  skip=0
  for a in "${ALLOW[@]:-}"; do
    [[ -z "$a" ]] && continue
    a="$(cd "$a" 2>/dev/null && pwd || echo "$a")"
    [[ "$f" == "$a" || "$f" == "$a/"* ]] && { skip=1; break; }
  done
  [[ $skip -eq 0 ]] && leaks+=("$f")
done <<< "$hits"

if [[ ${#leaks[@]} -eq 0 ]]; then
  echo "LEAK SWEEP clean: ${#MARKERS[@]} reference markers, 0 hits inside $TREE outside the allow-list."
  echo "NOTE this is a CONTENT sweep of one tree. It cannot see a reference reachable as a branch in"
  echo "another repository the run can read; only verify's byte-identity check covers that."
  exit 0
fi

echo "REFERENCE LEAK: ${#leaks[@]} file(s) inside $TREE carry reference-only markers."
echo "Each of these is a copy of the answer that an engineer reaches with one grep. Move them outside"
echo "the tree (a stale path in an old report is harmless once it resolves to nothing), or --allow them"
echo "if the run is genuinely supposed to read them."
# NAME THE MARKER, NOT JUST THE FILE. The post-Setup containment gate in kernel_workflow.js orders its
# agent to report the paths and NOT to open them -- so a bare path list gives the reader nothing to
# judge with, and the only way to tell a real copy from a coincidental name is to do the thing the gate
# forbids. Printing which identifiers matched keeps the triage inside the scanner's own output.
for f in "${leaks[@]}"; do
  echo "  $f"
  grep -o -P "$pattern" "$f" 2>/dev/null | sort | uniq -c | sort -rn | head -12 |
    while read -r n m; do echo "      ${n}x  $m"; done
done
exit 1
