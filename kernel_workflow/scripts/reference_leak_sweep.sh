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
#   6. a branch in a DIFFERENT repo the run can read (for example an external JIT checkout via
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
#   reference_leak_sweep.sh --tree <run tree root> [--allow <path>]... [--repo <path>]...
#                           [--markers <file>] [--derive <ref> <base>] [--given <path>]...
#
# --repo adds a repository OUTSIDE <tree> whose refs should also be scanned -- the AITER_JIT_DIR
# checkout, a sibling clone, anything the run can read by configuration rather than by walking.
#
# Exit 0 = clean. Exit 1 = leaks found (listed on stdout). Exit 2 = bad invocation.
#
# A "leak" is a file inside <tree> that contains a marker string that exists ONLY in the reference
# implementation. Markers are identifiers the reference introduces and the frozen baseline never uses,
# so a hit means reference-derived bytes, not a coincidental word. --derive regenerates the list from
# the two trees: added identifiers in `git diff base..ref` minus every identifier the baseline already
# contains. Regenerate it whenever the reference moves; a stale marker list silently under-reports.

set -uo pipefail

# MARKER_FILE is NOT cleared here. It used to be, which silently disabled the documented environment
# override: the `:=` default below only fires on an unset-or-empty value, so wiping it first meant an
# exported MARKER_FILE was always discarded and the default always won. Caught 2026-08-23 while testing
# that the sweep fails closed on a missing list -- it did not fail at all, because the bad path had
# been thrown away and the good default restored. `--markers` was the only override that ever worked.
TREE=""; ALLOW=(); EXTRA_REPOS=()
DERIVE_REF=""; DERIVE_BASE=""; GIVEN=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tree)    TREE="$2"; shift 2 ;;
    --allow)   ALLOW+=("$2"); shift 2 ;;
    --repo)    EXTRA_REPOS+=("$2/.git"); shift 2 ;;
    --markers) MARKER_FILE="$2"; shift 2 ;;
    --derive)  DERIVE_REF="$2"; DERIVE_BASE="$3"; shift 3 ;;
    --given)   GIVEN+=("$2"); shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# THE MARKER LIST DOES NOT LIVE IN SKILL_DIR, AND MUST NOT BE MOVED BACK.
#
# It is a list of the reference implementation's own symbol names -- `_publish_tok_ready`,
# `_S2_EPOCH_SLOT`, `mega_moe_fused_s2c`. Under SKILL_DIR it sat one `ls scripts/` away from every
# engineer the workflow runs, which makes the leak detector into a leak: an engineer who reads it is
# handed the design as a vocabulary list, and no amount of "do not read this" in a header survives
# curiosity. Distance is the only thing that contains it (uid 0 makes chmod inert; see handoff
# §14z-35). So it lives outside the walk path and a trusted launcher supplies MARKER_FILE out of
# band. Do not write that path into this script or launch_args.json: same-UID agents can read both.
#
# Fails CLOSED. If the file is absent the sweep exits 2 rather than scanning with an empty list, so a
# lost marker file is a loud abort and never a clean-looking zero-hit run.
: "${MARKER_FILE:?set MARKER_FILE to the out-of-band reference marker list}"
# Unconditional: a copy under scripts/ is the leak whether or not the verify-only path also exists.
# Gating this on "$MARKER_FILE is missing" would have made the common case -- someone restores the old
# file while the new one is still in place -- pass silently, which is the exact failure being guarded.
if [[ -f "$HERE/reference_leak_markers.txt" ]]; then
  echo "reference_leak_sweep: refusing to use $HERE/reference_leak_markers.txt -- the marker list is" >&2
  echo "verify-only and must not sit inside SKILL_DIR where engineers can read it. Move it to" >&2
  echo "$MARKER_FILE (or set MARKER_FILE) and delete the copy under scripts/." >&2
  exit 2
fi

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

  # SUBTRACT THE BASE REPO'S OTHER REFS, NOT ONLY ITS WORKING TREE.
  #
  # "Reference-unique" was defined as "added by the reference, minus what the baseline CHECKOUT
  # contains". That definition is too narrow, and the gap is measurable: on 2026-08-22 the list built
  # this way carried `combine_data_type`, `fused_combine` and `ready_base`, and a ref-aware sweep of a
  # plain ROCm/aiter clone found 23 blobs across 14 upstream branches (`wjx/exp_moe_ep`,
  # `zxe/fused_dispatch_gemm1_gfx1250`, `yanbo/ep_moe`, ...) carrying nothing else. Those names are not
  # reference-derived; they are what the team already calls these things on adjacent MoE branches. The
  # reference merely happened to reuse them.
  #
  # An over-inclusive marker list is not a safe default. It costs the sweep its readability -- 23 hits
  # on prior art reads exactly like 23 hits on a copy -- and a scanner whose output must be triaged by
  # hand is a scanner that gets switched off. So harvest identifiers from every ref tip of the base
  # repo too, deduped by SHA and bounded, and say on stderr what was and was not covered.
  refs_scanned=0; refs_total=0
  declare -A dseen=()
  while read -r sha _; do
    [[ -z "$sha" || -n "${dseen[$sha]:-}" ]] && continue
    dseen[$sha]=1; refs_total=$((refs_total+1))
  done < <(git -C "$DERIVE_BASE" for-each-ref --format='%(objectname) %(refname)' 2>/dev/null)
  # NOT `for ... done | sort -u >> "$baseline"`. A loop on the left of a pipe runs in a SUBSHELL, so
  # every refs_scanned increment is discarded when it exits and the counter reads 0 no matter how many
  # trees were harvested. That misfires the "NOT harvested / the list may over-report" warning below on
  # a run that in fact harvested everything -- and that warning is the operator's only signal that the
  # subtraction was incomplete, so a version of it that cries wolf is worse than none. Redirect per
  # iteration instead and keep the loop in this shell.
  for sha in "${!dseen[@]}"; do
    [[ $refs_scanned -ge ${REF_SCAN_MAX_TREES:-2000} ]] && break
    refs_scanned=$((refs_scanned+1))
    git -C "$DERIVE_BASE" grep -I -h -oE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$sha" 2>/dev/null >> "$baseline"
  done
  sort -u -o "$baseline" "$baseline"
  echo "--derive: subtracted the base working tree plus $refs_scanned of $refs_total unique ref trees." >&2
  [[ $refs_scanned -lt $refs_total ]] && echo "--derive: $((refs_total - refs_scanned)) ref tree(s) NOT harvested (REF_SCAN_MAX_TREES); the list may over-report." >&2
  [[ $refs_total -eq 0 ]] && echo "--derive: the base repo has no refs, so only its working tree was subtracted. If you stripped its refs for containment, derive the list BEFORE stripping." >&2

  # SUBTRACT WHAT THE RUN IS LEGITIMATELY GIVEN (--given <path>, repeatable).
  #
  # A marker only means something if an engineer could not have arrived at it honestly. The task text
  # names the opt-in flags on purpose and tells engineers to print `path=MEGA` vs `path=SCATTERED`; the
  # method cards name mechanisms. Every one of those tokens is also "added by the reference and absent
  # from the baseline", so the raw derive proposes them as markers -- and a list containing SCATTERED
  # fires on every honest candidate the wave produces. That is how a scanner earns the reputation that
  # gets it switched off.
  #
  # Until now the marker file's own header carried this as a manual instruction ("then re-apply the two
  # filters above by hand"). A hand-applied filter is one a tired operator skips, and skipping it fails
  # OPEN in the loud direction. So pass the material the run can read and let the tool do the
  # subtraction: --given <task dir> --given <knowledge dir> --given <roles dir>.
  if [[ ${#GIVEN[@]} -gt 0 ]]; then
    given_n=0
    for g in "${GIVEN[@]}"; do
      [[ -e "$g" ]] || { echo "--given: no such path, ignored: $g" >&2; continue; }
      given_n=$((given_n+1))
      grep -rhoE '\b[A-Za-z_][A-Za-z0-9_]{7,}\b' "$g" 2>/dev/null >> "$baseline"
    done
    sort -u -o "$baseline" "$baseline"
    echo "--derive: also subtracted identifiers appearing in $given_n path(s) the run is given." >&2
  else
    echo "--derive: no --given paths. Tokens the task text and knowledge cards name on purpose (the opt-in flags, path=SCATTERED, ...) will be proposed as markers and must be removed by hand before installing." >&2
  fi

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

# --- REFS ------------------------------------------------------------------------------------------
#
# A `find`-based sweep greps FILES. A branch that is not checked out has no files, so a repository can
# carry the entire reference on a second branch and still be walked and pronounced clean. That is not
# an exotic case; it is the default state of any repo with more than one branch, and on 2026-08-22 it
# was the actual state of THE FROZEN BASELINE THIS SWEEP WAS POINTED AT. A frozen baseline
# held a local branch `mega` -- 15 files, +3906 lines, 140 of 140 markers -- one `git checkout` away
# inside the directory the task text hands to every engineer. The sweep read it and reported clean, and
# I relayed that clean result to the user. A checker that answers the wrong question confidently is
# worse than no checker; that is the same failure the -P self-test above exists to prevent.
#
# So: enumerate every repository under <tree>, and grep the TREE AT EACH REF, not just the worktree.
# Refs are deduped by commit SHA -- N branches at one commit is one scan.
#
# What this still does not cover, stated so nobody reads a clean line as more than it is: unreferenced
# history (a marker in an ancestor commit whose content no ref tip carries), dangling objects, and
# repositories outside <tree> that are reachable some other way (AITER_JIT_DIR, a bundle, a symlink).
# Pass those with --repo. Beyond that, only verify's byte-identity check covers it.
gitdirs=()
while IFS= read -r g; do [[ -n "$g" ]] && gitdirs+=("$g"); done < <(
  find "$TREE" -name .git -maxdepth 6 -print 2>/dev/null | sort -u)
for r in "${EXTRA_REPOS[@]:-}"; do [[ -n "$r" ]] && gitdirs+=("$r"); done

ref_leaks=(); ref_skipped=0
# Same self-exclusion the find pass makes with -not -name: this scanner and its marker list quote
# markers by construction, and at a ref tip they are ordinary tracked blobs. Without these two
# exclusions the sweep's first act is to flag itself in every repo that carries it -- which is exactly
# the "trains the reader to skim" failure the find pass already guards against.
pathspec=(); for e in $EXTS; do pathspec+=("*.$e"); done
pathspec+=(':!*reference_leak_sweep.sh' ':!*reference_leak_markers.txt')

for g in "${gitdirs[@]:-}"; do
  [[ -z "$g" ]] && continue
  repo="$(dirname "$g")"; [[ -d "$g" ]] || repo="$(dirname "$g")"
  repo="$(cd "$repo" 2>/dev/null && pwd)" || continue
  skip=0
  for a in "${ALLOW[@]:-}"; do
    [[ -z "$a" ]] && continue
    a="$(cd "$a" 2>/dev/null && pwd || echo "$a")"
    [[ "$repo" == "$a" || "$repo" == "$a/"* ]] && { skip=1; break; }
  done
  [[ $skip -eq 1 ]] && continue

  # Dedupe by SHA: `%(objectname) %(refname)` over every ref, plus HEAD (a detached HEAD is a ref no
  # for-each-ref lists, and a detached HEAD at the reference is exactly how one would hide it).
  declare -A seen_sha=()
  while read -r sha rname; do
    [[ -z "$sha" ]] && continue
    [[ -n "${seen_sha[$sha]:-}" ]] && continue
    seen_sha[$sha]="$rname"
  done < <( { git -C "$repo" for-each-ref --format='%(objectname) %(refname)' 2>/dev/null
              printf '%s HEAD\n' "$(git -C "$repo" rev-parse HEAD 2>/dev/null)"; } )

  n=0
  for sha in "${!seen_sha[@]}"; do
    n=$((n+1))
    if [[ $n -gt ${REF_SCAN_MAX_TREES:-200} ]]; then ref_skipped=$((ref_skipped+1)); break; fi
    while IFS= read -r line; do
      [[ -n "$line" ]] && ref_leaks+=("$repo  ${seen_sha[$sha]} (${sha:0:8})  ${line#*:}")
    done < <(git -C "$repo" grep -I -l -P "$pattern" "$sha" -- "${pathspec[@]}" 2>/dev/null)
  done
  unset seen_sha
done

if [[ ${#ref_leaks[@]} -gt 0 ]]; then
  echo "REFERENCE LEAK IN GIT REFS: ${#ref_leaks[@]} blob(s) carry reference-only markers at a ref tip."
  echo "These are invisible to a working-tree grep and reachable with one \`git checkout\`. Remove the"
  echo "ref (or move the whole .git outside the run tree) before launching."
  printf '  %s\n' "${ref_leaks[@]}" | sort -u | head -60
  [[ ${#ref_leaks[@]} -gt 60 ]] && echo "  ... $(( ${#ref_leaks[@]} - 60 )) more"
fi
[[ $ref_skipped -gt 0 ]] && echo "NOTE $ref_skipped repo(s) had more unique ref trees than REF_SCAN_MAX_TREES=${REF_SCAN_MAX_TREES:-200}; the remainder were NOT scanned."

if [[ ${#leaks[@]} -eq 0 && ${#ref_leaks[@]} -eq 0 ]]; then
  echo "LEAK SWEEP clean: ${#MARKERS[@]} reference markers, 0 hits inside $TREE outside the allow-list."
  echo "Covered: working-tree files, and the tree at every ref tip of ${#gitdirs[@]} repository(ies)."
  echo "NOT covered: unreferenced history, dangling objects, and repos outside <tree> not passed with"
  echo "--repo. Only verify's byte-identity check covers those."
  exit 0
elif [[ ${#leaks[@]} -eq 0 ]]; then
  exit 1
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
