#!/usr/bin/env bash
# Assemble a runnable GEAK task workspace from three inputs and nothing else:
#
#   1. a baseline AITER checkout          (--baseline)
#   2. this workflow                      (the tree this script lives in)
#   3. the machine's runtime libraries    (MORI, ROCm, the AITER JIT cache — probed, not carried)
#
# Everything else a previous run produced — analyses, logs, accumulated patches, reference
# implementations — is deliberately NOT an input. If assembling a workspace ever seems to need one,
# that is the finding, not a missing file: the workflow is supposed to derive it.
#
# Usage:
#   bootstrap_task.sh --check
#   bootstrap_task.sh --baseline <aiter checkout> --out <workspace> [options]
#
# Options:
#   --task <name>        task template under tasks/ (default: megamoe_v2_ep8)
#   --exp-root <dir>     where the run writes rounds   (default: <out>/../geak_runs)
#   --state-dir <dir>    cross-round state             (default: <out>/../geak_state/<task>)
#   --mori-root <dir>    MORI checkout                 (default: $MORI_ROOT, else probed)
#   --jit-dir <dir>      prebuilt AITER JIT cache      (default: $AITER_JIT_DIR, else probed)
#   --known-reference <csv>  paths the provenance check must refuse (default: none)
#   --args-out <file>    where launch_args.json is written. Defaults to <out>/../launch_args.json,
#                        EXCEPT when --known-reference is set: those paths must not be written into
#                        the tree an engineer walks, so the default moves to
#                        $HOME/geak_launch/<workspace>/launch_args.json and an in-tree --args-out
#                        is refused.
#   --force              overwrite a non-empty --out
#   --check              probe the environment and exit; assemble nothing
#   --no-probe           assemble without probing. For inspecting a task off-machine and for the
#                        test suite. The workspace it writes is not certified runnable here.
#
# Exit codes: 0 ok, 1 usage, 2 environment structurally unfit, 3 fit but not launchable right now.
#
# Two kinds of unfitness, kept apart on purpose. STRUCTURAL (no 8-GPU group, no MORI) will not fix
# itself and blocks assembly. TRANSIENT (a co-tenant holding the VRAM) is the normal state of a
# shared box and blocks only the launch — assembling the workspace and then waiting for the pool is
# the correct sequence, so it must not be an error.
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=megamoe_v2_ep8
BASELINE= OUT= EXP_ROOT= STATE_DIR= FORCE=0 CHECK_ONLY=0 PROBE=1
MORI_ROOT_IN="${MORI_ROOT:-}"
JIT_DIR_IN="${AITER_JIT_DIR:-}"
KNOWN_REF=""
ARGS_OUT="${ARGS_OUT:-}"

die() { echo "bootstrap_task: $*" >&2; exit "${2:-1}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --baseline) BASELINE="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --task) TASK="${2:?}"; shift 2 ;;
    --exp-root) EXP_ROOT="${2:?}"; shift 2 ;;
    --state-dir) STATE_DIR="${2:?}"; shift 2 ;;
    --mori-root) MORI_ROOT_IN="${2:?}"; shift 2 ;;
    --jit-dir) JIT_DIR_IN="${2:?}"; shift 2 ;;
    --known-reference) KNOWN_REF="${2:?}"; shift 2 ;;
    --args-out) ARGS_OUT="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    --no-probe) PROBE=0; shift ;;
    -h|--help) sed -n '1,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

TASK_DIR="$SKILL_DIR/tasks/$TASK"
[ -d "$TASK_DIR" ] || die "no task template at $TASK_DIR (have: $(ls "$SKILL_DIR/tasks" 2>/dev/null | tr '\n' ' '))"

# ---------------------------------------------------------------- environment probe
# Each check names its own remedy. A wrong environment for this task does not produce a clear
# error at run time — it hangs inside the communication runtime or aborts in CheckStatusValid() —
# so the cost of finding out here is a second and the cost of finding out later is a lease.
fit=0 ready=0
note() { printf '  %-6s %s\n' "$1" "$2"; }

if [ "$PROBE" = 0 ] && [ "$CHECK_ONLY" = 0 ]; then
  echo "environment probe: SKIPPED (--no-probe). This workspace is not certified runnable here."
fi
if [ "$PROBE" = 1 ] || [ "$CHECK_ONLY" = 1 ]; then
echo "environment probe:"

# (1) GPUs. gfx950 x8 on one node. rocm-smi is authoritative; fall back to the sysfs enumeration
#     so a container without rocm-smi still gets a count rather than a false negative.
#     Count only devices that expose a VRAM counter: /sys/class/drm enumerates display and render
#     nodes too, and a raw card* count reads 65 on a box with 8 accelerators.
gfx="$(rocm-smi --showproductname --csv 2>/dev/null | grep -oi 'gfx[0-9]\+[a-z]*' | sort -u | tr '\n' ',' || true)"
ngpu="$(ls /sys/class/drm/card[0-9]*/device/mem_info_vram_total 2>/dev/null | wc -l)"
if [ "${ngpu:-0}" -ge 8 ]; then note ok "GPUs: $ngpu device(s)${gfx:+, arch=${gfx%,}}"
else note FAIL "GPUs: found ${ngpu:-0}, need 8 on one node. This task is an 8-rank intra-node collective; EP8 is baked into the guards. There is no smaller configuration of it."; fit=1; fi
case "$gfx" in
  *gfx950*|"") : ;;
  *) note warn "arch is ${gfx%,}, not gfx950 — the LDS/CU numbers in the task text were measured on MI355X and will not transfer. Re-measure before trusting any occupancy claim." ;;
esac

# (2) Free VRAM. 150 GiB/card is a floor, not a window: an arm that starts with just enough dies
#     when a co-tenant grows. Report the minimum across cards rather than the total.
minfree=""
for d in /sys/class/drm/card[0-9]*/device; do
  [ -r "$d/mem_info_vram_total" ] && [ -r "$d/mem_info_vram_used" ] || continue
  t=$(cat "$d/mem_info_vram_total"); u=$(cat "$d/mem_info_vram_used")
  f=$(( (t - u) / 1073741824 ))
  if [ -z "$minfree" ] || [ "$f" -lt "$minfree" ]; then minfree=$f; fi
done
if [ -z "$minfree" ]; then note warn "VRAM: could not read sysfs vram counters; check free memory by hand before launching"
elif [ "$minfree" -ge 150 ]; then note ok "VRAM: ${minfree} GiB free on the tightest card"
else note WAIT "VRAM: only ${minfree} GiB free on the tightest card, need >=150. A co-tenant is resident. Assemble now, launch later: wait for the pool rather than starting an arm that dies mid-lease, which is recorded as VOID, not as a latency, and leaves the round with nothing."; ready=1; fi

# (3) MORI. The kernels call mori_shmem at TRACE time, so its absence is not a run-time import
#     error you can catch — it aborts inside codegen.
if [ -z "$MORI_ROOT_IN" ]; then
  for c in /sgl-workspace/mori "$HOME/mori" /opt/mori; do
    [ -d "$c/python/mori" ] || [ -d "$c/mori" ] && { MORI_ROOT_IN="$c"; break; }
  done
fi
if [ -n "$MORI_ROOT_IN" ] && [ -d "$MORI_ROOT_IN" ]; then
  if PYTHONPATH="$MORI_ROOT_IN:$MORI_ROOT_IN/python:${PYTHONPATH:-}" python -c "import mori" 2>/dev/null; then
    note ok "MORI: $MORI_ROOT_IN (imports)"
  else
    note FAIL "MORI: $MORI_ROOT_IN exists but 'import mori' fails under it. Build it (its own README) — the fused kernels call mori_shmem during codegen, so this is not recoverable at run time."; fit=1
  fi
else
  note FAIL "MORI: not found. Pass --mori-root <checkout> or set MORI_ROOT. The symmetric heap and the p2p primitives come from it; without it the megakernel cannot even be traced."; fit=1
fi

# (4) JIT cache. A fresh directory here is not an error — it is a full C++ rebuild inside your
#     first lease, which is a worse outcome than a clear warning now.
if [ -z "$JIT_DIR_IN" ]; then
  for c in "$HOME/.aiter/jit" /opt/aiter/jit; do
    [ -d "$c" ] && { JIT_DIR_IN="$c"; break; }
  done
fi
if [ -n "$JIT_DIR_IN" ] && [ -d "$JIT_DIR_IN" ] && [ -n "$(ls -A "$JIT_DIR_IN" 2>/dev/null)" ]; then
  note ok "JIT cache: $JIT_DIR_IN (populated)"
else
  note warn "JIT cache: ${JIT_DIR_IN:-unset} is missing or empty. The first run will rebuild AITER's C++ modules from scratch, inside the lease. Warm it OUTSIDE a lease first, or expect the first round to measure nothing."
  JIT_DIR_IN="${JIT_DIR_IN:-$HOME/.aiter/jit}"
fi

fi  # end probe

if [ "$CHECK_ONLY" = 1 ]; then
  if [ "$fit" != 0 ]; then echo "environment is NOT fit for task '$TASK'; see FAIL lines above." >&2; exit 2; fi
  if [ "$ready" != 0 ]; then echo "environment is fit but NOT launchable right now; see WAIT lines above." >&2; exit 3; fi
  echo "environment is fit for task '$TASK'."; exit 0
fi
[ "$fit" = 0 ] || die "refusing to assemble a workspace this machine cannot run; re-run with --check after fixing the FAIL lines above" 2

# ---------------------------------------------------------------- assemble
[ -n "$BASELINE" ] || die "--baseline <aiter checkout> is required (or use --check)"
[ -n "$OUT" ] || die "--out <workspace> is required (or use --check)"
[ -d "$BASELINE" ] || die "--baseline $BASELINE is not a directory"
[ -d "$BASELINE/aiter/ops/flydsl" ] || die "--baseline $BASELINE does not look like an AITER checkout (no aiter/ops/flydsl)"
STRICT_TASK="$(python3 - "$TASK_DIR/launch_args.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    v = json.load(f).get("strict_autonomy", False)
print("1" if str(v).lower() == "true" else "0")
PY
)"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  if [ "$STRICT_TASK" = 1 ]; then
    die "strict_autonomy task refuses non-empty --out $OUT; use a NEW empty workspace (strict mode ignores --force)" 1
  fi
  [ "$FORCE" = 1 ] || die "--out $OUT exists and is not empty; pass --force to overwrite"
fi

BASELINE="$(cd "$BASELINE" && pwd)"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
PARENT="$(dirname "$OUT")"
EXP_ROOT="${EXP_ROOT:-$PARENT/geak_runs}"
STATE_DIR="${STATE_DIR:-$PARENT/geak_state/$TASK}"

# A strict capability proof cannot inherit any code, patch, note or hand-edited state. Check the
# whole state directory, not only STATE.json/best; an old patch elsewhere is still browseable.
if [ "$STRICT_TASK" = 1 ] && [ -d "$STATE_DIR" ] && [ -n "$(ls -A "$STATE_DIR" 2>/dev/null)" ]; then
  die "strict_autonomy task refuses inherited state at $STATE_DIR; pass --state-dir pointing at a NEW empty directory" 1
fi

# WHERE THE LAUNCH ARGS LAND, and why it is not next to the workspace.
#
# launch_args.json carries `known_reference_paths` -- the very paths the provenance check exists to
# refuse. The default used to be the workspace's own parent directory, i.e. the directory it sits
# in, which is exactly the tree an engineer walks. That publishes the answer's address in the one
# place the containment rule says it must never appear ("grep -rn <reference> over the workspace tree
# must come back empty except for the detector tooling"), and it does it silently, at assembly time,
# in a file whose whole purpose is to hold the paths nobody is allowed to read. Reference isolation
# is by DISTANCE -- under uid 0 a chmod quarantine is inert -- so the fix has to be the location.
#
# So: with no reference declared the old default is fine and stays. With one declared, the args go
# outside the workspace's tree unless the caller says otherwise, and the choice is printed.
if [ -z "$ARGS_OUT" ]; then
  if [ -n "$KNOWN_REF" ]; then ARGS_OUT="$HOME/geak_launch/$(basename "$OUT")/launch_args.json"
  else ARGS_OUT="$PARENT/launch_args.json"; fi
fi
mkdir -p "$(dirname "$ARGS_OUT")"
ARGS_OUT="$(cd "$(dirname "$ARGS_OUT")" && pwd)/$(basename "$ARGS_OUT")"
# Refuse rather than warn. A warning here scrolls past under the probe output, and the file is
# already written by the time anyone reads it.
case "$ARGS_OUT" in
  "$PARENT"/*|"$OUT"/*)
    [ -n "$KNOWN_REF" ] && die "refusing to write launch args naming a reference path to $ARGS_OUT: that is inside the workspace tree an engineer walks. Pass ARGS_OUT=<somewhere else>." 1 ;;
esac

echo "assembling '$TASK' into $OUT"
# The workspace IS the aiter that gets imported (every command sets PYTHONPATH="$PWD"), so it has to
# be a real copy, not a symlink or a worktree of the baseline — an engineer editing it must not be
# able to reach back and mutate the denominator.
tar -C "$BASELINE" -cf - --exclude=.git . | tar -C "$OUT" -xf -

ARGS_KNOWN_REF="$KNOWN_REF"
[ "$STRICT_TASK" = 1 ] && ARGS_KNOWN_REF=""
subst() {
  sed -e "s|\${MORI_ROOT}|$MORI_ROOT_IN|g" \
      -e "s|\${AITER_JIT_DIR}|$JIT_DIR_IN|g" \
      -e "s|\${WORKSPACE}|$OUT|g" \
      -e "s|\${SKILL_DIR}|$SKILL_DIR|g" \
      -e "s|\${EXP_ROOT}|$EXP_ROOT|g" \
      -e "s|\${STATE_DIR}|$STATE_DIR|g" \
      -e "s|\${KNOWN_REFERENCE_PATHS}|$ARGS_KNOWN_REF|g" "$1"
}
subst "$TASK_DIR/GEAK_TASK.md" > "$OUT/GEAK_TASK.md"
subst "$TASK_DIR/launch_args.json" > "$ARGS_OUT"
mkdir -p "$EXP_ROOT" "$STATE_DIR"

# INLINE THE TASK TEXT INTO THE LAUNCH ARGS, because nothing else carries it.
#
# GEAK_TASK.md lands in the workspace, and the workspace is tar-copied into EVAL_DIR/workspace and
# EVAL_DIR/baseline. That felt like enough. It is not: `kernel_workflow.js` sources the task from
# `args.task` and from nowhere else (`const TASK = A.task || ''`), and neither the script nor any
# role file mentions GEAK_TASK.md by name -- both director.md and tech_lead.md document TASK as
# "may be empty" and proceed. So a wave launched from a bootstrapped workspace, using the
# launch_args.json this script emits, ran with TASK='' and never saw the four route guards, the
# rank-max rule, the relL2 bar, the path-marker requirement, or the run commands. Whether an
# engineer happened to `cat` a markdown file at the top of the tree was left to luck.
#
# Inlining is the fix rather than "tell the roles to read $WORKSPACE/GEAK_TASK.md" because the
# workflow script cannot touch the filesystem, so a filename in args is a promise only an agent can
# keep, while text in args is in the prompt by construction. It happens BEFORE the placeholder
# check below so the emitted file's no-placeholder guarantee covers the task text too.
python3 - "$ARGS_OUT" "$OUT/GEAK_TASK.md" <<'PY' || die "failed to inline the task text into launch_args.json" 2
import json, sys
args_path, task_path = sys.argv[1], sys.argv[2]
with open(args_path) as f: args = json.load(f)
with open(task_path) as f: task = f.read()
if not task.strip(): raise SystemExit("GEAK_TASK.md is empty")
args["task"] = task
args["_task_comment"] = (
    "Inlined verbatim from the workspace's GEAK_TASK.md by scripts/bootstrap_task.sh. "
    "kernel_workflow.js reads the task from args.task and nowhere else, and no role file names "
    "GEAK_TASK.md, so a launch_args.json without this field runs the wave with an empty task.")
with open(args_path, "w") as f: json.dump(args, f, indent=2); f.write("\n")
PY

# Strict runs never persist a reference address. The trusted bootstrap process converts each
# reference file that differs from the frozen baseline into an opaque manifest:
#   hash(relative path), raw content hash, comments/whitespace-normalized content hash.
# A verifier can reject exact/comment-only copies from those values without learning where the
# reference lives or even the name of a reference-only file.
if [ "$STRICT_TASK" = 1 ] && [ -n "$KNOWN_REF" ]; then
python3 - "$ARGS_OUT" "$BASELINE" "$KNOWN_REF" <<'PY' || die "failed to build opaque reference hash manifest" 2
import hashlib, json, os, re, sys

args_path, baseline, refs_csv = sys.argv[1:4]
refs = [p.strip() for p in refs_csv.split(",") if p.strip()]
exts = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".hip", ".cu", ".cuh", ".mlir", ".s"}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def normalized(data):
    text = data.decode("utf-8", "ignore")
    text = re.sub(r"(?m)^\s*(?:#|//).*$", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return digest(re.sub(r"\s+", "", text).encode())

rows = []
seen = set()
for root in refs:
    if not os.path.isdir(root):
        raise SystemExit(f"known reference is not a directory: {root}")
    for base_dir, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            if os.path.splitext(name)[1].lower() not in exts:
                continue
            path = os.path.join(base_dir, name)
            if os.path.islink(path):
                continue
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            data = open(path, "rb").read()
            baseline_path = os.path.join(baseline, rel)
            if os.path.isfile(baseline_path) and open(baseline_path, "rb").read() == data:
                continue
            row = (digest(rel.encode()), digest(data), normalized(data))
            if row in seen:
                continue
            seen.add(row)
            rows.append({"path_sha256": row[0], "sha256": row[1], "normalized_sha256": row[2]})

with open(args_path) as f:
    args = json.load(f)
args["known_reference_paths"] = ""
args["known_reference_hashes"] = rows
args["_reference_hash_comment"] = (
    "Generated out of band by bootstrap_task.sh. Contains only path/content digests for reference "
    "files that differ from the frozen baseline; no reference address or filename is persisted.")
with open(args_path, "w") as f:
    json.dump(args, f, indent=2)
    f.write("\n")
PY
fi

# Run the content/ref sweep in this trusted bootstrap process, before any agent starts. The marker
# path is supplied out of band through the environment and is never written to launch args.
if [ "$STRICT_TASK" = 1 ]; then
  [ -n "${MARKER_FILE:-}" ] && [ -f "$MARKER_FILE" ] ||
    die "strict_autonomy requires MARKER_FILE in the trusted bootstrap environment" 1
  scan_roots=("$PARENT" "$OUT" "$EXP_ROOT" "$STATE_DIR" "$SKILL_DIR" "$SKILL_DIR/../perf_knowledge")
  for scan_root in "${scan_roots[@]}"; do
    [ -d "$scan_root" ] || continue
    set +e
    MARKER_FILE="$MARKER_FILE" REF_SCAN_MAX_TREES=100000 \
      bash "$SKILL_DIR/scripts/reference_leak_sweep.sh" --tree "$scan_root"
    scan_rc=$?
    set -e
    [ "$scan_rc" = 0 ] ||
      die "strict_autonomy containment preflight failed for $scan_root (exit $scan_rc)" 1
  done
  for scan_root in "${scan_roots[@]}"; do
    [ -d "$scan_root" ] || continue
    set +e
    bash "$SKILL_DIR/scripts/skill_address_scan.sh" \
      --skills-dir "$SKILL_DIR/../perf_knowledge/expert_skills" \
      --scan-root "$scan_root" --repo "$SKILL_DIR/.."
    address_rc=$?
    set -e
    [ "$address_rc" = 0 ] ||
      die "strict_autonomy skill-address preflight failed for $scan_root (exit $address_rc)" 1
  done
  workflow_revision="$(git -C "$SKILL_DIR/.." rev-parse HEAD 2>/dev/null)" ||
    die "strict_autonomy requires GEAK to be a git checkout" 1
  if ! python3 - "$ARGS_OUT" "$MARKER_FILE" "$workflow_revision" "${scan_roots[@]}" <<'PY'
import hashlib, json, sys
args_path, marker_path, workflow_revision, *roots = sys.argv[1:]
with open(args_path) as f:
    args = json.load(f)
with open(marker_path, "rb") as f:
    marker_sha = hashlib.sha256(f.read()).hexdigest()
args["containment_preflight"] = {
    "clean": True,
    "marker_manifest_sha256": marker_sha,
    "reference_hash_count": len(args.get("known_reference_hashes", [])),
    "workflow_revision": workflow_revision,
    "roots": sorted(set(roots)),
    "content_scan": "clean",
    "skill_address_scan": "clean",
}
with open(args_path, "w") as f:
    json.dump(args, f, indent=2)
    f.write("\n")
PY
  then
    die "failed to record containment attestation" 2
  fi
fi

# A placeholder that survived substitution is a silent misconfiguration: the run starts, the path
# does not resolve, and the failure surfaces an hour later as an import error inside a lease.
if grep -n '\${[A-Z_]*}' "$OUT/GEAK_TASK.md" "$ARGS_OUT"; then
  die "unresolved placeholders above — the workspace is not runnable" 2
fi
if [ -z "$KNOWN_REF" ]; then
  echo "  note   known_reference_paths is empty, so the provenance check is DISABLED. That is correct"
  echo "         for a fresh environment with no reference implementation on disk. If one DOES exist"
  echo "         here, pass --known-reference <csv> — an unlisted reference is not refused, it is copied."
fi

cat <<EOF

done.
  workspace   $OUT
  task        $OUT/GEAK_TASK.md
  launch args $ARGS_OUT
  exp root    $EXP_ROOT
  state dir   $STATE_DIR

Before launching: read $OUT/GEAK_TASK.md, then verify the baseline runs unmodified. A denominator
you have not seen pass is not a denominator.
EOF
