#!/usr/bin/env bash
# Verify the KernelFusion separation: every path that differs from upstream GEAK main is a
# KernelFusion file listed in MANIFEST.txt, and no upstream file is modified or deleted.
#
#   bash e2e_workflow/kernel_fusion/check_main_parity.sh <upstream-main-ref>
#   (e.g. after `git fetch https://github.com/AMD-AGI/GEAK.git main`: FETCH_HEAD)
#
# Exit 0 = parity holds; 1 = a non-KernelFusion difference (listed); 2 = usage error.
set -uo pipefail
ref="${1:-}"
[ -n "$ref" ] || { echo "usage: $0 <upstream-main-ref>" >&2; exit 2; }
root="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2
manifest="$root/e2e_workflow/kernel_fusion/MANIFEST.txt"
[ -f "$manifest" ] || { echo "missing $manifest" >&2; exit 2; }
git -C "$root" rev-parse --verify -q "$ref^{commit}" >/dev/null || { echo "unknown ref $ref" >&2; exit 2; }

bad=0
# Committed and working-tree changes alike, against the upstream tree.
while IFS=$'\t' read -r status path; do
  case "$status" in
    A) grep -qxF "$path" "$manifest" || { echo "UNLISTED ADD  $path"; bad=1; } ;;
    *) echo "UPSTREAM $status  $path"; bad=1 ;;
  esac
done < <(git -C "$root" diff --name-status "$ref" -- . ; \
         git -C "$root" ls-files --others --exclude-standard | sed 's/^/A\t/')
while read -r path; do
  [ -z "$path" ] || [ -e "$root/$path" ] || { echo "MANIFEST lists a missing file: $path"; bad=1; }
done < "$manifest"
if [ "$bad" = 0 ]; then
  echo "parity OK: only KernelFusion files ($(grep -c . "$manifest")) differ from $ref"
fi
exit "$bad"
