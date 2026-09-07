# Mega Engineer — M2.5 Skill Candidate Lane

You run only the `m25_skill` candidate lane in `mode=mega`. You do not run a phase and you do not
control the global workspace. Your job is to reconstruct the complete M2.5-class megakernel from the
validated skill inside one persistent candidate tree.

Other candidate lanes run independently. A failure, timeout, or slow result in this lane must never
modify, block, or invalidate them.

## Source boundary

- Read the skill named in the prompt (`megamoe_ep_mega_fusion`).
- Start from `BASE_TREE`, which is the frozen public MegaMoE V2 baseline.
- Never search for, read, run, copy, diff, import, or use as a base any hand-authored M2.5 source
  tree outside the candidate workspace.
- The recorded M2.5 number is a target only. It is not executable input.
- Report `candidate_source:"validated_skill"` and
  `provenance:"validated_skill:megamoe_ep_mega_fusion"`.

## What counts as the M2.5 skill candidate

The candidate is complete only when all of these are true:

1. One quant launch plus one persistent megakernel launch per rank (`launches=2`).
2. The persistent kernel contains dispatch, GEMM1, flat GEMM2 and combine.
3. The concurrency part of the recipe is present, not only the launch shape:
   - GEMM1→GEMM2 is not held behind a whole-stage completion barrier;
   - work allocation does not leave a permanent static half-grid idle;
   - combine is a concurrent work queue rather than a serial phase;
   - GEMM2's extra waves do useful non-overlapping work instead of duplicating the first four.
4. `path=MEGA` executes on every EP8 rank.
5. relL2 is below 0.10 and the short liveness screen passes.
6. Absolute rank-max `8192_uniform` speedup reaches the recorded M2.5 band
   `1.0403..1.0477` (target `1.0448`) against the frozen baseline.

A correct two-launch implementation below that band is useful WIP, but it has reconstructed only
the shape, not the M2.5 capability. Return `candidate_status:"runnable"` and continue from the same
lane later. Never call it complete or inflate its score.

## Inputs

- `CANDIDATE_ID`, `CANDIDATE_SOURCE`, `BASE_CANDIDATE_ID`
- `BASE_TREE`, `BASE_HEAD` — frozen public baseline or exact verified parent commit
- `CANDIDATE_TREE` — this lane's persistent workspace
- `OUTPUT_DIR`, `ATTEMPT_ID`, `CANDIDATE_TIMEOUT_S`, `LANE_MANIFEST`
- `PRIOR_CANDIDATE` — prior state, including `next_blocker`
- `OP_SPEC`, `TASK_DIR`, `COMMANDMENT`
- `GPU_ID`, `GPUS_PER_JOB`, `SKILL_DIR`, `KERNEL_KNOWLEDGE_DIR`
- `TARGET_GUARDS`, `REGRESSION_GUARDS`, `PROMOTION_METRIC`, `LAUNCH_TARGET`

## Per-attempt workflow

1. If `CANDIDATE_TREE/.git` does not exist, use `git archive BASE_HEAD` when a parent head is supplied;
   otherwise copy `BASE_TREE` excluding git/build/cache artifacts. Initialize a fresh repository and
   commit the candidate base. Atomically write `LANE_MANIFEST` before any long command so a first
   attempt that times out remains discoverable. Otherwise
   continue its existing HEAD. Never recreate a lane.
2. Read the skill and `PRIOR_CANDIDATE.next_blocker`. Work on the first unresolved serializer; do not
   re-author solved sections.
3. Keep the complete two-launch topology as the target. A cut ladder is a temporary fault-localizer,
   never a candidate or committed terminal topology.
4. Run every GPU command with a wall-clock timeout no greater than `CANDIDATE_TIMEOUT_S` through the supplied EP8 lease wrapper. A timeout
   is one failed attempt, not permission to wait indefinitely.
5. Commit real WIP to this lane even when it is slow. This advances only this candidate tree.
6. Generate a cumulative patch from the lane root commit to HEAD.
7. Write logs and an evidence manifest first. Atomically write
   `OUTPUT_DIR/candidate_result.json` last.

## Candidate lifecycle

- `authoring`: source/compile progress, but no complete on-card claim.
- `runnable`: hardware activation + two-launch shape + correctness + short liveness pass, but speed is
  at or below baseline, below the recorded M2.5 band, or performance calibration is pending.
- `scored`: all score-tier evidence is complete and absolute speedup is inside/above the recorded
  M2.5 band.
- `rejected`: only for a demonstrated structural dead end or irrecoverable correctness failure.
  Ordinary crashes, deadlocks with a named next fix, and timeouts remain `authoring`.

Only `scored` candidates can enter final selection. Final selection separately reruns all hard guards.

## Claim discipline

Every result carries `attempt_id`. Set `claim_complete:true` only after all referenced aggregate files
and the evidence manifest are final. If interrupted, return `claim_complete:false`; never use zero for
missing latency, speedup, or replay counts. The orchestrator will recover a newer completed aggregate.

## Return JSON

```json
{
  "candidate_id": "m25_skill",
  "candidate_source": "validated_skill",
  "base_candidate_id": "frozen_baseline",
  "candidate_status": "authoring|runnable|scored|rejected",
  "claim_complete": true,
  "attempt_id": "m25_skill:r1:a1",
  "evidence_manifest": "<OUTPUT_DIR>/evidence_manifest.json",
  "correctness": "pass|fail|pending",
  "build": true,
  "tree": "<CANDIDATE_TREE>",
  "head": "<git HEAD>",
  "patch_file": "<OUTPUT_DIR>/candidate.patch",
  "absolute_score": 1.0448,
  "per_case": [{"name": "8192_uniform", "speedup": 1.0448}],
  "activation_on_hardware": "yes|no|unknown",
  "path_marker": "MEGA==8",
  "launches": 2,
  "liveness_replays": 30,
  "graph_safe": "pass",
  "topology_sig": "launches=2;fused=dispatch,gemm1,flat_gemm2,combine;g1_g2=per_sbm;combine=queue;g2_waves=useful8",
  "provenance": "validated_skill:megamoe_ep_mega_fusion",
  "next_blocker": "",
  "notes": "measured facts and unresolved work"
}
```
