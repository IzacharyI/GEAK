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

## Stability edit (only when the prompt carries `STABILITY-EDIT AUTHORIZED`)

By default this lane measures and characterizes; it does not blind-land a cross-rank system-scope edit.
When — and only when — the Engineer prompt contains the literal phrase `STABILITY-EDIT AUTHORIZED`, you
are cleared to LAND the scoped fix for the bimodal slow-state that keeps a correct `runnable` candidate
below a stable in-band rank-max.

- **Root cause (measured, `AITER_MEGAMOE_COMBINE_WAIT_STATS=1`):** the straggler wait lives in the
  megakernel's **combine queue per-destination-token arrival wait**, not the stage1 epoch handshake. It
  is bimodal (p50 ~sub-µs; per-wave max multiple ms) and only goes e2e-slow when a straggler lands on the
  critical path; it can escalate to a full grid hang. The fused kernel slips MORE than the scattered
  baseline because it has no per-launch resync point to drain a hiccup.
- **Scoped fix:** ~64-way **shard** the per-destination-token arrival counter; publish arrivals with a
  **workgroup/agent-scope release**, NEVER a system-scope per-token atomic; **parity double-buffer** the
  arrival wait so the all-peer wait is no longer load-bearing for single-buffer safety.
- **Mandatory safe protocol (all required before you keep the edit):**
  1. Gate the edit behind an **in-kernel DEFAULT-OFF flag** (env or compile-time) so the runnable
     baseline can never regress if the edit is disabled.
  2. **Positive-control the straggler latch on a free GPU window** first (reproduce the slow tail, then
     show the edit suppresses it) — do not measure into a contended pool.
  3. Keep **paired relL2 < 0.10** and pass the short **liveness screen with ZERO hangs**.
  4. **A/B the slow-state occupancy** (edit-on vs edit-off, same window): keep the edit only if it drives
     candidate slow-state toward the scattered-baseline rate **without** regressing fast-state speedup or
     correctness. Otherwise **revert** and record the failure mode in `next_blocker`.
- **Promotion sub-goal:** with the edit, the target is a **stable in-band rank-max** (`8192_uniform`
  inside `1.0403..1.0477`), not a fast-state-only median. Reducing slow-state occupancy toward the
  baseline rate is an explicit part of reaching `scored`. Commit the edit to this lane like any other WIP;
  never touch another lane.

## Staged authoring (only when the prompt carries `STAGED-AUTHORING AUTHORIZED`)

By default this lane may return `authoring` with only source/compile progress — which has repeatedly
degenerated into **scaffold-and-bail**: a turn emits its structured result 2 minutes in, far under its
lease, with the body never authored (`gpu_used=false`, HEAD unmoved from the scaffold), because "author
the complete 2-launch topology in one design" is read as all-or-nothing. When — and only when — the
Engineer prompt contains the literal phrase `STAGED-AUTHORING AUTHORIZED`, this discipline is in force:

- **No scaffold-and-bail.** A turn holding a GPU lease may NOT emit its structured result with only
  scaffolding, characterization, or planning as progress while the pool is free. Spend the lease
  executing. If the lease was denied / the pool was contended, bank what exists and say so in `notes`.
- **Author the NEXT single stage**, in this fixed order, when the complete fold exceeds one turn:
  dispatch → +GEMM1 → +flat GEMM2 (grid-wide GEMM1→GEMM2 coarse barrier) → +combine as the 3rd ticketed
  queue → `ret=None` launch drop (`launches==2`). Read `PRIOR_CANDIDATE.next_blocker` for where you are.
- **Build + run on-card before you emit.** Compile and run the current stage through the EP8 lease
  wrapper and record a real on-card result (`path=MEGA`, relL2, launches). Commit the verified stage;
  set `next_blocker` to the next stage.
- **Staging is a fault-localizer path, not a deliverable.** Intermediates stay `candidate_status:"authoring"`,
  are NEVER returned `runnable`/`scored`, and are NEVER committed as a *terminal* topology — the terminal
  target is always the complete 2-launch shape (this preserves the "no intermediate terminal topology"
  rule above; a cut ladder localizes faults, it is not the candidate).
- **This supersedes the skill's "one design / no intermediate topologies" as a ban on transient stages.**
  Mega-mode note #1 bans committing an intermediate *shape* as a carried-forward deliverable and (notes
  #2/#3) bans *freelancing* the low-level arithmetic — that freelanced arithmetic + an unhardened
  `mega_moe_stage1.py` spin-wait init is the real r12 cause, NOT the act of staging. Implement the skill's
  EXACT arrival-ticket/epoch-parity/spin-wait arithmetic; a per-stage on-card build+run CATCHES an
  r12-class latent fault EARLIER than one all-at-once author, so verified staging is the safer path.
- **Deadlock rule.** If an on-card arm hangs past its command timeout it is a cross-rank / grid-barrier
  deadlock on a mis-authored arrival-ticket / epoch-parity edge — let the wrapper reap it, diagnose the
  address arithmetic (implement the skill's spec exactly; hardening `mega_moe_stage1.py`'s spin-wait init
  is in scope), do not wedge the pool. Only a real on-card `[RESULT]` counts; never narrate an unmeasured
  `launches=2`.

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
  "activation": {"mode": "switch", "switches": [{"switch_name": "AITER_MEGAMOE_ROLE_PARTITION", "switch_value": "1"}, {"switch_name": "AITER_MEGAMOE_FINE_READY", "switch_value": "1"}, {"switch_name": "AITER_MEGAMOE_PIPELINE_DEPTH", "switch_value": "2"}], "switch_name": "AITER_MEGAMOE_FINE_READY", "switch_value": "1", "path_marker": "fine_ready gate", "marker_how": "grep in the built kernel / a log marker on the concurrent path"},
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

**`topology_sig` is the human-readable mirror of the structured topology descriptor** (`MEGA_TOPOLOGY_SCHEMA`:
`launches`, `fused_stages`, `combine_mode`, `g2_waves`, `site1{work_shards,dispatch_cu}`,
`site2{persist_cu,skew_cu}`, `combine_knobs{block_num,warp_num}`). Keep the two consistent: each
`key=value` clause in `topology_sig` must name a real lever the candidate actually realizes on hardware —
`launches=N` = measured launch count, `combine=queue` = combine folded as a third ticketed queue,
`g2_waves=useful8` = the GEMM2 reclaim wave scheme, and any `site1_*/site2_*/combine_*` clause = the
concurrency knob you set. If the direction carries a `DIRECTION.target_topology`, realize exactly those
levers and echo them back; do not claim a lever in `topology_sig` that the kernel does not truly take.

**AUTHOR THE COUPLED GRID AND DECLARE ALL ITS SWITCHES, or your concurrency work measures as the SERIAL
FLOOR.** M2.5's +4.71% is NOT any single lever — it is the concurrency SITES co-designed and ALWAYS-ON
as one grid. Measured one-at-a-time each site fails, and that is a trap already fallen into: SITE-4
useful8 helps (~7.35ms) but SITE-3 fine-ready ALONE regresses (atomic traffic with nothing to overlap),
SITE-1 as a STATIC tail% partition is a measured DUD (partition-on-serial-dep null), SITE-2 pipelining
was never attempted. **The sites are COUPLED:** SITE-3's overlap needs SITE-1's REAL producer/consumer
partition to have anything to overlap; SITE-2's deeper pipeline raises the register/LDS pressure SITE-3's
fences must survive. So in a deep-fusion lease author them TOGETHER:
- **SITE-1 as a DYNAMIC / load-proportional CU partition** — claim role from the work-pool head sized to
  the ACTUAL per-stage tile counts, NOT the static tail% that already measured a dud — gate
  `AITER_MEGAMOE_ROLE_PARTITION` (default `"0"`).
- **SITE-3 fine per-SBM readiness that ACTUALLY overlaps GEMM1||GEMM2 on that partition** — gate
  `AITER_MEGAMOE_FINE_READY` (default `"0"`).
- **SITE-2 depth-2 per-WG software prefetch pipeline** — gate a new default-off flag, e.g.
  `AITER_MEGAMOE_PIPELINE_DEPTH` (default `"1"` = off, `"2"` = one-tile-ahead).

Rule the build with the in-kernel **PHASE METER** (see `overlap_instrument.md` / the skill's phase-meter
section): measure REAL GEMM1||GEMM2 overlap, never per-stage roofline — stage timers rise while `mega_e2e`
falls, so summing them is meaningless. Gate each lever behind its own DEFAULT-OFF flag so the runnable
launches=2 baseline can never regress, AND declare **all** of them together in the `activation` object as
`{mode:"switch", switches:[{switch_name,switch_value}, ...], path_marker, marker_how}`. Verify runs a
paired A/B and, on `mode:"switch"` with `switches[]`, exports the WHOLE set for the CANDIDATE arm ONLY,
leaving the base arm serial — that is the only way the coupled grid is actually measured. If you declare
only ONE switch (or leave `activation` undeclared, or `mode:"default_on"` without the env set), verify
measures the rest OFF and your candidate reads ~0.447x (the ~10.5ms serial floor) no matter how correct
the code is. That was exactly the r3:a50 outcome: real fine-ready code, `launches=2`/`relL2=0.0298`
correct, yet `cand=10.5ms` because the flag stayed `0`. Staged on-card build+run each lever as you add it
(catch the r12-class substrate fault early), but the DELIVERABLE arm has ALL coupled levers ON together;
echo every flag name in a `topology_sig` clause too.
