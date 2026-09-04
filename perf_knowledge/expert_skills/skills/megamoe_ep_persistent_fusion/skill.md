---
id: megamoe_ep_persistent_fusion
title: 'MegaMoE EP: fuse dispatch+GEMM1+GEMM2+combine into one persistent kernel per rank (gfx950, intranode
  P2P)'
kind: expert_skill
authors:
- zhengyao
scope: kernel
match:
  operator: moe_dispatch_combine
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp8_e4m3   # a8w4: A = MXFP8 (e4m3, 1x32 block scale)
  - mxfp4        # a8w4: W = MXFP4 (e2m1, 1x32 block scale)
  regimes:
  - prefill
  - decode
  from_backend: ''
  to_backend: flydsl
  profile_signature:
    op_name_regex: mega_moe|dispatch_combine|p2p_scatter
    min_pct_gpu: 20.0
  # Precise applicability. All must hold for this card to apply; a route/precision/parallel
  # outside this envelope is a different problem and the numbers below do not transfer.
  config:
    framework: MegaMoE_v2
    parallel: EP8              # expert-parallel, 8 ranks, one intranode XGMI group
    arch: gfx950               # CDNA4 / MI355X
    precision: a8w4            # A=MXFP8(e4m3,1x32), W=MXFP4(e2m1,1x32); f32 accum; scaled-MFMA
    graph: cuda_graph_captured
    shapes:
      tokens_per_rank: [512, 8192]
      routes: [uniform, skew]  # uniform is the TARGET route; skew is diagnostic-only, out of scope
      topk: measured_per_config
expects:
  isolated_speedup_min: 1.01
  e2e_delta_min_pct: 1.0
  parity: required
# Provenance — this card is a REUSE of a human-validated capability, not an autonomous finding.
provenance:
  source: validated_skill
  origin: human_validated_capability   # hand-built persistent megakernel ("M2.5" / version D), measured on-box
  reuse_mode: production_optimization
  reporting_rule: >
    A candidate reproduced FROM this skill MUST be reported with source=validated_skill and MUST NOT
    be presented as an autonomous derivation ("GEAK discovered M2.5"). The blind autonomy proof runs
    with use_expert_skills=OFF and never reads this card; the two ledgers stay separate.
# Incumbent, not ceiling. M2.5 is the known-good performance FLOOR (保底). A candidate that beats
# these numbers on the same A/B against the immutable oracle wins and supersedes this incumbent.
incumbent:
  label: M2.5_persistent_megakernel
  is_ceiling: false
  measured_gain_vs_baseline_pct:
    tokens512_uniform: 1.49
    tokens8192_uniform: 4.71   # target route, tightest spread -> the positive control
validation:
  status: draft
  last_verified: ''
  gpu: ''
  model: ''
  measured:
    isolated: ''
    e2e_pct: ''
    parity: ''
  artifact: ''
role: advisory_prior
supersedes: []
---

## When to use

**Applies exactly when:** `MegaMoE_v2`, expert-parallel `EP8` on one intranode XGMI group, `gfx950`
(CDNA4 / MI355X), precision `a8w4` (A = MXFP8 e4m3 1×32, W = MXFP4 e2m1 1×32), CUDA-graph captured,
at `tokens_per_rank ∈ {512, 8192}` on the **uniform** route (skew is diagnostic-only, out of scope).
A route/precision/parallel-degree outside that envelope is a different problem and the numbers here
do not transfer — re-measure before reusing.

An expert-parallel MoE layer on a single intranode group (measured on 8×MI355X / gfx950, EP8) whose
dispatch → GEMM1 → GEMM2 → combine stages run as **separate kernel launches** with **zero measured
overlap** between them, and whose profile shows the cost concentrated in *waiting* rather than in
math or in bytes moved. The diagnostic that selects this skill: an all-rank barrier in combine whose
peer-wait blows up under routing skew (measured rank-max p95 `191 µs` uniform → **`876.6 µs` skew**)
while logical remote bytes differ by `<0.1%` between the two routes. If the skew regression tracked
*bytes* or *padding* instead, this is the wrong skill.

Do **not** reach for this to cut launch overhead. Measured launch cost for the whole four-kernel
chain is `≈6.4 µs` — roughly 0.1% of the skew-route runtime, so launching less is not the win.

**Where the win actually comes from (corrected 2026-09-03 against the aiter_mega M2.5 source and
831_handoff.md).** M2.5's measured +4.71% (8192_uniform) ships **default-ON** through
`AITER_MEGAMOE_FUSE_ALL` — the full-megakernel wiring of **Steps 4–5** below (CU-role-partition
GEMM1/GEMM2 + combine folded in as a third work queue), running `path=MEGA` on all ranks with no env
vars set. It does **not** come from a per-token cross-rank readiness edge. That edge (Steps 1–2) is a
**measured regression** on this hardware: M2.5's own `mega_moe_fused_s2c.py` carries it opt-in and
default-OFF because it is *slower* (8192-uniform Stage2+combine `2.0777 → 2.2334 ms`) and wedges
intermittently. On MI355X (8 XCDs, each with a private L2) any system-scope release/atomic lowers to a
cross-L2 flush — a fixed ~6.6 ms, contention-bound cost that coarsening cannot remove — so leading
with it measures ~0.4× baseline, not a gain. **Do the wiring (Steps 4–5) first; treat the readiness
edge as optional and hardware-gated, not the headline mechanism.**

## Mechanism

Four separate launches enforce a **global barrier per stage boundary** that the algorithm does not
require. Stage *N+1* cannot begin until the slowest CTA of stage *N* on the slowest rank retires,
even though most of its work depends on only a few tiles. Under skew the fast ranks pay for the hot
ranks twice — once at the GEMM1/GEMM2 boundary and again at the combine barrier — which is exactly
the shape of the `+685 µs` skew delta.

Two readiness edges are *structurally absent* from the scattered form, and both are cheap to add:

- **GEMM1→GEMM2** is intra-rank, so it needs only agent scope (`fence_agent_release` +
  `atomic_add_agent`), and it indexes cleanly because `SBM` is already the Stage1↔Stage2 metadata
  alignment (`m_row//SBM → tile_row_base`).
- **Stage2/P2P→Combine** is cross-rank. The existing payload store uses `cache_modifier=2` (an SLC
  *hint*), which is neither a release nor a fence — nothing currently tells the peer its rows are
  ready, which is precisely why the all-rank barrier exists. **Caveat (MI355X):** making this edge
  per-token with a *system-scope* atomic is a measured regression — a cross-L2 flush per token (see
  "Where the win actually comes from"). If it is attempted at all, keep the publication **local**
  (agent- or workgroup-scope release, `atomic_add_agent`) and shard the arrival counter ~64 ways to
  spread cache-line contention across the XCDs, exactly as M2.5's `mega_moe_fused_s2c.py` does; a
  cross-rank system-scope per-token atomic must never be the first move.

Once both edges exist, the stages can share one persistent kernel and a token's combine reduction
starts as soon as *its own* `topk` partials land. That is where the gain is: the exposed tail of the
slowest rank is overlapped with useful work on the fast ranks, instead of being a barrier.

The single most important reason this is worth doing as *fusion* rather than as better scheduling
between launches: the P2P payload cost is real and currently fully exposed. A matched no-payload
control moves skew Stage2+combine `2.5205 → 1.5568 ms` (**−38.2%**). That `0.96 ms` is not
removable — a correct implementation still sends the data — but it is *hideable*, and only a fused
kernel can hide it under compute from other tokens.

## Procedure

Build the fused path **opt-in behind `AITER_MEGAMOE_FUSE_ALL=1`** (`mega_moe_v2.py:433`) with the scattered path
retained, so it is measurable as a one-flag A/B rather than as a rebuild. Build it that way from the
start — the flag is what makes every number below reproducible, and it doubles as the run's
positive control.

1. **Cross-rank readiness edge — OPTIONAL, hardware-gated, do this LAST (measured regression on
   MI355X; see "Where the win actually comes from").** Attempt this only if a trace shows an exposed
   cross-rank tail *after* Steps 4–5 are landed, and only with a **local** publication
   (agent/workgroup-scope release, `atomic_add_agent`) on a **~64-way sharded** per-destination-token
   arrival counter — **never** a `fence_system_release()` + `atomic_add_system` per-token atomic,
   which is a cross-L2 flush here (~6.6 ms fixed) and is why M2.5 ships this edge default-OFF. In
   `p2p_scatter_epilog` (`mega_moe_stage2.py`), after the payload and scale stores, emit the local
   release then the sharded arrival increment in the peer's symmetric heap. Model the new state on the
   existing `TILE_READY`/`P2P_TILE_READY` `DispatchSlot`s (`dispatch.py`); the producer/consumer slot
   addressing already matches (`slot = dest_lid*topk + s` on both sides), so no index decode is
   needed.
2. **Replace the consumer barrier.** In `flydsl_dispatch_combine_intranode_kernel.py`, swap the
   Stage-2 all-rank barrier for a per-token `wait_until_equals(arrival[tok], topk_expected[tok])`
   followed by `fence_system_acquire()` immediately before that token's Stage-3 reduction.
3. **Break the single-buffer hazard.** `shmem_comb_inp_tok` is single-buffered and zeroed once at
   construction. That reuse is the *actual reason* the barrier was required, so removing the barrier
   without fixing it is a stale-read bug that passes single-shot correctness. Double-buffer it with
   a parity index (reuse dispatch's proven epoch/parity discipline), and reset arrival counters for
   the *next* parity, never the current one.
4. **Absorb GEMM1 with a CU-role partition, not an LDS union.** Stage1 sits at `159744 B` LDS
   (97.5% of a CU) → 1 WG/CU; Stage2 needs `66560 B`. They cannot co-reside. Assign disjoint CTA
   sets to GEMM1 and GEMM2 roles via the existing arrival-ticket mechanism
   (`mega_moe_stage1.py:210-225`), each with its own footprint, sized to the 256-CU budget.
5. **Fold combine in as a third work queue** rather than a phase. Lifting the Stage-3 reduction into
   a per-work-item emitter is the prerequisite, and is worth ~2.2pp on its own — the megakernel with
   combine still a separate launch is only ~+2.5-3.0pp of the total.
6. **Print a path marker once per process.** An opt-in fast path that silently fails
   its predicate produces a plausible wrong number that reads as "fusion didn't help". A result
   without the marker is void, not zero.

**Measured effect of steps 1–5**, isolated (same tree, only `AITER_MEGAMOE_FUSE_ALL` varying, quant a
separate launch on both arms, A,B,A,B ×3 pairs per guard, rank-max `mega_e2e`, path marker verified
on all 24 runs):

| guard | scattered (med) | megakernel (med) | gain (med) | per-pair range |
|---|---|---|---|---|
| 512 uniform | 0.7264 ms | 0.7156 ms | **+1.49%** | +1.24 .. +1.79% |
| 512 skew | 0.7858 ms | 0.7662 ms | **+1.54%** | **−0.76** .. +3.70% |
| 8192 uniform | 4.6754 ms | 4.4699 ms | **+4.71%** | +3.55 .. +4.93% |
| 8192 skew | 5.5267 ms | 5.3775 ms | **+2.09%** | +1.48 .. +3.13% |

Large-uniform (`+4.71%`, tightest spread) is the guard to use as a positive control; 512-skew is
**not** — one of its three pairs came back negative.

## Executable verification

The procedure above is the apply-template; **`verify.sh` (next to this card) is the executable gate**
that decides whether a candidate reproduces the skill. It takes no reference source — it drives the
one-flag A/B on the candidate tree only, and encodes the five acceptance checks as pass/fail:

1. **Two-launch shape.** Assert the terminal form is exactly *quant, then one megakernel* — grep the
   captured graph for two ops per rank, fail on a fused-quant regression.
2. **Path marker.** `AITER_MEGAMOE_FUSE_ALL=1` must print the once-per-process fusion marker; a run
   without the marker is **void, not zero** (`grep -c 'path=MEGA'`, expected == world_size).
3. **relL2 parity.** Fused vs scattered output, `relL2 ≤ 0.10` at `tokens=8192`; parity is `required`.
4. **1000-replay stress.** ≥1000 CUDA-graph replays with a wall-clock timeout, comparing the *last*
   iteration to the first — catches an arrival counter that desyncs on generation 2 (invisible to a
   single-shot check).
5. **Paired performance.** A,B,A,B ×3 interleaved pairs per guard, rank-max `mega_e2e`, on the four
   guards; report medians and per-pair ranges. `uniform` guards gate; `skew` is reported, not gated.

Run it as `bash verify.sh --tree <candidate_aiter> --world-size 8 [--incumbent <m2.5_aiter>]`. It
writes a JSON verdict (`selected`, `decision`, `vs_incumbent_pct`) and exits non-zero if any *gating*
check fails. See `megamoe_ep_persistent_fusion.validation.yaml` for the shapes/routes the on-box
validator feeds it.

## Three-way comparison (Baseline / M2.5 incumbent / new candidate)

`verify.sh --three-way` measures all three arms under one identical route/iteration command and one
denominator (the **frozen public-AITER baseline**):

| arm | what it is | role |
|---|---|---|
| **Baseline** | four serialized launches/rank (quant → Stage1 → Stage2 → Combine), no fusion | denominator; every % below is vs this |
| **M2.5 (incumbent)** | this skill's persistent megakernel; `source=validated_skill` | **known-good floor (保底)**, not the ceiling |
| **New candidate** | whatever the current run produced | must clear Baseline; **is allowed to beat M2.5** |

M2.5 is the incumbent to **match-or-beat**, not a target to converge to. Pass `--incumbent <m2.5_aiter>`
and the floor is **enforced**, not merely reported: the candidate's fused path is paired against the
incumbent's fused path (A,B,A,B ×3, rank-max, on the decisive `8192_uniform` guard) and the selection
is **`max(candidate, M2.5)`**:

- candidate beats M2.5 **past the noise floor** (`NOISE_PCT`, default 1.45%) → `decision: supersede`,
  `selected: candidate` — ship it, bump `incumbent.label`, re-record gains, keep `is_ceiling: false`.
- within noise → `decision: tie_keep_incumbent`, `selected: M2.5_incumbent` — the capability is
  reproduced (not a regression), but deployment keeps the known-good incumbent.
- candidate slower → `decision: regress_keep_incumbent` — **the floor holds; M2.5 ships, never the
  slower candidate.**

So the floor guarantees deployment is **never below M2.5**, while a real win is always allowed to
supersede it. The base gate (beat Baseline + parity + marker + 1000-replay) still decides whether the
candidate is *valid at all*; the floor rule decides *what ships* among the valid arms. Add
`--require-improvement` to make a tie/regress (no gain over the incumbent) a non-zero gate exit for
callers that specifically want an improvement, not just a safe reproduction.

## Knobs & pitfalls

- `payload_chunk_rows` — `256` at the 512-token bucket, `384` at 8192. This was measured, not
  reasoned; it does not follow a monotone rule, so re-sweep it per bucket rather than extrapolating.
- **Write-through P2P** on the payload store is a separate accepted win; keep it when fusing.
- **Every wait needs a paired acquire.** `mori_shmem.*_wait_until_*` performs a *relaxed system load
  that does not invalidate L2*. A wait without `fence_system_acquire()` before consuming the guarded
  data reads stale bytes and is correct on the first iteration.
- **Grid-wide barriers require full co-residency** (`_check_block_num_resident`, cap = #CU). A
  participant index outside the resident window deadlocks at one problem size and runs fine at
  another, so correctness must be run at both the smallest and largest shape the harness offers.
- **Single-shot correctness cannot see this class of bug.** Add a ≥1000-iteration repeat stress with
  a wall-clock timeout comparing the *last* iteration to the first: a counter that desyncs on
  generation 2 is invisible to a first-iteration comparison.
- **Rank-max, never rank-mean.** A collective is gated by its slowest rank; rank-mean can improve
  while the operator gets slower.

## Do-no-harm notes

- **Do not fold quantization into the megakernel.** Tried and retired: an ingress quant role reaching
  exactly one launch per rank measured **`+0.15%` to `+1.2%` slower** across the guards. Quant belongs
  in the upstream producer's epilogue (a `--prequant` path already exists). "DeepGEMM-like" does not
  mean one launch — DeepGEMM consumes pre-quantized FP8 and ships the cast separately, maximising
  occupancy inside the GEMM body rather than minimising surrounding launch count. The terminal form
  here is **two launches: quant, then one megakernel.** The code is kept default-off as a control
  experiment rather than deleted.
- **Do not try to reach 2 WG/CU on Stage1 by shrinking LDS.** The `cs_size*4 = 131072 B` f32
  CShuffle slab dominates `lds_pool_bytes`, and the configuration is additionally pinned at the
  **256-VGPR** ceiling. Static ISA screening closed this: the target `<81920 B` is not reachable
  without changing the tile shape in a way that costs more than the occupancy buys. Stage1
  *source-level scheduling* is a different, still-open direction (an early measurement showed
  −0.77% geomean, 4/4 favourable pairs at 8192) — do not close that one on this evidence.
- **Keep the all-rank barrier behind a fallback flag.** A liveness failure in the field needs an
  immediate escape hatch, and the gate needs the A/B.
- The gains here are **single-digit percent**, and the measured per-case noise floor on this harness
  reaches **1.45%** at 1 rep. Any claim in that range measured with fewer than 5 interleaved pairs
  and no null arm is unreadable, not a result. See the workflow's positive-control gate.
- Denominator discipline: measure against the **frozen public-AITER baseline** under an identical
  route/iteration command. Do not quote a speedup against MORI or Taco.

## Provenance & reporting

`provenance.source: validated_skill`. This card is the **reuse of a human-validated capability** (the
hand-built persistent megakernel, "M2.5" / version D, measured on-box) — legitimate in
**production-optimization mode** as reuse of validated capability, *not* plagiarism. Two hard rules
keep the two ledgers from contaminating each other:

- **A result reproduced from this card is reported with `source=validated_skill`.** It must **not** be
  presented as an autonomous derivation — never "GEAK discovered M2.5". The three-way report labels
  the arm accordingly.
- **The blind autonomy proof does not read this card.** Strict/blind runs set `use_expert_skills=OFF`
  (default), so the byte-identity criterion is untouched; this channel is opt-in and production-only.

The card carries **mechanism, never the built copy**: no reference branch/commit/path, no literal
diff of the answer. That is the same containment rule `scripts/skill_address_scan.sh` enforces —
"reproducible because it describes what to build, not because it says where the built copy is parked."

## Sources

- Isolated fusion A/B: 3 interleaved pairs per guard, rank-max `mega_e2e`, path marker verified on all
  24 runs. The medians and per-pair ranges are the table above — that table **is** the log, digested.
  The raw logs are **deliberately not cited by path**, for the same reason the commit is not: they are
  the fused arm's own output, and a run that opens them has confirmed the answer instead of deriving it.
- Step-2 bottleneck evidence — kernel DAG showing zero overlap, instrumented combine peer-wait,
  matched no-payload control, ATT waitcnt/barrier shares, LDS residency. **Also not cited by path.**
  Every number this card relies on is quoted inline above; reproducing the evidence is part of the
  work, not a prerequisite handed over with it.

  **When your own profile disagrees with a number here, the size of the disagreement decides what to
  do, and "trust your own profile" is only correct for the small case.**
  - Within ~2x: trust your profile, say so, move on. Different tooling, different machine state.
  - Beyond ~5x: **you and this card are not measuring the same thing.** Do not pick a side, and in
    particular do not close an optimisation axis on the strength of your own control. Enumerate the
    plausible readings of the experiment, run each one, and report which reading reproduces which
    number. A closed axis backed by an unadjudicated 5x disagreement is a defect, not a finding.

  This rule exists because it was already violated once, expensively. Wave 14 closed the axis
  "hiding the Stage2 P2P payload cost" — the exact mechanism behind this card's largest measured win
  (+4.71% at `8192_uniform`) — because its own no-payload control moved `stage2_combine` by <=1.2%
  while this card reports -38.2%. That is ~30x. The two controls are almost certainly not the same
  experiment: deleting the P2P store instructions is not the same as deleting the store instructions
  **and** the arrival wait that follows them, and only the second can move the number by a third.
  Nobody ran both. The highest-payoff axis in the campaign was closed on the coin-flip.
- Implementation history: **deliberately not cited here by branch or commit.** This card is injected
  into capability-evaluation runs, where an engineer is asked to derive the megakernel; a commit
  address turns that question into a fetch. Two waves already filed candidates 8-of-11 and 11-of-12
  files byte-identical to a reachable copy, and this card named six commits at the time. The history
  is archived outside every run tree and the address is recorded in the project's private ledger
  (handoff, "reference containment"), not in engineer-readable knowledge. Everything needed to
  rebuild it is the mechanism above; `scripts/reference_leak_sweep.sh` enforces that this stays true.
- Reusable in-kernel machinery: `mega_moe_stage1.py:210-225` (arrival-ticket roles), `:230-240` +
  `:266-279` (ABA-safe epoch/parity), `dispatch.py:576-597` + `:204-219` (tile-ready counters),
  `communication_ops_utils.py:49-154` (system release/acquire).
- Prior art for the "unfused megakernel" control arm (one launch, one global event, identical
  operator code — the cleanest isolation of fine-grained-dependency gain): Event Tensor, MLSys 2026,
  arXiv 2604.13327v2 §4.5. Note its distributed dynamic-scheduler arm measured **0.82–0.89×**, i.e.
  slower — remote task-queue push overhead. Treat dynamic cross-rank scheduling as unproven here.
