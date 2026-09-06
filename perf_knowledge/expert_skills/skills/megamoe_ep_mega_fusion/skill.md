---
id: megamoe_ep_mega_fusion
title: 'MegaMoE EP mega mode: author the complete dispatch+GEMM1+GEMM2+combine persistent megakernel in one design (gfx950, intranode P2P)'
kind: expert_skill
authors:
- zhengyao
mode: mega
scope: kernel
match:
  operator: moe_dispatch_combine
  arch_class:
  - '*'
  gens:
  - gfx950
  dtypes:
  - mxfp8_e4m3
  - mxfp4
  regimes:
  - prefill
  - decode
  from_backend: ''
  to_backend: flydsl
  profile_signature:
    op_name_regex: mega_moe|dispatch_combine|p2p_scatter
    min_pct_gpu: 20.0
  config:
    framework: MegaMoE_v2
    parallel: EP8
    arch: gfx950
    precision: a8w4
    graph: cuda_graph_captured
    shapes:
      tokens_per_rank:
      - 512
      - 8192
      routes:
      - uniform
      - skew
      topk: measured_per_config
expects:
  isolated_speedup_min: 1.01
  e2e_delta_min_pct: 1.0
  parity: required
provenance:
  source: mega_mode_reproduction
  origin: author_plus_optimize_fusion
  reuse_mode: mega_fusion
  reporting_rule: 'This card is injected ONLY under MODE=mega (megafusion mode: author+optimize combined,
    capability_eval=false, strict_autonomy=false). In mega mode faithful REPRODUCTION of the M2.5 topology
    is the GOAL, not a violation — there is no derived-not-copied / criterion-5 / artifact-distinctness
    requirement here (that ledger belongs to the separate strict/blind task and is untouched). Report the
    resulting arm plainly as the mega-mode fused candidate measured against baseline and against the M2.5
    floor. The ONE boundary that is NOT lifted: never read anything under /root/geak_reference/ — the
    concrete arithmetic in this card was authored from the reachable in-tree baseline (mega_moe_stage1.py)
    plus the r12 root-cause diagnosis, not copied from any out-of-tree reference source.

    '
incumbent:
  label: M2.5_persistent_megakernel
  is_ceiling: false
  measured_gain_vs_baseline_pct:
    tokens512_uniform: 1.49
    tokens8192_uniform: 4.71
validation:
  status: mega_only   # NOT 'validated' on purpose: this fork is injected by MODE=mega (explicit path),
                      # never by index.yaml auto-match — so it can never collide with the persistent_fusion
                      # card in optimize/author runs. measured.* below still records the M2.5 capability.
  last_verified: '2026-09-04'
  gpu: gfx950/MI355X
  model: ''
  measured:
    isolated: 1.0448
    e2e_pct: 4.48
    parity: pass
  artifact: /sgl-workspace/megamoe/geak_handD_eval
role: advisory_prior
supersedes: []
---

## READ THIS FIRST — mega runs in TWO phases; which one you are decides which half of this card you follow

mode=mega is not one job. It is a **Reproduce phase** followed by an **Optimize phase**, and this card
serves both — but they read DIFFERENT halves of it. Find your phase before you read further.

- **You are the `repro_engineer` (Reproduce phase).** Your ONLY job is to **faithfully reproduce the
  whole flat 2-launch fused operator as the FLOOR (保底)**. Your recipe is the **"Construction skeleton —
  the single persistent grid"** section below (the concrete arrival-ticket / epoch-parity / spin-wait
  arithmetic) — **plus** the r12 substrate init fix. **GEMM2 stays FLAT.** You do **NOT** innovate, do
  **NOT** do incremental half-fusion, do **NOT** do the `g2-collapse`, and do **NOT** treat the
  cut-ladder as a build method (it is a bring-up diagnostic only — you commit the whole flat operator,
  never a cut). **Everything under the banner "⚠️ OPTIMIZATION-PHASE FAILURE LORE" is NOT a build method
  for you** — skip it as construction guidance; it exists to warn the Optimize phase off the moves that
  faulted. If you find yourself designing something this card does not describe, that is drift — stop and
  reproduce.
- **You are an optimize-phase role (`tech_lead` / `engineer` / `deep_engineer`).** The Reproduce phase
  has ALREADY committed the whole flat floor as HEAD. Your job is to **beat the floor** — stack speed
  rungs on the whole operator, or offer an alternative COMPLETE fused candidate — never re-decompose the
  floor into a single-edge ladder, never ship a half-fused intermediate. The FAILURE LORE section IS for
  you: it tells you which cleverness (notably the `g2-collapse`) carries which fault risk. The floor is
  permanently retained; a candidate is admitted only if it beats the floor.

Both phases share the arithmetic authorization, the r12 first-class substrate fix, and the one
un-lifted boundary: **never read anything under `/root/geak_reference/`.**

## Mega mode — what this card is, and how it differs from the persistent_fusion card

This is the **mega-mode** fork of `megamoe_ep_persistent_fusion`. It is injected by `MODE=mega`
(the megafusion workflow: author-mode + optimize-mode combined), not by `index.yaml` auto-match. Its
mechanism, measured numbers, knobs and do-no-harm notes are identical to the persistent_fusion card —
that engineering is correct and unchanged. **Three things are deliberately different**, and they are the
whole reason this fork exists:

1. **No bankable-rung ladder.** The persistent_fusion card builds the fused body as an incremental
   chain of enabling rungs (A-resident GEMM2 → barrier-gated fold → CU-role partition → optional
   cross-rank edge), each committed and carried forward. That ladder existed to survive a shared,
   un-screenable lease under the *blind* methodology. **Mega mode does not use it.** You author the
   **complete 2-launch topology in one design** (quant, then one persistent megakernel wiring
   dispatch→GEMM1→GEMM2→combine). Do **not** invent intermediate half-fused topologies — the GEMM1-only
   single-launch, the barrier-only-no-combine shape, etc. Those intermediate states are new artifacts
   M2.5 never had, and the last wave's fatal fault lived precisely inside one of them (see r12 below).
2. **Concrete machinery, not lossy-by-design.** The persistent_fusion card states its address
   arithmetic is "generic on purpose … not code to transcribe" and points only at line-refs, because
   its leak-sweep KEEP forced it to stay lossy. **That KEEP is lifted for mega mode.** This card spells
   out the exact arrival-ticket / epoch-parity / spin-wait address arithmetic below, authored from the
   reachable in-tree baseline. You reproduce it faithfully instead of re-deriving it — re-derivation is
   exactly what produced the r12 defect.
3. **Fixing the baseline substrate is a first-class action.** `mega_moe_stage1.py` was inherited by the
   last wave as "clean substrate" and never suspected until r12 traced the fatal fault into its
   spin-wait address init. In mega mode, hardening that init **is allowed and expected** — it is not
   out of scope, it is the known blocker.

**The one boundary that is NOT lifted:** never read anything under `/root/geak_reference/`. The
concrete arithmetic here was authored from `mega_moe_stage1.py` (in-tree, reachable) + the r12
diagnosis, not from any reference copy.

## When to use

**Applies exactly when:** `MegaMoE_v2`, expert-parallel `EP8` on one intranode XGMI group, `gfx950`
(CDNA4 / MI355X), precision `a8w4` (A = MXFP8 e4m3 1×32, W = MXFP4 e2m1 1×32), CUDA-graph captured,
at `tokens_per_rank ∈ {512, 8192}` on the **uniform** route (skew is diagnostic-only, out of scope).
A route/precision/parallel-degree outside that envelope is a different problem and the numbers here
do not transfer — re-measure before reusing.

An expert-parallel MoE layer on a single intranode group (measured on 8×MI355X / gfx950, EP8) whose
dispatch → GEMM1 → GEMM2 → combine stages run as **separate kernel launches** with **zero measured
overlap** between them, and whose profile shows the cost concentrated in *waiting* rather than in
math or in bytes moved. The diagnostic: an all-rank barrier in combine whose peer-wait blows up under
routing skew (measured rank-max p95 `191 µs` uniform → **`876.6 µs` skew**) while logical remote bytes
differ by `<0.1%` between the two routes.

Do **not** reach for this to cut launch overhead. Measured launch cost for the whole four-kernel chain
is `≈6.4 µs` — roughly 0.1% of the skew-route runtime, so launching less is not the win.

**Where the win actually comes from (corrected 2026-09-03 against the aiter_mega M2.5 source and
831_handoff.md).** M2.5's measured +4.71% (8192_uniform) ships **default-ON** through
`AITER_MEGAMOE_FUSE_ALL` — the full-megakernel wiring of **Steps 4–5** below (CU-role-partition
GEMM1/GEMM2 + combine folded in as a third work queue), running `path=MEGA` on all ranks with no env
vars set. It does **not** come from a per-token cross-rank readiness edge. That edge (Steps 1–2) is a
**measured regression** on this hardware: M2.5's own `mega_moe_fused_s2c.py` carries it opt-in and
default-OFF because it is *slower* (8192-uniform Stage2+combine `2.0777 → 2.2334 ms`) and wedges
intermittently. On MI355X (8 XCDs, each with a private L2) any system-scope release/atomic lowers to a
cross-L2 flush — a fixed ~6.6 ms, contention-bound cost that coarsening cannot remove — so leading with
it measures ~0.4× baseline, not a gain. **Do the wiring (Steps 4–5); treat the readiness edge as
optional and hardware-gated, not the headline mechanism.**

## Mechanism

Four separate launches enforce a **global barrier per stage boundary** that the algorithm does not
require. Stage *N+1* cannot begin until the slowest CTA of stage *N* on the slowest rank retires,
even though most of its work depends on only a few tiles. Under skew the fast ranks pay for the hot
ranks twice — once at the GEMM1/GEMM2 boundary and again at the combine barrier — which is exactly the
shape of the `+685 µs` skew delta.

Two readiness edges are *structurally absent* from the scattered form:

- **GEMM1→GEMM2** is intra-rank, so it needs only agent scope (`fence_agent_release` +
  `atomic_add_agent`), and it indexes cleanly because `SBM` is already the Stage1↔Stage2 metadata
  alignment (`m_row//SBM → tile_row_base`).
- **Stage2/P2P→Combine** is cross-rank. The existing payload store uses `cache_modifier=2` (an SLC
  *hint*), which is neither a release nor a fence. **Caveat (MI355X):** making this edge per-token with
  a *system-scope* atomic is a measured regression — a cross-L2 flush per token. If attempted at all,
  keep the publication **local** (agent- or workgroup-scope release, `atomic_add_agent`) and shard the
  arrival counter ~64 ways; a cross-rank system-scope per-token atomic must never be the first move.

Once both edges exist, the stages share one persistent kernel and a token's combine reduction starts
as soon as *its own* `topk` partials land. The P2P payload cost is real and currently fully exposed: a
matched no-payload control moves skew Stage2+combine `2.5205 → 1.5568 ms` (**−38.2%**). That `0.96 ms`
is not removable but it is *hideable*, and only a fused kernel can hide it under compute from other
tokens.

## Procedure

Build the fused path **behind `AITER_MEGAMOE_FUSE_ALL=1`** (`mega_moe_v2.py:433`) with the scattered
path retained, so it is measurable as a one-flag A/B and doubles as the run's positive control. In mega
mode you author all of Steps 2–6 in the **first** design pass (Step 1 stays optional/last), not as a
staircase.

1. **Cross-rank readiness edge — OPTIONAL, hardware-gated, do this LAST** (measured regression on
   MI355X; see "Where the win actually comes from"). Attempt only if a trace shows an exposed cross-rank
   tail *after* Steps 4–5 land, and only with a **local** publication (agent/workgroup-scope release,
   `atomic_add_agent`) on a **~64-way sharded** per-destination-token arrival counter — **never** a
   `fence_system_release()` + `atomic_add_system` per-token atomic. In `p2p_scatter_epilog`
   (`mega_moe_stage2.py`), after the payload and scale stores, emit the local release then the sharded
   arrival increment in the peer's symmetric heap. Model the new state on the existing
   `TILE_READY`/`P2P_TILE_READY` `DispatchSlot`s (`dispatch.py`); the producer/consumer slot addressing
   already matches (`slot = dest_lid*topk + s` on both sides).
2. **Replace the consumer barrier.** In `flydsl_dispatch_combine_intranode_kernel.py`, swap the Stage-2
   all-rank barrier for a per-token `wait_until_equals(arrival[tok], topk_expected[tok])` followed by
   `fence_system_acquire()` immediately before that token's Stage-3 reduction.
3. **Break the single-buffer hazard.** `shmem_comb_inp_tok` is single-buffered and zeroed once at
   construction. That reuse is the *actual reason* the barrier was required, so removing the barrier
   without fixing it is a stale-read bug that passes single-shot correctness. Double-buffer it with a
   parity index (reuse dispatch's proven epoch/parity discipline), and reset arrival counters for the
   *next* parity, never the current one.
4. **Absorb GEMM1 with a CU-role partition, not an LDS union.** Stage1 sits at `159744 B` LDS (97.5% of
   a CU) → 1 WG/CU; Stage2 needs `66560 B`. They cannot co-reside. Assign disjoint CTA sets to GEMM1 and
   GEMM2 roles via the existing arrival-ticket mechanism (`mega_moe_stage1.py:210-225`), each with its
   own footprint, sized to the 256-CU budget.
5. **Fold combine in as a third work queue** rather than a phase. Lifting the Stage-3 reduction into a
   per-work-item emitter is the prerequisite, worth ~2.2pp on its own — the megakernel with combine
   still a separate launch is only ~+2.5-3.0pp of the total.
6. **Print a path marker once per process.** An opt-in fast path that silently fails its predicate
   produces a plausible wrong number that reads as "fusion didn't help". A result without the marker is
   void, not zero.

### Round-1 whole-topology (mega mode — replaces the bankable-rung ladder)

The persistent_fusion card lands the body as a staircase of enabling rungs because its blind
methodology could not author the whole body in one lease. **Mega mode discards that.** You are
author-mode + optimize-mode combined: you author the **complete 2-launch topology in one design**,
aligned to M2.5's full operator decomposition (the same function boundaries, the same control flow, the
same outputs), and then optimize/measure it as the existing optimize pipeline does.

Rules for mega mode:

- **One design, whole topology.** dispatch → GEMM1 → GEMM2 → combine, all four stages wired into ONE
  persistent launch (plus the separate quant launch = the terminal 2-launch shape) in the first pass.
- **No intermediate half-fused topologies.** Do not build a GEMM1-only single launch, a barrier-only
  shape with combine still a separate launch, or any other partial as a *committed step*. They are new
  artifacts M2.5 never had and are where the last wave's fatal fault lived (r12). The prerequisites the
  ladder called "rungs" (A-resident GEMM2, parity double-buffer, CU-role partition) are just *parts of
  the one design* here — author them together, not as separately-promoted topologies.
- **Positive control ≠ intermediate topology.** You still keep the `AITER_MEGAMOE_FUSE_ALL` A/B and the
  cut-ladder (crash_bisection) as *diagnostics* on the whole body — those are the tools that localize a
  fault inside the one topology. They are not an instruction to ship a half-fused kernel.
- **Fix the substrate when it blocks you.** If the whole-topology body faults or hangs in the
  arrival-ticket / spin-wait substrate (it did — r12), harden `mega_moe_stage1.py` directly (see the
  concrete arithmetic and the r12 hazard below). That file is in the modifiable set.

### Construction skeleton — the single persistent grid (CONCRETE, mega mode)

ONE launch. Grid = the full CU budget (256 CU on gfx950), launched co-resident so a grid-wide barrier
is legal — participant count must equal the resident count, or a participant outside the resident
window deadlocks at one size and runs at another (see Knobs). Every workgroup stays resident and runs
one top loop until all three queues drain:

    role   = role_of(block_id)      # disjoint partition of resident WGs (Step 4):
                                    #   GEMM1 | GEMM2 | COMBINE, each sized to its own LDS footprint.
    parity = replay_parity          # flips every graph replay (Step 3); selects the double-buffered
                                    #   slice this generation reads/writes.

    loop:
        item = claim_next(queue[role], parity)     # atomic ticket off this role's counter
        if item is DRAINED: break

        if role is GEMM1:
            r = gemm1(item)
            publish(ready1[item], scope=agent)      # fence_agent_release + atomic_add_agent;
                                                    #   index by SBM (m_row//SBM -> tile_row_base)
        elif role is GEMM2:
            wait_until(ready1[deps(item)]); acquire(scope=agent)
            r = gemm2(item)     # FLAT per-tile compute — port the body of `run_unit` from the in-tree,
                                #   reachable `mega_moe_stage2.py` (the a8w4 MXFP4 GEMM2 tile: routed-token
                                #   gather -> dequant -> MFMA -> epilog) into this work-pool role, ONE plain
                                #   tile per claimed ticket. This is the genuinely UNBUILT join — author it
                                #   flat, NOT as a g2-collapse write layout. NO collapse = no shared-consumer
                                #   gather under SGPR pressure = the cut4 fault is never created (that is the
                                #   whole bet: cut3's flat GEMM1-through path is already clean+correct on-card).
            store_payload_writethrough(peer, item, parity)
            # Cross-rank publish OPTIONAL/hardware-gated/LAST (Step 1). Base form: leave the all-rank
            # barrier as COMBINE's gate behind the fallback flag.
        elif role is COMBINE:                       # the THIRD queue (Step 5), not a phase:
            wait_until(arrival[item] == topk_expected[item])
            acquire(scope=system)
            out[item] = reduce_topk(inp[item, parity])
            reset(arrival[item, other(parity)])     # reset the NEXT parity, never this one

    once_per_process: emit "path=MEGA"

**Concrete arrival-ticket / epoch-parity / spin-wait address arithmetic.** This is the exact machinery
the persistent_fusion card left lossy. It is authored from the reachable in-tree baseline
`mega_moe_stage1.py` (compact-dispatch owner/producer handshake, lines ~185–282). Reproduce it
faithfully; do not re-derive it. Identifiers below are the baseline's own.

*Setup (per workgroup, top of kernel):*

    disp_rsrc     = make_buffer_from_addr(addr_disp, Int64)     # dispatch-slot table
    parity_rsrc   = make_buffer_from_addr(addr_parity, Int32)   # 1 lane: current parity
    expected_rsrc = make_buffer_from_addr(addr_expected, Int32) # per-parity expected count
    a_entry_count = disp_ptr(ENTRY_COUNT)      # base pointers, each = buffer_load(disp_rsrc, slot, Int64)
    a_epoch_gate  = disp_ptr(EPOCH_GATE)
    a_launch_ready= disp_ptr(LAUNCH_READY)     # p_launch_ready = disp_ptr(P2P_LAUNCH_READY) for peers

*Ticket claim (tid 0 does the atomic, broadcasts through LDS):*

    ticket64   = atomic_add_agent(a_entry_count + grid_epoch_slot*8, 1)   # Int64 slot stride = 8 bytes
    generation = ticket64 // launch_grid_x
    ticket     = Int32(ticket64 - generation*launch_grid_x)
    gate_addr  = a_epoch_gate + grid_epoch_slot*4                         # Int32 gate stride = 4 bytes
    gate_epoch = Int32(generation + 1)
    compact_owner    = (ticket == 0)
    compact_producer = (0 < ticket <= dispatch_blocks); producer_slot = ticket - 1

*Owner path (ticket==0) — ABA-safe epoch/parity flip, then publish the gate:*

    old_parity       = buffer_load(parity_rsrc, 0, Int32)
    next_parity_lane = old_parity ^ 1
    previous_expected= buffer_load(expected_rsrc, next_parity_lane, Int32)
    next_expected    = previous_expected + fz_npes
    buffer_store(expected_rsrc, next_parity_lane, next_expected, Int32)
    launch_epoch_lane= (next_expected // fz_npes)*2 - next_parity_lane
    next_parity  = readfirstlane(next_parity_lane)      # collapse to a scalar before any store
    launch_epoch = readfirstlane(launch_epoch_lane)
    ... (cross-rank launch_ready handshake over peers, system release/acquire) ...
    s_waitcnt(0); fence_agent_release(); buffer_store(parity_rsrc, 0, next_parity, Int32)
    s_waitcnt(0); fence_agent_release(); store_i32_system(gate_addr, 0, gate_epoch)  # PUBLISH

*Non-owner path — spin-wait on the gate, then acquire:*

    int32_wait_until_equals(gate_addr, gate_epoch)      # base=gate_addr, predicate=gate_epoch
    fence_agent_acquire()

*After the gate, every WG loads the just-published parity and its expected count:*

    payload_parity   = buffer_load(parity_rsrc, 0, Int32, SC0)
    payload_expected = buffer_load(expected_rsrc, payload_parity, Int32, SC0)

Three invariants separate "passes single-shot" from "passes the 1000-replay stress":

- **Parity discipline (Step 3).** COMBINE reads parity `P` while the next generation fills `P`'s
  complement; the counter reset touches the complement, never `P`. This is the ABA-safe flip above
  (`next_parity_lane = old_parity ^ 1`, expected updated on the *next* lane). Skipping it is the stale
  read the barrier was hiding — correct on iteration 1, desynced on iteration 2.
- **Disjoint roles (Step 4).** GEMM1 at `159744 B` LDS and GEMM2 at `66560 B` cannot co-reside; the
  partition is by block, enforced by the ticket, never by an LDS overlay.
- **Combine is a queue, not a barrier-gated phase (Step 5).** Its items unlock per-token as partials
  arrive, so combine-role CUs drain the exposed P2P tail concurrently with the last GEMM work.

### Six fixes the persistent grid must be BORN with (replay-confirmed on-card, in-tree-derivable)

These are not optimizations and not debugging lore — they are construction requirements. Each was
confirmed by on-lease replay, and each is derivable from the reachable in-tree baseline
(`mega_moe_stage1.py` / `mega_moe_v2.py` / `mega_moe_stage2.py` / `gemm2.py`). Build the grid WITH them from round 1;
authoring without them reproduces a known device fault or a silent-wrong result, not a faster path.
This is design experience, NOT a transcript of any reference source.

> **TILE-CONFIG GATE — validate the TARGET tile config, not the easy one. The floor is measured at
> 8192_uniform, which routes through `SBM=128 / g2_BM=64 / kMChunks=4`. The small shapes (bs=128/512)
> route through `BM=32`, a DIFFERENT and easier GEMM2 compute path. A clean bs=128 proves the topology
> and the handshake — it does NOT prove the floor, because the flat GEMM2 at BM=64 is a separate,
> harder path that has never validated. Treat "bs=128 clean" as bring-up, never as done. Gate
> `authoring_status:"complete"` on the BM=64 target route passing relL2<0.10 — attack BM=64 FIRST, not
> after the BM=32 shapes are polished.**

1. **The GEMM1→GEMM2 release fence lives in the COLD post-drain region, never the hot MFMA loop.**
   A GEMM1-role block does its full MFMA compute loop with NO cross-role fence inside it; then, ONCE,
   after the block has drained its pool, it emits a single `fence_agent_release()` + one
   `atomic_add_agent` bump of `ready1[0]`. Placing any release fence *inside* the hot consumer loop
   triggers a spill-independent backend regalloc device fault (SGPR pressure on the MFMA body) — it is
   NOT a source bug you can chase; the fix is purely placement. Index the ready flag by `SBM`
   (`m_row // SBM → tile_row_base`), and gate GEMM2 on the coarse `ready1 == NUM_G1_BLOCKS` count.

2. **GEMM2 compute scratch must NOT overlap the epilog's LDS metadata slots.** The persistent grid
   packs GEMM1 (`~159744 B`), GEMM2 (`~66560 B`) and combine into one launch; their LDS regions are
   disjoint by role, but within the GEMM2 role the flat compute scratch (`g2_lds_base_i32`, taken from
   the a_buf pool base) and the epilog's cached-metadata slots (`fa_lds_packed_off`,
   `fa_lds_weight_off`, `fa_lds_peer_off`, laid out above `fa_compute_lds_bytes` at
   `packed = compute`, `weight = packed + g2_BM*4`, `peer = weight + …`) share the same physical LDS.
   If the compute scratch address range overlaps those slots it clobbers the cached packed/weight to 0
   and every output row collapses to peer0/tok0/slot0 (looks like relL2≈1.0, "empty scatter"). Assign
   the offsets so the ranges are strictly non-overlapping — verify the arithmetic, do not assume.

3. **Under `AITER_MEGAMOE_FUSE_ALL=1`, force `grid_mult=1`.** The persistent grid needs the whole
   launch co-resident (1 WG/CU) so the grid-wide `ready1` handoff/barrier is legal — participant count
   must equal resident count. The bounded bucket=512 default selects `grid_mult=2`, which
   oversubscribes the grid so it is NOT co-resident and the `ready1` wait hangs (bs=512 deadlock). Pin
   `grid_mult=1` on the fused path in `mega_moe_v2.py`; the scattered (default-OFF) path keeps its own
   selection.

4. **The scatter epilog reads packed/weight DIRECTLY from global, not from an LDS cache.** In
   `p2p_scatter_epilog`, plumb `stids_rsrc` / `sweights_rsrc` / `srcmap_row_base` and read
   `sorted_pos = srcmap_row_base + row`, then `buffer_load(stids_rsrc, sorted_pos, …)` /
   `buffer_load(sweights_rsrc, sorted_pos, …)` per row — mirroring the already-working peer-table
   direct read. This is the robust form even independent of fix (2): it removes the epilog's dependence
   on an LDS slot that the fused GEMM2 scratch can reach. AMD buffer-resource loads bounds-check
   (OOB→0), so pad rows are safe and are NOT a correctness hazard here.

5. **The B-scale (e8m0) view MUST be bounded to the tensor's exact byte extent.** In `gemm2.py`, the
   a8w4 GEMM2 dequant reads a per-block e8m0 scale through `bscale_views` (`make_bscale_view`). If that
   view is built WITHOUT a record-count clamp (while the A-scale `make_ascale_view` IS bounded), the
   HIGHEST n-block of a compact-path tile whose expert index `e` lands at/past the last local expert
   reads OFF THE END of the scale region — a garbage e8m0 exponent byte (~`0xFE`) becomes a scale
   ~`2^127`, and an fp4-max-6 accumulator explodes to ~`2^126` → genuine Inf/NaN in exactly the top
   n-block (looks like ~51 poisoned rows at bs=512). Clamp the view: plumb the scale's total dword
   extent (`fa_bscale_total_dw` through `mega_moe_stage1.py`) and pass it as the `make_bscale_view`
   record bound so OOB→0. **Diagnostic caution learned on-card:** `comb_inp` is `int16` storage — read
   it with `.view(torch.bfloat16)` (bitcast), NOT `.float()` (int cast), or a probe will report a
   phantom `2^15=32768` and mislead the root-cause. Verify the value as bf16 bits before theorizing.

6. **The GEMM2 role must clamp its addressing wave to the tile's NATIVE wave count (4), not inherit the
   grid's launch wave count.** This is the single most load-bearing fusion-specific fix — it was the
   root cause of BOTH the bs=8192 catastrophic garbage AND the bs=512 finite-huge residual (one bug,
   not two). The flat GEMM2 tile geometry is structurally **4-wave**: `wave_n = BN//4` (BN=256 → 4
   waves × 64 cols), and the A-load `gather_base_row = wave*(BM//4)`, the B-address
   `n_block*BN + wave*(BN//4)`, and the epilog column `wave*wave_n` ALL assume 4 waves. The standalone
   (scattered) stage2 launches GEMM2 with `block=(256,1,1)` = exactly 4 waves, so it is correct. But
   the persistent fused grid inherits **GEMM1's** `num_waves=8` (`_select_large_stage1` selects 8 for
   all buckets), so if the GEMM2 role computes `g2_wave = tid//64 = 0..7` UNGUARDED, waves 4–7 compute
   the NEXT n-block's columns 256–511 and scatter them under THIS tile's `n_block_idx` → OOB / mis-scatter.
   The damage is tile-config-dependent (why fix (5)'s frontier looked scale-shaped): **catastrophic at
   BM=64** (the 8192 target route), **finite-garbage at BM=32 when SBM>BM** (the bs=512 residual), and
   **harmless only at bs=128** where `SBM==BM==32`. Fix: clamp the GEMM2 addressing wave to `(wave & 3)`
   so the role uses 4-wave geometry regardless of the grid's launch wave count. With this in place the
   fused path matches the scattered baseline at all three shapes (bs=8192 relL2≈0.0592, byte-for-byte
   the scattered number).

**Correctness is COMPLETE with fixes 1–6.** The fused path now validates clean at ALL three shapes —
bs=128 (relL2≈0.053), bs=512 (relL2≈0.053), and the **target bs=8192 (relL2≈0.0592, matching the
scattered baseline byte-for-byte)** — with `path=MEGA` ×8. The GEMM2 flat work-pool is proven at the
target tile config (BM=64). No g2-collapse was ever committed; the whole flat bet held.

**The 2-launch structural floor is COMPLETE.** Combine is folded in as the THIRD role/queue (Step 5 of
the skeleton): a CU partition running the per-token top-k reduction, gated by the GEMM2→COMBINE arrival
handshake with the parity double-buffer (reset the NEXT parity, never the current), WITHOUT disturbing
the now-clean GEMM2, GEMM2 kept FLAT (no g2-collapse). The trailing `comb_op.combine_no_stage1` launch
is dropped (`AITER_MEGAMOE_LAUNCH_TRACE` shows `combine_dropped=True`) → terminal `launches == 2`
(`quant` + ONE persistent megakernel carrying `dispatch → GEMM1 → GEMM2(flat) → combine`), 8192_uniform
validated with `fp8_blockwise_1x32` p2p_quant, ≥30 clean replays across {128,8192}×{uniform,skew}.
`authoring_status:"complete"`.

**What is NOT yet done is PERFORMANCE.** The complete, correct 2-launch floor measures **~0.23×** the
scattered baseline — it reproduces M2.5's shape, not M2.5's speed. Closing that ~4.5× gap is the
concurrency work deconstructed in the next section ("The concurrency M2.5 has that the correctness-floor
does NOT"); it is the OPTIMIZE phase's job, not a structural gap in the floor.

*Methodology caution (learned on-card, kept because it cost several rounds):* a "force scale to neutral"
probe must be verified to actually produce 1.0 after every dequant word transform — the
`G2_FORCE_BSCALE/ASCALE` hack wrote a raw `0x7F7F7F7F` word that `shift_scale_word` turned into a HUGE
scale, so "forced neutral" was actually "forced huge," and a whole round's "scale-independent"
conclusion was built on it and had to be retracted. The real bug was never scale-related at all (it was
the 4-vs-8 wave fault, fix 6). Confirm a probe's effect before trusting any result built on it, and be
ready to discard a localization when the instrument under it is shown to be broken.

### The concurrency M2.5 has that the correctness-floor does NOT — deconstructed from the scattered path

**Read this as the SECOND HALF of the deconstruction.** The six fixes above make the fused operator
*correct* — same 2-launch shape as M2.5, all shapes relL2-clean, `path=MEGA` ×8. They do NOT make it
*fast*: the correctness-floor measures **~0.23× the scattered baseline** (19.9 ms vs 4.56 ms @
8192_uniform), while M2.5 is **+4.71%** (1.047×). That whole ~4.5× gap is **pure concurrency** — the
floor reproduces M2.5's SHAPE but serializes what M2.5 OVERLAPS. The "where the win comes from" note
above (Steps 4–5 wiring: CU-role partition + combine-as-queue) is topology-true but **concurrency-
silent**: it says WHAT to wire, not how the wiring stays overlapped instead of collapsing into a
sequence. Building the wiring without the discipline below is exactly what produces a correct 0.23×
floor. This section is the missing half — and it is **derivable from the in-tree scattered path**, whose
full-width overlapped execution is a readable, already-validated known-good (NOT from any reference).

**The concurrency yardstick is the scattered path (in-tree, readable).** With `FUSE_ALL` OFF, `_run_joint`
launches each stage SEPARATELY (`mega_moe_v2.py:327-328`): `_run_fused_stage1` then `_run_stage2`. Each
launch (a) occupies the **full** `cu_num` grid (`mega_moe_stage2.py:565` `launch_cu_num=min(cu_num,persist_cu)`,
`:575` `grid_blocks=launch_cu_num`, `:538` `block=(256,1,1)`), and (b) keeps its **own tuned `grid_mult>1`**
oversubscription for latency hiding (`mega_moe_v2.py:291` "scattered path keeps its tuned grid_mult").
So each scattered stage runs at full CU width with latency hidden. That is the effective width+throughput
the single fused grid has to REBUILD inside one launch — the fused operator does not get to lose it for free.

**The four sites where the single-grid wiring serializes or under-utilizes what must overlap** — each is
a real **correctness price** the floor paid for single-launch co-residency, NOT a free slowdown you can
just delete. The win is the realization that keeps correctness AND recovers the overlap:

1. **Static CU-role partition → each stage runs on a CU SUBSET (~half width), not the full grid.**
   The floor splits blocks into GEMM1-role and GEMM2-role by ticket (Step 4, "Disjoint roles"). Price:
   GEMM1 (`159744 B` LDS) and GEMM2 (`66560 B`) genuinely cannot co-reside on one CU, so a partition is
   mandatory. But a STATIC ~50/50 split idles CUs whenever the two stages' work is unbalanced or phased.
   *Derived target:* make the partition **load-proportional / dynamic** (claim role from a work-pool head
   sized to actual per-stage tile counts) so neither role strands CU width — realization on-card.

2. **`grid_mult=1` (fix 3) → the per-stage oversubscription latency-hiding is gone.** Co-residency forces
   1 WG/CU, so the scattered path's `grid_mult>1` throughput trick is unavailable. Price: the grid-wide
   `ready1` handoff is only legal when participant count == resident count. *Derived target:* hide the
   same latency WITHIN the co-resident grid — more tiles in flight per role (deeper per-WG software
   pipelining / more k-chunks staged) instead of more WGs — realization on-card.

3. **Coarse `ready1 == NUM_G1_BLOCKS` barrier → ZERO GEMM1‖GEMM2 overlap.** Fix 1 gates GEMM2 on ALL
   GEMM1 blocks finishing (`mega_moe_v2.py:114-116`) — GEMM2 cannot touch a tile until the last GEMM1
   block drains. Price: the coarse count was the correctness-SAFE handshake (a per-tile fence in the hot
   loop faults — fix 1). *Derived target:* **fine-grained per-SBM readiness** — a GEMM2 tile starts the
   instant ITS `tile_row_base` GEMM1 inputs land, so GEMM1 and GEMM2 run concurrently across the grid.
   Realization on-card, and expect it to re-expose the exact co-residency / hot-loop-fence traps the
   correctness phase hit (fixes 1 & 3) — that coupling is WHY it is hard, and why it is a genuine
   on-card discovery rather than a source edit. Combine (Step 5) is already a per-token queue draining
   concurrently — keep it that way, never re-gate it behind a phase barrier.

4. **The 4-wave clamp (fix 6) leaves HALF the GEMM2 role's waves doing redundant work → the fused GEMM2
   runs at ~half its scattered wave-throughput.** This is a WAVE-level under-utilization distinct from
   site 1's CU-level half-width. The persistent grid launches at GEMM1's `num_waves=8` (`_select_large_stage1`),
   but GEMM2 tile geometry is natively 4-wave, so fix 6 clamps addressing to `(wave & 3)` — waves 4–7
   then recompute waves 0–3's columns (redundant), producing nothing useful. The scattered stage2 avoids
   this by launching `block=(256,1,1)` = exactly 4 waves (no idle waves). The fused grid CANNOT relaunch
   the GEMM2 role at 4 waves — a single launch has uniform block dims fixed by the GEMM1 role's 8-wave
   need. Price: correctness (fix 6) required neutralizing waves 4–7, and neutralizing was safer than
   using them. *Derived target:* give waves 4–7 the **NEXT n-block's** columns 256–511 (advance the
   B-address / epilog-column base by `+BN` for the high half) so all 8 waves do useful, non-overlapping
   GEMM2 work — doubling the role's wave-throughput and recovering what the correctness clamp gave up.
   Realization on-card; this touches the same n-block addressing that fix 6 had to get right, so it is a
   direct extension of a confirmed fix, not a new subsystem.

#### D1 CONFIRMED on-card (optimize round_1, +1.322× over the coarse-barrier floor) — the site-3 realization, folded like fixes 1–6

Site 3's *derived target* (fine-grained per-SBM readiness) is now a **replay-confirmed construction**:
`MEGA_FINE_READY=1` measured **15.042 vs 19.885 ms rank-max @8192_uniform (paired) = +1.322×** over the
committed floor, correctness intact (relL2<0.10 all shapes, `path=MEGA`×8, near-identical to the floor).
Same binary topology, gated by `MEGA_FINE_READY` (default 1 = this; 0 = the coarse-barrier floor → a
clean A/B). Derived entirely from the in-tree modifiable files (`mega_moe_stage1.py`,
`communication_ops_utils.py`) — NOT from `/root/geak_reference/`. The construction, exactly:

1. **Publish/acquire via `AtomicRMWOp`, never a `FenceOp`.** Add two helpers in
   `communication_ops_utils.py`: `atomic_add_agent_release(addr,val)` (agent-scope RELEASE fetch-and-add
   = the GEMM1→GEMM2 publish, ordering all prior output stores to the whole-GPU/cross-XCD-L2 coherence
   point WITHOUT a standalone fence) and `atomic_acquire_agent(addr)` (agent-scope ACQUIRE via
   fetch-add-0), both lowering to a single `_llvm_d.AtomicRMWOp(add, ptr, val, ordering, syncscope=Agent)`.
   **`AtomicRMWOp` is the proven-safe-in-loop instruction class; `FenceOp` (fence_agent_release/acquire)
   faults as a spill-independent regalloc artifact inside the high-SGPR loop** — this is why the r3 floor
   had to hoist its fence to the cold region. The atomic replaces the fence and can sit in-loop.

2. **THE KEY NEGATIVE FINDING — the wall is PLACEMENT, not the atomic and not a source-init.** A 3-arm
   `cut=8` bisection proved **ANY agent atomic in the producer POST-MFMA region — monotonic OR release,
   offset-0 OR per-tile, ANY address pin (readfirstlane / manual lo-hi / none) — faults with a
   spill-independent backend regalloc wild-address, the SAME class as cut4**, while the SAME instruction
   is safe at the producer loop-top and in the cold post-drain region. So the cure is atomic **PLACEMENT**,
   not a pin and not an init. This **falsifies the r3 "atomic exonerated, the fence is the poison" claim**
   (the atomic faults in the hot region too) and supersedes the earlier "readfirstlane-pin fixes the
   runtime wild address" theory — the pin is neither necessary nor sufficient in the post-MFMA region.

3. **PRODUCER publish — DEFERRED one iteration to the low-pressure loop-top.** `_fine_publish(tile)` is
   called **unconditionally at the `scf.while` producer loop-top** (the same position as the safe
   `a_work_head` atomic), so only a scalar tile index — not the atomic — crosses the MFMA. Inside it:
   `s_waitcnt(0)` + `fx.barrier()` run on **every thread every iteration** (uniform control flow — a
   workgroup barrier inside a runtime `if` can hang on AMD even when the predicate is uniform), draining
   the tile's GEMM1 output stores; only the counter bump is `tid==0`- and valid-tile-guarded, via the
   agent-RELEASE atomic at `g2_ready1 + (tile//N_TILES)*4`. The just-produced tile is carried to the next
   loop-top by `prev_work`, a loop-carried `scf.while` iter_arg — which **MUST be bound unconditionally
   before the loop** (`prev_work = fx.Int32(-1)`); initializing it inside `if const_expr(...)` leaves the
   `__while_after` block referencing an unbound value → **UnboundLocalError at trace time** (this was the
   "DEFER0 crash" — a trace-time bug, not a device fault).

4. **CONSUMER acquire gate — per-tile, no whole-stage barrier.** Each GEMM2 tile blocks until ITS
   sort-block has all `N_TILES` GEMM1 n-subtiles released — `int32_wait_until_greater_than(ready1[sb],
   N_TILES-1)` with `sb == m_block` (holds because `g2_BM==g2_SBM`) — then an `atomic_acquire_agent`
   add-0 makes the released stores visible to this consumer's A-loads. **r12 idiom: readfirstlane-pin the
   per-tile gate address** (`m_block` is wave-uniform via the `g2_ws` broadcast) so the base can't be
   recomputed from a clobbered VGPR. There is **no whole-stage pre-loop barrier** under fine_ready — each
   tile acquires its own sort-block, so GEMM2 consumes ready tiles while GEMM1 still produces later ones
   (the actual overlap). Correspondingly **skip the coarse `ready1[0]→NUM_G1_BLOCKS` handoff entirely**
   under fine_ready (bumping `ready1[0]` would corrupt sort-block 0's per-tile count, and the fine
   consumer never waits on it).

This is the site-3 serializer removed. Sites 1, 2, 4 remain *derived targets* to attack on the floor that
now sits at +1.322× — attack one serializer per lease, positive-control the overlap meter first, fold each
back here as confirmed exactly as this rung was.

**Two second-order axes, RESOLVED by in-tree inspection (do not chase them as open levers).** An audit
asked whether intra-stage inner-loop tuning and occupancy are additional un-deconstructed gaps. They are
not — the code settles both:

- **Inner-loop schedule is already at parity — it is a PRESERVED INVARIANT, not a derived target.** The
  fused GEMM2 role reuses `gemm2_compute_v2` from `.gemm2` **byte-for-byte** (`mega_moe_stage1.py:28-37`
  module comment: "reuse the in-tree stage2/gemm2 building blocks so the GEMM2 tile computed inside the
  persistent grid is byte-for-byte the same math as the scattered"), and the GEMM1 role IS the native
  stage1 body. So the k-loop pipeline depth, MFMA schedule and A/B LDS double-buffering are NOT degraded
  relative to scattered — there is no inner-loop throughput gap to recover. The obligation is to KEEP it
  that way: site 4's wave-reclaim (and any role restructure) must keep calling the same
  `gemm2_compute_v2`, never hand-roll a de-pipelined loop to make the addressing simpler.
- **Occupancy is pinned by co-residency, so "more WGs" is a dead lever.** `grid_mult=1` (fix 3) forces
  1 WG/CU; cross-WG occupancy cannot rise without breaking the `ready1` handoff. The ONLY throughput
  levers left are therefore intra-WG and already captured above: pipeline depth (site 2) and reclaiming
  the idle GEMM2 waves (site 4). Do not spend a lease trying to raise occupancy by adding WGs — it is
  structurally unavailable on the fused path.

**Discipline — this half obeys the same rule as the six fixes.** The concurrency MODEL here (full-width,
overlapped, latency-hidden) is *reconstructed from the readable scattered path*, whose overlap is a
known-good in-tree fact — not copied from `/root/geak_reference/`. The specific lowering-safe
REALIZATION of each derived target is what the OPTIMIZE phase confirms on-card (positive-control overlap
meter first, then attack one serializer per lease) and folds back here as a confirmed construction step —
exactly as fixes 1–6 were folded after replay. Until a target is on-card-confirmed it is labeled *derived
target*, never asserted as the fix. Do not write a guessed concurrency edit in as fact — that repeats the
retracted-assertion failure (methodology caution above).

### The ruler for this half — build it IN the kernel when the external A/B can't attribute overlap
The one-flag A/B differential meter (fuse OFF vs ON, paired rank-max — also the run's positive control)
answers *"did e2e move"*. It does NOT answer *"did it move BECAUSE the intended GEMM1‖GEMM2 edge started
overlapping"* — a latency win with no measured overlap change is SUSPICIOUS, not a win (a config side
effect can move e2e without creating overlap). When attribution is ambiguous, the vendor SoL path is a
dead end on this workload (rocprofiler cannot attribute PMCs across the 8-proc + CUDA-graph replay of one
long persistent kernel — do not spend a lease fighting it). **We own the kernel source, so put the ruler
INSIDE the kernel** — and for THIS problem an in-kernel phase meter is *better* than SoL, because the
quantity we need is phase-overlap and CU-fill, not generic unit utilization:
- **Phase cycle split:** read the on-chip realtime counter (`s_memrealtime`) at each phase boundary
  inside the persistent grid; accumulate GEMM1-role / coarse-barrier-idle / GEMM2-role / combine cycles
  per block into a **preallocated global scratch** (never a host sync mid-kernel — that is illegal during
  graph capture, same rule the COMMANDMENT flags for `G2_DEBUG .item()/synchronize`); dump post-kernel.
  → the direct read of *where the 19.9 ms goes* and whether barrier-idle falls when a fix lands.
- **Barrier spin count:** count the `ready1` spin-wait iterations a GEMM2-role block burns → a DIRECT
  serialization-vs-overlap measure (falls toward 0 as per-tile readiness starts real overlap).
- **Concurrent-CU count:** each block records its CU (`HW_ID`) + a timestamp → reconstruct how many CUs
  are live per phase → the DIRECT half-width-idle measure (site 1 / D2).
This turns *one lease = one yes/no fault test* into *one lease = a full phase table* — it directly buys
down the falsification-round cost. **Caveat (non-negotiable): the instrument perturbs.** It adds
instructions and SGPR/VGPR pressure, and this kernel is pressure-sensitive (the cut4 fault was
regalloc-under-pressure, not a source bug). So keep it lightweight, **env-gated OFF by default**, and it
must carry **its own null control** — prove the instrumented build's floor timing matches the
uninstrumented floor within the noise floor BEFORE trusting any reading. A dead/ timing-moving instrument
gives confident wrong numbers; that exact trap cost the OPTIMIZE phase two rounds (see FAILURE LORE
"instrument silently dead for two rounds"). Verify the ruler before you trust a rung verdict.

## ⚠️ OPTIMIZATION-PHASE FAILURE LORE — NOT a build method (repro_engineer: skip this whole section)

> **Everything from here down to "Measured effect of steps 1–5" is a record of how the OPTIMIZE phase
> faulted when it introduced the `g2-collapse` (a clever collapsed GEMM2 write layout that is NOT part
> of the faithful floor).** It is debugging history — the cut4 wild-address fault, the r3/r6/r7/r8 ISA
> diagnosis, the rocgdb instrument, the falsified axes — kept so the optimize phase does not re-walk a
> closed axis.
>
> **`repro_engineer`: this is NOT your recipe and NOT a sequence of build steps. Your recipe is the
> "Construction skeleton" ABOVE, with GEMM2 FLAT and NO g2-collapse.** The whole reason the faithful
> flat floor is worth building is that it **never introduces the g2-collapse write set**, so it never
> creates the SGPR pressure that drives this fault. If your faithful flat operator faults somewhere,
> use the cut-ladder to *locate* which stage and fix the flat operator or the r12 substrate init — do
> NOT adopt any collapse/redirect "fix" from below as construction. Skip to "Measured effect" and
> "Three-way comparison".
>
> **Optimize phase:** if you reintroduce the g2-collapse to chase speed, THIS is the fault you own and
> the axes already closed against it. Read it before spending a lease here.

### r12 — the cut4 g2-collapse fault (OPTIMIZE-phase history; the g2-collapse is not in the floor)

> **★★★ r7 CORRECTION — round_7 (this wave), on-device distinct-binary arm + ISA + fault-VA form-shift.
> READ THIS FIRST; it RETIRES the round_6 "MECHANISM UNIFIED / faulting atom = the shared GEMM1
> `tile_row_base` gather" claim (kept below only as history) and RESOLVES the r3↔r8 tension in favor of
> r8: the fault is a c4-backend SGPR-pressure artifact on a SOURCE-INVISIBLE shared-consumer gather, NOT
> `tile_row_base` and NOT any source-visible routing/offset index.**
>
> round_7 made the `tile_row_base` descriptor `s[24:27]` **uniform AND valid**: `readfirstlane` the loaded
> `v6` token-row index BEFORE the `0x1c00 (=7168=model_dim)` stride, collapsing the `.LBB0_182` waterfall in
> the cut4 ISA (lane-waterfall markers 214→3; `s[24:27]` now built purely from scalars). **cut4 STILL
> faults** (rc=1, 8 distinct per-rank wild VAs) → **`tile_row_base` is NOT the faulting descriptor.**
> DECISIVE NEGATIVE #13.
>
> **Form-shift that pins the mechanism:** with `tile_row_base` uniform, the fault VAs **returned to the
> CANONICAL band (<2^47)** — unlike r6's non-canonical (>2^47) garbage-high-bit VAs. Making the source
> token-row index uniform-valid **removed the garbage-index SYMPTOM but not the fault** → the wild access is
> now a **valid-LOOKING but wrong** effective address on a **DIFFERENT (backend-perturbed) descriptor** —
> exactly the r8 signature: a shared-consumer off-form atom's i64 base perturbed by the g2 write set's SGPR
> pressure, **spill-independent and source-invisible** (no source offset expression touches it).
>
> **CLOSED on-device — do NOT re-spend a lease on the source-index axis (r5/r6/r7 all closed it):**
> - index-VALUE clamp (r5: `_ic1` read-index, `_ic2` loaded-expert-id),
> - uniform-**broadcast** of `srcmap_row_base` (r6),
> - `readfirstlane` the loaded token-row index → uniform-VALID descriptor (r7).
>
> r7 proved **`tile_row_base` / any source-visible index is non-causal**; clamping or pinning it again
> repeats the r6 "not on the descriptor register" mistake. The old lever (c) ("predicate the shared GEMM1
> gather") targeted `tile_row_base`, so **it too is closed as framed.**
>
> **The two live levers (STATE `suggest_next`), pick one per lease:**
> - (1) **LIVE rocgdb re-attach on the r7 uniform-`tile_row_base` binary** (`HSA_ENABLE_INTERRUPT=1`,
>   precise-memory ON, hang the faulting wave, `info reg pc` + registers) to read the **NEW** faulting PC
>   directly — with `tile_row_base` ruled out, this **names the DIFFERENT shared-consumer gather** now
>   faulting. Highest-information move: stop inferring the atom, read it (this is the r3 method, re-run on
>   the r7 binary).
> - (2) the **stronger real SGPR decouple to ~61** (cut3's `.sgpr_count`) with a **same-variant
>   `dc1`-vs-`dc0` A/B** — the r8 driver, still never driven to target (prior decouple stalled at
>   `sgpr=76`, cut3 is `61`).
>
> cut3 stays clean+correct (relL2 0.059205 @ 8192, path=MEGA==8) — the controls are valid instruments.
> THIRTEEN axes falsified; D0 open after 16 attempts. Confirm the current step in STATE before acting.

> **★ ROOT CAUSE FLIPPED — round_3 (this wave), LIVE rocgdb attach with a hardware register read. READ
> THIS BEFORE the falsification ledger below; it SUPERSEDES the "wild uninitialized base / c4-backend
> regalloc SGPR-pressure" verdict that the rest of this section builds toward.**
>
> The 11-round "wild *base* / backend regalloc" verdict is **wrong**. The post-mortem coredump avenue is
> dead (torchrun SIGKILLs rank-0 at 2 GB of a 59 GB dump, register file truncated away), so the PC was
> read by a **live rocgdb attach** (`HSA_ENABLE_INTERRUPT=1`, hang the faulting wave, `info reg pc` +
> registers in-hardware). It named the faulting atom directly:
> - **Fault atom:** `buffer_load_dwordx4` @cut4 ISA **line1776**, inside the `.LBB0_182` **readfirstlane
>   descriptor waterfall** — a **data-dependent MoE gather**.
> - **It is fed a WILD LOADED ROUTING INDEX, not a wild base:** register `v6` reads **lane0 = 0x1 (valid),
>   lanes 1-63 = 0x7676eef1 (garbage)**, and only under `D2_CUT>=4` (g2-collapse). The gather forms a
>   ~4.5 TB wild VA from that garbage index and faults.
> - **The index PRODUCER is the single un-falsified lever:** `buffer_load_dword v6, v6, s[24:27]` (the
>   `.LBB0_182` descriptor-waterfall load that *produces* the routing index) returns garbage in **lanes
>   1-63 only, and only when g2 collapses.**
>
> **This is data-flow corruption of a *loaded index*, not a base, and not register allocation.** It
> reconciles the two things the base/regalloc verdict never could: (a) round_1's live-range **decouple did
> not clear the fault** (it only weakly cut SGPR, 76 vs cut3's 61) — because the fault was never a
> base/pressure problem; (b) every source-*base*/offset/extent/write axis below was correctly falsified —
> the corrupted quantity is a **value loaded at runtime**, which none of those axes touch.
>
> **DO NOT pursue "lower c4 SGPR pressure / decouple the base live-range" (old fix 2c) as the cut4 fix — it
> is falsified on-hardware.** The lever is the **routing/offset-index producer**: find why
> `buffer_load_dword v6,v6,s[24:27]` returns per-lane garbage in lanes 1-63 exclusively under g2-collapse
> (a divergent-descriptor / EXEC-mask / per-lane-offset defect in the collapsed layout, not a pressure
> knob). Everything below is the **historical falsification ledger** — true as elimination, superseded as
> to the fix. Confirm the current step in STATE before acting.

> **LIVE DIAGNOSIS is in the wave's STATE, not here — READ THIS FIRST, then read STATE.** cut4 has been
> through **seven falsified axes** and, worse, **two rounds of broken instrument**. Do not treat any single
> fix recipe in this section as "the answer"; they are the falsification record. What is durable:
>
> **Why the diagnosis thrashed (round_5 root-caused it): the instrument was silently dead for two rounds.**
> round_3/round_4's on-device readings measured **nothing** — (a) the carried fused body **never traced at
> cut4** (FlyDSL trace-time `NameError`: a point-of-use pin defined above a `while`/`if` is invisible inside
> the generated branch closures), and (b) A/B toggles were written as a **bare `if PYCONST:`**, which in
> FlyDSL is **not** a compile-time branch — it lowers to a device `scf.if` and measures the wrong thing.
> Both are fixed in `mega_moe_stage1.py`. **This retroactively VOIDS round_4's "write-to-read-only-page /
> undersized-write-set" reframe** (it was read off a body whose guards were no-ops). See
> `knowledge/crash_bisection.md` → "Verify the instrument before you trust a rung verdict" — assert a
> per-cut trace/compile marker and use `const_expr(...)` for every cut/toggle, or you are debugging air.
>
> **The durable signature (instrument fixed): verdict-(iii) — a wild 64-bit effective address on the g2
> READ substrate, NOT a write and NOT a register the `ready1` publish perturbs.** Evidence that survives:
> the 8 ranks fault at **distinct, non-deterministic VAs** in the same high-canonical mmap band
> (`0x7e..`/`0x7f..`), each canary/redirect at a *different* VA — a wild address being *formed*, not an
> index overrun of a valid base and not a fixed RO page. cut3 stays **clean + correct** and bit-identical
> scattered-vs-fused (relL2 0.059205 @ 8192, path_mega 0 vs 8) — the fused body through GEMM1 is correct on
> all 8 ranks; only the cut4 g2-collapse access faults.
>
> **The narrowing (do NOT re-open any of these — NINE axes exhausted on-device):** (2a) gate
> base/predicate SGPR-pin + source-visible base-pinning, (2b) g2 buffer-resource extent-clamp, (2c)
> `addr_ready1` live-range, the two write-atom suspects (ready1 / `a_work_tail`), the up-front regfix pins,
> **and (round_6) all four g2 READ *bases* — `comb_inp`/`trb`/`stids`/`swts` — plus the num_records
> extent-clamp, and (round_7) the source-visible raw-pointer OFFSET itself (ready1 offset-canary).**
> Two structural falsifications pin the class: **clamp-immune** (round_6 — clamping the buffered read extent
> did not move the fault ⇒ the wild access is **not** a `create_buffer_resource` read) and **offset-canary-
> immune** (round_7 — perturbing the source g2 offset arithmetic did not move it ⇒ it is **not** any
> source-visible offset either). **The fault is now out of the source entirely — every source axis (base,
> offset, write, extent) is exhausted.** The single live axis is the **compiler backend**, and round_8's
> ISA live-diff **named the mechanism** (the two-branch OR is now resolved): a **c4-backend SGPR-pressure
> regalloc perturbation** of a shared-consumer **off-form `flat_load`/`global_store` i64 base** —
> **spill-independent** (it faults at `spill=0`, so it is a live-range/allocation perturbation, **not** a
> spill-reload bug). It is not a stray emitted atom with a bad constant; it is the *base register* of a
> real shared-consumer access going wild **only under c4's higher SGPR pressure**.
>
> **On the r3/r4 "register-corruption" revival — read this so you don't mis-file it as a regression:** r3/r4's
> *measurements* were void (dead instrument — untraced body + non-`const_expr` guards; see
> `crash_bisection.md`), but the regalloc/scheduling *hypothesis space* was therefore **never soundly
> tested**. Rounds 7–8 tested it soundly (instrument fixed, source axes exhausted) and **confirmed it** —
> so the revival is legitimate, and the naming instruments (coredump **PC**, per-emitted-atom `0xBEEF`
> canary, and **ISA-diff c4 vs cut3 shared-consumer live-ranges**) have already done their job: the
> mechanism above is named, not hypothesized.
>
> **[SUPERSEDED by the round_3 flip at the top of this section — the fix below is FALSIFIED on-hardware.]**
> The round_8 verdict read the mechanism as: *"the live axis is the FIX: lower c4's SGPR pressure so
> regalloc stops perturbing that shared-consumer i64 base — decouple the g2-collapse substrate's
> live-ranges."* round_1 executed exactly that decouple and it **did not clear the fault**, and round_3's
> live rocgdb read then showed **why**: the wild quantity is a *loaded routing index* (`v6`), not a base,
> so no amount of base/pressure decoupling can fix it. Keep this paragraph only as the record of the
> falsified backend hypothesis; the live lever is the **routing/offset-index producer** (top of section).

The last wave (12 rounds) authored the whole body and got it on-card, but the credit-bearing fused
terminal never ran clean: the launch-collapse faulted at cut4 (SIGABRT / illegal access). The fault was
bracketed to **one instruction** and root-caused as a **latent use-of-uninitialized / miscomputed
base-or-predicate ADDRESS in the arrival-ticket spin-wait loops** in `mega_moe_stage1.py`. It is
**deterministically exposed by the register-allocation shift** that adding *any* `ready1` publish forces
— a benign constant `0xBEEF` store (no atomic, no computed pointer) faults identically, with identical
instruction/mem-op mix, identical LDS, zero spill. Falsified on-card: the publish formulation,
OOB/buffer-sizing, dispatch-table delivery, and fence/atomic pairing — **none** is the cause. It is NOT
a counter target-unreachable; it is a **regalloc-exposed substrate address defect.**

> **FALSIFIED on-card, 2026-09-05 (round_2, this operator) — read before you touch the gate.**
> The wave's D0 engineer applied the gate base/predicate SGPR-pin below **correctly** (direct 64-bit
> `readfirstlane`, matching stage2:126) AND base-pinned every source-visible g2 substrate base
> (`addr_ready1`, `addr_g2_comb_inp/trb/stids/sweights`, +cut5 inputs) in both substrate copies. cut3
> stayed clean+correct at **both** bs=128 and bs=8192 (relL2 0.0656/0.0592, path=MEGA==8), but
> **cut4→cut8 faulted identically** — same wild-global SIGABRT, `0x7e7e…` poison address, first forward,
> all 8 ranks. **Conclusion: base-pinning the source-visible bases is necessary hygiene but is NOT the
> cut4 fix** — consistent with the `0xBEEF`-constant-store faulting identically, the corrupted register
> is one the publish's regalloc shift perturbs that is **not** a source-visible base. The cut4 fault is
> **before combine** (`path_mega=0`), so it is in the **GEMM2-role substrate**, and the poison address is
> a **wild table index**, not the flat-system gate (which base-pinning already covered and did not fix).
> So do the base-pin (it is correct and cheap), but the **primary** cut4 fix is the resource-extent clamp
> in "fix 2b" below, and the converging diagnostic is the ISA diff. Do not re-spend a round re-deriving
> base-pinning — that experiment is done.

Where to look, concretely, in the arithmetic above:

- **LDS-aliased ticket scratch.** The baseline broadcasts the ticket through the LDS pool itself:
  `ticket_scratch = recast_iter(Int64, a_buf.ptr)` where `a_buf = lds.pool` — the **same** pool the
  GEMM staging view aliases (`c_tile = recast_iter(Float32, lds.pool.ptr)`). When a `ready1` publish
  shifts allocation, the ticket-scratch base and live GEMM data share storage under a *different*
  layout. Harden this: give the ticket broadcast its own small dedicated LDS slot (not the pool it
  shares with staging), or barrier-fence the broadcast so the recast base is provably stable across the
  new allocation.
- **fix 2a — the gate base/predicate SGPR-pin (hygiene: necessary, but round_2 proved NOT sufficient
  for cut4; do it anyway, then go to fix 2b).** Read the exact baseline shape first (in-tree, reachable —
  `mega_moe_stage1.py`, line numbers are the frozen baseline's):

  ```
   94:  grid_epoch_slot = GRID_MULT_VALUES.index(grid_mult)   # Python int → COMPILE-TIME const
  199:  a_epoch_gate    = _disp_ptr(DispatchSlot.EPOCH_GATE)  # 64-bit ptr, a buffer_load → VGPR
  219:  generation      = ticket64 // launch_grid_x           # runtime → VGPR
  221:  gate_addr       = a_epoch_gate + grid_epoch_slot*4    # base = VGPR + constexpr
  222:  gate_epoch      = generation + 1                      # predicate = VGPR
  227:  if compact_owner:  ... ~50 lines, heavy reg pressure ...
  272:      store_i32_system(gate_addr, 0, gate_epoch)        # OWNER publishes here
  277:  else: int32_wait_until_equals(gate_addr, gate_epoch)  # NON-OWNER spins here
  ```

  First correct a red herring: **`grid_epoch_slot` is NOT a runtime register** — it is
  `GRID_MULT_VALUES.index(grid_mult)`, a Python int, so `*4` / `*8` are constexpr offsets and cannot be
  left stale. The only runtime inputs to the gate are **`a_epoch_gate`** (the 64-bit base, a VGPR out of
  the dispatch-table `buffer_load`) and **`generation`** (a VGPR out of the ticket64 division). Those two
  are computed **once in the common path** (221–222) but consumed on two branches of wildly unequal
  length — the ~50-line owner path (parity dance + peer loop + work_head init + multiple fences) versus
  the 3-line non-owner spin. Note the baseline **already** `readfirstlane`s `next_parity`/`launch_epoch`
  (239–240) but does **not** do it for `gate_addr`/`gate_epoch`: those stay in VGPRs across the whole
  divergent region, so they are exactly what the `ready1`-publish regalloc shift perturbs — the owner
  stores to one address, the waiter spins on another (peer illegal access), or the predicates differ
  (hang).

  **The fix, drop-in, at line 222 in the common path — before the `if compact_owner` split** (so both
  branches provably consume the *same* wave-uniform scalars, mirroring what 239–240 already do for
  parity/epoch):

  ```python
  # r12 fix 2: pin the gate base AND predicate to SGPR in the COMMON path, unconditionally,
  # so neither the owner-store (272) nor the non-owner-wait (277) recomputes them from a VGPR
  # the ready1 publish's regalloc shift can leave stale.
  #
  # a_epoch_gate is an fx.Int64 dispatch pointer (_disp_ptr -> _buffer_load(..., fx.Int64), a VGPR).
  # There IS a direct 64-bit readfirstlane: the combine substrate already pins its own 64-bit peer
  # pointer with exactly this call (mega_moe_stage2.py:126,
  #   peer_base = rocdl.readfirstlane(T.i64, peer_base.ir_value())).
  # fx.rocdl IS flydsl.expr.rocdl (stage2 imports `rocdl` from flydsl.expr; both files do
  # `import flydsl.expr as fx`), so the i64 form is valid in stage1's own fx.rocdl namespace:
  a_epoch_gate = fx.rocdl.readfirstlane(T.i64, a_epoch_gate.ir_value())   # base now SGPR (uniform i64)
  gate_addr    = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)             # SGPR base + constexpr offset
  # gate_epoch is the Int32 predicate; pin it with the SAME i32 readfirstlane the owner branch already
  # uses on next_parity/launch_epoch (mega_moe_stage1.py:239-240):
  gate_epoch   = fx.Int32(generation + fx.Int64(1))
  gate_epoch   = fx.Int32(fx.rocdl.readfirstlane(T.i32, gate_epoch))     # predicate now SGPR
  ```

  Every line above mirrors a proven same-operator idiom at its exact bit width: the 64-bit base pin is
  the combine substrate's own `rocdl.readfirstlane(T.i64, …ir_value())` (stage2:126), and the 32-bit
  predicate pin is stage1's own `fx.rocdl.readfirstlane(T.i32, <fx.Int32>)` (239–240). Do **not** hand-roll
  a 32-bit split with bitwise `&`/`>>`/`<<`/`|` on `fx.Int64` — those ops are demonstrated only on
  `fx.Int32` in this tree and are unnecessary once you use the direct i64 readfirstlane. Pinning both
  makes the whole `store_i32_system`/`int32_wait_until_equals` address+predicate a scalar the allocator
  cannot reintroduce as a publish-perturbed VGPR. Do the identical i64 pin on the
  `a_entry_count + grid_epoch_slot*8` atomic base at 214 if the ISA diff implicates it (same class, lower
  risk — it is inside the `tid==0` guard and used once). This is authored from the reachable baseline
  arithmetic above **plus** the r12 regalloc diagnosis — **not** transcribed from any out-of-tree
  reference. It is correct hygiene — the ISA diff must show the gate base/predicate now in SGPRs — but
  round_2 proved it does **not** by itself clear cut4; go to fix 2b.

- **fix 2b — resource-extent clamp the GEMM2-role tables (hygiene; FALSIFIED round_3 as the cut4 fix —
  keep it, then go to the register-pressure fix below).**

  > **FALSIFIED on-card, 2026-09-05 (round_3, this operator) — read before you re-spend a round on 2b.**
  > The wave's engineer applied 2b **correctly** — bound the g2 peer/trb/stids/sweights tables to their
  > byte extents (`num_records_bytes=`, stage1 ~646–667) **and** kept 2a's source-visible base-pins **and**
  > added H-b (an extra `readfirstlane` pin of `addr_ready1` at stage1:574 before the drain publish).
  > **cut4 still faulted identically.** The converging finding: the corrupted access is the **raw
  > `addr_ready1` spin-wait at stage1 ~686** (`int32_wait_until_greater_than(addr_ready1 + _st*4, …)`),
  > which goes through **`mori_shmem` on a raw i64 address — NOT a `buffer_resource`** — so an extent clamp
  > can never reach it. This is now falsified on **three source-level axes** (grid-gate hardening,
  > source-visible base-pinning, g2 buffer-resource extent-clamp) **and** the H-b pin. **Conclusion: the
  > cut4 fault is a register-allocation problem, not a source-level address problem.** No further form of
  > *source-level* pinning or clamping will fix it — go to the register-pressure fix (fix 2c) below.

  The cut4 fault is a **wild table index** (poison `0x7e7e…`/`0xBEEF` canary), before combine, in the
  GEMM2-role substrate — not the flat-system gate. The g2 tables are read through AMD **buffer resources**
  (`buffer_ops.create_buffer_resource_from_addr(addr)`), and several are created **unbounded** (e.g.
  `gemm2.py:354  bq_rsrc = create_buffer_resource_from_addr(arg_bq)` — no `num_records_bytes`). An
  unbounded resource **faults** on a wild index; a resource **bound to its real byte extent** makes the
  hardware **clamp the OOB access to 0** instead of faulting. Because cut4/cut5 **discard** the GEMM2
  output, a clamped-to-0 read cannot corrupt their correctness — so this is a real shot at cut4/cut5
  clean *even if the index is genuinely wild under the shifted allocation*. Bind each g2 table to its
  extent, exactly as the combine substrate already binds its own peer resource:

  ```python
  # PRECEDENT (reachable, in-tree): stage2 already bounds its 64-bit peer resource with a compile-time
  # byte-extent kwarg — mega_moe_stage2.py:127
  #   rsrc_dst = buffer_ops.create_buffer_resource_from_addr(peer_base, num_records_bytes=comb_inp_nbytes)
  # where comb_inp_nbytes is computed at mega_moe_v2.py:325 as a compile-time int and plumbed in.
  #
  # DO THE SAME for the g2 tables. Compute each table's byte extent in mega_moe_v2.py from its own torch
  # tensor (numel * element_size) and pass it through compile/run_mega_moe_stage1 as a compile-time kwarg;
  # fall back to None (unbounded) only if the extent is 0/unknown:
  #   tile_row_base : op.tile_row_base.numel() * 4     (int32; op.tile_row_base = zeros(metadata_blocks,int32))
  #   srcmap_em     : op.srcmap_em.numel()  * op.srcmap_em.element_size()
  #   wts_em        : op.wts_em.numel()     * op.wts_em.element_size()
  # then at each g2-table create_buffer_resource_from_addr site (e.g. gemm2.py:354):
  g2_trb_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_g2_trb, num_records_bytes=trb_nbytes)
  g2_sm_rsrc  = buffer_ops.create_buffer_resource_from_addr(addr_g2_srcmap, num_records_bytes=srcmap_nbytes)
  g2_wts_rsrc = buffer_ops.create_buffer_resource_from_addr(addr_g2_wts, num_records_bytes=wts_nbytes)
  ```

  This is the same `num_records_bytes=` kwarg the tree already passes at stage2:127 and gemm2:140/329 —
  no new API, and the extents are the tables' own torch shapes (reachable), **not** transcribed from any
  out-of-tree reference. Guard the whole thing behind the fusion `const_expr` so the scattered/default
  path stays byte-op-identical. Verify: the `0xBEEF` benign-store control now runs **clean at cut4/cut5**
  (their g2 output is discarded, so the clamp is invisible to correctness), and cut6+ tell you whether a
  *genuine* wild index remains (there the g2 output is live, so a clamp would move relL2 — if it does, the
  index itself is wrong upstream, not just unbounded). If cut4 still faults after both 2a and 2b, the
  corrupted access is not table-resource-bounded: fall to the register-pressure reduction + ISA diff below.

- **fix 2c — shorten the `addr_ready1` live range + cut concurrent SGPR pins (the register-pressure fix;
  the remaining lever after 2a/2b/H-b are falsified).** round_3 localized the fault to the raw
  `addr_ready1` spin-wait at **stage1 ~686**. Read the substrate as authored and the mechanism is a
  **too-long live range under too many concurrent pins**, not a bad source address:

  ```
  574:  addr_ready1 = readfirstlane(i64, addr_ready1)     # H-b pins the base to an SGPR here...
  575–613: while consumer_active:  ... GEMM1 drain, atomic_add publish at 610 ...   # publish forces the shift
  630–642: g2 substrate setup: addr_g2_comb_inp/trb/stids/sweights each readfirstlane-pinned (4 SGPRs)
  671–675: cut5+: addr_g2_aq/ascale/bq/bscale/eids each readfirstlane-pinned (5 more SGPRs)
  686:  int32_wait_until_greater_than(addr_ready1 + _st*4, N_TILES-1)   # ...but the base is consumed HERE
  ```

  The pin at 574 must stay live across the **entire** GEMM1 drain (the `ready1` publish's regalloc shift)
  **and** 9 more concurrent `readfirstlane` pins (639–642, 671–675) before its consumer at 686. That is a
  textbook eviction: the allocator spills/reassigns the 574-pinned SGPR under the publish shift, so 686
  reads a clobbered base — exactly the `0xBEEF`-benign-store-faults-identically signature. The fix is to
  **reduce the pressure on that base's live range**, two concrete moves (do both; they compose):

  1. **Re-pin `addr_ready1` immediately before the wait, not once at 574.** Move (or duplicate) the
     `addr_ready1 = readfirstlane(T.i64, addr_ready1.ir_value())` to just inside `_g2_wait_ready`, right
     before the `int32_wait_until_greater_than` at ~686, so the pinned base has a **short** live range that
     does not straddle the publish shift or the g2-setup pins. A base materialized one instruction before
     its use cannot be the one the drain-loop allocation evicted.
  2. **Cut the count of simultaneously-live `readfirstlane` pins in the g2 setup.** The 9 g2 base pins
     (639–642, 671–675) and the ready1 base compete for the same SGPR file across the wait. Pin each g2
     base **at its own point of use** (inside the loop/closure that consumes it) instead of all up front,
     or drop the pins whose resource is already extent-bound (the buffer-resource create at 646/659–666
     re-materializes its base anyway). Fewer overlapping uniform-i64 live ranges = the allocator keeps the
     ready1 base resident through the publish.

  **The ISA diff is the arbiter, not this hypothesis.** diff the emitted ISA of the faulting cut
  (`isa_pub0`) against the clean canary (`isa_canary`) to see *which* SGPR the publish perturbs; the
  **leading candidate is the 574→686 `addr_ready1` base** for the reason above, but confirm it in the ISA
  before committing — if the diff implicates a different base, apply moves 1–2 to *that* base's live range
  instead. A rung that goes clean after move 1 alone confirms the live-range theory; check the ISA shows
  the ready1 wait base now materialized adjacent to its use with no spill across the publish.

Diagnostic that converges (from the r12 report's own next-step): **diff the emitted ISA** of the
faulting cut (`isa_pub0`) against the clean canary (`isa_canary`) round-over-round to see which
base/predicate register the publish perturbs; and **reduce substrate SGPR pressure** (fix 2c above) so
the `ready1` publish does not perturb the arrival-ticket allocation at all. Only once cut4 runs clean
does CUT5 (GEMM2 compute) → CUT6+ (4→3→2 launch collapse) become reachable, and only then is the paired
`8192_uniform` win read. Do **not** re-author the GEMM2 body, move the intra-rank edge off agent scope,
or re-measure the closed partial to chase this — it is a substrate address-init defect, fix it there.

## ⚠️ END OPTIMIZATION-PHASE FAILURE LORE — the faithful-floor recipe resumes here

> Back to material BOTH phases use: the measured M2.5 floor numbers, the executable verify gate, the
> three-way comparison, knobs, and do-no-harm notes. `repro_engineer`: this is the part after the
> Construction skeleton that you DO follow (how the floor is measured and verified).

**Measured effect of steps 1–5**, isolated (same tree, only `AITER_MEGAMOE_FUSE_ALL` varying, quant a
separate launch on both arms, A,B,A,B ×3 pairs per guard, rank-max `mega_e2e`, path marker verified on
all 24 runs):

| guard | scattered (med) | megakernel (med) | gain (med) | per-pair range |
|---|---|---|---|---|
| 512 uniform | 0.7264 ms | 0.7156 ms | **+1.49%** | +1.24 .. +1.79% |
| 512 skew | 0.7858 ms | 0.7662 ms | **+1.54%** | **−0.76** .. +3.70% |
| 8192 uniform | 4.6754 ms | 4.4699 ms | **+4.71%** | +3.55 .. +4.93% |
| 8192 skew | 5.5267 ms | 5.3775 ms | **+2.09%** | +1.48 .. +3.13% |

Large-uniform (`+4.71%`, tightest spread) is the guard to use as a positive control; 512-skew is
**not** — one of its three pairs came back negative.

## Executable verification

The procedure above is the apply-template; **`verify.sh` (next to the persistent_fusion card) is the
executable gate** that decides whether a candidate reproduces the skill. It takes no reference source —
it drives the one-flag A/B on the candidate tree only, and encodes five acceptance checks as pass/fail:

1. **Two-launch shape.** Assert the terminal form is exactly *quant, then one megakernel* — grep the
   captured graph for two ops per rank, fail on a fused-quant regression.
2. **Path marker.** `AITER_MEGAMOE_FUSE_ALL=1` must print the once-per-process fusion marker; a run
   without the marker is **void, not zero** (`grep -c 'path=MEGA'`, expected == world_size).
3. **relL2 parity.** Fused vs scattered output, `relL2 ≤ 0.10` at `tokens=8192`; parity is `required`.
4. **1000-replay stress.** ≥1000 CUDA-graph replays with a wall-clock timeout, comparing the *last*
   iteration to the first — catches an arrival counter that desyncs on generation 2.
5. **Paired performance.** A,B,A,B ×3 interleaved pairs per guard, rank-max `mega_e2e`, on the four
   guards; report medians and per-pair ranges. `uniform` guards gate; `skew` is reported, not gated.

Run it as `bash verify.sh --tree <candidate_aiter> --world-size 8 [--incumbent <m2.5_aiter>]`. It
writes a JSON verdict (`selected`, `decision`, `vs_incumbent_pct`) and exits non-zero if any *gating*
check fails. See `megamoe_ep_persistent_fusion.validation.yaml` for the shapes/routes the on-box
validator feeds it. `verify.sh` is a pure A/B on the candidate tree; the mega-mode leak-sweep lift does
not change what it checks (it never read a reference in the first place).

## Three-way comparison (Baseline / M2.5 incumbent / new candidate)

`verify.sh --three-way` measures all three arms under one identical route/iteration command and one
denominator (the **frozen public-AITER baseline**):

| arm | what it is | role |
|---|---|---|
| **Baseline** | four serialized launches/rank (quant → Stage1 → Stage2 → Combine), no fusion | denominator; every % below is vs this |
| **M2.5 (incumbent)** | this skill's persistent megakernel | **known-good floor (保底)**, not the ceiling |
| **New candidate** | whatever the current mega run produced | must clear Baseline; **is allowed to beat M2.5** |

M2.5 is the incumbent to **match-or-beat**, not a target to converge to. Pass `--incumbent <m2.5_aiter>`
and the floor is **enforced**: the candidate's fused path is paired against the incumbent's on the
decisive `8192_uniform` guard and the selection is **`max(candidate, M2.5)`**:

- candidate beats M2.5 **past the noise floor** (`NOISE_PCT`, default 1.45%) → `decision: supersede`,
  `selected: candidate` — ship it.
- within noise → `decision: tie_keep_incumbent` — reproduced, deployment keeps the known-good incumbent.
- candidate slower → `decision: regress_keep_incumbent` — the floor holds; M2.5 ships.

The base gate (beat Baseline + parity + marker + 1000-replay) decides whether the candidate is *valid at
all*; the floor rule decides *what ships*. In mega mode the goal is `decision: supersede` on
`8192_uniform` — beating the +4.71% floor.

## Knobs & pitfalls

- `payload_chunk_rows` — `256` at the 512-token bucket, `384` at 8192. Measured, not reasoned; re-sweep
  per bucket rather than extrapolating.
- **Write-through P2P** on the payload store is a separate accepted win; keep it when fusing.
- **Every wait needs a paired acquire.** `mori_shmem.*_wait_until_*` performs a *relaxed system load
  that does not invalidate L2*. A wait without `fence_system_acquire()` (or `fence_agent_acquire()` for
  the intra-rank/gate edge) before consuming the guarded data reads stale bytes and is correct only on
  the first iteration.
- **Grid-wide barriers require full co-residency** (`_check_block_num_resident`, cap = #CU). A
  participant index outside the resident window deadlocks at one problem size and runs fine at another,
  so correctness must be run at both the smallest and largest shape the harness offers.
- **Single-shot correctness cannot see the parity/counter class of bug.** Add a ≥1000-iteration repeat
  stress with a wall-clock timeout comparing the *last* iteration to the first.
- **Rank-max, never rank-mean.** A collective is gated by its slowest rank.

## Do-no-harm notes

- **Do not fold quantization into the megakernel.** Tried and retired: an ingress quant role reaching
  exactly one launch per rank measured **`+0.15%` to `+1.2%` slower** across the guards. Quant belongs
  in the upstream producer's epilogue (a `--prequant` path already exists). The terminal form here is
  **two launches: quant, then one megakernel.**
- **Do not try to reach 2 WG/CU on Stage1 by shrinking LDS.** The `cs_size*4 = 131072 B` f32 CShuffle
  slab dominates `lds_pool_bytes`, pinned at the **256-VGPR** ceiling. Static ISA screening closed this.
  Stage1 *source-level scheduling* is a different, still-open direction (an early measurement showed
  −0.77% geomean, 4/4 favourable pairs at 8192) — do not close that one on this evidence.
- **Keep the all-rank barrier behind a fallback flag.** A liveness failure in the field needs an
  immediate escape hatch, and the gate needs the A/B.
- The gains here are **single-digit percent**, and the measured per-case noise floor on this harness
  reaches **1.45%** at 1 rep. Any claim in that range measured with fewer than 5 interleaved pairs and
  no null arm is unreadable, not a result.
- Denominator discipline: measure against the **frozen public-AITER baseline** under an identical
  route/iteration command. Do not quote a speedup against MORI or Taco.

## Provenance & reporting

`provenance.source: mega_mode_reproduction`. This card is injected **only under `MODE=mega`** — the
megafusion workflow (author-mode + optimize-mode combined, `capability_eval=false`,
`strict_autonomy=false`). In mega mode:

- **Faithful reproduction of the M2.5 topology is the goal.** There is no derived-not-copied /
  criterion-5 / artifact-distinctness requirement in mega mode. Report the resulting arm plainly as the
  mega-mode fused candidate, measured against the frozen baseline and against the M2.5 floor. Do not
  spend effort making the artifact "distinct" — that ledger belongs to the separate strict/blind task
  (`use_expert_skills=OFF`), which is untouched by mega mode and does not read this card.
- **The concrete arithmetic above was authored from reachable knowledge**, not copied from a reference
  source: the in-tree baseline `mega_moe_stage1.py` (compact-dispatch handshake) + the r12 root-cause
  diagnosis. That is why it can be concrete here where the persistent_fusion card had to stay lossy.
- **The one boundary that is NOT lifted:** never read anything under `/root/geak_reference/`. The
  hand-authored M2.5 lives there; it is off-limits regardless of mode. Everything needed is in-tree +
  this card.

## Sources

- Isolated fusion A/B: 3 interleaved pairs per guard, rank-max `mega_e2e`, path marker verified on all
  24 runs. The medians and per-pair ranges are the table above — that table **is** the log, digested.
- Step-2 bottleneck evidence — kernel DAG showing zero overlap, instrumented combine peer-wait, matched
  no-payload control, ATT waitcnt/barrier shares, LDS residency. Every number this card relies on is
  quoted inline above.

  **When your own profile disagrees with a number here, the size of the disagreement decides what to
  do.** Within ~2x: trust your profile, say so, move on. Beyond ~5x: you and this card are not measuring
  the same thing — enumerate the plausible readings of the experiment, run each, report which reading
  reproduces which number. A closed axis backed by an unadjudicated 5x disagreement is a defect.

  (History: Wave 14 closed "hiding the Stage2 P2P payload cost" — the mechanism behind the +4.71% win —
  because its no-payload control moved `stage2_combine` by <=1.2% while this card reports −38.2% (~30x).
  Deleting the P2P store is not the same experiment as deleting the store **and** the arrival wait that
  follows it; only the second moves the number by a third. Nobody ran both.)

- Reusable in-kernel machinery (in-tree, reachable — the concrete arithmetic above is authored from
  these, not from any out-of-tree reference): `mega_moe_stage1.py:185-282` (compact-dispatch
  owner/producer arrival-ticket handshake, epoch/parity flip, gate spin-wait — the exact source of the
  address arithmetic and of the r12 defect), `dispatch.py:576-597` + `:204-219` (tile-ready counters),
  `communication_ops_utils.py:49-154` (system release/acquire).
- Prior art for the "unfused megakernel" control arm: Event Tensor, MLSys 2026, arXiv 2604.13327v2
  §4.5. Its distributed dynamic-scheduler arm measured **0.82–0.89×** (slower) — treat dynamic
  cross-rank scheduling as unproven here.
