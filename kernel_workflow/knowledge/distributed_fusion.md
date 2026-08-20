# Distributed Fusion — Collapsing a Multi-Launch Operator into One Persistent Kernel

The playbook for the `distributed` specialty: an operator that today issues **several serialized
kernel launches per rank** and coordinates with peer ranks, and whose target form is **one persistent
kernel per rank** with computation and communication genuinely overlapped (the "DeepGEMM-style"
shape). `geomean_levers.md` Lever 1 tells you to collapse dispatch count on a single-GPU op; this
file is what changes when the stages also talk to *other ranks*, because then the thing you are
removing is not launch latency — it is a **wait**.

Every number below is measured on a live 8×MI355X (gfx950/CDNA4, 256 CU, 8 XCDs, wave64) EP8 MoE
workload. Treat them as calibration for the *shape* of each effect, not as portable constants.

## When this file applies

All three must hold, or use `geomean_levers.md` instead:
- ≥2 GPU kernels per call that are **strictly serialized** (kernel-level DAG shows zero observed
  overlap between them);
- at least one of them exchanges data with peer ranks;
- the profile shows **waiting**, not arithmetic, dominating — e.g. accounted-cycle share for
  `waitcnt`+`barrier` above ~35% inside the kernels, or a cross-rank barrier whose cost explodes
  under load skew.

Diagnostic that settles it in one measurement: run a **no-payload control** (same compute, peer
stores removed). If the stage collapses — measured `2.5205 → 1.5568 ms`, −38.2% on the skewed route
— the transfer/visibility cost is real and currently *exposed*, i.e. not hidden under compute. That
gap is your budget. If the control barely moves, there is nothing for fusion to hide; stop here.

## Lever 1 — Enumerate the MISSING readiness edges before writing any code

Serialized stages are usually not serialized because they must be. They are serialized because **no
producer publishes and no consumer waits**, so the only ordering is kernel termination + stream
order. You can enumerate every fusion opportunity statically, before touching the kernel:

- Read the producer's epilogue. If it ends at a plain `buffer_store` — no release fence, no atomic,
  a cache hint at most — it publishes **nothing**.
- Grep the consumer for `wait_until` / any acquire. If there is none, the edge does not exist.

Each such (producer, consumer) pair is one missing edge, and each is a candidate. In the reference
workload there were exactly two: `GEMM1→GEMM2` (intra-rank) and `Stage2/P2P→Combine` (cross-rank).
Cross-rank readiness came solely from an all-rank barrier.

**An edge that does not exist cannot be optimized — it must first be added.** Budget for that: the
publication itself costs something (Levers 5 and 6 are about making it cheap).

## Lever 2 — Replace whole-phase joins with per-item readiness (highest payoff)

An all-rank barrier, or any single counter every consumer gates on, forces **every** consumer to pay
`max` over **all** producers. It is where load skew gets exposed, and it is almost always the largest
single line item.

Measured, instrumented peer-wait (`s_memrealtime`), 8 ranks × 20 replays:

| route | rank-max p95 peer wait |
|---|---|
| uniform | 191 µs |
| all-remote | 193 µs |
| **rank-mixed skew** | **876.6 µs** (+685.5 vs uniform) |

Per-rank split on the skewed route: ranks 0–3 waited `124.7 µs` mean, ranks 4–7 waited `651.3 µs`.
Fast ranks were blocking on hot-expert ranks — a pure serialization cost, not bytes: logical remote
bytes differed `<0.1%` between routes and skew firmware XGMI traffic was actually *4.3% lower*.

The fix is granularity, not removal: publish **per destination item** (per token, per tile) and have
each consumer wait only on the items it is about to read, then start its reduction. Fast items
proceed while slow ones are still arriving.

**Prerequisite most people miss:** a single-buffered staging buffer is *why* the barrier was needed —
the barrier is what stops iteration N+1 from overwriting rows iteration N has not consumed. Removing
the barrier therefore **requires** either double-buffering with a parity index or per-slot completion
counters. Do that first, or the fusion is correct in a single shot and silently wrong in a loop.

Keep the barrier behind a compile-time fallback flag. It gives the A/B an honest control and a
liveness failure an immediate escape hatch.

## Lever 3 — Charge the fusion for the kernel boundary it removes

This is the lever that decides most *negative* results, so read it before proposing a fusion.

A kernel boundary is not free ordering — it is a **hardware-managed, load-balanced join** over many
small workgroups. Replacing it with an in-kernel counter substitutes a **software max-of-N** over the
resident blocks, and the resident blocks are big, few, and unevenly dispatched across dies.

Measured: a quantization pass costing `0.0009 ms` standalone was folded into the persistent kernel as
a static grid-strided partition. The pass itself was free. The join was not:

| route | separate launch | fused ingress | Δ |
|---|---|---|---|
| 512 uniform | 0.6796 | 0.6806 | +1.0 µs (+0.15%) |
| 512 skew | 0.7549 | 0.7657 | +10.8 µs (+1.4%) |
| 8192 uniform | 4.4877 | 4.5042 | +16.5 µs (+0.37%) |
| 8192 skew | 5.3472 | 5.4122 | +65.0 µs (+1.2%) |

The delta is worst on the skewed routes — exactly where the grid is least evenly occupied, i.e. where
max-of-N is worst.

**Rule: fusion pays when it removes a *wait*, not when it merely removes a *launch*.** Launch is
~5–15 µs; if that is all you are removing and the phase must then be joined in-kernel, expect to
lose.

**Corollary — launch count is not an objective, so do not adopt it as one.** The pass above was
pursued to reach "exactly one launch per rank", which sounded like the definition of full fusion. It
is not, for three independent reasons, and the project later withdrew the milestone outright rather
than repair it:

1. **The measurement said no from the start.** The pass was 0.64% of e2e; the join it created cost
   more than the launch it removed. Nothing about that ratio was going to improve.
2. **Check the operator boundary before fusing across it.** That pass was an activation *cast*, which
   in a real serving graph belongs in the epilogue of whatever produced the activation — producer
   quantizes what it just wrote. Pulling it into the consumer megakernel competes with the better
   placement and forces the consumer to ingest unquantized input, defeating any pre-quantized fast
   path the operator already offers. **Ask "who is the natural producer of this data?" before
   fusing a phase in; if the answer is a different operator, the phase is not yours to fuse.**
3. **The reference architecture does not do it either.** "Fuse it like <fast library X>" was read as
   "one launch for everything". Fast GEMM libraries of this class consume pre-quantized inputs with
   their scales and ship the cast as a separate helper; collective dispatch/combine likewise live
   outside. What they maximize is occupancy and pipelining *inside* the main body, not the number of
   launches in the surrounding graph. **If you are citing a reference design as justification, state
   the specific mechanism you are copying. "It's one kernel" is usually not what the reference did.**

If you build such a fusion anyway, keep it in-tree **default-off as a labelled control experiment**,
not as an unfinished feature — the negative result is the deliverable, and leaving it filed as
"almost done" causes the next person to spend effort finishing something that should not ship.

**The corollary runs in BOTH directions — a small inter-kernel gap is NOT grounds to reject a
fusion.** The rule above says launch count is not a reason to fuse. It does not say launch gap is a
reason not to. These are different claims and conflating them has already killed a correctly-scoped
fusion on the wrong evidence: a planning role measured the total inter-kernel gap at 12.4 µs/iter
(0.22% of e2e at the large size, 1.42% at the small one) and closed the direction as
"net-negative before it starts", citing this very file. That reasoning is invalid. The gap is what
fusion removes **incidentally**; the wait is what it removes **on purpose**. On the same operator the
gap was 0.22% while the exposed cross-rank wait was a rank-max p95 of 876.6 µs, and the fusion that
absorbed part of that wait measured **+4.71%** on the large-uniform guard — twenty times the gap it
also happened to delete.

So the decision test is fixed, and it is not a launch-gap measurement:

- **Run the no-payload control** (Priority item 1). Replace the transferred payload with nothing and
  keep everything else identical. The delta is the exposed communication cost — the ceiling on what
  overlap can buy. On this operator it was −38.2% of Stage2+combine under skew.
- **Then measure the exposed wait directly** — an instrumented peer-wait timer, or the barrier's
  own duration. If a consumer is blocked on a producer for a time that is large relative to the
  target, fusion has something to absorb, *regardless of how small the launch gap is*.
- Only if BOTH come back near zero is "no fusion win available" the right conclusion. **"The kernels
  launch back-to-back" is evidence about the host, not about whether the GPU is idle waiting.**
  Kernel boundaries can be 12 µs apart and still serialize a millisecond of dependent work, which is
  exactly the case a persistent kernel with per-item readiness exists to fix.

Write the decision down with the number that drove it. "Rejected: no-payload control showed 0.3% of
e2e" is a durable result. "Rejected: launch gap is 0.22%" is a category error and will be re-litigated.

## Lever 4 — Straggler-gating has a variance signature; use it to diagnose

Same code, 8192 uniform, 100 iters, 3 interleaved reps, rank-max e2e:

| arm | rep1 | rep2 | rep3 | spread |
|---|---|---|---|---|
| separate launch | 4.4208 | 4.4217 | 4.4200 | **1.7 µs** |
| fused ingress | 4.4566 | 4.4310 | 4.4476 | **25.6 µs** |

Slower **and 15× noisier** ⇒ a gate paying max-of-N, not a throughput deficit. Slower with an equally
tight spread ⇒ genuinely less throughput. These call for opposite fixes, so **always report per-rep
spread alongside the median** — a median-only table cannot distinguish them.

Corollary on the straggler reading: adding participants makes max-of-N *worse*, not better; removing
them eventually runs out of bandwidth. Neither direction is the fix — the fix is Lever 2 (make it not
a join at all) or Lever 3 (do not fuse it).

## Lever 5 — Publication scope must match the cache-coherence domain count

On a multi-die accelerator the LLC is **per die** (8 XCDs on MI355X). Two consequences that produce
hangs rather than wrong answers, and reproduce only at some block counts:

1. A relaxed/plain load sees **only its own die's LLC**. The `*_wait_until_*` shmem helpers are
   relaxed loads. An `atomic_add` at *agent* scope lands in the adder's L2, so a waiter on another
   die never observes it. **Publish readiness with system-scope atomics; clear with a system-scope
   store.**
2. Because the wait is a relaxed load, it does **not** invalidate L1. Every wait must be paired with
   an explicit **acquire fence** before the guarded data is consumed. No exceptions — this is the
   single most common source of "works at bs=128, garbage at bs=8192".

Record the coherence-domain count in the profile alongside the cache sizes; the sizes do not predict
either of these.

## Lever 6 — Write-through stores instead of per-block release fences

The obvious way to publish is `fence_system_release()` then the completion atomic. On this part that
fence lowers to a **full LLC writeback (`buffer_wbl2`) per block**. With ~200 participating blocks it
measured **+0.411 ms**.

Cheaper and equivalent for this pattern: mark the payload and scale stores **write-through**, then
`s_waitcnt vmcnt(0)` before the completion atomic. On gfx95x the buffer cache-policy bits are
`bit0=sc0`, `bit1=nt`, `bit4=sc1`; write-through is `sc0|sc1 = 1|16 = 17`. Applied to the ingress
this recovered most of an 8.4 µs residual and removed the fence entirely.

Same trade the GEMM epilogue already makes for its activations — check whether your codebase has an
`out_cache_modifier`-style hook before adding a fence.

## Lever 7 — Residency is a correctness invariant, not a tuning knob

In a persistent kernel, roles are assigned by **arrival ticket** (`ticket = atomic_add(counter)`;
ticket 0 = plan owner, `1..K` = producers, and so on). If any block gates on a participant index that
is **not resident**, the grid deadlocks: the unscheduled block cannot run until a resident block
retires, and none can, because they are all waiting on it.

Measured: the largest stage ran at **1 WG/CU**, so exactly tickets `0..num_cu-1` were resident.
Excluding the owner from the participant set without shrinking the participant count pushed the last
share onto ticket `num_cu` → hard hang, all 8 GPUs at 100%.

**Invariant: every participant index must land inside `[0, resident_blocks)`.** And note the trap:
the resident count depends on the occupancy-binding resource and on per-bucket grid sizing, so a
participant count that is legal at one problem size can hang at another. Derive it, never hardcode
it, and make any override env-gated for bisection only.

## Lever 8 — Acyclicity: the coordinator must not owe the workers anything

If the block holding a coordinating role (emitting a plan, seeding counters) is also assigned a share
of work that the workers gate on, and that share is scheduled *after* the coordination point in
program order, you have closed a cycle: coordinator → workers → coordinator. Hard hang, at every
participant count you try.

**Rule: any work the workers gate on must complete before the coordinator reaches its coordination
point.** Placement, not logic — moving the phase earlier in the same block fixed it.

Bisection tell: if forcing the participant target to a value the *workers themselves* satisfy always
passes while every real value hangs, it is this, not a scope bug.

## Lever 9 — Reset-free counters

Never clear an arrival counter between generations; the clear races the next generation's publishers.
Instead let the counter rise monotonically and raise the *target* by the per-generation increment.
Pad any slot outside the live range so the target is still reachable.

Where a clear is genuinely unavoidable, do it in the **ticket-0 block before it publishes the epoch
gate** — that orders the clear ahead of every other block without needing a grid barrier.

Where a generation key is needed, prefer an explicit monotone epoch with a parity-selected buffer
half (ABA-safe). **Trap:** do not key a generation on a counter that also counts launches which
publish nothing — e.g. a variant compiled without the publishing role will bump the epoch and desync
every waiter.

## Anti-patterns (measured, do not re-walk)

**A — A work-stealing claim queue for an in-kernel phase.** Measured **~89 ns of end-to-end cost per
claimed chunk** (+0.020 ms at 512 tokens, +0.30 ms at 8192, scaling exactly with chunk count), because
every block's *next* claim atomic sits on its own critical path. A static grid-strided split with no
claim atomic at all beat it outright. Work-stealing is for a queue whose items have wildly unequal
cost; a uniform sweep does not qualify.

**B — Grid-stride unrolling to restore outstanding-load count.** A phase folded into a kernel pinned
at 1 WG/CU runs far fewer waves than the standalone launcher did, so more independent groups per
iteration "should" help. Measured at 8192 tokens, 100 iters, 3 reps: `4.4535 ms` at U=4 vs
`4.4476 ms` at U=1 — no. **And that is information:** if unrolling does not help, the residual is not
latency-bound, so stop looking there and go read Lever 4's spread instead.

**C — Cutting LDS to raise occupancy without checking which resource actually binds.** The stage was
at 97.5% of LDS/CU, which reads as an obvious 1→2 WG/CU opportunity. It was not: at **256
VGPR/thread** a 512-thread block is pinned to 1 WG/CU *regardless of LDS*. The only configuration
that cleared the LDS budget cost **+2.33 ms** e2e. **Report which resource binds, not just the LDS
figure** — and when occupancy is VGPR-bound, an LDS direction is a dead end before it is dispatched.

**D — A per-rank load test to detect intra-rank skew.** Skew that matters is often an *intra*-rank
property (a few hot experts inside one rank). A detector comparing rows received against
`tokens × topk` never fires, because the rank-level totals are balanced. Use a statistic at the
granularity the imbalance lives at — e.g. `max_expert_tiles` versus `rows / tile_m / experts_per_rank`.

**E — Pursuing "one launch" as the definition of full fusion.** See Lever 3's corollary. The
milestone was built, was correct, reached exactly one launch, measured slower on all four guards, and
was withdrawn as a goal rather than repaired. Cost: the fusion work itself, plus a planned follow-up
to convert its ingress join into per-item readiness that was only ever needed to rescue the wrong
target. **Before adopting a structural target, ask what it is worth in the profile.** If the phase in
question is under ~1% of e2e, the target is bookkeeping, not performance.

## Measurement discipline (this domain breaks the default assumptions)

- **Interleave the arms in ONE script.** Batch-to-batch drift measured up to **4%** at small sizes —
  larger than most of the effects here. A 1% apparent regression from sequential runs became 3-of-4
  paired **wins** when interleaved. Any margin under ~2% measured sequentially is noise.
- **Report per-rep spread with every median** (Lever 4).
- **Rank-max, not rank-mean,** is the number that matters: the collective is gated by the slowest rank.
- **Guard all four corners.** Small/large × uniform/skew. A fusion that wins at large uniform can lose
  at small skew, and small-batch regimes have proportionally more fixed cost to amortize.
- **Liveness is a separate gate from correctness.** 1000 CUDA-Graph replays per route, plus an
  explicit stale-read check that the readiness counters are epoch/parity-correct across back-to-back
  iterations. A fusion can be numerically correct on a single shot and deadlock or read stale data at
  a different block count (Levers 7–9 are all failures of exactly this kind).
- **A latency win with no measured overlap change is suspicious, not accepted.** Re-collect the
  kernel-level DAG and the instrumented peer-wait and show the wait actually shrank. Otherwise you
  have measured drift.
- **Name the denominator.** Compare against the frozen upstream baseline under an identical
  route/iteration command, never against a different library.

- **Two denominators, two different questions — run both.** "How much faster than upstream" and "how
  much did the fusion buy" are not the same number and the gap can be 7×. Measured on one operator:

  | guard | vs frozen upstream | fusion alone (same tree, flag off/on) |
  |---|---|---|
  | large uniform | −4.92% | +4.71% |
  | large skew | −2.08% | +2.09% |
  | small skew | **−10.96%** | **+1.54%** |

  The small-skew column is the warning: 9 of those 11 points came from unrelated improvements already
  on the branch (write-through payload stores, a payload-chunk tuning), not from fusion. Attribute a
  fusion win by running the **same tree with only the fusion flag toggled**, A,B,A,B. Comparing the
  fused build against a frozen baseline credits fusion with every other change on the branch.

- **Fusing a phase deletes the instrument that measured it.** A peer-wait timer living in the
  standalone kernel reads a flat `0.0` once that kernel is no longer called — which looks exactly
  like "the wait went to zero" and is really "nothing was measured". Before quoting an improvement in
  a wait/occupancy/bandwidth counter after a fusion, **prove the instrumented code path still
  executes.** Budget an in-kernel replacement instrument as part of the fusion, not as follow-up.

- **State the aggregation unit of an in-kernel timer before reading its output.** Two bugs, both
  producing confident wrong numbers, neither a timing bug:
  - *Mean vs total.* A per-item readiness scheme waits many short times where a barrier waited once
    for long. Reporting the per-wait **mean** made the fusion look worse for free. The comparable
    statistic to a barrier timer is the **per-launch total**. Publish both.
  - *Per-block vs per-wave slots.* One accumulator per block sums waves that wait **concurrently**,
    turning parallel time into serial time. It reported 8470 µs of wait inside a 5.4 ms kernel.
    **Any in-kernel wait exceeding the kernel's own wall time is an aggregation error** — that
    impossibility is the cheapest available self-check, so compute the ratio every time.

- **An opt-in fast path that falls back silently will be measured in its slow form.** A megakernel
  gated on `ENV == "1"` **and** a config predicate silently took the scattered path when either
  failed, and a full closeout was filed against the wrong path — it does not look like a bug, it
  looks like "the optimization didn't help". Make the kernel **announce which path it took**, once
  per process, and make the harness **refuse to report a number without that marker from the same
  run**. Do not use an unrelated field (a variant name, a config string) as a proxy for "fusion is
  on"; it will read plausible on both paths.

- **The static-ISA screen is a screen for RESOURCE hypotheses only. It cannot refute a latency-hiding
  one.** When the GPU group is a serialized lease, dumping the final ISA and diffing
  `amdhsa_private_segment_fixed_size` / `amdhsa_next_free_vgpr` / `amdhsa_group_segment_fixed_size`
  plus instruction-class counts is an excellent way to kill directions in minutes without queueing.
  It is sound for claims of the form *"this will free registers / shrink LDS / raise occupancy"* —
  those claims are **defined** by those numbers, so unchanged numbers do disprove them.
  It is **not** sound for claims of the form *"this will shorten the dependency chain / overlap a
  load with an MFMA / reduce time spent stalled"*. Those are properties of the dynamic schedule.
  Measured counterexample on this operator: an index-scalarization + prefetch-lead change left
  scratch, VGPR, LDS, `v_mfma` and `buffer_load` counts **all bit-identical**, was refuted on that
  basis, and then — when it was accidentally measured anyway — came back favourable on **4 of 4
  paired reps at the large size (−0.39% to −1.61%)**, three to four times that guard's calibrated
  noise floor. The static screen was not wrong about the ISA; it was answering a different question
  than the one asked.
  So: use the screen to **order** the queue and to catch compile-illegality for free, never as the
  sole grounds to close a scheduling-class direction. Recording "ISA unchanged" is useful; recording
  "ISA unchanged, therefore no effect" is a claim the instrument cannot support. If a scheduling
  hypothesis survives compile, it needs wall clock.

## Priority

1. **Run the no-payload control** and the DAG. No exposed wait ⇒ no fusion win available; stop.
2. **Enumerate the missing readiness edges** (Lever 1). This is static and cheap.
3. **Attack the largest exposed wait first** with per-item readiness (Lever 2) — usually the cross-rank
   barrier, and usually most of the available gain.
4. **Then the intra-rank producer→consumer edge**, agent scope, which is cheaper to publish.
5. **Do not fuse launch-only phases at all** (Lever 3) unless something other than launch count pays
   for it. If you build one to settle the question, land it default-off and labelled as a control.
6. Levers 5–9 are **prerequisites, not options** — get scope, fence cost, residency, acyclicity and
   counter-reset right, or the above will hang rather than run slowly.

Honest ceiling: perfectly hiding the *entire* worst-case combine wait was 13.46% of e2e, and the
no-payload delta is not removable (a correct implementation still sends the data). Real gains are a
fraction of these. A cumulative **2–5%** rank-max improvement from a full fusion of a
well-tuned distributed operator is a good outcome — size the effort accordingly.

**Outturn, for calibration.** Fusing dispatch + GEMM1 + the GEMM1→GEMM2 edge + GEMM2 + P2P publish +
combine into one persistent kernel (three launches → one, with the ingress cast deliberately left
outside) delivered, fusion-alone and paired:

| guard | fusion gain | per-pair range |
|---|---|---|
| small uniform | +1.49% | +1.24 .. +1.79% |
| small skew | +1.54% | **−0.76** .. +3.70% |
| large uniform | **+4.71%** | +3.55 .. +4.93% |
| large skew | +2.09% | +1.48 .. +3.13% |

Inside the predicted 2–5% band, and the *shape* is the useful part: the gain tracks how much
cross-rank wait there is to absorb. Large-uniform wins most; small batches sit near the noise floor
(one of three small-skew reps went negative — at that size the fixed costs fusion removes are the
same order as batch drift). Skew wins less than uniform **not because the fusion is untuned but
because its bottleneck is elsewhere** — the hot-expert tail, where in-kernel wait ran 27× higher on
the slowest rank than the fastest. Fusion cannot fix a producer-side imbalance; that needs the
producer's tile→block mapping changed so a token's contributions arrive progressively. Diagnose which
regime you are in (Lever 4's spread, plus the per-rank wait distribution) before spending more effort
on the fusion itself.

**Overlap claims have a proof ceiling you should know before promising one.** Once the operator is a
single kernel there are no boundaries left for a kernel-granularity trace to show overlap across, and
a wait timer shows that a wave *waited*, not that the CU executed something else meanwhile. Structural
evidence (kernel count in the trace, per-item readiness replacing a barrier) plus a bounded in-kernel
wait (measured: 8.6%–23% of kernel time) supports overlap but does not demonstrate it. A direct claim
needs a **residency-side** instrument — executing-wave sampling across the timed window, or an ATT
pass correlating wait regions against issued VALU. Plan for that instrument up front, or state the
overlap result as partial and say what is missing.
