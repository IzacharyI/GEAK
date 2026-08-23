# When NOT to Fuse — The Precondition Test for Fine-Grained Dependencies

A **method** card, and deliberately an adversarial one. Every other document in this tree tells you
how to fuse. This one exists so that "fuse it" has to survive a test before it becomes a plan.

The bias it corrects is structural, not personal: a workflow whose task statement says *"collapse N
launches into one"* will produce an analysis that concludes *"collapse N launches into one"*,
because that conclusion was in the prompt. An analysis that cannot reach the opposite conclusion on
evidence is not an analysis. **You are permitted — and expected — to return "the fine-grained form
does not pay here, and here is the measurement that says so."** That is a successful Analyze phase.

## The three conditions

Fusing to a persistent/megakernel form buys you exactly one thing the multi-launch form cannot have:
a consumer may start on *part* of a producer's output before the producer has finished. Everything
else fusion gives you (fewer launches, warm caches, no re-load of weights) is available more cheaply
by other means. So the fine-grained dependency pays **only if all three hold**:

1. **The producer's output can be handed over at a smaller granularity than "all of it."** There
   must exist a partial result that is independently consumable. If the consumer needs a reduction
   over the producer's entire output, there is no partial handover and no window.
2. **There is idle hardware for the consumer to run on while the producer is still running.** If the
   producer already saturates the machine, starting the consumer early does not make it finish
   earlier; it just interleaves. Check occupancy and CU utilization during the producer, not after.
3. **The producer does not complete in a single wave.** This is the condition that is skipped most
   often and kills the most proposals. If the producer's grid fits in one wave, its first output
   tile becomes ready at almost exactly the moment the whole producer finishes — the early-start
   window is ~0. There is nothing to overlap. Compute it: `ceil(blocks / (CUs × blocks_per_CU))`. If
   that is 1, stop; write it in the report and stop.

Condition 3 has a corollary worth stating separately: **a producer with many waves has a window
proportional to (waves − 1)/waves of its own duration.** That is your upper bound on what
fine-grained handover can recover from that edge, before any overhead. Compare it to the sync cost
per edge (`tile_task_graph.md`) *before* proposing the change. If the window is 3 µs and you need
200 edges each costing ≳1 µs of global sync, the arithmetic has already answered you.

Evaluate all three **per edge**, not for the operator as a whole. It is normal for one edge in a
chain to satisfy all three and the rest to fail condition 3.

## The cheaper-lever ladder

Fusion is the most expensive lever in the drawer — in engineering time, in verification risk, in
deadlock surface, in how hard the result is to change later. Work up the ladder and stop at the
first rung that closes the gap you measured:

1. **Remove the work.** Algorithmic or layout change, redundant compute, an unnecessary
   materialization, a precision choice. Consistently the largest wins, and consistently the last
   thing anyone tries.
2. **Fuse *adjacent* kernels** that share operands — trivially safe, no dependency machinery, gets
   the cache reuse without the scheduler.
3. **Run independent branches concurrently** (multiple streams/queues). This costs nothing in
   correctness and directly attacks the case where the DAG has parallelism the launch order is
   throwing away. It is the right answer whenever `tile_task_graph.md` found unconnected subgraphs.
4. **Cut the launch overhead itself** — graph capture / replay. Launch and teardown is on the order
   of ~2 µs uncaptured and ~1.3 µs captured; the residual bubble between adjacent nodes in a
   captured graph has been measured at **~300 ns**. Note what that means: *if your proposal's whole
   thesis is "we remove N launch boundaries", the prize is N × 300 ns.* For a handful of launches
   that is noise. Say so rather than shipping it.
5. **Overlap dependent consumers with their producers** without going persistent, where the platform
   offers it. On NVIDIA this is programmatic dependent launch (PDL), worth ~0.5–2 µs per dependency
   edge. **ROCm has no direct PDL equivalent** — so on AMD hardware this rung is thinner, and its
   substitutes (cooperative launch, in-kernel producer/consumer flags between co-resident
   workgroups, multi-queue) each carry more of fusion's complexity. That gap is a legitimate
   *argument for* fusion on this hardware, and it is one you have to make explicitly rather than
   assume.
6. **Persistent / megakernel form.** Only after 1–5, and only for the edges that passed the
   three-condition test.

## The calibration that should make you cautious

Published four-way ablation on a dependency chain, same work throughout (graph time / kernel time,
µs):

| form | time |
|---|---|
| linear chain, no dependent-launch | 65.5 / 61.6 |
| linear chain + dependent-launch | 59.4 / 54.2 |
| restructured branching DAG, no dependent-launch | 49.2 / 44.4 |
| restructured branching DAG + dependent-launch | **41.0 / 37.0** |

Read the second column against the third: **restructuring the DAG so independent work is expressed
as independent (rung 3) beat the fine-grained-dependency mechanism (rung 5) on the original
structure.** The mechanism and the structure compose, but the structure was worth more.

And a replication that should be quoted at anyone proposing a megakernel on launch-overhead grounds
alone: a small transformer decode step, hand-written as one megakernel, versus the same step split
into 81 ordinary kernels under graph capture — 983 µs fused, 1169 µs captured-with-dependencies,
**1010 µs captured + dependent launch**. The cheap mechanism recovered ~83% of what the kernel
boundaries cost; the persistent form led by **~2.7%**.

The honest summary of the public evidence: **fusion of adjacent kernels + concurrent independent
branches + a dependent-launch mechanism typically reaches 90–95% of an ideal megakernel.** A
megakernel proposal is therefore a proposal to spend most of the project's risk budget on the last
5–10% — which is sometimes exactly right (when rung 5 is unavailable, when the three conditions hold
strongly on a long chain, when the remaining 5% is the difference that matters), and which you must
argue *as such*, with the ladder above enumerated and each rung's expected recovery stated.

## Concurrency is not free of decisions

The moment you run branches concurrently (rung 3) or split roles inside a persistent kernel, you
have created a **resource allocation problem** that did not exist while everything was serialized:

- Which branch should get priority?
- How many CUs should each branch occupy?
- After you give a branch more resource and it gets faster — **has the critical path moved?**

Concurrency without answering these often produces *no speedup at all*, because the branch you
accelerated had slack and the one you starved did not. Serialized code has one bottleneck and it is
easy to find; concurrent code has a bottleneck that moves. See `resource_partition.md`.

## What to write in the report

Whatever your conclusion, the Analyze phase should contain, per candidate edge:

```
edge: <producer tile> -> <consumer tile>
  (1) partial handover possible?   yes/no + why
  (2) idle hardware during producer? measured occupancy/CU util
  (3) producer waves = ceil(blocks / (CUs * per_CU)) = N   window <= (N-1)/N * producer_us
  window_us: X    per-edge sync cost: Y (scope from tile_task_graph.md)
  verdict: PAYS / DOES NOT PAY / UNKNOWN(what measurement would settle it)
```

Plus the ladder, with the rung you are proposing and one line on why each cheaper rung is
insufficient. A proposal that skips the ladder is not ranked.

## Provenance

Distilled from public analyses arguing *against* the megakernel-by-default position, including
measured launch/bubble costs under graph capture, the dependent-launch-vs-persistence ablations
above, and an independent replication of a published megakernel result. Retained here specifically
because the rest of this knowledge tree argues the other way and an Analyze phase needs both. All
numbers are other people's, on other hardware, quoted as order-of-magnitude calibration.
**Measure yours.**
