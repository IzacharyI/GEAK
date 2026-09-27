# Generic lessons for whole-operator fusion runs

You get this file as `MEGA_GENERIC_LESSONS` only in a `mode=mega` run with Expert Skills disabled.

Each lesson below cost at least one full round in an earlier run. None of them names an operator or
a design. They are failure patterns of the *process*: measuring, debugging, and choosing what to
build next. They never override COMMANDMENT, your role file, or a measured result. If one conflicts
with a measurement, the measurement wins; say so in your notes.

Which sections to read:

| role | sections |
|---|---|
| planner (`mega_search_lead`) | all |
| engineer | §1 – §4 |
| verify / director | §1, §2 |

---

## §1 Measurement: what counts as a number

1. **Only an on-card, measured result counts.**
   - A plan, a handoff line, "should be faster", or "compiles" is not a result.
   - Carry an unmeasured claim forward as `unverified`, never as progress.
2. **The official metric is the rank-max.**
   - One slow rank sets the operator's time.
   - A change that speeds up the median rank but leaves one straggler is a regression.
   - Always look at the per-rank spread, not only the reduced number.
3. **Paired and interleaved, or it did not happen.**
   - Arms that are timed sequentially drift by more than the effects hunted here.
   - Compare only base/candidate pairs interleaved inside one lease.
4. **Every variant in a sweep gets fresh inputs and fresh output buffers, then a fresh process.**
   - A variant that silently skips work can read the previous variant's correct results from a
     reused buffer. It then looks both correct and fast.
   - Confirm every sweep winner in a separate fresh-process run before you report it.
5. **Prove the activation, don't assume it.**
   - A fused path behind a switch runs only if the measured arm actually exported the switch *and*
     every rank printed the path marker.
   - A flag you added but never exported in the harness measures the old path.
   - Map each new switch to the arm's environment explicitly.
6. **Measure the denominator; never copy it.**
   - If the baseline itself moves (machine state, clocks, a neighbour on the GPUs), every speedup
     moves with it.
   - Before believing a surprising gain or loss, re-time the frozen baseline against its recorded
     number. A whole-run shift in the baseline is a machine problem, not a code effect.
7. **Every guard, not only the target.**
   - A design tuned at the largest shape often regresses the small ones: more setup cost, worse
     occupancy, a heavier synchronization path.
   - A fallback that is *allowed* at small shapes is not *free*:
     - once fallback is allowed, the search tends to settle for it;
     - say per guard whether the fused path ran;
     - report the measured ratio there too.

## §2 Correctness and liveness of a fused multi-rank kernel

1. **Correct once is not correct.**
   - Readiness counters, flags and queues must be correct on replay N of a captured graph, with
     different inputs, not only on the first call.
   - Two workable patterns:
     - monotonic generation counting, where every producer bumps **exactly once per generation**
       and waiters target `generation × expected_count`;
     - an in-graph reset that is properly ordered against every reader.
   - A producer whose bump count depends on data or on how work was tiled breaks generation targets:
     the first run passes, and a later replay hangs.
2. **Everyone you wait on must be resident.**
   - A workgroup spinning on a producer that is not scheduled yet deadlocks.
   - Size a persistent grid from the measured occupancy, not from the problem size.
   - Keep the spin bounded while debugging.
3. **Debug hangs with instruments, not hypotheses.**
   - A blind "fix, run, hang, kill" loop costs a full lease per guess and returns no information.
   - Build three instruments first:
     - a host-readable dump of `{observed, target}` for every counter;
     - a bounded spin that sets a failure flag and exits;
     - the smallest reproduction (e.g. capture plus a few replays of one small case).
   - The graph-contract runner's watchdog already names the stuck phase.
   - If `observed` is 0 everywhere, the publishing path is not running (wiring, gate, dead code). It
     is not an address or offset bug.
4. **Do not abandon a correct protocol because of a black-box signal.**
   - When a design that is known to be sound hangs, first localize which counter never reaches its
     target.
   - Replacing the protocol under an unexplained hang usually trades one hang for another.
5. **Check correctness at every configuration you actually select.**
   - A tile or launch parameter past a supported limit can silently drop part of the output.
   - Measure accuracy per selected configuration (shape class), with fresh buffers (§1.4).
6. **Direct evidence only.** Compare the graph-captured candidate output with the task's numeric
   reference. "Equal to another variant that was correct" is not correctness evidence.

## §3 Code generation and compiler traps

1. **Keep atomics and other heavy side effects out of the hot loop.**
   - An atomic read-modify-write inside the main loop, or in its tail, can change register
     allocation for the whole kernel.
   - The result is a slowdown or a deterministic fault far from the edit.
   - Publish once per tile or work item, after the loop.
2. **When one edit makes a distant site fault or slow down, compare the kernel metadata first**
   (VGPR/SGPR count, spills, scratch, LDS) before and after. Register pressure is usually the link.
3. **Trace-based DSLs lower loops from the traced program.**
   - A loop-carried variable must exist on every path.
   - Creating one only under a compile-time-constant branch can break the loop lowering.
   - Derive it from values that already exist, and use integer 0/1 instead of a boolean.
4. **Static checks do not catch trace-time or compile-time errors.**
   - A structural checker can pass a kernel that fails the first time it is traced on the card.
   - Compile and run the smallest case on the card early in every lease.
5. **JIT caches are per machine, not per tree.**
   - A flag- or environment-gated arm must prove it compiled to a distinct binary.
   - Run every arm in its own fresh process.

## §4 Choosing what to build next

1. **Separate structural levers from tuned constants.**
   - Two kinds of change:
     - a structural lever is a switch: a new stage order, a fused boundary, a queue;
     - a constant is a tuned number: tile size, wave count, prefetch depth.
   - Adopt or reject a lever with its constants retuned for the new structure. Untuned constants
     make a good lever look bad.
2. **Wire the baseline's tuning into the fused path before inventing new overlap.**
   - The unfused baseline usually ships per-shape tuned configurations for each stage.
   - If the fused path runs a stage with a default configuration, that stage is slower than the
     baseline's. No amount of overlap recovers that.
   - Check each stage's configuration first.
3. **Overlap is not free.**
   - Running stages concurrently helps only when the time hidden exceeds the machinery added:
     readiness polling, extra synchronization, lost occupancy.
   - Measure `wait` against `busy` with the stage meter (MEGA_MEASUREMENT_GUIDE §2) before and after.
4. **Build in stages, each proven on the card.**
   - For a multi-step rebuild, land and measure each stage.
   - Do not scaffold every stage and hand off an untested whole.
   - A rebuild may regress on the way to its goal. Keep its branch across rounds instead of
     reverting it at the first slower measurement, and state the stage it is at.
5. **Name the blocker class honestly.** For each failed round, record one class:
   - measurement (void or noisy evidence);
   - correctness;
   - liveness;
   - speed;
   - flow (budget or timeout).

   The next plan should target that class, not repeat the previous plan with a new name.
