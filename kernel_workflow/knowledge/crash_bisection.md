# Crash Bisection — Locating a Fault Inside One Kernel Without a Debugger

A **method** card for the situation a fused kernel puts you in: the kernel dies with an illegal
memory access or hangs, the failure is 100% reproducible, and there is no line number. A fused
kernel is a single launch, so nothing about *where* it died is visible from the outside — the trace
has one record, and that record says "it failed".

The reflex is to read the code harder. That does not converge: a fused megakernel is thousands of
lines of index arithmetic and every line looks plausible. What converges is bisection, and the unit
being bisected is **how far the kernel is allowed to get**.

## The compile-time truncation ladder

Instrument the kernel with numbered cut points and select one at build time:

```
    cut 0   ──  return immediately after the prologue
    cut 1   ──  return after the descriptor/index computation
    cut 2   ──  return after the first tile's loads issue
    ...                       (15 cut points is a workable number)
    cut N   ──  no truncation; the whole kernel
```

Each cut is a `return` guarded on a compile-time constant read from an environment variable, so
every rung is a **different binary with no runtime branch on the hot path**. Then walk the ladder:
the first cut that fails is the segment that contains the fault. One measured instance ran at
**~33 s per rung** — fifteen rungs is under ten minutes of wall clock, and each rung's answer is
binary and uncontestable.

Why compile-time and not a runtime flag:

- A runtime branch changes register allocation and scheduling, so the truncated build and the full
  build are not the same kernel. A heisenbug that moves when the flag moves has told you nothing.
- Truncation deletes the code below the cut, so **the code that would have run cannot be blamed**.
  A runtime early-exit still compiles everything below it, and on a resource-limited kernel that
  code is still costing registers and LDS — which may be the actual fault.
- Dead-code elimination below the cut is a feature: if a rung stops failing because the compiler
  removed something, that something is implicated. Check the emitted ISA per rung and record the
  register/LDS/spill numbers alongside the pass/fail. The ladder produces an occupancy sweep for
  free.

## What the rungs must be

Cut at **semantic boundaries**, not every N lines. The useful boundaries in a fused kernel are the
ones where a class of fault becomes possible for the first time:

| rung is placed after | first becomes possible |
|---|---|
| prologue / role assignment | a block index outside the resident range |
| descriptor and index computation | an out-of-range table lookup |
| the first load issue | a bad address, a missing bounds clamp |
| the first barrier | a non-uniform barrier — a subset of waves reaching it |
| the first counter wait | a deadlock (target never reachable) |
| the first counter publish | a premature publish, visible only to a consumer |
| the epilogue store | a bad output address, an aliasing write |

Two rungs deserve special handling because they fail *differently*:

- **A rung that hangs instead of crashing has still answered.** Hang and illegal-access are
  different faults and the ladder separates them for free; run every rung under a timeout so a hang
  is a recorded result rather than a lost lease.
- **A rung that passes when the one below it failed is a finding, not a mistake.** It means the
  fault is masked by code that runs later — most often a clamp or a barrier that happens to make an
  out-of-range index harmless.

## Confirm the location before fixing it

The ladder gives a segment, not a cause. Before writing a fix, add a probe inside the implicated
segment that prints the quantities the segment computes, and check them against what they should be.

A worked example of why this step is not optional: a hang was attributed to a stale read past a
barrier, on the theory that the barrier does not invalidate the vector L1. A probe placed in the
implicated segment printed `num_valid=52864` and `total_work=4956`, and `4956 = ceil(52864/128)*12`
— the work count was **7.5× below the clamp**, so the clamp was never the reachable limit and the
theory was false. That probe cost one run. The fix derived from the theory would have cost a wave
and would not have worked.

## When the failure only appears on the second replay

A fused kernel that publishes results between its own stages can pass single-shot and fail under
graph replay. The mechanism is worth recognizing on sight, because the ladder above will show every
rung passing:

the payload buffer is zeroed once at allocation, and graph replay feeds identical inputs, so on
**replay 1** a premature or unordered read returns zeros and adds nothing — correctness passes and
the error metric does not move. On **replay 2** the same read returns the *previous generation's*
partial result. So the acceptance rule is not "it ran"; it is **it ran N times**, and any candidate
that publishes across a stage boundary must be replayed before its correctness number is believed.
See `gfx950_lowering.md` for the ordering bug that produces this on this target.

## Cost, and when not to do this

Fifteen rungs at half a minute each is cheap in wall-clock and free in reasoning risk, and it is
device-time that produces an answer rather than a hypothesis. Do it as soon as a fused candidate
fails, not after two rounds of reading the source.

It is the wrong tool when the failure is **not** reproducible. The ladder assumes rung N's verdict
is stable; against a race that fires one run in ten, a rung that passes means nothing. Establish
reproducibility first (run the untruncated kernel enough times to bound the failure rate), and if
the fault is intermittent, that fact is itself the finding — an intermittent illegal access in a
kernel with counter-based handover is an ordering bug, and the ladder will not localize it.
