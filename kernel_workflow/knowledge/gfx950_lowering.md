# gfx950 Lowering — What the Source Says vs What the ISA Does

A **measured** card, not a method card. Every number and every lowering below was read out of
emitted ISA on a gfx950 part, device-free, with `COMPILE_ONLY=1`. It exists because each of these
facts was independently re-derived by an engineer who then had no way to leave it anywhere, and the
next wave paid to find it again.

Two standing rules apply to everything here:

- **These are lowerings, not language semantics.** A compiler update can change any of them. The
  method in the last section costs no GPU lease — re-run it rather than trusting this file.
- **A source-level reading of any of these is not evidence.** Every entry below was believed to be
  something else until the ISA was dumped, and in two cases the source-level reading was the benign
  one.

## Synchronization primitives — what actually gets emitted

| what you wrote | what gfx950 emits | the trap |
|---|---|---|
| `fx.barrier()` (workgroup barrier) | `s_waitcnt lgkmcnt(0)` then `s_barrier` | **It does not drain `vmcnt`.** LDS traffic is ordered; outstanding `buffer_store`s are not |
| `fence_agent_release` | `buffer_wbl2 sc1` | A **cross-XCD L2 writeback**. Not a local fence |
| system-scope acquire | `buffer_inv sc0 sc1` | Invalidates; pairs with the above and must be on the consumer side |
| non-temporal load | `nt` bit on the load | Orders nothing. A `cache_modifier` is not a release |

### The publish bug this produces, in full

The natural way to publish a tile from a persistent producer is: every wave finishes its epilogue
stores, hit a barrier, then thread 0 bumps a readiness counter. **That orders nothing.** After
`fx.barrier()`, thread 0's own `vmcnt(0)` covers only thread 0's wave; the other waves' epilogue
`buffer_store`s are still in flight when the counter goes up. The consumer then reads a tile that is
partially written, and — because the payload buffer is zeroed once at allocation and graph replay
feeds identical inputs — **the first replay passes and relL2 never moves.** It fails on replay 2, as
the previous generation's partial sums, out of the consumer's own XCD L2.

The shape that is correct: take the **agent-scope release on every wave** before the barrier, then
the counter bump under an exec mask. Verify by reading back the emitted ISA — the correct form
reads `buffer_wbl2 sc1; s_waitcnt vmcnt(0); s_barrier` on all threads, followed by a single
`global_atomic_add`. Then verify three more things, none of which the producer side shows you:

1. the paired **acquire** exists on the consumer,
2. no load of the produced region was **hoisted above** the wait,
3. the readiness flag's **clear** cannot land after the producer's increment (that ordering is a
   hang, and it is the twin failure of the race above — fixing one direction can open the other).

### Price the release before you place it

`buffer_wbl2 sc1` is a cross-XCD writeback because agent scope on an 8-XCD part cannot be satisfied
inside one die's LLC. It executes **once per output tile per block**, so its cost scales with the
tile count, not with the edge count: 4608 executions at the large shape of one measured operator.
That is not obviously small and it is the reason a *publish-only* arm — producer publishes, nobody
consumes, the consumer still waits at the launch boundary — is worth a lease on its own. It prices
the producer half of the edge in isolation, and it is occupancy-neutral by construction, so its
reading is the publication cost and nothing else.

Cheaper alternative when a full release is more than the edge needs: on a `buffer_store`, the
cache-policy constant is the whole fix. Switching `cache_modifier` from non-temporal to
**write-through** was measured cheaper than a per-block release fence on the same edge. Check what
the edge's scope actually requires (`tile_task_graph.md`, the scope column) before paying for agent
scope.

## Occupancy arithmetic that only the ISA can settle

**The profile summary's VGPR count is reported in units of 2 on this target.** A summary saying
`VGPR=128` is an ISA `.vgpr_count` of **256**. Any occupancy derived from the summary is wrong by a
factor of two in the direction that makes the kernel look fine. Cross-check every one against
`.vgpr_count` in the ISA metadata before building an argument on it.

A measured example of the whole table, from one fused-MoE candidate, to show the shape of what to
collect (**these are that operator's numbers, not the card's — collect your own**):

| leg | LDS | VGPR | spill | wg/CU | wave/SIMD | limiter |
|---|---|---|---|---|---|---|
| stage1 @8192, tile 128×512, 512 thr | 159744 | 256 | 48 | 1 | 2.0 | **tied** |
| stage2 @8192, tile 64×256, 256 thr | 66112 | 202 | 0 | 2 | 2.0 | LDS |
| stage1 @512, tile 64×512, 512 thr | 79872 | 234 | 0 | 2 | 2 | VGPR |
| stage2 @512, tile 32×256, 256 thr | 33088 | 168 | 0 | 4 | 3 | VGPR |

### One VGPR count for both roles — the constraint that decides the shape

`pipe_occupancy.md` says a fused kernel has one resource shape and every op must live inside it.
The sharp form of that, and the one that actually killed a direction:

> **Register unification alone can halve the occupancy of the cheaper role, with zero LDS
> involvement, and no role partition can fix it.**

In the table above, fusing the two stages at the small shape gives both roles stage1's 234 VGPR,
dragging the stage2 role from 3 waves/SIMD to 2 — a 33% loss bought by nothing. LDS can be
negotiated (two 66112 B tiles in one 512-thread block = 132224 ≤ 159744, occupancy-neutral); the
register file cannot. So the honest outcome is a fused arm **gated to the large bucket**, falling
back to the scattered path at the small one — which is a legitimate design, not a failure to finish
(`pipe_occupancy.md`, "name the op with the worst fit and price it").

### Residency is a correctness precondition and it may already be binding

With 512-thread / 159744 B blocks the device holds 1 workgroup per CU, so **at most 256 blocks are
resident** on a 256-CU part. Consequences, in the order they bite:

- Any grid-wide join in the fused form must have every participant index inside `[0, resident)`.
  A wait still targeting the pre-fusion `block_num` **hangs**.
- One measured operator's combine leg already launches exactly 256 blocks at the large shape — the
  cap was *already* binding before any fusion, which is easy to miss because nothing had failed yet.
  Its fused participant count would be `min(persist_cu, num_cu) = 240`, so every 256-target wait
  becomes a hang.
- Check whether each leg has a residency assertion at all. In that same operator two legs had one
  and two had none.
- Repartitioning for residency is also where a block-quantization tail goes away: a 6720-block grid
  (13.1 waves) repartitioned to 252 resident blocks collapses to one persistent wave. Take the
  credit, but keep the barrier counts uniform across the halves of a split block, or the
  workgroup-wide barrier is no longer uniform and the kernel hangs for a second reason.

## The lease-free method that produced all of this

None of the above cost a GPU lease. The whole procedure:

1. Build with `COMPILE_ONLY=1` and dump the final ISA for each leg and each shape.
2. Read `.vgpr_count`, LDS bytes (both raw and 512-granule), spill count, and derive wg/CU and
   waves/SIMD yourself. Do not take them from the profile summary.
3. **`sha256` the off-arm ISA against the pristine tree.** Identical is the null control: it proves
   the switch is genuinely off and the arm cannot be a stale cached binary. On-arm must differ.
4. Diff the emitted instruction counts between arms (`+1 global_atomic_add`, `+1 s_barrier`,
   `spills 48→45`) — this is what makes "occupancy-neutral by construction" a claim you can check
   rather than a hope.

One thing this method **cannot** settle: JIT arm isolation by cache-directory diff. Under
`COMPILE_ONLY=1` the child aborts before anything is persisted and the cache gains zero directories
for either arm, so the directory-diff probe from `jit_arm_isolation.md` needs a real lease. What it
*can* prove device-free is weaker but not nothing: the two arms emit different ISA sha and carry
different kernel names, so they cannot be the same cached binary unless the cache is broken. Run the
directory diff on the first real leg anyway.

## Provenance

Measured on gfx950 (8 XCD, 256 CU) from emitted ISA during a fused-MoE investigation; the
synchronization lowerings, the VGPR-unit discrepancy, the residency cap and the publish race were
each found by dumping ISA after a source-level reading had said something else. Operator-specific
numbers are marked as such. **Re-dump before you rely on any of it.**
