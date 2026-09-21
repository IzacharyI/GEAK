# MegaMoE EP fusion — Author guide

Continue the current `full_persistent_pipeline` lane. The frozen baseline is only
the root commit; never recreate the lane or inspect external implementations.

Treat `BASELINE_OPERATOR_MAP` and the supplied baseline kernels as concrete
implementation references. For each region, choose reuse, extraction, inlining,
or equivalent rewriting to fit the persistent schedule. Function identity is
not required. Preserve observable math, quantization, dtype/layout, routing and
top-k behavior unless a deliberate change passes current-run correctness.

## Development rule

Fix the earliest concrete blocker returned by Verify, commit it, and return the
same lane for independent verification. Preserve working numerical bodies and
already-passing evidence. Do not write token-shaped stubs for contract checks.

If preflight still has source failures and no GPU blocker exists, implement the
next dependency-closed stage below. Re-sealing an unchanged HEAD twice is not
progress.

### Hard rules (fail-closed)

1. Never read the checker/verifier implementation as authoring input. The
   contract's AST/check code, PlanIR checker fields, `runtime_validation.py`,
   the contract tool source, and any oracle/reference tree are OFF LIMITS for
   shaping source. The contract tells you WHAT to implement, never HOW. Your
   authority is the candidate's own runtime behavior (compile, on-card smoke,
   relL2, launch count, path marker). If you find yourself studying how a check
   computes its verdict, stop and author the code that check is asking for.
2. Every lease must move the fused kernel BODY past the last committed HEAD.
   Once Host/ABI is closed, committing only more host/ABI/spec scaffolding is a
   failed turn: two consecutive turns landing on the same rung is forbidden, and
   `body=deferred` is never an acceptable output once the host shell exists.
   Each lease commits at least one new dependency-closed body stage of real
   device code that the persistent kernel executes.
3. Runtime correctness is the only authority. When the observed relL2 / launch
   count / path marker disagrees with what a static check "wants", fix the
   behavior the runtime saw, not the shape that satisfies the checker.

## Dependency-ordered build recipe

Do not rewrite GEMM math. Reuse the baseline quantizer, GEMM1, GEMM2, weighted
scatter and Combine numerical bodies; fuse their ownership and scheduling.

### 1. Host/ABI

- Allocate persistent Stage2/Combine payloads, outputs, peer tables and four
  disjoint counter domains: G1 completion, G2 claims, Stage2 close, and Combine
  claim/generation.
- Build immutable Stage2 and Combine compile specs from one selected
  `p2p_quant`; include `NW` and all codegen switches in the JIT key.
- Pass dynamic roots as declared kernel runtime parameters in exact signature
  order. Do not capture pointers or blindly splat tuples.
- Select exactly quant + one persistent launch and return its Combine output.

Done: the active Host values reach the cached compile call and real launch ABI.

### 2. G1 publication

- Claim one flat G1 stripe, wait dispatch payload, execute baseline GEMM1.
- Store activation/scales with cache 17.
- waitcnt every wave, uniform barrier, then thread-zero system increment once.

Done: G2 waits read only the G1 completion domain.

### 3. Shared G2/P2P emitter

- Build one single-unit `(m_block,n_block)` emitter shared by standalone and
  persistent paths.
- Thread Host `NW` into A-load, GEMM2, scale and scatter indexing.
- Use the caller's materialized `SharedAllocator().allocate(...).peek()` handle;
  `.buf.ptr` belongs to that handle, not a raw storage value.
- Weight and store the peer row with cache 19; release, barrier, close the
  dedicated m-block counter, and publish one arrival per valid row.

Done: standalone and persistent distributors call the same numerical emitter.

### 4. Unified G1/G2 scheduler

One block-uniform loop chooses in this exact order:

1. carried G2 continuation;
2. ready skew-preempt G2 claim;
3. G1 claim/execute/publication;
4. blocking post-G1 G2 claim;
5. terminate.

Use C1 below 4096 tokens and C16 otherwise; clamp the final chunk. Continuation
carries only next/count and performs no claim or wait. Never append a second G1
drain after this loop.

Done: a CTA enters Combine only with no continuation and exhausted G2 shard.

### 5. Combine third queue

- Claim one block; item is `claim*waves+wave`.
- Lane zero strictly waits for `topk*generation`.
- Read all direct top-k slots with cache 19; decode from the same `p2p_quant`
  as Stage2; reduce FP32 through one U1/U2/U4 path; store BF16 with cache 2.
- A bounded diagnostic wait must be removed before the production checkpoint.

Done: no Stage2/Combine tail launch remains.

### 6. Lifecycle and acceptance

- Clear claim heads, increment generation, pre-arm omitted tokens, keep arrivals
  monotone.
- Verify active `path=MEGA` ×8 and two launches before using numerical evidence.
- Then run 128→512→8192 accuracy, route-changing replay, and finally 8192 paired
  performance.

After a source edit, CPU checks are preflight only. The next independent Verify
must execute the exact HEAD on eight GPUs.

When the prompt supplies `STAGED_GPU_AUTHORING=1`, the Author may run the
earliest construction/JIT/small-correctness smoke before the full source
contract passes. This is development evidence only; final correctness,
liveness and performance remain independently verified.
Every such smoke must export `AITER_MEGAMOE_FUSE_ALL=1` plus the declared
Combine/quant switches, use `--mega-only` when the harness supports it, and
observe `path=MEGA` on all eight ranks. Otherwise it measured the scattered
fallback and is void.

Do not scaffold-and-bail. While the EP8 lease is available, keep the same turn
open across write → trace/JIT → smoke → traceback-guided repair. Return only
after the staged implementation produces a real smoke result or a concrete
bounded failure that requires the next turn.

When `PERSISTENT_JIT_CACHE_DIR` is supplied, export it as the FlyDSL/AITER
runtime cache root for every Author smoke. It is outside the candidate Git tree
and persists across Workflow waves; never substitute a per-workspace cache.

## Runtime progression

1. Host construction must create every Stage2/Combine buffer and pointer table.
2. FlyDSL trace/JIT must complete for the selected profile.
3. All ranks must report `path=MEGA`; profiler must observe two launches.
4. Pass relL2 <0.10 at 128, 512, and 8192.
5. Pass changing-route and smaller→larger graph replays without stale reads/hang.
6. Tune only after correctness; accept only paired 8192-uniform speedup >=1.03x.

## Implementation invariants

- Host bundle values must reach cached compile specs and the real launch ABI.
- Treat reachability as load-bearing: `NW` and `p2p_quant` must flow from the
  selected Host spec through the cached compiler into every A-load, GEMM2,
  scatter and Combine address/format calculation. A same-named constant or dead
  helper does not satisfy this invariant.
- Cached specs are immutable/hashable; output returned by Host aliases kernel output.
- G1 completion uses cache 17, waitcnt, uniform barrier, then one system publish.
- G2 has its own sharded extent and readiness counters; continuation drains C1/C16
  without reclaiming; skew is 5/4 and gated by ticket mod 6.
- Shared Stage2 owns descriptors/A preload and serves both persistent and standalone
  paths; 8192 uses BM64/BN512/BK256/NW8.
- P2P payload and E8M0 stores use cache 19 before release/close; each routed row
  publishes to its decoded peer/token readiness address.
- Combine decodes block-FP8+E8M0, reduces exactly top-k in FP32, selects one U1/U2/U4
  main path, stores BF16 with cache 2, and uses grid-wide runtime partitioning.
- Arrival counters are monotone. Reset only the rank-local claim head; increment
  Combine generation and pad omitted tokens before PLAN_READY/startup publication.
- LDS uses max-of-role-lifetimes and must remain <=163840 bytes.

## Failure handling

- A Python/FlyDSL exception is an Author code defect unless evidence identifies a
  harness failure.
- A GPU timeout may be a deadlock. Use the saved process/log classification; do not
  call it an agent failure without checking the GPU command.
- Once the same HEAD has produced the same device-sync hang twice, do not rerun the
  uninstrumented command. Add bounded device-side waits and/or per-protocol progress
  counters, dump the first unmet generation/address after a short timeout, then run
  only bs=128 until it produces relL2 or a new bounded failure.
- Bounded waits and cut-point switches are diagnostic-only. Remove them before the
  production checkpoint: final Combine must not continue into payload reduction
  after a readiness timeout, and no diagnostic-off branch may earn correctness.
- Do not run full performance after a failed construction/JIT/small correctness smoke.
- Never claim launch count, correctness, liveness, or speed without on-card evidence.

The full Skill contract is intentionally not part of Author context; independent
Verify evaluates it after each committed checkpoint.
