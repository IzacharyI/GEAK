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
next dependency-closed batch in this order: Host/ABI → G1 cache/publication →
shared Stage2+P2P → unified G1/G2 scheduler → Combine+generation. Re-sealing an
unchanged HEAD twice is not progress.

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
