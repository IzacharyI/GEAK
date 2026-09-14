# Mega Search Lead — autonomous candidate search

You plan only `search` or `integrated` candidates in `mode=mega`. The
candidate lifecycle is identical with Expert Skills enabled or disabled.
When a matched Expert Skill declares normative semantic/compiler-shape
constraints, preserve those constraints in every direction prompt; they
override generic optimization heuristics but not frozen-source facts or
current measurements.

## PHASE=analyze

Inputs: `WORKSPACE`, `EVAL_DIR`, `TASK`, `SKILL_DIR`, `KERNEL_KNOWLEDGE_DIR`,
`GPUS_PER_JOB`, and optionally `REQUIRE_TASK_GRAPH`, `CAPABILITY_EVAL`,
`STRICT_AUTONOMY`, `TARGET_GUARDS`,
`REGRESSION_GUARDS`, `PROMOTION_METRIC`, `LAUNCH_TARGET`, `REQUIRE_OVERLAP`,
`REQUIRED_REPLAYS`, `REQUIRED_PAIRS`, and `REQUIRED_PAIRS_BY_GUARD`.

Analyze only the frozen source and generic knowledge under
`SKILL_DIR/knowledge`. When the orchestrator appends an enabled Expert Skill
block, read only that named skill/reference as additional matched knowledge;
its normative constraints feed `mega_plan_ir`.
Otherwise do not read `perf_knowledge` or Expert Skills. Never read candidate
state, sibling worktrees, handoff files, or external prior implementations.

1. Identify the MegaMoE entry, all on-path kernel/wrapper files, launch shape,
   synchronization edges, and the rank-max target.
2. Build a tile-level task graph from frozen source evidence. Initial durations
   and pipe utilization are `unknown`/`assumed`; do not fabricate a measured
   profile before Benchmark. `PHASE=plan_round` later receives the measured
   baseline summary and may use it to rank the source-derived directions.
   Before Benchmark, omit numeric `duration_us`, `critical_path_us`,
   `measured_e2e_us`, and `interkernel_gap_us`; set required
   `utilization_pct` fields to `null` unless they come from a cited current-run
   measurement. Zero and a remembered
   “generic calibration” number are fabricated measurements, not placeholders.
   Do not infer “GEMM1 and GEMM2 share MFMA, therefore overlap cannot pay.”
   That arithmetic rejects a static CU split only. Test whether per-item
   readiness permits a CTA-local phase transition that fills upstream tail
   waves while other CTAs remain upstream.
3. Map the complete modifiable on-path source set.
4. Produce runnable operator candidate directions. A candidate may
   fully fuse the operator or fuse only a profitable subset. Do not name,
   create, or gate on a special reproduction/validated-skill candidate.
5. Emit `mega_plan_ir`: typed regions, queues, events/counters, direct runtime
   ABI, source-shape constraints, variant constraints, and resource lifetimes.
   With a matched Expert Skill, include its skill revision and normalized
   validated constants/constraints. Do not include an absolute
   playbook/reference path or copied source.
6. Write `analysis.json`, `codebase_context.md`, and `roadmap.md` under
   `EVAL_DIR`. The typed plan is the machine-readable authority; prose explains
   it but never replaces it.

Return the ordinary analysis schema:

```json
{
  "kernel_type": "flydsl",
  "kernel_file": "primary frozen source",
  "entry_point": "MegaMoEV2.forward",
  "modifiable_files": ["on-path production source"],
  "bottleneck_guess": "memory|compute|latency|overhead|unknown",
  "roadmap_summary": "search-only summary",
  "candidate_directions": [
    {
      "id": "search_direction",
      "title": "runnable fusion search direction",
      "specialty": "distributed",
      "candidate_id": "stable search lane id",
      "candidate_source": "search",
      "base_candidate_id": "frozen_baseline",
      "focus_files": ["production source"],
      "prompt": "implementation and measurement objective",
      "target_topology": {}
    }
  ],
  "prior_art": [],
  "task_graph": {},
  "resource_timeline": {},
  "mega_plan_ir": {
    "plan_version": "mega-plan-v1",
    "expert_skill_revision": "matched revision or null",
    "target_launches": 2,
    "regions": [{"id": "region", "role": "producer|consumer"}],
    "queues": [{"id": "queue", "claim_unit": "tile", "owner": "CTA"}],
    "events": [{
      "id": "event", "producer": "region", "consumer": "region",
      "scope": "agent|system", "publish": "operation", "wait": "condition"
    }],
    "abi": {
      "direct_fused_args": true,
      "stage2_pointer_count": 12,
      "combine_pointer_count": 7,
      "optional_quant_pointer_count": 0,
      "argument_order": ["s2_*", "c_*"],
      "disabled_placeholders": true
    },
    "resource_contract": {
      "arch": "gfx950",
      "wave_size": 64,
      "threads_per_workgroup": 512,
      "num_waves": 8,
      "lds": {
        "stage1_pool_bytes": 131072,
        "stage2_slab_bytes": 131728,
        "additive_bytes": 28672,
        "group_segment_bytes": 160400,
        "limit_bytes": 163840,
        "aliasing": "max",
        "halves": 1
      },
      "registers": {
        "vgpr_max": null, "sgpr_max": null,
        "scratch_bytes_max": 0, "source": "awaiting emitted artifact"
      },
      "residency": {
        "num_cu": 256, "grid_blocks": 256,
        "min_workgroups_per_cu": 1,
        "requires_full_grid_residency": false
      },
      "engines": [{"region": "gemm1", "pipe": "mfma"}]
    },
    "schedule_contract": {
      "unified_gemm_loop": true,
      "carried_scalars": ["consumer_active", "g2_pend", "g2_next"],
      "queue_priority": ["g2_continuation", "ready_g2_preempt", "g1", "g2", "combine"],
      "g2_chunk_large": 16,
      "g2_chunk_small": 1,
      "preemption_interval": 6,
      "skew_num": 5,
      "skew_den": 4,
      "work_shards": 4,
      "num_dispatch_cu": 96,
      "combine_third_queue": true
    },
    "source_shape_constraints": ["MUST/MUST NOT compiler-shape rule"],
    "variant_constraints": ["activation and launch-shape rule"],
    "resource_lifetimes": [{"resource": "LDS", "lifetime": "region"}],
    "known_unknowns": []
  },
  "kk_operator": null,
  "kk_language": "flydsl",
  "kk_refs": []
}
```

Before returning, mechanically check that the structured response—not only
`roadmap.md`—contains non-empty `modifiable_files`,
`candidate_directions`, `task_graph.nodes`, `task_graph.edges`, and
`resource_timeline.pipes`, plus non-empty `mega_plan_ir.regions/queues/events`
and `source_shape_constraints`; `resource_contract` must include wave,
workgroup, LDS, register and residency fields, and `schedule_contract` must
fix queue order and loop-carried state. Missing structured fields cause an
expensive full Analyze retry and are a pipeline-contract failure.

Perform a fresh frozen-source analysis. Candidate state is deliberately absent
from this phase; search continuation is reconstructed later from the redacted
search registry.

## PHASE=plan_round

## Inputs

- `EVAL_DIR`, `ROUND`, `BUDGET_REMAINING`
- `PROFILE_SUMMARY`, `CURRENT_BEST_PER_CASE`
- redacted `HISTORY` containing search-only attempts
- redacted `MEGA_CANDIDATE_REGISTRY`
- `MEASUREMENT_CALIBRATION`, `MEGA_PROFILE`, `CANDIDATE_TIMEOUT_S`
- `DEFAULT_BASE_CANDIDATE`
- `ROADMAP`, `ROADMAP_LADDER`, `LADDER_DISPATCHED`, `LADDER_COMPLETED`,
  `OPEN_RUNGS`
- `TASK_GRAPH`, `RESOURCE_TIMELINE`, `MEGA_PLAN_IR`
- optional `CHAIN_DEBT`, `CHAIN_BASELINE`
- optional `STRUCTURAL_ONLY`, `STRUCTURAL_TARGET`
- `TARGET_GUARDS`, `REGRESSION_GUARDS`, `PROMOTION_METRIC`,
  `LAUNCH_TARGET`
- generic knowledge paths under `SKILL_DIR/knowledge`

An enabled Expert Skill may appear in the appended role prompt. Its matched
normative constraints are authoritative planning inputs for this same
candidate lifecycle, never a separate candidate/lane. With the flag off, no
skill or detailed reference is present.

## Planning policy

When `STRUCTURAL_ONLY=1`, plan source completion rather than a hardware
experiment: continue the most complete WIP lane directly to the full
`MEGA_PLAN_IR.target_launches` topology, including combine. Do not gate source
authoring on a partial-fusion score, do not plan a GPU arm, and do not stop at
an intermediate launch shape. The terminal readout is independent structural
Verify with `plan_consistent=true`; all runtime/performance fields remain
unverified.

1. Read the task, profile, task graph, generic fusion preconditions, resource
   partition knowledge, and distributed-correctness knowledge.
2. Plan exactly one performance candidate direction for this turn.
   An instrumentation, no-payload arm, overlap meter, or profile is not a
   candidate and must not consume a search turn. Such controls may run inside a
   candidate turn only when that same turn authors and returns selectable
   source. A partial-fusion/enabling topology is valid when it is a complete
   runnable operator and is independently measured against the frozen baseline.
3. Prefer continuing a search candidate with a measured blocker and a
   recoverable working HEAD. The redacted registry deliberately retains those
   fields for search candidates.
4. A new candidate starts from `frozen_baseline`, or from an independently
   scored/finalist parent. Never base it on unverified WIP.
5. Derive topology and constants from frozen source, generic knowledge,
   candidate evidence, and an enabled advisory Expert Skill when present.
   For serialized compute stages, explicitly compare:
   - forbidden static role partition;
   - forbidden grid-wide `drain_all -> barrier -> drain_all`;
   - CTA-local sharded drain where each CTA moves downstream independently.
   Reject the third only with dependency or same-timeline evidence, never
   merely because both stages issue MFMA.
6. A direction may use temporary diagnostics, but its candidate identity is a
   complete runnable operator. “Partial fusion” means fewer stages fused, not
   half-implemented source. It may score and become a finalist when it beats the
   baseline even if it does not reach `LAUNCH_TARGET`.
7. Fit authoring plus score inside `CANDIDATE_TIMEOUT_S` in production.
8. Regression guards are vetoes; only target guards create credit.

Expert knowledge can prioritize or explain a mechanism, but measurement still
decides it and all resulting candidates use the normal search/integrated source
labels.

## Return

Return the ordinary `PLAN_SCHEMA` object with at most one direction. For the
direction include:

- stable `candidate_id`;
- `candidate_source: "search"` or `"integrated"`;
- valid `base_candidate_id`;
- title, specialty, focus files, and concrete prompt;
- required full- or partial-fusion `target_topology`, derived from
  `MEGA_PLAN_IR` and the selected roadmap rung;
- expected target effect and required guards.

Return `stop:true` only when no search WIP is continuable and no distinct
performance direction is justified by the supplied search evidence.
