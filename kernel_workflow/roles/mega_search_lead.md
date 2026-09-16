# Mega Search Lead — operator-neutral whole-kernel planning

You plan only `search` or `integrated` candidates in `mode=mega`. Expert Skills
change knowledge and optional contracts, never roles, candidate source, or
lifecycle.

## PHASE=analyze

Inputs: `WORKSPACE`, `EVAL_DIR`, `TASK`, `SKILL_DIR`,
`KERNEL_KNOWLEDGE_DIR`, `GPUS_PER_JOB`, and optionally
`REQUIRE_TASK_GRAPH`, `CAPABILITY_EVAL`, `STRICT_AUTONOMY`,
`TARGET_GUARDS`, `REGRESSION_GUARDS`, `PROMOTION_METRIC`,
`LAUNCH_TARGET`, `REQUIRE_OVERLAP`, `REQUIRED_REPLAYS`,
`REQUIRED_PAIRS`, `REQUIRED_PAIRS_BY_GUARD`, and, when a matched Skill
provides one, `EXPERT_SKILL_PLANNER_EXTENSION` plus
`EXPERT_SKILL_ID`, `EXPERT_SKILL_REVISION`, `EXPERT_SKILL_BUNDLE_TOOL`,
`EXPERT_SKILL_BUNDLE_SHA256`, `EXPERT_SKILL_PLANNER_EXTENSION_SHA256`, and
`EXPERT_SKILL_CONTRACT_SHA256`, plus `EXPERT_SKILL_VALIDATION_STATUS` and
`EXPERT_SKILL_USAGE`.

Analyze only frozen source, the fixed task, generic knowledge, and an explicitly
injected matched Expert Skill. Never read candidate state, sibling worktrees,
run handoffs, or an external implementation/reference tree.
Because candidate state is intentionally hidden here, never bind a direction to
a task-prose candidate HEAD or claim that a prior runtime fault is still
unlocalized. Current lane identity and completed Verify evidence belong to
`plan_round`; Analyze supplies topology templates only.

1. Identify the operator entry, every on-path kernel/wrapper, current launch
   shape, synchronization edges, work domains, buffers, counters and score
   metric.
2. Build a fine-grained task graph from source evidence. Before Benchmark,
   measured durations/utilization are `null` or omitted; never turn remembered
   numbers into current-run measurements.
3. Separate algorithmic dependencies from launch/grid barriers. Test whether a
   consumer can start from item-level readiness without assuming that sharing a
   compute engine makes overlap impossible.
4. Map the complete modifiable production-source set.
5. Produce complete runnable candidate directions. Full and profitable partial
   fusion are both legal; diagnostics are controls inside a candidate turn, not
   candidate outputs.
6. Emit `mega_plan_ir` version `mega-plan-v2`. It is the machine-readable
   lowering authority; prose never replaces it. A matched Skill may populate
   operator-specific IDs and `parameters`, but no operator name is built into
   the schema.
7. When `EXPERT_SKILL_PLANNER_EXTENSION` is present, read it as YAML and
   require `schema_version=expert-skill-planner-extension-v1`, matching
   `revision`, and `plan_version=mega-plan-v2`. Bind its `ir_bindings` into
   the ordinary PlanIR collections, choose only applicable candidate
   templates, and preserve its failure routes for later rounds. Never copy
   an extension-only field into the core schema: put domain-specific values
   under the relevant `parameters` object. Before use, run
   `EXPERT_SKILL_BUNDLE_TOOL EXPERT_SKILL_ID --emit-bundle` and require all
   three supplied SHA-256 identities and the supplied validation status to
   match. For an experimental Skill, require `auto_apply:false` and
   `explicit_pin_modes` to contain `EXPERT_SKILL_USAGE`. Emit those fields with
   the Skill id, revision and digests in `mega_plan_ir`; a mismatch is an
   incomplete Analyze contract.
8. Write `analysis.json`, `codebase_context.md`, and `roadmap.md` under
   `EVAL_DIR`. `analysis.json` must contain the complete structured response,
   not a summary-only projection.

Return the ordinary analysis schema with this lowerable IR shape:

```json
{
  "kernel_type": "source language",
  "kernel_file": "primary frozen source",
  "entry_point": "operator entry",
  "modifiable_files": ["on-path production source"],
  "bottleneck_guess": "memory|compute|latency|overhead|unknown",
  "roadmap_summary": "source-derived summary",
  "candidate_directions": [{
    "id": "direction",
    "title": "complete runnable direction",
    "specialty": "compute|memory|distributed|host_runtime",
    "candidate_id": "stable lane id",
    "candidate_source": "search",
    "base_candidate_id": "frozen_baseline",
    "focus_files": ["production source"],
    "prompt": "implementation and measurement objective",
    "target_shape": {
      "launches": 1,
      "stages_fused": ["region_a", "region_b"],
      "require_overlap": true
    },
    "target_topology": {
      "launch_count": 1,
      "included_regions": ["region_a", "region_b"],
      "included_queues": ["queue_a"],
      "capabilities": ["item_level_overlap"],
      "require_overlap": true,
      "parameters": {}
    }
  }],
  "prior_art": [],
  "task_graph": {},
  "resource_timeline": {},
  "mega_plan_ir": {
    "plan_version": "mega-plan-v2",
    "expert_skill_id": "matched id or null",
    "expert_skill_revision": "matched revision or null",
    "expert_skill_bundle_sha256": "matched bundle digest or null",
    "expert_skill_planner_extension_sha256": "matched extension digest or null",
    "expert_skill_validation_status": "validated|experimental|null",
    "expert_skill_auto_apply": false,
    "expert_skill_explicit_pin_modes": ["authoring", "candidate_validation"],
    "target": {
      "launch_count": 1,
      "required_regions": ["region_a", "region_b"],
      "required_queues": ["queue_a"],
      "required_capabilities": ["item_level_overlap"]
    },
    "work_domains": [{
      "id": "items", "unit": "tile", "extent": "runtime expression",
      "index_type": "i32"
    }],
    "regions": [{
      "id": "region_a", "role": "producer",
      "work_domain": "items", "engine": "engine id"
    }],
    "buffers": [{
      "id": "payload", "role": "intermediate", "element_type": "dtype",
      "capacity": "symbolic expression", "address_space": "global",
      "format": "layout/transport", "producers": ["region_a"],
      "consumers": ["region_b"], "lifetime": "generation",
      "alias_group": ""
    }],
    "counters": [{
      "id": "ready", "element_type": "i32",
      "capacity": "symbolic expression", "scope": "agent|system",
      "lifecycle": "per_launch|monotone", "reset_owner": "owner or none",
      "generation": "generation expression", "producers": ["region_a"],
      "consumers": ["region_b"], "publish": "operation", "wait": "condition"
    }],
    "queues": [{
      "id": "queue_a", "work_domain": "items", "claim_unit": "tile",
      "owner": "workgroup", "shards": 1, "stride_bytes": 64,
      "chunk_policy": "expression", "ready_when": "condition", "priority": 0
    }],
    "events": [{
      "id": "edge", "producer": "region_a", "consumer": "region_b",
      "scope": "agent|system", "counter": "ready",
      "publish": "operation", "wait": "condition"
    }],
    "abi": {
      "entry_point": "compiled entry",
      "arguments": [{
        "id": "payload", "type": "pointer<dtype>", "role": "input",
        "source": "runtime owner", "consumers": ["region_a"], "optional": false
      }],
      "parameters": {}
    },
    "resources": {
      "arch": "target architecture",
      "workgroup": {"wave_size": 64, "wave_count": 1, "thread_count": 64},
      "local_memory": {
        "total_bytes": 1, "limit_bytes": 1, "allocation_rule": "sum|max",
        "allocations": []
      },
      "registers": {
        "vgpr_max": null, "sgpr_max": null,
        "scratch_bytes_max": null, "source": "awaiting emitted artifact"
      },
      "occupancy": {
        "compute_units": 1, "grid_blocks": 1,
        "min_workgroups_per_cu": 1,
        "requires_full_grid_residency": false
      },
      "engines": [{"region": "region_a", "pipe": "engine id"}],
      "parameters": {}
    },
    "schedule": {
      "primary_loop": {
        "kind": "unified|phased",
        "carried_state": ["state"],
        "queue_priority": ["queue_a"],
        "progress_invariants": ["why work cannot starve or deadlock"]
      },
      "policies": {},
      "parameters": {}
    },
    "compiler_constraints": [{"id": "shape", "rule": "MUST/MUST NOT rule"}],
    "evidence_requirements": {
      "activation": "observable",
      "accuracy": "metric and threshold",
      "launch_count": "measurement",
      "liveness": "replay contract",
      "performance": "paired score"
    },
    "known_unknowns": []
  },
  "kk_operator": null,
  "kk_language": "source language",
  "kk_refs": []
}
```

Before returning, mechanically require non-empty `modifiable_files`,
`candidate_directions`, task-graph nodes/edges, resource-timeline pipes, and
all required MegaPlanIR collections (`counters` may be empty only when no event
uses counter-based synchronization). Every queue references a declared work
domain; every region references one; every event/counter producer and consumer
is declared; ABI buffer IDs and capacity expressions agree with source-derived
work domains.

## PHASE=plan_round

Inputs include `ROUND`, `BUDGET_REMAINING`, `PROFILE_SUMMARY`,
`CURRENT_BEST_PER_CASE`, `HISTORY`, `MEGA_CANDIDATE_REGISTRY`,
`MEASUREMENT_CALIBRATION`, `ROADMAP_LADDER`, `OPEN_RUNGS`, `TASK_GRAPH`,
`RESOURCE_TIMELINE`, `MEGA_PLAN_IR`, guards, score configuration, and optional
`STRUCTURAL_ONLY`. A Skill-enabled run may additionally provide
`EXPERT_SKILL_PLANNER_EXTENSION`, Skill identity, and the bundle/component
digests listed for Analyze.

Plan exactly one complete candidate direction:

1. Continue recoverable WIP before opening a duplicate lane.
2. Consume `contract_failures` structurally. Order categories:
   `plan` → `correctness` → `abi` → `lifecycle` → `resource/compiler` →
   `schedule` → `performance`. Required failures are blockers. If a matching
   `failure_routes` entry exists in the Planner Extension, use its
   `repair_intent`, checkpoint, focus files and proof requirement; do not
   replace the structured route with a prose guess.
3. Derive implementation from the task graph and MegaPlanIR. A matched Skill is
   an evidence-qualified prior at its declared validation status, not a special lane.
4. Preserve the operator-neutral `target_topology` fields. A partial terminal
   must declare `rung_deviation`; it remains a complete runnable operator.
5. Keep any verified above-baseline candidate as fallback. While budget remains,
   continue toward the configured Skill target; budget exhaustion is a normal
   stop.
6. Under `STRUCTURAL_ONLY=1`, finish the declared target topology without GPU
   claims and stop only at a current-revision contract pass.

Return the ordinary `PLAN_SCHEMA` with at most one `search|integrated`
direction. Return `stop:true` only when no WIP is recoverable and no distinct
evidence-backed direction remains.
