---
id: megamoe_ep_mega_fusion
title: 'MegaMoE EP8 persistent tile-pipeline playbook'
kind: expert_skill
mode: mega
reference_revision: m25-repro-v2
reference_file: recipe_v1.md
constraint_profile: semantic_and_compiler_shape
authors:
  - GEAK Team
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
expects:
  isolated_speedup_min: 1.01
  e2e_delta_min_pct: 1.0
  parity: required
provenance:
  source: validated_knowledge
  origin: deconstructed_capability
  reuse_mode: common_mega_candidate_lifecycle
incumbent:
  label: M2.5_persistent_megakernel
  is_ceiling: false
  measured_gain_vs_baseline_pct:
    tokens512_uniform: 1.49
    tokens8192_uniform: 4.71
validation:
  status: mega_only
  last_verified: '2026-09-04'
  gpu: gfx950/MI355X
  measured:
    isolated: 1.0448
    e2e_pct: 4.48
    parity: pass
  artifact: recorded_measurement_only
role: normative_prior
supersedes: []
---

# MegaMoE EP8 persistent tile-pipeline playbook

Use this skill as validated design knowledge inside the ordinary `mode=mega`
Analyze → Plan → Author → Verify loop. It does not create a reproduction mode,
reserved lane, source label, or scheduler. When the match and baseline revision
apply, the detailed recipe's `MUST`/`MUST NOT` source-shape constraints override
generic optimization heuristics and Engineer improvisation; correctness and
current hardware evidence still override the recipe.

## Required reading

Read repository-relative `recipe_v1.md` as a detailed validated reference. It
records one design known to reach M2.5-class behavior, including synchronization
and geometry details plus the FlyDSL source shapes on which it was validated.
Re-derive applicability from the current frozen source/task graph. Do not
replace the direct fused ABI, nested emitter, unified queue loop, publication
scope or generation protocol unless current evidence proves the replacement
and the deviation is recorded.

## When to use

Use for `mode=mega`, EP8 MegaMoE V2 on gfx950. With
`use_expert_skills=false`, this skill and its reference must be absent and the
same roles derive candidates without M2.5 knowledge.

## Mechanism

The target is not wrapper-level launch fusion. Build a tiled/instruction-level
persistent pipeline whose task graph, resource ownership and readiness edges are
explicit:

- CTAs move from GEMM1 to ready GEMM2 work when their local shard drains,
  without a global stage barrier;
- GEMM2/P2P and combine can likewise be phase-staggered across CTAs;
- LDS/VGPR/CU residency, publication/acquire scope, monotone generations and
  graph replay safety are correctness constraints;
- full fusion is the known M2.5 design, while a measured faster partial fusion
  is also a valid Mega candidate.

For the pinned baseline, the following are closed source shapes, not equivalent
implementations: separate GEMM1-then-GEMM2 loops, fused pointers hidden in
trailing dispatch-table slots, a module-level Stage2 device body replacing the
nested closure/JIT emitter, and an agent-scope G1 completion publish. The
recipe specifies the validated alternatives.

## Procedure

Run the normal Mega analysis and candidate loop:

1. Build the tile task graph and identify missing readiness edges.
2. Use the reference to prioritize a credible persistent pipeline; do not copy
   or read an external M2.5 source tree.
3. Author resumable production-source checkpoints in the ordinary candidate
   lane.
4. Debug JIT, direct correctness and graph liveness on the actual candidate.
5. Use paired rank-max measurement against frozen MegaMoE V2. Continue
   optimizing beyond the reference when evidence supports it.

When GPUs are unavailable, run the recipe's independent structural tier after
authoring. The source oracle is visible only to the read-only verifier, never
to the author. A byte-exact oracle copy is checker calibration, not capability
evidence.

## Completion

An M2.5-class full-fusion candidate confirms:

- `path=MEGA` on all eight ranks;
- exactly two launches per rank;
- direct graph-path `relL2 < 0.10`;
- rank-max paired performance against the frozen scattered baseline;
- the required liveness and finalist guard contract.

## Do-no-harm notes

The reference never overrides measurement. A skill-enabled candidate uses the
same correctness, artifact, graph, liveness, regression and frozen-baseline
speed gates as a skill-disabled candidate.

## Sources

- `recipe_v1.md`: detailed validated design reference.
- Frozen baseline identity is supplied and verified by the workflow; no
  resolvable repository address is exposed to the authoring Skill.
