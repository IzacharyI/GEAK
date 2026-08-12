# Analysis skills — index (kernel_workflow)

Pluggable **profile-analysis** skills for Kernel Workflow. `profile_engineer` runs at most ONE of
these, after its own bottleneck classification (compute/memory/latency/lds/balanced/overhead), to
enrich that classification with operator-specific structure the generic labels cannot express — e.g.
a distributed MoE kernel's route/expert imbalance and Stage1/Stage2/combine time-share.

Selected by the `analysis_skill` arg (`kernel_workflow.js`). `analysis_skill=none` (the default) or an
unknown/unreadable skill dir disables the step entirely and the run behaves exactly as it did before
this feature existed — see "Degradation" in each skill.

This mirrors `e2e_workflow/knowledge/analysis_skills/` (the `roofline` skill there is the model this
was built from) but is scoped to Kernel Workflow's single-op profiling loop rather than an e2e serving
run. The two directories are independent; a skill added here does not appear in e2e's list and
vice versa.

| skill | dir | what it adds | when to use |
|---|---|---|---|
| `moe_bottleneck` | `moe_bottleneck/` | route/expert-imbalance signal, per-stage (dispatch/GEMM/combine) rank-mean/max/tail-spread decomposition, candidate optimization directions | a distributed MoE kernel (dispatch+GEMM+combine style) with multi-rank profiling data available |
| `none` | — | nothing (pre-feature behavior) | any non-MoE kernel; reproducing an old run byte-for-byte; skill is misbehaving |

## Contract every skill must honour

1. **Advisory only.** A skill may ADD fields, ADD annotations and SUGGEST an ordering. It may never
   prune a candidate, never overwrite the measured `bottleneck` classification or `top_kernels`, and
   never be the sole reason a direction is or isn't taken. The on-box measurement (the isolated A/B
   against the frozen oracle) is always the judge.
2. **Markdown-first.** The skill's logic lives in its `SKILL.md` so an agent can execute it by reading.
   Helper scripts are OPTIONAL mechanical primitives (parsing, unit math) built on
   `kernel_workflow/scripts/multi_rank_analysis/` (the generic, operator-agnostic library). If a helper
   is missing or raises, the agent completes the analysis by hand from `SKILL.md` — a broken script
   must not disable the skill, and a broken skill must not fail the run.
3. **Degrade, never fail.** Every skill defines an explicit degradation ladder ending in "emit nothing
   and let the caller fall back to the pre-skill behavior".
4. **Declare confidence.** Every emitted number carries a confidence level, and the consumer
   (`tech_lead.md`'s `plan_round`) is told what it is allowed to do at each level.

## Adding a skill

Drop a new directory here containing a `SKILL.md` that follows the contract above, add one row to the
table, and pass `analysis_skill=<dirname>`. No orchestration or role change is required — the
`ANALYSIS_SKILL`/`ANALYSIS_SKILL_ON`/`ANALYSIS_SKILL_INPUTS` gating block in `kernel_workflow.js`
resolves any `<dirname>` under this directory generically.
