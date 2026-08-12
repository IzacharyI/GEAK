# SKILL — MoE dispatch/GEMM/combine bottleneck advisory

Turn a distributed MoE kernel's generic bottleneck label (`compute`/`memory`/`latency`/`lds`/
`balanced`/`overhead`, from `profile_engineer` step 4) into an answer to: **which stage is actually
slow, is the slowness data-dependent (routing) or structural (kernel config), and what class of fix
follows?**

You (an agent) can execute this skill by reading this file. It is built on
`kernel_workflow/scripts/multi_rank_analysis/` (the generic, operator-agnostic library) — use its
functions if they import; do the arithmetic by hand from the formulas below if they don't. **A broken
helper must not disable this skill; a broken skill must not fail the run.**

---

## 0. The one doctrine that matters

> This skill explains **why** a multi-rank MoE kernel classified `latency`- or `overhead`-bound is
> slow, in terms the generic classification cannot express (which stage, whether it's routing-data-
> dependent). It **never** re-labels `bottleneck`, **never** overrides a `top_kernels` entry, and
> **never** prunes a `directions[]` candidate the TechLead was otherwise going to propose.

Corollary: this skill is **advisory**. `bottleneck` and `top_opportunities` from `profile_engineer`
remain the primary signal; this skill only ever adds a parallel, more specific hypothesis. See §8.

---

## 1. Inputs

- `EVAL_DIR/baseline_metrics.json` (or `round_N_metrics.json`) — `profile_engineer`'s step-4
  classification output (required: `bottleneck`, `key_metrics`, `top_kernels`).
- A **multi-rank trace directory** — one Chrome/Perfetto trace JSON per rank
  (`torch.profiler.export_chrome_trace` output), produced by the same distributed benchmark the
  COMMANDMENT's PROFILE command runs. Path comes from the COMMANDMENT or is passed explicitly by the
  caller (required for §3 steps 1–2; without it, degrade per §6 L3).
- A **category map** — `{category_name: regex}` describing how this MoE kernel's raw kernel names
  split into logical stages. This skill ships `default_moe_category_map.json` as a **worked example**
  (matching a dispatch→GEMM1→GEMM2→P2P-combine MoE kernel family: `stage1`/`stage2`/`combine`/
  `quantize`); if the kernel under analysis uses different stage names, the caller/TechLead supplies
  its own map instead (optional; falls back to the shipped default, confidence lowered — §6 L1).
- **Route/expert load summary**, if the benchmark prints one (e.g. `[ROUTES]`/`[EXPERTS]`-style
  per-destination-rank counts and per-expert max/mean route counts) — optional; enables §4's
  route-imbalance correlation (absent → skip that section only, §6 L4).

---

## 2. Output artifact

Write `EVAL_DIR/profile_moe_advisory.json` (or `round_N_moe_advisory.json` for reprofile) and a
human-readable `.md` beside it. Report its path as `moe_advisory_json` in `profile_engineer`'s return.

```jsonc
{
  "skill": "moe_bottleneck", "skill_version": "1",
  "category_map_source": "caller|default",
  "categories": {
    "<cat>": {
      "rank_mean_ms": 0.0, "rank_max_ms": 0.0, "rank_min_ms": 0.0,
      "rank_tail_spread_pct": 0.0,           // (max-min)/min*100 — the "is this stage skewed" signal
      "share_of_classified_pct": 0.0,        // this category's share of total classified time
      "kernel_names": ["..."],
      "missing_ranks": []                     // ranks whose trace lacked this category (degraded, not dropped)
    }
  },
  "unclassified_share_pct": 0.0,              // time in kernels no category matched — never silently hidden
  "route_imbalance": {                        // OMITTED entirely if route/expert data unavailable (§6 L4)
    "active_experts": 0, "expert_max_to_mean": 0.0,
    "correlates_with_tail_category": "combine|stage1|stage2|none"
  },
  "directions": [{
    "hypothesis": "...",                      // one sentence, tied to a specific number above
    "specialty": "algorithm|memory|compute|host_runtime",  // matches tech_lead.md's direction taxonomy
    "confidence": "low|medium|high",
    "data_dependent": false,                  // true = routing-driven; a static kernel-config sweep will not fix it
    "evidence": "the exact category/metric this hypothesis is derived from"
  }],
  "degraded": [{"reason": "..."}],
  "skill_errors": []
}
```

---

## 3. Procedure

0. Read `EVAL_DIR/baseline_metrics.json` for `bottleneck`/`key_metrics`/`top_kernels`. If `bottleneck`
   is `compute`- or `memory`-bound with high utilization on that axis, this skill still runs (a
   compute-bound GEMM stage can coexist with a latency-bound combine stage) but should note in
   `directions[].evidence` that the generic classification already points at a structural fix for that
   axis — do not manufacture a competing hypothesis for the same evidence.
1. For each rank's trace file, call `multi_rank_analysis.trace_categories.bucket_trace_events(path,
   category_map)`. A trace file that fails to load returns `{"error": ...}` (§6 L2) — exclude that
   rank from every category's aggregate and record it, do not abort.
2. Merge across ranks with `multi_rank_analysis.aggregate.merge_rank_records(records,
   ["per_category_ms.<cat>" for each cat in category_map])` to get `rank_mean_ms`/`rank_max_ms`/
   `rank_min_ms`/`rank_tail_spread_pct` per category. Compute `share_of_classified_pct` as
   `rank_mean_ms / sum(rank_mean_ms over all categories) * 100` — **never** against total wall time,
   since profiler instrumentation overhead inflates the denominator (this mirrors the existing
   evidence's explicit note: "Chrome-trace category durations include profiler overhead and are used
   only for within-profile decomposition, never as the speedup denominator").
3. If route/expert data is available, compute `expert_max_to_mean` (max per-expert route count /
   mean per-expert route count across all experts) and check whether the category with the highest
   `rank_tail_spread_pct` also has a rank ordering that correlates with the per-destination-rank route
   counts (the rank receiving the most tokens is also the rank with the highest time in that
   category). This is the signal, not a guess: it is the exact mechanism recorded as evidence for one
   real MoE kernel (§4 worked pattern below) — skewed routing (`expert_max_to_mean` ~48x) made a
   specific stage's rank-max blow out ~98% while GEMM stages moved only ~10–17%.
4. Apply §4's routing rules to produce `directions[]`. Every hypothesis MUST cite the specific
   category/metric it came from (`evidence` field) — no direction may be justified by "MoE kernels are
   generally slow at X".
5. Sanity-check: `unclassified_share_pct` should be small. If it is large (e.g. >30%), the category
   map is a poor fit for this kernel — lower confidence on every entry and say so, rather than silently
   presenting confident numbers derived mostly from unclassified time.

---

## 4. Domain model — routing rules

These are the general principles this skill applies; they are illustrated with (not limited to) the
worked pattern from one real distributed-MoE dispatch/GEMM/combine kernel's measured evidence.

- **High tail-spread in a "publish to peers" / combine-style category, correlated with route skew**
  → hypothesis: "this stage's cost is destination-dependent — a rank that is a popular routing target
  receives disproportionate P2P/scatter writes and serializes on them." →
  `specialty: "algorithm"`, `data_dependent: true`. Candidate fix class: rebalance the stage's grid/CU
  assignment to the actual per-destination load, or overlap this stage's publication with the
  preceding GEMM stage's completion (pipeline across the two rather than a hard barrier between them).
  Worked evidence: one MoE kernel's combine-stage rank-max delta between uniform and skewed routing
  was ~98% of that route pair's total per-category delta, while its two GEMM stages moved only
  ~10–17% — i.e. **the imbalance concentrates almost entirely in the combine/publish stage**, not the
  GEMMs, even though the GEMMs are the larger absolute time.
- **Low tail-spread in the GEMM-shaped categories but high tail-spread in the combine-shaped
  category** → the bottleneck is **stage-specific to combine**, not a general GEMM tiling problem.
  Steers the TechLead away from another GEMM `block_m`/`block_n`/`num_stages` sweep and toward
  `algorithm`/`host_runtime` directions targeting the combine/publish kernel specifically.
- **Route-imbalance-driven latency is flagged `data_dependent: true`** whenever
  `expert_max_to_mean` is large (order ~10x or more) — this is a hard signal that **static kernel
  launch-geometry knobs cannot fix it**: a knob tuned for one routing distribution will not generalize
  to another, because the imbalance is a property of the input data, not the kernel's code. A round of
  static-knob sweeps (persistent-CU counts, strided-vs-contiguous work assignment, fixed skew-CU
  splits) that each come back near-parity with the default is the expected, predictable outcome of
  trying to fix a data-dependent problem with a data-independent knob — report this as confirming
  evidence for a code-level (route-aware) direction, not as "nothing helped, give up."
- **All categories show low tail-spread (routing looks uniform) but the generic classification is
  still `latency`-bound** → this skill has nothing more specific to add than `profile_engineer`
  already found; return `directions: []` rather than inventing an MoE-flavored story where none is
  supported by the data.

---

## 5. Confidence stages

| stage | source | confidence | consumer may |
|---|---|---|---|
| **A** default category map, no route data | shipped `default_moe_category_map.json`, `route_imbalance` omitted | `low` | display + annotate only |
| **B** caller-supplied category map, no route data | operator-specific map, still no imbalance correlation | `medium` | cite in `directions[].evidence` |
| **C** caller-supplied category map + route/expert data present | full correlation check (§3 step 3) | `high` | rank as a secondary key for `directions[]` ordering |

---

## 6. Degradation ladder — every level is non-fatal

| level | trigger | behavior |
|---|---|---|
| **L0** | `analysis_skill=none`, skill dir missing/unreadable | Emit nothing. Caller behaves exactly as before this feature existed. |
| **L1** | no category map supplied | Fall back to `default_moe_category_map.json`; every entry's confidence capped at stage A. |
| **L2** | one rank's trace file missing/unreadable | Exclude that rank from every category's aggregate (via `merge_rank_records`'s `missing_ranks`); note it in `degraded[]`; other ranks unaffected. |
| **L3** | no trace directory available at all | Skip categories/route sections entirely; return `directions: []` with a note that only `baseline_metrics.json` was available — never fail the profile step. |
| **L4** | route/expert data unavailable | Omit `route_imbalance` entirely (not a null-filled stub); every `directions[].data_dependent` defaults to `false` since imbalance cannot be confirmed. |
| **L5** | anything else raises | Catch it, append to `skill_errors[]`, write whatever categories succeeded, and continue. **The run never fails because of this skill.** |

---

## 7. Routing table — mapping `directions[]` into `plan_round`'s taxonomy

| `directions[].specialty` | maps to `tech_lead.md` direction `specialty` | typical fix shape |
|---|---|---|
| `algorithm` | `algorithm` | rebalance grid/CU assignment to load; pipeline/overlap two stages instead of a hard barrier |
| `host_runtime` | `host_runtime` | reduce launch/dispatch overhead for a many-small-kernel stage |
| `compute`/`memory` | same | only emitted when this skill's category-level evidence sharpens (not replaces) what `profile_engineer` already found on that axis |

No new specialty is invented — every `directions[]` entry must map onto one of the TechLead's existing
four specialties (`algorithm|memory|compute|host_runtime`; `deep_explore` is never proposed by this
skill, that decision is the TechLead's alone).

---

## 8. Guarding against being wrong

1. **The advisory can be wrong.** The isolated A/B against the frozen oracle is the ONLY acceptance
   criterion for any candidate this skill's `directions[]` inspired — matching this repo's
   `roofline` skill's §8.6 "never sole authority" doctrine.
2. **Do not confuse correlation with proof.** §3 step 3's route/tail-spread correlation is a
   hypothesis generator, not a measurement. A `data_dependent: true` flag means "worth checking against
   a route-aware redesign first", not "confirmed root cause."
3. **A rejected static-knob sweep is evidence, not noise.** If the TechLead already tried several
   static launch-geometry knobs for a category this skill flags `data_dependent: true`, and they all
   came back within the acceptance noise band, that is **confirming** evidence for this skill's
   hypothesis (see §4) — record it in `HISTORY`/the insight ledger rather than re-trying more knobs
   from the same family.
4. **No candidate is ever pruned because of this skill.** It only ever adds candidate directions or
   annotates evidence for/against a direction already under consideration.
