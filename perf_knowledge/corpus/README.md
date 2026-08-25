# corpus/ — source evidence plus actionable development decisions

A FlyDSL author writing a GEMM has to answer, by hand, roughly twenty questions that Triton answers
for them: which MFMA instruction, what tile, how many waves, how LDS is laid out and swizzled, where
to pin instruction order. AITER already contains the same operator family answered six ways — in
FlyDSL, Triton, Gluon, CK, HIP and hand-written assembly. Source presence alone says nothing about
whether an answer was measured or won.
This directory turns them into two different products so the first move on a new kernel is a lookup
rather than a guess:

1. **Source evidence** — what an implementation literally contains, with a content-bound ID and
   reproducible `file:line`.
2. **Development decision cards** — when a pattern applies, what to try, why, alternatives, evidence
   strength and limits.

## The one rule

**Never make the reader infer a recommendation from a source-match count.**

Source evidence can establish: *this file, at this line, selects
`mfma_f32_16x16x32_bf16`.* It cannot establish: *that instruction is right for this shape.* The old
page stopped at the first sentence and called each regex category a "decision axis", leaving the
author to reverse-engineer the actual advice from a raw hit dump. That is an evidence index, not knowledge.

`gemm_decisions.md` supplies the missing semantic layer. Every card has conditions, actions,
alternatives, rationale, evidence strength and limitations:

- `source_observed` means "include this implementation precedent as a candidate", never "prefer it";
- `shipped_config` means "AITER selected this parameter seed", not "the corpus has its benchmark";
- measured guidance stays in learned cards/expert skills behind their existing feature switches.

Measured claims remain in learned cards or `perf_knowledge/expert_skills/`, attached to the environment
and run that earned them. The always-on corpus does not copy them and bypass those systems' switches.

## Layout

| path | what it is |
|---|---|
| `_extract_impl_facts.py` | reads an AITER checkout and writes raw source/tuning evidence. Needs `--aiter`. |
| `evidence/gemm_source.yaml` | machine-readable source observations: stable ID, question category, language, `file:line`, match, arch scope, verbatim excerpt. |
| `evidence/gemm_tuned_configs.yaml` | AITER's shipped selected Triton configs, each with stable `cfg_…` attribution ID, grouped by `(gfx, variant, M bucket)` with exact source JSON paths. |
| `gemm_source_evidence.md` | generated evidence index for tracing a card back to source; not the first page an author should read. |
| `decisions/gemm.yaml` | curated development cards with conditions, actions, alternatives and evidence strength. |
| `_render_decisions.py` | validates card citations and combines curated cards with shipped tuning evidence. |
| `gemm_decisions.md` | generated, actionable page the Workflow reads first. |
| `test_corpus.py` | source reproducibility, card grounding and generated-page contracts. |

The extraction/rendering split matters: extraction needs a source tree that is not part of this repo,
while both renderers must not. That is what lets `--check` run in CI on a box with no aiter and no GPU, and a
generated doc that nothing checks is how `kernel_families.md` came to state a default tile that no
longer existed.

## How to refresh

```bash
python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /path/to/aiter --emit
python3 perf_knowledge/corpus/_render_facts.py --emit
python3 perf_knowledge/corpus/_render_decisions.py --emit
```

Commit the evidence, cards and both generated pages together. CI rejects either page when it does not
match its inputs, and rejects a decision card whose content-bound evidence ID is absent or stale.

**Extraction refuses a dirty source.** If any file that produces evidence has uncommitted changes, the
emit aborts, because its `file:line` would resolve for you and for nobody else — and a citation index
whose citations do not resolve is not a weaker index, it is a different document that looks like one.
This is not hypothetical: the first version of this corpus shipped 56 records pointing into a locally
modified kernel, under a commit that did not contain those edits, with the whole suite green. The test
asserted a commit was *present*; it never asked whether the commit *identified* what was read.

The fix when your working copy is dirty is a throwaway clean checkout, which leaves your edits alone:

```bash
git -C /path/to/aiter worktree add /tmp/aiter-clean <commit>
python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /tmp/aiter-clean --emit
```

`--allow-dirty` extracts anyway and marks every affected record `unreproducible: true`, which is
useful for a local look and is rejected by `test_corpus.py` for the committed artifact. The refusal is
scoped to files that actually *contributed* evidence — an unrelated edit elsewhere in aiter cannot make a
citation unresolvable, and refusing on it would be superstition rather than a check.

Provenance records `aiter_origin` (the remote) rather than a local path, since the pair that
identifies a source is repository plus commit. A path like `/tmp/aiter-clean` names a directory that
no longer exists and never meant anything to anyone else.

## Reading the source-evidence table without drawing the wrong conclusion

The table counts **where an implementation question is answered in source**, per question per
language. The zeros mislead if read as absences of the choice itself:

- `scheduling` is 132 in FlyDSL and empty everywhere else. Triton kernels *are* scheduled — by a
  compiler pass. CK picks a `BlockGemmPipelineScheduler` enum. Neither writes a per-instruction
  ordering statement, so neither appears. The real content of that row is "FlyDSL is the language
  where this becomes your problem", which is worth knowing and is not the same sentence.
- `tile_shape` is nearly empty for Triton. Its tiles are `tl.constexpr` parameters, so the source
  states *that there is a knob* (`tunable_param`, 459 of them) while the values live in the shipped
  JSON. Two different source observations, in two different places, both recorded.
- An empty cell can also just mean no rule matches that idiom yet. The rules are a list at the top of
  `_extract_impl_facts.py`. A missing idiom is a fixable gap, not a finding about the language.

## Why regex, and where that runs out

Six languages is six grammars, and a Python AST pass cannot read a `.cu`. For a citation index what
matters is that a hit points at the right *line* — the reader opens the file. Where structure is
genuinely required there is an AST pass, applied only to the `.py` files where it is valid.

Comments are stripped before matching. This is not hypothetical: `detect_language.py` learned it by
classifying five plainly-HIP files as CK on the strength of `ck_tile::` mentions that appeared only
inside commented-out code.

Two things the regex approach deliberately does not attempt:

- **CK's positional template arguments.** Forty-odd unnamed positions are exactly what a regex gets
  subtly wrong. What is recorded is the instance template chosen and the two arguments CK names —
  loop scheduler and pipeline version — because those have an analogue a FlyDSL author must decide.
- **Instruction-level detail inside the `.co` binaries.** The launchers (`asm_gemm_*.cu`) publish the
  assembly's contract: object name, workgroup size, grid formula, argument block layout. The
  instructions themselves need a disassembly pass, which is separate work.

## Invariants `test_corpus.py` enforces

1. Every source record carries a language from the same vocabulary `kernel_workflow/scripts/kb.py` uses, so a
   corpus record and a learned card can be filtered by the same term.
2. Every source record carries `file` and a positive integer `line`, and the excerpt is verbatim — no
   paraphrase, because a paraphrase is a new artifact that can be wrong, while a quote can only be
   stale, and staleness is detectable from the commit.
3. No source-evidence record contains a performance number. Timings there would be measurements without
   a machine, which is the shape of an unfalsifiable claim.
4. Missing subtrees are reported in `missing`, never silently dropped. A section that is empty because
   nothing was found and one that is empty because the search never ran look identical otherwise, and
   the difference decides whether the next person repeats the search.
5. `gemm_source_evidence.md` matches the evidence, and `gemm_decisions.md` matches cards plus evidence.
6. The committed corpus resolves at the commit it names: `aiter_dirty_sources` is empty and no record
   is flagged `unreproducible`. This is the promise in the first paragraph, checked rather than
   assumed — "a commit is recorded" and "the commit identifies what was read" are different
   properties, and only the second one is worth anything here.
7. Every decision card states conditions, actions, rationale and limitations; source IDs must resolve
   to unchanged evidence, and measured guidance is rejected from this always-on layer.
