# corpus/ — the same operator, written six ways

A FlyDSL author writing a GEMM has to answer, by hand, roughly twenty questions that Triton answers
for them: which MFMA instruction, what tile, how many waves, how LDS is laid out and swizzled, where
to pin instruction order. AITER already contains the same operator family answered six ways — in
FlyDSL, Triton, Gluon, CK, HIP and hand-written assembly — and those answers were, mostly, measured.
This directory indexes them so that the first move on a new kernel is a lookup rather than a guess.

## The one rule

**Facts here, claims elsewhere.**

A fact is something readable from source at a named commit: *this file, at this line, selects
`mfma_f32_16x16x32_bf16`.* A claim is something only a measurement can support: *that instruction is
the right one for this shape.* This directory holds only the first kind.

The split is not fastidiousness, it is what makes the directory maintainable. Facts can be
regenerated, and a regenerated fact that has gone stale shows up as a diff. Claims cannot be
regenerated — they are tied to the run, the machine and the version that produced them — so a claim
that drifts out of date looks exactly like one that is still true. Claims belong in
`perf_knowledge/expert_skills/`, attached to the validation artifact that earned them, and aggregate
priors belong in `perf_knowledge/index/run_recurrence.md`, which counts how often an axis paid across
distinct kernels without restating what any card concluded.

Concretely, a line in `gemm_family.md` may say "13 places state a cache policy, here they are". It
may not say "use `.cg`". If you catch yourself wanting to write the second, you want an expert skill.

## Layout

| path | what it is |
|---|---|
| `_extract_impl_facts.py` | reads an aiter checkout, writes `facts/*.yaml`. Needs `--aiter`. |
| `_render_facts.py` | reads `facts/*.yaml`, writes `gemm_family.md`. Needs nothing else. |
| `facts/gemm_family.yaml` | one record per rule hit: category, language, `file:line`, match, arch scope, verbatim excerpt. |
| `facts/gemm_tuned_configs.yaml` | AITER's shipped Triton sweep results, grouped by `(gfx, variant, M bucket)`. |
| `gemm_family.md` | the rendered page. Generated; CI fails if it disagrees with the facts. |
| `test_corpus.py` | the invariants below, enforced. |

The two-script split matters: extraction needs a source tree that is not part of this repo, and
rendering must not. That is what lets `--check` run in CI on a box with no aiter and no GPU, and a
generated doc that nothing checks is how `kernel_families.md` came to state a default tile that no
longer existed.

## How to refresh

```bash
python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /path/to/aiter --emit
python3 perf_knowledge/corpus/_render_facts.py --emit
```

Commit the YAML and the `.md` together. They are a pair; a commit containing one without the other is
the stale-doc failure this directory is built to prevent.

**Extraction refuses a dirty source.** If any file that produces facts has uncommitted changes, the
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
scoped to files that actually *contributed* facts — an unrelated edit elsewhere in aiter cannot make a
citation unresolvable, and refusing on it would be superstition rather than a check.

Provenance records `aiter_origin` (the remote) rather than a local path, since the pair that
identifies a source is repository plus commit. A path like `/tmp/aiter-clean` names a directory that
no longer exists and never meant anything to anyone else.

## Reading the coverage table without drawing the wrong conclusion

The table counts **where a decision is stated in source**, per axis per language. The zeros mislead
if read as absences of the decision itself:

- `scheduling` is 132 in FlyDSL and empty everywhere else. Triton kernels *are* scheduled — by a
  compiler pass. CK picks a `BlockGemmPipelineScheduler` enum. Neither writes a per-instruction
  ordering statement, so neither appears. The real content of that row is "FlyDSL is the language
  where this becomes your problem", which is worth knowing and is not the same sentence.
- `tile_shape` is nearly empty for Triton. Its tiles are `tl.constexpr` parameters, so the source
  states *that there is a knob* (`tunable_param`, 459 of them) while the values live in the shipped
  JSON. Two different facts, in two different places, both recorded.
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

1. Every fact carries a language from the same vocabulary `kernel_workflow/scripts/kb.py` uses, so a
   corpus record and a learned card can be filtered by the same term.
2. Every fact carries `file` and a positive integer `line`, and the excerpt is verbatim — no
   paraphrase, because a paraphrase is a new artifact that can be wrong, while a quote can only be
   stale, and staleness is detectable from the commit.
3. No fact record contains a performance number. Timings in a fact file would be measurements without
   a machine, which is the shape of an unfalsifiable claim.
4. Missing subtrees are reported in `missing`, never silently dropped. A section that is empty because
   nothing was found and one that is empty because the search never ran look identical otherwise, and
   the difference decides whether the next person repeats the search.
5. `gemm_family.md` matches the facts byte for byte.
6. The committed corpus resolves at the commit it names: `aiter_dirty_sources` is empty and no record
   is flagged `unreproducible`. This is the promise in the first paragraph, checked rather than
   assumed — "a commit is recorded" and "the commit identifies what was read" are different
   properties, and only the second one is worth anything here.
