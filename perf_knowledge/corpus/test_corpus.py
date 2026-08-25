#!/usr/bin/env python3
"""Invariants for the operator corpus.

Two kinds of test here, kept apart on purpose.

The hermetic ones build tiny source trees and assert on what the extractor does with them. They run
anywhere and they are where a rule's behaviour is pinned — a rule tested only against real aiter is
tested against a moving target, and when it breaks you cannot tell whether the rule regressed or the
upstream file changed.

The committed-artifact ones read `evidence/*.yaml` as it stands in the repo and assert the properties
the README promises: language vocabulary shared with `kb.py`, verbatim excerpts, no performance
numbers, gaps stated rather than dropped, and the rendered page matching its inputs. These are the
tests that catch a bad regeneration, which is the failure mode that actually happens — nobody
hand-edits a 17k-line YAML, but plenty of people run the extractor against the wrong tree.

There is no test asserting a particular count. `flydsl == 339` would fail on every aiter bump and
teach the next person to update the number rather than look at why it moved.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
EVIDENCE = os.path.join(HERE, "evidence")
FAMILY = os.path.join(EVIDENCE, "gemm_source.yaml")
TUNED = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
DECISIONS = os.path.join(HERE, "decisions", "gemm.yaml")
SOURCE_DOC = os.path.join(HERE, "gemm_source_evidence.md")
DECISION_DOC = os.path.join(HERE, "gemm_decisions.md")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = _load("_extract_impl_facts", os.path.join(HERE, "_extract_impl_facts.py"))
RE_ = _load("_render_facts", os.path.join(HERE, "_render_facts.py"))
RD = _load("_render_decisions", os.path.join(HERE, "_render_decisions.py"))


@pytest.fixture(scope="module")
def source():
    if not os.path.exists(FAMILY):
        pytest.skip(f"{FAMILY} not generated")
    return RE_.load_yaml(FAMILY)


@pytest.fixture(scope="module")
def tuned():
    if not os.path.exists(TUNED):
        pytest.skip(f"{TUNED} not generated")
    return RE_.load_yaml(TUNED)


@pytest.fixture(scope="module")
def decisions():
    if not os.path.exists(DECISIONS):
        pytest.skip(f"{DECISIONS} not written")
    return RE_.load_yaml(DECISIONS)


def scan(tmp_path, name, body, language="flydsl"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    got, _, _ = EX.scan_file(str(p), name, language)
    return got


def cats(records):
    return {r["category"] for r in records}


# --------------------------------------------------------------------------------------------
# Rules, against trees we control
# --------------------------------------------------------------------------------------------
def test_a_commented_out_intrinsic_is_not_a_fact(tmp_path):
    """The failure `detect_language.py` hit, guarded here too: a mention is not a use."""
    got = scan(tmp_path, "k.py", """
        # rocdl.mfma_f32_16x16x16f16(a, b, c)   <- tried this, was slower
        x = 1
        """)
    assert not got, f"matched inside a comment: {got}"


def test_a_real_intrinsic_is_a_fact_with_its_line(tmp_path):
    got = scan(tmp_path, "k.py", """
        x = 1
        acc = rocdl.mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0)
        """)
    hits = [g for g in got if g["category"] == "mfma_intrinsic"]
    assert len(hits) == 1
    assert hits[0]["line"] == 3, "line number must survive comment stripping"
    assert hits[0]["captured"] == ["mfma_f32_16x16x32_bf16"]


def test_c_style_block_comments_are_stripped_without_shifting_lines(tmp_path):
    got = scan(tmp_path, "k.cu", """
        /* __shared__ float buf[256];
           more commentary */
        __shared__ float real[256];
        """, language="hip")
    hits = [g for g in got if g["category"] == "lds_alloc"]
    assert len(hits) == 1, "the commented allocation must not count"
    assert hits[0]["line"] == 4


def test_arch_scope_is_recorded_when_a_decision_sits_in_a_gfx_branch(tmp_path):
    got = scan(tmp_path, "k.py", """
        if arch == "gfx950":
            BLOCK_M = 256
        """)
    hits = [g for g in got if g["category"] == "tile_shape"]
    assert hits, "expected a tile fact"
    assert "gfx950" in (hits[0]["arch_scope"] or ""), hits[0]


def test_arch_scope_is_empty_for_an_unconditional_decision(tmp_path):
    got = scan(tmp_path, "k.py", "BLOCK_M = 128\n")
    hits = [g for g in got if g["category"] == "tile_shape"]
    assert hits and not hits[0]["arch_scope"]


def test_a_scalar_arch_scope_renders_as_one_label_not_six_characters():
    assert RE_.arch_scopes({"arch_scope": "gfx950"}) == ["gfx950"]
    page = "\n".join(RE_.question_section("tile_shape", [{
        "category": "tile_shape", "language": "flydsl", "file": "x.py", "line": 1,
        "captured": ["128"], "match": "BLOCK_M = 128", "arch_scope": "gfx950",
    }]))
    assert "[gfx950]" in page
    assert "[2, 4, 9, f, g, x]" not in page


def test_a_tunable_parameter_is_recorded_separately_from_a_tile_value(tmp_path):
    """The distinction the coverage table depends on: a knob's existence is not its value."""
    got = scan(tmp_path, "k.py", """
        def kernel(
            BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr,
        ):
            pass
        """, language="triton")
    assert cats(got) == {"tunable_param"}
    assert {g["captured"][0] for g in got} == {"BLOCK_SIZE_M", "BLOCK_SIZE_N"}


def test_the_asm_launcher_contract_is_captured(tmp_path):
    """What a hand-written assembly kernel publishes is its call shape, not its source."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        struct __attribute__((packed)) KernelArgs
        {
            void* ptr_c;
        };
        int blockSizeX = 256;
        gdx = (Ndim / SUBN) * blockSizeX;
        """, language="asm")
    assert {"asm_argblock", "asm_launch"} <= cats(got)


def test_a_grid_formula_inside_a_format_string_is_not_a_launch_fact(tmp_path):
    """`printf("gdx=%d, gdy=%d")` is a log line, not a grid. Found in the rendered page, where a
    fragment of a format string sat in the asm launch table looking like geometry."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        printf("gdx=%d, gdy=%d, gdz=%d\\n", gdx, gdy, gdz);
        """, language="asm")
    assert not [g for g in got if g["category"] == "asm_launch"], got


def test_a_zero_initialiser_is_not_a_grid_formula(tmp_path):
    """`int gdx = 0;` states no decision. Three of those per launcher crowded out the real formula."""
    got = scan(tmp_path, "asm_gemm_x.cu", """
        int gdx = 0;
        gdx = (Ndim / SUBN) * blockSizeX;
        """, language="asm")
    hits = [g for g in got if g["category"] == "asm_launch"]
    assert len(hits) == 1, hits
    assert hits[0]["captured"] == ["(Ndim / SUBN) * blockSizeX"]


def test_ck_records_the_named_pipeline_choice_and_not_the_positional_soup(tmp_path):
    got = scan(tmp_path, "g.cuh", """
        using Instance = ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3
            <Row, Row, 256, 256, 128, 64, 16, 4, 32, 32, 4, 2,
             ck::BlockGemmPipelineScheduler::Interwave,
             ck::BlockGemmPipelineVersion::v1>;
        """, language="ck")
    assert "ck_instance" in cats(got)
    assert {g["captured"][0] for g in got if g["category"] == "ck_pipeline"} == {"Interwave", "v1"}
    # The 40-position argument list must not have been mined for tile evidence: a number at an unnamed
    # position is not a tile until somebody has counted the positions, and that is not this tool.
    assert "tile_shape" not in cats(got)


def test_evidence_is_verbatim_source_not_a_summary(tmp_path):
    body = "acc = rocdl.mfma_f32_16x16x32_bf16(a, b, acc, 0, 0, 0)\n"
    got = scan(tmp_path, "k.py", body)
    ev = "\n".join(got[0]["evidence"])
    assert "rocdl.mfma_f32_16x16x32_bf16" in ev
    assert re.match(r"^\d+: ", got[0]["evidence"][0]), "excerpt lines carry their line number"


def test_a_missing_subtree_is_reported_not_skipped(tmp_path):
    _, _, _, seen, missing = EX.collect(str(tmp_path))
    assert not seen
    langs = {m["language"] for m in missing}
    assert {l for l, _, _ in EX.IMPLS} <= langs, f"every language must account for itself: {missing}"
    assert all(m["why"] for m in missing), "a gap without a reason is not a gap, it is a hole"


# --------------------------------------------------------------------------------------------
# Filename parsing for the shipped sweep results
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,gfx,tags,shape", [
    ("gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json", "gfx942",
     ["GEMM", "A8W8_BLOCKSCALE"], {"n": 1024, "k": 8192}),
    ("gfx1250-FUSED-GEMM-AFP4WFP4-A16W16.json", "gfx1250",
     ["FUSED", "GEMM", "AFP4WFP4", "A16W16"], {}),
    ("gfx1250-FF-A16W16-fused.json", "gfx1250", ["FF", "A16W16", "fused"], {}),
    ("gfx950-GEMM-A16W16-ATOMIC.json", "gfx950", ["GEMM", "A16W16", "ATOMIC"], {}),
])
def test_tuned_names_split_into_gfx_tags_and_shape(name, gfx, tags, shape):
    """A splitter, not a grammar. An earlier regex insisting on kernel-then-variant dropped 239 of
    257 files into an empty-field bucket, and an empty field is indistinguishable from a real one."""
    got = EX.parse_tuned_name(name)
    assert got == {"gfx": gfx, "tags": tags, "shape": shape}


def test_a_name_without_a_gfx_prefix_is_refused_rather_than_guessed():
    assert EX.parse_tuned_name("GEMM-A8W8.json") is None


def test_a_knob_constant_across_shapes_is_separated_from_one_that_moves():
    rows = [
        {"file": "a.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 4}},
        {"file": "b.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 8}},
    ]
    g = EX.summarize_tuned(rows)
    assert len(g) == 1
    assert g[0]["same_across_configs"] == {"BLOCK_SIZE_K": 128}
    assert "num_warps" in g[0]["varies_by_shape"]
    assert g[0]["shape_configs"] == 2
    assert g[0]["source_files"] == ["a.json", "b.json"]


def test_a_single_shipped_shape_config_does_not_claim_agreement():
    """`same_across_configs` over one row would assert an invariant on evidence that cannot show it."""
    g = EX.summarize_tuned([{"file": "a.json", "gfx": "gfx942", "tags": ["GEMM"], "shape": {},
                             "m_bucket": "any", "knobs": {"BLOCK_SIZE_K": 64}}])
    assert g[0]["shape_configs"] == 1
    assert "same_across_configs" not in g[0]
    assert g[0]["knobs"] == {"BLOCK_SIZE_K": 64}


def test_varying_values_render_as_value_colon_count():
    """`.cgx23` reads as a cache modifier named `.cgx23`; the count has to be unambiguous."""
    g = EX.summarize_tuned([
        {"gfx": "g", "tags": [], "shape": {}, "m_bucket": "b", "knobs": {"cache_modifier": ".cg"}},
        {"gfx": "g", "tags": [], "shape": {}, "m_bucket": "b", "knobs": {"cache_modifier": None}},
    ])
    assert ": " in g[0]["varies_by_shape"]["cache_modifier"]


# --------------------------------------------------------------------------------------------
# The YAML round trip
# --------------------------------------------------------------------------------------------
def test_the_fallback_parser_agrees_with_pyyaml_on_the_real_files():
    """Two parse paths that disagree is worse than one that is absent: `--check` would pass or fail
    depending on whether PyYAML happened to be installed."""
    yaml = pytest.importorskip("yaml")
    for path in (FAMILY, TUNED, DECISIONS):
        if not os.path.exists(path):
            pytest.skip(f"{path} not generated")
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f) == RE_._mini_yaml(path), f"parsers diverge on {path}"


def test_a_scalar_containing_a_newline_stays_on_one_line():
    """A CK template head matches across a line break. Emitted raw it put a second physical line
    inside a quoted scalar, which PyYAML folds back but which breaks the one-record-per-line property
    the whole hand-rolled format rests on, and misattributes the diff."""
    assert EX.yaml_quote("DeviceGemm\n<") == '"DeviceGemm\\n<"'
    assert "\n" not in EX.yaml_quote("a\tb\r\nc")


def test_escapes_survive_the_round_trip():
    for original in ('a"b', "a\\b", "a\nb", "a\\nb", "a\tb", 'q"\\n'):
        assert RE_._unquote(EX.yaml_quote(original)) == original, original


# --------------------------------------------------------------------------------------------
# The committed artifacts
# --------------------------------------------------------------------------------------------
def test_every_source_record_uses_the_language_vocabulary_kb_uses(source):
    """A corpus keyed on its own language names cannot be joined to a learned card, which is the
    entire point of putting `language` on cards in the first place."""
    kb = _load("kb", os.path.join(ROOT, "kernel_workflow", "scripts", "kb.py"))
    vocab = set(kb.AUTHORING_LANGUAGES)
    used = {e["language"] for e in source["source_evidence"]}
    assert used <= vocab, f"not in kb.AUTHORING_LANGUAGES: {sorted(used - vocab)}"
    assert {l for l, _, _ in EX.IMPLS} <= vocab, "a declared implementation language must be known"


def test_every_source_record_points_at_a_file_and_a_line(source):
    for record in source["source_evidence"]:
        assert record["file"] and not os.path.isabs(record["file"]), record
        assert isinstance(record["line"], int) and record["line"] > 0, record
        assert record["evidence"], f"source evidence with no verbatim excerpt: {record}"


def test_every_source_record_has_a_unique_content_bound_id(source):
    records = source["source_evidence"]
    ids = [record.get("evidence_id") for record in records]
    assert all(re.fullmatch(r"src_[0-9a-f]{16}", str(value)) for value in ids)
    assert len(ids) == len(set(ids))
    assert all(record["evidence_id"] == EX.source_evidence_id(record) for record in records)


def test_no_source_record_carries_a_performance_number(source):
    """A timing in source evidence is a measurement with no machine
    attached, which is the shape of an unfalsifiable claim."""
    banned = re.compile(r"\b\d+(?:\.\d+)?\s*(?:us|ms|ns|µs|TFLOPS|tflops|GB/s|gbps)\b|"
                        r"\b(?:speedup|faster|slower|x\s*speedup)\b", re.IGNORECASE)
    for record in source["source_evidence"]:
        blob = f"{record.get('match', '')} {' '.join(record.get('evidence') or [])}"
        assert not banned.search(blob), (
            f"performance language in source evidence: {record['file']}:{record['line']}"
        )


def test_provenance_names_the_commit_the_source_was_read_at(source):
    prov = source["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", str(prov["aiter_commit"])), prov
    assert prov["aiter_commit"] != "unknown", "source evidence without a commit has no expiry date"


def test_the_committed_corpus_resolves_at_the_commit_it_names(source):
    """The corpus's one promise, checked rather than assumed.

    An earlier version asserted only that a commit was PRESENT, and passed while 56 records pointed
    into a locally-modified kernel that the named commit does not contain — `file:line` resolved for
    the person who ran the extractor and nobody else. Present and identifying are different
    properties, and only the second one is worth anything in a citation index.
    """
    dirt = source["provenance"].get("aiter_dirty_sources")
    assert dirt == [] or dirt is None or not dirt, (
        f"extracted from a tree with uncommitted changes in {dirt}; re-extract from a clean "
        f"checkout (`git -C <aiter> worktree add /tmp/aiter-clean <commit>`)")
    assert not [e for e in source["source_evidence"] if e.get("unreproducible")], \
        "the committed corpus must not contain records flagged unreproducible"


def test_extraction_refuses_a_dirty_source_by_default(tmp_path, monkeypatch):
    """The refusal, not just the label. A marker on an artifact nobody re-reads is a footnote."""
    calls = {}

    def fake_dirty(path):
        calls["asked"] = path
        return {"aiter/ops/flydsl/kernels/splitk_hgemm.py"}

    monkeypatch.setattr(EX, "dirty_paths", fake_dirty)
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "aiter/ops/flydsl/kernels/splitk_hgemm.py", "line": 1, "language": "flydsl",
          "category": "mfma_intrinsic", "match": "x", "captured": [], "arch_scope": "",
          "evidence": ["1: x"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--emit"])
    assert EX.main() == 3, "a dirty contributing file must abort the emit"
    assert calls["asked"] == str(tmp_path)


def test_allow_dirty_records_but_marks_every_affected_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(EX, "dirty_paths", lambda a: {"dirty.py"})
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "dirty.py", "line": 1, "language": "flydsl", "category": "tile_shape",
          "match": "x", "captured": [], "arch_scope": "", "evidence": ["1: x"]},
         {"file": "clean.py", "line": 2, "language": "flydsl", "category": "tile_shape",
          "match": "y", "captured": [], "arch_scope": "", "evidence": ["2: y"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--json", "--allow-dirty"])
    assert EX.main() == 0
    payload = json.loads(capsys.readouterr().out)
    got = {e["file"]: e.get("unreproducible")
           for e in payload["evidence"]["source_evidence"]}
    assert got == {"dirty.py": True, "clean.py": None}, got
    assert payload["evidence"]["provenance"]["aiter_dirty_sources"] == ["dirty.py"]


def test_an_unrelated_dirty_file_does_not_block_extraction(tmp_path, monkeypatch, capsys):
    """Scoped to contributors. Refusing on any edit anywhere in aiter would be superstition — an
    unrelated change cannot make a citation unresolvable."""
    monkeypatch.setattr(EX, "dirty_paths", lambda a: {"docs/README.md", "aiter/unrelated.py"})
    monkeypatch.setattr(EX, "collect", lambda a: (
        [{"file": "clean.py", "line": 1, "language": "flydsl", "category": "tile_shape",
          "match": "x", "captured": [], "arch_scope": "", "evidence": ["1: x"]}], [], [], [], []))
    monkeypatch.setattr(EX, "aiter_commit", lambda a: "0" * 40)
    monkeypatch.setattr(sys, "argv", ["x", "--aiter", str(tmp_path), "--json"])
    assert EX.main() == 0
    assert json.loads(capsys.readouterr().out)["evidence"]["provenance"]["aiter_dirty_sources"] == []


def test_dirty_paths_parses_renames_and_quoted_names():
    """`XY old -> new`: the new name is what was read. A parser that kept the old one would clear a
    file that is in fact modified."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent("""
        import subprocess, importlib.util, sys
        spec = importlib.util.spec_from_file_location("x", sys.argv[1])
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        class R:
            stdout = ' M a/b.py\\nR  old/x.py -> new/y.py\\n?? "sp ace.py"\\n'
        subprocess.run = lambda *a, **k: R()
        print(sorted(m.dirty_paths("/nowhere")))
        """), os.path.join(HERE, "_extract_impl_facts.py")],
        capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "['a/b.py', 'new/y.py', 'sp ace.py']", out.stdout


def test_gaps_are_stated_with_a_reason(source):
    for m in source.get("missing") or []:
        assert m.get("language") and m.get("subtree") and m.get("why"), m


def test_every_language_declared_actually_produced_source_evidence(source):
    """A language present in `IMPLS` but absent from the output means the globs no longer match the
    upstream layout — which is how `ck` sat at zero while looking like a deliberate omission."""
    declared = {l for l, _, _ in EX.IMPLS}
    got = {e["language"] for e in source["source_evidence"]}
    assert declared == got, f"declared but empty: {sorted(declared - got)}"


def test_the_rendered_source_page_matches_the_evidence():
    if not os.path.exists(SOURCE_DOC):
        pytest.skip("doc not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py"), "--check"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, f"stale doc; re-run --emit\n{r.stdout}{r.stderr}"


def test_the_source_page_states_no_ranking(source):
    """The prose is hand-written, so it is the one place a claim could slip in. Checked as text."""
    if not os.path.exists(SOURCE_DOC):
        pytest.skip("doc not generated")
    with open(SOURCE_DOC, encoding="utf-8") as f:
        body = f.read()
    for phrase in ("is faster", "is better", "outperforms", "you should use", "the best "):
        assert phrase not in body.lower(), f"the page ranks: {phrase!r}"


def test_curated_decision_cards_are_actionable_and_grounded(decisions, source):
    cards = decisions.get("cards") or []
    assert cards, "an evidence index without decision cards still leaves the author to infer the answer"
    assert RD.validate(cards, source["source_evidence"]) == []
    for card in cards:
        assert card["conditions"], card
        assert card["actions"], card
        assert card["why"], card
        assert card["limitations"], card


def test_always_on_corpus_does_not_bypass_measured_knowledge_switches(decisions):
    levels = {card["evidence_level"] for card in decisions["cards"]}
    assert levels == {"source_observed"}
    for card in decisions["cards"]:
        assert card["source_evidence"] and not card["measurement_evidence"], card


def test_a_loose_file_line_reference_is_rejected(decisions, source):
    card = dict(decisions["cards"][0], source_evidence=["x.py:1"])
    errors = RD.validate([card], source["source_evidence"])
    assert any("content-bound" in error for error in errors)


def test_the_rendered_decision_page_matches_its_inputs():
    if not os.path.exists(DECISION_DOC):
        pytest.skip("decision doc not generated")
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "_render_decisions.py"), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"stale decision doc; re-run _render_decisions.py --emit\n{result.stdout}{result.stderr}"
    )


def test_shipped_configs_become_seed_candidate_and_vary_next_advice(decisions, source, tuned):
    body = RD.render(decisions, source, tuned)
    assert "## Shipped configuration seeds" in body
    assert "seed candidate" in body and "vary next" in body
    assert "not measured winners" in body


def test_tuned_groups_carry_arch_and_bucket(tuned):
    """A knob set without its gfx and its M bucket reads as a global recommendation. It is not one."""
    ids = []
    for g in tuned["tuned_configs"]:
        assert re.fullmatch(r"cfg_[0-9a-f]{16}", str(g["config_id"])), g
        ids.append(g["config_id"])
        assert str(g["gfx"]).startswith("gfx"), g
        assert g["m_bucket"], g
        assert int(g["shape_configs"]) >= 1, g
        assert g["source_files"], g
        assert ("same_across_configs" in g) != ("knobs" in g), \
            f"exactly one of the two forms, chosen by shape_configs: {g}"
    assert len(ids) == len(set(ids)), "config IDs must identify exactly one (gfx, variant, M bucket)"


def test_the_extractor_refuses_to_run_without_a_source_tree():
    """`--emit` with no `--aiter` would write an empty corpus over a good one."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "_extract_impl_facts.py"), "--emit"],
                       capture_output=True, text=True, check=False)
    assert r.returncode != 0
    assert "--aiter" in (r.stderr + r.stdout)


def test_the_extractor_requires_an_explicit_output_mode(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(HERE, "_extract_impl_facts.py"), "--aiter", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "--emit" in (result.stderr + result.stdout)
    assert "--json" in (result.stderr + result.stdout)


def test_the_renderer_runs_without_an_aiter_checkout():
    """The property that makes `--check` a usable gate: rendering reads only committed YAML."""
    if not os.path.exists(FAMILY):
        pytest.skip("source evidence not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py")],
                       capture_output=True, text=True, cwd="/", check=False)
    assert r.returncode == 0, r.stderr
    assert "## Source-evidence coverage" in r.stdout


def test_json_mode_is_machine_readable():
    if not os.path.exists(FAMILY):
        pytest.skip("source evidence not generated")
    aiter = os.environ.get("AITER_PATH", "/sgl-workspace/aiter")
    if not os.path.isdir(aiter):
        pytest.skip("no aiter checkout")
    # `--allow-dirty` because this asserts the OUTPUT PARSES, and whether the source tree is clean has
    # nothing to do with that. Without it the test asserts reproducibility twice and fails on any
    # developer box with a modified aiter, which teaches people to skip it.
    r = subprocess.run([sys.executable, os.path.join(HERE, "_extract_impl_facts.py"),
                        "--aiter", aiter, "--json", "--allow-dirty"],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["evidence"]["source_evidence"] and payload["tuned"]["tuned_configs"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
