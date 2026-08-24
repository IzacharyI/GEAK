#!/usr/bin/env python3
"""Invariants for the operator corpus.

Two kinds of test here, kept apart on purpose.

The hermetic ones build tiny source trees and assert on what the extractor does with them. They run
anywhere and they are where a rule's behaviour is pinned — a rule tested only against real aiter is
tested against a moving target, and when it breaks you cannot tell whether the rule regressed or the
upstream file changed.

The committed-artifact ones read `facts/*.yaml` as it stands in the repo and assert the properties
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
FACTS = os.path.join(HERE, "facts")
FAMILY = os.path.join(FACTS, "gemm_family.yaml")
TUNED = os.path.join(FACTS, "gemm_tuned_configs.yaml")
DOC = os.path.join(HERE, "gemm_family.md")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EX = _load("_extract_impl_facts", os.path.join(HERE, "_extract_impl_facts.py"))
RE_ = _load("_render_facts", os.path.join(HERE, "_render_facts.py"))


@pytest.fixture(scope="module")
def facts():
    if not os.path.exists(FAMILY):
        pytest.skip(f"{FAMILY} not generated")
    return RE_.load_yaml(FAMILY)


@pytest.fixture(scope="module")
def tuned():
    if not os.path.exists(TUNED):
        pytest.skip(f"{TUNED} not generated")
    return RE_.load_yaml(TUNED)


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
    # The 40-position argument list must not have been mined for tile facts: a number at an unnamed
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
        {"gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 4}},
        {"gfx": "gfx942", "tags": ["GEMM"], "shape": {}, "m_bucket": "M_LEQ_32",
         "knobs": {"BLOCK_SIZE_K": 128, "num_warps": 8}},
    ]
    g = EX.summarize_tuned(rows)
    assert len(g) == 1
    assert g[0]["same_across_shapes"] == {"BLOCK_SIZE_K": 128}
    assert "num_warps" in g[0]["varies_by_shape"]
    assert g[0]["shapes_swept"] == 2


def test_a_single_swept_shape_does_not_claim_agreement():
    """`same_across_shapes` over one row would assert an invariant on evidence that cannot show it."""
    g = EX.summarize_tuned([{"gfx": "gfx942", "tags": ["GEMM"], "shape": {},
                             "m_bucket": "any", "knobs": {"BLOCK_SIZE_K": 64}}])
    assert g[0]["shapes_swept"] == 1
    assert "same_across_shapes" not in g[0]
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
    for path in (FAMILY, TUNED):
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
def test_every_fact_uses_the_language_vocabulary_kb_uses(facts):
    """A corpus keyed on its own language names cannot be joined to a learned card, which is the
    entire point of putting `language` on cards in the first place."""
    kb = _load("kb", os.path.join(ROOT, "kernel_workflow", "scripts", "kb.py"))
    vocab = set(kb.AUTHORING_LANGUAGES)
    used = {f["language"] for f in facts["facts"]}
    assert used <= vocab, f"not in kb.AUTHORING_LANGUAGES: {sorted(used - vocab)}"
    assert {l for l, _, _ in EX.IMPLS} <= vocab, "a declared implementation language must be known"


def test_every_fact_points_at_a_file_and_a_line(facts):
    for f in facts["facts"]:
        assert f["file"] and not os.path.isabs(f["file"]), f
        assert isinstance(f["line"], int) and f["line"] > 0, f
        assert f["evidence"], f"a fact with no excerpt is a claim: {f}"


def test_no_fact_carries_a_performance_number(facts):
    """The README's one rule, enforced. A timing in a fact file is a measurement with no machine
    attached, which is the shape of an unfalsifiable claim."""
    banned = re.compile(r"\b\d+(?:\.\d+)?\s*(?:us|ms|ns|µs|TFLOPS|tflops|GB/s|gbps)\b|"
                        r"\b(?:speedup|faster|slower|x\s*speedup)\b", re.IGNORECASE)
    for f in facts["facts"]:
        blob = f"{f.get('match', '')} {' '.join(f.get('evidence') or [])}"
        assert not banned.search(blob), f"performance language in a fact: {f['file']}:{f['line']}"


def test_provenance_names_the_commit_the_facts_were_read_at(facts):
    prov = facts["provenance"]
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", str(prov["aiter_commit"])), prov
    assert prov["aiter_commit"] != "unknown", "facts without a commit have no expiry date"


def test_the_committed_corpus_resolves_at_the_commit_it_names(facts):
    """The corpus's one promise, checked rather than assumed.

    An earlier version asserted only that a commit was PRESENT, and passed while 56 records pointed
    into a locally-modified kernel that the named commit does not contain — `file:line` resolved for
    the person who ran the extractor and nobody else. Present and identifying are different
    properties, and only the second one is worth anything in a citation index.
    """
    dirt = facts["provenance"].get("aiter_dirty_sources")
    assert dirt == [] or dirt is None or not dirt, (
        f"extracted from a tree with uncommitted changes in {dirt}; re-extract from a clean "
        f"checkout (`git -C <aiter> worktree add /tmp/aiter-clean <commit>`)")
    assert not [f for f in facts["facts"] if f.get("unreproducible")], \
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
    got = {f["file"]: f.get("unreproducible") for f in payload["facts"]["facts"]}
    assert got == {"dirty.py": True, "clean.py": None}, got
    assert payload["facts"]["provenance"]["aiter_dirty_sources"] == ["dirty.py"]


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
    assert json.loads(capsys.readouterr().out)["facts"]["provenance"]["aiter_dirty_sources"] == []


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


def test_gaps_are_stated_with_a_reason(facts):
    for m in facts.get("missing") or []:
        assert m.get("language") and m.get("subtree") and m.get("why"), m


def test_every_language_declared_actually_produced_facts(facts):
    """A language present in `IMPLS` but absent from the output means the globs no longer match the
    upstream layout — which is how `ck` sat at zero while looking like a deliberate omission."""
    declared = {l for l, _, _ in EX.IMPLS}
    got = {f["language"] for f in facts["facts"]}
    assert declared == got, f"declared but empty: {sorted(declared - got)}"


def test_the_rendered_page_matches_the_facts():
    if not os.path.exists(DOC):
        pytest.skip("doc not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stale doc; re-run --emit\n{r.stdout}{r.stderr}"


def test_the_page_states_no_ranking(facts):
    """The prose is hand-written, so it is the one place a claim could slip in. Checked as text."""
    if not os.path.exists(DOC):
        pytest.skip("doc not generated")
    with open(DOC, encoding="utf-8") as f:
        body = f.read()
    for phrase in ("is faster", "is better", "outperforms", "you should use", "the best "):
        assert phrase not in body.lower(), f"the page ranks: {phrase!r}"


def test_tuned_groups_carry_arch_and_bucket(tuned):
    """A knob set without its gfx and its M bucket reads as a global recommendation. It is not one."""
    for g in tuned["tuned_configs"]:
        assert str(g["gfx"]).startswith("gfx"), g
        assert g["m_bucket"], g
        assert int(g["shapes_swept"]) >= 1, g
        assert ("same_across_shapes" in g) != ("knobs" in g), \
            f"exactly one of the two forms, chosen by shapes_swept: {g}"


def test_the_extractor_refuses_to_run_without_a_source_tree():
    """`--emit` with no `--aiter` would write an empty corpus over a good one."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "_extract_impl_facts.py"), "--emit"],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "--aiter" in (r.stderr + r.stdout)


def test_the_renderer_runs_without_an_aiter_checkout():
    """The property that makes `--check` a usable gate: rendering reads only committed YAML."""
    if not os.path.exists(FAMILY):
        pytest.skip("facts not generated")
    r = subprocess.run([sys.executable, os.path.join(HERE, "_render_facts.py")],
                       capture_output=True, text=True, cwd="/")
    assert r.returncode == 0, r.stderr
    assert "## Coverage" in r.stdout


def test_json_mode_is_machine_readable():
    if not os.path.exists(FAMILY):
        pytest.skip("facts not generated")
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
    assert payload["facts"]["facts"] and payload["tuned"]["tuned_configs"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
