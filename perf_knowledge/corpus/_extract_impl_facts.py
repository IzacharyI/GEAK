#!/usr/bin/env python3
"""Extract source evidence from AITER's six GEMM implementations, with file:line provenance.

    # from the repo root, with an aiter checkout to read
    python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /sgl-workspace/aiter --emit
    python3 perf_knowledge/corpus/_extract_impl_facts.py --aiter /sgl-workspace/aiter --json
    python3 perf_knowledge/corpus/_render_facts.py --emit             # evidence YAML -> source index
    python3 perf_knowledge/corpus/_render_decisions.py --emit         # cards + evidence -> advice

WHAT PROBLEM THIS SOLVES
GEAK has to author FlyDSL, and the FlyDSL corpus is thin: 33 files against Triton's 318. The
knowledge needed to close that gap is already on disk — the same operators are implemented six ways
in AITER, and every one of those files is a record of somebody's decision about tiling, LDS staging,
layout and which MFMA to issue. What was missing is a way to LOOK ONE UP. `kernel_families.md` is
the hand-written attempt and it shows the limits: file-level pointers, no line numbers, and claims
like "default tile 128x128x64" that no longer have anything checking them.

So this walks the implementations and emits one record per source observation, each carrying a
content-bound identity, `file:line` and the source lines themselves.

WHAT IT DOES NOT DO — and this is the whole safety argument
It records what the source SAYS. It never says why a choice is faster, never ranks two
implementations, and never turns a constant into a recommendation. Those are claims, claims need
measurement, and nothing here measured anything: this is a citation index over code that already
exists. The docs that DO make claims (`languages/flydsl/*.md`) become checkable against it, which is
the actual win — a reader can go from "FlyDSL picks a different MFMA on gfx950" to the four lines
that do it.

The evidence excerpt is copied verbatim for the same reason. A paraphrase of a tile derivation is a
new artifact that can be wrong; the lines themselves can only be stale, and staleness is detectable
because every record carries the commit it was read at.
"""
import argparse
import ast
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(HERE)
ROOT = os.path.dirname(PK)
EVIDENCE = os.path.join(HERE, "evidence")

# The GEMM family, as the pilot. Each entry is (language, subtree, glob) — the subtree is where that
# language's implementations live in aiter, and languages are kept apart because "which language is
# this" must not be inferred here: `kernel_workflow/scripts/detect_language.py` exists for source
# whose language is unknown, and a corpus keyed on a guess is worse than a smaller correct one.
IMPLS = [
    ("flydsl", "aiter/ops/flydsl", ("kernels/splitk_hgemm.py", "kernels/small_m_hgemm.py",
                                   "kernels/preshuffle_gemm.py",
                                   "kernels/mfma_preshuffle_pipeline.py",
                                   "kernels/moe_gemm_2stage.py",
                                   "kernels/mixed_moe_gemm_2stage.py",
                                   "kernels/hgemm_dispatch.py", "kernels/mfma_epilogues.py",
                                   "kernels/layout_utils.py", "gemm_kernels.py")),
    ("triton", "aiter/ops/triton/_triton_kernels/gemm", ("basic/*.py", "batched/*.py", "fused/*.py")),
    ("gluon", "aiter/ops/triton", ("_gluon_kernels/*/moe/*gemm*.py", "_gluon_kernels/*/gemm/*.py",
                                   "gluon/*gemm*.py")),
    # HIP and CK both live under `csrc/` and both are `.cu`/`.cuh`, so the split is by DIRECTORY,
    # not by extension: `ck_*` subtrees are CK template instantiations, `py_itfs_cu/` and `kernels/`
    # are hand-written HIP. Getting this backwards is the mis-filing `detect_language.py` guards
    # against on the source side — a CK instance read as HIP teaches a FlyDSL author to look for a
    # hand-written loop where there is a template parameter list.
    # Hand-written HIP GEMM in aiter is the `opus_gemm` tree. The kernels themselves are the per-arch
    # `include/gfx*/opus_gemm_pipeline_*.cuh` and `..._traits_*.cuh` — the top-level `.cu` is only a
    # dispatcher — and the codegen scripts hold the tile tables the instances are stamped from. All
    # three layers are wanted: the pipeline says how, the traits say with what, the codegen says which
    # combinations were thought worth building.
    ("hip", "csrc", ("py_itfs_cu/gemm_common.cu", "include/gemm_common.h",
                     "include/gemm_dispatch_utils.h", "opus_gemm/*.cu",
                     "opus_gemm/include/*.cuh", "opus_gemm/include/*.h",
                     "opus_gemm/include/gfx*/opus_gemm_pipeline_*.cuh",
                     "opus_gemm/include/gfx*/opus_gemm_traits_*.cuh",
                     "opus_gemm/codegen/gen_instances_*.py", "opus_gemm/opus_gemm_common.py")),
    ("ck", "csrc", ("ck_gemm*/*.cu", "ck_gemm*/include/*.cuh", "ck_gemm*/include/*.h",
                    "ck_batched_gemm*/*.cu", "ck_batched_gemm*/include/*.cuh",
                    "cktile_gemm*/**/*.cu", "cktile_gemm*/**/*.cuh",
                    "ck_tile_gemm_moe_2stages/**/*.cu", "ck_tile_gemm_moe_2stages/**/*.cuh",
                    "ck_deepgemm/**/*.cu", "ck_deepgemm/**/*.cuh")),
    # Hand-written assembly is not readable from `hsa/**/*.co` by a text pass, and the binary itself
    # is not where the decisions are stated anyway. What IS readable is the launcher: `asm_gemm_*.cu`
    # names the `.co` to load, the block/grid it is launched with, and the argument block layout the
    # assembly expects. That is the asm contract, and it is the part a FlyDSL author has to match.
    # The instructions inside the binary are a separate, deeper layer. They require a future
    # disassembly evidence pass; this source pass does not pretend that file:line can describe a
    # binary instruction.
    ("asm", "csrc/py_itfs_cu", ("asm_gemm_*.cu", "asm_flatmm_*.cu",
                                "asm_a8w8_blockscale_bpreshuffle.cu")),
]

# Per-shape selected knob sets that AITER ships for its Triton GEMMs: one JSON per
# `{gfx}-{KERNEL}-{VARIANT}[-N=..-K=..]`, each mapping an M bucket to a concrete config. The source
# tree preserves neither the benchmark archive nor rejected alternatives, so these are useful seeds,
# not measured winners.
TUNED_CONFIG_DIR = "aiter/ops/triton/configs/gemm"


# ---------------------------------------------------------------------------------------------
# Fact rules. Each is a category plus a matcher over a stripped-comment view of the file.
#
# Regex rather than a per-language parser, deliberately and with its limit stated: six languages
# means six grammars, and a Python AST pass over Triton would still not read a `.cu`. What matters
# for a citation index is that a hit points at the right LINE, not that the construct is fully
# parsed — the reader opens the file. Where structure is genuinely needed (Triton's autotune config
# lists) there is an AST pass below, applied only to the .py files where it is valid.
#
# Comments are stripped before matching. `detect_language.py` learned this the expensive way: five
# plainly-HIP files were classified CK on the strength of `ck_tile::` mentions that appeared only in
# commented-out code. A corpus that cites a commented-out tile size is worse than one that misses it.
# ---------------------------------------------------------------------------------------------
RULES = [
    # --- which matrix instruction, and under what condition -----------------------------------
    ("mfma_intrinsic", r"\brocdl\.(mfma_[a-z0-9_]+)\s*\("),
    ("mfma_intrinsic", r"\b(v_mfma_[a-z0-9_]+)\b"),
    ("mfma_intrinsic", r"\b(?:tl|ttgl)\.dot\w*\s*\("),
    ("mfma_shape", r"\b(?:WMMA|MFMA|INSTR)_(?:M|N|K)\s*=\s*(\d+)"),
    ("mfma_shape", r"\bmatrix_instr_nonkdim\s*[=:]\s*(\d+)"),
    ("mfma_shape", r"\binstr_shape\s*[=:]\s*[\(\[]([^)\]]*)[\)\]]"),
    ("mfma_shape", r"\bm16n16k(\d+)\b"),

    # --- tiling ---------------------------------------------------------------------------------
    ("tile_shape", r"\b(BLOCK_SIZE_[MNK]|BLOCK_[MNK]|TILE_[MNK]|tile_[mnk])\s*[=:]\s*([0-9]+)"),
    ("tile_shape", r"\b(BM|BN|BK)\s*=\s*([0-9]+)"),
    ("waves", r"\b(BLOCK_[MNK]_WARPS|num_warps|NUM_WARPS|WARP_SIZE|waves_per_eu)\s*[=:]\s*([0-9]+)"),
    ("split_k", r"\b(SPLIT_K|split_k|SPLITK|NUM_KSPLIT)\s*[=:]\s*([0-9]+)"),
    ("split_k", r"\b(IS_SPLIT_K|IS_SLICE_K)\s*=\s*"),
    ("persistent", r"\b(persistent|PERSISTENT|NUM_XCDS|num_xcds)\w*\s*[=:]\s*"),

    # --- LDS ------------------------------------------------------------------------------------
    ("lds_stage", r"\b(STAGES|lds_stage|LDS_STAGE|num_stages|NUM_STAGES)\s*[=:]\s*([0-9]+)"),
    ("lds_swizzle", r"\bdef\s+(swizzle_\w+)\s*\("),
    ("lds_swizzle", r"\b(swizzle_xor\w*|xor_swizzle|XOR_SWIZZLE)\b"),
    ("lds_pad", r"\b(\w*(?:pad|PAD)\w*)\s*[=:]\s*([0-9]+)"),
    ("lds_alloc", r"\b(?:fx\.)?(?:alloc_shared|shared_alloc|allocate_shared|smem_alloc)\s*\("),
    ("lds_alloc", r"\b__shared__\b"),

    # --- global memory / async copy ---------------------------------------------------------------
    ("async_copy", r"\b(ASYNC_COPY|use_async_copy|async_copy)\s*[=:]\s*"),
    ("access_width", r"\b(DMA_BYTES|LDG_VEC_SIZE|DTYPE_BYTES|VEC_SIZE)\s*=\s*([0-9]+)"),
    ("cache_policy", r"\b(?:cache_modifier|CACHE_MODIFIER)\s*[=:]\s*['\"]?(\.?[a-z]+)"),
    ("cache_policy", r"\b(?:nt|sc0|sc1|nontemporal|GLC|SLC)\s*=\s*True"),

    # --- layout ------------------------------------------------------------------------------------
    ("layout_preshuffle", r"\b(b_preshuffle|preshuffle|BPRESHUFFLE|shuffle_weight)\w*\s*[=:(]"),
    ("layout_transpose", r"\b(?:trans_?[ab]|TRANS_[AB]|transpose_[ab])\s*[=:]\s*"),
    ("layout_epilogue", r"\b(use_cshuffle_epilog|cshuffle|CShuffle|c_shuffle)\w*\s*[=:(]"),

    # --- scheduling ---------------------------------------------------------------------------------
    ("scheduling", r"\brocdl\.(sched_\w+|s_setprio|iglp_opt)\s*\("),
    ("scheduling", r"\b(?:tl|ttgl)\.(?:async_wait|barrier|debug_barrier)\s*\("),

    # --- the asm contract, as stated by the launcher ------------------------------------------------
    # A hand-written assembly kernel publishes no source. What it publishes is the shape of the call:
    # the object to load, the workgroup it assumes, the grid formula its block-id decode expects, and
    # the byte layout of its argument block. A FlyDSL author reimplementing one of these has to match
    # all four, and every one of them is written down here rather than in the binary.
    ("asm_object", r"\b(co_name|CO_NAME|hsaco|\w+\.co)\b"),
    ("asm_launch", r"^\s*(?:\w[\w:<>*&\s]*?\s)?blockSize[XYZ]?\s*=\s*([0-9]+)\s*;"),
    # Anchored to the start of a statement, and the right-hand side must name something. Both
    # restrictions come from real misses: unanchored, `gdx=` matched inside a
    # `printf("gdx=%d, gdy=%d")` format string; and without the letter requirement the `int gdx = 0;`
    # declarations were reported alongside the real formula, three records of `0` per launcher saying
    # nothing. A grid formula that mentions no dimension is not a grid formula.
    ("asm_launch", r"^\s*(?:int\s+)?gd[xyz]\s*=\s*([^;\"]*[A-Za-z_][^;\"]*);"),
    ("asm_launch", r"\bhipModuleLaunchKernel\s*\("),
    ("asm_argblock", r"struct\s+__attribute__\(\(packed\)\)\s+(\w+)"),
    ("asm_tile", r"\bSUB[MNK]\s*=\s*(.+?);"),
    ("asm_tile", r"\bcfg\.(tile_[mnk]|splitK)\b"),

    # --- decisions encoded in a name --------------------------------------------------------------
    # aiter's hand-written HIP GEMM stamps its instances from a naming scheme that carries the whole
    # configuration: `opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128` is block tile, then the warp
    # grid, then the MFMA shape, then the scale granularity. Matching the name is not a shortcut for
    # reading the template — it is the only place the four are stated together, and it makes the set
    # of combinations somebody chose to build enumerable.
    ("instance_name", r"\b(opus_gemm_\d+x\d+x\d+(?:x\d+)?_\d+x\d+_\d+x\d+x\d+(?:_[0-9x]+)?)\b"),
    # The pipeline variants are the HIP-side design axes, named: persistent vs not, mono-tile,
    # flat-mm, split-k, and the 4GB-safe addressing sibling that exists because a 32-bit offset
    # overflows on large tensors.
    ("pipeline_variant", r"opus_gemm_(?:pipeline|traits)_(\w+?)_gfx\d+"),
    ("pipeline_variant", r"\b(is_4g_safe|_4g_safe)\b"),
    # CK states its configuration as a positional template argument list, and the corpus does not try
    # to decode it — 40-odd unnamed positions are exactly the kind of thing a regex would get subtly
    # wrong. What it records instead is the instance template chosen and the two arguments CK does
    # name, the loop scheduler and the pipeline version, because those are the decisions with an
    # analogue an author can act on: interwave vs intrawave is a scheduling choice a FlyDSL kernel
    # also has to make by hand.
    ("ck_instance", r"\b(DeviceGemm\w*|DeviceBatchedGemm\w*|GemmPipelineAgBgCr\w*)\s*<?"),
    ("ck_pipeline", r"BlockGemmPipeline(?:Scheduler|Version)::(\w+)"),
    ("ck_pipeline", r"\b(GemmSpecialization::\w+)"),

    # --- the tunable surface, as declared ---------------------------------------------------------
    # Which knobs a kernel even has. In Triton this is the `tl.constexpr` parameter list, and it is
    # worth its own category because it answers a different question from a tile VALUE: the values
    # for these live in the shipped JSON, so the source can only tell you what is adjustable. An
    # author porting to FlyDSL needs the list before the numbers.
    ("tunable_param", r"^\s*(\w+)\s*:\s*tl\.constexpr\s*,?\s*$"),
]

# Which gfx a decision is conditional on. Captured separately from the rules because it applies to
# ALL of them: a tile size inside `if arch == "gfx942"` is a different fact from the same number in
# a shared path, and "only true on one arch" is precisely what the user of this corpus needs told.
ARCH_GATE = re.compile(
    r"""(?x)
    (?: GPU_ARCH | arch | ARCH | gpu_arch | target | get_rocm_arch\(\) )
    \s* (?: == | \sin\s | \.startswith ) \s*
    [\(\[]? \s* ['"]? (gfx\d+[a-z]*) """)
ARCH_ANY = re.compile(r"\b(gfx\d{3,4}[a-z]?)\b")


def strip_comments(text, python):
    """Blank out comment bodies, preserving line count and column positions.

    Line numbers are the product here, so nothing may shift: a record whose excerpt points three
    lines off is worse than no record, because the reader trusts it enough not to double-check.
    """
    if python:
        return re.sub(r"#[^\n]*", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)
    return re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.DOTALL)


def arch_context(lines, idx):
    """The gfx this line sits under, by walking back through enclosing conditions.

    Indentation-based for Python and brace-agnostic for C++, so it is a hint and labelled as one:
    `arch_scope` is reported, never used to filter. An over-eager arch attribution would quietly
    narrow a fact that is actually general, and the reader has the line number to check.
    """
    for j in range(idx, max(-1, idx - 40), -1):
        m = ARCH_GATE.search(lines[j])
        if m:
            return m.group(1)
    return ""


def excerpt(lines, idx, span=2):
    lo, hi = max(0, idx - span), min(len(lines), idx + span + 1)
    return [f"{n + 1}: {lines[n].rstrip()}" for n in range(lo, hi) if lines[n].strip()]


def source_evidence_id(record):
    """Content-bound identity used by decision cards instead of a loose `file:line` pointer.

    `file:line` is useful provenance but not an identity: after an upstream refresh the same line can
    contain a different construct and a card would still appear grounded. Including the match and
    excerpt means such a change invalidates the reference and forces the card to be reviewed.
    """
    fields = {
        key: record.get(key)
        for key in ("category", "language", "file", "line", "match", "captured",
                    "arch_scope", "evidence")
    }
    canonical = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "src_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def scan_file(path, rel, language):
    """Every rule hit in one file, as source-evidence records.

    A hit answers only "what is written here?". It is not yet a development decision: deciding
    whether to copy, benchmark or reject the pattern requires conditions and evidence strength, which
    live in `decisions/gemm.yaml`.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    view = strip_comments(raw, python=path.endswith(".py"))
    vlines, rlines = view.splitlines(), raw.splitlines()
    out = []
    for category, pattern in RULES:
        # MULTILINE so a rule can anchor to a line, which is what a declaration-per-line construct
        # like Triton's `NAME: tl.constexpr,` parameter list needs. No rule uses `$` to mean
        # end-of-file, so the wider meaning of the anchors costs nothing here.
        for m in re.finditer(pattern, view, re.MULTILINE):
            idx = view.count("\n", 0, m.start())
            if idx >= len(rlines):
                continue
            groups = [g for g in m.groups() if g] if m.groups() else []
            record = {
                "category": category,
                "language": language,
                "file": rel,
                "line": idx + 1,
                "match": m.group(0).strip(),
                "captured": groups,
                "arch_scope": arch_context(vlines, idx),
                "evidence": excerpt(rlines, idx),
            }
            record["evidence_id"] = source_evidence_id(record)
            out.append(record)
    # Overlapping rules occasionally describe the exact same source construct (notably asm launcher
    # aliases). One piece of evidence must have one identity; keeping duplicates inflates coverage and
    # makes an ID ambiguous to a decision card.
    unique = {}
    for record in out:
        unique.setdefault(record["evidence_id"], record)
    return list(unique.values()), raw, vlines


def autotune_configs(path, rel):
    """Triton's `@triton.autotune(configs=[Config({...}, num_warps=..), ...])`, via AST.

    The one place a parser earns its keep: the tuned search space is a literal in the decorator, and
    it is the single most reusable thing in a Triton kernel — a FlyDSL author starting from scratch
    wants the tile shapes somebody already swept, not a guess. Regex cannot read a nested dict
    reliably, and a half-read config list would be an invented search space.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if not name.endswith("Config"):
            continue
        knobs = {}
        for arg in node.args:
            if isinstance(arg, ast.Dict):
                for k, v in zip(arg.keys, arg.values):
                    try:
                        knobs[ast.literal_eval(k)] = ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        pass
        for kw in node.keywords:
            try:
                knobs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                pass
        if knobs:
            out.append({"file": rel, "line": node.lineno, "knobs": knobs})
    return out


TUNED_SHAPE_SEG = re.compile(r"^(?P<dim>[MNK])=(?P<val>\d+)$")


def parse_tuned_name(name):
    """Split `gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json` into gfx, tags, shape.

    A splitter rather than a grammar, on purpose. The names carry more forms than a regex should
    claim to know — `FUSED-GEMM-AFP4WFP4-A16W16` has two dtype tags, `FF-A16W16-fused` mixes case,
    `GEMM-A16W16-ATOMIC` and `-gated` are epilogue markers — and a pattern that insisted on
    kernel-then-variant silently dropped 239 of 257 files into an empty-field bucket. Splitting on
    `-` and keeping the middle segments as an ordered tag list records what the name says without
    inventing a taxonomy the filenames do not actually promise.
    """
    stem = name[:-len(".json")] if name.endswith(".json") else name
    segs = stem.split("-")
    if not segs or not segs[0].startswith("gfx"):
        return None
    gfx, shape, tags = segs[0], {}, []
    for seg in segs[1:]:
        m = TUNED_SHAPE_SEG.match(seg)
        if m:
            shape[m.group("dim").lower()] = int(m.group("val"))
        else:
            tags.append(seg)
    return {"gfx": gfx, "tags": tags, "shape": shape}


def tuned_configs(aiter):
    """The shipped per-shape knob sets, as one record per (file, M bucket).

    Flattened rather than summarised. The obvious compression — "here are the tile shapes AITER
    uses" — throws away the two things that make the table worth having: which gfx it was tuned on,
    and which M bucket it applies to. A knob set is only meaningful with both, and a corpus that
    dropped them would read as a global recommendation, which is exactly what it is not.
    """
    base = os.path.join(aiter, TUNED_CONFIG_DIR)
    if not os.path.isdir(base):
        return [], [{"language": "triton", "subtree": TUNED_CONFIG_DIR,
                     "why": "no such directory (no shipped tuned configs to read)"}]
    out, unparsed = [], []
    for name in sorted(os.listdir(base)):
        if not name.endswith(".json"):
            continue
        meta = parse_tuned_name(name)
        if meta is None:
            unparsed.append(name)
            continue
        try:
            with open(os.path.join(base, name), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            unparsed.append(name)
            continue
        if not isinstance(data, dict):
            unparsed.append(name)
            continue
        for bucket, knobs in data.items():
            if not isinstance(knobs, dict):
                continue
            out.append({"file": os.path.join(TUNED_CONFIG_DIR, name),
                        "gfx": meta["gfx"], "tags": meta["tags"], "shape": meta["shape"],
                        "m_bucket": str(bucket),
                        "knobs": {k: v for k, v in sorted(knobs.items())}})
    gap = []
    if unparsed:
        gap.append({"language": "triton", "subtree": TUNED_CONFIG_DIR,
                    "why": f"{len(unparsed)} config filename(s) not in `gfx…-TAG…[-N=..]` form, "
                           f"skipped rather than guessed: {', '.join(sorted(unparsed)[:4])}"})
    return out, gap


def summarize_tuned(rows):
    """Group the shipped knob sets by (gfx, tags, M bucket) and report each knob's value spread.

    Grouped rather than copied. 1742 verbatim rows is a re-encoding of JSON that already exists in
    aiter — a second copy with a shelf life, and 34k lines nobody diffs. The grouping is not a
    space trick though: it surfaces the distinction the raw rows bury. Within one bucket some knobs
    take a single value across every shipped shape config (`BLOCK_SIZE_K` is 128 in all nine gfx942
    A8W8_BLOCKSCALE M<=32 files, `num_stages` always 2) while others move with the shape. The first
    kind is a useful candidate to seed; the second is a dimension they still have to vary. Both are
    read off the counts, not asserted — `unanimous` is literally "one distinct value".

    Exact per-shape rows are deliberately not reproduced: an author who needs N=4096,K=7168 should
    open the source JSONs, whose exact paths are retained on every group below.
    """
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["gfx"], " ".join(r["tags"]), r["m_bucket"])].append(r)
    out = []
    for (gfx, tags, bucket), members in sorted(groups.items()):
        spread = collections.defaultdict(collections.Counter)
        for mem in members:
            for k, v in mem["knobs"].items():
                spread[k][json.dumps(v)] += 1
        fixed, varies = {}, {}
        for k, counter in sorted(spread.items()):
            if len(counter) == 1:
                fixed[k] = json.loads(next(iter(counter)))
            else:
                # `value: count`, not `valueXcount` — `.cgx23` reads as a cache modifier named
                # `.cgx23` rather than `.cg` seen 23 times, and `32x20` as a tile shape.
                varies[k] = ", ".join(f"{json.loads(val)}: {n}" for val, n in counter.most_common())
        rec = {
            "config_id": "cfg_" + hashlib.sha256(
                json.dumps([gfx, tags, bucket], separators=(",", ":")).encode()
            ).hexdigest()[:16],
            "gfx": gfx,
            "tags": tags,
            "m_bucket": bucket,
            "shape_configs": len(members),
            "source_files": sorted({member["file"] for member in members if member.get("file")}),
        }
        if len(members) == 1:
            # One shape config means there is no cross-shape agreement to report, so the field is not
            # called agreement. `same_across_configs` over a single row would read as an invariant
            # backed by evidence that does not exist.
            rec["knobs"] = fixed
        else:
            # Every shipped shape config in this bucket carries this value. Useful seed evidence, not
            # proof of a constraint — the source does not preserve alternatives or benchmark results.
            rec["same_across_configs"] = fixed
            rec["varies_by_shape"] = varies
        out.append(rec)
    return out


def aiter_commit(aiter):
    r = subprocess.run(["git", "-C", aiter, "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or "unknown"


def aiter_origin(aiter):
    """The repository the source evidence came from, as a remote URL where one exists.

    Not the local path. A reader who wants to check a citation needs to know WHICH aiter, and the
    honest way to extract from a dirty working copy is a throwaway clean worktree — which made the
    recorded path `/tmp/aiter-clean`, a directory that no longer exists and never meant anything to
    anyone else. The remote plus the commit is the pair that actually identifies the source.
    """
    r = subprocess.run(["git", "-C", aiter, "remote", "get-url", "origin"],
                       capture_output=True, text=True, check=False)
    return r.stdout.strip() or f"(no remote; read from {os.path.abspath(aiter)})"


def dirty_paths(aiter):
    """Repo-relative paths with uncommitted changes, as a set.

    Needed per-file, not as a boolean. The boolean this replaced (`aiter_tree_dirty`) recorded the
    problem and did nothing with it: 56 records pointed into a locally-modified kernel while the
    provenance named a commit that did not contain those edits, so `file:line` did not resolve for
    anyone but the person who ran the extractor. That is the corpus's one promise, broken, with a
    green test suite — the test asserted a commit was PRESENT, never that it IDENTIFIED what was read.
    """
    r = subprocess.run(["git", "-C", aiter, "status", "--porcelain"],
                       capture_output=True, text=True, check=False)
    out = set()
    for line in r.stdout.splitlines():
        if len(line) > 3:
            # `XY path` — and for a rename, `XY old -> new`; the new name is what was read.
            out.add(line[3:].split(" -> ")[-1].strip().strip('"'))
    return out


def collect(aiter):
    """Walk the six implementation sets. Missing subtrees are reported, never silently empty."""
    import glob as globmod
    evidence, configs, seen, missing = [], [], [], []
    for language, subtree, patterns in IMPLS:
        base = os.path.join(aiter, subtree)
        if not os.path.isdir(base):
            missing.append({"language": language, "subtree": subtree, "why": "no such directory"})
            continue
        files = []
        for pat in patterns:
            files += sorted(globmod.glob(os.path.join(base, pat), recursive=True))
        files = [f for f in files if "__pycache__" not in f and os.path.isfile(f)]
        if patterns and not files:
            missing.append({"language": language, "subtree": subtree,
                            "why": f"no file matched {list(patterns)}"})
            continue
        for path in files:
            rel = os.path.relpath(path, aiter)
            got, raw, _ = scan_file(path, rel, language)
            evidence += got
            if language in ("triton", "gluon") and path.endswith(".py"):
                configs += [dict(c, language=language) for c in autotune_configs(path, rel)]
            seen.append({"language": language, "file": rel, "lines": raw.count("\n") + 1,
                         "evidence_records": len(got)})
    tuned, tuned_gap = tuned_configs(aiter)
    # An empty autotune list is a finding, not a failed pass, so it is stated rather than left as a
    # zero a reader would take for a bug. AITER's GEMM Triton kernels carry no `@triton.autotune` at
    # all: the search space lives in the shipped JSON instead. Worth saying out loud, because an
    # author who greps for autotune in a GEMM kernel and finds nothing will otherwise conclude the
    # kernel is untuned, when in fact it is the most heavily swept code in the tree.
    if not configs:
        missing.append({"language": "triton", "subtree": "aiter/ops/triton/_triton_kernels/gemm",
                        "why": "no @triton.autotune in the GEMM family — tuning is shipped as "
                               f"per-shape JSON under {TUNED_CONFIG_DIR} "
                               f"(see evidence/gemm_tuned_configs.yaml, {len(tuned)} rows)"})
    return evidence, configs, tuned, seen, missing + tuned_gap


# ---------------------------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------------------------
def yaml_quote(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v)
    # Newlines escaped, not passed through. A CK template head like `DeviceGemmMX_Xdl_CShuffleV3\n<`
    # matches across a line break, and writing that raw put a second physical line inside a quoted
    # scalar — which PyYAML folds back correctly, but which breaks the one-record-per-line property
    # the whole hand-rolled format depends on, and makes `git diff` attribute the change to the
    # wrong record. Tabs go too, for the same "a scalar occupies exactly one line" reason.
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n").replace("\t", "\\t")
    return '"' + s + '"'


def emit_yaml(payload, path):
    """Hand-rolled, because a corpus file has to be readable in a diff and PyYAML's dump is not.

    Only two shapes appear (a scalar map and a list of scalar maps), which is little enough to write
    out directly and keeps this tool importable on a box without PyYAML — the same reason `kb.py`
    carries its own front-matter parser.
    """
    L = ["# GENERATED by perf_knowledge/corpus/_extract_impl_facts.py — do not hand-edit.",
         "# These are SOURCE-EVIDENCE records: what a file states, not what an author should choose.",
         "# Actionable choices live in decisions/gemm.yaml and are rendered to gemm_decisions.md.", ""]
    L.append("provenance:")
    for k, v in payload["provenance"].items():
        if isinstance(v, list):
            L.append(f"  {k}: [{', '.join(yaml_quote(x) for x in v)}]")
        else:
            L.append(f"  {k}: {yaml_quote(v)}")
    L.append("")
    for section in ("coverage", "missing", "autotune_configs", "tuned_configs", "source_evidence"):
        if section not in payload:
            continue
        rows = payload[section]
        L.append(f"{section}:" if rows else f"{section}: []")
        for row in rows:
            first = True
            for k, v in row.items():
                lead = "  - " if first else "    "
                first = False
                if isinstance(v, list) and v and isinstance(v[0], str):
                    L.append(f"{lead}{k}:")
                    L += [f"      - {yaml_quote(x)}" for x in v]
                elif isinstance(v, dict):
                    L.append(f"{lead}{k}:" if v else f"{lead}{k}: {{}}")
                    L += [f"      {kk}: {yaml_quote(vv)}" for kk, vv in sorted(v.items())]
                elif isinstance(v, list):
                    L.append(f"{lead}{k}: [{', '.join(yaml_quote(x) for x in v)}]")
                else:
                    L.append(f"{lead}{k}: {yaml_quote(v)}")
        L.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aiter", help="an aiter checkout to read (required for --emit / --json)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true", help="write evidence/gemm_source.yaml")
    mode.add_argument("--json", action="store_true", help="dump to stdout instead")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="extract even when files that produce evidence have uncommitted changes; "
                         "the affected records are marked `unreproducible: true`")
    a = ap.parse_args()

    if not a.aiter:
        ap.error("--aiter is required: this tool reads a source tree that is not part of this repo")
    if not os.path.isdir(a.aiter):
        print(f"no such aiter tree: {a.aiter}", file=sys.stderr)
        return 2

    evidence, configs, tuned, seen, missing = collect(a.aiter)
    commit = aiter_commit(a.aiter)
    for record in evidence:
        record.setdefault("evidence_id", source_evidence_id(record))

    # Which files that CONTRIBUTED evidence are locally modified. Scoped to contributors rather than
    # the whole tree because that is the actual condition: an unrelated edit elsewhere in aiter cannot
    # make a citation unresolvable, and refusing on it would be superstition rather than a check.
    contributors = {f["file"] for f in evidence}
    unreproducible = sorted(contributors & dirty_paths(a.aiter))
    if unreproducible and not a.allow_dirty:
        print(f"REFUSING: {len(unreproducible)} file(s) that produced evidence have uncommitted changes, "
              f"so their `file:line` would not resolve at {commit[:12]} for anyone else:",
              file=sys.stderr)
        for p in unreproducible[:10]:
            print(f"  {p}", file=sys.stderr)
        print("\nA citation index whose citations do not resolve is not a weaker index, it is a\n"
              "different document that looks like one. Either commit those changes, or extract from a\n"
              "clean checkout:\n"
              f"  git -C {a.aiter} worktree add /tmp/aiter-clean {commit[:12]}\n"
              "Pass --allow-dirty to record the evidence anyway; the affected records are then marked\n"
              "and the committed corpus is not allowed to contain them.", file=sys.stderr)
        return 3

    prov = {
        # The commit is the expiry date. Without it a stale fact and a current one are the same
        # document, which is how `kernel_families.md` came to state a default tile nothing checks.
        "aiter_commit": commit,
        # Empty means every citation below resolves at that commit for anyone. Non-empty names exactly
        # which ones do not, which is the part a boolean could not say.
        "aiter_dirty_sources": unreproducible,
        "aiter_origin": aiter_origin(a.aiter),
        "extractor": "perf_knowledge/corpus/_extract_impl_facts.py",
        "operator_family": "gemm",
    }
    if unreproducible:
        dirt = set(unreproducible)
        for f in evidence:
            if f["file"] in dirt:
                f["unreproducible"] = True
    payload = {
        "provenance": dict(prov, evidence_records=len(evidence), autotune_configs=len(configs)),
        "coverage": seen, "missing": missing, "autotune_configs": configs,
        "source_evidence": evidence,
    }
    # The shipped config table changes on a different clock from the source (a reselection rewrites all
    # of it; a kernel edit touches a few lines), so it gets its own file and its own diff.
    grouped = summarize_tuned(tuned)
    tuned_payload = {
        "provenance": dict(prov, records=len(grouped), rows_summarised=len(tuned),
                           source=TUNED_CONFIG_DIR),
        "tuned_configs": grouped,
    }
    if a.json:
        print(json.dumps({"evidence": payload, "tuned": tuned_payload}, ensure_ascii=False, indent=2))
        return 0

    os.makedirs(EVIDENCE, exist_ok=True)
    out = os.path.join(EVIDENCE, "gemm_source.yaml")
    tuned_out = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
    emit_yaml(payload, out)
    emit_yaml(tuned_payload, tuned_out)
    by_lang = collections.Counter(f["language"] for f in evidence)
    print(f"OK: {len(evidence)} source-evidence records from {len(seen)} files "
          f"({', '.join(f'{k}={v}' for k, v in sorted(by_lang.items()))}), "
          f"{len(configs)} autotune -> {os.path.relpath(out, ROOT)}")
    print(f"OK: {len(tuned)} shipped tuned rows in {len(grouped)} (gfx, tags, M) groups "
          f"-> {os.path.relpath(tuned_out, ROOT)}")
    for m in missing:
        print(f"  gap: {m['language']} — {m['why']} ({m['subtree']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
