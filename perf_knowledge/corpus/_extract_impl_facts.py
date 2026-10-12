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
import csv
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
    # `gemm_tune/…_common.py` is where the A8 (fp8/int8) preshuffle family states its search space —
    # per-arch tile catalogs, the lds_stage x cshuffle x async x waves_per_eu sweep and the register
    # cap on waves_per_eu. Without it the corpus could cite that family's kernel but not the set of
    # configurations AITER is willing to build for it. `utils.py` carries the per-arch LDS budget
    # every tiling estimate is compared against.
    # Patterns are the UNION of the layouts across the indexed trees, not the layout of any one of
    # them. The FlyDSL GEMM family was restructured between a6bb4993 and 4faf8713 — `splitk_hgemm.py`
    # and `small_m_hgemm.py` became `gemm_a16w16_gfx950.py`, `hgemm_dispatch.py` became
    # `gemm_a16w16_policy.py`, `utils.py` moved under `kernels/` — and a pattern list that tracked
    # only the newer one would read the older tree as empty, which is the opposite of why it is
    # indexed. Each tree reports its own non-matching patterns in `missing`, qualified by commit, so
    # the gap list states what the restructure moved instead of hiding it.
    ("flydsl", "aiter/ops/flydsl", (
        # Dense bf16/fp16. First three are <= a6bb4993, the a16w16 family is the rewrite.
        "kernels/splitk_hgemm.py", "kernels/small_m_hgemm.py", "kernels/hgemm_dispatch.py",
        "kernels/gemm_a16w16*.py", "gemm_a16w16_policy.py",
        # Pipeline, epilogue and layout: shared by dense and grouped, and the part that survived the
        # rewrite in place — which is why evidence here is the strongest in the corpus.
        "kernels/preshuffle_gemm.py", "kernels/mfma_preshuffle_pipeline.py",
        "kernels/mfma_epilogues.py", "kernels/layout_utils.py", "kernels/splitk_epilogue.py",
        "kernels/kernels_common.py", "gemm_kernels.py",
        # Grouped (MoE) two-stage GEMM: gemm1 (gate+up, fused activation) and gemm2 (down, weighted
        # combine). Globbed as families because the same two kernels appear under four names across
        # the two trees — `moe_gemm_2stage.py`, `moe_gemm_2stage_gfx942.py`, the a16wmix pair split
        # into explicit `gemm1.py`/`gemm2.py`, and the mega_moe pair.
        "kernels/moe_gemm_2stage*.py", "kernels/mixed_moe_gemm_2stage*.py",
        "kernels/moe_2stage_a16wmix/*.py", "kernels/mega_moe/gemm*.py",
        "kernels/mxfp4_gemm*.py",
        # The mxfp4 (a4w4 block-scaled) arm of the grouped family is split across four files that
        # the `kernels/mxfp4_gemm*.py` glob alone does not reach, and all four are on the path a
        # deployed gfx950 MoE actually takes. `mxfp4_kname.py` is the config space itself — the
        # kernel NAME is the configuration, so its grammar, its variant whitelist and its
        # name-level legality checks are the only statement of which configs exist. Without it the
        # corpus can describe tiles it cannot name. `mxmoe_dispatcher.py` is the entry point that
        # turns a shape into one of those names, `mxmoe_gemm_v2.py` is the second gemm2 family
        # behind that front-end, and the two `mxfp4_gemm{1,2}_kernels.py` wrappers hold the host
        # side. These live at the package root, not under `kernels/`.
        "mxfp4_kname.py", "kernels/mxmoe_dispatcher.py", "kernels/mxmoe_gemm_v2.py",
        "mxfp4_gemm1_kernels.py", "mxfp4_gemm2_kernels.py",
        # Where the grouped launch is actually assembled: grid_y from the sorted-block count,
        # persist_m, the sorted-scale buffer sizing, and the stage-2 reduction call. The GEMM
        # codegen files state the tile; this states what the tile is launched over.
        "moe_kernels.py", "moe_sorting.py",
        # The topk combine is not always stage 2's epilogue: the `reduce` path leaves stage 2's
        # output unsummed and sums it in a separately compiled, separately launched kernel. Without
        # this file the corpus can only cite the atomic half of that choice.
        "kernels/moe_reduce.py",
        # Ragged M is the grouped family's defining constraint and it is not stated in the GEMM
        # kernel: the sorting kernel is where tokens are bucketed per expert and padded to a block
        # multiple, which is what fixes the tile the GEMM can use.
        "kernels/moe_sorting_kernel.py", "moe_common.py",
        # Per-arch LDS budget every tiling estimate is compared against; moved under `kernels/`.
        "utils.py", "kernels/utils.py",
        "gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py",
        "test_flydsl_splitk_hgemm.py")),
    # The grouped/MoE parity gates. The dense family's gate is citable because its test happens to
    # live under `aiter/ops/flydsl/`; the MoE ones live under `op_tests/`, so without this entry the
    # corpus could state what a grouped kernel must COMPUTE but not what it must PASS — and an
    # author would reasonably assume the dense elementwise gate carries over, which it does not.
    # Only the files that declare a gate are listed: routing/sorting (which assert exactness) and
    # the two-stage and grouped GEMM tests (which assert a cosine bound).
    ("flydsl", "op_tests", ("test_moe_2stage.py", "test_moe_sorting.py", "test_moe_ep.py")),
    ("flydsl", "op_tests/flydsl_tests", ("test_flydsl_moe.py", "test_flydsl_grouped_gemm.py")),
    ("triton", "aiter/ops/triton/_triton_kernels/gemm", ("basic/*.py", "batched/*.py", "fused/*.py")),
    # Triton's grouped GEMM. `moe_op_gemm_*` is the two-stage kernel per quantization scheme;
    # `activations.py` is the fused gate/up activation that makes stage 1 more than a GEMM, and
    # reading the GEMM without it would present a fusion decision as an unexplained epilogue.
    # `moe_routing/` is deliberately excluded: routing is upstream of the GEMM seam, and a corpus
    # that mixed the two would teach an author to optimize a part the task holds fixed.
    ("triton", "aiter/ops/triton/_triton_kernels/moe", ("moe_op_gemm*.py", "moe_op_mxfp4*.py",
                                                        "activations.py", "reduce.py")),
    # The GEMM kernels above call `remap_xcd` / `pid_grid` but do not define them; the definitions —
    # which are the actual workgroup-to-XCD and grouped-M rasterization — live in a shared helper.
    ("triton", "aiter/ops/triton/utils/_triton", ("pid_preprocessing.py",)),
    # How a shape reaches a shipped Triton config: a per-(N, K) file if one exists, else the generic
    # file, then M_LEQ/M_GEQ buckets, then `any`. An untuned shape gets the generic `any` entry at
    # every M, which is where a frozen Triton baseline for such a task actually sits.
    ("triton", "aiter/ops/triton/utils", ("gemm_config_utils.py",)),
    # `**` on the gemm arm: the newer tree nests the kernels one level deeper
    # (`_gluon_kernels/gfx1250/gemm/basic/gemm_a16w16.py`), and the single-segment `*` silently
    # matched nothing there while the moe arm kept the language non-empty — the failure the
    # per-pattern gap reporting below exists to catch.
    ("gluon", "aiter/ops/triton", ("_gluon_kernels/*/moe/*gemm*.py", "_gluon_kernels/*/gemm/**/*.py",
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
    # `kernels/custom_kernels.cu` is AITER's hand-written skinny GEMM/GEMV family (`wvSplitK*`,
    # `HGEMV_WFPerRow`) — the default bf16/fp16 route for M <= 16 — and `include/opus/opus.hpp` is
    # the single-header HIP micro-DSL (buffer load/store, MFMA wrappers, layouts) AITER's HIP kernels
    # are written in. The `opus_gemm/` tree lives on AITER branches newer than the pinned commit, so
    # its patterns report as `missing` until an AITER bump brings it in.
    ("hip", "csrc", ("py_itfs_cu/gemm_common.cu", "include/gemm_common.h",
                     "include/gemm_dispatch_utils.h", "kernels/custom_kernels.cu",
                     "include/opus/opus.hpp", "opus_gemm/*.cu",
                     "opus_gemm/include/*.cuh", "opus_gemm/include/*.h",
                     "opus_gemm/include/gfx*/opus_gemm_pipeline_*.cuh",
                     "opus_gemm/include/gfx*/opus_gemm_traits_*.cuh",
                     "opus_gemm/codegen/gen_instances_*.py", "opus_gemm/opus_gemm_common.py",
                     # `opus_moe/` is the HIP grouped-GEMM tree, present from 4faf8713. Unlike
                     # `opus_gemm/` it ships no codegen layer — the instances are written out — so
                     # only the dispatcher and the per-arch headers are declared.
                     "opus_moe/*.cu", "opus_moe/include/*.cuh", "opus_moe/include/*.h",
                     "opus_moe/include/gfx*/*.cuh")),
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
    # `asm_fmoe.cu` and `asm_moe_2stage.cu` are the same contract for the grouped family, and they
    # state the part that is specific to it: the sorted-token and expert-id buffers the assembly
    # expects alongside A and B, which is the argument block a FlyDSL grouped kernel has to match.
    ("asm", "csrc/py_itfs_cu", ("asm_gemm_*.cu", "asm_flatmm_*.cu",
                                "asm_a8w8_blockscale_bpreshuffle.cu",
                                "asm_fmoe.cu", "asm_moe_2stage.cu")),
    # The binaries publish no source, but each `hsa/<gfx>/<kind>/*.csv` states which kernels exist
    # for that architecture: tile, B-preshuffle and split-K capability. One record per row, per gfx.
    # `fmoe*/` carries the grouped equivalent, including the block-M the sorting must pad to.
    ("asm", "hsa", ("gfx*/*gemm*/*.csv", "gfx*/fmoe*/*.csv")),
]

# AITER's runtime dispatch — which backend a shape is routed to by default. Dispatch is library code,
# not a kernel language, so each routing statement is filed under the language of the backend it
# routes TO; library targets (`hipblaslt`, `torch`) are not languages and are not recorded.
ROUTING_FILES = ("aiter/tuned_gemm.py", "aiter/ops/gemm_op_a8w8.py", "aiter/ops/gemm_op_a16w16.py",
                 "aiter/ops/gemm_op_a4w4.py")
ROUTING_RULES = (
    r"\[\s*['\"]libtype['\"]\s*\]\s*=\s*['\"](\w+)['\"]",
    r"\blibtype['\"]?\s*\]?\s*==\s*['\"](\w+)['\"]",
    r"\breturn\s+(\w+)\s*\(",
    r"\bdef\s+(is_\w+_shape)\s*\(",
)
# Matched against `_`-separated name tokens, first hit wins: `blockscale_ck` is CK, `block` is not.
ROUTE_TARGETS = (("cktile", "ck"), ("skinny", "hip"), ("flydsl", "flydsl"), ("triton", "triton"),
                 ("gluon", "gluon"), ("asm", "asm"), ("ck", "ck"))

# Per-shape selected knob sets that AITER ships for its Triton GEMMs, each mapping an M bucket to a
# concrete config. The source tree preserves neither the benchmark archive nor rejected
# alternatives, so these are useful seeds, not measured winners.
#
# The root, not `…/configs/gemm`, because AITER has moved these twice over and the directory the
# older constant named no longer exists: `parse_tuned_location` decodes both the flat
# `{gfx}-{KERNEL}-{VARIANT}[-N=..-K=..].json` form and the nested
# `<arch>/<backend>/<op>/<dtype>/DEFAULT.json` tree that replaced it. Naming the root also brings
# the grouped (`op == moe`) tables in, which the GEMM-only path could not reach.
TUNED_CONFIG_DIR = "aiter/ops/triton/configs"


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

    # --- the shipped search space, as opposed to one point in it -----------------------------------
    # The rules above capture a tile VALUE: `TILE_M = 128` in some file. That answers "what did this
    # kernel use", which is the wrong question for somebody choosing a configuration — they need the
    # set the library is willing to compile, and the derivation that narrows it for a shape. AITER
    # states both, as module-level option tuples and as the `*_options(...)` helpers that filter them.
    # Missing this was a real gap and not a subtle one: a reader of the corpus could learn that the
    # default tile_m is 128 without learning that tile_m 16 exists, which is precisely the knob a
    # skinny-M shape needs. A single value read as the whole space is worse than no value.
    # Anchored to column 0: a module-level constant is the library's declared space, an indented
    # ALL-CAPS tuple is a local intermediate inside somebody's kernel body. The first is a fact about
    # what can be configured; the second is arithmetic, and mixing them makes the space look larger
    # and vaguer than it is.
    ("config_space", r"^([A-Z][A-Z0-9_]{2,})\s*=\s*[\(\[]"),
    ("config_space", r"\bdef\s+(\w*_options|iter_\w+_configs?|\w*_registry_variants)\s*\("),
    # The grouped (MoE) arm states its space as module-level DICTS and as a kernel-NAME grammar, so
    # the two rules above read `mxfp4_kname.py` -- which is the entire grouped config space -- as
    # nearly empty. There the configuration is not an argument at all: the kernel name IS the config,
    # and it is defined by a `re.compile` grammar plus token sets (`_MXMOE_G1_FLAG_TOKENS` and
    # friends) that enumerate which flags may appear. A leading underscore is normal for both, hence
    # `_?`; the existing rule's `[A-Z]` start misses every one of them. Dicts are looser than tuples
    # and do pull in a few module-level caches (`_GROUPED_CONFIG_CACHE = {}`), which is tolerable:
    # an empty dict carries no values to mislead a reader, while the variant whitelists it reaches
    # (`MXFP4_G1_VARIANTS`) are the only statement anywhere of which grouped kernels exist.
    ("config_space", r"^(_?[A-Z][A-Z0-9_]{2,})\s*=\s*re\.compile\("),
    ("config_space", r"^(_?[A-Z][A-Z0-9_]{2,})\s*=\s*\{"),
    # And the two ends of that grammar: the function that BUILDS a kernel name from a config and the
    # one that parses it back. Where the name is the config, these are the encode/decode pair for the
    # whole space -- `build_flydslv2_gemm2_name` is the only statement anywhere of which tokens a
    # stage-2 grouped kernel may carry and in what order, and a reader who has the token sets but not
    # the builder can still assemble a name no compiler will accept. Deliberately narrow (9 sites in
    # the indexed tree) because a loose `\bdef (\w*name\w*)` would sweep up every getter.
    ("config_space",
     r"\bdef\s+((?:build|make|format|parse)_\w*(?:_name|kname|_kernels?)|\w*kname\w*)\s*\("),
    # The same fact in CK-tile's grammar. The two rules above are Python's: a module-level tuple and
    # an `*_options()` helper. CK-tile states its space as a traits struct of `static constexpr`
    # members, so a Python-shaped rule set reads every `ck_tile::` file as empty — 60 of the indexed
    # CK files yielded nothing, which read as "CK has no configuration surface" rather than "this
    # extractor cannot see it". The members are forwarded template parameters (`M_Tile = M_Tile_`),
    # so the VALUE here is a name, not a number: the numbers live in build-time generated instances
    # that are not in the tree. That is the honest fact to capture — which knobs CK-tile exposes for
    # a family — and it is what makes CK comparable to the FlyDSL config spaces beside it. Safe to
    # match loosely because `.cu`/`.cuh` are comment-stripped as C++, so the commented-out branches
    # these files are full of are already blanked.
    ("config_space", r"\bstatic\s+constexpr\s+(?:ck_tile::index_t|int|bool|auto)\s+"
                     r"(M_Tile|N_Tile|K_Tile|[MNK]_Warp|[MNK]_Warp_Tile|kBlockPerCu|kPad[MNK]|"
                     r"Scheduler|DoubleSmemBuffer|NumWaveGroups|TransposeC|TiledMMAPermuteN|"
                     r"UseStructuredSparsity|TileParitioner\w*)\s*="),
    # Caps and floors: the boundary of the space, which is a different fact from its members. Written
    # as `HGEMM_MAX_SPLIT_K`, `MAX_LDS_BYTES`, `SMALL_M_KERNEL_MAX` — prefix optional, suffix optional.
    ("config_limit", r"\b((?:[A-Z][A-Z0-9_]*_)?(?:MAX|MIN)(?:_[A-Z0-9_]+)?)\s*=\s*([0-9]+)"),
    # What makes a configuration legal at all: LDS-budget estimators and the validators that reject a
    # tiling before it reaches the compiler. An optimizer that cannot tell "slow" from "will not
    # build" spends its budget on the second.
    ("config_validity", r"\bdef\s+(\w*(?:validate|_check_)\w*)\s*\("),
    ("config_validity", r"\bdef\s+(\w*estimate\w*|\w*lds_bytes\w*)\s*\("),
    # Which kernel FAMILY a shape is routed to. Named constants rather than a comment, and the
    # families do not share a config surface — a knob that exists in one is absent from the other.
    ("kernel_family", r"\b(KERNEL_FAMILY_[A-Z0-9_]+)\s*=\s*['\"](\w+)['\"]"),
    # The comparison sites only — those are the routing decision. The dozen places that merely pass
    # `kernel_family=` through say the same thing a dozen times.
    ("kernel_family", r"\bkernel_family\s+in\s+[\(\[]|\bkernel_family\s*==\s*(\w+)"),
    # Architecture gates. Whether a code path exists on the box you are on outranks every tuning fact
    # about it, and the corpus was silent on this: the small-M family raises outright on gfx942, and
    # nothing in the extracted evidence said so. A card that recommends a path the target architecture
    # refuses to compile does not cost a little accuracy, it costs a whole round. Match the comparison
    # sites against an arch name, not every mention of one.
    ("arch_gate", r"(?:gpu_arch|GPU_ARCH|arch|target_gfx|get_gfx\(\)|get_rocm_arch\(\))\s*[=!]=\s*['\"](gfx[0-9a-z]+)['\"]"),
    ("arch_gate", r"['\"](gfx[0-9a-z]+)['\"]\s*(?:not\s+)?in\s+|\bin\s+\{?\(?\s*['\"](gfx[0-9a-z]+)['\"]"),

    # --- LDS ------------------------------------------------------------------------------------
    ("lds_stage", r"\b(STAGES|lds_stage|LDS_STAGE|num_stages|NUM_STAGES)\s*[=:]\s*([0-9]+)"),
    ("lds_swizzle", r"\bdef\s+(swizzle_\w+)\s*\("),
    ("lds_swizzle", r"\b(swizzle_xor\w*|xor_swizzle|XOR_SWIZZLE)\b"),
    # `padded_m = 16` is the dispatcher's M-bucket granularity for a tuned-config lookup, not an LDS
    # row pad; it used to be the only "lds_pad" CK and HIP had, which read as CK padding its LDS.
    # It is recorded under `dispatch_padding` below instead.
    ("lds_pad", r"\b(?!padded_[mnk]\b)(\w*(?:pad|PAD)\w*)\s*[=:]\s*([0-9]+)"),
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

    # --- added with the A8 / epilogue / split-K / XCD cards -----------------------------------------
    # Every rule below is additive: none of them can match a construct an earlier rule already
    # records, so adding them leaves every existing `src_…` identity — and therefore every card that
    # cites one — unchanged.
    #
    # An MFMA chosen as a function object rather than called in place. The A8 kernel selects
    # `mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8` per dtype and calls `mfma_fn(...)` later, so the
    # call-site rule above saw only the scaled-MFMA branch and none of the fp8/int8/f16/bf16 choices.
    ("mfma_intrinsic", r"\brocdl\.(mfma_[a-z0-9_]+)\b(?!\s*\()"),
    ("mfma_intrinsic", r"\bgetattr\(\s*rocdl\s*,\s*['\"](mfma_[a-z0-9_]+)['\"]"),
    # Arch gates spelled through `get_hip_arch()` or a prefix test. The A8 family gates its gfx950
    # preload table and its FP4 path this way, and the `==` rule above cannot see either.
    ("arch_gate", r"\bget_hip_arch\(\)\)?\s*[=!]=\s*['\"](gfx[0-9a-z]+)['\"]"),
    ("arch_gate", r"\.startswith\(\s*['\"](gfx[0-9a-z]+)['\"]\s*\)"),
    # The declared space, in the two spellings the upper-case rule misses: private module constants
    # (`_LDS_STAGES = (1, 2)`, `_TILE_PRELOAD_TABLE = {`) and lower-case tile catalogs / per-arch
    # kernel lists (`_base_tiles_lds2_common = [`, `kernels_list_942 = …`, `default_kernels_dict_950`).
    ("config_space", r"^(_[A-Z][A-Z0-9_]{2,})\s*=\s*[\(\[\{]"),
    ("config_space", r"^(\w*tiles\w*)\s*=\s*\["),
    ("config_space", r"^(\w*kernels_(?:list|dict)\w*)\s*="),
    # Legality stated as the message of the exception that enforces it. This is where FlyDSL writes
    # most of its contract — `tile_k_bytes must be divisible by 64`, `tile_n must be divisible by
    # (CShuffleNLane*EVec)` — and a checker you cannot cite is a constraint an optimizer rediscovers
    # by compiling. Anchored on the `raise`, so the excerpt shows the condition above it.
    # The second line of verbs is the grouped-MoE vocabulary: that path states most of its contract
    # as a rejection rather than a requirement ("illegal mxmoe g2 name ... atomic incompatible with
    # f4out|cshuffle", "unknown token", "sorting layout mismatch"), and the original verb list --
    # written from the dense FlyDSL kernels, which phrase everything as `must be` -- matched none of
    # them. Still keyed on the message vocabulary rather than on `raise` alone, because a bare
    # `raise` rule matches ~1400 sites in this tree and most are argument hygiene, not legality.
    ("config_validity", r"\braise\s+(?:ValueError|NotImplementedError|RuntimeError|AssertionError)"
                        r"\(\s*f?[\"']([^\"'\n]*(?:must be|must equal|must match|requires|"
                        r"only support|not supported|not yet supported|exceed|divisible|fixes|"
                        r"illegal|unsupported|unknown token|incompatible|invalid|mismatch)"
                        r"[^\"'\n]*)"),
    # How split-K partials are combined: the module-level state constants and the helpers that rotate
    # the per-stream signal state. `SPLIT_K\s*[=:]` above only sees the split factor itself.
    ("split_k", r"^(SPLIT_K_[A-Z0-9_]+)\s*(?::[^=\n]+)?="),
    ("split_k", r"\bdef\s+(_?\w*split_k_(?:signal|semaphore)\w*)\s*\("),
    # The kernel side of that combine: the helpers that initialise the output and wait on it, and the
    # `if const_expr(IS_SPLIT_K):` blocks that hold it. The host constants above say a combine
    # exists; these lines are how it is written, and `end_line` carries each block's extent.
    ("split_k", r"\bdef\s+(zero_c|\w*split_k_barrier\w*)\s*\("),
    ("split_k", r"^[ \t]*if\s+const_expr\(\s*(IS_SPLIT_K)\s*\)\s*:"),
    # Accumulating into the output with atomics. Split-K partials and MoE top-k scatter both combine
    # this way, and it fixes two things a store does not: the output must hold its initial value
    # before the first add, and the rounding order is whatever order the workgroups arrive in.
    ("atomic_combine", r"\bllvm\.AtomicRMWOp\(\s*llvm\.AtomicBinOp\.(\w+)"),
    ("atomic_combine", r"\btl\.atomic_add\("),
    # Workgroup-to-tile/XCD mapping: Triton's shared `remap_xcd` / `pid_grid` helpers and the FlyDSL
    # MoE kernel's `xcd_swizzle` remap. Kept apart from `persistent`, which is a different decision.
    ("grid_mapping", r"\bdef\s+(remap_xcd\w*|pid_grid\w*)\s*\("),
    ("grid_mapping", r"\b(xcd_swizzle)\s*(?::\s*int\s*)?=|\b(xcd_swizzle)\s*>\s*0"),
    ("grid_mapping", r"\b(\w*per_xcd\w*)\s*="),
    # Which scale granularity a quantized kernel consumes. A per-token/per-channel kernel and a
    # 128x128 block-scale kernel are different math contracts, and that is decided here, not by dtype.
    # `*_repr` names are Triton's kernel-name serialisers, which carry the granularity in the kernel
    # name and consume no scale.
    ("scale_operand", r"\b(?!\w*_repr\b)(\w*(?:per_token|per_tensor|per_channel|block_?scale)\w*)"
                      r"\s*=(?!=)"),
    # Where the scale meets the accumulator, which is the granularity written as arithmetic. Inside
    # the K loop (`acc += dot(a, b) * a_scale[:, None] * …`) every K block's partial is scaled before
    # it joins the fp32 accumulator: the 128x128 block-scale contract. After the loop
    # (`acc *= a_scale[:, None] * b_scale[None, :]`) one multiply finishes a per-token/per-channel
    # accumulator.
    ("scale_in_k_loop", r"(\w+)\s*\+=\s*\(?\s*(?:tl\.dot\([^)]*\)|\w+)\s*\*\s*(\w*scale\w*)"
                        r"\[\s*(?::\s*,\s*None|None\s*,\s*:)\s*\]"),
    ("scale_after_k_loop", r"(\w+)\s*\*=\s*(\w*scale\w*)\[\s*:\s*,\s*None\s*\]\s*\*\s*"
                           r"(\w*scale\w*)\[\s*None\s*,\s*:\s*\]"),
    # Gluon states its LDS layout as a constructor, which is the same bank-conflict question the
    # FlyDSL XOR16 helper answers by hand.
    ("lds_swizzle", r"\b(?:gl|ttgl)\.(SwizzledSharedLayout|PaddedSharedLayout)\s*\("),
    # The dispatcher's M granularity for tuned-config lookups (see the `lds_pad` note above).
    ("dispatch_padding", r"\b(padded_[mnk])\s*[=:]\s*([0-9]+)"),
    # The lookup itself: which shipped config a shape is handed.
    ("config_lookup", r"\bdef\s+(get_gemm_config)\s*\("),
    # Hand-written HIP: MFMA builtins, the declared workgroup size, the skinny family's tiling knobs
    # (one record per kernel template) and the size test that picks a skinny variant.
    ("mfma_intrinsic", r"\b__builtin_amdgcn_(mfma_[a-z0-9_]+)\s*\("),
    ("waves", r"__launch_bounds__\s*\(\s*([^)]+?)\s*\)"),
    ("tunable_param", r"template\s*<([^<>]*\bYTILE\b[^<>]*)>"),
    ("kernel_family", r"\bif\s*\(\s*\(?\s*(K_in\s*\*\s*N_in\s*<=\s*[^)&]+?)\s*\)"),
    # Sizing the grid against the CU count. AITER's ASM launchers pick tile and split by the number
    # of rounds `ceil(workgroups / num_cu)` and the idle CUs in the last one; the HIP skinny launcher
    # sizes waves per group with `mindiv(rows, CuCount * YTILE, …)`. It is the grid-fill rule written
    # as code, and the same on every architecture because the CU count is read at run time.
    ("grid_fill", r"\b(__wvPrGrp)\s*=\s*(mindiv\([^;]*\))"),
    ("grid_fill", r"\b(\w*round\w*)\s*=\s*\(\s*(\w+)\s*\+\s*(num_cu|cu_num|CuCount)\s*-\s*1\s*\)"
                  r"\s*/\s*(?:num_cu|cu_num|CuCount)"),
    ("grid_fill", r"\b(max_splitk)\s*=\s*([^;]*num_cu[^;]*);"),
    # Partial sums reduced across the lanes of a wave: how a skinny kernel splits K inside a wave.
    ("lane_reduction", r"\b(__shfl(?:_xor|_down|_up)?)\s*\("),
    # Which lane holds which MFMA fragment element: the HGEMM's A/B load and C store index math, the
    # row epilogue's `lane_div_16 * 4` row offset and the A8 kernel's 4x16 lane layout. Scaling an
    # accumulator per row or per column, or writing it back, needs exactly this mapping.
    ("fragment_layout", r"\b((?:ldmatrix|stmatrix)_[abc]_\w+_idx)\s*=\s*([^\n]+)"),
    ("fragment_layout", r"\b(lane_div_16_mul4)\s*=\s*([^\n]+)"),
    ("fragment_layout", r"\b(layout_lane\d+)\s*=\s*(fx\.make_layout\([^\n]*)"),
    ("layout_epilogue", r"\bdef\s+(default_epilog|mfma_epilog)\s*\("),
    # The A8/MoE data-movement helpers, named by width (16/8/4-byte XOR16 LDS stores, dwordx4 global
    # copies) and the preshuffled layouts they read.
    ("access_width", r"\bdef\s+((?:lds_(?:store|load)|buffer_copy|load_b)_\w+)\s*\("),
    ("layout_preshuffle", r"\bdef\s+(make_preshuffle_\w+_layout)\s*\("),
    # Host contract: the launch grid (which block index is which tile, which is the K split), the
    # compiled-kernel cache and the kernel-name grammar a tuned row is parsed with.
    ("grid_mapping", r"\.launch\(\s*grid\s*=\s*\(([^)]*)\)"),
    ("jit_cache", r"@functools\.lru_cache\(\s*(?:maxsize\s*=\s*)?(\w*)\s*\)\s*\n\s*def\s+(\w+)"),
    ("instance_name", r"^(_\w*KERNEL_RE)\s*=\s*re\.compile\("),
    # The gate an implementation's own test applies against its reference, including the cases it
    # relaxes — which is where the implementation says its numerics drift.
    ("parity_tolerance", r"^(DEFAULT_(?:ATOL|RTOL|PASS_PCT))\s*=\s*([0-9.eE+-]+)"),
    ("parity_tolerance", r"[\"'](pass_pct|max_delta_limit)[\"']\s*:\s*([0-9.eE+-]+)"),
    # The grouped/MoE tests state their gate differently from the dense ones, and reading only the
    # dense idiom above would present the MoE gate as missing rather than as a different shape:
    # the pass/fail criterion there is a COSINE distance constant, with the elementwise comparison
    # demoted to an informational print. Captured as three separate idioms because they are three
    # separate decisions — the named constant, the inequality it is used in, and the per-call
    # override of the shared `checkAllclose` defaults (an explicit `atol=0` is how the routing and
    # sorting outputs declare they are exact, not approximate).
    # `[^\S\n]*`, not `\s*`: under MULTILINE `\s` still matches the newline, so `^\s*` happily
    # starts the match on an earlier blank line — or on a comment line, which `strip_comments` has
    # turned into spaces — and reports that line instead. That is exactly the three-lines-off record
    # `strip_comments`'s docstring says is worse than no record at all.
    ("parity_tolerance", r"^[^\S\n]*(\w*(?:LOGITS_DIFF_TOL|VERIFY_TOL\w*|logits_diff_tol))[^\S\n]*=\s*([0-9.eE+-]+)"),
    ("parity_tolerance", r"\b(logits_diff)\s*>\s*([0-9.eE+-]+)"),
    ("parity_tolerance", r"\bcheckAllclose\([^)]*?\b(atol|rtol|tol_err_ratio)\s*=\s*([0-9.eE+-]+)"),
    ("parity_tolerance", r"\bdef\s+(_check_close)\s*\(([^)]*)\)"),
    # opus's buffer access and wait builtins — the HIP spelling of the loads FlyDSL issues.
    ("access_width", r"\b__builtin_amdgcn_raw_buffer_(load|store)_(\w+)\s*\("),
    ("scheduling", r"\b__builtin_amdgcn_(s_waitcnt\w*|s_barrier|sched_group_barrier|sched_barrier|"
                   r"s_setprio)\s*\("),
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


PERF_CLAIM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:us|ms|ns|µs|TFLOPS|tflops|GB/s|gbps)\b|"
                        r"\b(?:speedup|faster|slower)\b", re.IGNORECASE)


def quotable(line, anchor=False):
    """Whether a source line may be quoted as context around a fact.

    Blank lines say nothing. A comment such as "cross this boundary instead is ~1.5% slower" says
    something, but it is an upstream timing with no machine, shape or date attached: quoting it
    inside the corpus would hand a card author a performance number dressed as evidence, which is
    exactly the unfalsifiable claim the corpus exists to replace with measurement. The construct the
    comment sits above is still a fact, so the record stays and only the claim is left out — the
    gap in the quoted line numbers shows where. An anchor line is never dropped: it IS the fact, and
    a perf claim on it is a conflict worth surfacing rather than hiding.
    """
    return bool(line.strip()) and (anchor or not PERF_CLAIM.search(line))


def excerpt(lines, idx, span=2):
    lo, hi = max(0, idx - span), min(len(lines), idx + span + 1)
    return [f"{n + 1}: {lines[n].rstrip()}"
            for n in range(lo, hi) if quotable(lines[n], anchor=n == idx)]


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


def fact_identity(record):
    """Identity of the STATEMENT, deliberately blind to where it is written.

    `source_evidence_id` is bound to `file`/`line` because a citation must stop resolving when the
    cited line changes. That is right for a citation and useless for the question a refresh actually
    raises, which is whether the fact survived: the FlyDSL GEMM rewrite moved `swizzle_xor16` out of
    `splitk_hgemm.py` into `mfma_preshuffle_pipeline.py`, and under the citation identity that reads
    as 1227 dead records when 817 of them are the same sentence one directory over.

    So the two live side by side. Dropping `file`, `line` and `evidence` (whose excerpt carries line
    numbers) leaves what the record asserts, which is what lets a relocated fact be found again.
    Equal fact_ids are the same statement, not necessarily the same occurrence — a tile shape used in
    nine kernels is one fact and nine citations, and the renderer already treats it that way.
    """
    fields = {key: record.get(key)
              for key in ("category", "language", "match", "captured", "arch_scope")}
    canonical = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "fact_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def statement_ends(raw):
    """Map each line that starts a Python statement to the last line of that statement.

    A record is one line plus a two-line excerpt, which is enough for a constant and not for a
    mechanism: a combine written as a 50-line `if` block, or a helper `def`, needs its extent for a
    reader to know where to stop. The AST gives it exactly. Only `.py` has a parser here; a file that
    does not parse simply gets no extents.
    """
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return {}
    ends = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.stmt) and getattr(node, "end_lineno", None):
            ends[node.lineno] = max(ends.get(node.lineno, 0), node.end_lineno)
    return ends


INVENTORY_KEYS = {
    "tile_m": "tile_m", "tilem": "tile_m", "tile_n": "tile_n", "tilen": "tile_n",
    "splitk": "split_k_capable", "bpreshuffle": "bpreshuffle", "pf": "prefetch", "tn": "tn",
    "subk": "sub_k", "bias": "bias",
}
GFX_DIR = re.compile(r"(?:^|/)(gfx[0-9]+[a-z]?)/")


def inventory_records(raw, rel, language):
    """One record per kernel row of an ASM inventory CSV (`hsa/<gfx>/<kind>/*.csv`).

    Read by header, not by position: the kinds disagree on column order and spelling (`tileM`,
    `tile_m`, `tile_M`). `splitK` is a capability flag, not a split factor — the launchers accept a
    split above 1 only for a kernel whose flag is 1 — so it is recorded as `split_k_capable`. The
    architecture is the directory the CSV sits in.
    """
    lines = raw.splitlines()
    if not lines:
        return []
    header = [cell.strip() for cell in lines[0].split(",")]
    gfx = GFX_DIR.search(rel)
    out = []
    for idx in range(1, len(lines)):
        cells = [cell.strip() for cell in lines[idx].split(",")]
        if len(cells) != len(header):
            continue
        captured = [f"{INVENTORY_KEYS[key.lower()]}={value}"
                    for key, value in zip(header, cells) if key.lower() in INVENTORY_KEYS]
        record = {
            "category": "asm_tile",
            "language": language,
            "file": rel,
            "line": idx + 1,
            "match": lines[idx].strip(),
            "captured": captured,
            "arch_scope": gfx.group(1) if gfx else "",
            "evidence": [f"1: {lines[0].rstrip()}", f"{idx + 1}: {lines[idx].rstrip()}"],
        }
        record["evidence_id"] = source_evidence_id(record)
        out.append(record)
    return out


def route_language(name):
    tokens = name.lower().split("_")
    for token, language in ROUTE_TARGETS:
        if token in tokens:
            return language
    return None


def routing_records(path, rel):
    """Routing statements in AITER's dispatch code, each filed under the backend it routes to.

    The excerpt reaches five lines up, because the condition that selects a route sits above the
    statement that names it. No architecture is inferred: a route in the `else` of `gfx == "gfx942"`
    is the other architectures', and a nearest-literal guess would file it under gfx942. The excerpt
    shows the condition; the card that cites it states the architecture.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    view = strip_comments(raw, python=True)
    rlines = raw.splitlines()
    ends = statement_ends(raw)
    out = []
    for pattern in ROUTING_RULES:
        for m in re.finditer(pattern, view, re.MULTILINE):
            language = route_language(m.group(1))
            if language is None:
                continue
            idx = view.count("\n", 0, m.start())
            record = {"category": "kernel_family", "language": language, "file": rel,
                      "line": idx + 1}
            if ends.get(idx + 1, 0) > idx + 1:
                record["end_line"] = ends[idx + 1]
            record.update({
                "match": m.group(0).strip(),
                "captured": [m.group(1)],
                "arch_scope": "",
                "evidence": [f"{n + 1}: {rlines[n].rstrip()}"
                             for n in range(max(0, idx - 5), min(len(rlines), idx + 2))
                             if quotable(rlines[n], anchor=n == idx)],
            })
            record["evidence_id"] = source_evidence_id(record)
            out.append(record)
    return list({record["evidence_id"]: record for record in out}.values()), raw


def scan_file(path, rel, language):
    """Every rule hit in one file, as source-evidence records.

    A hit answers only "what is written here?". It is not yet a development decision: deciding
    whether to copy, benchmark or reject the pattern requires conditions and evidence strength, which
    live in `decisions/gemm.yaml`.

    `end_line` is location, not identity: it is left out of `source_evidence_id`, so an edit deep
    inside a long function does not invalidate every card that cites the function's first line.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if path.endswith(".csv"):
        return inventory_records(raw, rel, language), raw, raw.splitlines()
    python = path.endswith(".py")
    view = strip_comments(raw, python=python)
    vlines, rlines = view.splitlines(), raw.splitlines()
    ends = statement_ends(raw) if python else {}
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
            }
            if ends.get(idx + 1, 0) > idx + 1:
                record["end_line"] = ends[idx + 1]
            record.update({
                "match": m.group(0).strip(),
                "captured": groups,
                "arch_scope": arch_context(vlines, idx),
                "evidence": excerpt(rlines, idx),
            })
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


TUNED_CONFIG_OPS = ("gemm", "moe")


def parse_tuned_location(rel):
    """Decode a shipped config's identity from its path, under either of AITER's two layouts.

    Up to a6bb4993 the arch and tags were in the filename and the directory was flat
    (`configs/gemm/gfx942-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json`). From 4faf8713 the tree is
    `configs/<arch>/<backend>/<op>/<dtype>/DEFAULT.json`, the filename carries no arch at all, and
    `configs/gemm/` does not exist — so a reader of the old constant sees an empty directory and
    concludes AITER ships no tuned Triton configs, when it ships 700 of them.

    `<dtype>` is `config_name.lower().replace("-", "_")`, which is lossy: `GEMM-A8W8_BLOCKSCALE` and
    `GEMM_A8W8_BLOCKSCALE` both land on `gemm_a8w8_blockscale`, so it is kept as ONE tag rather than
    split back into the two the old form had. Splitting would be a guess, and a tag list that looks
    like the old one while being reconstructed differently is worse than one that plainly differs.
    """
    parts = rel.replace(os.sep, "/").split("/")
    flat = parse_tuned_name(parts[-1])
    if flat is not None:
        return flat
    if len(parts) < 5 or not parts[0].startswith("gfx") or parts[2] not in TUNED_CONFIG_OPS:
        return None
    arch, backend, op, dtype = parts[0], parts[1], parts[2], parts[3]
    stem = parts[-1][:-len(".json")]
    tags, shape = [backend, op.upper(), dtype.upper()], {}
    for seg in ([] if stem == "DEFAULT" else stem.split("-")):
        m = TUNED_SHAPE_SEG.match(seg)
        if m:
            shape[m.group("dim").lower()] = int(m.group("val"))
        else:
            tags.append(seg)
    return {"gfx": arch, "tags": tags, "shape": shape}


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
    # Walked rather than listed: the current layout is four directories deep, and `os.listdir` on
    # the root of it returns nine `gfx*` directories and no configs at all.
    for dirpath, dirs, names in os.walk(base):
        dirs.sort()
        rel_dir = os.path.relpath(dirpath, base)
        # An attention or conv config is not this corpus's subject. Skipping it by path keeps it out
        # of `unparsed`, which is for files that LOOK like ours and could not be read — a gap list
        # padded with out-of-scope filenames stops being read.
        seg = rel_dir.replace(os.sep, "/").split("/")
        if len(seg) >= 3 and seg[0].startswith("gfx") and seg[2] not in TUNED_CONFIG_OPS:
            dirs[:] = []
            continue
        for name in sorted(names):
            if not name.endswith(".json"):
                continue
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            meta = parse_tuned_location(rel)
            if meta is None:
                unparsed.append(rel)
                continue
            try:
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, ValueError):
                unparsed.append(rel)
                continue
            if not isinstance(data, dict):
                unparsed.append(rel)
                continue
            for bucket, knobs in data.items():
                if not isinstance(knobs, dict):
                    continue
                out.append({"file": os.path.join(TUNED_CONFIG_DIR, rel),
                            "gfx": meta["gfx"], "tags": meta["tags"], "shape": meta["shape"],
                            "m_bucket": str(bucket),
                            "knobs": {k: v for k, v in sorted(knobs.items())}})
    gap = []
    if unparsed:
        gap.append({"language": "triton", "subtree": TUNED_CONFIG_DIR,
                    "why": f"{len(unparsed)} config file(s) in neither the flat `gfx…-TAG…[-N=..]` "
                           f"form nor `<arch>/<backend>/<op>/<dtype>/`, skipped rather than guessed: "
                           f"{', '.join(sorted(unparsed)[:4])}"})
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


# ---------------------------------------------------------------------------------------------
# AITER's tuned GEMM databases: `aiter/configs/**/*_tuned_*gemm*.csv`
# ---------------------------------------------------------------------------------------------
# The Triton JSON above records one backend's selected knobs. The CSV databases record something the
# JSON cannot: for each shipped (gfx, CU count, M, N, K) AITER raced its backends — asm, ck, cktile,
# flydsl, triton — and kept one. That is the only place in AITER where "which implementation for
# which shape regime" is written down, and it is where the FlyDSL configurations AITER actually ships
# live. What is kept is the selection; what is dropped on read is how long anything took.
TUNED_DB_DIRS = ("aiter/configs", "aiter/configs/model_configs")
TUNED_DB_KIND = re.compile(
    r"(?P<kind>(?:bf16|a8w8|a4w4)(?:_blockscale)?(?:_bpreshuffle)?_tuned_(?:batched_)?gemm)")
# Timings from a box this corpus cannot name. The always-on layer carries no performance numbers, so
# these columns never enter a record.
TUNED_DB_TIMING = ("bw", "errRatio", "err_ratio", "tflops", "us")
TUNED_DB_CONTEXT = ("bias", "bpreshuffle", "dtype", "outdtype", "q_dtype_w", "scaleAB")
M_BUCKETS = ((16, "M<=16"), (64, "M17-64"), (256, "M65-256"), (1024, "M257-1024"),
             (4096, "M1025-4096"), (None, "M>4096"))
M_BUCKET_ORDER = {label: i for i, (_, label) in enumerate(M_BUCKETS)}
# AITER's two serialisations of a FlyDSL configuration: `gemm_kernels.flydsl_kernel_name` for the
# split-K/small-M HGEMM, and the A8 tune `kernelInstance.name` (upstream spells it `bpreshuflle`).
FLYDSL_HGEMM_NAME = re.compile(
    r"^flydsl_gemm(?P<stage>\d+)_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_split_k(?P<split_k>\d+)_"
    r"block_m_warp(?P<block_m_warps>\d+)_block_n_warp(?P<block_n_warps>\d+)_"
    r"async_copy(?P<async_copy>True|False)_b_to_lds(?P<b_to_lds>True|False)_"
    r"b_preshuffle(?P<b_preshuffle>True|False)_c_to_lds(?P<c_to_lds>True|False)"
    r"(?P<small_m>_small_m(?:_nr(?P<n_tile_repeat>\d+))?(?:_pn(?P<persistent_n_tiles>\d+))?"
    r"(?:_wpe(?P<waves_per_eu>\d+))?(?:_ur(?P<b_to_lds_unroll>\d+))?)?"
    r"_(?P<target_gfx>gfx[0-9a-z]+)$")
FLYDSL_A8_NAME = re.compile(
    r"^flydsl_bpreshu(?:ffle|flle)_(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"(?P<a_dtype>[A-Z0-9]+)_(?P<w_dtype>[A-Z0-9]+)_(?P<out_dtype>[A-Z0-9]+)_"
    r"(?P<lds_stage>\d+)x(?P<use_cshuffle_epilog>\d+)x(?P<use_async_copy>\d+)x(?P<waves_per_eu>\d+)_"
    r"(?P<scheduler>\w+)$")


def m_bucket(m):
    for bound, label in M_BUCKETS:
        if bound is None or m <= bound:
            return label
    return M_BUCKETS[-1][1]


def decode_flydsl_kernel(name):
    """`(family, dtype, knobs)` encoded in a shipped FlyDSL kernel name, or None.

    The names are AITER's own serialisation of the configuration, so this reads a configuration
    rather than guessing one. `tile` and `warps` stay single knobs: their parts vary together, and
    three independent spreads for tile_m/n/k would advertise combinations nobody selected.
    """
    m = FLYDSL_HGEMM_NAME.match(name or "")
    if m:
        g = m.groupdict()
        knobs = {
            "tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}",
            "split_k": int(g["split_k"]),
            "warps": f"{g['block_m_warps']}x{g['block_n_warps']}",
            "b_to_lds": g["b_to_lds"] == "True",
            "b_preshuffle": g["b_preshuffle"] == "True",
            "async_copy": g["async_copy"] == "True",
            "c_to_lds": g["c_to_lds"] == "True",
            "stages": int(g["stage"]),
        }
        for key in ("n_tile_repeat", "persistent_n_tiles", "waves_per_eu", "b_to_lds_unroll"):
            if g.get(key):
                knobs[key] = int(g[key])
        family = "small_m" if g["small_m"] else "hgemm"
        return family, f"a{g['a_dtype']}_w{g['w_dtype']}_{g['out_dtype']}", knobs
    m = FLYDSL_A8_NAME.match(name or "")
    if m:
        g = m.groupdict()
        knobs = {
            "tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}",
            "lds_stage": int(g["lds_stage"]),
            "use_cshuffle_epilog": int(g["use_cshuffle_epilog"]),
            "use_async_copy": int(g["use_async_copy"]),
            "waves_per_eu": int(g["waves_per_eu"]),
            "scheduler": g["scheduler"].lower(),
        }
        dtype = f"a{g['a_dtype'].lower()}_w{g['w_dtype'].lower()}_{g['out_dtype'].lower()}"
        return "a8_preshuffle", dtype, knobs
    return None


def tuned_db_rows(aiter):
    """Rows of AITER's tuned GEMM databases, without their timing columns.

    Kept: what makes a row comparable (gfx, CU count, M/N/K, the dtype/bias/preshuffle key) and what
    was selected (backend, kernel name). A database without a backend column is inventoried and
    reported as a gap rather than attributed to a backend by guessing from kernel-name prefixes.
    """
    rows, inventory, gaps = [], [], []
    for sub in TUNED_DB_DIRS:
        base = os.path.join(aiter, sub)
        if not os.path.isdir(base):
            gaps.append({"language": "aiter_tuned_db", "subtree": sub, "why": "no such directory"})
            continue
        for name in sorted(os.listdir(base)):
            if not name.endswith(".csv") or "untuned" in name or "fmoe" in name:
                continue
            kind = TUNED_DB_KIND.search(name)
            if not kind:
                continue
            rel = os.path.join(sub, name)
            with open(os.path.join(base, name), encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                columns = list(reader.fieldnames or [])
                file_rows = list(reader)
            has_backend = "libtype" in columns
            inventory.append({
                "file": rel, "db_kind": kind.group("kind"), "rows": len(file_rows),
                "backend_column": has_backend,
                "gfx": sorted({r["gfx"] for r in file_rows if r.get("gfx")}),
            })
            if file_rows and not has_backend:
                gaps.append({"language": "aiter_tuned_db", "subtree": rel,
                             "why": "no `libtype` column: the selected backend is not recorded, so "
                                    "these rows are inventoried but not counted as a selection"})
            for r in file_rows:
                try:
                    m, n, k, cu = int(r["M"]), int(r["N"]), int(r["K"]), int(r["cu_num"])
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({
                    "file": rel, "db_kind": kind.group("kind"), "gfx": r.get("gfx") or "",
                    "cu_num": cu, "m": m, "n": n, "k": k,
                    "libtype": (r.get("libtype") or None) if has_backend else None,
                    "kernel_name": r.get("kernelName") or "",
                    "context": {c: r[c] for c in TUNED_DB_CONTEXT if c in r},
                })
    return rows, inventory, gaps


def _short_id(prefix, parts):
    return prefix + hashlib.sha256(
        json.dumps(parts, separators=(",", ":")).encode()).hexdigest()[:16]


def _value_text(value):
    return "true" if value is True else "false" if value is False else str(value)


def knob_spread(members):
    """`(fixed, varies)` over a group's knob dicts, counting absence as a value.

    Not every knob appears on every member: a shipped name carries a flag only when the tuner set
    it, so `kw` present on one of two members is a knob that MOVED, and reporting it as `kw: 2`
    would read as "the group agreed on 2" when half the group did not ask for it at all. `absent`
    is spelled out for the same reason a gap is reported rather than dropped.
    """
    spread = collections.defaultdict(collections.Counter)
    for knobs in members:
        for knob, value in knobs.items():
            spread[knob][json.dumps(value)] += 1
    fixed, varies = {}, {}
    for knob, counter in sorted(spread.items()):
        if len(counter) == 1 and sum(counter.values()) == len(members):
            fixed[knob] = json.loads(next(iter(counter)))
            continue
        parts = [f"{_value_text(json.loads(v))}: {n}" for v, n in counter.most_common()]
        gap = len(members) - sum(counter.values())
        if gap:
            parts.append(f"absent: {gap}")
        varies[knob] = ", ".join(parts)
    return fixed, varies


def summarize_flydsl_tuned(rows):
    """FlyDSL selections grouped by (gfx, CUs, database, family, dtype, M bucket).

    Same contract as `summarize_tuned`: a knob every selected row in the group shares is a seed, a
    knob that moved with the shape is one to vary. Nothing here says a selection beat anything by a
    margin — the database keeps the winner, not the race.
    """
    groups = collections.defaultdict(list)
    unparsed = collections.Counter()
    for r in rows:
        if r["libtype"] != "flydsl":
            continue
        decoded = decode_flydsl_kernel(r["kernel_name"])
        if decoded is None:
            unparsed[r["file"]] += 1
            continue
        family, dtype, knobs = decoded
        groups[(r["gfx"], r["cu_num"], r["db_kind"], family, dtype, m_bucket(r["m"]))].append(
            (r, knobs))
    out = []
    for key in sorted(groups, key=lambda t: (t[0], t[1], t[2], t[3], t[4], M_BUCKET_ORDER[t[5]])):
        gfx, cu, kind, family, dtype, bucket = key
        members = groups[key]
        fixed, varies = knob_spread([knobs for _, knobs in members])
        rec = {
            "config_id": _short_id("cfg_", ["aiter_tuned_db", gfx, cu, kind, family, dtype, bucket]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "family": family, "dtype": dtype,
            "m_bucket": bucket, "shape_configs": len(members),
            "source_files": sorted({r["file"] for r, _ in members}),
        }
        if len(members) == 1:
            rec["knobs"] = fixed
        else:
            rec["same_across_configs"] = fixed
            rec["varies_by_shape"] = varies
        out.append(rec)
    gaps = [{"language": "aiter_tuned_db", "subtree": path,
             "why": f"{n} flydsl kernelName(s) in neither shipped naming form; skipped, not guessed"}
            for path, n in sorted(unparsed.items())]
    return out, gaps


def summarize_backend_selection(rows):
    """How many shipped rows AITER's tuner assigned to each backend, per (gfx, CUs, database, M).

    A count of selections, not of speedups. A shape shipped in two model databases counts twice,
    and the competing set, tuner box and toolchain are not recorded — the counts say where each
    backend was chosen, never by how much.
    """
    groups = collections.defaultdict(collections.Counter)
    files = collections.defaultdict(set)
    for r in rows:
        if not r["libtype"]:
            continue
        key = (r["gfx"], r["cu_num"], r["db_kind"], m_bucket(r["m"]))
        groups[key][r["libtype"]] += 1
        files[key].add(r["file"])
    out = []
    for key in sorted(groups, key=lambda t: (t[0], t[1], t[2], M_BUCKET_ORDER[t[3]])):
        gfx, cu, kind, bucket = key
        out.append({
            "selection_id": _short_id("sel_", ["aiter_tuned_db", gfx, cu, kind, bucket]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "m_bucket": bucket,
            "rows": sum(groups[key].values()),
            "selected_backend_rows": dict(sorted(groups[key].items())),
            "source_files": sorted(files[key]),
        })
    return out


# --------------------------------------------------------------------------------------------
# Fused-MoE tuned databases
#
# A separate reader rather than a wider `TUNED_DB_KIND`, because these tables are not keyed on a
# GEMM shape at all. A dense row says (M, N, K); a fused-MoE row says (token, model_dim, inter_dim,
# expert, topk) and then names TWO kernels, one per stage. Folding them into the dense reader would
# have required inventing an M/N/K for each row, which is the kind of guess the rest of this file
# refuses to make.
# --------------------------------------------------------------------------------------------
FMOE_DB_KIND = re.compile(r"(?P<kind>tuned_(?:grouped_)?fmoe)")
FMOE_SHAPE_KEYS = ("model_dim", "inter_dim", "expert", "topk")
FMOE_QUANT_KEYS = ("act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type",
                   "use_g1u1", "doweight_stage1")
# Timings from a box this corpus cannot name, plus the accuracy columns of the same run, plus the
# model tag, which names the sweep rather than the configuration.
FMOE_DROP = ("bw", "err1", "err2", "max_abs", "rel_l2", "tflops", "us", "us1", "us2", "_tag")
FMOE_KEYS = ("gfx", "cu_num", "token", "kernelName", "kernelName1", "kernelName2")
# AITER's own serialisations of a fused-MoE configuration. Three forms ship, and the per-stage knobs
# live INSIDE the name, so a decoded name is a read configuration rather than an inferred one. The
# flag tokens are kept under AITER's own spelling (`w3`, `xcd4`, `sbm32`, `atomic`, `nt`) — a
# friendlier name would be my interpretation of a token I cannot check against a definition.
FLYDSL_MOE_NAME = re.compile(
    r"^flydsl_moe(?P<stage>\d+)(?P<layout>_layout)?_"
    r"a(?P<a_dtype>[a-z0-9]+)_w(?P<w_dtype>[a-z0-9]+)_(?P<out_dtype>[a-z0-9]+)_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)(?P<flags>(?:_[a-z0-9]+)*)$")
FLYDSL_MXMOE_NAME = re.compile(
    r"^flydsl_mxmoe_g(?P<stage>\d+)_(?P<dtype>a\d+w\d+)_"
    r"(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)(?P<flags>(?:_[a-z0-9]+)*)$")
FLYDSL_GROUPED_NAME = re.compile(
    r"^grouped_gemm(?P<stage>\d+)_(?P<dtype>[a-z0-9]+)(?P<flags>(?:_[a-z0-9]+)*)$")
FLAG_WITH_COUNT = re.compile(r"^(?P<knob>[a-z]+)(?P<count>\d+)$")
# Which project generated a name, read off the serialisation each one emits: CK's generated instance
# name, the mangled symbol of an aiter assembly kernel, FlyDSL's own name builder. A prefix that is
# none of these is left unattributed and counted as a gap — `opus`, `impl`, `cktile` and `swiglu`
# names ship here too, and a fourth guess would be exactly that.
FMOE_BACKEND_PREFIXES = (("moe_ck2stages", "ck"), ("_ZN5aiter", "asm"),
                         ("flydsl_", "flydsl"), ("grouped_gemm", "flydsl"))


def fmoe_backend(name):
    for prefix, backend in FMOE_BACKEND_PREFIXES:
        if (name or "").startswith(prefix):
            return backend
    return None


def decode_flydsl_moe_kernel(name):
    """`(family, dtype, knobs)` for a shipped FlyDSL fused-MoE kernel name, or None."""
    m = FLYDSL_MOE_NAME.match(name or "")
    if m:
        g = m.groupdict()
        family = f"moe_stage{g['stage']}" + ("_layout" if g["layout"] else "")
        dtype = f"a{g['a_dtype']}_w{g['w_dtype']}_{g['out_dtype']}"
        knobs = {"tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}"}
    else:
        m = FLYDSL_MXMOE_NAME.match(name or "")
        if m:
            g = m.groupdict()
            family = f"mxmoe_stage{g['stage']}"
            dtype = g["dtype"]
            knobs = {"tile": f"{g['tile_m']}x{g['tile_n']}x{g['tile_k']}"}
        else:
            m = FLYDSL_GROUPED_NAME.match(name or "")
            if not m:
                return None
            g = m.groupdict()
            family, dtype, knobs = f"grouped_stage{g['stage']}", g["dtype"], {}
    tags = []
    for token in [t for t in g["flags"].split("_") if t]:
        counted = FLAG_WITH_COUNT.match(token)
        if counted:
            knobs[counted.group("knob")] = int(counted.group("count"))
        else:
            tags.append(token)
    if tags:
        knobs["tags"] = ",".join(sorted(tags))
    return family, dtype, knobs


def fmoe_db_rows(aiter):
    """Rows of AITER's fused-MoE tuned databases, without their timings.

    One record per CSV row, carrying both stages: the stage-1 and stage-2 kernel names and every
    launcher column the sweep wrote. Launcher columns are kept under AITER's own names, including
    its `tile_m` / `tile_m2` stage spelling — re-keying them into a stage-1/stage-2 structure would
    mean deciding that `num_buffer_stage2` pairs with `num_buffers`, which is a reading of the
    schema and not something the schema states.

    `m` is `token * topk`, the row count the stage-1 grouped GEMM actually sees, so a fused-MoE
    record buckets on the same axis as a dense one. `token` is kept beside it: the two differ by a
    factor of topk, and only one of them is a GEMM dimension.
    """
    rows, inventory, gaps = [], [], []
    for sub in TUNED_DB_DIRS:
        base = os.path.join(aiter, sub)
        if not os.path.isdir(base):
            continue  # already reported by `tuned_db_rows`
        for name in sorted(os.listdir(base)):
            if not name.endswith(".csv") or "untuned" in name:
                continue
            kind = FMOE_DB_KIND.search(name)
            if not kind:
                continue
            rel = os.path.join(sub, name)
            with open(os.path.join(base, name), encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                columns = list(reader.fieldnames or [])
                file_rows = list(reader)
            launch = [c for c in columns
                      if c not in FMOE_DROP and c not in FMOE_KEYS
                      and c not in FMOE_SHAPE_KEYS and c not in FMOE_QUANT_KEYS]
            unattributed = collections.Counter()
            inventory.append({
                "file": rel, "db_kind": kind.group("kind"), "rows": len(file_rows),
                "backend_column": False, "stages": 2,
                "gfx": sorted({r["gfx"] for r in file_rows if r.get("gfx")}),
            })
            for r in file_rows:
                try:
                    shape = {k: int(r[k]) for k in FMOE_SHAPE_KEYS}
                    token, cu = int(r["token"]), int(r["cu_num"])
                except (KeyError, TypeError, ValueError):
                    continue
                backend = fmoe_backend(r.get("kernelName1"))
                if backend is None:
                    unattributed[(r.get("kernelName1") or "").split("_")[0]] += 1
                rows.append({
                    "file": rel, "db_kind": kind.group("kind"), "gfx": r.get("gfx") or "",
                    "cu_num": cu, "token": token, "m": token * max(shape["topk"], 1),
                    "shape": shape, "libtype": backend,
                    "kernel_stage1": r.get("kernelName1") or "",
                    "kernel_stage2": r.get("kernelName2") or "",
                    "quant": {k: r[k] for k in FMOE_QUANT_KEYS if r.get(k) not in (None, "")},
                    "launch": {c: r[c] for c in launch if r.get(c) not in (None, "")},
                })
            for prefix, n in sorted(unattributed.items()):
                gaps.append({"language": "aiter_tuned_db", "subtree": rel,
                             "why": f"{n} row(s) name a `{prefix}…` stage-1 kernel, which is none of "
                                    "the shipped naming forms; the backend is left unattributed"})
    return rows, inventory, gaps


def summarize_fmoe_tuned(rows):
    """FlyDSL fused-MoE selections grouped by (gfx, CUs, database, family, dtype, MoE shape, M).

    The MoE shape is part of the key, not of the spread. `inter_dim` and `expert` change which
    kernel is reasonable rather than how it is configured, so a group that mixed two models would
    report "tile varies" where the truth is "these are two different problems".

    Within a group the members are the token counts of one M bucket, so a fixed knob is one the
    sweep kept across the whole bucket and a varying one is a knob that tracks the token count.
    """
    groups = collections.defaultdict(list)
    unparsed = collections.Counter()
    for r in rows:
        if r["libtype"] != "flydsl":
            continue
        decoded = decode_flydsl_moe_kernel(r["kernel_stage1"])
        if decoded is None:
            unparsed[r["file"]] += 1
            continue
        family, dtype, knobs = decoded
        knobs = {f"stage1.{k}": v for k, v in knobs.items()}
        stage2 = decode_flydsl_moe_kernel(r["kernel_stage2"])
        if stage2:
            knobs["stage2.family"] = stage2[0]
            knobs.update({f"stage2.{k}": v for k, v in stage2[2].items()})
        knobs.update(r["launch"])
        shape = "x".join(str(r["shape"][k]) for k in FMOE_SHAPE_KEYS)
        groups[(r["gfx"], r["cu_num"], r["db_kind"], family, dtype, shape,
                m_bucket(r["m"]))].append((r, knobs))
    out = []
    for key in sorted(groups, key=lambda t: (t[:6], M_BUCKET_ORDER[t[6]])):
        gfx, cu, kind, family, dtype, shape, bucket = key
        members = groups[key]
        fixed, varies = knob_spread([knobs for _, knobs in members])
        rec = {
            "config_id": _short_id("cfg_", ["aiter_fmoe_db", gfx, cu, kind, family, dtype,
                                            shape, bucket]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "family": family, "dtype": dtype,
            "moe_shape": dict(zip(FMOE_SHAPE_KEYS, (int(v) for v in shape.split("x")))),
            "m_bucket": bucket, "shape_configs": len(members),
            "tokens": sorted({r["token"] for r, _ in members}),
            "source_files": sorted({r["file"] for r, _ in members}),
        }
        if len(members) == 1:
            rec["knobs"] = fixed
        else:
            rec["same_across_configs"] = fixed
            rec["varies_by_shape"] = varies
        out.append(rec)
    gaps = [{"language": "aiter_tuned_db", "subtree": path,
             "why": f"{n} flydsl fused-MoE kernelName(s) in no shipped naming form; "
                    "skipped, not guessed"}
            for path, n in sorted(unparsed.items())]
    return out, gaps


def summarize_fmoe_backend(rows):
    """Which project's kernel the fused-MoE sweep kept, per (gfx, CUs, database, MoE shape, M).

    Counted per stage, because the two stages are tuned independently and are not always won by the
    same project: a row can keep a FlyDSL gate+up and a CK down-projection. A single count keyed on
    stage 1 would have reported that row as wholly FlyDSL.
    """
    groups = collections.defaultdict(collections.Counter)
    files = collections.defaultdict(set)
    for r in rows:
        shape = "x".join(str(r["shape"][k]) for k in FMOE_SHAPE_KEYS)
        for stage, column in ((1, "kernel_stage1"), (2, "kernel_stage2")):
            backend = fmoe_backend(r[column])
            if backend is None:
                continue
            key = (r["gfx"], r["cu_num"], r["db_kind"], shape, m_bucket(r["m"]), stage)
            groups[key][backend] += 1
            files[key].add(r["file"])
    out = []
    for key in sorted(groups, key=lambda t: (t[:4], M_BUCKET_ORDER[t[4]], t[5])):
        gfx, cu, kind, shape, bucket, stage = key
        out.append({
            "selection_id": _short_id("sel_", ["aiter_fmoe_db", gfx, cu, kind, shape,
                                               bucket, stage]),
            "gfx": gfx, "cu_num": cu, "db_kind": kind, "m_bucket": bucket, "stage": stage,
            "moe_shape": dict(zip(FMOE_SHAPE_KEYS, (int(v) for v in shape.split("x")))),
            "rows": sum(groups[key].values()),
            "selected_backend_rows": dict(sorted(groups[key].items())),
            "source_files": sorted(files[key]),
        })
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
        files, empty_patterns = [], []
        for pat in patterns:
            hits = [f for f in sorted(globmod.glob(os.path.join(base, pat), recursive=True))
                    if "__pycache__" not in f and os.path.isfile(f)]
            if not hits:
                empty_patterns.append(pat)
            files += hits
        if patterns and not files:
            missing.append({"language": language, "subtree": subtree,
                            "why": f"no file matched {list(patterns)}"})
            continue
        # Reported per pattern as well. A language with one matching file used to hide every other
        # pattern that matched nothing: HIP kept `gemm_common.cu`, so the whole `opus_gemm/` tree
        # it was declared to read could vanish upstream without a line in `missing`.
        for pat in empty_patterns:
            missing.append({"language": language, "subtree": subtree,
                            "why": f"declared pattern matched no file: {pat}"})
        for path in files:
            rel = os.path.relpath(path, aiter)
            got, raw, _ = scan_file(path, rel, language)
            evidence += got
            if language in ("triton", "gluon") and path.endswith(".py"):
                configs += [dict(c, language=language) for c in autotune_configs(path, rel)]
            seen.append({"language": language, "file": rel, "lines": raw.count("\n") + 1,
                         "evidence_records": len(got)})
            # A file that matched a declared pattern and yielded nothing is a gap in the RULES, not
            # a fact about the file, and it used to be invisible: `missing` only reported patterns
            # that matched NO file, so a declared-and-read file could contribute zero evidence
            # forever without a line anywhere. That is the worst kind of gap, because coverage looks
            # complete. Reported only above a size floor — a 40-line forwarding header legitimately
            # states nothing, and listing those would bury the real cases in noise.
            if not got and raw.count("\n") + 1 >= 120:
                missing.append({"language": language, "subtree": subtree,
                                "why": f"indexed but no rule matched: {rel} "
                                       f"({raw.count(chr(10)) + 1} lines)"})
    for rel in ROUTING_FILES:
        path = os.path.join(aiter, rel)
        if not os.path.isfile(path):
            missing.append({"language": "routing", "subtree": os.path.dirname(rel),
                            "why": f"declared dispatch file not found: {rel}"})
            continue
        got, raw = routing_records(path, rel)
        evidence += got
        seen.append({"language": "routing", "file": rel, "lines": raw.count("\n") + 1,
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


MERGE_BOOKKEEPING = ("aiter_commits", "retuned_at", "relocated_to")


def merge_by_identity(store, order, rows, key, commit):
    """Fold one tree's rows into the cross-tree table, keyed by `key`.

    Two different kinds of key arrive here and they must not be treated alike. `evidence_id` hashes
    the CONTENT, so a repeat across trees is literally the same statement and the only new fact is
    that it also holds at another commit. `config_id` and `selection_id` hash a LOOKUP KEY — (gfx,
    tags, M bucket), (gfx, CU, shape) — and AITER retunes those tables: 36 of the 41 shared
    backend-selection keys carry a different winner at the two commits indexed here.

    Appending the older commit to such a row would assert an agreement that does not exist, and
    dropping the row would lose the fact that the answer changed. So the primary tree's answer is
    kept, and the commits where the same key held a different one are named in `retuned_at` — which
    is also the signal a card quoting a shipped number needs, because that number has since moved.
    """
    def body(row):
        return json.dumps({k: v for k, v in row.items() if k not in MERGE_BOOKKEEPING},
                          ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    for row in rows:
        prev = store.get(row[key])
        if prev is None:
            row["aiter_commits"] = [commit]
            store[row[key]] = row
            order.append(row[key])
        elif commit in prev["aiter_commits"] or commit in prev.get("retuned_at", []):
            continue
        elif body(prev) == body(row):
            prev["aiter_commits"].append(commit)
        else:
            prev.setdefault("retuned_at", []).append(commit)


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
    for section in ("coverage", "inventory", "missing", "autotune_configs", "tuned_configs",
                    "backend_selection", "source_evidence"):
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
    ap.add_argument("--aiter", action="append", metavar="PATH",
                    help="an aiter checkout to read (required for --emit / --json). Repeat to index "
                         "several checkouts: the first is PRIMARY (it names provenance.aiter_commit "
                         "and its answer wins where two trees disagree); every tree contributes "
                         "evidence, so a fact whose file was moved or deleted upstream stays citable "
                         "at a commit that still contains it")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true", help="write evidence/gemm_source.yaml")
    mode.add_argument("--json", action="store_true", help="dump to stdout instead")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="extract even when files that produce evidence have uncommitted changes; "
                         "the affected records are marked `unreproducible: true`")
    a = ap.parse_args()

    trees = a.aiter or []
    if not trees:
        ap.error("--aiter is required: this tool reads a source tree that is not part of this repo")
    for tree in trees:
        if not os.path.isdir(tree):
            print(f"no such aiter tree: {tree}", file=sys.stderr)
            return 2
    primary, multi = trees[0], len(trees) > 1

    # Each record carries the commits it was OBSERVED at, merged by content-bound id across trees.
    #
    # An upstream refactor does not falsify a fact, it relocates it: when `splitk_hgemm.py` was folded
    # into `gemm_a16w16_gfx950.py`, 309 records stopped resolving at HEAD while every one of them
    # remained a true statement about a real kernel at the commit it was read from. Discarding them
    # would throw away knowledge over a rename; silently re-pointing them at HEAD would be a citation
    # index that lies. So the commit becomes a dimension of the record instead, and the three cases a
    # reader actually cares about fall out of `aiter_commits` without anyone curating them: present in
    # every tree (survived the refactor, strongest), new tree only (current practice), old tree only
    # (the technique is sound, the implementation moved — go find where).
    merged, order = {}, []
    tuned_store, tuned_order = {}, []
    flydsl_store, flydsl_order = {}, []
    sel_store, sel_order = {}, []
    coverage, missing, inventory, db_missing = [], [], [], []
    commits, dirty_all = [], []
    configs, rows_summarised, db_rows_read = [], 0, 0

    for pos, tree in enumerate(trees):
        evidence, tree_configs, tree_tuned, seen, gaps = collect(tree)
        db_rows, db_inventory, db_gaps = tuned_db_rows(tree)
        moe_rows, moe_inventory, moe_gaps = fmoe_db_rows(tree)
        commit = aiter_commit(tree)

        # Which files that CONTRIBUTED evidence are locally modified. Scoped to contributors rather
        # than the whole tree because that is the actual condition: an unrelated edit elsewhere in
        # aiter cannot make a citation unresolvable, and refusing on it would be superstition rather
        # than a check. The shipped JSON and CSV tables contribute too: a locally re-tuned database
        # would otherwise be published under a commit that never contained it.
        contributors = ({f["file"] for f in evidence} | {r["file"] for r in tree_tuned}
                        | {r["file"] for r in db_rows} | {r["file"] for r in moe_rows})
        dirt = contributors & dirty_paths(tree)
        if dirt and not a.allow_dirty:
            print(f"REFUSING: {len(dirt)} file(s) that produced evidence have uncommitted changes, "
                  f"so their `file:line` would not resolve at {commit[:12]} for anyone else:",
                  file=sys.stderr)
            for p in sorted(dirt)[:10]:
                print(f"  {p}", file=sys.stderr)
            print("\nA citation index whose citations do not resolve is not a weaker index, it is a\n"
                  "different document that looks like one. Either commit those changes, or extract from a\n"
                  "clean checkout:\n"
                  f"  git -C {tree} worktree add /tmp/aiter-clean {commit[:12]}\n"
                  "Pass --allow-dirty to record the evidence anyway; the affected records are then marked\n"
                  "and the committed corpus is not allowed to contain them.", file=sys.stderr)
            return 3

        commits.append(commit)
        # Qualified by commit only when there is more than one tree, where an unqualified path would
        # not say which checkout is dirty.
        dirty_all += [f"{commit[:12]}:{p}" if multi else p for p in sorted(dirt)]
        for record in evidence:
            record.setdefault("evidence_id", source_evidence_id(record))
            if record["file"] in dirt:
                record["unreproducible"] = True
            record.setdefault("fact_id", fact_identity(record))
        # Identical content at a second commit is one record with two anchors: the ID is bound to the
        # match and excerpt, so a repeat is the same statement, not a near-duplicate.
        merge_by_identity(merged, order, evidence, "evidence_id", commit)

        # The shipped tables get the same treatment, and for the same reason. Fifteen cards cite
        # `cfg_…` IDs as evidence of a bucket STRUCTURE, not as a current recommendation; reading
        # those tables from the primary tree alone silently unmoored every one of them, because the
        # Triton config tree was reorganised upstream and not one of the 386 older group IDs
        # survives into the new layout.
        flydsl_groups, flydsl_gaps = summarize_flydsl_tuned(db_rows)
        moe_groups, moe_unparsed = summarize_fmoe_tuned(moe_rows)
        merge_by_identity(tuned_store, tuned_order, summarize_tuned(tree_tuned), "config_id", commit)
        merge_by_identity(flydsl_store, flydsl_order, flydsl_groups + moe_groups,
                          "config_id", commit)
        merge_by_identity(sel_store, sel_order, summarize_backend_selection(db_rows),
                          "selection_id", commit)
        merge_by_identity(sel_store, sel_order, summarize_fmoe_backend(moe_rows),
                          "selection_id", commit)
        if pos == 0:
            configs = tree_configs
        rows_summarised += len(tree_tuned)
        db_rows_read += len(db_rows) + len(moe_rows)
        # Per-tree facts, so what a refresh dropped is legible as a diff rather than inferred from a
        # record count that moved.
        coverage += [dict(row, aiter_commit=commit) if multi else row for row in seen]
        inventory += [dict(row, aiter_commit=commit) if multi else row
                      for row in db_inventory + moe_inventory]
        missing += [dict(row, aiter_commit=commit) if multi else row for row in gaps]
        db_missing += [dict(row, aiter_commit=commit) if multi else row
                       for row in db_gaps + flydsl_gaps + moe_gaps + moe_unparsed]

    evidence = [merged[i] for i in order]
    grouped = [tuned_store[i] for i in tuned_order]
    flydsl_groups = [flydsl_store[i] for i in flydsl_order]
    selection = [sel_store[i] for i in sel_order]
    commit, unreproducible = commits[0], dirty_all

    # A citation that stopped resolving, repaired in place where the statement can still be found.
    #
    # Without this a card grounded in `splitk_hgemm.py:27` leaves its reader at a deleted file with
    # nothing but a commit to check out, and the practical response is to distrust the card. With it
    # the record says where the same line lives now, so the knowledge survives the rename at the cost
    # of one field. Only the primary tree is a redirect target: pointing at another stale commit
    # would move the problem rather than solve it.
    if multi:
        current = collections.defaultdict(list)
        for f in evidence:
            if commit in f["aiter_commits"]:
                current[f["fact_id"]].append(f"{f['file']}:{f['line']}")
        for f in evidence:
            if commit not in f["aiter_commits"] and f["fact_id"] in current:
                here = current[f["fact_id"]]
                # Capped: the point is an entry point into the current tree, and a record that
                # listed sixty call sites would be answering a different question.
                f["relocated_to"] = sorted(here)[:4]

    prov = {
        # The commit is the expiry date. Without it a stale fact and a current one are the same
        # document, which is how `kernel_families.md` came to state a default tile nothing checks.
        # Singular stays singular: it names the primary tree, the one the config tables and the
        # rendered pages speak for. The full list is beside it.
        "aiter_commit": commit,
        "aiter_commits": commits,
        # Empty means every citation below resolves at that commit for anyone. Non-empty names exactly
        # which ones do not, which is the part a boolean could not say.
        "aiter_dirty_sources": unreproducible,
        "aiter_origin": aiter_origin(primary),
        "extractor": "perf_knowledge/corpus/_extract_impl_facts.py",
        "operator_family": "gemm",
    }
    seen = coverage
    payload = {
        "provenance": dict(prov, evidence_records=len(evidence), autotune_configs=len(configs)),
        "coverage": seen, "missing": missing, "autotune_configs": configs,
        "source_evidence": evidence,
    }
    # The shipped config table changes on a different clock from the source (a reselection rewrites all
    # of it; a kernel edit touches a few lines), so it gets its own file and its own diff.
    tuned_payload = {
        "provenance": dict(prov, records=len(grouped), rows_summarised=rows_summarised,
                           source=TUNED_CONFIG_DIR),
        "tuned_configs": grouped,
    }
    db_payload = {
        "provenance": dict(prov, records=len(flydsl_groups), selection_groups=len(selection),
                           rows_read=db_rows_read, source=", ".join(TUNED_DB_DIRS),
                           timing_columns_dropped=list(TUNED_DB_TIMING)),
        "inventory": inventory,
        "missing": db_missing,
        "tuned_configs": flydsl_groups,
        "backend_selection": selection,
    }
    if a.json:
        print(json.dumps({"evidence": payload, "tuned": tuned_payload, "tuned_db": db_payload},
                         ensure_ascii=False, indent=2))
        return 0

    os.makedirs(EVIDENCE, exist_ok=True)
    out = os.path.join(EVIDENCE, "gemm_source.yaml")
    tuned_out = os.path.join(EVIDENCE, "gemm_tuned_configs.yaml")
    db_out = os.path.join(EVIDENCE, "gemm_csv_tuned_configs.yaml")
    emit_yaml(payload, out)
    emit_yaml(tuned_payload, tuned_out)
    emit_yaml(db_payload, db_out)
    by_lang = collections.Counter(f["language"] for f in evidence)
    print(f"OK: {len(evidence)} source-evidence records from {len(seen)} files "
          f"({', '.join(f'{k}={v}' for k, v in sorted(by_lang.items()))}), "
          f"{len(configs)} autotune -> {os.path.relpath(out, ROOT)}")
    if multi:
        # The split is the point of indexing more than one tree, so it is printed rather than left
        # for a reader to count: what survived the refresh, what the refresh added, and what now
        # resolves only at an older commit and therefore needs a card that says where it went.
        held = sum(1 for f in evidence if len(f["aiter_commits"]) == len(commits))
        stale = [f for f in evidence if commit not in f["aiter_commits"]]
        moved = [f for f in stale if f.get("relocated_to")]
        print(f"     commit anchors: {held}/{len(evidence)} cited identically in all {len(commits)} "
              f"trees; {len(stale)} do not resolve at the primary tree {commit[:12]}, of which "
              f"{len(moved)} are the same statement relocated (`relocated_to`) and "
              f"{len(stale) - len(moved)} are absent from it entirely")
        absent = collections.Counter(f["file"] for f in stale if not f.get("relocated_to"))
        for path, n in absent.most_common(8):
            print(f"       gone: {n:4d} records  {path}")
    print(f"OK: {rows_summarised} shipped tuned rows in {len(grouped)} (gfx, tags, M) groups "
          f"-> {os.path.relpath(tuned_out, ROOT)}")
    moe_cfg = sum(1 for g in flydsl_groups if g.get("moe_shape"))
    moe_sel = sum(1 for s in selection if s.get("moe_shape"))
    print(f"OK: {db_rows_read} tuned-DB rows from {len(inventory)} CSVs -> {len(flydsl_groups)} "
          f"FlyDSL groups ({moe_cfg} fused-MoE), {len(selection)} backend-selection groups "
          f"({moe_sel} fused-MoE) -> {os.path.relpath(db_out, ROOT)}")
    if multi:
        # Not a warning. A shipped table that answers the same lookup differently at two commits is
        # the single most useful thing a cross-tree index can tell an author quoting one.
        retuned = sum(1 for g in grouped + flydsl_groups + selection if g.get("retuned_at"))
        print(f"     {retuned} shipped group(s) answer the same (gfx, tags, shape) key differently "
              f"at an older commit (`retuned_at`)")
    for m in missing + db_missing:
        print(f"  gap: {m['language']} — {m['why']} ({m['subtree']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
