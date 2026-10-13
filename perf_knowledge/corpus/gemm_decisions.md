# GEMM development decision cards

This is the actionable layer. It does not dump regex matches and ask the reader to infer a recommendation. Every curated card states **when it applies, what to try, why, alternatives, evidence strength and limits**. The raw, reproducible source observations remain in [`gemm_source_evidence.md`](gemm_source_evidence.md).

Source baseline: AITER `4faf87137cfb5e4d5538e010bb4fefa6c8c5df8d` · 43 curated decision card(s) citing 342 of 5167 source-evidence records · 617 FlyDSL shipped-seed groups · 1292 backend-selection groups · 2042 Triton seed groups (on [`gemm_triton_seeds.md`](gemm_triton_seeds.md)).

How to read this page:

- Steps follow the order a kernel is written in, from its math contract to validation. A card sits under the step of its first axis — the question it answers — and every other step its axes touch links back to it: choosing split-K in step 3 is listed again in step 8, where the partials are combined, and step 9, where the combine's host state is written.
- Within a step, constraint cards come first. Read every constraint card before costing a candidate: architecture gates are in step 2 and build legality in step 3.
- A card's *Same question in other implementations* block shows how Triton, Gluon, CK or ASM answer the same question in AITER, with their own source lines. Transfer the intent, not the spelling: use the target language's docs to write it.
- Evidence is `file:line`, or `file:line-end` when the cited construct spans lines; open the range to read the whole mechanism.
- The two shipped tables after the cards are what AITER's tuner selected — FlyDSL configurations per bucket, and which backend it chose per bucket. They seed a search; they never end one.

Evidence levels:

- `source_observed`: implementation precedent only; add a candidate and measure it.
- `shipped_config` / shipped selection: parameter seeds and backend choices recorded in AITER's shipped tables; alternatives and benchmark results are not attached, so vary and measure locally.
- Background references point at always-on docs under `perf_knowledge/` (hardware facts, how-to), never at measurements.
- Measured guidance is deliberately not copied here: it stays in learned cards/expert skills behind their existing feature switches.

## Design traits

A card whose match `requires` a trait is **deferred** by `_select_candidates.py` until you pass `--trait <name>`. A trait describes your design, which the selector cannot see, so deferred means "applies once your kernel does this", not "does not apply".

- `lds_staging` — The design stages at least one operand tile through LDS: it is stored to LDS and the MFMA fragments are read back from it.
- `reused_weight` — B, the weight, is reused across calls, so a one-time host-side layout transform such as a preshuffle is paid once per weight rather than per call.
- `split_k_hot_loop` — The K loop is hand-scheduled with ROCm scheduling directives over LDS-staged operands, as the FlyDSL split-K HGEMM hot loop is.

## Coverage by architecture

What this corpus holds for each target, so a plan knows how much it is standing on. A card counts for a gfx when its match does not exclude it. *Arch-specific source* counts gates that name the gfx (or a prefix of it, such as `gfx95`) plus records whose code sits under such a gate. ASM kernels are the rows of AITER's per-arch kernel inventories (`hsa/<gfx>/<kind>/*.csv`). Counts, not rankings.

| gfx | cards that can apply | arch-specific source records | ASM kernels by kind | FlyDSL seed groups | backend-selection groups | Triton seed groups |
|---|---|---|---|---|---|---|
| `gfx950` | 42 of 43 | 19 | bf16gemm 24, f4f6gemm 6, f4gemm 35, f6f4gemm 7, f6gemm 8, fmoe_2stages 182, fp8gemm_blockscale 6, i8gemm 8 | 337 | 504 | 720 |
| `gfx942` | 33 of 43 | 37 | bf16gemm 22, fmoe_2stages 186, fp8gemm_blockscale 6, i8gemm 9 | 36 | 146 | 224 |
| `gfx1250` | 33 of 43 | 5 | f4gemm 8, mxfp8fp4gemm 12 | 49 | 121 | 957 |
| `gfx1201` | 33 of 43 | 1 | — | 0 | 24 | 58 |
| `gfx11` | 33 of 43 | 3 | — | 0 | 0 | 0 |
| `gfx1100` | 33 of 43 | 1 | — | 0 | 0 | 28 |
| `gfx1150` | 33 of 43 | 1 | — | 0 | 0 | 29 |
| `gfx1151` | 33 of 43 | 1 | — | 0 | 0 | 26 |

## Development order

| step | axes | cards that answer it | cards from other steps that bear on it |
|---|---|---|---|
| 1. Interface and math semantics | — | — | — |
| 2. Architecture capability and implementation path | `architecture_capability`, `implementation_variant` | `flydsl-gfx942-path-gates` (constraint), `flydsl-small-m-hgemm-family` (performance_candidate), `gemm-shape-regime-routing` (performance_candidate), `gemm-default-backend-routing` (performance_candidate), `grouped-moe-two-stage1-families-compete-at-the-same-shape` (performance_candidate) | `flydsl-a8-legality`, `flydsl-half-mfma-call-forms`, `flydsl-a8-mfma-selection`, `flydsl-async-copy-by-family`, `gemm-blockscale-shipped-bucket-structure`, `flydsl-preshuffle-b-layout-contract`, `gemm-skinny-decode-wave-split`, `flydsl-host-launch-contract`, `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract`, `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline` |
| 3. Grid, workgroup tile, waves and split-K | `workgroup_tile`, `work_partition`, `execution_geometry`, `configuration_constraints` | `flydsl-hgemm-tiling-validity` (constraint), `flydsl-a8-legality` (constraint), `grouped-moe-sort-block-m-couples-both-stages` (constraint), `grouped-moe-mxfp4-stage1-variant-whitelist` (constraint), `gemm-workgroup-xcd-mapping` (performance_candidate), `gemm-split-k-when-grid-underfills` (performance_candidate), `flydsl-hgemm-config-space` (performance_candidate), `flydsl-a8-preshuffle-config-space` (performance_candidate), `gemm-blockscale-shipped-bucket-structure` (performance_candidate), `gemm-asm-tile-inventory-and-selection` (performance_candidate), `gemm-skinny-decode-wave-split` (performance_candidate) | `flydsl-a8-mfma-selection`, `gemm-blockscale-k-loop-scale-application`, `flydsl-cshuffle-epilogue`, `flydsl-xor16-lds-layout-consistency`, `flydsl-preshuffle-b-layout-contract`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `flydsl-splitk-reduction-contract`, `flydsl-small-m-hgemm-family`, `gemm-shape-regime-routing`, `flydsl-host-launch-contract`, `flydsl-test-parity-gate`, `flydsl-lds-pipeline-depth`, `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract`, `grouped-moe-two-stage1-families-compete-at-the-same-shape`, `grouped-moe-mxfp4-kernel-name-is-the-config-space`, `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract`, `grouped-moe-mxfp4-stage2-epilogue-picks-the-output-format`, `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate` |
| 4. Layout and address computation | `operand_layout` | `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract` (constraint), `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate` (constraint), `flydsl-mfma-fragment-lane-layout` (semantic), `flydsl-xor16-lds-layout-consistency` (performance_candidate), `flydsl-preshuffle-b-layout-contract` (performance_candidate) | `flydsl-a8-data-movement-helpers`, `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` |
| 5. Global, LDS and register data movement | `memory_access` | `flydsl-async-copy-by-family` (performance_candidate), `flydsl-a8-data-movement-helpers` (performance_candidate), `gemm-streamed-operand-cache-policy` (performance_candidate), `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline` (performance_candidate) | `flydsl-gfx942-path-gates`, `gemm-workgroup-xcd-mapping`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `gemm-skinny-decode-wave-split`, `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` |
| 6. MFMA / dot compute | `compute_instruction` | `flydsl-a8-scale-granularity-contract` (constraint), `flydsl-a8-mfma-selection` (semantic), `gemm-blockscale-k-loop-scale-application` (semantic), `flydsl-half-mfma-call-forms` (performance_candidate) | `flydsl-gfx942-path-gates`, `flydsl-splitk-hot-loop-schedule-bundle`, `flydsl-a8-interleaved-schedule`, `flydsl-mfma-fragment-lane-layout` |
| 7. Pipeline and synchronization | `pipeline_schedule` | `flydsl-splitk-hot-loop-schedule-bundle` (performance_candidate), `flydsl-a8-interleaved-schedule` (performance_candidate), `flydsl-lds-pipeline-depth` (performance_candidate) | `flydsl-hgemm-tiling-validity`, `flydsl-async-copy-by-family`, `flydsl-a8-preshuffle-config-space`, `flydsl-a8-data-movement-helpers`, `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline`, `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate`, `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` |
| 8. Reduction, epilogue and write-back | `epilogue` | `flydsl-splitk-reduction-contract` (constraint), `grouped-moe-mxfp4-stage2-epilogue-picks-the-output-format` (constraint), `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` (semantic), `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` (semantic), `flydsl-cshuffle-epilogue` (performance_candidate), `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` (performance_candidate) | `flydsl-a8-scale-granularity-contract`, `gemm-split-k-when-grid-underfills`, `flydsl-a8-preshuffle-config-space`, `flydsl-mfma-fragment-lane-layout`, `flydsl-test-parity-gate`, `grouped-moe-parity-gate-is-cosine-not-elementwise` |
| 9. Host configuration, ABI and launch | `runtime_contract` | `flydsl-host-launch-contract` (constraint), `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract` (constraint) | `flydsl-a8-scale-granularity-contract`, `gemm-blockscale-k-loop-scale-application`, `gemm-split-k-when-grid-underfills`, `flydsl-preshuffle-b-layout-contract`, `flydsl-splitk-reduction-contract`, `gemm-default-backend-routing`, `grouped-moe-parity-gate-is-cosine-not-elementwise`, `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag`, `grouped-moe-mxfp4-kernel-name-is-the-config-space`, `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract`, `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` |
| 10. Correctness validation and tuning | `validation_contract`, `tunable_surface` | `flydsl-test-parity-gate` (constraint), `grouped-moe-parity-gate-is-cosine-not-elementwise` (constraint), `grouped-moe-mxfp4-kernel-name-is-the-config-space` (semantic) | `flydsl-hgemm-config-space`, `flydsl-a8-preshuffle-config-space`, `gemm-blockscale-shipped-bucket-structure`, `gemm-streamed-operand-cache-policy`, `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag`, `grouped-moe-mxfp4-stage1-variant-whitelist` |

## 1. Interface and math semantics

Fixed by the task rather than chosen: `math_contract` and `dtype` in `comparison_context` above, read from the task's meta.json, with the operator's numerics.md for parity. Its consequences are placed where they are implemented — scale application in step 6, the argument and layout contract in step 9.

No card answers this step yet.

## 2. Architecture capability and implementation path

- `architecture_capability` — Which implementation paths and instruction/data formats exist on the target architecture?
- `implementation_variant` — Which kernel family or opaque backend implementation is selected for this scenario?

Also bears on this step: `flydsl-a8-legality` (step 3), `flydsl-half-mfma-call-forms` (step 6), `flydsl-a8-mfma-selection` (step 6), `flydsl-async-copy-by-family` (step 5), `gemm-blockscale-shipped-bucket-structure` (step 3), `flydsl-preshuffle-b-layout-contract` (step 4), `gemm-skinny-decode-wave-split` (step 3), `flydsl-host-launch-contract` (step 9), `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract` (step 9), `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline` (step 5).

### Which FlyDSL GEMM paths does gfx942 (MI300X) actually give you, and which does it silently switch or refuse?

**Card:** `flydsl-gfx942-path-gates` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `architecture_capability`, `compute_instruction`, `memory_access`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `gfx`: `gfx942`

#### Use when

- operator_family=gemm
- target_language=flydsl
- target architecture is gfx942 — read this before costing any other card on this page

#### Try

- Read the architecture gates before the tuning tables. Whether a path exists on your box outranks every fact about how to tune it, and several gates in this library key on gfx942 alone.
- Do not fund async copy for the split-K HGEMM family on gfx942. That family computes its async-copy flag as `architecture is not gfx942`, rejects any request that disagrees, and its kernel hard-codes async off on this branch — there is no knob to flip. This does NOT carry over to the A8 (fp8/int8) preshuffle family, which keeps `use_async_copy` as a real knob on gfx942 with a 4-byte A copy instead of 16 — see flydsl-async-copy-by-family.
- Do not fund the small-M kernel family on gfx942. Its compile entry point raises, and its config registry returns nothing. Port the design if you want it; you cannot call it.
- Expect the HGEMM to use the m16n16k16 fragment with 4-byte DMA and two MFMA steps per warp-K, not the m16n16k32 / 16-byte / one-step bundle the other architectures get. The choice is made for you by one architecture test — see flydsl-half-mfma-call-forms.
- Expect the A8 family on gfx942 to take its hand-written gfx942 schedule branch, K=32 fp8 MFMAs instead of the K=128 scaled MFMA, and tile_k=64 tiles that the gfx950 catalog omits — see flydsl-a8-mfma-selection and flydsl-a8-interleaved-schedule.
- Expect no shipped FlyDSL seed: every gfx942 fp8 row in AITER's tuned databases went to CK or ASM. Budget a FlyDSL kernel on this box as authoring work, not as a lookup — see gemm-shape-regime-routing.
- On gfx950 these gates open: the wider fragment, the 16-byte DMA, async copy and the small-M family are all in play. Other architectures are not excluded by these gates, but AITER ships no FlyDSL configuration for them.

#### Why this is a candidate

- Six gfx942 tests exist in the FlyDSL GEMM sources and they do different things: one switches the HGEMM fragment/DMA/MFMA bundle, one fixes HGEMM async copy library-wide, two shut the small-M family, one narrows the A8 async A-copy width and one selects the A8 gfx942 schedule.
- This is the cheapest knowledge on the page. A gate costs nothing to check and a whole round to discover, and none of it is visible from the tuning tables, which record what was selected on the architectures where the path was open.

#### Keep as alternatives

- On gfx942, spend the search on what remains genuinely open — the tile geometry, split_k, B staging and the epilogue — rather than on the gated paths.
- If a gated path is the real ceiling for your shape, vendor and port it deliberately as a structural change, and budget it as one.

#### Evidence

- `src_c4c19187bcb7933c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/gemm_kernels.py:41`
- `src_a0f1066a6a2d6667` — `config_validity` `Current kernel fixes async_copy from the active GPU architecture;` at `aiter/ops/flydsl/gemm_kernels.py:123-126`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128-137`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411-412`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331-332`
- `src_42146c2d073d655e` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:243`
- `src_fbe3baca81526883` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:255`
- `sel_5aecc3631bf688e9` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 230 · cktile 17 (rows)
- `sel_438e88e4bc86460c` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: ck 402 · cktile 13 · asm 4 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 26 · opus 9 (rows)

#### Background (always-on reference docs, not measurements)

- [`hardware/cdna3_mi300/memory_hierarchy.md`](../hardware/cdna3_mi300/memory_hierarchy.md)
- [`languages/flydsl/knobs.md:71-75`](../languages/flydsl/knobs.md)
- [`languages/flydsl/kernel_families.md:19-32`](../languages/flydsl/kernel_families.md)

#### Limits

- These are the gates that name an architecture literally. A path can also be effectively closed by something this card does not see — a missing dtype, an LDS budget, a compiler version.
- Gates are per family: a gate in one family's source says nothing about another family's, which is exactly how the async-copy statement used to be over-read.
- Gates move between releases. Every citation carries its file and line; re-read them against the checkout you are building on before trusting the list.
- That a path is open says nothing about whether it is fast.

### What does AITER do differently when M is a handful of rows, and which knobs does that open?

**Card:** `flydsl-small-m-hgemm-family` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `architecture_capability`, `workgroup_tile`, `work_partition`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `skinny_gemv_decode`
- `target_languages`: `flydsl`
- `exclude_gfx`: `gfx942`
- `dtypes`: `bf16`
- `shape`: `m_max=16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- M is small — the shipped routing threshold is M < 17
- dtype is bf16

#### Try

- Check the architecture first. gfx942 is excluded by the source: the compile entry point raises and the config registry returns an empty set before it looks at the shape. gfx950 is where the registry is exercised; other architectures are not excluded by the gate but have no shipped configuration. On gfx942 read this card as a porting reference.
- Treat small M as a different kernel family rather than the generic kernel with a smaller tile: AITER routes it to a separate implementation selected by an explicit kernel_family value.
- In that family tile_m is fixed at 16 with block_m_warps=1 and 2 stages — the M dimension stops being a search axis, and the search moves entirely onto N and K.
- Search tile_n far wider than the generic family allows — the shipped set runs (32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024) — with tile_k in (32, 64, 96, 128, 160, 192, 256) and split_k up to 32.
- Consider the axes this family adds and the generic one does not: n_tile_repeat, persistent_n_tiles, waves_per_eu and a b_to_lds unroll factor.
- Respect the combination rules: n_tile_repeat > 1 only without B-in-LDS and only for the two shipped shapes; persistent_n_tiles > 1 only with B-in-LDS, tile_n >= 128, block_n_warps >= 2 and no more persistent tiles than N/tile_n.
- Give the generic family a real arm, not a token one: all 86 FlyDSL rows AITER ships for gfx950 bf16 at M<=16 are generic-family kernels (mostly tile_m=32 with split_k 4-16), none from this family. The database does not say whether this family was in that race.

#### Why this is a candidate

- A skinny-M GEMM is bounded by how much of the machine the few M tiles can occupy, and every axis this family adds — repeat, persistent tiles, waves per EU — is an occupancy axis rather than a tiling one.
- The routing threshold, the fixed tile_m and the option sets are stated as named constants in the source, so the boundary between the two families is readable rather than folklore.

#### Keep as alternatives

- Stay on the generic family with a small tile_m and split_k > 1 when a second kernel cannot be written or maintained; that is what AITER shipped for these buckets and is the cheaper first move.
- Port the HIP skinny kernels' structure instead — rows spread over CUs, K split across the lanes of a wave, no combine (gemm-skinny-decode-wave-split). That is what AITER's default route runs for an untuned bf16 decode case, on both architectures.
- Keep the generic default configuration as the control arm regardless of which family is chosen.

#### Evidence

- `src_38699a8ce57faf0c` — `kernel_family` `KERNEL_FAMILY_HGEMM, hgemm` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:8`
- `src_8ffea30406cf0355` — `kernel_family` `KERNEL_FAMILY_SMALL_M, small_m` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:9`
- `src_bca963564a69909c` — `kernel_family` `KERNEL_FAMILY_SMALL_M` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:56-71`
- `src_b446c07f43b11542` — `config_limit` `SMALL_M_KERNEL_MAX, 17` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:69`
- `src_034b3cc6e3fd6aa8` — `tile_shape` `TILE_M, 16` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:70`
- `src_390ae8578e28ce34` — `config_space` `SMALL_M_TILE_N_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:83-96`
- `src_60717171485c58f5` — `config_space` `SMALL_M_TILE_K_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:81`
- `src_ee5e839c6374f38f` — `config_limit` `SMALL_M_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:82`
- `src_a7c0de9870934fc0` — `config_space` `SMALL_M_N_TILE_REPEAT_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:102`
- `src_28936b493eeb6f6f` — `config_space` `SMALL_M_PERSISTENT_N_TILE_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:103`
- `src_1dcdf9c7b6794c6e` — `config_space` `SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:97`
- `src_44b5602aacc7e7df` — `config_space` `SMALL_M_B_TO_LDS_UNROLL_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:101`
- `src_cceaeb92fb86b540` — `config_validity` `_validate_small_m_registry_config` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:172-228`
- `src_bc6d04baf2182ba4` — `config_limit` `MAX_LDS_BYTES, 163840` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:76`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411-412`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331-332`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:77-80`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:78-82`](../languages/flydsl/patterns.md)
- [`operators/skinny_gemv_decode/backends/flydsl.md:19-28`](../operators/skinny_gemv_decode/backends/flydsl.md)

#### Limits

- On gfx942 none of this is callable; see the first action. The rest of the card describes the gfx950 path.
- That AITER maintains a separate family for small M is a design precedent, not a measurement that it beats a well-tuned generic configuration on your shape.
- The option sets are what the shipped registry enumerates; they are candidates to search, and nothing here says which member wins.
- The family is bf16-only and its LDS budget is its own; do not carry its constants into the generic kernel.

### For each shape bucket, which backend did AITER's own tuner ship, and what does that imply for a FlyDSL kernel?

**Card:** `gemm-shape-regime-routing` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `workgroup_tile`, `work_partition`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a workload with one or more (M, N, K) cases, with gfx and dtype known

#### Try

- Look up each case's bucket in the backend-selection table below before choosing directions. It says which implementation AITER shipped for comparable shapes on that gfx and CU count — what a FlyDSL kernel competes with in production, even when the task's frozen baseline is Triton.
- gfx950 bf16: FlyDSL was selected for 86 of 165 rows at M<=16 and 52 of 174 at M 65-256, but for only 7 of 186 rows above M=1024, where ASM took 164. Large-M bf16 is where FlyDSL has no shipped precedent: plan structural work (wave schedule, tile, workgroup mapping), not a config lookup.
- gfx950 fp8 per-token with preshuffled B: FlyDSL was selected in every bucket, most often at the extremes (72 of 111 rows at M<=16, 77 of 133 above M=4096) — seed from the FlyDSL A8 groups.
- fp8 128x128 block scale on either architecture, and every gfx942 fp8 database: no FlyDSL row exists — CK, CK-Tile and ASM own them. A FlyDSL kernel there is authored from the A8 precedents plus a block-scale K loop, with no shipped FlyDSL seed (see flydsl-a8-scale-granularity-contract and gemm-blockscale-k-loop-scale-application); which configuration the Triton baseline actually runs per case — often one fallback tile for every M — is gemm-blockscale-shipped-bucket-structure.
- Route small-M bf16 on gfx950 through both the generic HGEMM (what AITER shipped) and the small-M family (flydsl-small-m-hgemm-family); route quantized operands to the A8 family.
- Treat buckets as separate problems: when a workload spans buckets (decode plus prefill), expect different winners per case and keep a per-case dispatch rather than one global configuration.
- For a case no database row covers, read AITER's default route instead (gemm-default-backend-routing): the HIP skinny kernels for small-M bf16/fp16, hipblaslt (gfx942) or ASM (elsewhere) for B-preshuffled bf16, CK for fp8 — that is what a FlyDSL kernel competes with there.

#### Why this is a candidate

- The tuned databases are the only place AITER records which implementation it chose for which shape regime.
- A frozen Triton baseline says nothing about where FlyDSL already wins in production and where it never has; this table does.

#### Keep as alternatives

- Ignore the shipped selection and search blind — the control arm, and the only option for a bucket the database does not cover.

#### Evidence

- `src_38699a8ce57faf0c` — `kernel_family` `KERNEL_FAMILY_HGEMM, hgemm` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:8`
- `src_bca963564a69909c` — `kernel_family` `KERNEL_FAMILY_SMALL_M` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:56-71`
- `src_b446c07f43b11542` — `config_limit` `SMALL_M_KERNEL_MAX, 17` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:69`
- `src_f854f40de1f28585` — `scale_operand` `_needs_per_token_scale` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:403`
- `sel_81411b8982b03045` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 451 · triton 77 · asm 30 · torch 27 · skinny 23 · opus 5 · hipblaslt 3 (rows)
- `sel_4ff2330404509760` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: flydsl 541 · torch 78 · asm 42 · triton 12 · opus 10 (rows)
- `sel_0bf8dd0fd028cea8` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: flydsl 237 · torch 139 · asm 52 · opus 16 · hipblaslt 2 (rows)
- `sel_c472ec443f7b30a0` — backend selected by AITER's tuner in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: torch 128 · flydsl 116 · asm 85 · opus 23 · triton 1 (rows)
- `sel_7c8d8e410633ad1e` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 274 · ck 57 · cktile 20 (rows)
- `sel_1b5489e6d41507a3` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: flydsl 280 · cktile 40 · ck 26 (rows)
- `sel_8b543c4558078855` — backend selected by AITER's tuner in `a8w8_blockscale_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: ck 8960 · cktile 1065 (rows)
- `sel_438e88e4bc86460c` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: ck 402 · cktile 13 · asm 4 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/tuning.md`](../operators/dense_gemm/tuning.md)
- [`operators/splitk_streamk_gemm/tuning.md:16-19`](../operators/splitk_streamk_gemm/tuning.md)
- [`languages/flydsl/patterns.md:41-53`](../languages/flydsl/patterns.md)

#### Limits

- Counts are rows, not unique shapes: a shape shipped in two model databases counts twice.
- The tuner's box, competing set and toolchain are not recorded; a selection says where AITER chose a backend, never by how much.
- The table reflects AITER at the pinned commit; re-extract after an AITER bump.

### Which backend does AITER's own dispatch pick when no tuned row exists, and how does that differ between gfx942 and gfx950?

**Card:** `gemm-default-backend-routing` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a case whose production path, or whose frozen baseline, is AITER's dispatch rather than a fixed kernel

#### Try

- bf16/fp16 (`tuned_gemm.get_GEMM_A16W16_config`): a tuned row wins first, and a `flydsl` row is served only when `is_flydsl_available()` and its kernel name parses — otherwise the row is dropped. With no row: gfx12 goes to torch; a B-preshuffled request goes to hipblaslt on gfx942 and, elsewhere, to ASM when the input is bf16, the output bf16 or fp32, and N and K are multiples of 64 (it asserts otherwise); an unshuffled request goes to the HIP skinny kernels when `is_skinny_default_shape` holds, else to torch.
- The skinny route owns bf16/fp16 decode by default: K % 8 == 0 and M = 1 with N <= 2*CU and K <= 9216, M <= 4 with N <= CU and K <= 9216, M <= 8 with N <= CU and K <= 5120, or M <= 16 with N <= CU and K <= 256 (gemm-skinny-decode-wave-split).
- fp8 128x128 block scale (`gemm_op_a8w8.gemm_a8w8_blockscale`): a B-preshuffled call goes to ASM only on gfx950 with m >= 16, k >= 512 and bf16 output, and asserts otherwise; an unshuffled call goes to CK or CK-Tile by tuned row, CK by default. The B-preshuffled block-scale entry (`gemm_a8w8_blockscale_bpreshuffle`) picks CK-Tile, CK or ASM by tuned row — that is where gfx942 reaches its ASM block-scale kernels.
- fp8 per-token with preshuffled B (`gemm_a8w8_bpreshuffle`, fp8 weights only at the pinned commit): CK, CK-Tile or FlyDSL by tuned row (`gemm_a8w8_bpreshuffle_flydsl`, only when FlyDSL is available), CK with no row.
- Read these as what a FlyDSL kernel competes with for a case AITER never tuned; where it did tune, the backend-selection table (gemm-shape-regime-routing) says what it picked.

#### Why this is a candidate

- The default route is code, not a table: it is what production runs for every shape the tuned databases do not cover, and it differs by architecture.
- It names FlyDSL as a routed backend for bf16 and for per-token fp8, and never for block scale.

#### Keep as alternatives

- Ignore AITER's dispatch and compare only against the task's frozen baseline — the control arm.

#### Evidence

- `src_2577a978dcbebafb` — `kernel_family` `is_skinny_default_shape` at `aiter/tuned_gemm.py:76-97`
- `src_c9b676e9661f7f1e` — `kernel_family` `flydsl` at `aiter/tuned_gemm.py:133-141`
- `src_57c3a751ec4ad78b` — `kernel_family` `asm` at `aiter/tuned_gemm.py:172`
- `src_07dceb286dc43c55` — `kernel_family` `skinny` at `aiter/tuned_gemm.py:182`
- `src_44e19b6d045bd0ec` — `kernel_family` `gfx950_a8w8_blockscale_ASM` at `aiter/ops/gemm_op_a8w8.py:668`
- `src_9dd4c645f2e22c23` — `kernel_family` `ck` at `aiter/ops/gemm_op_a8w8.py:677-682`
- `src_ae977d7a08feed5e` — `kernel_family` `cktile` at `aiter/ops/gemm_op_a8w8.py:679-682`
- `src_08f91e081b402af2` — `kernel_family` `gemm_a8w8_blockscale_ck` at `aiter/ops/gemm_op_a8w8.py:683`
- `src_f4e4d3f780fda60a` — `kernel_family` `cktile` at `aiter/ops/gemm_op_a8w8.py:734-743`
- `src_24f53ab3d301476f` — `kernel_family` `asm` at `aiter/ops/gemm_op_a8w8.py:738-743`
- `src_bcbb91da2b9d2eb8` — `kernel_family` `flydsl` at `aiter/ops/gemm_op_a8w8.py:629-630`
- `src_df8a47f1c59069da` — `kernel_family` `gemm_a8w8_bpreshuffle_ck` at `aiter/ops/gemm_op_a8w8.py:632`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/patterns.md:41-53`](../languages/flydsl/patterns.md)
- [`operators/skinny_gemv_decode/tuning.md:16-33`](../operators/skinny_gemv_decode/tuning.md)

#### Limits

- The routing conditions are AITER's at the pinned commit; its dispatch changes between releases, so re-read `tuned_gemm.py` and `ops/gemm_op_a8w8.py` after an AITER bump.
- A route says which backend runs, not how fast it is; hipblaslt and torch routes are libraries this corpus does not index.

### For one fixed grouped-MoE shape, AITER ships two different FlyDSL stage-1 kernel families rather than one family swept over knobs — which applies, and why does that matter for authoring?

**Card:** `grouped-moe-two-stage1-families-compete-at-the-same-shape` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `implementation_variant`, `workgroup_tile`, `work_partition`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`

#### Use when

- target_language=flydsl
- gfx950
- a stage-1 grouped GEMM exists at the exact (model_dim, inter_dim, expert, topk) shape in more than one FlyDSL source module

#### Try

- Check which source module is live for a given shape before tuning knobs inside just one of them: the 'moe_stage1' family compiles out of the generic 2-stage builder (`mixed_moe_gemm_2stage_common.py` at current main; the same builder lived in `moe_gemm_2stage.py` at commit `a6bb4993`, which upstream has since split per-arch) with block_m=32 in every shipped row for this shape, while the 'mxmoe_stage1' family compiles out of the separate `mxfp4_gemm1.py` module (block_m=16, its own grid helper `gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, ...)`, its own epilogue tags `f16in`/`hpf`).
- Expect AITER's shipped table to switch between the two families by token-count bucket at this exact shape (gfx950, 6144x256, 257 experts, topk 9, mxfp4): `moe_stage1` at the smallest (`token=1,2`) and largest (`token>=256`) buckets, `mxmoe_stage1` in between (`token=4..128`) — not a single cutover in one direction.
- Treat a retune as scoped to one family's own knob space (its own tile catalog, its own flags) until the bucket boundary itself is swept; moving a knob inside one family does not touch the other family's shipped rows for the neighboring buckets.

#### Why this is a candidate

- Two source modules implementing the same math for the same shape is a maintenance fact worth knowing before retuning: a change to `moe_gemm_2stage.py`'s tile catalog does nothing for the buckets the shipped table currently routes to `mxfp4_gemm1.py`, and vice versa.

#### Keep as alternatives

- Standardize on a single stage-1 family across all buckets for this shape — simpler to maintain and tune, but gives up whatever each specialized family wins in its own M range.

#### Evidence

- `src_c49c81eef10d1108` — `tile_shape` `BN, 256` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:47-54`
- `src_74352923ff7ef43b` — `tile_shape` `BM, 32` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1498`
- `src_ada62c940d6fc161` — `jit_cache` `1024, compile_moe_gemm1` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:87`
- `cfg_58db8a1bd4f97080` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M<=16`: 1 shipped row(s)
- `cfg_cff530f26fedf80c` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M17-64`: 1 shipped row(s)
- `cfg_da3c10c1b64033c6` — FlyDSL `mxmoe_stage1` `a4w4` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M17-64`: 1 shipped row(s)
- `cfg_87387b8c0155e756` — FlyDSL `mxmoe_stage1` `a4w4` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M65-256`: 2 shipped row(s)
- `cfg_039e19576885eb1d` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M1025-4096`: 1 shipped row(s)

#### Limits

- This is read off one model's shipped CSV (`glm5_fp4_tuned_fmoe.csv`) for one shape; the bucket boundary is a fact about what AITER currently ships, not a proven optimum, and a different shape could keep one family the whole range.
- The two families are not pinned to the same upstream commit: the `mxmoe_stage1` citations are live at current main, while the `moe_stage1` builder is cited at `a6bb4993`, so confirm which module name carries that family in the tree actually being built before acting on the module names here.
- The device-side dispatch for a specific deployed build can differ from what the static shipped table alone would predict (padding, batching and runtime routing all sit between the table and the launch); treat the per-bucket family as a seed to verify against the actual launch, not as the final word.

## 3. Grid, workgroup tile, waves and split-K

- `workgroup_tile` — How much M/N/K work does one workgroup own?
- `work_partition` — How is work partitioned or persisted across tiles and the K dimension?
- `execution_geometry` — How many waves/warps and workitems cooperate on the tile?
- `configuration_constraints` — Which combinations are legal with respect to divisibility, LDS/register budgets and fixed parameters?

Also bears on this step: `flydsl-a8-mfma-selection` (step 6), `gemm-blockscale-k-loop-scale-application` (step 6), `flydsl-cshuffle-epilogue` (step 8), `flydsl-xor16-lds-layout-consistency` (step 4), `flydsl-preshuffle-b-layout-contract` (step 4), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `flydsl-splitk-reduction-contract` (step 8), `flydsl-small-m-hgemm-family` (step 2), `gemm-shape-regime-routing` (step 2), `flydsl-host-launch-contract` (step 9), `flydsl-test-parity-gate` (step 10), `flydsl-lds-pipeline-depth` (step 7), `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract` (step 9), `grouped-moe-two-stage1-families-compete-at-the-same-shape` (step 2), `grouped-moe-mxfp4-kernel-name-is-the-config-space` (step 10), `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract` (step 4), `grouped-moe-mxfp4-stage2-epilogue-picks-the-output-format` (step 8), `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate` (step 4).

### Which FlyDSL HGEMM configurations will fail to build at all, before any of them can be slow?

**Card:** `flydsl-hgemm-tiling-validity` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `configuration_constraints`, `workgroup_tile`, `pipeline_schedule`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- candidate configurations are being generated automatically or swept

#### Try

- Filter candidates before compiling them: N must satisfy N >= tile_n and N % tile_n == 0, and K/split_k must satisfy K/split_k >= tile_k and (K/split_k) % tile_k == 0.
- Require tile_k >= 32 and tile_k % 32 == 0, tile_m % (block_m_warps*16) == 0 and tile_n % (block_n_warps*16) == 0.
- Require each of tile_m*tile_k, tile_n*tile_k and tile_m*tile_n to be divisible by 8*64*block_m_warps*block_n_warps, the per-block vectorised load width.
- Estimate LDS as a_lds = max(stages*tile_m*tile_k*2, tile_m*tile_n*2) when B is not staged through LDS; when it is, the estimate is not a_lds plus something, it becomes align_up(a_lds,16) + stages*tile_n*tile_k*2. Drop candidates over the block budget, which the library takes from `addressable_lds_bytes_for_gfx`: 163840 bytes on gfx950 and 65536 on the other CDNA targets.
- Do not spend a round on stages, pack_n or c_to_lds in the generic family: the kernel compiles a fixed 2-stage pipeline, pack_n=1 and c_to_lds off, and rejects anything else. They read like tuning knobs in the signature and are not.
- Do not combine b_preshuffle=True with b_to_lds=True; the kernel rejects it.
- Check the split-K reduction counter capacity separately — see flydsl-splitk-reduction-contract.

#### Why this is a candidate

- AITER performs all of these checks in host code before the kernel is built, so they are the stated contract rather than an inference from a crash.
- An optimizer that cannot separate 'illegal' from 'slow' spends its budget re-discovering the first, and a budgeted search has few rounds to spend.

#### Keep as alternatives

- Let the compiler reject the configuration and catch the exception — acceptable only when the search is small, since each rejection still costs a compile.

#### Evidence

- `src_9ae627bec0096537` — `config_validity` `_validate_hgemm_tiling` at `aiter/ops/flydsl/gemm_kernels.py:330-456`
- `src_d3f3cb90295b0c40` — `config_validity` `_estimate_hgemm_lds_bytes` at `aiter/ops/flydsl/gemm_kernels.py:249-268`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `src_7c7ddd401ff4aaeb` — `config_limit` `SPLIT_K_COUNTER_MAX_LEN, 128` at `aiter/ops/flydsl/gemm_kernels.py:37`
- `src_c139f07a63569327` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MIN, 2` at `aiter/ops/flydsl/gemm_kernels.py:77`
- `src_539c4b7bf95dca07` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MAX, 8` at `aiter/ops/flydsl/gemm_kernels.py:78`
- `src_0f8216bb2c53105f` — `config_validity` `Invalid tile_k={tile_k}; latest kernel requires tile_k >= 32` at `aiter/ops/flydsl/gemm_kernels.py:358-360`
- `src_4404f2040e4191a0` — `config_validity` `Invalid tile_k={tile_k}; latest kernel requires tile_k % 32 == 0` at `aiter/ops/flydsl/gemm_kernels.py:362-364`
- `src_d76e1615be7ab856` — `config_validity` `Invalid tile combination: tile_m * tile_k must be divisible by` at `aiter/ops/flydsl/gemm_kernels.py:415-418`
- `src_cbd1aeb93c0eb00c` — `config_validity` `Current kernel only supports stage={FIXED_STAGE}; got stage={stage}` at `aiter/ops/flydsl/gemm_kernels.py:119-121`
- `src_c01de8a05cd112e2` — `config_validity` `Current kernel only supports `pack_n=1`; got pack_n={pack_n}` at `aiter/ops/flydsl/gemm_kernels.py:372-374`
- `src_d0f75ad1bcdf6730` — `config_validity` `Current kernel only supports c_to_lds={FIXED_C_TO_LDS};` at `aiter/ops/flydsl/gemm_kernels.py:128-131`
- `src_e248a6669d5daaaf` — `config_validity` `Current kernel requires b_to_lds=False when b_preshuffle=True` at `aiter/ops/flydsl/gemm_kernels.py:161-163`
- `src_ce6c8804ca232bf6` — `config_validity` `addressable_lds_bytes_for_gfx` at `aiter/ops/flydsl/utils.py:14-22`
- `src_c694123000988e0b` — `arch_gate` `gfx950` at `aiter/ops/flydsl/utils.py:18-19`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_optimization.md:75-93`](../languages/flydsl/authoring_optimization.md)
- [`languages/flydsl/knobs.md:20-40`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:21-39`](../languages/flydsl/patterns.md)

#### Limits

- The LDS formula is the library's own estimate for this kernel shape; a rewritten staging scheme changes it and the estimate stops applying.
- Passing every check means the configuration builds, not that it is fast, and not that it is correct in a modified kernel.

### Which FlyDSL A8 / FP4 preshuffle-GEMM configurations are rejected before they can be slow?

**Card:** `flydsl-a8-legality` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `configuration_constraints`, `architecture_capability`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`, `int4`, `fp4_e2m1`, `mxfp4`

#### Use when

- target_language=flydsl
- quantized GEMM built on the A8 preshuffle family (fp8, int8/int4, or fp4 on gfx950)
- candidate configurations are being generated or swept

#### Try

- Filter before compiling: tile_k*elem_bytes must be a multiple of 64; tile_m*tile_k*elem_bytes (halved for fp4) must divide evenly over the fixed 256-thread block; and each thread's share of the A tile must be a multiple of the 16-byte load.
- Accept only in_dtype in {fp8, int8, int4, fp16, bf16, fp4} and out_dtype in {fp16, bf16}; there is no fp32 output.
- On the scaled-MFMA path (gfx95x with non-integer, non-half inputs) require tile_k to be a multiple of 128 — see flydsl-a8-mfma-selection.
- Treat FP4 as gfx950-only: it raises on other architectures, fp8 activations with fp4 weights are not supported, and FP4 with tile_k=128 supports only lds_stage=2.
- For int8, confirm the installed FlyDSL exposes `mfma_i32_16x16x32i8` (or the `_i8` spelling); its absence raises instead of falling back.

#### Why this is a candidate

- Every item is a check and a raise in the compile entry point, so it is the stated contract rather than an inference from a crash.
- The fixed 256-thread block and the 16-byte A load make several tile shapes that look plausible from the HGEMM space illegal here.

#### Keep as alternatives

- Let the compiler reject configurations only when the search is small; each rejection still costs a compile.

#### Evidence

- `src_e1aea77c2c7f9600` — `config_validity` `tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:198-201`
- `src_6a9c3bef7e7f1bd0` — `config_validity` `tile_m*tile_k*elem_bytes/a_elem_vec_pack must be divisible by` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:232-235`
- `src_d2906b5ee9f2c4b6` — `config_validity` `bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:240-242`
- `src_3479ce3bb9436d9d` — `config_validity` `in_dtype must be one of (` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:165-168`
- `src_65bcb0094b2ef2f3` — `config_validity` `out_dtype must be` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:170`
- `src_bc08df5300de01f3` — `config_validity` `tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:855-857`
- `src_7e2a5d7fc5c13904` — `config_validity` `FP4 GEMM requires gfx950, got {get_hip_arch()}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1706`
- `src_68781327203b224d` — `config_validity` `fp8-A not yet supported with MXFP4 kernel (op_sel_a overflow)` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1702-1704`
- `src_f5c5e87d02b8a8e5` — `config_validity` `FP4 tile_k=128 currently only supports lds_stage=2` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:211`
- `src_c9b3df8a49dab9d2` — `config_validity` `FP4 requires tile_k=128 or tile_k >= {64 * 2 * a_elem_vec_pack}` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:205-209`
- `src_f9f2544162138ea2` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:215-217`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:82-87`](../languages/flydsl/knobs.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:60-78`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- These are the checks in AITER's vendored kernel at the pinned commit; an authored kernel with its own staging has its own contract.
- Passing every check means the configuration builds, not that it is fast.

### When a 2-stage grouped MoE GEMM sorts tokens to experts once and both GEMMs read that sorted order, what tile-M choices are actually free?

**Card:** `grouped-moe-sort-block-m-couples-both-stages` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- the design is a 2-stage grouped (MoE) GEMM sharing one token-to-expert sort/pad step between stage 1 and stage 2

#### Try

- Pick stage 1's tile_m first: AITER's generic 2-stage compiler fixes the sort block size to it directly (`sort_block_m = tile_m`, no stage-1 override) — stage 1 is never free to sort at a size other than its own tile.
- Give stage 2 either that same tile_m (the common case) or a smaller one that evenly divides it: `compile_mixed_moe_gemm2` accepts its own `sort_block_m` but raises unless it already equals stage 2's tile_m or is an exact multiple of it.
- Read the `_sbm{N}` tag in a stage-2 name as a declaration, not as a deviation flag: `build_flydslv2_gemm2_name` appends it whenever `sbm` is non-zero, with no `sbm != tm` guard, and its only caller passes `sbm=block_m` for every candidate tile — so the shipped GLM-5.2 names carry `_sbm16`/`_sbm32`/`_sbm64`/`_sbm128` each equal to their own `t{M}`. A name with no `_sbm` tag means the sort block was left unstated, not that it matches.
- Treat tile_m=16 on stage 2 as the one genuinely coupled case: the BM16 A-scale chunk addressing is selected by `SBM == BM`, not by `BM` alone, so a 16-wide stage-2 tile is only legal when the shared sort block is also 16 — which in turn forces stage 1's BM to 16 (that is the same condition `native_scale_layout_for` imposes). At every other tile_m the multiple-of rule is the whole constraint.

#### Why this is a candidate

- The sort/pad step runs once and lays every expert's rows out at one granularity; a stage tiled at a different, non-dividing granularity would read across a row boundary that does not exist in its own output.
- The raise exists to catch exactly that at compile time: `sort_block_m % tile_m != 0` is rejected before launch, not left to a runtime out-of-bounds read.

#### Keep as alternatives

- An unshared sort per stage — not what AITER's launcher does; it would cost a second sort/pad pass to buy tiling independence the multiple-of rule already gives for free in the common case.

#### Evidence

- `src_d90c21da2fd744e2` — `config_validity` `sort_block_m ({_sort_block_m}) must be a multiple of tile_m ({tile_m})` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:3169-3171`
- `src_b25a5418edbd6918` — `config_validity` `Heterogeneous B requires MXFP4 routed weights` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:164`
- `src_684ee007ec1d2fd1` — `config_space` `build_flydslv2_gemm2_name` at `aiter/ops/flydsl/moe_kernels.py:349-372`
- `src_7fdfffec65089511` — `config_space` `_FLYDSL_V2_GEMM2_RE` at `aiter/ops/flydsl/mxfp4_kname.py:56-61`
- `src_0a0738d9fdf9642d` — `config_space` `MXFP4_G1_VARIANTS` at `aiter/ops/flydsl/mxfp4_kname.py:26-42`
- `cfg_58db8a1bd4f97080` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M<=16`: 1 shipped row(s)
- `cfg_da3c10c1b64033c6` — FlyDSL `mxmoe_stage1` `a4w4` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M17-64`: 1 shipped row(s)
- `cfg_cff530f26fedf80c` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M17-64`: 1 shipped row(s)

#### Limits

- Two different stage-2 compilers are in play and this card merges them: `mixed_moe_gemm_2stage_common.py`'s generic one (which supplies the multiple-of rule) and the mxfp4 `flydsl_moe2_layout_*` family (which supplies the `_sbm` naming and the SBM==BM==16 gate). They agree on the rule but were read separately; the older per-kernel-pair compilers were not confirmed to expose `sort_block_m` independently of `tile_m` at all.
- Only the legality boundary is sourced here — nothing in this corpus measures whether taking the freedom (sort_block_m larger than stage 2's own tile_m) is ever faster than matching it exactly.
- `sort_block_m` being a knob does not make the sort step's *grid* one, and on the mxfp4 path it is not. Two unrelated kernels both answer to the word 'sort' and must not be conflated. (a) `moe_sorting` — the FlyDSL one, open body, host-settable grid: `n_grid_blocks: fx.Int32` is a kernel parameter (`aiter/ops/flydsl/kernels/moe_sorting_kernel.py:755`) passed straight to `grid=(n_grid_blocks,1,1)` (`:773`) and computed host-side at `:1951`. (b) `sort_quant` / `moe_sort_quant` — what a decode trace on the mxfp4 two-stage path actually dispatches, and a *compiled HIP* kernel in `csrc/kernels/mxfp4_moe/moe_aux/moe_sort_quant.cuh`, built into `module_moe_mxfp4_aux.so`. There its CTA count is a template parameter, not an argument: `template <int N_CTAS, int THREADS_PER_CTA>` (`:34`) reaching `<<<N_CTAS, THREADS_PER_CTA, 0, stream>>>` (`:355`, `:405`, `:450`, `:466`). Decisively, the runtime dispatch is a string-keyed lookup into pre-codegen'd instances whose key is `"aux_sort_quant_NE{NE}_TOPK{TOPK}_MB{MB}_H{H}"` (`csrc/kernels/mxfp4_moe/moe_mxfp4_aux.cu:122`, `:125`) — **the key carries no CTA field**, and `gen_instances.py:287-290` emits exactly one instance per shape at `mb in (32,)`. So there is no second grid to select: changing it means editing the codegen and rebuilding the `.so`, which is neither a config-string edit nor a single-kernel-file edit.
- Do not reach for the open FlyDSL sorting kernel as the escape hatch without first checking it is the one running. In the GLM-5.2 vendored seam that path is gated off by default — `_USE_CK_MOE_SORTING` / `_USE_FLYDSL_MOE_SORTING` both read `os.environ.get(..., "0") == "1"` (`fused_moe_seam.py:2711-2712`) and nothing in that task or its harness sets either — so `moe_sorting()` falls through to the compiled `moe_sorting_opus_fwd`. Flipping the flag to open the body is not a free optimisation either: it swaps the baseline kernel for a different implementation, so any speedup measured after it is a comparison between two kernels, not a tuning result.

### Which stage-1 tile_m / cache-policy / inline-quant combinations will actually compile for an mxfp4 grouped MoE, and which are only apparently free?

**Card:** `grouped-moe-mxfp4-stage1-variant-whitelist` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `configuration_constraints`, `workgroup_tile`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- the a4w4 / mxfp4 stage-1 grouped GEMM (`mxfp4_gemm1.py`, dispatched as `flydsl_mxmoe_g1_a4w4_*`)

#### Try

- Pick `(BM, use_nt, inline_quant)` as ONE choice from `MXFP4_G1_VARIANTS[a_dtype]`, not as three independent knobs: fp4 admits `{(32,T,F),(32,F,F),(64,T,F),(64,F,F),(128,F,F),(16,T,T)}` and fp8 the same set minus `(64,T,F)`. Anything else raises `NotImplementedError` from `_assert_supported` even when every other check passes.
- Read the practical consequence directly off that set: tile_m 16 exists ONLY together with non-temporal B loads and inline quantization — which is exactly the deployed decode config `16x256x256_f16in_nt` — and tile_m 128 exists ONLY without non-temporal B loads. There is no `(16, *, False)` and no `(128, True, *)` to try.
- Treat BK as welded, not tuned: stage 1 asserts `BN in (64,128,256) and BK == 256`, so the K tile is not a knob and `K % 256 == 0` is a hard shape requirement (`D_HIDDEN % BK == 0`), as is `(2*D_INTER) % BN == 0` on the fused gate+up N.
- Expect the narrow combinations to carry extra riders: `BN == 64` is legal only at `BM == 32` with fp4 in AND fp4 out and no inline quant; `num_waves == 2` requires `BN == 64`; `k_wave > 1` requires `BM == 32`, forbids inline quant, and caps `num_waves * k_wave <= 8`.

#### Why this is a candidate

- The whitelist is not a style choice: each entry corresponds to a code path with its own LDS layout and operand staging, and the ones absent from it were never written, so a sweep over the cross product is a sweep over mostly-missing kernels.
- The `(16, True, True)` entry is why the GLM-5.2 decode config looks over-specified — at tile_m 16 the other two tokens are not independent decisions at all.

#### Keep as alternatives

- Sweeping BM, use_nt and inline_quant as a cross product — the obvious thing to try, and it spends most of its budget on combinations that cannot compile.

#### Evidence

- `src_0a0738d9fdf9642d` — `config_space` `MXFP4_G1_VARIANTS` at `aiter/ops/flydsl/mxfp4_kname.py:26-42`
- `src_c86e3bf3667b6875` — `config_validity` `flydsl mxfp4 gemm1 unsupported variant` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:180-184`
- `src_50249c22f9a4c75e` — `config_validity` `flydsl mxfp4 gemm1 requires a_dtype in {tuple(MXFP4_G1_VARIANTS)},` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:85-88`
- `src_96686bfe03b0b2ff` — `config_validity` `flydsl mxfp4 gemm1 requires BN in (64, 128, 256), got {BN}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:136-138`
- `src_c58e3e46708c46bc` — `config_validity` `flydsl mxfp4 gemm1 requires D_HIDDEN (K) % {BK} == 0, got H={D_HIDDEN}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:117-119`
- `src_4b58ee5ae49ae29c` — `config_validity` `flydsl mxfp4 gemm1 requires 2*D_INTER (N_OUT) % {BN} == 0,` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:131-134`
- `src_ae5bccba3310fa36` — `config_validity` `flydsl mxfp4 GEMM1 requires num_waves in (2, 4), got {num_waves}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:146-148`
- `src_a8b88b9d2224fb6b` — `config_validity` `flydsl mxfp4 GEMM1 two-wave specialization requires effective BN64` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:150-152`
- `src_ed75bfa8a52c3bfa` — `config_validity` `flydsl mxfp4 GEMM1 requires k_wave in (1, 2, 4), got {k_wave}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:158-160`
- `src_3d98038b022c3b55` — `config_validity` `D_HIDDEN/BK={k_tiles} must be divisible by k_wave={k_wave}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:172-174`

#### Limits

- The whitelist is read at current aiter main; it is a per-commit fact and an upstream change that adds a kernel widens it. Re-read `MXFP4_G1_VARIANTS` against the tree actually being built before concluding a combination is unavailable.
- Nothing here measures which admitted combination is fastest for a shape; the set is the legality boundary only.
- A deployment that vendors this kernel by flattening it into a single seam may carry a NARROWER set than upstream, and the narrowing is silent. Observed on the GLM-5.2 MXFP4 fused-MoE seam: its inlined copy of the variant set is the **fp8** set verbatim, so an fp4 `(64, True, False)` candidate — admitted by `MXFP4_G1_VARIANTS['fp4']` upstream — raises `NotImplementedError` inside that seam. Resolve the whitelist against the copy the build actually calls, not only against aiter main.
- **Whether `BN`/`BK` are knobs at all is a per-tree fact, and the two trees disagree — resolve it against the build, exactly as the limitation above says.** On aiter main the tile token is live: `_MXMOE_TILE_RE` parses `t{BM}x{BN}x{BK}` into `p1['BN']` (`mxfp4_kname.py:81-85`, `:107`), `_mxfp4_a4w4_stage1_fw` forwards `BN=p1['BN']` (`fused_moe.py:2505`), it is a `@functools.cache` key on `_get_compiled_mxfp4_gemm1_port` (`mxfp4_gemm1_kernels.py:13-23`), and `compile_gemm1_a4w4_port` accepts `BN in (64,128,256)` (`mxfp4_gemm1.py:1541-1552`). In the GLM-5.2 vendored seam NONE of that holds: there is no tile regex and no `nums['BN']` at all, `_mxfp4_a4w4_stage1_fw` reads only `p1['BM']`, `p1['inline_quant']`, `p1['use_nt']` (`fused_moe_seam.py:4514-4552`), and the compile entry hard-asserts `BN == 256 and BK == 256` (`:1380`, `:1588`). So in that build `BN`, `BK`, `hpf`, `num_waves`, `k_wave` and `kSplitK` are genuinely INERT NAME TEXT — a renamed tier is the same compiled kernel, and a `BN=128` row is a `BN=256` kernel wearing a `BN=128` name. Measured: sweeping them at tiers 8/16 returned the null floor, leaving `xcd_swizzle` as the only live integer there (0.99916 / 0.99719, both null).
- Do not generalise that inertness back onto aiter main, and do not let the reverse mistake through either: reading main's richer parser and concluding the deployed null was 'a measured negative on a live knob' is wrong on this seam. Real `BN=128` here is a DEVICE-CODE change, so it is not reachable from the config string in either direction. Two concrete blockers in `_gemm1_body`, both on the non-interleaved arm: `tile_il = n_block_idx*16 + wave*4 + j` (`fused_moe_seam.py:843`) has the BN=256 geometry baked in as literals, where the interleaved arm one branch up (`:840`) still writes `BN` and `BN//4` symbolically; and `np_gate = n_block_idx*(BN//64) + wave` (`:855`) tiles exactly only while `BN//64 == 4` matches `wave in 0..3` — at `BN=128` it becomes `n_block*2 + wave` and the upper waves run into the next n-block's scales.
- **'Inert' scopes the config string, not the axis.** A later campaign paid the device-code change those two blockers describe and BN 256->128 then moved weighted 1.1294 -> 1.1502 as a standalone edit on gfx950 decode, anchoring a ~1.32x stack — see the learned card `narrow-bn-to-multiply-the-decode-grid-on-an-mxfp4-two-stage--moe-grouped-gemm-gfx950-decode`. So read this entry as 'you cannot reach BN from the tier name in that seam', never as 'narrow BN does not pay'; a cost ceiling fitted to the wide-tile implementation mispriced it by ~10pp.

### How should a FlyDSL GEMM map workgroup IDs onto output tiles and XCDs, and where does AITER already do it?

**Card:** `gemm-workgroup-xcd-mapping` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `memory_access`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a multi-XCD part (MI300X / MI355X: 8 XCDs, each with its own L2)
- the grid launches enough workgroups that their placement across XCDs matters

#### Try

- Know the starting point: no workgroup remap was found in the dense FlyDSL HGEMM or A8 sources scanned here (only the grouped-GEMM kernel has one), so consecutive workgroups follow the hardware's round-robin across XCDs and neighbours that share an A or B panel land on different L2s (background).
- Port the remap as a block-id permutation in the kernel prologue, the way the FlyDSL grouped-GEMM kernel does behind `xcd_swizzle`: with 8 XCDs, wgid = (linear_id % 8) * (num_wgs / 8) + linear_id / 8, then rasterise M in groups of `xcd_swizzle` rows before decoding the (m, n) tile.
- Or follow Triton's shared helpers: `remap_xcd` / `remap_xcd_chunked` regroup program IDs per XCD, then `pid_grid(..., GROUP_SIZE_M)` walks M in groups so consecutive programs reuse one B panel.
- Keep the unmapped kernel as the control arm (`xcd_swizzle=0` is the grouped kernel's default), gate the remap by grid size, and prove every tile is still visited exactly once when the grid is not a multiple of the XCD count.

#### Why this is a candidate

- Two implementations in AITER already solve this and state the formula in source; the dense FlyDSL kernels are the ones that do not.
- Whether it pays depends on how much panel reuse crosses XCDs for the shape, which is why it is a candidate with a control arm and not a default.

#### Keep as alternatives

- No remap — the default dispatch order.
- Grouped-M rasterisation alone (`GROUP_SIZE_M`) without an XCD remap.

#### Same question in other implementations

- **triton** — `remap_xcd` / `remap_xcd_chunked` plus `pid_grid` with `GROUP_SIZE_M` in the shared pid helper, called from the GEMM kernels with `NUM_XCDS`. Evidence: `src_1b8ded5e4825e5e1` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:28-54`), `src_a0815635126a1ea6` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:10-24`), `src_d760ac9f0e4bff02` (`aiter/ops/triton/utils/_triton/pid_preprocessing.py:58-80`), `src_334e95f3f9406108` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:105`), `src_de6d336c2ee7601b` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:76`)

#### Evidence

- `src_69a5695f3c33fd50` — `grid_mapping` `xcd_swizzle` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:132`
- `src_3a83a99245462451` — `grid_mapping` `xcd_swizzle` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:523-558`
- `src_923af36a5869aea6` — `grid_mapping` `_wgs_per_xcd` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:544`

#### Background (always-on reference docs, not measurements)

- [`optimization/xcd_l2_locality.md:36-49`](../optimization/xcd_l2_locality.md)
- [`hardware/shared/l2_xcd_swizzle.md`](../hardware/shared/l2_xcd_swizzle.md)

#### Limits

- The grouped kernel's tile decode after the remap is MoE-specific; the remap ports, the decode has to be rewritten for a dense grid.
- Measured effects of XCD remapping live in learned cards and expert skills behind their switches; this card carries none.

### When does a GEMM case need split-K to fill the machine, and what does choosing it commit the rest of the kernel to?

**Card:** `gemm-split-k-when-grid-underfills` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `workgroup_tile`, `execution_geometry`, `epilogue`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a case whose output tile count ceil(M/tile_m)*ceil(N/tile_n) is well below the CU count — 304 on MI300X, 256 on MI350X/MI355X — the usual state of decode and skinny-M shapes
- K is deep enough to split: every split still runs several tile_k iterations (the HGEMM derivation asks for 2 to 8 block-K loops per split for its non-base split factors)

#### Try

- Count output tiles against CUs before tuning anything else in the case. A grid that is a fraction of the CU count leaves the rest of the machine idle, and no schedule or layout change later in the kernel recovers it: this is a partition decision, made before the pipeline work.
- Raise split_k until the workgroup count approaches the CU count, taking values from the kernel's legal set — for the FlyDSL HGEMM, split_k divides K and K/split_k is a multiple of tile_k (flydsl-hgemm-config-space). Past that point a larger split adds combine traffic without adding parallelism.
- Count rounds, not workgroups: AITER's ASM launchers score each (tile, split) by ceil(tiles*split / CUs) and keep the fewest rounds, preferring fewer idle CUs in the last round (gemm-asm-tile-inventory-and-selection). One workgroup past a CU multiple costs a whole extra round, so step the split down or the tile up rather than overshoot.
- Choose the combine together with the split, because split-K is not a knob on its own: the FlyDSL HGEMM initialises the output from split 0 and atomically adds the others (flydsl-splitk-reduction-contract), Triton writes fp32 partials and runs a separate reduce kernel, ASM multiplies the launch grid by the split. An authored kernel has to pick one and supply its host-side state.
- Re-check numerics with the split: a different accumulation order — and, in FlyDSL's atomic combine, adding partials already rounded to the output dtype — changes the result, so gate against the task's oracle at its stated tolerance.
- Seed from where AITER shipped split-K: FlyDSL bf16 on gfx950 selects split_k > 1 in 83 of 86 rows at M<=16 and 38 of 45 at M 17-64, and split_k = 1 in every row above M = 256; Triton's gfx942 fp8 block-scale configs vary NUM_KSPLIT (1 to 16) only in the buckets up to M<=512 and fix it at 1 from M<=1024 on.

#### Why this is a candidate

- This is the grid-fill rule of the background docs — compute the output tiles and split K when they are far below the CU count. The FlyDSL derivation checks divisibility only, so the occupancy half of the decision is written nowhere in the kernel.
- Both backends' shipped selections put split-K where that rule predicts it, in the small-M buckets, and turn it off for large M.

#### Keep as alternatives

- Keep split_k = 1 and shrink tile_m / tile_n to raise the tile count: nothing to combine, but less MFMA work per workgroup (background).
- Stream-K or a persistent grid sized to the CU count, which also balances a tile count just off a CU multiple — no FlyDSL precedent in AITER.
- split_k = 1 as the control arm.

#### Same question in other implementations

- **triton** — NUM_KSPLIT partitions K inside the kernel and SPLITK_BLOCK_SIZE sizes each partition; a separate reduce kernel sums the fp32 partials. The block-scale kernel carries the same two parameters. Evidence: `src_0f50a57ed622a4f4` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:77`), `src_8f1fdafc50973349` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:209`), `src_4fbd4a0e0992c880` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:80`), `src_a6edf5790cd2f1f1` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:81`)
- **asm** — the launcher scores every (kernel, split) of the architecture by rounds = ceil(tiles*split / CUs), keeps the fewest rounds, and multiplies the launch grid by the split; the bf16 launcher caps split at min(CUs / output tiles, 16, K / sub_k). Evidence: `src_4044f249c79abf90` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:250`), `src_b07a8dc43c913b9d` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:353`), `src_3528bd18e9f8f895` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:119`), `src_787ebf8727d0c7e1` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:138`)
- **hip** — the decode skinny kernels split K across the lanes of a wave instead of across workgroups: weight rows are spread over CUs x YTILE (`mindiv`) and each lane's partial is reduced with cross-lane shuffles, so nothing is combined in memory. Evidence: `src_4afbdde799d96232` (`csrc/kernels/custom_kernels.cu:1738`), `src_0b8de37c2b304d4a` (`csrc/kernels/custom_kernels.cu:350`)

#### Evidence

- `src_642476c9b693d5f1` — `config_space` `_hgemm_split_k_options` at `aiter/ops/flydsl/gemm_kernels.py:229-246`
- `src_58d4bf39e7ca8f03` — `config_validity` `Invalid split-K: K={k} must be divisible by split_k={split_k}` at `aiter/ops/flydsl/gemm_kernels.py:397-399`
- `src_fce6be7eb6b247d6` — `config_space` `HGEMM_BASE_SPLIT_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:75`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_ac0c03746ab634f6` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 45 shipped row(s)
- `cfg_ae017b25b6473300` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M257-1024`: 29 shipped row(s)
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_c23460f537b313ac` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_16` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_da4a49e152c50056` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_1024` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/tuning.md:16-37`](../operators/splitk_streamk_gemm/tuning.md)
- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)
- [`languages/flydsl/knobs.md:52-60`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:68-76`](../languages/flydsl/patterns.md)

#### Limits

- Whether split-K pays for a shape is a measurement; the tile count says when it is worth trying, not that it wins.
- The CU counts are hardware facts from the background docs, not from this source.
- A split that fills the grid can still be illegal for its combine: the FlyDSL HGEMM counters allow at most 128 output tiles when split_k > 1.

### Which tile, warp and split-K values will the FlyDSL HGEMM family actually accept, and how does a shape narrow them?

**Card:** `flydsl-hgemm-config-space` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- a configuration is being chosen or searched for a known M, N, K

#### Try

- Search tile_m within (16, 32, 48, 64, 80, 96, 112, 128, 160, 256), tile_n within (64, 128, 160, 192, 256) and tile_k within (64, 96, 128, 160, 256) rather than treating the 128x128x64 default as the space.
- Search the warp layout and B staging as one knob: the registry enumerates exactly five (block_m_warps, block_n_warps, b_to_lds) variants — (1,2,F), (1,4,F), (2,2,F), (1,4,T), (2,2,T).
- Cap tile_m at max(96, align_up(2*M, 16)): a tile_m far above M pays for rows that do not exist, and the shipped derivation refuses those candidates outright.
- Take split_k from 1..32, keeping only values that divide K and leave K/split_k divisible by tile_k; every candidate must clear that filter, including 1, 2, 4, 8 and 16. Those five need nothing further; any other divisor also has to leave between 2 and 8 block-K loops per split.
- Whether split_k > 1 is worth generating at all is the grid-fill decision in gemm-split-k-when-grid-underfills: the derivation here checks divisibility, never occupancy.
- Seed from what AITER shipped for your bucket before sweeping (FlyDSL tuned-DB table below): on gfx950 bf16 every shipped FlyDSL selection used either (2,2,b_to_lds=True) or (1,4,b_to_lds=False); the M<=64 buckets are dominated by tile_m=32 with split_k 4-16; every bucket above M=256 uses split_k=1, and M 257-1024 favours the non-power-of-two tile_m=96.

#### Why this is a candidate

- AITER states the space as module-level option tuples and two derivation helpers, not as one default; the default is one point inside it.
- The derivations encode the shape-dependence directly — tile_m is bounded by M, and split_k by the divisibility of K — so the narrowing rule is readable rather than guessed.
- This card says which candidates are legal and worth generating. It does not say which one wins; that is what the benchmark is for.

#### Keep as alternatives

- Keep the library default 128x128x64 with split_k=1 as the control arm to measure against.
- Fix the tile and search only split_k when N or K constrain the tiling to one legal shape.

#### Evidence

- `src_426b8960fb14a04f` — `config_space` `HGEMM_TILE_M_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:74`
- `src_62ce0ec6b152ecaa` — `config_space` `HGEMM_TILE_N_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:72`
- `src_3c72689d786ca8be` — `config_space` `HGEMM_TILE_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:73`
- `src_fce6be7eb6b247d6` — `config_space` `HGEMM_BASE_SPLIT_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:75`
- `src_96320a5bab17a8f5` — `config_limit` `HGEMM_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/gemm_kernels.py:76`
- `src_562e9940a52a202f` — `config_space` `_hgemm_tile_m_options` at `aiter/ops/flydsl/gemm_kernels.py:222-226`
- `src_642476c9b693d5f1` — `config_space` `_hgemm_split_k_options` at `aiter/ops/flydsl/gemm_kernels.py:229-246`
- `src_27d99a67a1d27ccc` — `config_space` `KERNEL_CONFIG_VARIANTS` at `aiter/ops/flydsl/gemm_kernels.py:79-105`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_ac0c03746ab634f6` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 45 shipped row(s)
- `cfg_147eed9b10acbc4e` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: 52 shipped row(s)
- `cfg_ae017b25b6473300` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M257-1024`: 29 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/tuning.md:16-19`](../operators/splitk_streamk_gemm/tuning.md)
- [`languages/flydsl/knobs.md:42-50`](../languages/flydsl/knobs.md)
- [`languages/flydsl/authoring_gemm_levers.md:103-117`](../languages/flydsl/authoring_gemm_levers.md)
- [`optimization/occupancy_and_registers.md:54-67`](../optimization/occupancy_and_registers.md)

#### Limits

- These are the values the library is willing to compile, not values anybody measured as best; every candidate still has to be timed on the target box.
- Shipped selections come from AITER's tuner on its own box and competing set; they seed a search, they do not end one.
- The option tuples are the generic HGEMM family's. The small-M family has its own, wider, N space — see flydsl-small-m-hgemm-family.
- A candidate that is inside the space can still be rejected at compile time by the tiling and LDS rules — see flydsl-hgemm-tiling-validity.

### Which tile, LDS-stage, epilogue, async and occupancy combinations will the FlyDSL A8 (fp8/int8) preshuffle GEMM build, and how do they differ by architecture?

**Card:** `flydsl-a8-preshuffle-config-space` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `tunable_surface`, `pipeline_schedule`, `epilogue`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`

#### Use when

- target_language=flydsl
- fp8 or int8 GEMM with a preshuffled B (the A8 preshuffle family, or an authored kernel modelled on it)
- a configuration is being chosen or searched

#### Try

- Search tiles from the family's own catalogs, not from the HGEMM tuples: a shared lds_stage=2 catalog spanning tile_m 16-256 (tile_k up to 512 for small M), a smaller separate catalog for lds_stage=1, the tile_k=64 tiles 64x256x64 and 128x128x64 only on gfx942, and 256x256x128 only on gfx950.
- Treat lds_stage, use_cshuffle_epilog, use_async_copy and waves_per_eu as a four-way sweep: the tuner crosses lds_stage in {1,2}, cshuffle in {0,1}, async in {0,1} and waves_per_eu in {0..4}.
- Drop waves_per_eu values above the register estimate before compiling, as the tuner does with `_estimate_max_wpe`: tile_m*tile_n/256 accumulator VGPRs per thread for the 256-thread block, times 1.5 for pipeline buffers, against the 512-VGPR file.
- Estimate LDS with the family's own formula: only the A tile is staged (tile_m*tile_k*elem_bytes, halved for fp4); a CShuffle epilogue adds a 2*tile_m*tile_n staging buffer that shares the ping/pong space; lds_stage=2 keeps two buffers. Compare with `max_lds_bytes_for_tune()`, the per-arch budget.
- Start from the defaults the tuner falls back to — gfx942 128x128x128, gfx950 128x256x256, both lds_stage=2 and waves_per_eu=2 — plus the small-M defaults 16x64x512 and 32x64x512.
- Then seed from what AITER shipped for your bucket (FlyDSL tuned-DB table below): on gfx950 fp8 the M<=16 selections are mostly 16x64x512 and the M>1024 selections mostly 128x256x128 or 256x128x128, while lds_stage, CShuffle, async copy and waves_per_eu each take both values inside the same bucket — those four stay search axes.

#### Why this is a candidate

- AITER states the A8 space as module-level catalogs, a combo sweep and two estimators in its tune module, so the space is readable rather than guessed.
- The catalogs differ by architecture in ways a copied config will not reveal: a gfx942 tile_k=64 tile does not exist on gfx950.

#### Keep as alternatives

- Keep the family's per-arch default kernel as the control arm.
- Use the HGEMM family when the operands are not quantized; the two families share neither knobs nor catalogs.

#### Evidence

- `src_f6f28f20adae28cd` — `config_space` `_base_tiles_lds2_common` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:188-218`
- `src_af9ebdf13bdc947e` — `config_space` `_base_tiles_lds2_942_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:221-224`
- `src_273e37ba3a8cefa8` — `config_space` `_base_tiles_lds2_950_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:227-229`
- `src_6bd4c9e3168467b4` — `config_space` `_base_tiles_lds1` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:232-241`
- `src_055d381c0e94f538` — `config_space` `_LDS_STAGES` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:246`
- `src_84d3653b2aaf49f3` — `config_space` `_CSHUFFLE_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:247`
- `src_623470961bcb8c24` — `config_space` `_ASYNC_COPY_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:248`
- `src_32599cca99565a9f` — `config_space` `_WAVES_PER_EU` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:249`
- `src_ff07751a8278d3bb` — `config_validity` `_estimate_max_wpe` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:267-279`
- `src_c701f7377a04c988` — `config_validity` `preshuffle_gemm_estimated_lds_bytes` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:119-157`
- `src_771ddc6685eb37b6` — `config_validity` `max_lds_bytes_for_tune` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:177-179`
- `src_bdc2de724bf3f124` — `config_space` `kernels_list_942` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:298-300`
- `src_89bc3925dd87e7aa` — `config_space` `kernels_list_950` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:301-303`
- `src_865f0e9972c72d26` — `config_space` `default_kernels_dict_942` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:306-314`
- `src_63fcc4e0eff52ae1` — `config_space` `default_kernels_dict_950` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:316-321`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 274 shipped row(s)
- `cfg_2f03fe974e5d2b5c` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M17-64`: 133 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 661 shipped row(s)
- `cfg_9636311ea3a3ce0d` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: 178 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:82-87`](../languages/flydsl/knobs.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:60-78`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- These are the configurations the tuner will build and the ones it shipped, not a ranking; every candidate still needs the target box.
- The catalogs belong to AITER's vendored FlyDSL at the pinned commit; an authored kernel with a different staging scheme needs its own estimate.
- Shipped FlyDSL selections exist only for gfx950 per-token fp8; there is none for gfx942 and none for block-scaled fp8.

### Which configuration does the Triton block-scale baseline run for each case, and what does AITER ship instead where it tuned small and large M separately?

**Card:** `gemm-blockscale-shipped-bucket-structure` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`, `implementation_variant`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`

#### Use when

- target_language=flydsl
- an fp8 128x128 block-scale workload measured against AITER's Triton kernel, especially one whose cases span decode and prefill (for example M=8 and M=32768 in one kernel)

#### Try

- Find the baseline's configuration for every case before planning, in the task's frozen baseline or through the lookup: a per-(N, K) file if AITER tuned that shape, else the generic file, then M_LEQ / M_GEQ buckets, then `any`. gfx942 ships block-scale files for nine (N, K) shapes; the generic file's only entry is `any` — a 128x128x128 tile, NUM_KSPLIT = 1, 4 warps — so an untuned shape runs that one configuration at every M.
- Price the fallback per case: at decode M=8 with N=4096 that tile launches 32 workgroups on a 304-CU MI300X, a grid-fill gap a FlyDSL decode kernel can close (gemm-split-k-when-grid-underfills); at large M the same tile sits inside the range AITER ships for the shapes it did tune.
- Where AITER did tune a shape, small and large M are different structures: on gfx942 the M<=32 buckets ship BLOCK_SIZE_M mostly 4 to 16, BLOCK_SIZE_N mostly 16 and mostly one or two warps, and vary NUM_KSPLIT from 1 to 16; from M<=1024 on NUM_KSPLIT is 1, with BLOCK_SIZE_M 64 to 256, BLOCK_SIZE_N mostly 128 and mostly four to eight warps.
- Hold the scale block fixed across regimes: BLOCK_SIZE_K = 128 and the 16x16 MFMA (matrix_instr_nonkdim = 16) are shared by every gfx942 bucket, so decode and prefill differ in tile, split and warps, not in how K is blocked (gemm-blockscale-k-loop-scale-application).
- Mirror the tuned shapes rather than the fallback: one entry point that dispatches on M to a kernel specialised for each regime, since the tuned tiles differ by more than an order of magnitude in BLOCK_SIZE_M.
- Know the production competitors per regime too: AITER's tuned block-scale B-preshuffle database on gfx942 selects CK in 16 of 18 rows at M<=16 and ASM in 32 of 38 above M=4096.

#### Why this is a candidate

- In a Triton-to-FlyDSL block-scale task the frozen baseline is this Triton kernel with whatever the lookup returned, so the per-case target is readable before any kernel is written.
- The shipped tables are the only record of which tile and split AITER chose per regime once it did tune a shape.

#### Keep as alternatives

- One configuration for every case — simpler, and the control arm; expect it to trail on the regime it was not chosen for.

#### Evidence

- `src_2a228a58ade64092` — `config_lookup` `get_gemm_config` at `aiter/ops/triton/utils/gemm_config_utils.py:142-177`
- `src_318b927528cb6cda` — `tunable_param` `BLOCK_SIZE_M` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:76`
- `src_caaad70b7145147d` — `tunable_param` `BLOCK_SIZE_N` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:77`
- `src_4fbd4a0e0992c880` — `tunable_param` `NUM_KSPLIT` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:80`
- `src_a6edf5790cd2f1f1` — `tunable_param` `SPLITK_BLOCK_SIZE` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:81`
- `cfg_d6c5b6bf7798a092` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_4` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_c23460f537b313ac` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_16` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_89a57b3ca5ee84c7` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_32` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_658519ce57ddcf17` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_512` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_da4a49e152c50056` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_1024` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_1311810498769365` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_2048` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_3eff28954f18b5fe` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `any` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `sel_f7e394faa23e37e6` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 66 · triton 4 · asm 2 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 26 · opus 9 (rows)

#### Background (always-on reference docs, not measurements)

- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)

#### Limits

- Shipped configurations are selections without timings; the tuner's box and competing set are not recorded.
- These are Triton knob names (every group is on gemm_triton_seeds.md): carry over tile shape, split and warp count as intent, not as spellings.
- The bucket groups pool AITER's nine tuned (N, K) files; a task shape outside them has no shipped small-M configuration at all.

### Which tiles and split-K capabilities does AITER's hand-written ASM ship per architecture, and how does its launcher choose among them for a shape?

**Card:** `gemm-asm-tile-inventory-and-selection` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `workgroup_tile`, `work_partition`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a tile and split are being chosen per case, and the vendor's own choices for the same dtype and architecture are worth knowing

#### Try

- Read the per-architecture inventory as the vendor's tile set: fp8 128x128 block scale is the same six kernels on gfx942 and gfx950 — tile_n = 128, tile_m in 32, 48, 64, 80, 96, 128, all split-K capable and B-preshuffled. bf16 is 22 kernels on gfx942 (tile_n = 64, tile_m 32 to 160, 12 split-K capable) and 24 on gfx950 (adding two 256x256 kernels). fp4 exists only on gfx950 (35 kernels, 2 split-K capable).
- Choose tile and split the way the launcher does, per case rather than once: for every kernel of the architecture (the block-scale launcher also requires tile_n to divide N), and every split from 1 to 16 it supports, count the rounds `ceil(ceil(M/tile_m) * ceil(N/tile_n) * split / num_cu)`; keep the fewest rounds, and among equal rounds the one with fewer idle CUs in the last round or the larger `tile_m * tile_n / (tile_m + tile_n)`.
- The bf16 launcher also caps the split at `min(num_cu / output_tiles, 16, K / sub_k)` — split only up to what fills the machine.
- Under block scale keep tile_n a search axis, with 128 as ASM's point: every block-scale ASM kernel uses tile_n = 128 on both architectures (one N tile, one W scale per K block), while Triton's gfx942 block-scale configs fix GROUP_N = 128 but ship BLOCK_SIZE_N 16 or 64 at M <= 8 and mostly 128 at M 1025-2048 (gemm-blockscale-k-loop-scale-application).

#### Why this is a candidate

- The inventories state what exists per architecture; the launchers state the rule that picks from it. Both are the vendor's grid-fill heuristic written as code, and `num_cu` is read at run time, so the same rule serves the 304-CU MI300X and the 256-CU MI355X.

#### Keep as alternatives

- A single fixed tile for every case — the control arm, and what an untuned Triton baseline does.

#### Evidence

- `src_3528bd18e9f8f895` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:119`
- `src_c2ed9024f2039d1f` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a16w16.cu:145`
- `src_787ebf8727d0c7e1` — `grid_fill` `max_splitk, std::min(std::min(static_cast<int>(num_cu / pure_tg_num), …` at `csrc/py_itfs_cu/asm_gemm_a16w16.cu:138`
- `src_0558cdca6ff86ab9` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a8w8.cu:98`
- `src_fa4eb9ba8bba7192` — `grid_fill` `local_round, tg_num, num_cu` at `csrc/py_itfs_cu/asm_gemm_a4w4.cu:124`
- `src_3865c44e30f36de9` — `asm_tile` `tile_m=128, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`
- `src_fcf5d73d12271d55` — `asm_tile` `tile_m=32, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`
- `src_c792a003b578b049` — `asm_tile` `tile_m=48, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:4`
- `src_acee84984c040f98` — `asm_tile` `tile_m=64, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:5`
- `src_fe82852692f7eaca` — `asm_tile` `tile_m=80, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:6`
- `src_8475d71b5a90595e` — `asm_tile` `tile_m=96, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:7`
- `src_8866d3dbeb612f3a` — `asm_tile` `tile_m=128, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`
- `src_a4937eaa160077d1` — `asm_tile` `tile_m=32, tile_n=128, split_k_capable=1, bpreshuffle=1` at `hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_1311810498769365` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_2048` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))

#### Background (always-on reference docs, not measurements)

- [`optimization/wave_and_grid_sizing.md:49-61`](../optimization/wave_and_grid_sizing.md)
- [`operators/splitk_streamk_gemm/tuning.md:16-37`](../operators/splitk_streamk_gemm/tuning.md)

#### Limits

- The inventory lists what AITER ships, not which kernel wins a shape; the launcher's rule is a heuristic, and the task's benchmark decides.
- ASM tiles are binaries: the tile shapes transfer as intent; the MFMA schedule inside them is not visible.

### How does AITER's hand-written HIP skinny GEMM organise work for M <= 16, and what can a FlyDSL decode kernel take from it?

**Card:** `gemm-skinny-decode-wave-split` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `work_partition`, `execution_geometry`, `memory_access`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `skinny_gemv_decode`, `scaled_quant_gemm`
- `target_languages`: `flydsl`
- `regimes`: `decode`

#### Use when

- target_language=flydsl
- decode: M is a handful of rows (AITER's skinny route covers M <= 16), so an MFMA tile along M is mostly padding

#### Try

- Stage the whole activation in LDS when it fits and keep it there for the workgroup's lifetime: the `_sml_` variant reserves 64 KB — one workgroup per CU on gfx942's 64 KB LDS, as the source comment intends, while gfx950's 160 KB would fit two — and is chosen when K*M <= 32K elements for bf16/fp16 (64K for the fp8 variant); up to 1.2x of that the `_hf_` variant, beyond it `_big_`.
- Split the output rows over every CU and the K dimension over the lanes: each wave owns YTILE rows of the weight, waves per workgroup are sized with `mindiv(rows, CuCount * YTILE, WvPrGrp)`, and each lane takes A_CHUNK elements of K per step.
- Reduce the lanes' partial dot products inside the wave with cross-lane shuffles (`__shfl_xor`, `__shfl`) — no second kernel and no atomics.
- Treat the knobs as a family, not constants: THRDS, YTILE, WvPrGrp, A_CHUNK and UNRL are template parameters of every variant.
- Compare with the FlyDSL small-M HGEMM (flydsl-small-m-hgemm-family), which answers the same regime with 16-row MFMA tiles: choose the structure for a FlyDSL decode kernel by measurement, not by precedent.

#### Why this is a candidate

- It is AITER's default bf16/fp16 decode kernel (gemm-default-backend-routing), and its launcher is the grid-fill rule for decode written in HIP.

#### Keep as alternatives

- An MFMA tile with split-K across workgroups (gemm-split-k-when-grid-underfills) — what the FlyDSL HGEMM and the Triton baselines do at decode.

#### Evidence

- `src_b8e76e6d00fd7ca3` — `kernel_family` `K_in * N_in <= 64 * 1024` at `csrc/kernels/custom_kernels.cu:2229`
- `src_abfa46c903cb04ea` — `kernel_family` `K_in * N_in <= 32 * 1024` at `csrc/kernels/custom_kernels.cu:1736`
- `src_abb2ef768886f8dc` — `kernel_family` `K_in * N_in <= 32 * 1024 * 1.2` at `csrc/kernels/custom_kernels.cu:1742`
- `src_4afbdde799d96232` — `grid_fill` `__wvPrGrp, mindiv(M_in, CuCount * _YTILEs, _WvPrGrp)` at `csrc/kernels/custom_kernels.cu:1738`
- `src_e32202e7a829f4ab` — `grid_fill` `__wvPrGrp, mindiv(M_in, CuCount * _YTILEs, _WvPrGrp)` at `csrc/kernels/custom_kernels.cu:2231`
- `src_0b8de37c2b304d4a` — `lane_reduction` `__shfl_xor` at `csrc/kernels/custom_kernels.cu:350`
- `src_c99341de831ed9d8` — `lane_reduction` `__shfl` at `csrc/kernels/custom_kernels.cu:1951`
- `src_f9a2106c23412f25` — `lds_alloc` `__shared__` at `csrc/kernels/custom_kernels.cu:713`
- `src_d33d9c364eba88bd` — `waves` `WvPrGrp* THRDS` at `csrc/kernels/custom_kernels.cu:688`
- `src_d78de056f0337d85` — `tunable_param` `typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int…` at `csrc/kernels/custom_kernels.cu:687`
- `src_5147e7cddd350323` — `tunable_param` `typename scalar_t, typename fp8_t, int THRDS, int YTILE, int WvPrGrp, …` at `csrc/kernels/custom_kernels.cu:1775`
- `src_2577a978dcbebafb` — `kernel_family` `is_skinny_default_shape` at `aiter/tuned_gemm.py:76-97`

#### Background (always-on reference docs, not measurements)

- [`operators/skinny_gemv_decode/tuning.md:16-33`](../operators/skinny_gemv_decode/tuning.md)
- [`languages/flydsl/knobs.md:77-80`](../languages/flydsl/knobs.md)

#### Limits

- The HIP family is fp16/bf16 plus an fp8 variant that multiplies by two scalar (per-tensor) scales at the end; a block-scale decode kernel still needs the per-K-block scale loop.
- Its structure is a precedent for a FlyDSL decode kernel, not a drop-in: FlyDSL has no skinny kernel in AITER to copy.

## 4. Layout and address computation

- `operand_layout` — How are operands transformed before and inside LDS?

Also bears on this step: `flydsl-a8-data-movement-helpers` (step 5), `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` (step 8).

### Stage 1 writes block scales that stage 2 reads — what decides their layout, and who else has to agree?

**Card:** `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `configuration_constraints`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- a two-stage mxfp4 grouped MoE where stage 1's fp4 output and its e8m0 scales are stage 2's A operand

#### Try

- Do not tune the scale layout: it is derived, by `native_scale_layout_for(BM, out_dtype)`, which is true exactly when `BM == 16 and out_dtype == 'fp4'`. Stage 1 enforces the same predicate as a hard check, so setting the flag independently of the tile is not possible.
- Expect FOUR values to move together when stage 1 picks tile_m 16: stage-1 BM, the sort block size, stage-2 tile_m, and stage-2 `SBM`. Stage 2's BM16 A-scale chunk addressing is selected by `SBM == BM`, not by `BM` alone, and the host refuses a mismatch at runtime (`FlyDSL v2 stage2 sorting layout mismatch: moe_sorting uses block_m=..., but the kernel expects sort_block_m=...`).
- Read the two layouts as different shapes, not different strides: native emits one padded chunk per BM16 block with each scale byte duplicated into both halves of a 16-bit store (so stage 2's two N16 MFMA selectors each find one), while the regular layout packs two BM16 blocks into one 32-row chunk with byte-granular stores. Changing one side alone produces a silently wrong result, not a crash.
- When down-tiling stage 2 for a small-M bucket, stop at the sort block: tile_m 16 is only a candidate when the sort block is also 16, which is why AITER's own stage-2 candidate generator excludes it otherwise.

#### Why this is a candidate

- The layout is a serialization format between two kernels; a format needs one owner, and AITER makes the owner a pure function of the tile so that no caller can disagree with another.
- The failure mode if they do disagree is numerical, not structural: stage 2 reads a real scale from the wrong row, so parity degrades without any out-of-bounds signal.

#### Keep as alternatives

- One scale layout for all tile sizes — fewer coupled values and no cross-stage contract, but it costs stage 2 an atomic merge at BM16 that the duplicated-byte narrow store replaces.

#### Evidence

- `src_200b3893f45ec954` — `config_validity` `flydsl mxfp4 GEMM1 native scale layout requires BM16 FP4 output` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:154-156`
- `src_200b3893f45ec954` — `config_validity` `flydsl mxfp4 GEMM1 native scale layout requires BM16 FP4 output` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:154-156`
- `src_271f42b2696a0d4c` — `config_validity` `SBM ({SBM}) must be a multiple of BM ({BM})` at `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:158`
- `src_271f42b2696a0d4c` — `config_validity` `SBM ({SBM}) must be a multiple of BM ({BM})` at `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:158`
- `src_9c1b248e8f431365` — `scale_operand` `block_scale` at `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py:1131-1134`

#### Limits

- The stage-1 side is cited at its enforcement site (`_assert_supported`); `native_scale_layout_for` itself is a short helper that the extractor's rules do not reach, so its body is quoted from a read rather than from an evidence record.
- No runtime assertion was found that stage 1's parsed `out_dtype` equals stage 2's parsed `a_dtype` on the dedicated mxfp4 path — that agreement is trusted to whichever tuner paired the two names in the CSV row. Treat a hand-edited CSV pair as unchecked.

### What stops an mxfp4 grouped-MoE tile from asking for more LDS than the hardware has?

**Card:** `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `pipeline_schedule`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- an mxfp4 grouped-MoE stage whose A tile and f32 accumulator both live in LDS

#### Try

- Do not expect a guard rail: nothing in either stage compares the computed `lds_bytes` against the hardware LDS size. The only budget constant, `LDS_2WG_LIMIT` (65 KiB, two workgroups per CU), feeds `default_k_stages`, which tries the deep pipeline, then the shallow one, and if NEITHER fits returns the deep one anyway. An over-budget tile is a link-time or occupancy surprise, not a compile error.
- Size the tile by hand from the stage's own formula before trusting it. Stage 1 UNIONS the A region with the accumulator: `max(k_wave*kAStages*BM*KH_TILE + kSubBlocks*(K/BK)*256, k_wave*(BM/epi_splits)*(BN+4)*4)`, where `KH_TILE` is `BK` for fp8 A and `BK/2` for fp4. Stage 2 does NOT union — its layout is sequential, A slots then accumulator — so the same tile numbers cost more LDS in stage 2 than in stage 1.
- Keep the four dwords of accumulator row padding when editing the epilogue: `ACC_LDS_STRIDE = BN + 4` rotates each four-row store group by 16 banks on gfx950 and exists to avoid a 4-way bank conflict from a BN-aligned stride. Stage 2's accumulator has no such padding, so a store pattern copied from one into the other inherits the conflict.
- Watch for the side effect of the pipeline depth landing on 1: that case also flips a compiler hint (`amdgpu-sched-strategy: iterative-minreg`), so a shallow-K config is not only shallower, it is scheduled differently.

#### Why this is a candidate

- Treating the union as two separate allocations overstates stage 1's LDS by the smaller of the two regions, which is enough to reject a tile that actually fits.
- An optimizer that assumes an illegal tile will fail to build will read the resulting occupancy drop as 'this tile is slow' instead of 'this tile did not fit'.

#### Keep as alternatives

- An explicit assert against the arch LDS size — it would turn a silent occupancy loss into a build failure, which is what the dense FlyDSL tiling-validity card expects, but AITER does not do it here.

#### Evidence

- `src_2f735e2089e8927d` — `config_limit` `MIN_N_BLOCKS_FOR_SHALLOW_K, 4` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1475`
- `src_dfcd02dec23c4cef` — `lds_pad` `ACC_LDS_PAD_DW, 4` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:44`
- `src_74352923ff7ef43b` — `tile_shape` `BM, 32` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1498`
- `src_ad8ce49f3cc468b4` — `tile_shape` `BN, 256` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1506`
- `src_393f8ee6cc563eba` — `tile_shape` `BK, 256` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1507`
- `src_f3e5a9635914fa7d` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:240`
- `src_06aa7c5b011611ea` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:256`

#### Limits

- `LDS_2WG_LIMIT` and the union/sequential layout comments are quoted from a read of `mxfp4_gemm1.py` and `mxfp4_gemm2.py`; the extractor's `config_limit` rule reaches the neighbouring `MIN_N_BLOCKS_FOR_SHALLOW_K` but not the limit constant itself, so the cited evidence anchors the surrounding derivation rather than the constant.
- Whether exceeding the two-workgroup budget actually costs throughput at a given shape is unmeasured here; the card only states that nothing prevents it.

### Which lane holds which row and column of a 16x16 MFMA fragment in AITER's FlyDSL kernels, and what does that fix for loads, in-register scaling and stores?

**Card:** `flydsl-mfma-fragment-lane-layout` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `epilogue`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a kernel loads MFMA fragments, scales an accumulator per row or per column, or writes it back

#### Try

- Load fragments by lane: in the HGEMM, lane l reads A row `l % 16` at K offset `(l // 16) * frag_values * mfma_steps` (`ldmatrix_a_m_idx`, `ldmatrix_a_k_vec_idx`), and B mirrors it along N (`ldmatrix_b_n_idx`, `ldmatrix_b_k_vec_idx`).
- Read the accumulator as lane l holding four consecutive rows starting at `(l // 16) * 4`, in column `l % 16`: the row epilogue's row offset is `lane_div_16 * 4 + ii` for ii in 0..3, the HGEMM store uses `stmatrix_c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES` and `stmatrix_c_n_idx = w_tid % WMMA_N`, and the A8 kernels name the same 4x16 lane grid `make_layout((4, 16), (16, 1))`.
- Scale in registers with that mapping: a lane multiplies its four rows by their row scales and its one column by that column's scale — the in-loop block-scale multiply (gemm-blockscale-k-loop-scale-application) and a per-token epilogue both need it.
- Derive loads, scales and stores from one mapping: a transposed reading on any side permutes the output without failing to compile.

#### Why this is a candidate

- Every FlyDSL GEMM in AITER writes this mapping as index arithmetic, so it can be read off the source instead of being rediscovered by debugging a permuted result.

#### Keep as alternatives

- Stage the accumulator through LDS with a C-shuffle epilogue (flydsl-cshuffle-epilogue) and index it row-major there, paying LDS for a simpler mapping.

#### Evidence

- `src_f01ef3793a7bd87c` — `fragment_layout` `ldmatrix_a_m_idx, w_tid % WMMA_M` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:288`
- `src_7e41979f22eae1f9` — `fragment_layout` `ldmatrix_a_k_vec_idx, w_tid // WMMA_M * WMMA_A_FRAG_VALUES * MFMA_PER_…` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:289`
- `src_44f9bf23c53fc7bc` — `fragment_layout` `ldmatrix_b_n_idx, w_tid % WMMA_N` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:290`
- `src_8ba4dcf6e9d68f6e` — `fragment_layout` `ldmatrix_b_k_vec_idx, w_tid // WMMA_N * WMMA_B_FRAG_VALUES * MFMA_PER_…` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:291`
- `src_91899d8d1f4341f6` — `fragment_layout` `stmatrix_c_m_vec_idx, w_tid // WMMA_N * WMMA_C_FRAG_VALUES` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:813`
- `src_074f27a8a5c89fcf` — `fragment_layout` `stmatrix_c_n_idx, w_tid % WMMA_N` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:814`
- `src_5cae13453d752cfb` — `fragment_layout` `lane_div_16_mul4, lane_div_16 * 4` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:73`
- `src_1743dd27416aa823` — `layout_epilogue` `default_epilog` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:50-82`
- `src_29a6126027710513` — `fragment_layout` `layout_lane16, fx.make_layout((4, 16), (16, 1))` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:426`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_attention_levers.md:49-62`](../languages/flydsl/authoring_attention_levers.md)
- [`languages/flydsl/authoring_tile_programming.md:201-277`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- The mapping is for the 16x16 MFMA atom these kernels use; a 32x32 instruction has a different lane layout.
- K per lane depends on the instruction (four bf16 or eight fp8 elements per lane for the 16x16 forms), which the frag-value constants carry.

### How should a FlyDSL GEMM apply an XOR16 LDS layout without making stores and loads disagree?

**Card:** `flydsl-xor16-lds-layout-consistency` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- operator_family=gemm
- target_language=flydsl
- an operand is staged through LDS (in the A8 family only A is; B is read preshuffled from global memory)
- XOR16 is included as a bank-conflict candidate

#### Try

- Define the byte-column transform once as col_in_bytes XOR ((row modulo k_blocks16) times 16).
- Apply the identical transform at every site: the global-to-LDS store and the LDS-to-fragment load, for each operand that goes through LDS.
- Re-derive the swizzle period per architecture: gfx942 LDS has 32 banks in 64 KiB, gfx950 has 64 banks in 160 KiB, so a stride that fully aliases on one may only partially conflict on the other (background).

#### Why this is a candidate

- The AITER implementation calls one helper at every write and read site; copying only one side changes addressing semantics rather than performance.
- The cited sites are A store, A fragment load, B store and B fragment load — the set that has to move together.

#### Keep as alternatives

- Keep the unswizzled layout as the control arm.
- Pad the LDS row instead of swizzling when LDS headroom allows — the FlyDSL grouped-GEMM kernels pad K by 8 elements unless they use the 128-bit LDS layout.
- Benchmark a different padding or swizzle scheme when the tile geometry changes.

#### Same question in other implementations

- **gluon** — `gl.SwizzledSharedLayout(...)` declared per staged operand (and per block-scale buffer); the compiler applies the same permutation at store and load. Evidence: `src_99b2357d48e96781` (`aiter/ops/triton/gluon/gemm_a8w8.py:123-125`), `src_4e349fde44ffedd3` (`aiter/ops/triton/gluon/gemm_a8w8.py:126-128`), `src_0526e1f18b1c8460` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:128-130`)

#### Evidence

- `src_2e329cfe46a52fb9` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:27-28`
- `src_d1f233838e74f022` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:455-457`
- `src_6f222b94149ab249` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:530`
- `src_d2c808ad3cd9ce7c` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:625-627`
- `src_0438c6e24a955ea2` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:675`
- `src_03338c20f236373f` — `lds_pad` `pad_k, 0` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:206`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_optimization.md:75-93`](../languages/flydsl/authoring_optimization.md)
- [`hardware/shared/memory_model_lds_bank.md`](../hardware/shared/memory_model_lds_bank.md)
- [`optimization/lds_and_bank_conflicts.md`](../optimization/lds_and_bank_conflicts.md)
- [`languages/flydsl/authoring_gemm_levers.md:133-148`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- The citations prove address consistency, not that XOR16 reduces bank conflicts on every tile.
- Retain the unswizzled implementation and choose by target-box measurement.

### What does enabling B preshuffle on a FlyDSL HGEMM actually require?

**Card:** `flydsl-preshuffle-b-layout-contract` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `operand_layout`, `runtime_contract`, `implementation_variant`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `reused_weight`

#### Use when

- operator_family=gemm
- target_language=flydsl
- B preshuffle is being considered as a candidate
- the regime of B is known or can be read from the caller: a weight reused across calls, or per-call data

#### Try

- Put the preshuffled B path on the candidate list on the generic family instead of deferring it: b_preshuffle is the default of the public entry point (flydsl_hgemm) and of the internal compile helper.
- Do not read that default as the deployment state: AITER's tuned database keys on `bpreshuffle`, and every FlyDSL bf16 selection it ships for gfx950 has b_preshuffle=False — those deployments pass unshuffled weights. Which layout you may assume is set by the caller's weight contract, not by the entry-point default.
- Treat b_preshuffle as a change of B's storage layout, not as a boolean tuning flag: the kernel expects B already permuted into the family's shuffle layout, and the host side must either shuffle it or be told it is pre-shuffled.
- Do not decline preshuffle on the grounds that no offline shuffle is available. The library performs the permutation itself with shuffle_weight(b, layout=_get_flydsl_shuffle_layout(pack_n)), exposes auto_shuffle_b to apply it at call time, and skips it when the tensor already carries the is_shuffled marker — the shuffle, the layout selector and the once-per-tensor skip are all readable from the entry point, so a copy of the kernel that cannot import them can still reimplement the same three pieces.
- Hoist the permutation out of the measured region rather than counting it against the win: key a cached shuffled copy on the identity of the source tensor so it is paid once per distinct B and reused by every later launch. is_shuffled is that idea in the library; a cache that also detects in-place mutation of the source is the same idea made safe.
- Scope the work item to the host layer before assigning it. The permutation happens on the host side of the entry point, before the launcher is called — it is not reachable from the kernel body. A direction that hands an engineer only the kernel source will come back 'not attempted, the host file is outside my lane', which is a decomposition failure and not a finding about preshuffle. The owner of this direction needs the configuration/launch module and permission to add a host-side module beside it.
- Make the cache invalidation-safe or do not cache: a freed tensor's address can be handed to a later allocation, so identity alone can alias two different matrices. Pair the address with something that changes when the contents do, and treat a mismatch as a re-shuffle rather than a stale hit.
- Settle the accounting question first, then act on the answer instead of parking the lever because the question exists. Read the benchmark loop: if it builds one B and calls the kernel repeatedly against it, with warmup iterations before the timed ones, then a once-per-tensor shuffle lands in warmup by construction and the steady-state cost is the honest number for that workload. Cache it, and write one line in the report saying B is treated as a reused operand and the permutation is amortised — a declared accounting choice is a complete answer, whereas deferring the lever leaves the fastest available configuration untested and reports nothing.
- Where B is genuinely per-call data the same reading gives the opposite instruction: do not cache, charge the shuffle to every launch, and expect the unshuffled path to win. The rule is that the regime decides, and the regime is readable from the caller.
- Keep b_to_lds=false whenever b_preshuffle=true; the current kernel rejects the combination.
- Keep b_preshuffle=false on the small-M family, which rejects it outright.
- Retune the tile shortlist on the preshuffled tree rather than carrying over the configuration that won without it; the two paths feed the MFMA differently and their best tiles need not agree.
- The A8 (fp8/int8) family has no such choice: it always consumes B preshuffled, so its host side must supply the shuffled weight (and the matching scale layout) — see flydsl-a8-preshuffle-config-space.

#### Why this is a candidate

- The generic HGEMM path supports preshuffled B and its dispatch default is even b_preshuffle=true, so this is a supported layout rather than a forbidden one — but the guard rails around it are all layout guard rails.
- The two rejections in the source are specific and different in kind: an unsupported combination (with b_to_lds) and an unsupported family (small-M). Neither is a statement about speed.
- The reason to name the amortisation machinery explicitly is that the layout requirement reads like a blocker and is not one. An author who stops at 'the kernel wants B permuted' concludes the harness cannot supply it; the entry point shows the library supplying it three different ways in the same nine lines.

#### Keep as alternatives

- Use the unshuffled B path as the correctness and performance control.
- Shuffle B once at load time outside the timed region when the deployment allows it, and measure both accounting rules explicitly.
- If B genuinely changes every call, the amortisation argument does not apply and the unshuffled path is the right answer — say which of the two regimes you are in before choosing.

#### Same question in other implementations

- **asm** — the bf16 assembly GEMM takes a `preshuffle` flag in its launcher and selects a preshuffled object. Evidence: `src_2616db93f0f6b2b7` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:105`)
- **ck** — separate B-preshuffle device-op templates (`*_BPreshuffle`) with their own tuned databases. Evidence: `src_24c2dc21486cdfe7` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:92`), `src_6352032268991afb` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:93`)

#### Evidence

- `src_93c0f2962c52fdfb` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:32`
- `src_4c74ced4671a1e5b` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:162`
- `src_3ee379ecc88d0ad1` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:767-768`
- `src_7b5432b56d9acd63` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:768`
- `src_520242be3bd7400e` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:886`
- `src_e7db1b030bf2e5c6` — `layout_preshuffle` `shuffle_weight` at `aiter/ops/flydsl/gemm_kernels.py:883`
- `src_886a3fb0a447c9a7` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:729`
- `src_2aa8afff0a4748b0` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:863`
- `src_e248a6669d5daaaf` — `config_validity` `Current kernel requires b_to_lds=False when b_preshuffle=True` at `aiter/ops/flydsl/gemm_kernels.py:161-163`
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)
- `cfg_147eed9b10acbc4e` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M65-256`: 52 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/knobs.md:62-69`](../languages/flydsl/knobs.md)
- [`languages/flydsl/patterns.md:55-66`](../languages/flydsl/patterns.md)

#### Limits

- A shipped default is a precedent, not a benchmark. That the library chose b_preshuffle=true says the path is first class and worth trying early; it does not say it wins on your shape, and the unshuffled control still has to be run.
- The amortisation only holds where one B is reused across launches. Under an accounting rule that charges the first shuffle to a single call, the path can lose; state which rule you are reporting.
- A benchmark that reuses one pre-shuffled B across iterations measures the steady state, not the first call; state which one you are reporting.
- The card is grounded in the vendored library's contract. A kernel copy that has been edited may have moved these checks.

## 5. Global, LDS and register data movement

- `memory_access` — Which access width and cache policy does the data path use?

Also bears on this step: `flydsl-gfx942-path-gates` (step 2), `gemm-workgroup-xcd-mapping` (step 3), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `gemm-skinny-decode-wave-split` (step 3), `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` (step 8).

### Is global-to-LDS async copy a knob, a fixed fact, or unavailable — per FlyDSL kernel family and architecture?

**Card:** `flydsl-async-copy-by-family` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `architecture_capability`, `pipeline_schedule`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- an operand tile is staged through LDS

#### Try

- Split-K HGEMM: not a knob. The flag is computed from the architecture (off on gfx942, on elsewhere) and a request that disagrees raises. Every FlyDSL bf16 selection AITER ships for gfx950 has async_copy on.
- A8 preshuffle (fp8/int8): a real knob on both architectures. The A tile is copied 4 bytes per lane on gfx942 and 16 bytes elsewhere, and the tuner sweeps async on and off in both catalogs.
- Search it per M bucket rather than fixing it: in AITER's gfx950 fp8 selections, async copy is on in 17 of 72 rows at M<=16 but in 39 of 50 at M 1025-4096 and 65 of 77 above 4096.
- When async copy is on, drop the ds_write group from any hand schedule: A lands in LDS without passing through VGPRs.
- For an authored kernel, size the copy to the hardware: direct global-to-LDS moves up to 32 bits per lane on CDNA3 and up to 128 bits on CDNA4 (background), which is why the gfx942 A8 path copies 4 bytes.

#### Why this is a candidate

- The same word — async copy — names an architecture fact in one family and a searched knob in the other; reading one family's gate as the other's rule removes a live candidate.
- The copy width is written in the A8 compile entry point, and the sweep values are module constants in its tune catalog.

#### Keep as alternatives

- Synchronous global-to-register-to-LDS staging, which is the gfx942 HGEMM path and the async=0 arm of the A8 sweep.

#### Evidence

- `src_c4c19187bcb7933c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/gemm_kernels.py:41`
- `src_a0f1066a6a2d6667` — `config_validity` `Current kernel fixes async_copy from the active GPU architecture;` at `aiter/ops/flydsl/gemm_kernels.py:123-126`
- `src_8f59fe9a6d65aaf3` — `async_copy` `ASYNC_COPY` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:132`
- `src_14bff946c5d70663` — `async_copy` `use_async_copy` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:136`
- `src_42146c2d073d655e` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:243`
- `src_623470961bcb8c24` — `config_space` `_ASYNC_COPY_VALS` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:248`
- `src_6dd333649bf42b12` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1291`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 274 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 661 shipped row(s)
- `cfg_9636311ea3a3ce0d` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: 178 shipped row(s)
- `cfg_973fe63f897809df` — FlyDSL `hgemm` `abf16_wbf16_bf16` selections in `bf16_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 86 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`hardware/cdna3_mi300/memory_hierarchy.md`](../hardware/cdna3_mi300/memory_hierarchy.md)
- [`hardware/cdna4_mi350/memory.md:51-55`](../hardware/cdna4_mi350/memory.md)
- [`languages/flydsl/deep.md:65-73`](../languages/flydsl/deep.md)
- [`languages/flydsl/knobs.md:71-75`](../languages/flydsl/knobs.md)
- [`languages/flydsl/authoring_gemm_levers.md:119-131`](../languages/flydsl/authoring_gemm_levers.md)
- [`languages/flydsl/authoring_tile_programming.md:396-426`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- Selection shares show where AITER's tuner chose async copy, not by how much it helped.
- The HGEMM rule is tied to the current kernel; an authored HGEMM can stage however the hardware allows.

### Which global-to-LDS and LDS-to-register helpers do the FlyDSL A8 and MoE GEMMs share, and which access widths do they fix?

**Card:** `flydsl-a8-data-movement-helpers` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `operand_layout`, `pipeline_schedule`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- an operand is staged through LDS and the other is read preshuffled from global memory, as in the A8 and MoE kernels

#### Try

- Move A at one width end to end: the MoE kernels take the largest of 16, 8 or 4 bytes that divides each thread's share of the tile (fp16/bf16 must divide by 16), load that many bytes from global (`buffer_copy_gmem16_dwordx4` for 16, raw dword loads for 8 and 4) and store them with the matching XOR16 helper — `lds_store_16b_xor16`, `lds_store_8b_xor16` or `lds_store_4b_xor16`. The A8 GEMM uses the same 16-byte global copy and writes LDS inline, or skips the register hop with async copy (flydsl-async-copy-by-family).
- Read A back from LDS one 8-byte K-pack per K32 MFMA step through the same XOR16 swizzle the store used (`lds_load_pack_k32`: an 8-byte load, or a 16-byte load split in halves); the GEMM and MoE kernels wrap it in a local 64-K variant, `lds_load_packs_k64`.
- Never stage B: read it preshuffled straight from global into MFMA packs (`load_b_pack_k32`, `load_b_packs_k64`, `load_b_tile`) in the layout `make_preshuffle_b_layout` defines — K-packs of 8 or 16 bytes over 1- or 2-byte elements; it raises on anything else. The scales have their own preshuffled layout (`make_preshuffle_scale_layout`).
- Reuse the layer rather than rewriting it: the 2-stage MoE kernels import the same pipeline module and write their B loads the same way.

#### Why this is a candidate

- These helpers are the data-movement half of the A8 family, named by the widths they fix; a FlyDSL fp8 kernel built on them inherits a movement path AITER already ships.

#### Keep as alternatives

- Stage B through LDS as well (the HGEMM's `b_to_lds`), paying LDS capacity for an unshuffled weight.

#### Evidence

- `src_54e4ca3a8185528d` — `access_width` `buffer_copy_gmem16_dwordx4` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:467-489`
- `src_f419311ce8a02db3` — `access_width` `lds_store_16b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:492-516`
- `src_c77d2ddf58185db3` — `access_width` `lds_store_8b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:519-543`
- `src_f735e442dfbc314e` — `access_width` `lds_store_4b_xor16` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:546-570`
- `src_498bdd55bb2b600e` — `access_width` `lds_load_pack_k32` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:573-604`
- `src_aa04a6d4ea345675` — `access_width` `load_b_pack_k32` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:346-444`
- `src_2a3468398e382ec9` — `access_width` `load_b_packs_k64` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:525-550`
- `src_53b921a28e9cd675` — `access_width` `load_b_tile` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:552-581`
- `src_36c3ee38d337b470` — `layout_preshuffle` `make_preshuffle_b_layout` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:159-223`
- `src_11e30e2c665aafa7` — `layout_preshuffle` `make_preshuffle_scale_layout` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:96-148`
- `src_1c0548a53c4cf22e` — `config_validity` `kpack_bytes must be 8 or 16, got {kpack_bytes!r}` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:175`
- `src_2bc956feedfd346d` — `config_validity` `elem_bytes must be 1 or 2, got {elem_bytes!r}` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:183`
- `src_14d7a32aaedf7288` — `access_width` `load_b_tile` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:639-692`
- `src_6d43092520bc779c` — `config_validity` `[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by …` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:442-444`
- `src_0e09838af93de4ba` — `config_validity` `bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to us…` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:454-456`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/authoring_gemm_levers.md:119-131`](../languages/flydsl/authoring_gemm_levers.md)
- [`optimization/vectorization_and_coalescing.md:33-47`](../optimization/vectorization_and_coalescing.md)

#### Limits

- The helpers are the A8/MoE family's; the HGEMM stages its operands with its own code, and a block-scale kernel adds scale loads these helpers do not cover.
- Which width is fastest for a tile is a measurement; the helpers state which widths are legal.

### Which operand loads and output stores should carry a streaming (non-temporal) cache policy, and which should keep the default?

**Card:** `gemm-streamed-operand-cache-policy` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- one operand is read once per call while the other is reused across tiles — typically the weight B at decode, where every workgroup streams its own slice of W while the few rows of A are shared — or an output is written once and not read back by this kernel

#### Try

- Make the cache policy a per-operand knob on the global loads, not a kernel-wide switch: FlyDSL's `buffer_ops.buffer_load` takes `cache_modifier`, and AITER's FlyDSL MoE GEMM exposes it for its B loads only, as `b_nt` (default 0 in one of its kernels and 2 in the other), while its other loads pass `cache_modifier=0`.
- Search the streamed operand's load policy — the default against the streaming setting — and leave the reused operand's loads at the default.
- Store a write-once output non-temporally when nothing re-reads it soon: the FlyDSL MoE kernel writes its output with `llvm.StoreOp(..., nontemporal=True)`.

#### Why this is a candidate

- AITER's Triton a16w16 GEMM makes the same choice per operand: `cache_modifier` is a kernel parameter applied to the B loads only, never to A. Of the 306 shipped Triton GEMM config groups, 88 set it, and 87 of those choose `.cg`.
- A line fetched once and never re-read only displaces lines that would have been reused; the reused operand is the one worth keeping resident.

#### Keep as alternatives

- The default cache policy everywhere — the control arm.

#### Same question in other implementations

- **triton** — `cache_modifier: tl.constexpr` on the a16w16 kernels, passed to `tl.load(b_ptrs, cache_modifier=...)` for B and never for A; the value is carried per bucket in the shipped configs. Evidence: `src_15cea189ebb33d4c` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:81`), `src_92959715207084c1` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:159`), `src_9667faa67584ba59` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:168`)

#### Evidence

- `src_6a42df54a6f6b458` — `cache_policy` `cache` at `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py:71`
- `src_e93e7dd39f7adedc` — `cache_policy` `b` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:893`
- `src_106bcf16a01dfc38` — `cache_policy` `b` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:3553`
- `src_e59fe6013e3decd4` — `cache_policy` `nontemporal=True` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:2284`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/hbm_infinity_fabric.md:73`](../hardware/shared/hbm_infinity_fabric.md)
- [`hardware/cdna3_mi300/memory_hierarchy.md:48`](../hardware/cdna3_mi300/memory_hierarchy.md)

#### Limits

- The spelling differs per backend — a Triton modifier string, a FlyDSL integer `cache_modifier`, an LLVM `nontemporal` store flag — and the corpus does not map one onto another; read what the target API accepts.
- Whether streaming helps depends on the reuse the benchmark reproduces: a timing loop that re-reads the same weight from a warm cache measures a different reuse than serving, where each layer reads its own weight.

### Stage 1's `f16in` / `iq` token — what does inline quantization actually remove, and when is it available?

**Card:** `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `memory_access`, `implementation_variant`, `pipeline_schedule`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- stage-1 mxfp4 grouped GEMM whose activation arrives as bf16 rather than pre-quantized

#### Try

- Read `f16in` (parsed to `inline_quant`, tagged `iq` in the device symbol) as a change of pipeline, not an extra step: the kernel takes the bf16 hidden state as an extra argument and quantizes it per 32-element group into the swizzled A-LDS slot directly, so the global->LDS DMA for A and the whole A-scale load are SKIPPED, not supplemented.
- Expect the saving to be a removed memory pipeline and the cost to be in-loop ALU: a packed-u16 amax over four dwords plus two DPP reduction steps per group, then `cvt_scalef32_pk_fp4_bf16` straight into the position the MFMA fragment reader expects.
- Do not plan on combining it: inline quant is available only at `BM == 16` (and so only with `use_nt=True`), is rejected with `k_wave > 1` and with `BN == 64`, and is a PREREQUISITE for `prefetch_hidden` (`hpf`), which hoists the next tile's bf16 loads one iteration ahead.
- When the caller already holds a quantized activation, prefer the non-inline variants (`nt` / `cached` in the tag position): the three are mutually exclusive in the name, so this is a choice about where quantization happens, not whether.

#### Why this is a candidate

- At decode M the stage-1 GEMM is memory-bound on a skinny A, and the inline path deletes two separate global streams (quantized A and its scales) in exchange for arithmetic that the MFMA pipeline is not using.
- The tile restriction is not incidental: the whole point is a BM16 kernel whose A tile is small enough to quantize inside the K loop without spilling, which is why the whitelist has exactly one inline-quant entry.

#### Keep as alternatives

- Quantize the activation in a separate kernel before stage 1 — one extra pass over the hidden state and an extra buffer, but it keeps stage 1 on the plain A-load path and leaves tile_m free.
- `hpf` (prefetch_hidden) on top of inline quant — deeper hiding of the bf16 loads, at the cost of holding another tile's worth of registers.

#### Evidence

- `src_0a0738d9fdf9642d` — `config_space` `MXFP4_G1_VARIANTS` at `aiter/ops/flydsl/mxfp4_kname.py:26-42`
- `src_9bff422d7e1ffc9d` — `config_validity` `flydsl mxfp4 GEMM1 hidden prefetch requires inline quantization` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:113-115`
- `src_ae5bccba3310fa36` — `config_validity` `flydsl mxfp4 GEMM1 requires num_waves in (2, 4), got {num_waves}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:146-148`
- `src_a8b88b9d2224fb6b` — `config_validity` `flydsl mxfp4 GEMM1 two-wave specialization requires effective BN64` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:150-152`
- `src_c86e3bf3667b6875` — `config_validity` `flydsl mxfp4 gemm1 unsupported variant` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:180-184`
- `src_74352923ff7ef43b` — `tile_shape` `BM, 32` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1498`
- `cfg_58db8a1bd4f97080` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M<=16`: 1 shipped row(s)

#### Limits

- No measurement in this corpus compares inline quant against a pre-quantized stage 1 at the same shape; the deployed GLM-5.2 decode rows select it, which is a seed and not a benchmark.
- Because inline quant forces `BM == 16`, any A/B against a non-inline config also changes the tile — the two effects are not separable by swapping the token alone.

## 6. MFMA / dot compute

- `compute_instruction` — Which matrix-compute family and instruction shape does the implementation request?

Also bears on this step: `flydsl-gfx942-path-gates` (step 2), `flydsl-splitk-hot-loop-schedule-bundle` (step 7), `flydsl-a8-interleaved-schedule` (step 7), `flydsl-mfma-fragment-lane-layout` (step 4).

### Which scale granularity does the FlyDSL A8 kernel implement, and what changes for a 128x128 block-scale contract?

**Card:** `flydsl-a8-scale-granularity-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `epilogue`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`, `int8`

#### Use when

- target_language=flydsl
- fp8/int8 GEMM whose math contract names its scale granularity (per-tensor, per-token/per-channel, or [128,128] blocks)

#### Try

- Read the A8 family as per-token x per-channel: for fp8/int8 it loads one fp32 scale per output row of A and one per output column of B at the last tile and applies them in the epilogue. FP4 is its only block-scaled path (per-32 MX scales on gfx950).
- Do not reuse it unmodified for a [128,128] block-scale contract: those scales change every 128 elements of K, so they have to be applied per K block inside the loop (promote the partial to fp32 per block, multiply, accumulate), not once at the end — the loop body is gemm-blockscale-k-loop-scale-application.
- Do not fold arbitrary fp32 block scales into the gfx950 scaled MFMA: its scale operand is E8M0 (a power of two), so a general fp32 scale is rounded and parity fails (background).
- Settle the math contract against the task's oracle before timing anything: a per-token kernel measured against a block-scale oracle fails parity, it does not lose on speed.

#### Why this is a candidate

- The A8 compile entry point decides `_needs_per_token_scale` from the dtype and loads the row and column scales only for the last tile, which is the per-token contract written down.
- AITER's block-scale databases contain no FlyDSL row at all — every block-scale selection went to CK, CK-Tile or ASM — so there is no shipped FlyDSL block-scale kernel to copy.

#### Keep as alternatives

- For block scales, keep the A8 kernel's MFMA, staging and schedule as the precedent and change only where the scale is applied.

#### Same question in other implementations

- **triton** — one kernel per granularity, and the difference is where the multiply sits: the per-token kernel scales the finished accumulator once after the K loop (`accumulator *= a_scale[:, None] * b_scale[None, :]`), the block-scale kernel and its preshuffled variant scale every K block's `tl.dot` partial inside the loop. Evidence: `src_7a0ec7ada35d7647` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py:177`), `src_426b43e7e6389512` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:190-194`), `src_359d07953d540490` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:407-411`)
- **gluon** — the fp8 block-scale kernel stages the A and B scales through LDS with their own shared layouts, next to the operands, and multiplies each MFMA partial by them inside the loop. Evidence: `src_1ba1837b9ca062ee` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:134-136`), `src_927f1e26846a58fd` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:137-139`), `src_db2ae65ef56818a7` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:301`)

#### Evidence

- `src_f854f40de1f28585` — `scale_operand` `_needs_per_token_scale` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:403`
- `src_1255666f8d08f366` — `mfma_intrinsic` `mfma_scale_f32_16x16x128_f8f6f4` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:934`
- `sel_b7d6386940696a48` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: ck 293 · asm 40 · triton 28 · cktile 12 (rows)
- `sel_8b543c4558078855` — backend selected by AITER's tuner in `a8w8_blockscale_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: ck 8960 · cktile 1065 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 26 · opus 9 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/numerics.md:36-52`](../operators/dense_gemm/numerics.md)
- [`quantization/block_scaling_mxfp.md:40-49`](../quantization/block_scaling_mxfp.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:96-103`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- The per-K-block promote is the arithmetic a block-scale contract requires; its cost on a given shape is a measurement, and measured recipes live behind the expert-skill switch.
- The E8M0 limitation is a hardware fact from the background docs, not from this source.

### Which MFMA does a FlyDSL quantized GEMM issue per dtype and architecture, and what does that fix about tile_k?

**Card:** `flydsl-a8-mfma-selection` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `architecture_capability`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`, `int8`, `int4`, `fp4_e2m1`, `mxfp4`

#### Use when

- target_language=flydsl
- quantized GEMM (fp8, int8/int4, fp4)

#### Try

- fp8: issue `mfma_f32_16x16x32_fp8_fp8` (K=32 per instruction); on gfx95x the A8 kernel switches fp8 to the block-scaled `mfma_scale_f32_16x16x128_f8f6f4` (K=128 per instruction).
- int8, and int4 widened to int8: issue `mfma_i32_16x16x32i8` and keep i32 accumulation until the scale is applied.
- f16 / bf16 inside the same kernel: `mfma_f32_16x16x16f16` / `mfma_f32_16x16x16bf16_1k`.
- fp4 (gfx950 only): the same x128 scaled MFMA with its operand-format codes set for FP4 and M/N/K packed by two.
- Derive tile_k from the instruction K: a tile_k that is not a multiple of 128 cannot take the gfx950 scaled path, which is why the gfx950 A8 catalog has no tile_k=64 tiles.
- When porting between architectures, keep the fp8 format per arch in view: gfx942 is FNUZ e4m3 (bias 8, max 240) and gfx950 is OCP e4m3 (bias 7, max 448), so FNUZ bytes bit-copied onto gfx950 are off by a factor of two (background).

#### Why this is a candidate

- The dtype branches and the gfx95 switch are explicit in one function, and the instruction K is what sets the catalog's tile_k granularity.
- The instruction choice changes the loop trip count, the scale plumbing and the legal tiles together, so it is the first decision rather than a late tuning knob.

#### Keep as alternatives

- In an authored kernel, keep the unscaled K=32 fp8 MFMA as a control arm on gfx950 when the math contract does not need the hardware scale operand.

#### Same question in other implementations

- **triton** — `tl.dot` over fp8 tiles inside the block-scale K loop; the compiler selects the fp8 MFMA. Evidence: `src_f4a3d14442cafc0c` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:191`)
- **gluon** — an explicit `instr_shape=[16, 16, 32]` MFMA layout in the fp8 block-scale kernel. Evidence: `src_cf70b9a5bae73954` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:123`)

#### Evidence

- `src_113ea8d0f42a693d` — `mfma_intrinsic` `mfma_f32_16x16x32_fp8_fp8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1009`
- `src_1255666f8d08f366` — `mfma_intrinsic` `mfma_scale_f32_16x16x128_f8f6f4` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:934`
- `src_84fa26c5447f28f7` — `arch_gate` `gfx95` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:848`
- `src_bc08df5300de01f3` — `config_validity` `tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_…` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:855-857`
- `src_f9f2544162138ea2` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:215-217`
- `src_a5d9e08624d3f251` — `mfma_intrinsic` `mfma_i32_16x16x32i8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:220`
- `src_722ebc17457eec5c` — `mfma_intrinsic` `mfma_f32_16x16x16f16` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1005`
- `src_d9a92f8642e73d56` — `mfma_intrinsic` `mfma_f32_16x16x16bf16_1k` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1007`
- `src_011cf8d1222b7906` — `arch_gate` `gfx950` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:254`
- `src_273e37ba3a8cefa8` — `config_space` `_base_tiles_lds2_950_extra` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:227-229`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/dtype_numerics.md:40-49`](../hardware/shared/dtype_numerics.md)
- [`hardware/cdna4_mi350/matrix_core_blockscale.md`](../hardware/cdna4_mi350/matrix_core_blockscale.md)
- [`quantization/fnuz_vs_ocp.md`](../quantization/fnuz_vs_ocp.md)
- [`operators/scaled_quant_gemm/backends/flydsl.md:30-58`](../operators/scaled_quant_gemm/backends/flydsl.md)

#### Limits

- This is instruction selection in AITER's vendored kernel; it says nothing about which instruction is faster for a shape.
- The FNUZ/OCP statement comes from the background hardware docs, not from this source.

### How does a 128x128 block-scale GEMM apply its scales inside the K loop, and what does that fix about tile_k and the scale indexing?

**Card:** `gemm-blockscale-k-loop-scale-application` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `workgroup_tile`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz_blockscale`, `fp8_e4m3fn_blockscale`

#### Use when

- target_language=flydsl
- the math contract is Y = (X*x_scale) @ (W*w_scale)^T with one fp32 scale per row of X per 128 elements of K and one per 128x128 block of W

#### Try

- Scale every K block's partial before it joins the accumulator: for each 128-wide K block, take the tile's fp32 MFMA partial, multiply it by the outer product of that block's per-row X scales and per-column-group W scales, and add it into a separate fp32 accumulator. Triton writes the loop body as `accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]`; Gluon as `acc += mfma_out * cur_a_scale[:, None] * cur_b_scale[None, :]`.
- Make one K tile exactly one scale block. The Triton kernel takes GROUP_K and BLOCK_SIZE_K as separate parameters, its docstring requires them equal, and every gfx942 configuration AITER ships sets BLOCK_SIZE_K = GROUP_K = GROUP_N = 128 — on gfx942 that is four 16x16x32 fp8 MFMA K-steps between scale multiplies.
- Index the scales per K block: X's scale is [M, ceil(K/128)] and advances one column per K tile; W's holds one value per 128x128 block, so each output column reads the scale of its 128-wide group for the current K block (Triton gathers it as `offs_bn // GROUP_N`).
- Keep the MFMA unscaled and the scale in software: on gfx942 issue the plain fp8 MFMA (flydsl-a8-mfma-selection); on gfx950 do not fold arbitrary fp32 block scales into the scaled MFMA, whose scale operand is E8M0 (flydsl-a8-scale-granularity-contract, background).
- If K is split across workgroups, start every split on a scale-block boundary: Triton offsets the scale column by the split start divided by GROUP_K, and every SPLITK_BLOCK_SIZE it ships for gfx942 (512 to 2048) is a multiple of 128.

#### Why this is a candidate

- Every block-scale kernel AITER ships multiplies per K block inside the loop, and its per-token kernel multiplies once after it: the placement is the math contract, not an optimisation.
- AITER has no FlyDSL block-scale kernel (flydsl-a8-scale-granularity-contract), so a FlyDSL author writes this loop body from these precedents.

#### Keep as alternatives

- Stage the scales through LDS with their own layouts, next to the operands, as the Gluon kernel does; the background pitfall is feeding the global scale layout straight to the MFMA.
- The same arithmetic sits inside CK's A/B-scale device operation and ASM's block-scale kernels, as template or binary configuration rather than a loop you write.

#### Same question in other implementations

- **gluon** — scales staged through LDS with their own shared layouts; each MFMA partial multiplied by the current block's scales inside the loop. Evidence: `src_db2ae65ef56818a7` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:301`), `src_1ba1837b9ca062ee` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:134-136`), `src_927f1e26846a58fd` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:137-139`)
- **ck** — an A/B-scale device operation (`DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3`), plus a block-scale B-preshuffle variant. Evidence: `src_73c6170a2792b675` (`csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh:98`), `src_6352032268991afb` (`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/include/gemm_a8w8_blockscale_bpreshuffle_common.cuh:93`)
- **asm** — precompiled block-scale kernels with B preshuffled — the same six on gfx942 and gfx950, every one tile_n = 128 (one W scale group per tile) with tile_m 32 to 128 and split-capable; the launcher picks tile_m and a split per shape and multiplies the grid by the split. Evidence: `src_381f987b35a586b7` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:116`), `src_b07a8dc43c913b9d` (`csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu:353`), `src_3865c44e30f36de9` (`hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`), `src_fcf5d73d12271d55` (`hsa/gfx942/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:3`), `src_8866d3dbeb612f3a` (`hsa/gfx950/fp8gemm_blockscale/fp8gemm_bf16_blockscale.csv:2`)

#### Evidence

- `src_426b43e7e6389512` — `scale_in_k_loop` `accumulator, a_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:190-194`
- `src_359d07953d540490` — `scale_in_k_loop` `accumulator, a_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:407-411`
- `src_7a0ec7ada35d7647` — `scale_after_k_loop` `accumulator, a_scale, b_scale` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py:177`
- `src_4991e94450b3832a` — `tunable_param` `GROUP_K` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:73`
- `src_4f4e8313f9775c89` — `tunable_param` `BLOCK_SIZE_K` at `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py:78`
- `cfg_d2642e1347a53ea1` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `M_LEQ_8` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `cfg_3eff28954f18b5fe` — Triton seed `G E M M   A 8 W 8 _ B L O C K S C A L E` on `gfx942`, `any` (see [`gemm_triton_seeds.md`](gemm_triton_seeds.md))
- `sel_f7e394faa23e37e6` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M<=16`: ck 66 · triton 4 · asm 2 (rows)
- `sel_f1384cc1069c9823` — backend selected by AITER's tuner in `a8w8_blockscale_bpreshuffle_tuned_gemm` on `gfx942`/80 CUs, `M>4096`: asm 32 · ck 26 · opus 9 (rows)

#### Background (always-on reference docs, not measurements)

- [`operators/dense_gemm/numerics.md:36-52`](../operators/dense_gemm/numerics.md)
- [`operators/scaled_quant_gemm/tuning.md:50-57`](../operators/scaled_quant_gemm/tuning.md)
- [`operators/dense_gemm/numerics.md:59-63`](../operators/dense_gemm/numerics.md)

#### Limits

- The per-block multiply is VALU work on every K block; whether it hides behind the MFMAs is a measurement.
- These are Triton and Gluon spellings; FlyDSL expresses the same loop with its own MFMA, vector and conversion ops.
- Triton's scale strides are its wrapper's layout choice; take the x_scale / w_scale shapes from the task's math contract rather than from Triton.

### Which legal MFMA call forms can seed a FlyDSL f16/bf16 GEMM?

**Card:** `flydsl-half-mfma-call-forms` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `compute_instruction`, `architecture_capability`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `batched_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `f16`, `bf16`

#### Use when

- operator_family=gemm
- target_language=flydsl
- input_dtype in {f16,bf16}

#### Try

- For an m16n16k16 candidate, pair bf16 with mfma_f32_16x16x16bf16_1k and f16 with mfma_f32_16x16x16f16.
- Check the target architecture before treating the K shape as a free axis: the shipped split-K kernel does not let you pick it. On gfx942 it selects m16n16k16 with 4-byte DMA and two MFMA steps per warp-K; on every other architecture it selects m16n16k32 with 16-byte DMA and one. Benchmarking the K=32 form on gfx942 means rewriting that branch, not passing a flag.
- Also benchmark the m16n16k32 bf16/f16 forms where the architecture branch does select them and the fragment shape and register budget allow.
- Keep 16x16 fragments as the default shape and reach for 32x32 only when the tile needs it: a 16x16 accumulator block costs 4 VGPRs per lane, a 32x32 block 16, at the same nominal rate (background).

#### Why this is a candidate

- AITER's FlyDSL split-K GEMM implements both K=16 and K=32 classes and branches explicitly on bf16 versus f16.
- The two classes are not offered side by side at runtime: one architecture test picks the WMMA implementation, the DMA width and the MFMA-per-warp-K count together, as one bundle.
- This establishes legal implementation precedents; it does not establish which form is faster for a new shape.

#### Keep as alternatives

- Use the m16n16k16 form when the larger-K fragment does not fit the surrounding tile or register budget.
- Let a higher-level backend select the instruction when direct MFMA control is not required.

#### Same question in other implementations

- **triton** — `tl.dot` over the tile; the compiler picks the MFMA, steered by `matrix_instr_nonkdim` in the shipped config. Evidence: `src_cb8f85060eec45f8` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:170`)
- **gluon** — an explicit MFMA layout with `instr_shape` (e.g. [16, 16, 32] or [32, 32, 16]) chosen per kernel. Evidence: `src_cf70b9a5bae73954` (`aiter/ops/triton/gluon/gemm_a8w8_blockscale.py:123`), `src_422867b3bf9dfa13` (`aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py:88`)
- **ck** — an XDL device-op template; the MFMA shape is a positional template argument the corpus does not decode. Evidence: `src_780fadebcc954d06` (`csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh:86`)

#### Evidence

- `src_24517333d1cd6722` — `mfma_intrinsic` `mfma_f32_16x16x16bf16_1k` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:56-58`
- `src_0ed3b5780c589479` — `mfma_intrinsic` `mfma_f32_16x16x16f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:59-61`
- `src_387e95ce58e0449b` — `mfma_intrinsic` `mfma_f32_16x16x32_bf16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:77-79`
- `src_79cfb57473ed817f` — `mfma_intrinsic` `mfma_f32_16x16x32_f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:80-82`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128-137`

#### Background (always-on reference docs, not measurements)

- [`hardware/shared/matrix_core_mfma_smfmac.md`](../hardware/shared/matrix_core_mfma_smfmac.md)
- [`hardware/cdna4_mi350/matrix_core_blockscale.md`](../hardware/cdna4_mi350/matrix_core_blockscale.md)
- [`languages/flydsl/authoring_tile_programming.md:201-277`](../languages/flydsl/authoring_tile_programming.md)
- [`languages/flydsl/authoring_tile_programming.md:503-518`](../languages/flydsl/authoring_tile_programming.md)

#### Limits

- No latency or occupancy comparison is attached; treat both K shapes as candidates and re-measure.
- API spelling is version-sensitive; consult languages/flydsl/version_map.md before copying.

## 7. Pipeline and synchronization

- `pipeline_schedule` — How are loads, LDS buffering and matrix instructions pipelined and ordered?

Also bears on this step: `flydsl-hgemm-tiling-validity` (step 3), `flydsl-async-copy-by-family` (step 5), `flydsl-a8-preshuffle-config-space` (step 3), `flydsl-a8-data-movement-helpers` (step 5), `grouped-moe-mxfp4-inline-quant-replaces-the-a-pipeline` (step 5), `grouped-moe-mxfp4-lds-budget-is-a-heuristic-not-a-gate` (step 4), `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` (step 8).

### What explicit instruction-order bundle can seed a hand-scheduled FlyDSL split-K GEMM hot loop?

**Card:** `flydsl-splitk-hot-loop-schedule-bundle` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `workgroup_tile`, `execution_geometry`, `memory_access`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `split_k_hot_loop`

#### Use when

- operator_family=gemm
- target_language=flydsl
- the hot loop is hand-scheduled with ROCm scheduling directives (a split-K or tiled K loop with both operands in LDS)

#### Try

- Copy the ORDER, not a set of magic numbers: A-fragment ds-reads, then B-fragment ds-reads, then the A and B global loads as vmem, then the MFMA issues, closed by a scheduling barrier.
- Derive each group's count from the loop geometry that produces it — warp K steps times warp M or N steps for the ds-reads, the per-block LDG register counts for the vmem groups, and the full MFMA count for the mfma group — rather than hard-coding the counts observed here.
- Move or retune the bundle as a unit against the exact load/MFMA loop, then compare with the unscheduled control.

#### Why this is a candidate

- The directives are adjacent inside one scheduler function and describe a single ordering pattern; copying one directive in isolation loses that context.
- An earlier version of this card recorded the literal group sizes vmem(2)/dsrd(2)/dsrd(4)/mfma(8). Upstream now emits unit-count directives inside geometry-derived loops, so those numbers described one build of one tile and expired quietly. The order survived the rewrite; the constants did not, which is the reason this card states a rule and not a bundle of integers.

#### Keep as alternatives

- Leave instruction ordering to the compiler; that is the control arm and it is often the right answer.
- Use the A8 family's interleaved scheduler instead, which spreads ds-reads and vmem across the MFMAs one at a time rather than emitting the groups back to back — see flydsl-a8-interleaved-schedule.

#### Same question in other implementations

- **ck** — a named loop scheduler (`BlockGemmPipelineScheduler::Intrawave` / `Interwave`) plus a pipeline version chosen per template instance. Evidence: `src_75b377d9dfc10083` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:68`), `src_bb0d15e277ff0a86` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:69`), `src_6a2107cadf863683` (`csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale_common.cuh:85`)

#### Evidence

- `src_5da6811a509d9806` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:688`
- `src_b0da890c19bc671e` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:690`
- `src_ed1134f4e4cea08c` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:694`
- `src_86d7e5af1117454e` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:698`
- `src_0f18d7f50ce33cbb` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:702`
- `src_dca1b81d9d2b2edf` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:703`

#### Background (always-on reference docs, not measurements)

- [`optimization/mfma_scheduling.md`](../optimization/mfma_scheduling.md)
- [`languages/flydsl/authoring_gemm_levers.md:150-165`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- Instruction scheduling is the most build-specific thing in this corpus; treat it as a candidate to try late, after tiling and split-K are settled.
- Changing tile geometry, LDS stages or access width changes the schedule that must be re-measured.

### How does the FlyDSL A8 kernel interleave LDS reads, global loads and MFMAs, and how does that differ between gfx942 and gfx950?

**Card:** `flydsl-a8-interleaved-schedule` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `memory_access`, `workgroup_tile`, `compute_instruction`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `dtypes`: `fp8_e4m3fnuz`, `fp8_e4m3fn`, `int8`

#### Use when

- target_language=flydsl
- a K loop whose A tile goes through LDS and whose B is read straight from global memory
- instruction order is being pinned by hand

#### Try

- On gfx950 for fp8/int8, seed the preload depth from the family's per-tile table: `dsrd_preload` / `dvmem_preload` default to -1, which looks (tile_m, tile_n, tile_k) up in `_TILE_PRELOAD_TABLE` (values 2-8); every other architecture and dtype falls back to (0, 0).
- After the preload, issue one MFMA at a time and spread the remaining ds-reads and global loads evenly over the MFMA sequence (the kernel computes the per-MFMA counts with a ceiling-division distribution), then close with a sched barrier.
- Place the next tile's A ds-writes after the last ds-read of the current tile, and skip the ds-write group entirely when async copy writes A straight into LDS.
- On gfx942 the kernel uses a separate fixed pattern — vmem, a group of MFMAs, one ds-read, another group of MFMAs, with ds-writes near the tail — derived from num_acc_n and k_unroll; do not carry the gfx950 preload table over.
- Recompute the counts from your own geometry (LDS loads, global loads, MFMA total) whenever the tile, stage count or access width changes, instead of copying the table.

#### Why this is a candidate

- This is the scheduler of the FlyDSL family AITER selects most often in its gfx950 fp8 database, and both of its architecture branches are written out in one function.
- The preload table is keyed by tile, so it is a per-geometry fact rather than a constant that transfers.

#### Keep as alternatives

- The split-K HGEMM's grouped order — see flydsl-splitk-hot-loop-schedule-bundle.
- No hand schedule: leave ordering to the compiler as the control arm.

#### Evidence

- `src_62445650dbe2e65e` — `config_space` `_TILE_PRELOAD_TABLE` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:31-111`
- `src_d4d6210e665ad849` — `config_space` `_TILE_PRELOAD_DEFAULT` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:113`
- `src_d645d4f951f04c13` — `arch_gate` `gfx950` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:156-159`
- `src_43e09b045b1cc853` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1266`
- `src_a76d2aba228be108` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1268`
- `src_69bc44a707a4ec5b` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1270`
- `src_5732b53b2a42edb6` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1276`
- `src_47d88e0587130c78` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1284`
- `src_6dd333649bf42b12` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1291`
- `src_fc0ce347afaa09c3` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1299`
- `src_81f135c38346b8a5` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1197`
- `src_be54e3472c40f00f` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1198`
- `src_5793d6410b67084b` — `scheduling` `sched_dswr` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1225`
- `sel_7c8d8e410633ad1e` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: flydsl 274 · ck 57 · cktile 20 (rows)
- `sel_1b5489e6d41507a3` — backend selected by AITER's tuner in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M>4096`: flydsl 280 · cktile 40 · ck 26 (rows)

#### Background (always-on reference docs, not measurements)

- [`optimization/mfma_scheduling.md`](../optimization/mfma_scheduling.md)
- [`optimization/memory_pipelining.md`](../optimization/memory_pipelining.md)
- [`languages/flydsl/authoring_gemm_levers.md:150-165`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- The table values are AITER's per-tile choices for gfx950 fp8/int8; they are seeds for that geometry only.
- Scheduling is the most build-specific knowledge here; settle the tile and staging first.

### How many LDS stages should the K loop keep in flight, and what does each one cost?

**Card:** `flydsl-lds-pipeline-depth` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `pipeline_schedule`, `configuration_constraints`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`
- `requires`: `lds_staging`

#### Use when

- target_language=flydsl
- the design stages operand tiles through LDS (trait lds_staging)

#### Try

- Treat the stage count as a knob that competes with the tile for LDS: the split-K HGEMM allocates `STAGES * BLOCK_M * BLOCK_K` elements for A and `STAGES * BLOCK_N * BLOCK_K` for B, so every extra stage costs one more A tile and one more B tile.
- Start from AITER's two stages — `STAGES = 2` in the split-K and small-M HGEMMs, `lds_stage=2` as the preshuffle GEMM's default — and try a deeper ring only where the tile is small enough for the LDS budget and the tiling validator to accept it.
- Keep the stage count in the kernel name, as AITER does (`_S{STAGES}`, `_lds{lds_stage}`), so two builds that differ only in depth never share a compiled-kernel cache entry.

#### Why this is a candidate

- A stage is how far the loader runs ahead of the MFMAs, and it is bought with LDS that the tile, occupancy and any LDS-staged epilogue also want.
- AITER fixes two stages in the HGEMMs and exposes the depth only on the preshuffle GEMM, where it also restricts it (FP4 with tile_k=128 supports only lds_stage=2): the depth is a design axis, not a constant.

#### Keep as alternatives

- More workgroups per CU instead of more stages: the same LDS spent on occupancy hides latency across workgroups rather than within one.

#### Evidence

- `src_dbdbdbf21af30e3c` — `lds_stage` `STAGES, 2` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:142`
- `src_8a036c537a13ff24` — `lds_stage` `STAGES, 2` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:72`
- `src_1a80adca750d9118` — `lds_stage` `lds_stage, 2` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:211`

#### Background (always-on reference docs, not measurements)

- [`optimization/lds_and_bank_conflicts.md:55-59`](../optimization/lds_and_bank_conflicts.md)
- [`optimization/lds_and_bank_conflicts.md:94-96`](../optimization/lds_and_bank_conflicts.md)
- [`operators/skinny_gemv_decode/backends/flydsl.md:59`](../operators/skinny_gemv_decode/backends/flydsl.md)

#### Limits

- The HGEMM families hard-code two stages; a different depth there means changing the kernel source, not passing a parameter.
- Deeper is not monotonically better: once global latency is hidden, further stages only take LDS away from occupancy.

## 8. Reduction, epilogue and write-back

- `epilogue` — How is the accumulator rearranged and written?

Also bears on this step: `flydsl-a8-scale-granularity-contract` (step 6), `gemm-split-k-when-grid-underfills` (step 3), `flydsl-a8-preshuffle-config-space` (step 3), `flydsl-mfma-fragment-lane-layout` (step 4), `flydsl-test-parity-gate` (step 10), `grouped-moe-parity-gate-is-cosine-not-elementwise` (step 10).

### How does the FlyDSL split-K HGEMM combine partial results, and what does that impose on the launcher?

**Card:** `flydsl-splitk-reduction-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `work_partition`, `runtime_contract`, `configuration_constraints`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- split_k > 1 is a candidate, or a split-K kernel is being ported or written

#### Try

- Write the combine in the kernel as the HGEMM does, in three parts. First, only the split with k index 0 initialises the output tile — to zero, or to the bias when there is one, so the bias is added exactly once — then writes back L2 and raises the tile's counter to 1 (`zero_c`). Second, every split spins on that counter until it is non-zero before touching the output (`split_k_barrier`). Third, each split rounds its fp32 accumulator to the output dtype and adds it into C with `llvm.AtomicRMWOp(fadd)`, two elements per atomic, at agent scope (the `if const_expr(IS_SPLIT_K):` store block).
- The small-M family writes the same combine per N tile (a tile-indexed counter and the same packed fadd), so either kernel is a precedent for an authored split-K.
- Keep one global semaphore buffer per stream, sized SPLIT_K_SIGNAL_STATE_COUNT (3) x SPLIT_K_COUNTER_MAX_LEN (128) counters, and pass the current signal state into every launch; split 0 also clears the previous state's counters while the current ones are in use.
- Advance the per-stream signal state after a launch only when split_k > 1, cycling modulo 3, so consecutive split-K launches on one stream use different counter slices.
- Reject configurations whose output-tile count exceeds the counters: with split_k > 1, ceil(M/tile_m)*(N/tile_n) must be at most 128 — for large M this bounds how small tile_m can go while split-K is on.
- When porting, keep the combine bookkeeping in the host/launcher contract: a kernel copied without the semaphore allocation and the state rotation does not have the counters its reduction waits on.

#### Why this is a candidate

- The capacity check, the state constants and the rotation helpers are all module-level code in the entry point, so the contract is stated rather than implied.
- The kernel half — initialise, wait, atomically add — is three blocks in the kernel body, cited with their extents, so a port can copy the whole mechanism rather than its first line.
- The capacity bound interacts with the tile search: a split-K candidate that is legal by divisibility can still be rejected by the counter check.

#### Keep as alternatives

- split_k=1, which needs none of this.
- A two-pass combine — partial buffers plus a reduce kernel — as the Triton GEMM does.

#### Same question in other implementations

- **triton** — partial buffers per K split plus a reduce kernel (`NUM_KSPLIT`, `ACTUAL_KSPLIT`/`MAX_KSPLIT`), or `tl.atomic_add` into the output in the atomic variant. Evidence: `src_0f50a57ed622a4f4` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:77`), `src_8f1fdafc50973349` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py:209`), `src_f5043ecafb832fa0` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py:55`), `src_9e9cdadae23d0f5f` (`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16_atomic.py:139`)
- **asm** — the split count is part of the selected object's config and becomes the launch grid's z dimension. Evidence: `src_05e16505bb238998` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:115`), `src_4044f249c79abf90` (`csrc/py_itfs_cu/asm_gemm_a16w16.cu:250`)

#### Evidence

- `src_1a4b9ad5c5b109de` — `split_k` `zero_c` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:296-391`
- `src_ad151ebc96795a5e` — `split_k` `split_k_barrier` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:393-432`
- `src_5ad1990942237536` — `split_k` `IS_SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:615-616`
- `src_0294fc9392d3670a` — `split_k` `IS_SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:830-905`
- `src_869d39483177e42c` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:875-882`
- `src_284881f3cc4c325a` — `split_k` `split_k_barrier` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:712-751`
- `src_39dac9712009eb08` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:966-973`
- `src_564badda94311abc` — `split_k` `SPLIT_K_SIGNAL_STATE_COUNT` at `aiter/ops/flydsl/gemm_kernels.py:38`
- `src_7c7ddd401ff4aaeb` — `config_limit` `SPLIT_K_COUNTER_MAX_LEN, 128` at `aiter/ops/flydsl/gemm_kernels.py:37`
- `src_194c17aa69828857` — `split_k` `SPLIT_K_GLOBAL_SEMAPHORE` at `aiter/ops/flydsl/gemm_kernels.py:66`
- `src_86aa6d1400d0f17f` — `split_k` `_advance_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:688-692`
- `src_94058066019b30f2` — `split_k` `_get_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:683-685`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695-707`
- `src_679a944e53f65823` — `config_validity` `Split-K counter capacity exceeded:` at `aiter/ops/flydsl/gemm_kernels.py:704-707`
- `src_67a72909c71307d1` — `split_k` `SPLIT_K_SIGNAL_STATE_COUNT` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:24`
- `src_217562a4a9b879dc` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:85`
- `src_636fbc44252bb90e` — `parity_tolerance` `max_delta_limit, 32.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:86`

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/numerics.md`](../operators/splitk_streamk_gemm/numerics.md)
- [`operators/splitk_streamk_gemm/backends/flydsl.md:77-85`](../operators/splitk_streamk_gemm/backends/flydsl.md)
- [`languages/flydsl/deep.md:105-114`](../languages/flydsl/deep.md)
- [`languages/flydsl/knobs.md:52-60`](../languages/flydsl/knobs.md)

#### Limits

- The atomic combine adds partials that were already rounded to the output dtype, in whatever order the splits arrive: error grows with split_k and results are not bit-reproducible run to run. Gate against the task's oracle tolerance, not against a split_k=1 output.
- AITER's own test of this kernel holds most split-K cases to its default gate but relaxes its two M=1 B-in-LDS cases, to 99.0% of elements close with a max |delta| of 8 at split_k 8 and 32 at split_k 16 (flydsl-test-parity-gate); a task oracle stricter than that is a reason to prefer a two-pass or fp32 combine.
- The counter capacity is a property of the current kernel; an authored split-K with a different combine has its own bound.
- Whether split-K pays for a shape is a measurement; this card only states what a legal split-K launch requires.

### On mxfp4 stage 2, which epilogue / output-dtype / persistence combinations exist, and which are mutually exclusive?

**Card:** `grouped-moe-mxfp4-stage2-epilogue-picks-the-output-format` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `configuration_constraints`, `execution_geometry`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- the a4w4 / mxfp4 stage-2 grouped GEMM

#### Try

- Treat epilogue and output dtype as one decision: `out_dtype='fp8'` requires `epilog='reduce'`, `out_dtype='fp4'` requires `epilog='scatter'` (and `scatter` in turn requires `BM == 128`), and only `bf16` is free to use the atomic path. There is no epilogue-independent output format.
- Do not combine the atomic combine with an fp4 output or a cshuffle epilogue — the name parser rejects it outright (`illegal mxmoe g2 name ... atomic incompatible with f4out|cshuffle`), because the atomic add is what performs the topk combine and a packed-fp4 or LDS-shuffled store has nothing to add into.
- Read `persist` as a property of the epilogue rather than a free flag on the native path: persistence is `epilog in ('nonatomic','nonatomic_mxfp4')`, and even then the grid is only clamped to the CU count when total tiles exceed `NUM_CU * 4` — below that it degenerates to one block per tile. The XCD swizzle is wired only into the persistent branch, so an `atomic` kernel carrying an xcd token in its name is not swizzling.
- Avoid the one combination that is broken rather than merely illegal: `persist=True` with an fp8 A operand and no `persist_flat` is rejected as a known-bad path (cosine 0 at large M), and `persist=True` requires a non-zero `cu_num` to clamp against.
- Keep the BN/BK choice inside the shipped set: stage 2 accepts `BN in {128,256,512}` and `BK in {128,256}` on the v2 dispatcher, but `out_dtype='fp4'` narrows BN to 128 or 256 because fp32 C staging at BN=512 would need 256 KB of LDS; the native stage-2 compiler welds both to 256 outright.

#### Why this is a candidate

- Each epilogue owns the output buffer differently (atomic accumulates into it, reduce leaves partials for a second kernel, scatter writes a packed quantized payload plus its scale bytes), so the output dtype is a consequence of who writes it.
- Stating the exclusions as a list beats discovering them one compile at a time: all of them are compile- or parse-time raises, so each wrong guess costs a build.

#### Keep as alternatives

- A single bf16 atomic epilogue everywhere — one code path, no exclusions to track, but it forfeits the fp8/fp4 routed-output formats that exist to avoid a separate requantization pass.

#### Evidence

- `src_0644ef057d70e344` — `config_validity` `illegal mxmoe g2 name {kname!r}: atomic incompatible with {bad}` at `aiter/ops/flydsl/mxfp4_kname.py:133-135`
- `src_b68a23ddea180f6f` — `config_validity` `out_dtype must be` at `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:171-173`
- `src_787c6bd2b6e02ef9` — `config_validity` `a8w4/fp8-A gemm2 persist is not supported (known-broken F2 path: cos=0…` at `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:288-291`
- `src_7ef4a25f18f24f3b` — `config_validity` `persist=True requires cu_num>0, got {cu_num}` at `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:285`
- `src_a05565e3105dc883` — `config_validity` `flydsl mxfp4 gemm2 unsupported variant` at `aiter/ops/flydsl/mxfp4_gemm2_kernels.py:96-99`
- `src_3da4fe5eeb03e3f1` — `config_space` `_SUPPORTED` at `aiter/ops/flydsl/mxfp4_gemm2_kernels.py:11-23`
- `src_9f05040744f6e010` — `tile_shape` `BN, 256` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:88`
- `src_6201541591cb4036` — `tile_shape` `BK, 256` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:89`
- `src_f5a63189caf8f4ef` — `grid_mapping` `grid_x, 1, 1` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:303`
- `src_99b9dabdc3cd562f` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:899-906`

#### Limits

- Two stage-2 implementations are merged here: the native `mxfp4_gemm2.py` (BN and BK welded to 256, persistence from `epilog`) and the v2 `mxmoe_dispatcher.py`/`mxmoe_gemm_v2.py` path (wider BN/BK, explicit `persist`). Check which one a name routes to before applying a bound from the other.
- Nothing here measures where atomic stops beating reduce; the existing `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` card carries the shipped per-bucket seed for that.

### Where does the gate/up activation happen in a fused stage-1 MoE GEMM, and why can it not stay in registers?

**Card:** `grouped-moe-mxfp4-stage1-fuses-activation-through-lds` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `operand_layout`, `pipeline_schedule`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- stage-1 grouped GEMM that fuses a gated activation (silu / swiglu / situv2) over a gate+up fused N

#### Try

- Plan for a full LDS round trip after the K loop, not an in-register epilogue: MFMA leaves the gate column and its paired up column in DIFFERENT waves' accumulators, so stage 1 barriers, writes every accumulator to an f32 LDS buffer, barriers again, and reads the pair back before multiplying. Any `k_wave` cross-wave reduction happens on that read.
- Add bias, if any, at that same point — after the pair is reassembled and BEFORE the activation. There is one site, and a bias applied anywhere else is a different function.
- Pick the activation by its clamping semantics, not just its name: `silu` clamps gate and up before the multiply (the host passes `+inf` for ordinary unclamped SiLU, so `swiglu_limit` is live even here), `swiglu` clamps gate from above only and up on both sides then uses a 1.702-alpha sigmoid with `(up + 1.0)`, and `situv2` is unclamped with host-supplied reciprocals. Each has its own positivity check at compile time.
- Treat the gate/up weight layout (`interleave`, tagged `il` vs `sep`) as a free compile-time flag with no validation: separated is the default and de-interleaves on the fly by assigning alternate 16-column MFMA tiles to gate and up. Either layout produces the same accumulator assignment, so it is a load-addressing choice, and the same config name can compile both.

#### Why this is a candidate

- The round trip is forced by the operand layout, not chosen for convenience: a gated activation needs two columns that the MFMA fragment layout places in different waves.
- Because the LDS accumulator is the meeting point, it is also the only place bias and the cross-wave k_wave sum can be applied, which is why all three collapse into one epilogue.

#### Keep as alternatives

- Keeping the activation in registers by tiling N so one wave owns both halves of a pair — no LDS round trip, but it constrains the N tiling to the gate/up split rather than to the MFMA shape.
- A separate activation kernel after stage 1 — simplest to reason about, at the cost of writing and re-reading the whole intermediate.

#### Evidence

- `src_759b694e2df7c154` — `config_validity` `act must be` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1531`
- `src_851a8600c1f27eb0` — `config_validity` `SiTUv2 beta values must be positive` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1533`
- `src_e55d60318b3ae2e7` — `config_validity` `Swiglu limit must be positive` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:1535`
- `src_85a0084052e0aba5` — `config_validity` `flydsl mxfp4 gemm1 requires act in` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:94-97`
- `src_21afbb8224a4d7a8` — `config_validity` `flydsl mxfp4 gemm1 requires situ_beta > 0, got {situ_beta!r}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:100-102`
- `src_94c01495845eaba9` — `config_validity` `flydsl mxfp4 gemm1 requires situ_linear_beta > 0,` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:104-107`
- `src_a95f47b4e798c75a` — `config_validity` `flydsl mxfp4 gemm1 requires swiglu_limit > 0, got {swiglu_limit!r}` at `aiter/ops/flydsl/mxfp4_gemm1_kernels.py:109-111`
- `src_dfcd02dec23c4cef` — `lds_pad` `ACC_LDS_PAD_DW, 4` at `aiter/ops/flydsl/kernels/mxfp4_gemm1.py:44`

#### Limits

- Read from the mxfp4 stage-1 kernel only; the generic 2-stage stage-1 builder was not re-read for this card and may stage its activation differently.
- Nothing here measures the cost of the round trip against a register-resident alternative — no such alternative exists in this tree to compare against.

### In the atomic stage-2 combine, where does the routing weight get applied and what shape does the atomic actually have?

**Card:** `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `runtime_contract`, `memory_access`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- mxfp4 stage-2 grouped GEMM using the atomic bf16 combine

#### Try

- Apply the topk weight in f32, then convert, then atomically add: the epilogue multiplies each accumulator pair by the per-row weight as f32, converts the pair to bf16, and issues ONE packed-bf16x2 atomic fadd (monotonic, agent scope, 4-byte aligned). Converting before weighting changes the rounding, and weighting after the atomic is not expressible.
- Expect a transpose through LDS on the way there, and budget for it: the MFMA leaves four strided rows per lane, while a packed 2-wide atomic needs contiguous column pairs, so the accumulators go through LDS purely to be re-laned. Each lane then issues four atomics 64 columns apart to cover BN=256.
- Hoist the sorted-position lookups: the token ids and routing weights are loaded at the very top of the epilogue as `invariant` loads, before the accumulator stores, so the long-latency gather overlaps the LDS traffic rather than stalling the atomic.
- Mask the token id before using it: the sorted id word carries a payload in its top 8 bits and only the low 24 are the token index, which is then bounds-checked against M. A raw compare against the packed word reads padding rows as real tokens.

#### Why this is a candidate

- The combine is the only place the per-token routing weights can be applied without materializing topk separate outputs, so its ordering is the numerics of the whole MoE.
- The packed-bf16x2 form is what makes the atomic affordable: one atomic per two output columns instead of per column, which is why the LDS re-lane is worth a barrier.

#### Keep as alternatives

- The `reduce` epilogue, which stores unsummed partials and runs a second kernel to sum over topk — no atomic contention, but an extra launch; see `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` for which bucket AITER ships which.
- Deferring the weight to the reduction kernel (`g2_defer_weight` on the v2 path) — available only on the reduce path, since the atomic has nowhere to defer to.

#### Evidence

- `src_99b9dabdc3cd562f` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:899-906`
- `src_39531c6fe9b3aac8` — `cache_policy` `nontemporal=True` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:808`
- `src_73b546c7214cd995` — `cache_policy` `b` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:452`
- `src_c90a1f56cfbe7215` — `access_width` `load_b_scale_tile` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:423-434`
- `src_cb67223b021205a3` — `access_width` `load_b_tile` at `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:436-456`

#### Limits

- Read from the native `mxfp4_gemm2.py` atomic epilogue; the v2 `mxmoe_gemm_v2.py` epilogue has additional output formats (fp8-routed reduce, deferred weighting, alternate store cache policies) whose weight ordering was not separately confirmed.
- The meaning of the top 8 bits of the packed sorted-id word is not decoded in the stage-2 source; only the masking is.

### When can a FlyDSL GEMM route its accumulators through an LDS CShuffle epilogue, and what does it cost?

**Card:** `flydsl-cshuffle-epilogue` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `configuration_constraints`, `workgroup_tile`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- the output store path (layout, vector width, atomics) is being chosen

#### Try

- Treat the epilogue as a choice between the row epilogue, which stores straight from the MFMA accumulator layout, and the LDS CShuffle epilogue, which writes the tile row-major into LDS, barriers, re-reads it with threads remapped to (MLane, NLane) = (8, 32) and emits vectorised half2 stores or atomics.
- Check legality first: block_size must divide by cshuffle_nlane (32), tile_m by CShuffleMLane (256/32 = 8 for the 256-thread A8 block), and tile_n by cshuffle_nlane*e_vec (64 by default).
- Price the LDS: the staging buffer is 2*tile_m*tile_n bytes and shares the A ping/pong space in the family's estimate; when it overflows, the helper can split the buffer between two wave groups.
- Use its `store_pair` hook for split-K or atomic accumulation: the same shuffled layout feeds either a store or an atomic.
- Keep both arms: in AITER's gfx950 fp8 selections CShuffle on and off both occur in every M bucket (22 of 72 rows on at M<=16, 32 of 50 at M 1025-4096).
- In the split-K HGEMM the equivalent (`c_to_lds`) is fixed off and rejected if requested.

#### Why this is a candidate

- The helper states its legality as raises and its data flow in its docstring, so the contract is readable rather than reverse-engineered.
- The epilogue is the only place the corpus has no other card for, and it is where store coalescing and split-K accumulation are decided.

#### Keep as alternatives

- Direct row store from the MFMA layout — the default and the control arm.

#### Same question in other implementations

- **ck** — the CShuffle epilogue is part of the device-op template (`CShuffleDataType`, `*_Xdl_CShuffle*` instances). Evidence: `src_5ebdfc64c47345bb` (`csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh:52`), `src_7e81eb9b49b377d5` (`csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_common.cuh:50`), `src_780fadebcc954d06` (`csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh:86`)

#### Evidence

- `src_1786f3708aa447c0` — `layout_epilogue` `c_shuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:15`
- `src_0a9c121156ff2c92` — `layout_epilogue` `c_shuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:85-416`
- `src_d0d34580300e6c02` — `config_validity` `block_size ({block_size}) must be divisible by cshuffle_nlane ({cshuff…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:132-134`
- `src_4c3052f8521fc401` — `config_validity` `tile_m must be divisible by CShuffleMLane ({cshuffle_mlane}), got tile…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:137-139`
- `src_949b583f0277e194` — `config_validity` `tile_n must be divisible by (CShuffleNLane*EVec) = {cshuffle_nlane*e_v…` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:143-145`
- `src_756a8fa4df5e93d9` — `layout_epilogue` `CShuffle` at `aiter/ops/flydsl/kernels/mfma_epilogues.py:160`
- `src_c701f7377a04c988` — `config_validity` `preshuffle_gemm_estimated_lds_bytes` at `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py:119-157`
- `src_f83d827dcfa59956` — `layout_epilogue` `use_cshuffle_epilog` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:134`
- `src_d0f75ad1bcdf6730` — `config_validity` `Current kernel only supports c_to_lds={FIXED_C_TO_LDS};` at `aiter/ops/flydsl/gemm_kernels.py:128-131`
- `cfg_909edfc582fce0bd` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M<=16`: 274 shipped row(s)
- `cfg_496162f129968cb9` — FlyDSL `a8_preshuffle` `af8_wf8_b16` selections in `a8w8_bpreshuffle_tuned_gemm` on `gfx950`/256 CUs, `M1025-4096`: 661 shipped row(s)

#### Background (always-on reference docs, not measurements)

- [`operators/gemm_epilogue_fused/backends/flydsl.md:20-26`](../operators/gemm_epilogue_fused/backends/flydsl.md)
- [`languages/flydsl/authoring_gemm_levers.md:167-180`](../languages/flydsl/authoring_gemm_levers.md)

#### Limits

- Legality is the helper's; the LDS cost interacts with the A staging budget, so re-run the LDS estimate for the combined configuration.
- Selection shares are where AITER chose each arm, not how much either arm helped.

### How does the stage-2 GEMM accumulate `topk` partial outputs per token into one row, and what does the choice cost at the host level?

**Card:** `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` · **type:** `performance_candidate` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `epilogue`, `runtime_contract`, `tunable_surface`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- the design combines topk expert outputs per token after stage 2 (a grouped-GEMM combine, not a single dense GEMM write)

#### Try

- Treat 'atomic' and 'reduce' as two different launch shapes, not two values of one flag: the atomic path adds every partial straight into the output inside stage 2's own epilogue (`llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, ...)`), while the reduce path leaves stage 2's output unsummed and runs a second kernel out of its own module (`moe_reduce.py`), separately `lru_cache`-compiled by `compile_moe_reduction` and separately launched on its own grid, to sum over the topk dimension.
- Pick between them by shape, following AITER's own shipped table rather than guessing: in the GLM-5.2 mxfp4/mxfp8 fused-MoE rows, every low-M bucket (`M<=16` through `M257-1024`) ships `atomic`[,`nt`], and every bucket at `M>4096` ships `reduce` (sometimes with `persist`) — the table switches somewhere in between, not at one fixed, universal cutover.
- Budget for the second kernel launch when reduce is selected: it is a real extra host-side dispatch, not an extra epilogue instruction.

#### Why this is a candidate

- Atomic serializes writers on the same output address across however many of the `topk` experts land a row there; at large M both the row count and the per-row contention grow, so AITER moves the summation off the atomic path into an unsummed write followed by one coalesced reduction kernel.
- The shipped table encodes this tradeoff per M bucket for the GLM-5.2 pilot shape, so it is a cheap seed to start from before resweeping.

#### Keep as alternatives

- Always atomic — one kernel, no separate reduction pass, but every partial write contends on the same output row.
- Always reduce — no atomic contention, but a mandatory second kernel launch even at the smallest M.

#### Evidence

- `src_a9035f819c6590bc` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:2483-2490`
- `src_7710996bc1db7f10` — `jit_cache` `1024, compile_moe_reduction` at `aiter/ops/flydsl/kernels/moe_reduce.py:202`
- `src_081b5f3cb3e34298` — `grid_mapping` `fx.Int64(i32_m_tokens` at `aiter/ops/flydsl/kernels/moe_reduce.py:288`
- `src_eb530d7276be8a22` — `atomic_combine` `fadd` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:2605-2612`
- `src_b20ee136736bd4b8` — `jit_cache` `1024, compile_moe_reduction` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:2703`
- `cfg_555872067e503a14` — FlyDSL `moe_stage1` `afp4_wfp4_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M>4096`: 7 shipped row(s)
- `cfg_f851ab2244623ab6` — FlyDSL `moe_stage1` `afp8_wfp8_bf16` selections in `tuned_fmoe` on `gfx950`/256 CUs, `M>4096`: 7 shipped row(s)
- `sel_71c693f9258640e7` — backend selected by AITER's tuner in `tuned_fmoe` on `gfx950`/256 CUs, `M>4096`: flydsl 14 (rows)

#### Limits

- `persist` further changes how the reduce kernel is scheduled across CUs (see `gemm-workgroup-xcd-mapping` for persistent-kernel XCD mapping); this card only separates atomic-epilogue from separate-reduction, not what tunes each.
- The bucket boundary is read off one shape family (gfx950, 6144x256x257/topk9); it is a seed for the M-bucket sweep, not a boundary proven to generalize to other MoE shapes.
- Both halves are cited live at current main (`mixed_moe_gemm_2stage_common.py` for the atomic epilogue, `moe_reduce.py` for the separate reduction); the two older `moe_gemm_2stage.py` citations are pinned at commit `a6bb4993`, where that module held both paths together, and that file no longer exists upstream.

## 9. Host configuration, ABI and launch

- `runtime_contract` — Which host-side layout, argument block, workspace and launch contract must the kernel satisfy?

Also bears on this step: `flydsl-a8-scale-granularity-contract` (step 6), `gemm-blockscale-k-loop-scale-application` (step 6), `gemm-split-k-when-grid-underfills` (step 3), `flydsl-preshuffle-b-layout-contract` (step 4), `flydsl-splitk-reduction-contract` (step 8), `gemm-default-backend-routing` (step 2), `grouped-moe-parity-gate-is-cosine-not-elementwise` (step 10), `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` (step 8), `grouped-moe-mxfp4-kernel-name-is-the-config-space` (step 10), `grouped-moe-mxfp4-native-scale-layout-is-a-cross-stage-contract` (step 4), `grouped-moe-mxfp4-stage2-combine-weights-before-the-atomic` (step 8).

### What does the host side of a FlyDSL GEMM have to provide in AITER — compile cache, launch grid, per-stream state and kernel name?

**Card:** `flydsl-host-launch-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `runtime_contract`, `work_partition`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `scaled_quant_gemm`, `grouped_gemm_moe`, `fused_moe_grouped_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a FlyDSL kernel is being launched, cached or registered for dispatch

#### Try

- Compile once per configuration and cache it: AITER's HGEMM, small-M, A8 preshuffle and 2-stage MoE compilers are `@functools.lru_cache(maxsize=1024)` (the mixed-precision MoE ones unbounded), keyed on the full configuration, so every new tile or split is a new compile — warm up before timing.
- Make the launch grid match what the kernel reads: the HGEMM and small-M launch `grid=(bm, bn, SPLIT_K)` — block x is the M tile, y the N tile, z the K split — while the A8 GEMM launches `(gx, gy, 1)`. A kernel that reads `block_idx.z` as its split needs the host to launch that z.
- Supply split-K's per-stream state: the counter buffer and the current signal state go into every split-K launch and the state advances after it (flydsl-splitk-reduction-contract).
- Name the kernel in the registry's grammar: tuned rows encode every knob in the kernel name matched by `_HGEMM_KERNEL_RE`, and `tuned_gemm` drops a `flydsl` row whose name does not parse.
- Check every FlyDSL symbol against the installed version before copying code across versions (version_map); a module that moved fails at import, not at compile.

#### Why this is a candidate

- Each of these is a failure that surfaces only at run time — a recompile per call, a grid that leaves splits unlaunched, counters that were never allocated, a tuned row silently dropped.

#### Keep as alternatives

- Call the library entry points (`flydsl_hgemm`, `flydsl_preshuffle_gemm_a8`), which own this contract, instead of launching an authored kernel directly.

#### Evidence

- `src_4d28023c79d1fe52` — `jit_cache` `1024, compile_hgemm_kernel` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:102`
- `src_1e5ae08be5adf756` — `jit_cache` `1024, compile_small_m_hgemm_kernel` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:388`
- `src_ca495db46be070cd` — `jit_cache` `1024, compile_preshuffle_gemm_a8` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:123`
- `src_ada62c940d6fc161` — `jit_cache` `1024, compile_moe_gemm1` at `aiter/ops/flydsl/kernels/moe_gemm_2stage.py:87`
- `src_821a059af3542249` — `grid_mapping` `bm, bn, SPLIT_K` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:927-931`
- `src_569b9cec48bd2103` — `grid_mapping` `bm, bn, SPLIT_K` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:1399-1403`
- `src_328a63de09b9204a` — `grid_mapping` `gx, gy, 1` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:1674-1678`
- `src_1e7dc0218412b0cf` — `instance_name` `_HGEMM_KERNEL_RE` at `aiter/ops/flydsl/gemm_kernels.py:44-63`
- `src_c9b676e9661f7f1e` — `kernel_family` `flydsl` at `aiter/tuned_gemm.py:133-141`
- `src_86aa6d1400d0f17f` — `split_k` `_advance_split_k_signal_state` at `aiter/ops/flydsl/gemm_kernels.py:688-692`
- `src_194c17aa69828857` — `split_k` `SPLIT_K_GLOBAL_SEMAPHORE` at `aiter/ops/flydsl/gemm_kernels.py:66`

#### Background (always-on reference docs, not measurements)

- [`languages/flydsl/deep.md:116-122`](../languages/flydsl/deep.md)
- [`languages/flydsl/version_map.md:278-315`](../languages/flydsl/version_map.md)
- [`languages/flydsl/patterns.md:104-108`](../languages/flydsl/patterns.md)

#### Limits

- This is AITER's host contract at the pinned commit; an authored kernel launched outside AITER's wrappers must provide the same pieces itself.

### What has to agree between the stage-1 and stage-2 compiles when a shared expert is fused into the same grouped-GEMM call as the routed experts?

**Card:** `grouped-moe-shared-expert-fusion-is-a-cross-stage-identity-contract` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `runtime_contract`, `configuration_constraints`, `implementation_variant`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a shared expert is appended to the routed experts as one more row of the same grouped GEMM (heterogeneous B), rather than run as its own dense GEMM

#### Try

- Give both compiles the same `shared_expert_id`, and make it the last expert index: `compile_mixed_moe_gemm1_common` and its stage-2 counterpart each raise unless `shared_expert_id == experts - 1`, so the identity is checked twice, once per stage, never inferred from the expert count alone.
- Only fuse a shared expert into an MXFP4-weight call: both stages separately raise 'Heterogeneous B requires MXFP4 routed weights' when `shared_expert_id` is set and the routed B dtype is not fp4 — there is no heterogeneous path here for fp8 or bf16 weights.
- Read a fused call's expert axis as routed-experts-plus-one before matching it against a tuned table: GLM-5.2's 256 routed experts plus its one shared expert is why its shipped fused-MoE rows key on `expert=257`, not 256.

#### Why this is a candidate

- A heterogeneous B tile (the shared expert's weight sitting at a different physical location than the routed experts') is only safe to address if the kernel that built the row-to-expert mapping and the kernel that reads it agree on exactly which row index is the special one; checking it in both compiles turns a silent mismatch into a compile-time `ValueError` in either stage.

#### Keep as alternatives

- Run the shared expert as its own ordinary (non-grouped) GEMM outside the grouped call — avoids the heterogeneous-B and ID-identity contract entirely, at the cost of a separate kernel launch and losing the fusion with the routed experts' combine.

#### Evidence

- `src_0a391fc56eb3134e` — `config_validity` `FHMoE stage1 requires shared_expert_id == experts - 1;` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:146-149`
- `src_9c2d5c214ffbc124` — `config_validity` `FHMoE stage2 requires shared_expert_id == experts - 1;` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:3162-3165`
- `src_b25a5418edbd6918` — `config_validity` `Heterogeneous B requires MXFP4 routed weights` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:164`
- `src_4ce856c5ae84c9a4` — `config_validity` `Heterogeneous B requires MXFP4 routed weights` at `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage_common.py:3193`

#### Limits

- The MXFP4-only restriction and the id-equals-last-expert rule are both read directly off `mixed_moe_gemm_2stage_common.py`; whether every FlyDSL 2-stage MoE variant in the tree enforces the same two constraints the same way was not checked file-by-file.

## 10. Correctness validation and tuning

- `validation_contract` — Which reference and tolerance does the implementation's own test hold it to, and where does it relax them?
- `tunable_surface` — Which knobs exist, which values are legal seeds, and which fields are derived rather than tunable?

The oracle, parity gate and timing protocol belong to the workflow's verify step, not to this corpus; cards here state only what a kernel's own contract changes about validation.

Also bears on this step: `flydsl-hgemm-config-space` (step 3), `flydsl-a8-preshuffle-config-space` (step 3), `gemm-blockscale-shipped-bucket-structure` (step 3), `gemm-streamed-operand-cache-policy` (step 5), `grouped-moe-stage2-combine-is-a-kernel-choice-not-a-flag` (step 8), `grouped-moe-mxfp4-stage1-variant-whitelist` (step 3).

### Which parity does AITER's own FlyDSL GEMM test demand, where does it relax it, and what does that say about validating an authored kernel?

**Card:** `flydsl-test-parity-gate` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `validation_contract`, `work_partition`, `epilogue`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `gemm`, `dense_gemm`, `splitk_streamk_gemm`, `skinny_gemv_decode`, `gemm_epilogue_fused`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a FlyDSL GEMM — especially a split-K one — is being validated against a reference

#### Try

- Gate against an fp32 reference first: AITER's split-K HGEMM test compares with `torch.isclose` at atol = rtol = 1e-2 and requires 99.9% of elements close.
- Relax only where the implementation had to: AITER's split-K precision cases at split_k 8 (M=104, K=7168) and split_k 4 (M=1, K=512) hold the default gate; only its two M=1 `b_to_lds` cases are relaxed, to 99.0% of elements with max |delta| <= 32 at split_k 16 (K=7168) and <= 8 at split_k 8 (K=1536). The drift comes from the atomic combine of partials already rounded to bf16 (flydsl-splitk-reduction-contract).
- Let the task's oracle and tolerance decide acceptance (the workflow's verify step); use AITER's gate to judge a failure — drift within it is the known cost of split-K, drift beyond it points at a bug.

#### Why this is a candidate

- The relaxed cases are the implementation stating where its numerics drift; without them a correct split-K kernel reads as broken, or a broken one as merely imprecise.

#### Keep as alternatives

- Split-K with an fp32 workspace and a separate reduce pass (the Triton pattern), which keeps the combine in fp32 at the cost of a second kernel.

#### Evidence

- `src_9d1d40ad1d82f015` — `parity_tolerance` `DEFAULT_ATOL, 1e-2` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:41`
- `src_f0baf9a97c14b111` — `parity_tolerance` `DEFAULT_RTOL, 1e-2` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:42`
- `src_1183d7ed5d869e8a` — `parity_tolerance` `DEFAULT_PASS_PCT, 99.9` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:43`
- `src_217562a4a9b879dc` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:85`
- `src_636fbc44252bb90e` — `parity_tolerance` `max_delta_limit, 32.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:86`
- `src_972b88aa5143a3f7` — `parity_tolerance` `pass_pct, 99.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:102`
- `src_f30b29895e6d16b2` — `parity_tolerance` `max_delta_limit, 8.0` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:103`
- `src_724636e4ecf6c5f6` — `config_validity` `_check_output` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:129-154`
- `src_dcea4e10d8996a4e` — `config_space` `SPLITK_PRECISION_CASES` at `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py:46-105`

#### Background (always-on reference docs, not measurements)

- [`operators/splitk_streamk_gemm/backends/flydsl.md:77-85`](../operators/splitk_streamk_gemm/backends/flydsl.md)
- [`operators/dense_gemm/numerics.md:59-63`](../operators/dense_gemm/numerics.md)

#### Limits

- These are AITER's regression cases for its own kernel; an authored kernel is judged by the task's oracle, not by this test.

### A grouped/MoE GEMM is being validated against a reference — which parity does AITER's own test actually demand, and why is it not the dense GEMM's elementwise gate?

**Card:** `grouped-moe-parity-gate-is-cosine-not-elementwise` · **type:** `constraint` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `validation_contract`, `epilogue`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`

#### Use when

- target_language=flydsl
- a grouped (MoE) GEMM — one or both stages, or the fused two-stage path — is being compared against a torch reference

#### Try

- Gate on the COSINE distance, not on elementwise closeness: the shipped criterion is `logits_diff = ||x-y||^2 / (||x||^2 + ||y||^2) < 0.01`, declared as `LOGITS_DIFF_TOL = 0.01` (`op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:90`) and asserted as `logits_diff > 0.01` in the two-stage test (`op_tests/test_moe_2stage.py:570`). Treat an elementwise rel-L2 or isclose ratio as an informational print beside it, which is exactly the role the grouped test gives it.
- Do not carry the dense family's elementwise gate over (flydsl-test-parity-gate). The stated reason it was abandoned is quantization, not kernel error: the EP test records that a path quantizing activations to fp8 'differs elementwise by more than atol/rtol=5e-2 even without EP', and switches to the cosine criterion for it (`op_tests/test_moe_ep.py:813`) while keeping `atol=rtol=5e-2` elementwise only off that path (`:821`).
- Check NaN separately and deterministically, as its own assert — never let the tolerance gate stand in for it. The two-stage test added an explicit NaN probe precisely because `checkAllclose`'s err and the cosine diff 'can be masked by atomic-reduction noise', and a stage-2 atomic combine with an uninitialised K-padding tail is the defect it was written to catch.
- When authoring one stage, check that stage against its own reference rather than only the fused output: the FlyDSL MoE suite compares stage 1 alone, and feeds the REFERENCE stage-1 output into stage 2 so a stage-2 failure cannot be blamed on stage 1. Its helper fixes the shape of the check — NaN assert, Inf assert, then a bounded mismatch ratio (`op_tests/flydsl_tests/test_flydsl_moe.py:54`).
- Validate routing and sorting as EXACT, not as approximate, and keep them out of the numerics gate: the sorting test compares `num_tokens_post_padded` with `atol=0` (`op_tests/test_moe_sorting.py:221`) and requires its whole error dict to be zero. The two-stage test instead holds routing constant — the same `topk_weights`/`topk_ids` go to both reference and kernel — so a routing bug would not show up there at all.

#### Why this is a candidate

- Once activations are quantized to fp4/fp8 the reference and the kernel disagree elementwise by construction, so an elementwise bound tight enough to catch a real bug fails a correct kernel, and one loose enough to pass it catches nothing. A direction-of-vector metric is insensitive to that per-element quantization noise while still catching a wrong result.
- The failure modes a grouped MoE actually has are not elementwise drift: an uninitialised padding lane, a mis-indexed expert, a wrong sorted offset. NaN and exact index comparison catch those; a tolerance does not.
- The expected magnitude separates signal from noise and is the reason a bound this loose still discriminates — correct mxfp4 quantization error lands near 5e-3, while a broken scale indexing lands around 0.7.

#### Keep as alternatives

- The dense elementwise gate (isclose at atol=rtol=1e-2 with a pass percentile) — what AITER uses for its split-K dense GEMM, and what the MoE tests deliberately moved away from once activations are quantized.
- Bit-exactness, which AITER does demand where a path is supposed to be a pure passthrough rather than a requantization — appropriate only when no arithmetic may differ, since any tolerance wide enough to absorb quantizer noise would also absorb a real defect.

#### Evidence

- `src_56406869ba3f3068` — `parity_tolerance` `LOGITS_DIFF_TOL, 0.01` at `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:90`
- `src_fbc202be8570fb18` — `parity_tolerance` `VERIFY_TOL_A4W4, 0.02` at `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:85`
- `src_e5452814c4e01160` — `parity_tolerance` `VERIFY_TOL_A8W4, 0.02` at `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py:86`
- `src_d741a22a37929535` — `parity_tolerance` `logits_diff, 0.01` at `op_tests/test_moe_2stage.py:570`
- `src_08328711fc7e1ae6` — `parity_tolerance` `atol, 0` at `op_tests/test_moe_sorting.py:221-226`
- `src_bb2f27c92eb262d3` — `parity_tolerance` `_logits_diff_tol, 0.01` at `op_tests/test_moe_ep.py:813`
- `src_47c8b68457e67034` — `parity_tolerance` `atol, 5e-2` at `op_tests/test_moe_ep.py:821`
- `src_d1781a7127dd5d52` — `config_validity` `_check_close` at `op_tests/flydsl_tests/test_flydsl_moe.py:54-60`
- `src_a1a66f43d47e0462` — `parity_tolerance` `_check_close, ref, out, label, atol=1.0, rtol=0.05, max_err_ratio=0.05` at `op_tests/flydsl_tests/test_flydsl_moe.py:54-60`

#### Limits

- These are AITER's regression gates for its own kernels; an authored kernel is judged by the task's oracle and its tolerance, not by this card. Use the card to tell a known quantization drift from a bug, and to decide WHICH comparison to run, not to overrule a task that demands something stricter.
- The gate is weaker than it looks in two places that are worth reading before relying on it. The two-stage assert is a CONJUNCTION — it fails only when the elementwise error is non-zero AND the cosine diff exceeds 0.01 — and its strictness is switchable: the CLI sweep path passes `strict_accuracy=False`, which degrades both the NaN and the accuracy failure to warnings. Separately, the shared `checkAllclose` is a percentile reporter that returns a mismatch ratio rather than raising; the asserting wrapper exists but no MoE test calls it.
- The attribution of the relaxation to quantization is sourced; a relaxation attributed to the stage-2 ATOMIC combine is NOT — no comment in these tests says the tolerance was widened for atomics. Atomics appear only in the opposite role, as noise that can MASK a failure, which is why the NaN check was added alongside. Do not cite this card for 'atomics justify a looser bound'.
- Scope: the cosine gate is read off the mxfp4/a8w4 grouped and two-stage tests. Several older bf16/asm MoE tests still call the elementwise helper with `atol=100`, which is not a tighter alternative but an effectively disabled elementwise check — they were not treated as evidence for a gate here.

### Where is the configuration of an mxfp4 grouped-MoE GEMM actually written down, and what can be changed without changing it?

**Card:** `grouped-moe-mxfp4-kernel-name-is-the-config-space` · **type:** `semantic` · **evidence:** `source_observed` · **status:** `candidate_only` · **axes:** `tunable_surface`, `configuration_constraints`, `runtime_contract`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

#### Machine match

- `operator_families`: `grouped_gemm_moe`, `fused_moe_grouped_gemm`
- `target_languages`: `flydsl`
- `gfx`: `gfx950`
- `dtypes`: `mxfp4`, `fp4_e2m1`

#### Use when

- target_language=flydsl
- gfx950
- the a4w4 / mxfp4 block-scaled grouped MoE path (`flydsl_mxmoe_g1_*` stage 1, `flydsl_moe2_layout_*` or `flydsl_mxmoe_g2_*` stage 2)

#### Try

- Treat the kernel NAME as the configuration object, not as a label on one: `mxfp4_kname.py` holds a token grammar (`_MXMOE_NUMERIC_TOKENS`, `_MXMOE_G1_FLAG_TOKENS`, `_MXMOE_G2_FLAG_TOKENS`, `_FLYDSL_V2_GEMM2_RE`) and a parse/build pair, and the tuned CSV stores a config by storing its name. Every tile size, dtype, epilogue and flag reaches the compiler through that string.
- Change a config by changing the name and letting the parser reject it, rather than by passing a new argument: an unknown token raises `bad mxmoe kernel name ... unknown token`, so an illegal configuration fails at parse time instead of compiling into something that merely runs.
- Check WHICH stage-2 family a name belongs to before reading its tokens — two coexist with different grammars. `parse_g2_kname_any` dispatches between the native `flydsl_mxmoe_g2_a4w4_*` form and the v2 layout form `flydsl_moe2_layout_a{a}_w{b}_{out}_t{M}x{N}x{K}_{epilog}[_persist][_nt][_sbm{N}]` built by `build_flydslv2_gemm2_name`. The same concept (`persist`, `nt`) is spelled differently in each, and tokens legal in one are unknown in the other.
- Do not read the host-side dispatch name and the device-side symbol as two kernels: `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt` and `gemm1_a4w4_port_h6144_i256_ne257_bm16_iq_sep` are the same kernel under two naming schemes, the first from the config space and the second from the compiler's `@flyc.kernel(name=...)`.

#### Why this is a candidate

- An optimizer that treats the name as cosmetic will edit a launch argument that the dispatcher overwrites from the parsed name on the next call, and conclude the knob does nothing.
- The grammar is also the completeness statement: the flag token sets are the only place that says which flags exist at all, so a knob absent from them is absent from the kernel, not merely undocumented.

#### Keep as alternatives

- A keyword-argument config object with the name derived from it — what the dense FlyDSL GEMM path does; it makes an illegal config a Python TypeError rather than a parse error, but it cannot be stored in a CSV cell, which is what the tuned-table lookup needs.

#### Evidence

- `src_0a644836120e1b24` — `config_space` `_MXMOE_NUMERIC_TOKENS` at `aiter/ops/flydsl/mxfp4_kname.py:6-10`
- `src_1d74af857400b4b5` — `config_space` `_MXMOE_G1_FLAG_TOKENS` at `aiter/ops/flydsl/mxfp4_kname.py:11-20`
- `src_b4aaef8ed904898d` — `config_space` `_MXMOE_G2_FLAG_TOKENS` at `aiter/ops/flydsl/mxfp4_kname.py:21`
- `src_7fdfffec65089511` — `config_space` `_FLYDSL_V2_GEMM2_RE` at `aiter/ops/flydsl/mxfp4_kname.py:56-61`
- `src_8080a3948e3593a9` — `config_space` `_parse_mxfp4_g1_kname` at `aiter/ops/flydsl/mxfp4_kname.py:99-121`
- `src_4ca84ec1053e1925` — `config_space` `_parse_mxfp4_g2_kname` at `aiter/ops/flydsl/mxfp4_kname.py:124-147`
- `src_d918b57fec77fa2f` — `config_space` `parse_flydsl_v2_gemm2_kernel` at `aiter/ops/flydsl/mxfp4_kname.py:156-173`
- `src_675a596bbd78b93f` — `config_space` `parse_g2_kname_any` at `aiter/ops/flydsl/mxfp4_kname.py:176-200`
- `src_684ee007ec1d2fd1` — `config_space` `build_flydslv2_gemm2_name` at `aiter/ops/flydsl/moe_kernels.py:349-372`

#### Limits

- `mxfp4_kname.py` parses an `SK{n}` split-K token into both stage dicts, but no consumer of that parsed field was found in the stage-1/stage-2 compilers or in the host dispatcher — treat split-K as parsed-but-inert on this path until a consumer is shown, rather than as an available knob.
- The `_bf16lds` and `_sp{N}` suffixes the v2 regex accepts are never emitted by `build_flydslv2_gemm2_name`; the equivalent knobs exist one level down, in the compiler's own `gemm2_a4w4_port_*` symbol, which is a different name space. Do not transplant a token between the two.

## Shipped configuration seeds: FlyDSL (AITER tuned databases)

Generated from the FlyDSL rows of AITER's tuned GEMM databases (`aiter/configs/**/*tuned*gemm*.csv`), decoded from AITER's own kernel-name serialisation. Match gfx, CU count, database, family, dtype and M bucket before using a row. **Seed candidate** means every selected row in the group carries the value; **vary next** lists knobs that moved with the shape, with the most frequent values first. The tuner's timings are not carried, so these are concrete candidates, not measured winners.

### `gfx950` · 256 CUs · `a8w8_bpreshuffle_tuned_gemm` · FlyDSL `a8_preshuffle` · `af8_wf8_b16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_909edfc582fce0bd` | `M<=16` | 274 | none shared | `lds_stage`: 0: 192, 1: 82; `scheduler`: default: 238, default_ks7: 18, default_ks4: 12 (+3 more); `tile`: 16x64x512: 164, 16x32x1024: 64, 16x64x256: 22 (+4 more); `use_async_copy`: 0: 239, 4: 35; `use_cshuffle_epilog`: 2: 70, 0: 66, 1: 65 (+2 more); `waves_per_eu`: 2: 203, 1: 71 |
| `cfg_2f03fe974e5d2b5c` | `M17-64` | 133 | none shared | `lds_stage`: 0: 90, 1: 43; `scheduler`: default: 116, default_ks4: 7, default_ks3: 3 (+3 more); `tile`: 32x64x512: 39, 16x64x512: 28, 32x64x256: 11 (+17 more); `use_async_copy`: 0: 94, 4: 39; `use_cshuffle_epilog`: 1: 40, 2: 26, 4: 25 (+2 more); `waves_per_eu`: 2: 93, 1: 40 |
| `cfg_382465b289aad520` | `M65-256` | 116 | none shared | `lds_stage`: 1: 66, 0: 50; `scheduler`: default: 109, default_ks3: 3, default_ks2: 3 (+1 more); `tile`: 64x64x256: 29, 32x64x512: 25, 64x128x256: 17 (+17 more); `use_async_copy`: 4: 75, 0: 41; `use_cshuffle_epilog`: 2: 33, 0: 30, 1: 23 (+2 more); `waves_per_eu`: 2: 94, 1: 22 |
| `cfg_8245d4268291ab55` | `M257-1024` | 106 | `scheduler=default` | `lds_stage`: 1: 83, 0: 23; `tile`: 128x128x256: 26, 64x128x256: 15, 64x192x256: 15 (+22 more); `use_async_copy`: 4: 82, 0: 24; `use_cshuffle_epilog`: 2: 28, 0: 26, 3: 21 (+2 more); `waves_per_eu`: 2: 93, 1: 13 |
| `cfg_496162f129968cb9` | `M1025-4096` | 661 | `scheduler=default` | `lds_stage`: 1: 502, 0: 159; `tile`: 128x128x128: 62, 128x256x128: 57, 128x192x128: 51 (+39 more); `use_async_copy`: 4: 389, 0: 272; `use_cshuffle_epilog`: 2: 205, 0: 178, 1: 176 (+2 more); `waves_per_eu`: 2: 521, 1: 140 |
| `cfg_9636311ea3a3ce0d` | `M>4096` | 178 | `scheduler=default` | `lds_stage`: 1: 160, 0: 18; `tile`: 128x256x128: 96, 256x128x128: 27, 96x256x128: 12 (+13 more); `use_async_copy`: 4: 104, 0: 74; `use_cshuffle_epilog`: 2: 115, 0: 32, 1: 25 (+2 more); `waves_per_eu`: 2: 147, 1: 31 |

### `gfx950` · 256 CUs · `bf16_tuned_gemm` · FlyDSL `hgemm` · `abf16_wbf16_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_973fe63f897809df` | `M<=16` | 86 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 61, false: 25; `split_k`: 8: 34, 16: 29, 4: 19 (+2 more); `tile`: 32x64x128: 25, 32x64x64: 24, 32x128x128: 14 (+10 more); `warps`: 2x2: 61, 1x4: 25 |
| `cfg_ac0c03746ab634f6` | `M17-64` | 45 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 35, false: 10; `split_k`: 4: 14, 8: 14, 16: 7 (+2 more); `tile`: 32x64x128: 23, 64x64x128: 6, 32x64x64: 5 (+5 more); `warps`: 2x2: 35, 1x4: 10 |
| `cfg_147eed9b10acbc4e` | `M65-256` | 52 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `stages=2` | `b_to_lds`: true: 36, false: 16; `split_k`: 8: 25, 1: 17, 4: 6 (+2 more); `tile`: 32x64x128: 16, 128x128x128: 14, 64x64x128: 9 (+5 more); `warps`: 2x2: 36, 1x4: 16 |
| `cfg_ae017b25b6473300` | `M257-1024` | 29 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `split_k=1`, `stages=2` | `b_to_lds`: true: 21, false: 8; `tile`: 96x64x128: 9, 96x128x64: 9, 128x128x128: 8 (+2 more); `warps`: 2x2: 21, 1x4: 8 |
| `cfg_6660b64132bd9e80` | `M1025-4096` | 4 | `async_copy=true`, `b_preshuffle=false`, `b_to_lds=true`, `c_to_lds=false`, `split_k=1`, `stages=2`, `warps=2x2` | `tile`: 96x128x64: 2, 96x64x128: 1, 128x64x64: 1 |
| `cfg_94529fdef02c8a2f` | `M>4096` | 3 | `async_copy=true`, `b_preshuffle=false`, `c_to_lds=false`, `split_k=1`, `stages=2` | `b_to_lds`: true: 2, false: 1; `tile`: 64x64x128: 1, 128x128x128: 1, 128x64x64: 1; `warps`: 2x2: 2, 1x4: 1 |

### `gfx950` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `abf16_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_510d2d6168bb3053` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage1.xcd=1`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_684e266cdb2db851` | `M<=16` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x32x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `xbf16=0` | `stage1.w`: 2: 1, absent: 1; `stage1.xcd`: 1: 1, absent: 1; `stage2.tile`: 16x128x256: 1, 16x256x256: 1 |
| `cfg_5ec580cb72b4102f` | `M17-64` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.xcd`: 1: 1, 4: 1, absent: 1; `stage2.bnt`: 2: 1, absent: 2; `stage2.xcd`: 4: 1, absent: 2 |
| `cfg_9fa479221970ab55` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 4: 1, 2: 1; `stage1.tile`: 16x32x256: 1, 16x64x256: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 1: 1, 4: 1; `stage2.tile`: 16x256x256: 1, 16x128x256: 1 |
| `cfg_618be9e3ca35d63b` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.xcd`: 1: 1, 4: 1; `stage2.bnt`: 2: 1, absent: 1 |
| `cfg_93ffcfcdcd999f95` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=2`, `stage1.tile=16x64x256`, `stage1.xcd=1`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.w`: 2: 1, 3: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 16x256x256: 1, 16x128x256: 1 |
| `cfg_1e2f82daa1ce45a1` | `M257-1024` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_cf547aaf3befe6ca` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.xcd=1`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | `stage2.xcd`: 4: 1, absent: 1 |
| `cfg_65379c14d19aeec6` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=1`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 16x192x128: 1, 16x32x128: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 16x256x256: 1, 16x256x128: 1 |
| `cfg_15488429319daa13` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tile`: 16x128x256: 1, 16x128x128: 1; `stage2.xcd`: 4: 1, absent: 1 |
| `cfg_eb5f5b5fe34206a8` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.xcd`: 4: 1, 1: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.xcd`: 4: 1, absent: 1 |
| `cfg_8a7b815e594f1cc8` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage1.w=2`, `stage1.xcd=1`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_ae27ffd3996fd592` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `block_m`: 32: 5, 16: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 32x192x128: 4, 16x128x128: 1, 32x128x128: 1; `stage1.xcd`: 1: 4, 4: 2; `stage2.bnt`: 2: 2, absent: 4; `stage2.tile`: 32x256x128: 4, 16x128x128: 1, 32x128x128: 1 |
| `cfg_7da9516444acb48d` | `M>4096` | 6 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 32x256x128: 4, 32x128x128: 2; `stage1.xcd`: 1: 4, 4: 2; `stage2.bnt`: 2: 2, absent: 4; `stage2.tile`: 32x256x128: 4, 32x128x128: 2 |

### `gfx950` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `afp4_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_472176da724cdb51` | `M<=16` | 2 | `block_m=64`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=64x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=atomic`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage2.tile`: 64x128x256: 1, 64x256x128: 1 |
| `cfg_187dd16f8142b62a` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_10c1c7889021588b` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 4: 1, 2: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_2ba2274e7bdb4064` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 4: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1 |
| `cfg_65dc5c20c775a570` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_a6051cd53d9d181e` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_1f77e1917e4d32b6` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_29bca6aa99017f98` | `M<=16` | 4 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `xbf16=0` | `stage1.bnt`: 0: 2, absent: 2; `stage1.kw`: 4: 2, 2: 1, absent: 1; `stage1.tile`: 32x32x256: 3, 32x64x256: 1; `stage1.w`: 3: 2, 4: 1, 2: 1; `stage2.family`: moe_stage2_layout: 3, moe_stage2: 1; `stage2.sbm`: 32: 3, absent: 1; `stage2.tags`: reduce: 3, atomic,persist: 1; `stage2.tile`: 32x128x256: 3, 32x256x256: 1 |
| `cfg_67e4a128addd2251` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 2, absent: 1; `stage1.kw`: 4: 2, absent: 1; `stage1.tile`: 32x32x256: 1, 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 2, 3: 1; `stage1.xcd`: 4: 1, absent: 2; `stage2.bnt`: 2: 1, absent: 2; `stage2.family`: moe_stage2_layout: 2, moe_stage2: 1; `stage2.sbm`: 32: 2, absent: 1; `stage2.tags`: reduce: 2, atomic,nt: 1 |
| `cfg_bb1a7d0dbb31e502` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 4: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tags`: reduce: 1, nt,reduce: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_5f5fb1d3a590f3f2` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.bnt`: 0: 2, absent: 1; `stage1.w`: 4: 2, 3: 1; `stage2.tags`: atomic,nt: 2, atomic: 1; `stage2.tile`: 32x128x128: 2, 32x256x128: 1 |
| `cfg_1c807a6d6f7e058e` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_58db8a1bd4f97080` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_06449fb79dd966ae` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_c82d2d1359de6e4e` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_960b16993a0b742c` | `M17-64` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `xbf16=0` | `block_m`: 64: 1, 32: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 64x128x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 64: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 64x256x128: 1, 32x128x256: 1 |
| `cfg_498c7c699bf33482` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=2`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x128x128: 1, absent: 1 |
| `cfg_2aa324b752fbb6e8` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_b7bdd91aefb66ea9` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_da219c734afd225a` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 4: 1, 2: 1; `stage1.w`: 3: 1, 2: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_b9fef5829aee49d2` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x32x256: 1; `stage2.tags`: atomic,nt: 1, persist,reduce: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_51d5ee4329db82ae` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 2: 1, absent: 1; `stage2.tags`: atomic: 1, nt,reduce: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_475b436c4dec216a` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1; `stage2.tags`: atomic: 1, reduce: 1 |
| `cfg_e905d687f182daf5` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_ee8dc75774bd0d43` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_587bee1948dbbadb` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=nt,reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_cca48b8043c4fdd8` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x32x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tags`: reduce: 1, atomic: 1; `stage2.tile`: 32x256x128: 1, 32x128x256: 1 |
| `cfg_abf1e2cdd8583f0a` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_23574c2208ddbb88` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_cff530f26fedf80c` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_80188b631574ee49` | `M65-256` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_121d008bc7320993` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_4f10e35349bd353e` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 2: 1, 4: 1; `stage2.bnt`: 2: 1, absent: 1 |
| `cfg_4eae696379d48402` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.family`: moe_stage2: 1, moe_stage2_layout: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_8db6208a1b96845a` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_21fd0edb0cc57952` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x32x256: 1, 32x64x256: 1; `stage1.w`: 3: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: reduce: 1, nt,reduce: 1 |
| `cfg_dca1635f29cce920` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1 |
| `cfg_dcd3822333fa492d` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1 |
| `cfg_c3460cd0a9d4ab7f` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=nt,reduce`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_10670d2722600a07` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.tile=32x128x256`, `xbf16=0` | `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_c6107dc0c3f4d458` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_33d6a64cf5221598` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x32x256: 1; `stage1.w`: 2: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1 |
| `cfg_e4523acc0997ddfc` | `M65-256` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,persist`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_f5615a47c00cc961` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_62297e76e8d00785` | `M65-256` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_fbc7c246007b7854` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_db2ca9ab3e580fd2` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1; `stage2.tile`: 32x256x128: 1, 32x128x256: 1 |
| `cfg_e65cd526bfd29ca2` | `M257-1024` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tile=32x128x256`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tags`: atomic: 1, atomic,persist: 1 |
| `cfg_b181398945b7c492` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1 |
| `cfg_5cc784d77d772606` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_613b71556adafcdc` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.tile`: 32x128x256: 1, 32x32x256: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 1, 32x128x256: 1 |
| `cfg_1b81136f31bf269e` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x128x256: 1, 32x32x256: 1; `stage1.w`: 2: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: nt,reduce: 1, atomic,nt: 1 |
| `cfg_fd253ab3fd8de6ae` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_279446275f445341` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 4: 1, absent: 1; `stage1.tile`: 32x32x256: 1, 32x128x256: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1 |
| `cfg_b1a3609e36832cde` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage2.tags`: nt,reduce: 1, atomic,nt: 1 |
| `cfg_6e3ddb551ddac5b0` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage2.tags`: nt,reduce: 1, atomic,nt: 1 |
| `cfg_5b3738a8a255fde5` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `xbf16=0` | `stage1.kw`: 2: 1, 4: 1; `stage1.tile`: 32x64x256: 1, 32x32x256: 1; `stage1.w`: 3: 1, 2: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_0b70d513f7f9d80a` | `M257-1024` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,persist`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_c62d85016a93add6` | `M257-1024` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_e4d4309ad29650c1` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `xbf16=0` | `stage1.kw`: 2: 1, 4: 1; `stage1.tile`: 32x64x256: 1, 32x32x256: 1; `stage1.w`: 3: 1, 2: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_d57c6036373d2bfa` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_518125ad6c0cc3b5` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_a99cb419719f5fb7` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 64x128x256: 1; `stage1.w`: 3: 1, 4: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tags`: atomic,nt: 1, atomic,persist: 1; `stage2.tile`: 32x128x256: 1, 64x256x256: 1 |
| `cfg_5ec49145c29b4f80` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=3`, `stage2.tags=atomic`, `stage2.tile=64x128x256`, `xbf16=0` | `block_m`: 128: 1, 64: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 128x128x256: 1, 64x128x256: 1; `stage2.family`: moe_stage2: 1, moe_stage2_layout: 1; `stage2.sbm`: 128: 1, 64: 1 |
| `cfg_b703ec3e5938ae7f` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tile=64x128x256`, `xbf16=0` | `block_m`: 64: 1, 128: 1; `stage1.tile`: 64x64x256: 1, 128x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.sbm`: 64: 1, 128: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_cd6cf220e0759352` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 64x128x256: 1 |
| `cfg_348802bce6dbcbf4` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_604263407d150919` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.tile`: 32x32x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tile`: 32x128x256: 1, 32x256x128: 1 |
| `cfg_26d911f093c89ce0` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_b900060d58f67ffb` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tags=nt,reduce`, `xbf16=0` | `block_m`: 64: 1, 128: 1; `stage1.tile`: 64x128x256: 1, 128x128x256: 1; `stage1.w`: 3: 1, absent: 1; `stage2.sbm`: 64: 1, 128: 1; `stage2.tile`: 64x256x256: 1, 128x256x256: 1 |
| `cfg_db333a6e56dfa8ef` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=64x128x256`, `xbf16=0` | `block_m`: 64: 1, 128: 1; `stage1.tile`: 64x128x256: 1, 128x128x256: 1; `stage1.w`: 2: 1, absent: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.sbm`: 128: 1, absent: 1 |
| `cfg_add5d4cdc14f74a4` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt,persist: 1, atomic: 1; `stage2.tile`: 32x128x256: 1, 64x128x256: 1 |
| `cfg_8f965c8c34d5ef43` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_ffe9c406a14d85cc` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_ba658b922de41ca1` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_eae7aafd5a838dfb` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 64x64x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 64x256x128: 1 |
| `cfg_039e19576885eb1d` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_3b777da4a2cfddae` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_7f23c1e419329f16` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_64a9d128b5ef14c6` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_459288b92e3479ab` | `M>4096` | 8 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 4, 64: 3, 32: 1; `stage1.bnt`: 0: 7, absent: 1; `stage1.tile`: 128x128x256: 4, 64x128x256: 3, 32x128x256: 1; `stage1.w`: 3: 7, 4: 1; `stage2.sbm`: 128: 4, 64: 3, 32: 1; `stage2.tags`: atomic,nt: 4, atomic,persist: 2, scatter: 2; `stage2.tile`: 128x256x256: 4, 64x256x128: 3, 32x256x256: 1 |
| `cfg_ed3954d3def9fff2` | `M>4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_78488b6f5ebf508b` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tile=64x128x256`, `xbf16=0` | `block_m`: 64: 5, 128: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x256: 5, 128x128x256: 1; `stage1.w`: 4: 3, 3: 3; `stage2.sbm`: 64: 5, 128: 1; `stage2.tags`: atomic,persist: 4, atomic,nt: 1, atomic: 1 |
| `cfg_82afd9d83a7f4652` | `M>4096` | 6 | `block_m=64`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=64x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=atomic`, `xbf16=0` | `stage1.w`: 4: 3, 2: 2, 3: 1; `stage2.tile`: 64x128x256: 5, 64x256x128: 1 |
| `cfg_47efedab482a4ebf` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic`, `xbf16=0` | `block_m`: 64: 5, 128: 1; `stage1.tile`: 64x128x256: 5, 128x128x256: 1; `stage1.w`: 3: 5, 2: 1; `stage2.sbm`: 64: 5, 128: 1; `stage2.tile`: 64x128x256: 4, 64x256x128: 2 |
| `cfg_1f64545f332644ee` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic`, `xbf16=0` | `block_m`: 64: 5, 128: 1; `stage1.bnt`: 0: 5, absent: 1; `stage1.tile`: 64x128x256: 5, 128x128x256: 1; `stage1.w`: 3: 4, 4: 2; `stage1.xcd`: 4: 1, absent: 5; `stage2.sbm`: 64: 5, 128: 1; `stage2.tile`: 64x128x256: 4, 64x256x128: 2 |
| `cfg_c1a47cfc000cf544` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 5, 32: 2; `stage1.bnt`: 0: 4, absent: 3; `stage1.tile`: 64x128x256: 4, 32x32x256: 2, 64x64x256: 1; `stage1.w`: 4: 5, 3: 1, 2: 1; `stage1.xcd`: 4: 4, absent: 3; `stage2.sbm`: 64: 5, 32: 2; `stage2.tags`: persist,reduce: 3, atomic,nt: 2, nt,persist,reduce: 2; `stage2.tile`: 64x128x256: 5, 32x128x128: 2 |
| `cfg_54bbb206eb4f771e` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 5, 32: 2; `stage1.bnt`: 0: 4, absent: 3; `stage1.kw`: 2: 1, absent: 6; `stage1.tile`: 64x128x256: 3, 32x64x256: 2, 64x64x256: 2; `stage1.w`: 3: 4, 4: 2, 2: 1; `stage1.xcd`: 4: 1, absent: 6; `stage2.sbm`: 64: 5, 32: 2; `stage2.tags`: persist,reduce: 4, atomic,nt: 3; `stage2.tile`: 64x128x256: 4, 32x256x256: 1, 32x128x256: 1 (+1 more) |
| `cfg_390787160a0a0bb2` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 32: 2, 128: 1; `stage1.bnt`: 0: 3, absent: 4; `stage1.kw`: 2: 1, absent: 6; `stage1.tile`: 32x64x256: 2, 64x64x256: 2, 64x128x256: 2 (+1 more); `stage1.w`: 3: 3, 4: 3, 2: 1; `stage2.sbm`: 64: 4, 32: 2, 128: 1; `stage2.tags`: persist,reduce: 4, atomic,nt: 2, atomic,nt,persist: 1; `stage2.tile`: 64x128x256: 5, 32x128x256: 2 |
| `cfg_1e4415f3522e32ff` | `M>4096` | 4 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 1; `stage1.tile`: 64x128x256: 3, 128x128x256: 1; `stage1.w`: 2: 1, 4: 1, 3: 1 (+1 more); `stage1.xcd`: 4: 3, absent: 1; `stage2.sbm`: 64: 3, 128: 1; `stage2.tags`: atomic: 2, reduce: 1, atomic,persist: 1; `stage2.tile`: 64x256x128: 2, 128x256x256: 1, 64x256x256: 1 |
| `cfg_a6f7503fc1177d97` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=4`, `xbf16=0` | `block_m`: 64: 4, 128: 1; `stage1.bnt`: 0: 4, absent: 1; `stage1.fp`: 4: 4, absent: 1; `stage1.tile`: 64x128x256: 4, 128x128x256: 1; `stage1.w`: 2: 3, 4: 1, 3: 1; `stage2.bnt`: 2: 4, absent: 1; `stage2.family`: moe_stage2: 4, moe_stage2_layout: 1; `stage2.sbm`: 128: 1, 64: 1, absent: 3; `stage2.tags`: atomic: 4, atomic,persist: 1; `stage2.tile`: 64x128x256: 4, 64x256x256: 1 |
| `cfg_e6bac9cab25f52d9` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `xbf16=0` | `block_m`: 128: 4, 64: 2; `stage1.bnt`: 0: 5, absent: 1; `stage1.tile`: 128x128x256: 4, 64x128x256: 2; `stage1.w`: 4: 2, 3: 1, 2: 1 (+1 more); `stage1.xcd`: 4: 3, absent: 3; `stage2.bnt`: 2: 1, absent: 5; `stage2.family`: moe_stage2_layout: 4, moe_stage2: 2; `stage2.sbm`: 128: 2, 64: 2, absent: 2; `stage2.tags`: atomic: 3, atomic,persist: 2, reduce: 1; `stage2.tile`: 128x128x256: 2, 128x256x128: 1, 64x128x256: 1 (+2 more) |
| `cfg_5ead3431822ed923` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x128x256: 3, 128x128x256: 2; `stage1.w`: 2: 3, 4: 2; `stage1.xcd`: 4: 3, absent: 2; `stage2.sbm`: 64: 3, 128: 2; `stage2.tags`: reduce: 2, atomic,nt: 1, nt,reduce: 1 (+1 more); `stage2.tile`: 64x256x128: 3, 64x128x128: 1, 128x128x128: 1 |
| `cfg_ae76ae55dba9d86d` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 32: 1, 128: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.kw`: 2: 1, absent: 5; `stage1.tile`: 64x128x256: 4, 32x64x256: 1, 128x128x256: 1; `stage1.w`: 2: 4, 4: 2; `stage2.sbm`: 64: 4, 32: 1, 128: 1; `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1, atomic,persist: 1 (+3 more); `stage2.tile`: 64x256x128: 5, 32x256x128: 1 |
| `cfg_fca852ebeb38b5f7` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 5, 128: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x256: 5, 128x128x256: 1; `stage1.w`: 2: 4, 3: 2; `stage1.xcd`: 4: 4, absent: 2; `stage2.sbm`: 64: 5, 128: 1; `stage2.tags`: persist,reduce: 2, atomic,nt: 1, atomic,persist: 1 (+2 more); `stage2.tile`: 64x128x256: 4, 64x256x128: 2 |
| `cfg_e324eebfa22e362e` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.tile`: 64x128x256: 3, 128x128x256: 2, 32x128x256: 1; `stage1.w`: 4: 2, 3: 2, 2: 1 (+1 more); `stage2.sbm`: 64: 3, 128: 2, 32: 1; `stage2.tags`: atomic,nt: 2, atomic,persist: 2, atomic,nt,persist: 1 (+1 more); `stage2.tile`: 64x256x128: 3, 32x128x128: 1, 128x256x256: 1 (+1 more) |
| `cfg_555872067e503a14` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 5, 32: 1, 128: 1; `stage1.bnt`: 0: 5, absent: 2; `stage1.kw`: 2: 1, absent: 6; `stage1.tile`: 64x128x256: 4, 32x64x256: 1, 64x64x256: 1 (+1 more); `stage1.w`: 3: 3, 4: 2, 2: 2; `stage1.xcd`: 4: 4, absent: 3; `stage2.sbm`: 64: 5, 32: 1, 128: 1; `stage2.tags`: reduce: 3, atomic,nt: 1, nt,reduce: 1 (+2 more); `stage2.tile`: 64x256x128: 3, 32x128x256: 1, 64x128x256: 1 (+2 more) |
| `cfg_4c978670631100a9` | `M>4096` | 1 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=128x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=nt,reduce`, `stage2.tile=128x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_5a71f559b83974b6` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 32: 1, 128: 1; `stage1.bnt`: 0: 2, absent: 3; `stage1.tile`: 64x128x256: 2, 32x64x256: 1, 64x64x256: 1 (+1 more); `stage1.w`: 3: 3, 2: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 4; `stage2.sbm`: 64: 3, 32: 1, 128: 1; `stage2.tags`: atomic,nt: 2, persist,reduce: 2, reduce: 1; `stage2.tile`: 64x256x128: 2, 32x256x128: 1, 64x128x128: 1 (+1 more) |
| `cfg_0f851dcc549fe026` | `M>4096` | 1 | `block_m=64`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=64x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=atomic,nt`, `stage2.tile=64x128x256`, `xbf16=0` | one row; nothing to compare |

### `gfx950` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `afp8_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_3403d72a55d6439c` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 2; `stage1.tile`: 32x128x256: 2, 32x64x256: 1; `stage1.w`: 3: 2, absent: 1 |
| `cfg_2a52ec5afc978fc9` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_a05747932c5c2ec7` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_3052de8752b74f06` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 2: 1, 3: 1; `stage2.tags`: nt,reduce: 1, reduce: 1 |
| `cfg_3747c04e0eaa344f` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,nt: 1, reduce: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_3f14ed44f3e17a8d` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 4: 2, 2: 1; `stage1.w`: 3: 2, 4: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1, reduce: 1; `stage2.tile`: 32x128x128: 2, 32x256x128: 1 |
| `cfg_98ac04a5d1c3f005` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.bnt`: 0: 2, absent: 1; `stage1.kw`: 4: 1, 2: 1, absent: 1; `stage1.w`: 3: 2, 4: 1; `stage2.tags`: reduce: 2, atomic,nt: 1; `stage2.tile`: 32x128x128: 2, 32x256x128: 1 |
| `cfg_901a9d92c09e1386` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.kw`: 4: 1, 2: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1 |
| `cfg_27ce4a8e5cb245ba` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 4: 1, 2: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic,nt: 1, reduce: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_41321ea1fa31a370` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=nt,reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_96d0a9e390626206` | `M<=16` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 2; `stage1.w`: 3: 1, 4: 1, 2: 1; `stage2.tags`: atomic: 2, atomic,nt: 1 |
| `cfg_708b25959b5be0b9` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_e9f5e1abe27b2a36` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, 3: 1 |
| `cfg_afcef0e668e857d6` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x128x256: 1, 32x64x256: 1 |
| `cfg_7f24fe3add8a8efd` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: persist,reduce: 1, reduce: 1; `stage2.tile`: 16x128x256: 1, 32x128x256: 1 |
| `cfg_81010fb89232f4c3` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1 |
| `cfg_744c9f43ec834350` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.w`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x128x128: 1, absent: 1 |
| `cfg_0d18101f069788c4` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_5d57c2723ae38529` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=4`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_5c2111a5f3ef8f5c` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_5db8dc282ca1c183` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_96c168ada10b1024` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: reduce: 1, atomic,nt: 1 |
| `cfg_63ae009956ca91f8` | `M17-64` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 2, absent: 1; `stage1.w`: 3: 2, 2: 1; `stage1.xcd`: 4: 1, absent: 2; `stage2.tags`: atomic: 1, nt,reduce: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 2, 32x256x128: 1 |
| `cfg_6addc94166f7698c` | `M17-64` | 3 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 2, absent: 1; `stage1.w`: 3: 2, 4: 1; `stage2.tags`: atomic,nt: 2, nt,reduce: 1; `stage2.tile`: 32x256x256: 2, 32x128x128: 1 |
| `cfg_21b8148dd9b55a1c` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.tags`: nt,reduce: 1, reduce: 1 |
| `cfg_6e67613d6001815a` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 4: 1, 2: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: nt,reduce: 1, reduce: 1 |
| `cfg_97b76c9da0c2a13d` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x128`, `xbf16=0` | `stage2.tags`: reduce: 1, atomic,nt: 1 |
| `cfg_f5479c3590e5c952` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 3: 1, 4: 1; `stage2.tags`: reduce: 1, atomic,nt: 1 |
| `cfg_30b4d6b64f7a595d` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 4: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: nt,reduce: 1, atomic: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_c9fabef20c8e9efc` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 3: 1, 2: 1; `stage2.tags`: nt,reduce: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 1, 32x256x128: 1 |
| `cfg_48e65ff193052aff` | `M17-64` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_8b8f1c589ea917db` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, absent: 1; `stage2.tags`: nt,reduce: 1, reduce: 1 |
| `cfg_e140698db62888f9` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 2: 1 |
| `cfg_37b9cf990420ec04` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_75674065e94ac4c2` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 4: 1, 2: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_0786faff3fad5a81` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=persist,reduce`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_07b07c66ac886282` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=nt,reduce`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_6ed7632235ebe16b` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_6db88ee1558b0fdb` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tags`: reduce: 1, nt,reduce: 1; `stage2.tile`: 32x128x128: 1, 32x256x128: 1 |
| `cfg_2acd667af7458d04` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_9a4e2c30f5d3413f` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.w`: 2: 1, absent: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_154dd05025ee8965` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1; `stage2.tile`: 32x256x128: 1, 32x128x256: 1 |
| `cfg_bab13566dfed676b` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1 |
| `cfg_e325ba2a48b68993` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.tags`: reduce: 1, nt,reduce: 1; `stage2.tile`: 32x128x128: 1, 32x128x256: 1 |
| `cfg_cf43991244eba0ba` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_d9e032d16c8964fb` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: nt,reduce: 1, absent: 1; `stage2.tile`: 32x128x128: 1, absent: 1 |
| `cfg_2fd21a46e3dab128` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `xbf16=0` | `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: nt,reduce: 1, absent: 1; `stage2.tile`: 32x128x128: 1, absent: 1 |
| `cfg_846167f1d3eb94b8` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x256x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1 |
| `cfg_e88f5055f2e7b209` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=16x128x256`, `stage1.w=3`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `xbf16=0` | `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_48f691d02b3ecce8` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_17e3bb4e11437f8e` | `M65-256` | 2 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.w=3`, `xbf16=0` | `block_m`: 32: 1, 16: 1; `stage1.fp`: 8: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 16x128x256: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic,persist: 1; `stage2.tile`: 32x256x128: 1, 16x128x256: 1 |
| `cfg_dba66608f81a2a95` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 3: 1 |
| `cfg_5629b39ef138544c` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_c0e8959694f596dd` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.w`: 3: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1 |
| `cfg_f4e8b2df18f77a1a` | `M65-256` | 2 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 32: 1, 16: 1; `stage1.fp`: 8: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 16x128x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1; `stage2.tile`: 32x256x128: 1, 16x128x128: 1 |
| `cfg_4a63f029476c6b78` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_615b2a0b4e42083d` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_04909b3820628779` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_2e3d088300fa6b1b` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage2.tags`: nt,reduce: 1, atomic,nt: 1 |
| `cfg_6b69dd1c75fd51c4` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=2`, `xbf16=0` | one row; nothing to compare |
| `cfg_e927dd4d9317af15` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_679ff95bc157f45a` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage2.tags`: nt,reduce: 1, atomic,nt: 1; `stage2.tile`: 32x256x128: 1, 32x256x256: 1 |
| `cfg_76b5c8dbc1dcffb7` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.w`: 2: 1, 3: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_4c859376081f08dd` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_0df9e297d1084942` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1 |
| `cfg_7082c06639587211` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `xbf16=0` | one row; nothing to compare |
| `cfg_884d48750208af22` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_ebbb9a872c73ce25` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_f71f8631a0ad39b0` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_e0b95cad6dba1acc` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x256`, `xbf16=0` | `stage2.tags`: reduce: 1, nt,reduce: 1 |
| `cfg_f2fd02d54de6a092` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 2: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2: 1, moe_stage2_layout: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 1, 32x256x128: 1 |
| `cfg_5fd32ffdd338e82e` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_3307ebe76ca5b6ef` | `M257-1024` | 2 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 1, 16: 1; `stage1.tile`: 32x128x256: 1, 16x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 32x128x128: 1, 16x128x128: 1 |
| `cfg_3596c034c6279dfb` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_55540dea1f50a625` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_d3d061b4c4a50ed7` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 3: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, moe_stage2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1; `stage2.tile`: 32x256x128: 1, 32x128x128: 1 |
| `cfg_27b13f304b746145` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 16x128x256: 1, 16x128x128: 1 |
| `cfg_148890abb3e24145` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_08e4306012bd83a7` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1 |
| `cfg_070b720cf0c13642` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_abf78f61ced752db` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_ad37430106bccbd4` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1 |
| `cfg_6cad5fface666968` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_2638de4a15955c8a` | `M257-1024` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `flat`: 0: 1, 49.07: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage1.w`: 3: 1, 2: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_737f0c0ac145298a` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_c1f5be99dc2ab2a1` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.tags`: nt,reduce: 1, atomic,nt: 1; `stage2.tile`: 32x256x128: 1, 32x256x256: 1 |
| `cfg_c1d7593d922bbd56` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_c964c1bc9ae1d0fb` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1; `stage1.w`: 2: 1, absent: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tile`: 32x256x256: 1, 64x256x256: 1 |
| `cfg_6c94aeb53763bc0d` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `xbf16=0` | `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x256x128: 1, absent: 1 |
| `cfg_891da464e4287b61` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `xbf16=0` | one row; nothing to compare |
| `cfg_6f9b81cc840ad4de` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage2.tags`: atomic,nt: 1, atomic: 1; `stage2.tile`: 32x128x128: 1, 32x128x256: 1 |
| `cfg_62bb20f35695370c` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage2.tile`: 32x128x128: 1, 32x128x256: 1 |
| `cfg_9ddd77f6023d9f5f` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_8be1e6a1c444dc97` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_96adfe2c02f9616b` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_fb2b0de5387c991e` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_984b7caa9d9eeca9` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage2.tile`: 32x256x128: 1, 32x256x256: 1 |
| `cfg_890b5fa276f55529` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_b46dfe89f4294a34` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_9da9d2143318c0bf` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tile`: 32x128x256: 1, 64x256x256: 1 |
| `cfg_e202307cf34ba2bc` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tags`: reduce: 1, atomic,persist: 1; `stage2.tile`: 32x128x256: 1, 64x256x256: 1 |
| `cfg_cf5a4c3d3a27506c` | `M1025-4096` | 2 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1; `stage1.w`: 4: 1, 3: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 64: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 64x256x256: 1, absent: 1 |
| `cfg_b7b58e61872df8b7` | `M1025-4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `xbf16=0` | one row; nothing to compare |
| `cfg_4d98fca89c1ac8cf` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.w=4`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1 |
| `cfg_85c214e8e50266a8` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `xbf16=0` | `flat`: 0: 1, 198.32: 1; `stage1.w`: 3: 1, 4: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_dfb79e030d1ab2fb` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_27f01d580e1207a0` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_baa0cb19d3b94a9e` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_7cf2447757b986a2` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic`, `xbf16=0` | `block_m`: 128: 3, 64: 2; `stage1.tile`: 128x256x256: 3, 64x256x256: 2; `stage1.w`: 2: 4, absent: 1; `stage2.sbm`: 128: 3, 64: 2; `stage2.tile`: 64x256x256: 3, 64x128x256: 2 |
| `cfg_6e8f1ce4c9dbd1c5` | `M>4096` | 1 | `block_m=64`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=64x128x256`, `stage1.w=3`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=reduce`, `stage2.tile=64x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_7f21738874b385a2` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 64: 5, 32: 2; `stage1.bnt`: 0: 4, absent: 3; `stage1.tile`: 64x128x256: 5, 32x128x256: 2; `stage1.w`: 2: 5, 3: 1, 4: 1; `stage1.xcd`: 4: 4, absent: 3 |
| `cfg_426f74ee29512555` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 32: 2, 64: 2, 128: 1; `stage1.bnt`: 0: 2, absent: 3; `stage1.tile`: 32x128x256: 2, 64x128x256: 2, 128x256x256: 1; `stage1.w`: 3: 3, 2: 2; `stage2.family`: moe_stage2_layout: 3, absent: 2; `stage2.sbm`: 32: 2, 128: 1, absent: 2; `stage2.tags`: atomic,nt: 2, reduce: 1, absent: 2; `stage2.tile`: 32x128x128: 2, 64x256x256: 1, absent: 2 |
| `cfg_faf3a246efe620f4` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 3, 64: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.fp`: 8: 5, absent: 1; `stage1.tile`: 128x256x256: 3, 32x128x256: 1, 64x128x256: 1 (+1 more); `stage1.w`: 3: 2, 4: 2, 2: 1 (+1 more); `stage2.sbm`: 128: 3, 64: 2, 32: 1; `stage2.tags`: atomic: 2, atomic,nt: 2, nt,reduce: 2; `stage2.tile`: 64x256x256: 2, 64x256x128: 2, 32x128x256: 1 (+1 more) |
| `cfg_be57f9efe5b2a87e` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 32: 2, 128: 1; `stage1.bnt`: 0: 2, absent: 5; `stage1.tile`: 64x256x256: 4, 32x128x256: 2, 128x256x256: 1; `stage1.w`: 3: 1, 4: 1, absent: 5; `stage1.xcd`: 4: 6, absent: 1; `stage2.sbm`: 64: 4, 32: 2, 128: 1; `stage2.tags`: reduce: 6, nt,reduce: 1; `stage2.tile`: 64x256x256: 5, 32x256x256: 2 |
| `cfg_2b5afc180252cae4` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x256x256: 2, 128x128x256: 1, 128x256x256: 1 (+1 more); `stage1.w`: 4: 2, 2: 2, 3: 1; `stage1.xcd`: 4: 1, absent: 4; `stage2.sbm`: 64: 3, 128: 2; `stage2.tags`: reduce: 2, atomic,nt: 1, nt,reduce: 1 (+1 more); `stage2.tile`: 64x256x128: 4, 128x256x128: 1 |
| `cfg_1c93c741705d9c20` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 128: 3, 64: 2, 32: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.fp`: 8: 5, absent: 1; `stage1.tile`: 128x128x256: 3, 64x128x256: 2, 32x128x256: 1; `stage1.w`: 3: 4, 4: 1, absent: 1; `stage2.family`: moe_stage2_layout: 5, moe_stage2: 1; `stage2.sbm`: 128: 3, 32: 1, 64: 1 (+1 more); `stage2.tags`: atomic: 4, atomic,nt: 2; `stage2.tile`: 64x256x128: 3, 64x128x128: 2, 32x256x128: 1 |
| `cfg_dbb8b39f1a4d3a93` | `M>4096` | 3 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt`, `xbf16=0` | `block_m`: 32: 1, 64: 1, 128: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1, 128x256x256: 1; `stage1.w`: 4: 1, 3: 1, absent: 1; `stage2.sbm`: 32: 1, 64: 1, 128: 1; `stage2.tile`: 32x256x128: 1, 64x256x128: 1, 128x256x128: 1 |
| `cfg_114274af36ed5e1a` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tile=64x256x128`, `xbf16=0` | `block_m`: 64: 3, 128: 2; `stage1.bnt`: 0: 4, absent: 1; `stage1.tile`: 64x128x256: 2, 128x256x256: 2, 64x256x256: 1; `stage1.w`: 4: 2, 3: 2, 2: 1; `stage2.sbm`: 64: 3, 128: 2; `stage2.tags`: atomic: 4, atomic,nt: 1 |
| `cfg_1497518479add862` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 128: 1; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x256x256: 2, 64x128x256: 2, 128x256x256: 1; `stage1.w`: 3: 2, 2: 2, 4: 1; `stage1.xcd`: 4: 2, absent: 3; `stage2.sbm`: 64: 4, 128: 1; `stage2.tags`: reduce: 4, nt,reduce: 1; `stage2.tile`: 64x256x128: 4, 128x256x128: 1 |
| `cfg_dd9e3d216f5fe121` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `nt=0`, `run_1stage=0`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 64: 3, 128: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.fp`: 8: 5, absent: 1; `stage1.tile`: 64x256x256: 2, 32x128x256: 1, 64x128x256: 1 (+2 more); `stage1.w`: 2: 3, 4: 2, 3: 1; `stage1.xcd`: 4: 1, absent: 5; `stage2.family`: moe_stage2_layout: 5, moe_stage2: 1; `stage2.sbm`: 64: 2, 128: 2, 32: 1 (+1 more); `stage2.tags`: reduce: 3, atomic,nt: 1, atomic: 1 (+1 more); `stage2.tile`: 64x256x128: 4, 32x256x128: 1, 64x128x128: 1 |
| `cfg_464c7da6b08d319b` | `M>4096` | 7 | `block_m=64`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `stage1.bnt`: 0: 6, absent: 1; `stage1.tile`: 64x256x256: 6, 64x128x256: 1; `stage1.w`: 2: 3, 4: 3, 3: 1; `stage1.xcd`: 4: 6, absent: 1 |
| `cfg_b3090db820f16494` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 128: 2; `stage1.bnt`: 0: 2, absent: 4; `stage1.tile`: 64x128x256: 2, 64x256x256: 2, 128x256x256: 2; `stage1.w`: 4: 3, 3: 2, 2: 1; `stage2.sbm`: 64: 4, 128: 2; `stage2.tags`: atomic: 3, atomic,nt: 2, nt,reduce: 1; `stage2.tile`: 64x128x256: 3, 64x256x128: 2, 128x128x256: 1 |
| `cfg_155f764e218823ae` | `M>4096` | 7 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2`, `stage2.sbm=128`, `xbf16=0` | `stage1.bnt`: 0: 4, absent: 3; `stage1.w`: 3: 2, absent: 5; `stage2.tags`: atomic,persist: 5, atomic: 2; `stage2.tile`: 64x256x256: 6, 64x128x256: 1; `stage2.xcd`: 4: 3, absent: 4 |
| `cfg_a7038c956e198811` | `M>4096` | 7 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=128x256x256`, `stage2.sbm=128`, `xbf16=0` | `stage1.bnt`: 0: 4, absent: 3; `stage1.w`: 3: 2, 2: 2, absent: 3; `stage2.family`: moe_stage2_layout: 5, moe_stage2: 2; `stage2.tags`: reduce: 3, atomic: 2, nt,reduce: 1 (+1 more); `stage2.tile`: 128x256x256: 4, 128x256x128: 1, 64x256x256: 1 (+1 more) |
| `cfg_39861a1a57f0836c` | `M>4096` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `xbf16=0` | one row; nothing to compare |
| `cfg_3b7c57e9162cca23` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `xbf16=0` | `block_m`: 64: 4, 32: 2; `flat`: 381.09: 1, 701.57: 1, 1164.49: 1 (+3 more); `stage1.bnt`: 0: 3, absent: 3; `stage1.tile`: 64x256x256: 3, 32x128x256: 2, 64x128x256: 1; `stage1.w`: 3: 2, 2: 2, 4: 2 |
| `cfg_3df6b68c3c5603e4` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.tile`: 128x256x256: 2, 64x256x256: 2, 32x128x256: 1 (+1 more); `stage1.w`: 3: 3, 4: 2, absent: 1; `stage1.xcd`: 4: 1, absent: 5; `stage2.sbm`: 64: 3, 128: 2, 32: 1; `stage2.tags`: reduce: 4, atomic,nt: 2; `stage2.tile`: 64x256x256: 4, 32x256x256: 1, 64x256x128: 1 |
| `cfg_192e79469164a768` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 3, 32: 2, 64: 1; `stage1.bnt`: 0: 2, absent: 4; `stage1.tile`: 128x256x256: 3, 32x128x256: 2, 64x256x256: 1; `stage1.w`: 3: 3, 4: 1, 2: 1 (+1 more); `stage2.sbm`: 128: 3, 32: 2, 64: 1; `stage2.tags`: atomic,nt: 3, reduce: 2, atomic: 1; `stage2.tile`: 64x256x128: 2, 32x256x128: 1, 32x128x128: 1 (+2 more) |
| `cfg_e20f2c5e67a2c42b` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 3, 64: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.tile`: 128x256x256: 3, 32x128x256: 1, 64x128x256: 1 (+1 more); `stage1.w`: 4: 4, 3: 1, absent: 1; `stage1.xcd`: 4: 3, absent: 3; `stage2.sbm`: 128: 3, 64: 2, 32: 1; `stage2.tags`: atomic: 3, atomic,nt: 2, reduce: 1; `stage2.tile`: 64x256x256: 2, 64x256x128: 2, 32x256x256: 1 (+1 more) |

### `gfx950` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `afp8_wfp8_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_17ef23285e81eb97` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_9eb40e615c096b52` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic,nt: 1, reduce: 1 |
| `cfg_e1b55e162ade504b` | `M<=16` | 1 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_cd8dd116e55bb90b` | `M<=16` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,nt: 1, reduce: 1; `stage2.tile`: 32x128x256: 1, 32x128x128: 1 |
| `cfg_c4f9add9515f5981` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 4: 1, absent: 1 |
| `cfg_d9616853798ea17a` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1 |
| `cfg_c86f6d0c599ae54c` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, absent: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_3ddbba3e4c3638fa` | `M17-64` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_acb3147b8d3458e2` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: reduce: 1, atomic,nt: 1 |
| `cfg_afb44fcd8860ea62` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_d5fba9b6435c54c4` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage2.tags`: reduce: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 1, 32x128x256: 1 |
| `cfg_e09df919e75a1d2e` | `M65-256` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic,nt: 1, nt,reduce: 1 |
| `cfg_fcb60923ad0f63de` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1 |
| `cfg_1e8bca4cbca49911` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.w=2`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage2.family`: moe_stage2: 1, moe_stage2_layout: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,persist: 1, atomic,nt: 1; `stage2.tile`: 32x128x128: 1, 32x256x128: 1 |
| `cfg_fe6136c1a7d658a7` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_bc8cf6aeff3c7c22` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_72725e3ce3d208ed` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 2: 1, 3: 1 |
| `cfg_715fc9ea9247914f` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_994da03b695e9032` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1 |
| `cfg_59a4681feb196ccf` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.kw=2`, `stage1.tags=gui`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_f851ab2244623ab6` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 4, 64: 2, 32: 1; `stage1.bnt`: 0: 4, absent: 3; `stage1.fp`: 8: 6, absent: 1; `stage1.tile`: 128x128x256: 4, 64x128x256: 2, 32x64x256: 1; `stage1.w`: 4: 2, 2: 2, 3: 1 (+1 more); `stage1.xcd`: 4: 4, absent: 3; `stage2.sbm`: 128: 4, 64: 2, 32: 1; `stage2.tags`: reduce: 4, atomic,nt: 2, nt,reduce: 1; `stage2.tile`: 128x256x256: 4, 32x128x128: 1, 64x128x256: 1 (+1 more) |
| `cfg_b8e6f32d3c554420` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.tile=64x128x128`, `xbf16=0` | `block_m`: 128: 4, 64: 2; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 128x256x256: 4, 64x128x256: 2; `stage1.w`: 2: 2, 4: 1, 3: 1 (+1 more); `stage1.xcd`: 4: 4, absent: 2; `stage2.bnt`: 2: 1, absent: 5; `stage2.family`: moe_stage2_layout: 3, moe_stage2: 3; `stage2.sbm`: 128: 4, 64: 1, absent: 1; `stage2.tags`: reduce: 3, atomic,nt: 1, atomic,persist: 1 (+1 more) |
| `cfg_0310a64de6c0ed31` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 3, 32: 1; `stage1.bnt`: 0: 4, absent: 3; `stage1.fp`: 8: 3, absent: 4; `stage1.kw`: 2: 1, absent: 6; `stage1.tile`: 64x128x256: 3, 128x256x256: 2, 32x64x256: 1 (+1 more); `stage1.w`: 4: 4, 2: 1, absent: 2; `stage2.sbm`: 64: 3, 128: 3, 32: 1; `stage2.tags`: atomic,nt: 5, atomic: 1, reduce: 1; `stage2.tile`: 64x128x128: 4, 32x128x128: 1, 128x128x128: 1 (+1 more) |
| `cfg_4f6dffa2bb366398` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 3; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x256: 3, 128x128x256: 2, 128x256x256: 1; `stage1.w`: 3: 3, 2: 2, 4: 1; `stage1.xcd`: 4: 1, absent: 5; `stage2.sbm`: 64: 3, 128: 3; `stage2.tags`: atomic: 4, atomic,nt: 2; `stage2.tile`: 64x128x128: 5, 128x128x128: 1 |

### `gfx950` · 256 CUs · `tuned_fmoe` · FlyDSL `mxmoe_stage1` · `a4w4`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_74ed3d2fd0ba941d` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.situv=2`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_d3a4e01a7e0f9354` | `M<=16` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage2.tags`: atomic,nt: 2, atomic: 1; `stage2.tile`: 16x128x128: 2, 16x256x128: 1 |
| `cfg_7256439c2029e9a5` | `M<=16` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_c9382b6fc3367e79` | `M<=16` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_e5becaab13e0085c` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_cee0788dc0ff9072` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_2d30be5ab8ac8d61` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_b09c2807c719ca22` | `M<=16` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_07fd27567258f660` | `M<=16` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x256`, `xbf16=0` | `stage1.xcd`: 2: 1, 4: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_a397174d69bdc17c` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.situv=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_b23eef55a16b396b` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic,persist: 1; `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_aa7ea5dcd1545625` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.xcd`: 2: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic,persist: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_44dfa9d19df66afb` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tile`: 16x128x256: 1, 16x256x256: 1 |
| `cfg_ec95406ebf9bca92` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1 |
| `cfg_92163c353aa549d1` | `M17-64` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_0e3587b6eded1e65` | `M17-64` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,persist`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_da3c10c1b64033c6` | `M17-64` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_6b2f5a8b24c2e048` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.xcd`: 4: 1, 2: 1; `stage2.tags`: atomic,persist: 1, atomic,nt: 1 |
| `cfg_636c245dfdb2e338` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x256`, `xbf16=0` | `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1 |
| `cfg_b1c334e23feaa9a0` | `M17-64` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1; `stage2.tile`: 16x128x256: 1, 16x256x128: 1 |
| `cfg_d4bae44a392bf643` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.situv=2`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.xcd`: 4: 1, absent: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_54303ba707fac769` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_ed687211fe4520c6` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_f2d1ed35cf8d6b1b` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `xbf16=0` | `stage1.tags`: f16in,nt: 1, f16in,hpf,nt: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_5c6ea8f679882de9` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tags`: atomic: 1, atomic,nt: 1 |
| `cfg_9d6ed0652cf4039f` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x256`, `xbf16=0` | `stage2.tags`: atomic: 1, atomic,nt: 1 |
| `cfg_3a2272a8d18f9667` | `M65-256` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt,persist`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_87387b8c0155e756` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_77bac9fc12f20dc5` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage1.tags`: f16in,nt,swiglu: 1, f16in,hpf,nt,swiglu: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 4: 1, 2: 1; `stage2.tags`: atomic: 1, atomic,persist: 1; `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_cbc4215d7b8ca8c3` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x256`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 2: 1, absent: 1 |
| `cfg_43a9953e6ed00080` | `M65-256` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_e17641900ff8c9fa` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.situv=2`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_0d60a673bf73f04e` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_b7ec579b5182f64d` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_0dee1ba424e50243` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.tags`: f16in,nt: 1, f16in,hpf,nt: 1; `stage1.xcd`: 2: 1, absent: 1 |
| `cfg_79771f26984be209` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage1.tags`: f16in,nt: 1, f16in,hpf,nt: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_4dfbdf0a00db8245` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tags`: f16in,nt: 1, f16in,hpf,nt: 1; `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_5510688ff79244c6` | `M257-1024` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_cad42080509365d6` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_f4d15196fb6173bc` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x256x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,persist`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_bb6a9355af4ace30` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `xbf16=0` | `stage2.tile`: 16x128x256: 1, 16x128x128: 1 |
| `cfg_2a014859b7f84cd0` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x256x256: 1; `stage1.w`: 2: 1, absent: 1; `stage1.xcd`: 4: 1, 2: 1 |
| `cfg_b57f3272aacdf2b1` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage2.tags`: atomic,nt,persist: 1, nt,reduce: 1; `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_f766a40e6add720f` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `xbf16=0` | `stage2.tags`: nt,reduce: 1, atomic,nt,persist: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_011458edc01e729c` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_f72c01523ab5aa3b` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_c55f67f3cd31bb49` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x256x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_267663541480cf50` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_06bf72bc940dd0ab` | `M1025-4096` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt,swiglu`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt,persist`, `stage2.tile=16x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_672b6950591cfba9` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.situv=2`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2, 16: 1; `stage1.tags`: f16in,nt: 1, nt: 1, absent: 4; `stage1.tile`: 64x256x256: 3, 128x256x256: 2, 16x128x256: 1; `stage1.xcd`: 2: 5, 1: 1; `stage2.sbm`: 64: 3, 128: 2, 16: 1; `stage2.sp`: 1601: 2, 2401: 1, absent: 3; `stage2.tags`: reduce: 3, atomic,nt: 1, nt,reduce: 1 (+1 more); `stage2.tile`: 64x256x128: 4, 16x128x128: 1, 128x256x128: 1 |
| `cfg_2368dbb1acbd472c` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 4, 64: 1; `stage1.tags`: nt: 1, absent: 4; `stage1.tile`: 128x256x256: 4, 64x256x256: 1; `stage1.xcd`: 2: 4, 4: 1; `stage2.sbm`: 128: 4, 64: 1; `stage2.tags`: atomic: 2, atomic,nt: 1, nt,reduce: 1 (+1 more); `stage2.tile`: 128x128x128: 3, 64x256x128: 1, 128x256x128: 1 |
| `cfg_1daf2116246548b2` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 3, 64: 2, 32: 1; `stage1.tags`: nt: 2, absent: 4; `stage1.tile`: 128x256x256: 3, 64x256x256: 2, 32x128x256: 1; `stage1.xcd`: 2: 5, 4: 1; `stage2.sbm`: 128: 3, 64: 2, 32: 1; `stage2.tags`: atomic: 3, atomic,nt: 2, atomic,nt,persist: 1; `stage2.tile`: 128x128x128: 3, 64x256x128: 2, 32x256x128: 1 |
| `cfg_a39f1a192e9cb590` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 32: 1, 128: 1; `stage1.kw`: 2: 1, absent: 5; `stage1.tags`: nt: 2, absent: 4; `stage1.tile`: 64x256x256: 4, 32x128x256: 1, 128x256x256: 1; `stage1.xcd`: 2: 4, 4: 2; `stage2.sbm`: 64: 4, 32: 1, 128: 1; `stage2.tags`: reduce: 3, atomic,nt,persist: 2, atomic: 1; `stage2.tile`: 64x256x128: 3, 32x256x128: 1, 64x128x128: 1 (+1 more) |
| `cfg_5d03df5328ba7819` | `M>4096` | 1 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=scatter`, `stage2.tile=128x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_2d4cd0a21b63b2a0` | `M>4096` | 1 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=atomic`, `stage2.tile=128x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_ee45f498b9f1a26c` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 128: 2; `stage1.tags`: swiglu: 4, nt,swiglu: 1; `stage1.tile`: 64x256x256: 3, 128x256x256: 2; `stage1.xcd`: 4: 3, 2: 2; `stage2.sbm`: 64: 3, 128: 2; `stage2.tags`: reduce: 2, scatter: 2, nt,reduce: 1; `stage2.tile`: 64x256x128: 2, 128x256x128: 2, 64x128x128: 1 |
| `cfg_b825494d74d85206` | `M>4096` | 2 | `block_m=128`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=scatter`, `stage2.tile=128x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_b39626e3f70978fe` | `M>4096` | 5 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tags=swiglu`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 128: 3, 64: 2; `stage1.tile`: 128x256x256: 3, 64x256x256: 2; `stage2.sbm`: 128: 3, 64: 2; `stage2.tags`: persist,reduce: 2, scatter: 2, reduce: 1; `stage2.tile`: 64x256x128: 2, 128x256x256: 2, 128x128x128: 1 |

### `gfx942` · 80 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `abf16_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_26f918f3ad066ac6` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=2`, `stage1.tile=16x32x128`, `stage1.w=4`, `stage1.xcd=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_af77afa82cdf767f` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x256x128`, `stage2.xcd=4`, `xbf16=0` | one row; nothing to compare |
| `cfg_772f2ed94e8a464b` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 2; `stage1.kw`: 2: 1, absent: 2; `stage1.tile`: 16x64x128: 2, 16x32x128: 1; `stage1.w`: 2: 1, 4: 1, absent: 1; `stage1.xcd`: 1: 1, absent: 2; `stage2.bnt`: 2: 2, absent: 1 |
| `cfg_00220e5028355b6d` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 2; `stage1.kw`: 2: 1, absent: 2; `stage1.tile`: 16x64x128: 2, 16x32x128: 1; `stage1.w`: 3: 1, 4: 1, absent: 1; `stage2.bnt`: 2: 1, absent: 2 |
| `cfg_e191a68256084620` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_6ececd31c0d49f88` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_fa9e7b878c7599b5` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_6f25f2af30ad1bf0` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage2.tile`: 16x256x128: 1, 16x128x128: 1 |
| `cfg_e5e5ce26339e7b85` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_8c2bf5bc90fbf91a` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=16x64x128`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_9597b45d2d9a6012` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 5, 16: 1; `stage1.bnt`: 0: 2, absent: 4; `stage1.tile`: 32x128x128: 3, 32x192x128: 2, 16x64x128: 1; `stage1.w`: 4: 4, 2: 2; `stage1.xcd`: 4: 3, absent: 3; `stage2.tags`: atomic,persist: 5, atomic: 1; `stage2.tile`: 32x128x128: 5, 16x128x128: 1 |
| `cfg_cdaf571239ce576d` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 5, 16: 1; `stage1.tile`: 32x128x128: 5, 16x64x128: 1; `stage1.w`: 4: 5, 2: 1; `stage1.xcd`: 4: 4, absent: 2; `stage2.bnt`: 2: 1, absent: 5; `stage2.tags`: atomic: 3, atomic,persist: 3; `stage2.tile`: 32x128x256: 5, 16x128x128: 1 |

### `gfx942` · 304 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `abf16_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_3357849154f40851` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=2`, `stage1.tile=16x32x128`, `stage1.xcd=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_c2b7d2f3c5b64d6e` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=4`, `stage1.tile=16x32x128`, `stage1.w=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x256x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_857b686eee7fda2f` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=2`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.w`: 2: 2, 3: 1; `stage1.xcd`: 4: 1, absent: 2 |
| `cfg_993dbede6f8627e0` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=2`, `stage1.tile=16x32x128`, `stage1.xcd=1`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `xbf16=0` | `stage1.w`: 4: 1, 3: 1, absent: 1; `stage2.tile`: 16x256x128: 2, 16x256x256: 1; `stage2.xcd`: 4: 2, absent: 1 |
| `cfg_e565b511aea5f2b5` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x256x128`, `stage2.xcd=4`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 16x64x256: 1, 16x64x128: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_b6ade0634320bf2f` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 16x64x256: 1, 16x64x128: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_14f887a79d585de6` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1; `stage1.xcd`: 1: 1, 4: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 16x128x128: 1, 16x256x128: 1 |
| `cfg_5c5b031186c3e8ec` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist`, `stage2.tile=16x128x256`, `xbf16=0` | `stage1.w`: 4: 1, absent: 1; `stage1.xcd`: 1: 1, absent: 1 |
| `cfg_6981ee2c3d17079e` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage1.xcd=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x256x128`, `xbf16=0` | `stage1.w`: 4: 1, 2: 1 |
| `cfg_e4a186805b8115dd` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage1.xcd=4`, `stage2.family=moe_stage2`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 16x256x128: 1, 16x128x256: 1 |
| `cfg_e66486f0cf007394` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=4`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 5, 16: 1; `stage1.tile`: 32x128x128: 3, 16x64x128: 1, 32x64x128: 1 (+1 more); `stage1.w`: 4: 4, 2: 1, absent: 1; `stage2.bnt`: 2: 5, absent: 1; `stage2.tags`: atomic: 5, atomic,persist: 1; `stage2.tile`: 32x128x128: 5, 16x256x128: 1; `stage2.xcd`: 4: 4, absent: 2 |
| `cfg_43b1aa7bcf2dc31d` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.w=4`, `stage2.family=moe_stage2`, `xbf16=0` | `block_m`: 32: 5, 16: 1; `stage1.tile`: 32x128x128: 4, 16x64x128: 1, 32x64x128: 1; `stage1.xcd`: 4: 5, 1: 1; `stage2.bnt`: 2: 1, absent: 5; `stage2.tags`: atomic,persist: 4, atomic: 2; `stage2.tile`: 32x128x256: 5, 16x256x128: 1; `stage2.xcd`: 4: 3, absent: 3 |

### `gfx942` · 304 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `abf16_wint4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_bb2b33aeb5d80e4a` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_ecf60803bb18a7f3` | `M<=16` | 1 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=4`, `stage1.tile=16x16x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_c191b75ca7a6606f` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `xbf16=0` | `stage2.tags`: atomic: 2, reduce: 1; `stage2.tile`: 16x128x128: 2, 16x128x256: 1 |
| `cfg_aa404458d00cda64` | `M17-64` | 3 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 2; `stage1.kw`: 2: 2, 4: 1 |
| `cfg_4340685dcf9f1973` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `stage2.tile=16x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_ee45340a9b25540c` | `M65-256` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `stage2.tile`: 16x128x128: 1, 16x128x256: 1 |
| `cfg_87caf76039bf045e` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x128`, `stage2.family=moe_stage2`, `xbf16=0` | `stage2.tags`: atomic: 1, reduce: 1; `stage2.tile`: 16x128x128: 1, 16x128x256: 1 |
| `cfg_639e0888c78a4e07` | `M257-1024` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_1b5b9cf960a747ee` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x128`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `stage2.tile=32x128x128`, `xbf16=0` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_6c4e708ec6be5c1f` | `M1025-4096` | 2 | `block_m=16`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128`, `xbf16=0` | `stage1.tile`: 16x64x128: 1, 16x128x128: 1 |
| `cfg_9c2e9aaabd9c70c3` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage2.family=moe_stage2`, `stage2.tags=reduce`, `xbf16=0` | `block_m`: 64: 5, 32: 1; `stage1.tile`: 64x128x128: 5, 32x128x128: 1; `stage2.tile`: 64x128x128: 5, 32x128x128: 1 |
| `cfg_51c6a84beb578232` | `M>4096` | 6 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `xbf16=0` | `block_m`: 32: 4, 16: 1, 64: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 32x128x128: 4, 16x128x128: 1, 64x128x128: 1; `stage2.tile`: 32x128x128: 2, 32x128x256: 2, 16x128x256: 1 (+1 more) |

### `gfx1250` · 192 CUs · `tuned_grouped_fmoe` · FlyDSL `grouped_stage1` · `a8w4`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_26752455d9d9cf38` | `M<=16` | 1 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=3`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_ec14e67100a8391f` | `M<=16` | 1 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_d52d97a002aac134` | `M<=16` | 1 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=3`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_k2=512`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_10b812b962f45051` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=16`, `tile_n=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `num_buffer_stage2`: 2: 1, absent: 1; `tile_k2`: 256: 1, absent: 1; `tile_m2`: 16: 1, absent: 1; `tile_n2`: 512: 1, absent: 1 |
| `cfg_bf3a0da6fa33948f` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `num_buffers`: 3: 1, 2: 1 |
| `cfg_72ce3d58135b31f1` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_m=16`, `tile_n=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `num_buffer_stage2`: 2: 1, absent: 1; `tile_k2`: 128: 1, absent: 1; `tile_m2`: 16: 1, absent: 1; `tile_n2`: 512: 1, absent: 1 |
| `cfg_c9f72656b66afb46` | `M65-256` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `next_stage_prefetch`: 1: 1, 0: 1; `tile_k`: 256: 1, 128: 1; `tile_n`: 256: 1, 512: 1; `tile_n2`: 256: 1, 512: 1 |
| `cfg_2d79208be913021b` | `M65-256` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_m=16`, `tile_n=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `num_buffer_stage2`: 3: 1, absent: 1; `tile_k`: 512: 1, 128: 1; `tile_k2`: 128: 1, absent: 1; `tile_m2`: 16: 1, absent: 1; `tile_n2`: 512: 1, absent: 1 |
| `cfg_b6ef1c11b9fd43cd` | `M65-256` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `num_buffer_stage2=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k2=512`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `next_stage_prefetch`: 1: 1, 0: 1; `num_buffers`: 3: 1, 2: 1; `tile_k`: 256: 1, 128: 1 |
| `cfg_d1a8bc7752d3d743` | `M257-1024` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_7171d8112343b2f2` | `M257-1024` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=128`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `num_buffers`: 2: 1, 3: 1; `tile_n`: 256: 1, 512: 1 |
| `cfg_572d7d32a80a7f0c` | `M257-1024` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_m=16`, `tile_n=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `next_stage_prefetch`: 1: 1, 0: 1; `num_buffer_stage2`: 2: 1, absent: 1; `tile_k`: 128: 1, 512: 1; `tile_k2`: 512: 1, absent: 1; `tile_m2`: 16: 1, absent: 1; `tile_n2`: 256: 1, absent: 1 |
| `cfg_980503582acd3a9e` | `M1025-4096` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=512`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_341ce62975b380d1` | `M1025-4096` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=1`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=128`, `tile_k2=256`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_a4540aef81706aad` | `M1025-4096` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `max_m=16`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=3`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k2=512`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `tile_k`: 512: 1, 128: 1 |
| `cfg_6e8a1c45774ae04f` | `M>4096` | 3 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=128`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 64: 1, 128: 1, 192: 1; `num_buffer_stage2`: 2: 2, 3: 1; `tile_k2`: 256: 2, 128: 1; `tile_m`: 64: 2, 128: 1; `tile_m2`: 64: 2, 128: 1; `tile_n`: 512: 2, 256: 1; `tile_n2`: 256: 2, 512: 1 |
| `cfg_443903543f37f9c3` | `M>4096` | 3 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=128`, `tile_n=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 64: 1, 128: 1, 192: 1; `num_buffer_stage2`: 3: 1, 2: 1, absent: 1; `tile_k2`: 128: 1, 256: 1, absent: 1; `tile_m`: 64: 2, 128: 1; `tile_m2`: 64: 1, 128: 1, absent: 1; `tile_n2`: 256: 2, absent: 1 |
| `cfg_1d9b4b4d57aab8a0` | `M>4096` | 3 | `cluster_n=-1`, `grouped_persistent_m=0`, `m_warp=1`, `n_warp=4`, `next_stage_prefetch=0`, `num_buffer_stage2=3`, `num_buffers=3`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=128`, `tile_k2=512`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 64: 1, 128: 1, 192: 1; `tile_m`: 64: 2, 128: 1; `tile_m2`: 64: 2, 128: 1; `tile_n`: 256: 2, 512: 1 |

### `gfx1250` · 256 CUs · `tuned_grouped_fmoe` · FlyDSL `grouped_stage1` · `fp4`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_82f07e7f676a4f4b` | `M<=16` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=16`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=16`, `tile_n=256`, `waves_per_tensor_tdm=-1` | `split_k1`: 4: 1, 1: 1; `split_k2`: 2: 1, 1: 1 |
| `cfg_5e332dbefdbaca00` | `M<=16` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_n=512`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `tile_k`: 256: 2, 128: 2; `tile_k2`: 256: 2, 128: 2; `tile_m`: 64: 2, 16: 2; `tile_m2`: 64: 2, 16: 2 |
| `cfg_8f050f96131ff4e6` | `M<=16` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=256`, `tile_m=64`, `tile_m2=64`, `tile_n=512`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_e947117c280d489e` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=16`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=16`, `tile_n=256`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_296a96f75a195f86` | `M17-64` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `waves_per_tensor_tdm=-1` | `max_m`: 192: 2, 384: 2; `num_buffer_stage2`: 2: 3, absent: 1; `tile_k`: 128: 2, 256: 1, 512: 1; `tile_k2`: 128: 3, absent: 1; `tile_m`: 16: 3, 32: 1; `tile_m2`: 16: 3, absent: 1; `tile_n`: 256: 2, 512: 2; `tile_n2`: 512: 2, 256: 1, absent: 1; `wave_specialized_tdm`: 1: 3, absent: 1 |
| `cfg_803f0e2fae38369b` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_n=256`, `waves_per_tensor_tdm=-1` | `max_m`: 192: 1, 384: 1; `num_buffer_stage2`: 2: 1, absent: 1; `tile_k`: 256: 1, 512: 1; `tile_k2`: 128: 1, absent: 1; `tile_m`: 32: 1, 16: 1; `tile_m2`: 16: 1, absent: 1; `tile_n2`: 256: 1, absent: 1; `wave_specialized_tdm`: 1: 1, absent: 1 |
| `cfg_4eaec2f6dc8eef7b` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_n=512`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `tile_k`: 256: 1, 128: 1; `tile_k2`: 256: 1, 128: 1; `tile_m`: 64: 1, 16: 1; `tile_m2`: 64: 1, 16: 1 |
| `cfg_7b30a145b48c5d82` | `M17-64` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_3e72e34908e681fb` | `M65-256` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=32`, `tile_n=256`, `waves_per_tensor_tdm=-1` | `max_m`: 16: 1, 192: 1 |
| `cfg_abd8f2aa5f09ef7a` | `M65-256` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 768: 2, 1536: 2; `num_buffers`: 2: 3, 3: 1; `tile_k`: 128: 3, 512: 1; `tile_k2`: 128: 3, 512: 1; `tile_m`: 16: 3, 32: 1; `tile_m2`: 16: 3, 32: 1; `tile_n`: 256: 2, 512: 2; `tile_n2`: 256: 2, 512: 2 |
| `cfg_c70cdbd23ba8529b` | `M65-256` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 768: 1, 1536: 1; `num_buffers`: 2: 1, 3: 1; `tile_k`: 512: 1, 128: 1; `tile_k2`: 128: 1, 512: 1 |
| `cfg_84fb275c58431b28` | `M65-256` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `waves_per_tensor_tdm=-1` | `max_m`: 384: 2, 192: 2; `num_buffer_stage2`: 2: 3, absent: 1; `tile_k`: 256: 2, 128: 2; `tile_k2`: 128: 2, 256: 1, absent: 1; `tile_m`: 16: 2, 64: 1, 32: 1; `tile_m2`: 16: 2, 64: 1, absent: 1; `tile_n`: 512: 3, 256: 1; `tile_n2`: 512: 3, absent: 1; `wave_specialized_tdm`: 1: 3, absent: 1 |
| `cfg_038d364fadc14f21` | `M65-256` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_2aea278069208fc6` | `M257-1024` | 5 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `waves_per_tensor_tdm=-1` | `max_m`: 384: 4, 768: 1; `num_buffer_stage2`: 2: 4, absent: 1; `tile_k`: 128: 4, 256: 1; `tile_k2`: 128: 4, absent: 1; `tile_m`: 16: 4, 32: 1; `tile_m2`: 16: 4, absent: 1; `tile_n`: 512: 4, 256: 1; `tile_n2`: 512: 4, absent: 1; `wave_specialized_tdm`: 1: 4, absent: 1 |
| `cfg_0da17d967b02c3cc` | `M257-1024` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 384: 2, 3072: 1, 6144: 1; `next_stage_prefetch`: 0: 3, 1: 1; `num_buffer_stage2`: 2: 3, 3: 1; `num_buffers`: 2: 3, 3: 1; `tile_k`: 128: 2, 256: 2; `tile_k2`: 128: 3, 256: 1; `tile_m`: 64: 2, 128: 1, 32: 1; `tile_m2`: 64: 2, 128: 1, 32: 1; `tile_n`: 512: 3, 256: 1; `tile_n2`: 512: 2, 256: 2 |
| `cfg_0418a2de253061a1` | `M257-1024` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 3072: 1, 6144: 1; `next_stage_prefetch`: 0: 1, 1: 1; `tile_m`: 64: 1, 128: 1; `tile_m2`: 64: 1, 128: 1; `tile_n`: 256: 1, 512: 1 |
| `cfg_aa3f2e93401caaaa` | `M257-1024` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 384: 2, 768: 2; `tile_k`: 512: 2, 128: 2; `tile_n`: 256: 2, 512: 2; `tile_n2`: 256: 2, 512: 2 |
| `cfg_14884c9b0e4a006e` | `M257-1024` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `max_m=384`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | one row; nothing to compare |
| `cfg_86689ad75c367477` | `M1025-4096` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=32`, `tile_n=256`, `waves_per_tensor_tdm=-1` | `max_m`: 1536: 1, 3072: 1 |
| `cfg_d1bafab262fa98fc` | `M1025-4096` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tile_k2=128`, `tile_n=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 12288: 2, 24576: 2; `next_stage_prefetch`: 1: 2, 0: 2; `num_buffer_stage2`: 2: 3, 4: 1; `tdm_as_in_prologue`: 0: 3, 1: 1; `tdm_b_th`: 1: 1, absent: 3; `tile_k`: 256: 2, 128: 2; `tile_m`: 128: 2, 32: 1, 64: 1; `tile_m2`: 128: 2, 32: 1, 64: 1; `tile_n2`: 256: 2, 512: 2 |
| `cfg_6ed4d611ba94cbb4` | `M1025-4096` | 2 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=1`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=128`, `tile_m2=128`, `tile_n=512`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 12288: 1, 24576: 1 |
| `cfg_542e7dc1f7a42bf5` | `M1025-4096` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 1536: 2, 384: 2; `next_stage_prefetch`: 0: 3, 1: 1; `num_buffer_stage2`: 2: 3, 3: 1; `num_buffers`: 3: 2, 2: 2; `tdm_as_in_prologue`: 0: 3, 1: 1; `tdm_b_th`: 1: 1, absent: 3; `tile_k`: 128: 3, 256: 1; `tile_k2`: 128: 3, 512: 1; `tile_m`: 64: 2, 16: 1, 32: 1; `tile_m2`: 64: 2, 16: 1, 32: 1; `tile_n`: 256: 2, 512: 2; `tile_n2`: 512: 3, 256: 1 |
| `cfg_2e9e6fc303b91d10` | `M1025-4096` | 4 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=0`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=128`, `tile_m=16`, `tile_m2=16`, `tile_n=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 384: 2, 512: 2 |
| `cfg_eaf7872e64a3290b` | `M>4096` | 6 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_n=256`, `waves_per_tensor_tdm=-1` | `max_m`: 6144: 1, 12288: 1, 24576: 1 (+3 more); `next_stage_prefetch`: 0: 3, 1: 3; `tile_m`: 64: 3, 256: 2, 128: 1 |
| `cfg_ce516fb752325780` | `M>4096` | 1 | `cluster_m=1`, `cluster_m2=-1`, `cluster_n=4`, `cluster_n2=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=1`, `m_warp=2`, `m_warp2=2`, `max_m=196608`, `n_warp=2`, `n_warp2=2`, `next_stage_prefetch=1`, `num_buffer_stage2=4`, `num_buffers=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=256`, `tile_m=256`, `tile_m2=256`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=2`, `waves_per_tensor_tdm2=2` | one row; nothing to compare |
| `cfg_553e1d912f8c93ed` | `M>4096` | 3 | `cluster_n=4`, `grouped_persistent_m=0`, `lds_soa_load_interleave=1`, `m_warp=2`, `m_warp2=2`, `n_warp=2`, `n_warp2=2`, `next_stage_prefetch=1`, `num_buffer_stage2=4`, `num_buffers=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=256`, `tile_m=256`, `tile_m2=256`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=1` | `cluster_m`: 1: 1, absent: 2; `cluster_m2`: 1: 1, absent: 2; `max_m`: 16384: 2, 196608: 1 |
| `cfg_4a5c50c495f1a7df` | `M>4096` | 1 | `cluster_m=1`, `cluster_m2=-1`, `cluster_n=4`, `cluster_n2=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=1`, `m_warp=2`, `m_warp2=2`, `max_m=196608`, `n_warp=2`, `n_warp2=2`, `next_stage_prefetch=1`, `num_buffer_stage2=4`, `num_buffers=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_k2=256`, `tile_m=256`, `tile_m2=256`, `tile_n=256`, `tile_n2=256`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=2`, `waves_per_tensor_tdm2=2` | one row; nothing to compare |
| `cfg_1285eb87a0e0d7d5` | `M>4096` | 8 | `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `wave_specialized_tdm=1` | `cluster_n`: -1: 4, 4: 4; `cluster_n2`: 1: 2, absent: 6; `ep_fused`: 1: 2, absent: 6; `m_warp`: 2: 5, 1: 3; `m_warp2`: 2: 5, 1: 3; `max_m`: 196608: 4, 98304: 3, 49152: 1; `n_warp`: 2: 5, 4: 3; `n_warp2`: 2: 5, 4: 3; `next_stage_prefetch`: 1: 6, 0: 2; `num_buffer_stage2`: 3: 6, 2: 1, 4: 1; `num_buffers`: 3: 6, 2: 2; `tdm_as_in_prologue`: 0: 6, 1: 2; `tdm_b_th`: 1: 2, absent: 6; `tile_k`: 256: 7, 128: 1; `tile_k2`: 128: 4, 256: 4; `tile_m`: 256: 5, 32: 2, 128: 1; `tile_m2`: 128: 3, 256: 3, 32: 2; `tile_n`: 256: 6, 512: 2; `tile_n2`: 256: 6, 512: 2; `waves_per_tensor_tdm`: -1: 5, 1: 3 |
| `cfg_f8ee7ba2653e94cf` | `M>4096` | 3 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `next_stage_prefetch=1`, `num_buffer_stage2=2`, `num_buffers=2`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_m=128`, `tile_m2=128`, `tile_n=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 49152: 1, 98304: 1, 196608: 1; `tile_k2`: 128: 2, 256: 1; `tile_n2`: 256: 2, 512: 1 |
| `cfg_541d03e28146ac18` | `M>4096` | 12 | `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `wave_specialized_tdm=1` | `cluster_n`: -1: 8, 4: 4; `m_warp`: 1: 8, 2: 4; `m_warp2`: 1: 8, 2: 4; `max_m`: 196608: 3, 384: 2, 12288: 2 (+3 more); `n_warp`: 4: 8, 2: 4; `n_warp2`: 4: 8, 2: 4; `next_stage_prefetch`: 1: 8, 0: 4; `num_buffer_stage2`: 2: 8, 3: 2, 4: 2; `num_buffers`: 2: 8, 3: 2, 4: 2; `tile_k`: 256: 8, 128: 4; `tile_k2`: 128: 8, 256: 4; `tile_m`: 64: 4, 128: 4, 256: 4; `tile_m2`: 64: 4, 128: 4, 256: 4; `tile_n`: 512: 8, 256: 4; `tile_n2`: 256: 8, 512: 4; `waves_per_tensor_tdm`: -1: 8, 1: 4 |
| `cfg_d1fa058cf4555bf6` | `M>4096` | 12 | `cluster_n=-1`, `grouped_persistent_m=0`, `lds_soa_load_interleave=0`, `m_warp=1`, `m_warp2=1`, `n_warp=4`, `n_warp2=4`, `split_k1=1`, `split_k2=1`, `stage1.tags=gugu`, `stage2.family=grouped_stage2`, `tdm_as_in_prologue=0`, `tile_k=256`, `tile_n2=512`, `wave_specialized_tdm=1`, `waves_per_tensor_tdm=-1` | `max_m`: 1024: 2, 2048: 2, 4096: 2 (+3 more); `next_stage_prefetch`: 0: 8, 1: 4; `num_buffer_stage2`: 2: 10, 3: 2; `num_buffers`: 2: 10, 3: 2; `tile_k2`: 128: 8, 256: 4; `tile_m`: 16: 8, 128: 4; `tile_m2`: 16: 8, 128: 4; `tile_n`: 256: 8, 512: 4 |

### `` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `abf16_wint4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_be73014232154709` | `M<=16` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_39d9db8a0629fef4` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_158ce9f52b378173` | `M<=16` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_db3faae2eb9b3e58` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_b9bdf69fbc1137ee` | `M<=16` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.tile`: 16x16x256: 1, 16x32x128: 1 |
| `cfg_0b051eebae6a71b1` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128` | one row; nothing to compare |
| `cfg_ce8e8e6ea9d3b203` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_d08f42fc69abee71` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage2.tile`: 16x128x256: 1, 16x128x128: 1 |
| `cfg_3c04edfe6bdc863d` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_321f97c27a30f271` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_ccf933b67355624c` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.kw=4`, `stage1.tile=16x32x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_137de8361dc62659` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=16x16x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_669382ad68bdd617` | `M65-256` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_5994f8f6a9d0347f` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.tile`: 16x64x128: 1, 16x128x128: 1 |
| `cfg_5f88e561fadc04e8` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.kw`: 4: 1, absent: 1; `stage1.tile`: 16x32x128: 1, 16x64x128: 1 |
| `cfg_89b053a201b37cb5` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.tile`: 16x16x128: 1, 16x32x128: 1 |
| `cfg_7037918a6bfd3361` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage1.kw`: 4: 1, absent: 1; `stage1.tile`: 16x32x128: 1, 16x64x128: 1; `stage2.tile`: 16x128x128: 1, 16x128x256: 1 |
| `cfg_a7d4505ce96e5931` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.kw`: 4: 1, absent: 1; `stage1.tile`: 16x32x128: 1, 16x128x128: 1 |
| `cfg_892745399f94d740` | `M257-1024` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128` | one row; nothing to compare |
| `cfg_ae4807d8a738eee0` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_74db988ddc9fa18f` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_a2400138b16287b3` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_98a050bad619f059` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_66e41850f9642ce8` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage2.tile`: 16x128x128: 1, 16x128x256: 1 |
| `cfg_d035d7659d09d2f1` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x128` | `stage1.tile`: 16x128x128: 1, 16x64x128: 1 |
| `cfg_fc52cfa51d18f726` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage2.tile`: 16x128x256: 1, 16x128x128: 1 |
| `cfg_15d7b16fc25c8fd4` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_98b3376dc3f60dfc` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_37bfabcab3b7c8eb` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x64x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=16x128x256` | one row; nothing to compare |
| `cfg_3bee68ed8077ac02` | `M1025-4096` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x128`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage2.tile`: 16x128x128: 1, 16x128x256: 1 |
| `cfg_bc34e7f526969a63` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 5, 32: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x128: 5, 32x128x128: 1; `stage2.tile`: 64x128x128: 5, 32x128x128: 1 |
| `cfg_883c7e50a711dbf9` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 3, 16: 1, 32: 1; `stage1.bnt`: 0: 2, absent: 3; `stage1.tile`: 64x128x128: 3, 16x128x128: 1, 32x128x128: 1; `stage2.tile`: 64x128x128: 3, 16x128x256: 1, 32x128x128: 1 |
| `cfg_5e36cba34782269c` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 4, 32: 2; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x128: 4, 32x64x256: 1, 32x128x128: 1; `stage2.tile`: 64x128x128: 4, 32x128x128: 2 |
| `cfg_a2612ca46ed54c71` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 3, 16: 1, 32: 1; `stage1.bnt`: 0: 2, absent: 3; `stage1.tile`: 64x128x128: 3, 16x64x128: 1, 32x64x256: 1; `stage2.tile`: 64x128x128: 3, 16x128x256: 1, 32x128x128: 1 |
| `cfg_c8f80af012851a71` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 4, 32: 2; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x128: 4, 32x128x128: 2; `stage2.tile`: 64x128x128: 4, 32x128x128: 2 |
| `cfg_7068b57387214c15` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 3, 16: 1, 32: 1; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x128x128: 3, 16x128x128: 1, 32x128x128: 1; `stage2.tile`: 64x128x128: 3, 16x128x128: 1, 32x128x128: 1 |

### `` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `afp4_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_9262492bdf8b3c88` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=4` | one row; nothing to compare |
| `cfg_99bba4f2c4efb340` | `M<=16` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x32x256: 1, 64x64x256: 1; `stage1.w`: 4: 1, 3: 1 |
| `cfg_1881c88b15bd7360` | `M<=16` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt` | `block_m`: 64: 1, 32: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 64x64x256: 1, 32x128x256: 1; `stage2.sbm`: 64: 1, 32: 1; `stage2.tile`: 64x128x128: 1, 32x128x128: 1 |
| `cfg_3d1e0ffb34b272cb` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.w=4` | `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 32x32x256: 1, 32x64x256: 1 |
| `cfg_527610ae24a0a688` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, absent: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: reduce: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_c32cac93b3c9fb59` | `M<=16` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.w=4` | `block_m`: 64: 1, 32: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 64x128x256: 1, 32x128x256: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_190c7f28d6ca1c4d` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4` | `stage1.bnt`: 0: 1, absent: 1; `stage1.fp`: 4: 1, absent: 1 |
| `cfg_cc4c7398e97b055a` | `M<=16` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=4` | one row; nothing to compare |
| `cfg_f814dca3b794ac81` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=3` | `stage1.bnt`: 0: 1, absent: 1 |
| `cfg_bb0d0ff7c3f196e3` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4` | `stage1.bnt`: 0: 1, absent: 1; `stage1.kw`: 4: 1, 2: 1; `stage1.tile`: 32x32x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: reduce: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_6835a80016b1ed6f` | `M<=16` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage1.w=2` | one row; nothing to compare |
| `cfg_4e36985d7f0cb212` | `M<=16` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_68c1637b87f4327b` | `M<=16` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kb=14`, `stage1.tile=32x128x256`, `stage1.w=3` | one row; nothing to compare |
| `cfg_05a9f45c97a609ee` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=3` | `stage1.kb`: 14: 1, 2: 1; `stage1.tags`: go: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.family`: moe_stage2: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_9b1e2aa57641d689` | `M<=16` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kb=4`, `stage1.tags=go`, `stage1.tile=32x64x256`, `stage1.w=3` | one row; nothing to compare |
| `cfg_f385e1ea78e367c1` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_ed19d961f597ea00` | `M17-64` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4` | `block_m`: 64: 1, 32: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.tile`: 64x64x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 3: 1 |
| `cfg_fb93a7944e91928f` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt` | `stage2.tile`: 32x128x128: 1, 32x256x128: 1 |
| `cfg_ba19b8170c6c20a7` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt` | `stage1.bnt`: 0: 1, absent: 1; `stage1.w`: 2: 1, 4: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_a06fbd76fe831ca0` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_7bb8c23e107206f6` | `M17-64` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_81d203898dd59f58` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | `stage1.bnt`: 0: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_f01b3e26be5be7c9` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tile=32x32x256`, `stage1.w=4` | `stage1.fp`: 4: 1, absent: 1; `stage1.kw`: 4: 1, 2: 1; `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: reduce: 1, absent: 1; `stage2.tile`: 32x128x256: 1, absent: 1 |
| `cfg_8967a10ebdc1b3ae` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.kw=4`, `stage1.tile=32x32x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=nt,reduce`, `stage2.tile=32x128x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.w`: 2: 1, 4: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_b65c76cf0628e3fe` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_e7acf37bd7e472db` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x256x256` | `stage1.kw`: 2: 1, absent: 1; `stage1.w`: 3: 1, 2: 1 |
| `cfg_6115458dd7db3ceb` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256` | `stage1.w`: 4: 1, 2: 1 |
| `cfg_53b393971ecfb699` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.w=3` | `stage1.kb`: 4: 1, 7: 1; `stage1.tags`: go: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage2.family`: moe_stage2: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_ca6edb7938987ad7` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.w=3`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1; `stage2.xcd`: 4: 1, absent: 1 |
| `cfg_e646675ea70dbd3e` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3` | `stage1.kb`: 7: 1, absent: 1; `stage1.xcd`: 4: 1, absent: 1; `stage2.family`: moe_stage2: 1, absent: 1; `stage2.tags`: atomic: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_5833d3005a573a49` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_baf6b9c0cf0baeea` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.bnt`: 0: 1, absent: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 4: 1 |
| `cfg_eda1d69e862b974e` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1 |
| `cfg_bec407113ddda532` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_398755f9b6bd67f6` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.w`: 2: 1, 4: 1 |
| `cfg_635be0d584a0f744` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_83a3e5e9e14856ec` | `M65-256` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_42fdb3fd23d51490` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=reduce`, `stage2.tile=32x128x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tile`: 32x32x256: 1, 32x128x256: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_87c81f1e7200dc06` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32` | `stage1.w`: 4: 1, 2: 1; `stage2.tags`: nt,reduce: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_a3a7667c07bb44d0` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_fc95d4c4ecf061eb` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x256x256` | `stage1.tile`: 32x32x256: 1, 32x64x256: 1; `stage1.w`: 3: 1, 2: 1 |
| `cfg_c25989cab91ddf00` | `M65-256` | 2 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2`, `stage2.tags=atomic` | `block_m`: 64: 1, 32: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 64x128x256: 1, 32x64x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.tile`: 64x128x256: 1, 32x128x256: 1 |
| `cfg_e001117d143fa21f` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_9e158a8fb0545982` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x128x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 3: 1 |
| `cfg_9a5836db884b96be` | `M257-1024` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_2d32c426237e4576` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_c62e147c3b3899b9` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_5d9b2681a11318af` | `M257-1024` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_0dab19730793e3d3` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x128x128` | one row; nothing to compare |
| `cfg_1c89cb045a5f5143` | `M257-1024` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_0780b97a1e643b9f` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_55bc74c8f2f2d8f8` | `M257-1024` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.tags=reduce` | `block_m`: 32: 1, 64: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x128x256: 1, 64x128x256: 1; `stage2.sbm`: 32: 1, 64: 1; `stage2.tile`: 32x128x256: 1, 64x128x256: 1 |
| `cfg_7103440fe980d7fc` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_10b2d8d729167fbe` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=3` | one row; nothing to compare |
| `cfg_0daf65c227fc834e` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=3` | `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_84500529523843b0` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_9385907df49344fa` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x256` | `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1 |
| `cfg_375d24c8d9fdc8fb` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt,persist`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_6bf9def201a29617` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_eeffd53997a262a3` | `M257-1024` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_8e975c7d1806b210` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage2.family=moe_stage2` | `stage1.w`: 2: 1, 3: 1; `stage2.bnt`: 2: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 16x128x256: 1, 32x128x256: 1 |
| `cfg_eea5efc3e5dde5aa` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.sbm=32`, `stage2.tile=16x256x256` | `stage1.w`: 2: 1, 4: 1; `stage2.tags`: atomic,persist: 1, atomic: 1 |
| `cfg_98d2d1d8009ed492` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x256x256` | `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 3: 1, 2: 1 |
| `cfg_039c719d01b14133` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tile`: 16x256x256: 1, 32x256x256: 1 |
| `cfg_ab2a619d34be4b21` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_5983e50a2ff47a4b` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt` | `stage1.w`: 4: 1, 2: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_0df0eb8e098caa7b` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1; `stage2.tile`: 32x256x256: 1, 32x128x256: 1 |
| `cfg_a2acc3474d7358e7` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x128`, `xbf16=0` | `stage1.w`: 2: 1, 4: 1 |
| `cfg_1171a8ac0328661d` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128` | one row; nothing to compare |
| `cfg_931793a9d55310b3` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x128x256`, `xbf16=0` | `stage1.w`: 3: 1, 2: 1; `stage2.tags`: atomic,nt: 1, reduce: 1 |
| `cfg_ea7d3927a2003a7a` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_394367b28010e4f7` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 1, 64: 1; `stage1.bnt`: 0: 1, absent: 1; `stage1.fp`: 4: 1, absent: 1; `stage1.tile`: 128x128x256: 1, 64x128x256: 1; `stage1.w`: 3: 1, 2: 1; `stage2.sbm`: 128: 1, 64: 1; `stage2.tags`: reduce: 1, atomic,persist: 1; `stage2.tile`: 128x128x256: 1, 64x128x128: 1 |
| `cfg_d2c458e2a6500c05` | `M1025-4096` | 2 | `block_m=64`, `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.fp=4`, `stage1.tile=64x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64` | `stage1.w`: 4: 1, 2: 1; `stage2.tags`: atomic,nt: 1, atomic,persist: 1; `stage2.tile`: 64x128x256: 1, 64x128x128: 1 |
| `cfg_fe1bd3b21742356c` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32` | `stage1.tile`: 32x128x256: 1, 32x32x256: 1; `stage2.tags`: reduce: 1, atomic,nt: 1; `stage2.tile`: 32x128x256: 1, 32x256x256: 1 |
| `cfg_de3317caac873345` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=3` | `stage2.family`: moe_stage2_layout: 1, absent: 1; `stage2.sbm`: 32: 1, absent: 1; `stage2.tags`: atomic,nt: 1, absent: 1; `stage2.tile`: 32x256x256: 1, absent: 1 |
| `cfg_dcb96fd23fa66fd1` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_27caae9796cccf11` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x256` | `stage2.tags`: atomic,nt,persist: 1, atomic,nt: 1 |
| `cfg_a9541a3af2d84709` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tile=32x256x256` | `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1 |
| `cfg_5c029750911bf276` | `M1025-4096` | 1 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_18e69beda9f9e9f2` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256`, `xbf16=0` | one row; nothing to compare |
| `cfg_b661937c7b3d67f7` | `M1025-4096` | 2 | `block_m=32`, `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.kw=2`, `stage1.tile=32x32x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x128`, `xbf16=0` | one row; nothing to compare |
| `cfg_0d90ca0b68458960` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tile=32x32x256`, `stage1.w=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.sbm=32` | `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 16x256x256: 1, 16x128x256: 1 |
| `cfg_4fd8ae2e93b6f6d6` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic,persist` | `stage2.sbm`: 32: 1, absent: 1; `stage2.tile`: 32x256x256: 1, 16x256x256: 1 |
| `cfg_e3b733ff0924b782` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.bnt=2`, `stage2.family=moe_stage2`, `stage2.tags=atomic`, `stage2.tile=32x256x256` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x64x256: 1, 64x128x256: 1; `stage1.w`: 2: 1, 3: 1; `stage2.sbm`: 64: 1, absent: 1 |
| `cfg_ca302b6fa92759fd` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 128: 1; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x128x256: 4, 128x128x256: 1; `stage1.w`: 4: 2, 3: 1, 2: 1 (+1 more); `stage2.sbm`: 64: 4, 128: 1; `stage2.tags`: atomic: 3, atomic,nt: 1, nt,reduce: 1; `stage2.tile`: 64x128x256: 4, 128x128x256: 1 |
| `cfg_fc676e4500b85e0e` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.tile=64x128x256` | `block_m`: 128: 4, 64: 2; `stage1.fp`: 4: 2, absent: 4; `stage1.tile`: 128x128x256: 4, 64x128x256: 2; `stage1.w`: 4: 2, absent: 4; `stage2.sbm`: 128: 4, 64: 2; `stage2.tags`: atomic,nt: 4, atomic,nt,persist: 2 |
| `cfg_292e2d047d80088a` | `M>4096` | 6 | `block_m=64`, `ksplit=0`, `run_1stage=0`, `stage1.tile=64x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tile=64x256x256` | `stage1.bnt`: 0: 4, absent: 2; `stage1.w`: 2: 3, 3: 2, 4: 1; `stage1.xcd`: 4: 4, absent: 2; `stage2.tags`: reduce: 5, persist,reduce: 1 |
| `cfg_8672bf14f20d0c93` | `M>4096` | 5 | `block_m=64`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage1.tile=64x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `xbf16=0` | `stage1.bnt`: 0: 4, absent: 1; `stage1.w`: 3: 3, 4: 1, 2: 1; `stage2.tags`: atomic,persist: 2, atomic: 2, atomic,nt: 1; `stage2.tile`: 64x256x256: 3, 64x128x256: 2 |
| `cfg_0f20fdc4ffd26fc0` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 64: 5, 32: 1; `stage1.bnt`: 0: 4, absent: 2; `stage1.tile`: 64x128x256: 5, 32x128x256: 1; `stage1.w`: 3: 3, 2: 2, 4: 1; `stage1.xcd`: 4: 4, absent: 2; `stage2.sbm`: 64: 5, 32: 1; `stage2.tags`: persist,reduce: 3, atomic,nt: 2, reduce: 1; `stage2.tile`: 64x256x128: 4, 32x128x128: 2 |
| `cfg_e55c67cefb43f7d3` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 4, 128: 1; `stage1.bnt`: 0: 3, absent: 2; `stage1.tile`: 64x128x256: 4, 128x128x256: 1; `stage1.w`: 3: 2, 4: 2, 2: 1; `stage2.sbm`: 64: 4, 128: 1; `stage2.tags`: persist,reduce: 3, nt,reduce: 2; `stage2.tile`: 64x128x256: 4, 128x128x256: 1 |
| `cfg_11ada3258102614f` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.tile=64x128x256` | `block_m`: 64: 5, 128: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.tile`: 64x64x256: 3, 64x128x256: 2, 128x128x256: 1; `stage2.sbm`: 64: 5, 128: 1; `stage2.tags`: reduce: 3, atomic,nt,persist: 2, atomic,nt: 1 |
| `cfg_cec7b9ae24b9fdc2` | `M>4096` | 2 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tile=128x256x128` | `stage1.fp`: 4: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage2.tags`: reduce: 1, atomic: 1 |
| `cfg_204f55f3a8697119` | `M>4096` | 3 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128` | `stage1.bnt`: 0: 1, absent: 2; `stage1.fp`: 4: 2, absent: 1; `stage1.w`: 4: 2, 3: 1; `stage2.tags`: reduce: 1, atomic: 1, atomic,persist: 1; `stage2.tile`: 128x256x128: 2, 128x128x256: 1 |
| `cfg_66add723b5d21c57` | `M>4096` | 1 | `block_m=64`, `ksplit=0`, `run_1stage=0`, `stage1.tile=64x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=reduce`, `stage2.tile=64x256x128` | one row; nothing to compare |
| `cfg_ab07cd11619ba84a` | `M>4096` | 3 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 32: 1, 64: 1, 128: 1; `stage1.tile`: 32x32x256: 1, 64x64x256: 1, 128x128x256: 1; `stage1.w`: 4: 2, 3: 1; `stage2.sbm`: 32: 1, 64: 1, 128: 1; `stage2.tags`: reduce: 2, atomic,nt: 1; `stage2.tile`: 32x256x256: 1, 64x256x256: 1, 128x256x256: 1 |
| `cfg_31f80319ce49c12a` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 1, 64: 1; `stage1.bnt`: 0: 3, absent: 3; `stage1.fp`: 4: 2, absent: 4; `stage1.tile`: 128x128x256: 4, 32x128x256: 1, 64x128x256: 1; `stage1.w`: 3: 3, 2: 2, 4: 1; `stage1.xcd`: 4: 3, absent: 3; `stage2.sbm`: 128: 4, 32: 1, 64: 1; `stage2.tags`: persist,reduce: 3, atomic,nt: 2, reduce: 1; `stage2.tile`: 64x256x256: 4, 32x256x256: 1, 32x128x256: 1 |
| `cfg_4441863abf089912` | `M>4096` | 2 | `block_m=64`, `ksplit=0`, `run_1stage=0`, `stage1.tile=64x64x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=64`, `stage2.tags=atomic,nt,persist`, `stage2.tile=64x256x256` | one row; nothing to compare |
| `cfg_5748ff2650855764` | `M>4096` | 3 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 64: 2, 32: 1; `stage1.bnt`: 0: 1, absent: 2; `stage1.tile`: 32x64x256: 1, 64x128x256: 1, 64x64x256: 1; `stage1.w`: 4: 2, 3: 1; `stage2.sbm`: 64: 2, 32: 1; `stage2.tags`: atomic,nt: 1, atomic,nt,persist: 1, persist,reduce: 1; `stage2.tile`: 32x256x256: 1, 64x256x256: 1, 64x128x256: 1 |
| `cfg_5cab9e20d75dd79f` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 64: 3, 128: 2, 32: 1; `stage1.bnt`: 0: 2, absent: 4; `stage1.fp`: 4: 2, absent: 4; `stage1.tile`: 64x128x256: 3, 128x128x256: 2, 32x128x256: 1; `stage1.w`: 2: 3, 3: 1, 4: 1 (+1 more); `stage1.xcd`: 4: 3, absent: 3; `stage2.sbm`: 64: 3, 128: 2, 32: 1; `stage2.tags`: persist,reduce: 4, atomic,nt: 1, atomic,nt,persist: 1; `stage2.tile`: 64x256x256: 4, 32x256x256: 1, 64x128x256: 1 |
| `cfg_3c18bb097fffe394` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `xbf16=0` | `block_m`: 64: 4, 32: 2, 128: 1; `stage1.bnt`: 0: 3, absent: 4; `stage1.tile`: 64x128x256: 4, 32x32x256: 2, 128x128x256: 1; `stage1.w`: 4: 4, 3: 3; `stage1.xcd`: 4: 3, absent: 4; `stage2.bnt`: 2: 1, absent: 6; `stage2.family`: moe_stage2_layout: 6, moe_stage2: 1; `stage2.sbm`: 64: 3, 32: 2, 128: 1 (+1 more); `stage2.tags`: reduce: 3, nt,reduce: 2, atomic,nt: 1 (+1 more); `stage2.tile`: 32x128x128: 2, 64x256x256: 2, 64x256x128: 2 (+1 more); `stage2.xcd`: 4: 1, absent: 6 |
| `cfg_d479a0c7d3d01fd7` | `M>4096` | 7 | `flat=0`, `ksplit=0`, `run_1stage=0`, `stage1.fp=4`, `stage2.family=moe_stage2_layout`, `xbf16=0` | `block_m`: 64: 3, 32: 2, 128: 2; `stage1.bnt`: 0: 3, absent: 4; `stage1.kw`: 2: 2, absent: 5; `stage1.tile`: 32x32x256: 2, 128x128x256: 2, 64x128x256: 2 (+1 more); `stage1.w`: 3: 5, 4: 1, 2: 1; `stage1.xcd`: 4: 3, absent: 4; `stage2.sbm`: 64: 3, 32: 2, 128: 2; `stage2.tags`: nt,reduce: 3, reduce: 2, atomic,nt,persist: 1 (+1 more); `stage2.tile`: 64x256x128: 2, 64x256x256: 2, 32x256x128: 1 (+2 more) |
| `cfg_5c0ede1be5cab5d9` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2` | `block_m`: 64: 4, 32: 2, 128: 1; `stage1.bnt`: 0: 4, absent: 3; `stage1.tile`: 64x128x256: 3, 32x32x256: 1, 32x64x256: 1 (+2 more); `stage1.w`: 2: 3, 4: 2, 3: 2; `stage1.xcd`: 4: 4, absent: 3; `stage2.bnt`: 2: 3, absent: 4; `stage2.sbm`: 32: 1, 128: 1, absent: 5; `stage2.tags`: persist,reduce: 3, reduce: 2, atomic,persist: 1 (+1 more); `stage2.tile`: 64x256x256: 5, 16x128x256: 1, 32x128x256: 1 |
| `cfg_637cae4971b45085` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2` | `block_m`: 128: 4, 64: 2, 32: 1; `stage1.bnt`: 0: 3, absent: 4; `stage1.fp`: 4: 3, absent: 4; `stage1.tile`: 128x128x256: 4, 64x128x256: 2, 32x128x256: 1; `stage1.w`: 2: 3, 4: 2, 3: 1 (+1 more); `stage1.xcd`: 4: 3, absent: 4; `stage2.bnt`: 2: 1, absent: 6; `stage2.sbm`: 128: 4, 64: 1, absent: 2; `stage2.tags`: reduce: 3, atomic,persist: 2, persist,reduce: 2; `stage2.tile`: 64x256x256: 5, 32x128x256: 2 |
| `cfg_999fc6cb7580b82b` | `M>4096` | 6 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2` | `block_m`: 64: 3, 128: 2, 32: 1; `stage1.bnt`: 0: 1, absent: 5; `stage1.fp`: 4: 2, absent: 4; `stage1.tile`: 64x128x256: 3, 128x128x256: 2, 32x64x256: 1; `stage1.w`: 4: 3, 2: 3; `stage1.xcd`: 4: 2, absent: 4; `stage2.bnt`: 2: 2, absent: 4; `stage2.sbm`: 128: 2, 64: 1, absent: 3; `stage2.tags`: atomic: 2, persist,reduce: 2, reduce: 2; `stage2.tile`: 64x256x256: 4, 32x256x256: 2; `stage2.xcd`: 4: 1, absent: 5 |

### `` · 256 CUs · `tuned_fmoe` · FlyDSL `moe_stage1` · `afp8_wfp4_bf16`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_d778128173610686` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.bnt`: 0: 1, absent: 1; `stage1.w`: 4: 1, 3: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_c481b6e4e5a3c9e8` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.w`: 4: 1, 2: 1 |
| `cfg_fea2e2b22b51ca25` | `M<=16` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_81b2d30c483d600f` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1 |
| `cfg_5565f1d325cebae4` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_6ea27a79c301e53d` | `M17-64` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.tile`: 32x128x256: 1, 32x64x256: 1; `stage1.w`: 2: 1, 4: 1 |
| `cfg_9332df825d2f564d` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | one row; nothing to compare |
| `cfg_de928c059f626282` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.tile`: 32x64x256: 1, 32x128x256: 1; `stage1.w`: 4: 1, 2: 1 |
| `cfg_56ad2948231de16b` | `M65-256` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.w`: 2: 1, 3: 1 |
| `cfg_32fc7550337b0c56` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_014b89db9ddedca3` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.w`: 3: 1, 4: 1 |
| `cfg_6e37150a23f8092c` | `M257-1024` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | `stage1.fp`: 8: 1, absent: 1; `stage1.w`: 3: 1, 4: 1 |
| `cfg_1c76a1587db639d4` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x128x256` | `stage1.w`: 4: 1, 3: 1 |
| `cfg_245a67b20d6706e2` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_1fce577fb081f69b` | `M1025-4096` | 2 | `block_m=32`, `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage1.tile=32x128x256`, `stage1.w=3`, `stage2.family=moe_stage2_layout`, `stage2.sbm=32`, `stage2.tags=atomic,nt`, `stage2.tile=32x256x256` | one row; nothing to compare |
| `cfg_6a5b3d8e114fed53` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.bnt=2`, `stage2.family=moe_stage2` | `block_m`: 32: 1, 64: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1; `stage1.w`: 2: 1, absent: 1; `stage2.tags`: atomic,persist: 1, atomic: 1; `stage2.tile`: 32x256x256: 1, 64x256x256: 1 |
| `cfg_ff1ca05db59ee45c` | `M>4096` | 3 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt` | `block_m`: 32: 1, 64: 1, 128: 1; `stage1.tile`: 32x128x256: 1, 64x256x256: 1, 128x256x256: 1; `stage1.w`: 4: 2, 3: 1; `stage2.sbm`: 32: 1, 64: 1, 128: 1; `stage2.tile`: 32x256x256: 1, 64x256x256: 1, 64x128x256: 1 |
| `cfg_4f26d327d326ee6d` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 64: 2, 32: 1; `stage1.bnt`: 0: 4, absent: 3; `stage1.tile`: 128x256x256: 4, 64x256x256: 2, 32x128x256: 1; `stage1.w`: 2: 2, 3: 1, absent: 4; `stage1.xcd`: 4: 2, absent: 5; `stage2.sbm`: 128: 4, 64: 2, 32: 1; `stage2.tags`: reduce: 6, atomic,nt: 1; `stage2.tile`: 64x256x256: 3, 128x256x256: 3, 32x256x256: 1 |
| `cfg_408ca7fc10a4b0c8` | `M>4096` | 8 | `ksplit=0`, `run_1stage=0`, `stage1.fp=8`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 5, 32: 2, 64: 1; `stage1.bnt`: 0: 4, absent: 4; `stage1.tile`: 128x256x256: 5, 32x128x256: 2, 64x128x256: 1; `stage1.w`: 3: 5, 2: 2, absent: 1; `stage2.sbm`: 128: 5, 32: 2, 64: 1; `stage2.tags`: reduce: 6, atomic,nt: 2; `stage2.tile`: 64x256x256: 4, 32x256x256: 2, 128x256x256: 2 |
| `cfg_6320621be8c14cde` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 64: 2, 32: 1; `stage1.bnt`: 0: 4, absent: 3; `stage1.fp`: 8: 6, absent: 1; `stage1.tile`: 128x256x256: 4, 64x256x256: 2, 32x128x256: 1; `stage1.w`: 3: 3, 2: 2, absent: 2; `stage1.xcd`: 4: 1, absent: 6; `stage2.sbm`: 128: 4, 64: 2, 32: 1; `stage2.tags`: reduce: 5, atomic,nt: 2; `stage2.tile`: 64x256x256: 6, 32x256x256: 1 |
| `cfg_2c5f39d06ba4b50f` | `M>4096` | 8 | `ksplit=0`, `run_1stage=0`, `stage1.tags=gui`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 2, 64: 2; `stage1.bnt`: 0: 4, absent: 4; `stage1.fp`: 8: 7, absent: 1; `stage1.tile`: 128x256x256: 4, 32x128x256: 2, 64x256x256: 2; `stage1.w`: 3: 4, 2: 2, absent: 2; `stage2.sbm`: 128: 4, 32: 2, 64: 2; `stage2.tags`: reduce: 6, atomic,nt: 2; `stage2.tile`: 64x256x256: 6, 32x256x256: 2 |
| `cfg_0f432786fdee10ea` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage1.bnt=0`, `stage1.tags=gui`, `stage2.family=moe_stage2` | `block_m`: 128: 3, 64: 2; `stage1.fp`: 8: 4, absent: 1; `stage1.tile`: 128x256x256: 3, 64x256x256: 2; `stage1.w`: 2: 4, absent: 1; `stage2.sbm`: 128: 3, absent: 2; `stage2.tags`: atomic,persist: 4, atomic: 1; `stage2.tile`: 64x256x256: 3, 64x128x256: 2; `stage2.xcd`: 4: 1, absent: 4 |

### `` · 256 CUs · `tuned_fmoe` · FlyDSL `mxmoe_stage1` · `a4w4`

| decision ref | M bucket | shipped rows | seed candidate | vary next |
|---|---|---|---|---|
| `cfg_6f821c96ff4e7301` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_5e730001a9f96a12` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_8752ad07824524c3` | `M<=16` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_3d8285bd8a63f3ed` | `M<=16` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_536225b0083dd921` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_be436458700a0a8a` | `M17-64` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x128` | one row; nothing to compare |
| `cfg_52036931dbb0a26b` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_c54943b19a05012a` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x256x256` | `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage2.tags`: atomic,nt: 1, atomic: 1 |
| `cfg_d3715ac449f3a7f6` | `M17-64` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.xcd=4`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | `stage1.tags`: f16in,nt: 1, f16in,hpf,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1 |
| `cfg_8ccd5d9ce5749cd1` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 4: 1, absent: 1 |
| `cfg_740e69c4073abafb` | `M65-256` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_c14a29c6a0e6c170` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage1.xcd`: 2: 1, absent: 1 |
| `cfg_b492dbe333c52c02` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16` | `stage2.tags`: atomic: 1, atomic,nt: 1; `stage2.tile`: 16x128x256: 1, 16x128x128: 1 |
| `cfg_3dc17250f2bddfc2` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic` | `stage2.tile`: 16x128x128: 1, 16x256x256: 1 |
| `cfg_a8307561332a29e3` | `M65-256` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | `stage1.xcd`: 4: 1, 2: 1 |
| `cfg_2faf535e744ee445` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_c02dbee4f5bab90d` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,hpf,nt`, `stage1.xcd=2`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tile=16x128x128` | `stage1.tile`: 16x256x256: 1, 16x128x256: 1; `stage2.tags`: atomic: 1, atomic,nt: 1 |
| `cfg_446411528df9e033` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.tile`: 16x128x256: 1, 16x256x256: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.tile`: 16x128x128: 1, 16x256x256: 1 |
| `cfg_685c8ed72a80e4d7` | `M257-1024` | 2 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tile=16x128x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x128x128` | `stage1.tags`: f16in,hpf,nt: 1, f16in,nt: 1; `stage1.xcd`: 4: 1, 2: 1 |
| `cfg_1dfb43dc341f0c2a` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt` | `block_m`: 16: 1, 32: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tags`: f16in,nt: 1, nt: 1; `stage1.tile`: 16x256x256: 1, 32x128x256: 1; `stage1.xcd`: 2: 1, absent: 1; `stage2.sbm`: 16: 1, 32: 1; `stage2.tile`: 16x256x256: 1, 32x256x256: 1 |
| `cfg_a17336384a4b707c` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt` | `block_m`: 16: 1, 32: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tags`: f16in,nt: 1, nt: 1; `stage1.tile`: 16x256x256: 1, 32x128x256: 1; `stage1.xcd`: 4: 1, 2: 1; `stage2.sbm`: 16: 1, 32: 1; `stage2.tile`: 16x256x256: 1, 32x256x256: 1 |
| `cfg_9db41410cd7db010` | `M1025-4096` | 1 | `block_m=16`, `ksplit=0`, `run_1stage=0`, `stage1.tags=f16in,nt`, `stage1.tile=16x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=16`, `stage2.tags=atomic,nt`, `stage2.tile=16x256x256` | one row; nothing to compare |
| `cfg_e7c9bbe21439921b` | `M1025-4096` | 2 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout`, `stage2.tags=atomic,nt` | `block_m`: 16: 1, 32: 1; `stage1.kw`: 2: 1, absent: 1; `stage1.tags`: f16in,hpf,nt: 1, nt: 1; `stage1.tile`: 16x128x256: 1, 32x128x256: 1; `stage1.xcd`: 2: 1, 4: 1; `stage2.sbm`: 16: 1, 32: 1; `stage2.tile`: 16x128x128: 1, 32x256x256: 1 |
| `cfg_e7795e91a772d382` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 3; `stage1.kw`: 2: 2, absent: 5; `stage1.tags`: nt: 2, absent: 5; `stage1.tile`: 128x256x256: 4, 32x128x256: 2, 32x256x256: 1; `stage1.xcd`: 2: 3, absent: 4; `stage2.sbm`: 128: 4, 32: 3; `stage2.tags`: scatter: 4, atomic: 2, atomic,nt: 1; `stage2.tile`: 128x256x256: 4, 32x256x256: 3 |
| `cfg_2b5501f48803869f` | `M>4096` | 2 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.sp=102`, `stage2.tags=atomic`, `stage2.tile=128x128x128` | one row; nothing to compare |
| `cfg_d46cc7c34f04b08a` | `M>4096` | 3 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=atomic`, `stage2.tile=128x128x128` | `stage1.xcd`: 6: 1, absent: 2; `stage2.sp`: 102: 1, absent: 2 |
| `cfg_f20d4eaa52d5f3dd` | `M>4096` | 1 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.sp=102`, `stage2.tags=atomic`, `stage2.tile=128x128x128` | one row; nothing to compare |
| `cfg_923b69fea2dc241c` | `M>4096` | 5 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 1; `stage1.tags`: nt: 1, absent: 4; `stage1.tile`: 128x256x256: 4, 32x256x256: 1; `stage1.xcd`: 2: 1, absent: 4; `stage2.sbm`: 128: 4, 32: 1; `stage2.tags`: scatter: 4, nt,reduce: 1; `stage2.tile`: 128x256x256: 4, 32x256x256: 1 |
| `cfg_b1645268dcdfadd1` | `M>4096` | 4 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128` | `stage1.xcd`: 5: 1, 2: 1, absent: 2; `stage2.sp`: 801: 1, 6404: 1, absent: 2; `stage2.tags`: reduce: 2, scatter: 2; `stage2.tile`: 128x256x128: 2, 128x256x256: 2 |
| `cfg_ba3c055385b3d718` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 3; `stage1.tags`: nt: 2, absent: 5; `stage1.tile`: 128x256x256: 4, 32x128x256: 2, 32x256x256: 1; `stage1.xcd`: 2: 3, 4: 2, absent: 2; `stage2.sbm`: 128: 4, 32: 3; `stage2.tags`: scatter: 4, atomic: 2, atomic,nt: 1; `stage2.tile`: 128x256x256: 4, 32x256x256: 3 |
| `cfg_fbb53e0c8545ff4e` | `M>4096` | 4 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=scatter`, `stage2.tile=128x256x256` | one row; nothing to compare |
| `cfg_1fd379c4d16779e4` | `M>4096` | 4 | `block_m=128`, `ksplit=0`, `run_1stage=0`, `stage1.tile=128x256x256`, `stage2.family=moe_stage2_layout`, `stage2.sbm=128`, `stage2.tags=scatter`, `stage2.tile=128x256x256` | one row; nothing to compare |
| `cfg_a58025e7fb80ec76` | `M>4096` | 7 | `ksplit=0`, `run_1stage=0`, `stage2.family=moe_stage2_layout` | `block_m`: 128: 4, 32: 3; `stage1.kw`: 2: 2, absent: 5; `stage1.tags`: nt: 2, absent: 5; `stage1.tile`: 128x256x256: 4, 32x128x256: 2, 32x256x256: 1; `stage1.xcd`: 2: 5, 4: 1, absent: 1; `stage2.sbm`: 128: 4, 32: 3; `stage2.tags`: scatter: 4, atomic: 2, atomic,nt: 1; `stage2.tile`: 128x256x256: 4, 32x256x256: 3 |

## Backend AITER's tuner selected, by shape bucket

Each row counts the shipped tuned-database rows whose selected backend (`libtype`) was each implementation, per gfx, CU count, database and M bucket. It says where AITER chose a backend, not by how much: timings, the competing set and the tuner's box are not recorded, and a shape shipped in two model databases counts twice. Use it to see which implementation a FlyDSL kernel competes with in production for a case — and where FlyDSL has never been selected.

### `gfx950` · 256 CUs · `a8w8_blockscale_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_b7d6386940696a48` | `M<=16` | 373 | ck 293 · asm 40 · triton 28 · cktile 12 |
| `sel_bdf0a0ce3c910fc6` | `M17-64` | 190 | ck 154 · asm 24 · triton 8 · cktile 4 |
| `sel_bc5f94a60e7ac2a4` | `M65-256` | 403 | ck 247 · asm 134 · cktile 16 · triton 6 |
| `sel_9f87c053af503719` | `M257-1024` | 659 | ck 444 · asm 172 · cktile 43 |
| `sel_a91bc7afb96c3708` | `M1025-4096` | 495 | ck 374 · cktile 91 · asm 30 |
| `sel_b6b0e3ba26774e37` | `M>4096` | 328 | ck 198 · cktile 122 · asm 8 |

### `gfx950` · 256 CUs · `a8w8_blockscale_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_4b9bcc0f7ab9c1d0` | `M<=16` | 382 | ck 376 · cktile 6 |
| `sel_2f4edda53f68e173` | `M17-64` | 286 | ck 282 · cktile 4 |
| `sel_a0e632e90c21cdfc` | `M65-256` | 820 | ck 782 · cktile 38 |
| `sel_ee2dd068224e320e` | `M257-1024` | 1343 | ck 1275 · cktile 68 |
| `sel_b9a37194ba7c274a` | `M1025-4096` | 3826 | ck 3678 · cktile 148 |
| `sel_8b543c4558078855` | `M>4096` | 10025 | ck 8960 · cktile 1065 |

### `gfx950` · 256 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_7c8d8e410633ad1e` | `M<=16` | 351 | flydsl 274 · ck 57 · cktile 20 |
| `sel_da4768bfb4a81119` | `M17-64` | 203 | flydsl 145 · ck 45 · cktile 13 |
| `sel_75a6d5abfd89238c` | `M65-256` | 272 | flydsl 174 · ck 59 · cktile 39 |
| `sel_a7859ab04377ab13` | `M257-1024` | 247 | flydsl 136 · ck 59 · cktile 52 |
| `sel_3b5495db028ae505` | `M1025-4096` | 1011 | flydsl 882 · ck 74 · cktile 55 |
| `sel_1b5489e6d41507a3` | `M>4096` | 346 | flydsl 280 · cktile 40 · ck 26 |

### `gfx950` · 256 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_81411b8982b03045` | `M<=16` | 616 | flydsl 451 · triton 77 · asm 30 · torch 27 · skinny 23 · opus 5 · hipblaslt 3 |
| `sel_49310b1f04381a1e` | `M17-64` | 407 | flydsl 311 · triton 39 · torch 35 · asm 20 · opus 2 |
| `sel_4ff2330404509760` | `M65-256` | 683 | flydsl 541 · torch 78 · asm 42 · triton 12 · opus 10 |
| `sel_bb4f91d843dec8df` | `M257-1024` | 674 | flydsl 408 · torch 174 · asm 77 · opus 10 · triton 3 · hipblaslt 2 |
| `sel_0bf8dd0fd028cea8` | `M1025-4096` | 446 | flydsl 237 · torch 139 · asm 52 · opus 16 · hipblaslt 2 |
| `sel_c472ec443f7b30a0` | `M>4096` | 353 | torch 128 · flydsl 116 · asm 85 · opus 23 · triton 1 |

### `gfx950` · 256 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_ad6477f0eea34a82` | `M<=16` | 3 | flydsl 3 |
| `sel_3082225901e9198e` | `M<=16` | 3 | flydsl 3 |
| `sel_bc10d2fe512e1daf` | `M<=16` | 3 | flydsl 2 · asm 1 |
| `sel_fcbdd288353953da` | `M<=16` | 2 | flydsl 2 |
| `sel_bcd42f0c874e5b0b` | `M<=16` | 2 | flydsl 2 |
| `sel_a81eaf7b6506fea8` | `M<=16` | 2 | flydsl 2 |
| `sel_10daa53d5eea193d` | `M<=16` | 2 | flydsl 2 |
| `sel_e0ca4ebe3c4852a6` | `M<=16` | 2 | flydsl 2 |
| `sel_a000e467c7b169c2` | `M<=16` | 2 | flydsl 2 |
| `sel_d836ba6dd49573d2` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_5f0cc9dbf2a6322a` | `M<=16` | 2 | flydsl 2 |
| `sel_26814e1193d614e8` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_c4235b14d1a7cf63` | `M<=16` | 2 | flydsl 2 |
| `sel_36a6878891da4abc` | `M<=16` | 2 | flydsl 2 |
| `sel_f4b679825de3f8f5` | `M<=16` | 2 | flydsl 2 |
| `sel_c3216ffc658ebbc0` | `M<=16` | 2 | flydsl 2 |
| `sel_2c799b871aa926b8` | `M<=16` | 2 | flydsl 2 |
| `sel_16d3083581624062` | `M<=16` | 2 | flydsl 2 |
| `sel_7444eca9ad25626d` | `M<=16` | 1 | flydsl 1 |
| `sel_ecab27c6bf3b381c` | `M<=16` | 1 | flydsl 1 |
| `sel_283a902026aff19f` | `M<=16` | 1 | flydsl 1 |
| `sel_1cade2e8655ea47d` | `M<=16` | 1 | flydsl 1 |
| `sel_9cef356ffe5f5a06` | `M<=16` | 1 | flydsl 1 |
| `sel_b374d3858d28d10e` | `M<=16` | 1 | flydsl 1 |
| `sel_37085b37399bd284` | `M<=16` | 6 | flydsl 6 |
| `sel_09bdc5435a922633` | `M<=16` | 6 | flydsl 6 |
| `sel_29acbb76a0285be2` | `M<=16` | 3 | flydsl 3 |
| `sel_d5fa8091dbd0b20f` | `M<=16` | 3 | flydsl 3 |
| `sel_2c5d2419f8ca186c` | `M<=16` | 4 | flydsl 4 |
| `sel_4ef77e26cf0daa34` | `M<=16` | 4 | flydsl 4 |
| `sel_b8b6302dcc380f04` | `M<=16` | 2 | flydsl 2 |
| `sel_62e8efa4fcb8c272` | `M<=16` | 2 | flydsl 2 |
| `sel_81e465d91819d5c1` | `M<=16` | 4 | flydsl 4 |
| `sel_e727646f5129875e` | `M<=16` | 4 | flydsl 4 |
| `sel_94482d1717e8129d` | `M<=16` | 3 | flydsl 3 |
| `sel_7c8862bfbc68a88d` | `M<=16` | 3 | flydsl 3 |
| `sel_4529f6bf65efe093` | `M<=16` | 3 | flydsl 3 |
| `sel_f3eb683651339b41` | `M<=16` | 3 | flydsl 3 |
| `sel_7b0064dc7937d651` | `M<=16` | 6 | flydsl 6 |
| `sel_23608691f3e6166f` | `M<=16` | 6 | flydsl 6 |
| `sel_759b0896465552b9` | `M<=16` | 4 | flydsl 4 |
| `sel_9557932182f8ff6d` | `M<=16` | 4 | flydsl 4 |
| `sel_631f61cf7fbaaf49` | `M<=16` | 1 | flydsl 1 |
| `sel_17543b014f718de3` | `M<=16` | 1 | flydsl 1 |
| `sel_c0101163ca55816c` | `M<=16` | 2 | flydsl 2 |
| `sel_0f73e57cd5eae240` | `M<=16` | 2 | flydsl 2 |
| `sel_a29d4d8f4e34175b` | `M<=16` | 1 | flydsl 1 |
| `sel_2100d4bfd21f4d9a` | `M<=16` | 1 | flydsl 1 |
| `sel_69cab4f70207df5f` | `M<=16` | 2 | ck 2 |
| `sel_c51c415eb2b222e1` | `M<=16` | 2 | ck 2 |
| `sel_ea8e164df205be52` | `M<=16` | 2 | flydsl 2 |
| `sel_d0a7541907cc3a15` | `M<=16` | 2 | flydsl 2 |
| `sel_286f34dd3ac72d2a` | `M<=16` | 4 | flydsl 4 |
| `sel_b41a030640d7fe8f` | `M<=16` | 4 | flydsl 4 |
| `sel_8301acf97ddc8e30` | `M<=16` | 2 | ck 2 |
| `sel_a5a73162ed60431b` | `M<=16` | 2 | ck 2 |
| `sel_03163f25d7bcbf89` | `M<=16` | 3 | flydsl 2 · ck 1 |
| `sel_507724046b9a8af1` | `M<=16` | 3 | flydsl 2 · ck 1 |
| `sel_e0b99e022ff4c7be` | `M<=16` | 6 | flydsl 6 |
| `sel_7ebf980fe1d48c46` | `M<=16` | 6 | flydsl 6 |
| `sel_f57450bb1d873df5` | `M<=16` | 2 | flydsl 2 |
| `sel_be6658d386a17b39` | `M<=16` | 2 | flydsl 2 |
| `sel_a614ec58fca27471` | `M<=16` | 2 | flydsl 2 |
| `sel_b515f9e576103452` | `M<=16` | 2 | flydsl 2 |
| `sel_eb598886b3722411` | `M<=16` | 2 | flydsl 2 |
| `sel_027df5fb48ac1952` | `M<=16` | 2 | flydsl 2 |
| `sel_93ac800154332b48` | `M<=16` | 2 | flydsl 2 |
| `sel_7fbf2730d05b916c` | `M<=16` | 2 | flydsl 2 |
| `sel_8a588c4344696a78` | `M<=16` | 1 | flydsl 1 |
| `sel_978bbef62f922043` | `M<=16` | 2 | flydsl 2 |
| `sel_dd99ef6bf903243e` | `M<=16` | 1 | flydsl 1 |
| `sel_a3d174d5e961d390` | `M<=16` | 2 | flydsl 2 |
| `sel_9115b66a3ed96620` | `M<=16` | 2 | flydsl 2 |
| `sel_3540c142c6b91c85` | `M<=16` | 2 | flydsl 2 |
| `sel_09367805a72b2a25` | `M<=16` | 2 | flydsl 2 |
| `sel_d8f6bb64ca313229` | `M<=16` | 2 | flydsl 2 |
| `sel_23d79215ae493ee0` | `M<=16` | 2 | flydsl 2 |
| `sel_0cbcc4d1f3e8e524` | `M<=16` | 2 | flydsl 2 |
| `sel_3d553e217018c820` | `M<=16` | 2 | flydsl 2 |
| `sel_ac1be55c8e10e201` | `M17-64` | 2 | flydsl 2 |
| `sel_2e81e649e12ba587` | `M17-64` | 2 | flydsl 2 |
| `sel_2d81bf2a28add252` | `M17-64` | 8 | flydsl 6 · asm 2 |
| `sel_73e1e2666cd3c91e` | `M17-64` | 6 | flydsl 6 |
| `sel_9304915ca5ba2300` | `M17-64` | 6 | flydsl 6 |
| `sel_3035c266361aa359` | `M17-64` | 6 | flydsl 6 |
| `sel_9ae2053c2898ca41` | `M17-64` | 2 | flydsl 2 |
| `sel_205bbe6471ecfd7a` | `M17-64` | 2 | flydsl 2 |
| `sel_280aae521b9c87c4` | `M17-64` | 2 | flydsl 2 |
| `sel_ddf63fbac622241a` | `M17-64` | 2 | ck 1 · flydsl 1 |
| `sel_8cc8a818b234d80b` | `M17-64` | 2 | flydsl 2 |
| `sel_db9b14b449c45030` | `M17-64` | 2 | flydsl 2 |
| `sel_f3423d358b192f7a` | `M17-64` | 2 | flydsl 2 |
| `sel_4dbe2b6008c8f49a` | `M17-64` | 2 | flydsl 2 |
| `sel_8b6a3347b1963f0a` | `M17-64` | 2 | flydsl 2 |
| `sel_36177277247362f8` | `M17-64` | 2 | flydsl 2 |
| `sel_79331b3725ceac09` | `M17-64` | 2 | flydsl 2 |
| `sel_7159c69637f64064` | `M17-64` | 2 | flydsl 2 |
| `sel_96812e4c43529eff` | `M17-64` | 2 | flydsl 2 |
| `sel_a149eb7e01d4e68e` | `M17-64` | 2 | flydsl 2 |
| `sel_0b681b0771fd60da` | `M17-64` | 2 | asm 2 |
| `sel_478b08fb118fff46` | `M17-64` | 1 | asm 1 |
| `sel_902ecebad2a1e077` | `M17-64` | 2 | flydsl 2 |
| `sel_2725fb9b6ffb8232` | `M17-64` | 2 | flydsl 2 |
| `sel_a1e17c8054c143d2` | `M17-64` | 2 | flydsl 2 |
| `sel_8e98d051771d58c8` | `M17-64` | 2 | flydsl 2 |
| `sel_bcd81882020eed19` | `M17-64` | 4 | flydsl 4 |
| `sel_3c7a1fd48ddf7c65` | `M17-64` | 4 | flydsl 4 |
| `sel_28f841b87e7da284` | `M17-64` | 2 | flydsl 2 |
| `sel_1af3e2f5dca84539` | `M17-64` | 2 | flydsl 2 |
| `sel_ad2d0e2cd4cacf51` | `M17-64` | 4 | flydsl 4 |
| `sel_9cd115eb0e648900` | `M17-64` | 4 | flydsl 4 |
| `sel_8002411fd4154c63` | `M17-64` | 2 | flydsl 2 |
| `sel_3c3f7ab83307ca56` | `M17-64` | 2 | flydsl 2 |
| `sel_69a1118e0838121e` | `M17-64` | 2 | flydsl 2 |
| `sel_15fa2d55d84ff7be` | `M17-64` | 2 | flydsl 2 |
| `sel_ea18f6abe5713cb7` | `M17-64` | 2 | flydsl 2 |
| `sel_248e2cb685b10414` | `M17-64` | 2 | flydsl 2 |
| `sel_b91c5d02198ee014` | `M17-64` | 3 | flydsl 3 |
| `sel_547ee4e595891e21` | `M17-64` | 3 | flydsl 3 |
| `sel_acae37edcbf4321c` | `M17-64` | 4 | flydsl 4 |
| `sel_21b8e17c4fc951d3` | `M17-64` | 4 | flydsl 4 |
| `sel_6a85ddc11611efab` | `M17-64` | 4 | flydsl 4 |
| `sel_733cd619cfa71186` | `M17-64` | 4 | flydsl 4 |
| `sel_15da7fa4243a14d4` | `M17-64` | 2 | flydsl 2 |
| `sel_dbba96e77743b0bf` | `M17-64` | 2 | flydsl 2 |
| `sel_f557284997be691a` | `M17-64` | 2 | flydsl 2 |
| `sel_6a560798919c5cc7` | `M17-64` | 2 | flydsl 2 |
| `sel_1a4be03bbccd3f33` | `M17-64` | 2 | flydsl 2 |
| `sel_1e9b66f63ff1917b` | `M17-64` | 2 | flydsl 2 |
| `sel_7dd59fb7362fce20` | `M17-64` | 1 | flydsl 1 |
| `sel_8ad85eaa2872cf8d` | `M17-64` | 1 | flydsl 1 |
| `sel_656c73dc67332e9b` | `M17-64` | 2 | ck 2 |
| `sel_ab15e6f09a0be626` | `M17-64` | 2 | ck 2 |
| `sel_f1d2b1965b9a3f82` | `M17-64` | 4 | flydsl 4 |
| `sel_63874f80d60efd9e` | `M17-64` | 4 | flydsl 4 |
| `sel_5cfe86d14449d9ca` | `M17-64` | 4 | flydsl 4 |
| `sel_73bd1a1f8b03b6e1` | `M17-64` | 4 | flydsl 4 |
| `sel_fbeffd67e0e45ed1` | `M17-64` | 1 | ck 1 |
| `sel_1b3b703a38b64f04` | `M17-64` | 1 | ck 1 |
| `sel_e03f3d72a204d9f2` | `M17-64` | 6 | flydsl 4 · ck 2 |
| `sel_9ba5a83cdbb28f04` | `M17-64` | 6 | flydsl 4 · ck 2 |
| `sel_007f05b1e8aa752f` | `M17-64` | 6 | flydsl 6 |
| `sel_21223c4c4c53d4f4` | `M17-64` | 6 | flydsl 6 |
| `sel_708c29d6bc0196f9` | `M17-64` | 2 | flydsl 2 |
| `sel_e2dd97e7dcec8c2e` | `M17-64` | 2 | flydsl 2 |
| `sel_58dd27f7f59a810c` | `M17-64` | 2 | flydsl 2 |
| `sel_0d557c60578063eb` | `M17-64` | 2 | flydsl 2 |
| `sel_da36eb8a2152a9f4` | `M17-64` | 2 | flydsl 2 |
| `sel_51531e89c8cf32ba` | `M17-64` | 2 | flydsl 2 |
| `sel_534a01fc829fc68b` | `M17-64` | 2 | flydsl 2 |
| `sel_ca3f599f654f6d43` | `M17-64` | 2 | flydsl 2 |
| `sel_88bc04a1f2044580` | `M17-64` | 2 | flydsl 2 |
| `sel_2e17b1f9d24af535` | `M17-64` | 2 | flydsl 2 |
| `sel_5f3a884d69e1377d` | `M17-64` | 2 | flydsl 2 |
| `sel_99373caf79dcd36c` | `M17-64` | 2 | flydsl 2 |
| `sel_17972cd2f954da54` | `M17-64` | 2 | flydsl 2 |
| `sel_597260c4d1832407` | `M17-64` | 2 | flydsl 2 |
| `sel_64d6c7fe2ee3ca0a` | `M17-64` | 2 | flydsl 2 |
| `sel_a09bc5355d77fb6c` | `M17-64` | 2 | flydsl 2 |
| `sel_a93b10220c1ce8dc` | `M17-64` | 2 | flydsl 2 |
| `sel_40fa699eca902864` | `M17-64` | 2 | flydsl 2 |
| `sel_fe1a09d476a29031` | `M17-64` | 2 | flydsl 2 |
| `sel_9abace0a502c3775` | `M17-64` | 2 | flydsl 2 |
| `sel_e1f8d84d8c35611e` | `M65-256` | 2 | flydsl 2 |
| `sel_4499f7975c195d9a` | `M65-256` | 2 | flydsl 2 |
| `sel_4e785eb092b1f029` | `M65-256` | 1 | flydsl 1 |
| `sel_faaaad958ce2d7db` | `M65-256` | 1 | flydsl 1 |
| `sel_f019c88200b738e3` | `M65-256` | 6 | flydsl 4 · asm 2 |
| `sel_ec5bc62ee9ea15de` | `M65-256` | 2 | flydsl 2 |
| `sel_e80b8810cbd74cd3` | `M65-256` | 4 | flydsl 4 |
| `sel_86c42c0be2c38ab0` | `M65-256` | 3 | flydsl 3 |
| `sel_47160b736c9969d0` | `M65-256` | 2 | flydsl 2 |
| `sel_4c839f07c7f99a37` | `M65-256` | 2 | flydsl 2 |
| `sel_34e12257df3ddfa6` | `M65-256` | 2 | flydsl 2 |
| `sel_c862b517698273e1` | `M65-256` | 2 | flydsl 2 |
| `sel_9cb984a97c4e2ece` | `M65-256` | 2 | flydsl 2 |
| `sel_db8570a24ea1bbdc` | `M65-256` | 2 | flydsl 2 |
| `sel_1d41d69f004910ab` | `M65-256` | 2 | flydsl 2 |
| `sel_8f4b21eb37779558` | `M65-256` | 2 | flydsl 2 |
| `sel_309ac19ca1e89eb8` | `M65-256` | 2 | flydsl 2 |
| `sel_45856e2dcdcd9894` | `M65-256` | 1 | flydsl 1 |
| `sel_05c9093b3a59630b` | `M65-256` | 2 | flydsl 2 |
| `sel_5a2772191209d01c` | `M65-256` | 2 | flydsl 2 |
| `sel_3eebe0fa23412932` | `M65-256` | 1 | asm 1 |
| `sel_f5181296cd9d22ab` | `M65-256` | 2 | flydsl 2 |
| `sel_b6e6a1aee8552684` | `M65-256` | 2 | flydsl 2 |
| `sel_bb9daaccdb29bca3` | `M65-256` | 1 | asm 1 |
| `sel_015f9ee4060d7501` | `M65-256` | 2 | flydsl 2 |
| `sel_13492c3140b0b9b8` | `M65-256` | 2 | flydsl 2 |
| `sel_026ff23299d1c60f` | `M65-256` | 2 | flydsl 2 |
| `sel_8665c425659abe04` | `M65-256` | 2 | flydsl 2 |
| `sel_4730ec8007381f14` | `M65-256` | 4 | flydsl 4 |
| `sel_aee3a36817913a8d` | `M65-256` | 4 | flydsl 4 |
| `sel_750e045aa069a024` | `M65-256` | 2 | flydsl 2 |
| `sel_511d613c62eae2d6` | `M65-256` | 2 | flydsl 2 |
| `sel_0560bb8386bf3c0a` | `M65-256` | 4 | flydsl 4 |
| `sel_9425064708196332` | `M65-256` | 4 | flydsl 4 |
| `sel_adb2ebe710eed94a` | `M65-256` | 2 | flydsl 2 |
| `sel_b1e4251663e22fa1` | `M65-256` | 2 | flydsl 2 |
| `sel_601c9de6e536246a` | `M65-256` | 2 | flydsl 2 |
| `sel_fcf3c7c39d1349ce` | `M65-256` | 2 | flydsl 2 |
| `sel_3e3e523d9d260fab` | `M65-256` | 2 | flydsl 2 |
| `sel_60804916c77582d2` | `M65-256` | 2 | flydsl 2 |
| `sel_19ab01360fe02e71` | `M65-256` | 4 | flydsl 4 |
| `sel_51238bc95ef8e4df` | `M65-256` | 4 | flydsl 4 |
| `sel_c3fdc6217d1458dd` | `M65-256` | 4 | flydsl 4 |
| `sel_83923bae9c09b275` | `M65-256` | 4 | flydsl 4 |
| `sel_9a7c077f718b1239` | `M65-256` | 4 | flydsl 4 |
| `sel_fef77a347b4ec456` | `M65-256` | 4 | flydsl 4 |
| `sel_8c47c6ec3042caa7` | `M65-256` | 2 | flydsl 2 |
| `sel_c9813890d9643fa1` | `M65-256` | 2 | flydsl 2 |
| `sel_943edfe02496134c` | `M65-256` | 2 | flydsl 2 |
| `sel_6b027861cdd6b8e5` | `M65-256` | 2 | flydsl 2 |
| `sel_e34ccf55fe8b854e` | `M65-256` | 2 | flydsl 2 |
| `sel_0b2ce275691ed6d3` | `M65-256` | 2 | flydsl 2 |
| `sel_8e6baf58726689c7` | `M65-256` | 2 | flydsl 2 |
| `sel_e4886ae2a84394a3` | `M65-256` | 2 | flydsl 2 |
| `sel_402c421d314417cf` | `M65-256` | 4 | flydsl 4 |
| `sel_d15bfc41e23f77b2` | `M65-256` | 4 | flydsl 4 |
| `sel_37c028040e8c4e35` | `M65-256` | 4 | flydsl 4 |
| `sel_4738aa0bd93af912` | `M65-256` | 4 | flydsl 4 |
| `sel_bc58021da983b25c` | `M65-256` | 6 | flydsl 4 · ck 2 |
| `sel_8e4a061214a6b9f9` | `M65-256` | 6 | flydsl 4 · ck 2 |
| `sel_b086034475f1decf` | `M65-256` | 6 | flydsl 6 |
| `sel_1179feb7aa68de6a` | `M65-256` | 6 | flydsl 6 |
| `sel_d02bcfebe867153a` | `M65-256` | 2 | flydsl 2 |
| `sel_9db602901eb320e0` | `M65-256` | 2 | flydsl 2 |
| `sel_c0f847f9bb61a268` | `M65-256` | 2 | flydsl 2 |
| `sel_c5d82f54789887f1` | `M65-256` | 2 | flydsl 2 |
| `sel_177c86161e0a659e` | `M65-256` | 2 | flydsl 2 |
| `sel_7788bebcbadabf73` | `M65-256` | 2 | flydsl 2 |
| `sel_26faa388ad7a24ab` | `M65-256` | 2 | flydsl 2 |
| `sel_e59c7583b83592e1` | `M65-256` | 2 | flydsl 2 |
| `sel_06ffff60c58d67b2` | `M65-256` | 2 | flydsl 2 |
| `sel_f8acbebf2e6bf3ec` | `M65-256` | 2 | flydsl 2 |
| `sel_7e69e955ce146607` | `M65-256` | 2 | flydsl 2 |
| `sel_c1f18ec51c26a1c8` | `M65-256` | 2 | flydsl 2 |
| `sel_81825922fc9dec25` | `M65-256` | 2 | flydsl 2 |
| `sel_39f6aa8b691c8252` | `M65-256` | 2 | flydsl 2 |
| `sel_d4b48d1b88f28c33` | `M65-256` | 2 | flydsl 2 |
| `sel_75af67f44fb50cd4` | `M65-256` | 2 | flydsl 2 |
| `sel_4d30b64634339558` | `M65-256` | 2 | flydsl 2 |
| `sel_541e0cd1bb499c40` | `M65-256` | 2 | flydsl 2 |
| `sel_d917cbb49cae1975` | `M257-1024` | 2 | flydsl 2 |
| `sel_1c64cb5477e988f9` | `M257-1024` | 2 | flydsl 2 |
| `sel_8d6b2f215f121df0` | `M257-1024` | 2 | flydsl 2 |
| `sel_9b904df5ae29bab6` | `M257-1024` | 6 | flydsl 5 · asm 1 |
| `sel_9c2d1aab0caa33e0` | `M257-1024` | 3 | flydsl 3 |
| `sel_cc180114bf3cca3f` | `M257-1024` | 4 | flydsl 4 |
| `sel_2ce91024106cb768` | `M257-1024` | 2 | flydsl 2 |
| `sel_1160ea2de35b42e7` | `M257-1024` | 2 | flydsl 2 |
| `sel_037d77d2d8e69ef4` | `M257-1024` | 2 | flydsl 2 |
| `sel_485359db53914d9e` | `M257-1024` | 2 | flydsl 2 |
| `sel_3654761515c695a6` | `M257-1024` | 2 | flydsl 2 |
| `sel_317f4eb39c9d8a34` | `M257-1024` | 2 | flydsl 2 |
| `sel_226b2c94dca5aa13` | `M257-1024` | 2 | flydsl 2 |
| `sel_879a404f5d62c16b` | `M257-1024` | 2 | flydsl 2 |
| `sel_f903f62f34cdb497` | `M257-1024` | 2 | flydsl 2 |
| `sel_5d3cd6907bf003f9` | `M257-1024` | 2 | flydsl 2 |
| `sel_9e425a4755356008` | `M257-1024` | 1 | flydsl 1 |
| `sel_427026dfe40db717` | `M257-1024` | 2 | flydsl 2 |
| `sel_4a4cb6f6f2b9c5ec` | `M257-1024` | 2 | flydsl 2 |
| `sel_51b6ef0b72b24eae` | `M257-1024` | 1 | asm 1 |
| `sel_4e041de2857c1be2` | `M257-1024` | 2 | flydsl 2 |
| `sel_4c711934e774dd46` | `M257-1024` | 2 | flydsl 2 |
| `sel_06eb2e8b4242e728` | `M257-1024` | 2 | flydsl 2 |
| `sel_e98e8fe4059f4832` | `M257-1024` | 2 | flydsl 2 |
| `sel_e502d39145965790` | `M257-1024` | 2 | flydsl 2 |
| `sel_a29ab2a165defcd8` | `M257-1024` | 2 | flydsl 2 |
| `sel_311bcf957560af94` | `M257-1024` | 4 | flydsl 4 |
| `sel_4a8c80a083b9074f` | `M257-1024` | 4 | flydsl 4 |
| `sel_d01522ab1c0a0aaa` | `M257-1024` | 2 | flydsl 2 |
| `sel_09b5e738f01d7b52` | `M257-1024` | 2 | flydsl 2 |
| `sel_5b54407a4c1618a7` | `M257-1024` | 4 | flydsl 4 |
| `sel_2d40e1db7934dd7b` | `M257-1024` | 4 | flydsl 4 |
| `sel_1f38900fadf924af` | `M257-1024` | 2 | flydsl 2 |
| `sel_473e58ef6bc95d4f` | `M257-1024` | 2 | flydsl 2 |
| `sel_50181ace30540429` | `M257-1024` | 2 | flydsl 2 |
| `sel_3f1c61fe5951138a` | `M257-1024` | 2 | flydsl 2 |
| `sel_87fa58d89522c6f1` | `M257-1024` | 2 | flydsl 2 |
| `sel_ea56b89d991f4fb3` | `M257-1024` | 2 | flydsl 2 |
| `sel_8d65a190fad7e3d0` | `M257-1024` | 4 | flydsl 4 |
| `sel_42171eac6020f236` | `M257-1024` | 4 | flydsl 4 |
| `sel_e7ea1ddfa014c316` | `M257-1024` | 4 | flydsl 4 |
| `sel_cc2465b0aa99bc36` | `M257-1024` | 4 | flydsl 4 |
| `sel_7675781bfaaf428c` | `M257-1024` | 4 | flydsl 4 |
| `sel_114ebaa717d9e792` | `M257-1024` | 4 | flydsl 4 |
| `sel_4b9ce362a4b89971` | `M257-1024` | 2 | flydsl 2 |
| `sel_74c7b655eee6d7d4` | `M257-1024` | 2 | flydsl 2 |
| `sel_52deb625281b962e` | `M257-1024` | 2 | flydsl 2 |
| `sel_160dc432448c31ad` | `M257-1024` | 2 | flydsl 2 |
| `sel_a44a98abd0778f96` | `M257-1024` | 2 | flydsl 2 |
| `sel_3d50a66c2e2df2a0` | `M257-1024` | 2 | flydsl 2 |
| `sel_9e935425ba662386` | `M257-1024` | 1 | flydsl 1 |
| `sel_68add4f1d12fdcc0` | `M257-1024` | 1 | flydsl 1 |
| `sel_fe72fe94703a1a51` | `M257-1024` | 4 | flydsl 4 |
| `sel_e1ce017ad2cec4b7` | `M257-1024` | 4 | flydsl 4 |
| `sel_7df71f2db75f42be` | `M257-1024` | 4 | flydsl 4 |
| `sel_298cc545367c4f8b` | `M257-1024` | 4 | flydsl 4 |
| `sel_ad05caee1382f420` | `M257-1024` | 4 | flydsl 4 |
| `sel_fb5a5f87d52d685f` | `M257-1024` | 4 | flydsl 4 |
| `sel_efa1e8391b2efc98` | `M257-1024` | 6 | flydsl 6 |
| `sel_3ef185f6d04fc26f` | `M257-1024` | 6 | flydsl 6 |
| `sel_749260f5caeb74a1` | `M257-1024` | 2 | flydsl 2 |
| `sel_b18ccbcc08403949` | `M257-1024` | 2 | flydsl 2 |
| `sel_58f460bb287ef620` | `M257-1024` | 2 | flydsl 2 |
| `sel_67d1b1cb27e6ce79` | `M257-1024` | 2 | flydsl 2 |
| `sel_447b7d50c6e0a639` | `M257-1024` | 2 | flydsl 2 |
| `sel_7b985e94fd965d62` | `M257-1024` | 2 | flydsl 2 |
| `sel_5523291647baf0a3` | `M257-1024` | 2 | flydsl 2 |
| `sel_5cdde282c807d285` | `M257-1024` | 1 | flydsl 1 |
| `sel_d614a6eb980beabc` | `M257-1024` | 2 | flydsl 2 |
| `sel_725f118250d78236` | `M257-1024` | 2 | flydsl 2 |
| `sel_0443d4fc23cec510` | `M257-1024` | 2 | flydsl 2 |
| `sel_80aa2f84bea9221b` | `M257-1024` | 1 | flydsl 1 |
| `sel_130805c4d1fb12c5` | `M257-1024` | 2 | flydsl 2 |
| `sel_a3671648d0f6bc3a` | `M257-1024` | 2 | flydsl 2 |
| `sel_481efcf2ec08c05f` | `M257-1024` | 2 | flydsl 2 |
| `sel_6bd04068c351cd25` | `M257-1024` | 2 | flydsl 2 |
| `sel_1b8059fedd41b400` | `M257-1024` | 2 | flydsl 2 |
| `sel_136d530f06301c11` | `M257-1024` | 2 | flydsl 2 |
| `sel_bcb1a3c341411251` | `M1025-4096` | 2 | flydsl 2 |
| `sel_452f3561b229959b` | `M1025-4096` | 2 | flydsl 2 |
| `sel_3eda7f940eb0aea7` | `M1025-4096` | 2 | flydsl 2 |
| `sel_158a8fabcf7f7b6a` | `M1025-4096` | 2 | flydsl 2 |
| `sel_963b9172a9ee8dfb` | `M1025-4096` | 6 | flydsl 6 |
| `sel_29ff53e4ea8af61f` | `M1025-4096` | 5 | flydsl 5 |
| `sel_37bbe5e5877ec90b` | `M1025-4096` | 4 | flydsl 4 |
| `sel_a521751cb2b939a8` | `M1025-4096` | 2 | flydsl 2 |
| `sel_e8a75a4d021aae34` | `M1025-4096` | 2 | flydsl 2 |
| `sel_31df1cbeb3f92eb5` | `M1025-4096` | 2 | flydsl 2 |
| `sel_b7fc80dec9baacde` | `M1025-4096` | 2 | flydsl 2 |
| `sel_62ed54cd669e71f2` | `M1025-4096` | 2 | flydsl 2 |
| `sel_45ba01c4bc6d08a7` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c3a27094ad33a3ad` | `M1025-4096` | 2 | flydsl 2 |
| `sel_320d4ff8b7f4b807` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c5906023105107b9` | `M1025-4096` | 2 | flydsl 2 |
| `sel_3ff3bc5a39711279` | `M1025-4096` | 2 | flydsl 2 |
| `sel_d68cbdb26e151e07` | `M1025-4096` | 2 | flydsl 2 |
| `sel_e6cd9da73e292530` | `M1025-4096` | 2 | flydsl 2 |
| `sel_6e13b62c9be20b68` | `M1025-4096` | 2 | flydsl 2 |
| `sel_8a76bafc592c53a9` | `M1025-4096` | 2 | asm 2 |
| `sel_16013ed6060d4bae` | `M1025-4096` | 2 | flydsl 2 |
| `sel_682440135a19df36` | `M1025-4096` | 2 | flydsl 2 |
| `sel_25bf689a5ed38ec6` | `M1025-4096` | 2 | flydsl 2 |
| `sel_7475d1756245cccd` | `M1025-4096` | 2 | flydsl 2 |
| `sel_42b0b4cf712c086e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_eeb90186c6400644` | `M1025-4096` | 2 | flydsl 2 |
| `sel_aaa116d064ee4518` | `M1025-4096` | 4 | flydsl 4 |
| `sel_3c761823dfd601e3` | `M1025-4096` | 4 | flydsl 4 |
| `sel_a671c2c52280811a` | `M1025-4096` | 1 | flydsl 1 |
| `sel_b0b060177d1828af` | `M1025-4096` | 1 | flydsl 1 |
| `sel_ce85ec256abaad58` | `M1025-4096` | 4 | flydsl 4 |
| `sel_70c85dad105b3e99` | `M1025-4096` | 4 | flydsl 4 |
| `sel_dabba09c677bc5f3` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c5445b54fe237f35` | `M1025-4096` | 2 | flydsl 2 |
| `sel_81eb2d007c66141e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_5eccac7e963c521c` | `M1025-4096` | 2 | flydsl 2 |
| `sel_dfff564880ac4684` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c1fa36ff88c49bfe` | `M1025-4096` | 2 | flydsl 2 |
| `sel_42c8d331a0f63a6f` | `M1025-4096` | 4 | flydsl 4 |
| `sel_851eac9af7a8f9b9` | `M1025-4096` | 4 | flydsl 4 |
| `sel_0364e1c20ceb6d06` | `M1025-4096` | 4 | flydsl 4 |
| `sel_725b0f00c775c015` | `M1025-4096` | 4 | flydsl 4 |
| `sel_8124f9c96115aef8` | `M1025-4096` | 4 | flydsl 4 |
| `sel_9482d86a872016a0` | `M1025-4096` | 4 | flydsl 4 |
| `sel_ff0f4c560cbab1de` | `M1025-4096` | 2 | flydsl 2 |
| `sel_d8bf4cc70ab78836` | `M1025-4096` | 2 | flydsl 2 |
| `sel_1de2ccf16560f53b` | `M1025-4096` | 2 | flydsl 2 |
| `sel_f5a3009226f6c340` | `M1025-4096` | 2 | flydsl 2 |
| `sel_bd89ba9caad68802` | `M1025-4096` | 2 | flydsl 2 |
| `sel_aff8f0f9b2fd0a8e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_0948fcd17b9aaec5` | `M1025-4096` | 1 | ck 1 |
| `sel_bfef4c0574616a4f` | `M1025-4096` | 1 | ck 1 |
| `sel_8046f3b4f7d84725` | `M1025-4096` | 4 | flydsl 4 |
| `sel_761f08b152d20f36` | `M1025-4096` | 4 | flydsl 4 |
| `sel_0b0b3c1e63456588` | `M1025-4096` | 4 | flydsl 4 |
| `sel_373db1eff1cc703d` | `M1025-4096` | 4 | flydsl 4 |
| `sel_b800ad4861e1e5da` | `M1025-4096` | 1 | ck 1 |
| `sel_827f4b0c5fd38300` | `M1025-4096` | 1 | ck 1 |
| `sel_0283281771d2b0d4` | `M1025-4096` | 5 | flydsl 4 · ck 1 |
| `sel_bd3637ead220616b` | `M1025-4096` | 5 | flydsl 4 · ck 1 |
| `sel_74bb7324fb7666b5` | `M1025-4096` | 5 | flydsl 5 |
| `sel_6d5bd395b8877236` | `M1025-4096` | 5 | flydsl 5 |
| `sel_6204f5c2754d9e7f` | `M1025-4096` | 2 | flydsl 2 |
| `sel_eaa8ab9ecc151bbd` | `M1025-4096` | 2 | flydsl 2 |
| `sel_f18abb06e22abb47` | `M1025-4096` | 2 | flydsl 2 |
| `sel_16662fad9adaa0b0` | `M1025-4096` | 2 | flydsl 2 |
| `sel_9ad98c675d93bf2f` | `M1025-4096` | 2 | flydsl 2 |
| `sel_11438dcb13067bc9` | `M1025-4096` | 1 | flydsl 1 |
| `sel_e327294968a07af2` | `M1025-4096` | 1 | flydsl 1 |
| `sel_5116f2db8a95a44f` | `M1025-4096` | 2 | flydsl 2 |
| `sel_73e126d4c0735672` | `M1025-4096` | 2 | flydsl 2 |
| `sel_d7f26f60fced3c5f` | `M1025-4096` | 1 | flydsl 1 |
| `sel_37be1a59d09e17db` | `M1025-4096` | 2 | flydsl 2 |
| `sel_1b9e8c0102d760d9` | `M1025-4096` | 2 | flydsl 2 |
| `sel_3bd84038ce567cb4` | `M1025-4096` | 2 | flydsl 2 |
| `sel_3a5dd9f3e4fa6205` | `M1025-4096` | 2 | flydsl 2 |
| `sel_71d2c9e7a0d5b5cf` | `M1025-4096` | 2 | flydsl 2 |
| `sel_1f00f28142044910` | `M1025-4096` | 2 | flydsl 2 |
| `sel_5fdf634a18f78cc2` | `M>4096` | 5 | flydsl 5 |
| `sel_891a53a8f7fdd9fd` | `M>4096` | 5 | flydsl 5 |
| `sel_0c42e001a0bd0ef3` | `M>4096` | 8 | flydsl 8 |
| `sel_ddc9538ff12539f7` | `M>4096` | 8 | flydsl 8 |
| `sel_93f394641d434992` | `M>4096` | 1 | flydsl 1 |
| `sel_b873fe9455b55e0b` | `M>4096` | 1 | flydsl 1 |
| `sel_42af36f6125dc0cb` | `M>4096` | 20 | flydsl 20 |
| `sel_7a20ab2b99066c02` | `M>4096` | 13 | flydsl 13 |
| `sel_fcc6d5f8946ce19f` | `M>4096` | 11 | flydsl 11 |
| `sel_2d342f81bd272d89` | `M>4096` | 9 | flydsl 9 |
| `sel_6fa4a30bfda25e9d` | `M>4096` | 6 | flydsl 6 |
| `sel_4225f3148c9a79b3` | `M>4096` | 6 | flydsl 6 |
| `sel_2ddca673ffab6c62` | `M>4096` | 6 | flydsl 6 |
| `sel_05417b68fa6f0802` | `M>4096` | 6 | flydsl 6 |
| `sel_3a6d311cfb3e0813` | `M>4096` | 6 | flydsl 6 |
| `sel_734bf6481bdf6217` | `M>4096` | 6 | flydsl 6 |
| `sel_c6f5c4d74c84b531` | `M>4096` | 6 | flydsl 6 |
| `sel_b0476035090f418c` | `M>4096` | 6 | flydsl 6 |
| `sel_de9927bfe4a434aa` | `M>4096` | 6 | flydsl 6 |
| `sel_f491051a79d9792a` | `M>4096` | 6 | flydsl 6 |
| `sel_5d787b572c52a292` | `M>4096` | 7 | flydsl 7 |
| `sel_2329d4486188e731` | `M>4096` | 7 | flydsl 7 |
| `sel_787c2ba3ca44b63d` | `M>4096` | 5 | asm 5 |
| `sel_e537007265d52898` | `M>4096` | 7 | flydsl 7 |
| `sel_a296d75be32a52d2` | `M>4096` | 7 | flydsl 7 |
| `sel_51b065491f0ae54b` | `M>4096` | 4 | asm 4 |
| `sel_6b3cc526e574ae56` | `M>4096` | 5 | asm 5 |
| `sel_ce5d300bce655ae2` | `M>4096` | 7 | flydsl 7 |
| `sel_46bdb2b22d8ee7be` | `M>4096` | 7 | flydsl 7 |
| `sel_839cb88cb6bad493` | `M>4096` | 7 | flydsl 7 |
| `sel_599ba6206c5db22c` | `M>4096` | 7 | flydsl 7 |
| `sel_67bed17f9d512ebc` | `M>4096` | 10 | flydsl 10 |
| `sel_8915c9af7d2789f0` | `M>4096` | 10 | flydsl 10 |
| `sel_3367722d627995c5` | `M>4096` | 12 | flydsl 12 |
| `sel_66b666d78f1de284` | `M>4096` | 12 | flydsl 12 |
| `sel_95880ac06929adfe` | `M>4096` | 3 | flydsl 3 |
| `sel_b3f04af27dcacaf0` | `M>4096` | 3 | flydsl 3 |
| `sel_6346fa830982f695` | `M>4096` | 4 | flydsl 4 |
| `sel_57dedcc60c958388` | `M>4096` | 4 | flydsl 4 |
| `sel_dcb17215169d06a3` | `M>4096` | 5 | flydsl 5 |
| `sel_3f9ef9bb19c9406e` | `M>4096` | 5 | flydsl 5 |
| `sel_63c5b89efa48e1ca` | `M>4096` | 11 | flydsl 11 |
| `sel_3cde15b59278a6a5` | `M>4096` | 11 | flydsl 11 |
| `sel_d987f3589de9e4d4` | `M>4096` | 10 | flydsl 10 |
| `sel_d4737fd0430c351c` | `M>4096` | 10 | flydsl 10 |
| `sel_f3adb3938fd79427` | `M>4096` | 12 | flydsl 12 |
| `sel_b5c1e6390e2cac75` | `M>4096` | 12 | flydsl 12 |
| `sel_c9888417c21e3e81` | `M>4096` | 7 | flydsl 7 |
| `sel_a5cd965f8e1f87a9` | `M>4096` | 7 | flydsl 7 |
| `sel_1887f91f7c39263d` | `M>4096` | 6 | flydsl 6 |
| `sel_5930f452033dc2a6` | `M>4096` | 6 | flydsl 6 |
| `sel_d995f89db8a33c41` | `M>4096` | 7 | flydsl 7 |
| `sel_d4ad4ddcbfe31066` | `M>4096` | 7 | flydsl 7 |
| `sel_eb417b127737e95a` | `M>4096` | 7 | ck 7 |
| `sel_b679e11c89644a60` | `M>4096` | 7 | ck 7 |
| `sel_6e5bea21c531ca0b` | `M>4096` | 14 | flydsl 14 |
| `sel_71c693f9258640e7` | `M>4096` | 14 | flydsl 14 |
| `sel_8a95c3a903cb2605` | `M>4096` | 7 | flydsl 7 |
| `sel_ccbec0a95f807fc0` | `M>4096` | 12 | flydsl 12 |
| `sel_8b20f6b90bc418b8` | `M>4096` | 12 | flydsl 12 |
| `sel_7255e583c55e4c14` | `M>4096` | 7 | ck 7 |
| `sel_af5aef8581a28527` | `M>4096` | 7 | ck 7 |
| `sel_ee02cc5a936247d5` | `M>4096` | 19 | flydsl 14 · ck 5 |
| `sel_b2ec4950a6ca0fd6` | `M>4096` | 19 | flydsl 14 · ck 5 |
| `sel_88f9d62972f016c3` | `M>4096` | 12 | flydsl 12 |
| `sel_3874bb94c2d97a0c` | `M>4096` | 12 | flydsl 12 |
| `sel_0d01da759a0981d9` | `M>4096` | 6 | flydsl 6 |
| `sel_dea61bfd4dbdca9b` | `M>4096` | 6 | flydsl 6 |
| `sel_1c99974f37086e16` | `M>4096` | 7 | flydsl 7 |
| `sel_d3d4fe9b2d076a17` | `M>4096` | 7 | flydsl 7 |
| `sel_ffc0eb2a0de45fb4` | `M>4096` | 7 | flydsl 7 |
| `sel_c78189da211e0a7e` | `M>4096` | 7 | flydsl 7 |
| `sel_a1280864c56dd7ad` | `M>4096` | 1 | flydsl 1 |
| `sel_09b3ff825c9d5780` | `M>4096` | 6 | flydsl 6 |
| `sel_6fe724ef6b1c88d3` | `M>4096` | 6 | flydsl 6 |
| `sel_5df12e57dcbedede` | `M>4096` | 6 | flydsl 6 |
| `sel_a6734ac98371eff8` | `M>4096` | 6 | flydsl 6 |
| `sel_c9b67e8c629faba0` | `M>4096` | 6 | flydsl 6 |
| `sel_658a15b30212cce0` | `M>4096` | 6 | flydsl 6 |
| `sel_f0df3d34220a03c1` | `M>4096` | 6 | flydsl 6 |

### `gfx942` · 80 CUs · `a8w8_blockscale_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_f7e394faa23e37e6` | `M<=16` | 72 | ck 66 · triton 4 · asm 2 |
| `sel_abae8b787c473637` | `M17-64` | 66 | ck 39 · asm 27 |
| `sel_989b9db8a18a0bb0` | `M65-256` | 144 | asm 93 · ck 50 · opus 1 |
| `sel_a216d21d97247805` | `M257-1024` | 263 | asm 157 · ck 96 · opus 10 |
| `sel_6d485fcb23ffd0fe` | `M1025-4096` | 231 | ck 105 · asm 90 · opus 36 |
| `sel_f1384cc1069c9823` | `M>4096` | 67 | asm 32 · ck 26 · opus 9 |

### `gfx942` · 80 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_5aecc3631bf688e9` | `M<=16` | 247 | ck 230 · cktile 17 |
| `sel_5d90eccfeec27af1` | `M17-64` | 125 | ck 113 · cktile 11 · asm 1 |
| `sel_def10f3ed4823c11` | `M65-256` | 307 | ck 282 · asm 18 · cktile 7 |
| `sel_3316f2390f8a8002` | `M257-1024` | 394 | ck 384 · asm 9 · cktile 1 |
| `sel_bb0d68aae12eba45` | `M1025-4096` | 133 | ck 115 · cktile 14 · asm 4 |
| `sel_438e88e4bc86460c` | `M>4096` | 419 | ck 402 · cktile 13 · asm 4 |

### `gfx942` · 80 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_80605140d94f35fc` | `M<=16` | 80 | opus 50 · hipblaslt 24 · torch 4 · skinny 2 |
| `sel_170ff4ea1946a3d5` | `M17-64` | 45 | opus 25 · hipblaslt 10 · asm 7 · torch 3 |
| `sel_4d6a47f7f4d4d10e` | `M65-256` | 71 | opus 52 · hipblaslt 10 · asm 6 · torch 3 |
| `sel_04a72cfcbd6bc412` | `M257-1024` | 59 | opus 44 · hipblaslt 10 · torch 4 · asm 1 |
| `sel_5d979de3c64cf4db` | `M1025-4096` | 19 | hipblaslt 10 · opus 9 |
| `sel_27f9fc657457689d` | `M>4096` | 22 | hipblaslt 10 · torch 7 · opus 3 · triton 2 |

### `gfx942` · 80 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_1cb0d23b48f53a36` | `M<=16` | 1 | flydsl 1 |
| `sel_478810487308b7fc` | `M<=16` | 1 | flydsl 1 |
| `sel_23b90fa889e1be84` | `M<=16` | 1 | flydsl 1 |
| `sel_84b5fbf8064a1f7e` | `M<=16` | 1 | flydsl 1 |
| `sel_af6f6cacb2e0ab28` | `M<=16` | 1 | asm 1 |
| `sel_e0c5277061b12ac8` | `M<=16` | 1 | ck 1 |
| `sel_c2ba9a03e3a0d0f3` | `M17-64` | 3 | flydsl 3 |
| `sel_f2b78213c606365f` | `M17-64` | 3 | flydsl 3 |
| `sel_9794a2f7bb050812` | `M17-64` | 3 | flydsl 3 |
| `sel_147330df3b4254f7` | `M17-64` | 3 | flydsl 3 |
| `sel_baf55819cd2f1bf7` | `M17-64` | 2 | asm 2 |
| `sel_44fcdd43a0bf1f45` | `M17-64` | 3 | asm 3 |
| `sel_63d031bd8eb23ac9` | `M17-64` | 2 | ck 2 |
| `sel_017c9ba73fd3a7e6` | `M17-64` | 1 | ck 1 |
| `sel_193f477d9777428a` | `M17-64` | 1 | ck 1 |
| `sel_81197648736497a3` | `M65-256` | 1 | ck 1 |
| `sel_7b4709fe54a8728c` | `M65-256` | 1 | ck 1 |
| `sel_307ab5ecf13ef571` | `M65-256` | 2 | flydsl 2 |
| `sel_9cfef85e8bd832be` | `M65-256` | 2 | flydsl 2 |
| `sel_599a2831831e1516` | `M65-256` | 2 | flydsl 2 |
| `sel_980da24b79c9dfa3` | `M65-256` | 2 | flydsl 2 |
| `sel_272c9b34022baf1e` | `M65-256` | 3 | asm 3 |
| `sel_45a70de34b95c403` | `M65-256` | 2 | ck 2 |
| `sel_9eee8ccbfd17b056` | `M65-256` | 2 | ck 2 |
| `sel_7629913c96550464` | `M65-256` | 2 | ck 2 |
| `sel_b805c50aa6551f61` | `M65-256` | 1 | ck 1 |
| `sel_e07816c4f411f07d` | `M65-256` | 1 | ck 1 |
| `sel_4341f1dc0c6e2287` | `M257-1024` | 2 | ck 2 |
| `sel_6b4c4d478cbdbca3` | `M257-1024` | 2 | ck 2 |
| `sel_afd84b1de0d8aaea` | `M257-1024` | 2 | flydsl 2 |
| `sel_6fb3983f99ba9755` | `M257-1024` | 2 | flydsl 2 |
| `sel_500d99bc37be5285` | `M257-1024` | 2 | flydsl 2 |
| `sel_c2a264bcaa709add` | `M257-1024` | 2 | flydsl 2 |
| `sel_f82f9144593bafde` | `M257-1024` | 2 | asm 2 |
| `sel_cbed27cefc972275` | `M257-1024` | 2 | ck 2 |
| `sel_204ea22d48dd15ff` | `M257-1024` | 2 | ck 2 |
| `sel_aaa04491511c0f4c` | `M257-1024` | 2 | ck 2 |
| `sel_aa26b6b8a444d01c` | `M257-1024` | 2 | ck 2 |
| `sel_dbd0a3748e22d279` | `M257-1024` | 2 | ck 2 |
| `sel_14e622cedf83c856` | `M1025-4096` | 1 | ck 1 |
| `sel_c833bffbd1ca7e0e` | `M1025-4096` | 1 | ck 1 |
| `sel_d8759701e5715875` | `M1025-4096` | 2 | asm 1 · ck 1 |
| `sel_3e79abb56f6023ce` | `M1025-4096` | 2 | ck 2 |
| `sel_a135d7e18487c96d` | `M1025-4096` | 2 | flydsl 2 |
| `sel_cabaaeeae54e1a41` | `M1025-4096` | 2 | flydsl 2 |
| `sel_beff5ca6be1efb59` | `M1025-4096` | 2 | flydsl 2 |
| `sel_049584a37d853572` | `M1025-4096` | 2 | flydsl 2 |
| `sel_9a716022a8d8557a` | `M1025-4096` | 1 | ck 1 |
| `sel_0d5a5630e83664d7` | `M1025-4096` | 1 | ck 1 |
| `sel_5cecca791dc8a8be` | `M1025-4096` | 1 | ck 1 |
| `sel_d9d9ca3908bdbc26` | `M1025-4096` | 1 | ck 1 |
| `sel_07cd743aa1c09f80` | `M1025-4096` | 2 | asm 2 |
| `sel_d57f87338668a10f` | `M1025-4096` | 2 | ck 2 |
| `sel_e4a02a0470fb0161` | `M1025-4096` | 1 | ck 1 |
| `sel_9acb20b5a4b6e54b` | `M1025-4096` | 1 | ck 1 |
| `sel_c72e1c6a8904dcd7` | `M1025-4096` | 2 | ck 2 |
| `sel_1a300eae8bb1d0bb` | `M1025-4096` | 2 | ck 2 |
| `sel_324b62909d055ddf` | `M>4096` | 1 | ck 1 |
| `sel_6689aceb81346d72` | `M>4096` | 1 | ck 1 |
| `sel_e611794d55460504` | `M>4096` | 1 | ck 1 |
| `sel_ea1b3f2bf8c2e54f` | `M>4096` | 1 | ck 1 |
| `sel_c7671da3b010318c` | `M>4096` | 6 | flydsl 6 |
| `sel_4149efe60eda204b` | `M>4096` | 6 | flydsl 6 |
| `sel_7c78905f3db9b1ab` | `M>4096` | 6 | flydsl 6 |
| `sel_8522e0a80834ac5a` | `M>4096` | 6 | flydsl 6 |
| `sel_119998d838c49ced` | `M>4096` | 2 | asm 2 |
| `sel_f1c1a0914977610e` | `M>4096` | 7 | asm 7 |

### `gfx942` · 304 CUs · `a8w8_blockscale_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_c4b9d0637eac7164` | `M<=16` | 273 | ck 272 · cktile 1 |
| `sel_db36e93ec66ad939` | `M17-64` | 248 | ck 248 |
| `sel_4b9dbdd6e980af7f` | `M65-256` | 840 | ck 839 · cktile 1 |
| `sel_e3ab01fa0d52961f` | `M257-1024` | 1214 | ck 1214 |
| `sel_b284340d57114232` | `M1025-4096` | 2369 | ck 2369 |
| `sel_7e2fd7216f3c4c9b` | `M>4096` | 19875 | ck 19875 |

### `gfx942` · 304 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_5cf769a02e468084` | `M17-64` | 2 | asm 2 |
| `sel_9609d454b46e0fab` | `M65-256` | 4 | asm 4 |
| `sel_20bfb36c5f1ee347` | `M257-1024` | 4 | asm 4 |
| `sel_6bb0baf26eda8568` | `M1025-4096` | 4 | asm 4 |
| `sel_4914d4c661ff1e2a` | `M>4096` | 18 | asm 18 |

### `gfx942` · 304 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_8c569b363eea4fc1` | `M<=16` | 1 | flydsl 1 |
| `sel_75b1b8a439e4ef7b` | `M<=16` | 1 | flydsl 1 |
| `sel_edc44fd332428b53` | `M<=16` | 2 | flydsl 2 |
| `sel_3f8c1a5eabb0b97f` | `M<=16` | 2 | flydsl 2 |
| `sel_dd2f1b4d8c6353b2` | `M<=16` | 1 | flydsl 1 |
| `sel_d8f66e7003d1d64e` | `M<=16` | 1 | flydsl 1 |
| `sel_70e80b8d1077aabe` | `M<=16` | 2 | asm 2 |
| `sel_03481ca6bc57d1d2` | `M<=16` | 1 | asm 1 |
| `sel_47a61e3d0f909b5d` | `M17-64` | 3 | flydsl 3 |
| `sel_61d65b7d96d82a3c` | `M17-64` | 3 | flydsl 3 |
| `sel_dba805fb1fdfbe6f` | `M17-64` | 6 | flydsl 6 |
| `sel_bd4e6f091d20ce76` | `M17-64` | 6 | flydsl 6 |
| `sel_ea2fce03efd1d9d9` | `M17-64` | 3 | flydsl 3 |
| `sel_c29c19ff9425f134` | `M17-64` | 3 | flydsl 3 |
| `sel_d81828be9c319fd7` | `M17-64` | 2 | asm 2 |
| `sel_6517d207d34385c3` | `M17-64` | 2 | asm 2 |
| `sel_2c5e78baf229df2f` | `M17-64` | 2 | asm 2 |
| `sel_643b8d3beb1020c2` | `M65-256` | 2 | flydsl 2 |
| `sel_e9cad8ad4aae93db` | `M65-256` | 2 | flydsl 2 |
| `sel_57acd7228416bf10` | `M65-256` | 4 | flydsl 4 |
| `sel_13305c353ca21867` | `M65-256` | 4 | flydsl 4 |
| `sel_057179e554f659de` | `M65-256` | 2 | flydsl 2 |
| `sel_6acd72060b0da4f6` | `M65-256` | 2 | flydsl 2 |
| `sel_bae7813677e6d778` | `M65-256` | 2 | asm 2 |
| `sel_e03d73da3ec0f5c0` | `M65-256` | 2 | asm 2 |
| `sel_600e93eebee4129d` | `M257-1024` | 2 | flydsl 2 |
| `sel_d0bbd34b90601bc8` | `M257-1024` | 2 | flydsl 2 |
| `sel_b4cff96355d5696c` | `M257-1024` | 4 | flydsl 4 |
| `sel_9aa95886b29699ba` | `M257-1024` | 4 | flydsl 4 |
| `sel_8f8c0e857ec86459` | `M257-1024` | 2 | flydsl 2 |
| `sel_a616e85a4cc7ca08` | `M257-1024` | 2 | flydsl 2 |
| `sel_d2c0918c699b72ea` | `M257-1024` | 2 | asm 2 |
| `sel_58aa5d2026e45839` | `M257-1024` | 2 | asm 2 |
| `sel_3349f29171c134bd` | `M1025-4096` | 2 | flydsl 2 |
| `sel_ad301dfaf9f5065c` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c362382969144924` | `M1025-4096` | 4 | flydsl 4 |
| `sel_8dad5cfca89833b1` | `M1025-4096` | 4 | flydsl 4 |
| `sel_e3a9cd1f243c8c1c` | `M1025-4096` | 2 | flydsl 2 |
| `sel_6af364b9049cc5ae` | `M1025-4096` | 2 | flydsl 2 |
| `sel_8ce62b0db9aacebb` | `M1025-4096` | 1 | asm 1 |
| `sel_2c2d2167a77dc495` | `M1025-4096` | 2 | asm 2 |
| `sel_55f5d52f188f44ab` | `M>4096` | 6 | flydsl 6 |
| `sel_695f50c422fb7fe8` | `M>4096` | 6 | flydsl 6 |
| `sel_752e2bcab5ccb1e6` | `M>4096` | 12 | flydsl 12 |
| `sel_6f2e31cfe5604ae1` | `M>4096` | 12 | flydsl 12 |
| `sel_cdd102ff477ab5d5` | `M>4096` | 6 | flydsl 6 |
| `sel_f9b883e8a56bd4a7` | `M>4096` | 6 | flydsl 6 |
| `sel_06ef08162c207d6c` | `M>4096` | 5 | asm 5 |
| `sel_4e7a26238de6ee95` | `M>4096` | 6 | asm 6 |
| `sel_c851cf86cee13a24` | `M>4096` | 5 | asm 5 |

### `gfx1250` · 96 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_373e97b41a41c5ac` | `M<=16` | 47 | triton 45 · torch 2 |
| `sel_b6d99e3b30b41f8d` | `M17-64` | 27 | triton 27 |
| `sel_5d282acc28a070f3` | `M65-256` | 68 | triton 68 |
| `sel_19525d053df6d7a9` | `M257-1024` | 22 | triton 20 · torch 2 |
| `sel_6b04d4ff20b37469` | `M1025-4096` | 25 | triton 25 |
| `sel_d604b3ae1da352db` | `M>4096` | 22 | triton 22 |

### `gfx1250` · 192 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_9944675561ed2906` | `M<=16` | 3 | torch 2 · triton 1 |
| `sel_642384ab356cc71c` | `M257-1024` | 2 | triton 2 |
| `sel_f5a4274057e2fb70` | `M>4096` | 4 | triton 4 |

### `gfx1250` · 192 CUs · `tuned_grouped_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_48d40ddc8ccb66c1` | `M<=16` | 1 | flydsl 1 |
| `sel_f29a9a52df38cc2e` | `M<=16` | 1 | flydsl 1 |
| `sel_083da3956191e271` | `M<=16` | 1 | flydsl 1 |
| `sel_3d0a9576aadbf4b3` | `M<=16` | 1 | flydsl 1 |
| `sel_1cd71caee161352a` | `M<=16` | 1 | flydsl 1 |
| `sel_e4eae93b32b61dc6` | `M<=16` | 1 | flydsl 1 |
| `sel_d195f8ca498e1cf8` | `M17-64` | 2 | flydsl 2 |
| `sel_038081157917b4b9` | `M17-64` | 2 | flydsl 2 |
| `sel_4fba2e4dde34ffd9` | `M17-64` | 2 | flydsl 2 |
| `sel_2ee8a2c0b7e8cf7f` | `M17-64` | 2 | flydsl 2 |
| `sel_bde6298fbc4231b5` | `M17-64` | 2 | flydsl 2 |
| `sel_f509c202456e149c` | `M17-64` | 2 | flydsl 2 |
| `sel_76578170d65702a4` | `M65-256` | 2 | flydsl 2 |
| `sel_788417c68a635140` | `M65-256` | 2 | flydsl 2 |
| `sel_2d40ceca7f17534f` | `M65-256` | 2 | flydsl 2 |
| `sel_e754b774dffd04ec` | `M65-256` | 2 | flydsl 2 |
| `sel_d422b532fa38323d` | `M65-256` | 2 | flydsl 2 |
| `sel_3eaab848678f411f` | `M65-256` | 2 | flydsl 2 |
| `sel_b891fb3c7d1c41ba` | `M257-1024` | 2 | flydsl 2 |
| `sel_cd46e986d314bef9` | `M257-1024` | 2 | flydsl 2 |
| `sel_4308b4b59997507e` | `M257-1024` | 2 | flydsl 2 |
| `sel_15152de41e4794bc` | `M257-1024` | 2 | flydsl 2 |
| `sel_50ebf54a51b52cb1` | `M257-1024` | 2 | flydsl 2 |
| `sel_d25970ab27824957` | `M257-1024` | 2 | flydsl 2 |
| `sel_1bdbc14ed8328531` | `M1025-4096` | 2 | flydsl 2 |
| `sel_389f81fa89d88d01` | `M1025-4096` | 2 | flydsl 2 |
| `sel_861405cf62b2027d` | `M1025-4096` | 2 | flydsl 2 |
| `sel_aad67b55e55deb84` | `M1025-4096` | 2 | flydsl 2 |
| `sel_dffbeb2421ef6a4e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_4252ed83b619be90` | `M1025-4096` | 2 | flydsl 2 |
| `sel_2c310ca7dde89ffc` | `M>4096` | 3 | flydsl 3 |
| `sel_042a6f80767c4f7b` | `M>4096` | 3 | flydsl 3 |
| `sel_279cc221274f6ff1` | `M>4096` | 3 | flydsl 3 |
| `sel_1f0554507c7b302d` | `M>4096` | 3 | flydsl 3 |
| `sel_3a40cc4b65e955b1` | `M>4096` | 3 | flydsl 3 |
| `sel_136d84403e5bb62f` | `M>4096` | 3 | flydsl 3 |

### `gfx1250` · 256 CUs · `a8w8_blockscale_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_c10b8f49bc49f682` | `M<=16` | 6 | triton 6 |
| `sel_fe35250d70caaf1e` | `M17-64` | 5 | triton 5 |

### `gfx1250` · 256 CUs · `a8w8_bpreshuffle_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_5c177728ca6c374a` | `M<=16` | 4 | flydsl 4 |
| `sel_eb3ae4fae77268f1` | `M17-64` | 8 | flydsl 8 |
| `sel_6a7ad0f5d6e0fbd3` | `M65-256` | 8 | flydsl 8 |
| `sel_d167813c1f586d17` | `M257-1024` | 8 | flydsl 8 |
| `sel_ec0c48b0adb0fb6c` | `M1025-4096` | 8 | flydsl 8 |
| `sel_7d358a3d7c9fb4c7` | `M>4096` | 12 | flydsl 12 |

### `gfx1250` · 256 CUs · `bf16_tuned_gemm`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_45549acaa38e8de0` | `M<=16` | 222 | triton 149 · opus 73 |
| `sel_6242a12940bee830` | `M17-64` | 102 | triton 64 · opus 35 · flydsl 3 |
| `sel_3b2a90c0f5a60165` | `M65-256` | 112 | triton 61 · opus 51 |
| `sel_9a950caece1a9cec` | `M257-1024` | 97 | triton 49 · opus 48 |
| `sel_514b40589ffe3e8f` | `M1025-4096` | 113 | triton 65 · opus 48 |
| `sel_0cdbe7d4bb0be0ee` | `M>4096` | 107 | triton 58 · opus 49 |

### `gfx1250` · 256 CUs · `tuned_grouped_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_06471d1dbda64caa` | `M<=16` | 2 | flydsl 2 |
| `sel_519b07aa3b1c2b0d` | `M<=16` | 2 | flydsl 2 |
| `sel_3ed621f2cdd7021b` | `M<=16` | 4 | flydsl 4 |
| `sel_c8c066acd4102c14` | `M<=16` | 4 | flydsl 4 |
| `sel_5de4d566872b82b0` | `M<=16` | 2 | flydsl 2 |
| `sel_51638e48897ff957` | `M<=16` | 2 | flydsl 2 |
| `sel_f4ca394e0ba62510` | `M17-64` | 2 | flydsl 2 |
| `sel_ce24983ecbe55502` | `M17-64` | 2 | flydsl 2 |
| `sel_86f9d523c5d8ebdb` | `M17-64` | 4 | flydsl 4 |
| `sel_bb840ba625007781` | `M17-64` | 4 | flydsl 4 |
| `sel_92b7a0d08d13901d` | `M17-64` | 2 | flydsl 2 |
| `sel_e64bfae8be2971d6` | `M17-64` | 2 | flydsl 2 |
| `sel_ba472e05a0adae68` | `M17-64` | 2 | flydsl 2 |
| `sel_5484bb222b4c9530` | `M17-64` | 2 | flydsl 2 |
| `sel_3d616f216a15c791` | `M17-64` | 2 | flydsl 2 |
| `sel_666193976ff96ba2` | `M17-64` | 2 | flydsl 2 |
| `sel_986b08136de525b8` | `M65-256` | 2 | flydsl 2 |
| `sel_5f05b5c6f0ac0b44` | `M65-256` | 2 | flydsl 2 |
| `sel_cd974b85bbb32df9` | `M65-256` | 4 | flydsl 4 |
| `sel_9e2efd37775266ef` | `M65-256` | 4 | flydsl 4 |
| `sel_93309c18e932f2d8` | `M65-256` | 2 | flydsl 2 |
| `sel_def901ae2250da8a` | `M65-256` | 2 | flydsl 2 |
| `sel_8ce166f9ce73adb3` | `M65-256` | 4 | flydsl 4 |
| `sel_6e5206d9f7b70388` | `M65-256` | 4 | flydsl 4 |
| `sel_8b1179c331fdac3a` | `M65-256` | 4 | flydsl 4 |
| `sel_9b7d45c6299c6e67` | `M65-256` | 4 | flydsl 4 |
| `sel_daa51dbd438acce9` | `M257-1024` | 5 | flydsl 5 |
| `sel_a37b3b91b3e25029` | `M257-1024` | 5 | flydsl 5 |
| `sel_c311162571dd1a1f` | `M257-1024` | 4 | flydsl 4 |
| `sel_11a516ac3d884abd` | `M257-1024` | 4 | flydsl 4 |
| `sel_daf71fcd1edc4b56` | `M257-1024` | 2 | flydsl 2 |
| `sel_42d4a60cfb7d1570` | `M257-1024` | 2 | flydsl 2 |
| `sel_6904a16fd5a4fb82` | `M257-1024` | 4 | flydsl 4 |
| `sel_50abb234b3849d0c` | `M257-1024` | 4 | flydsl 4 |
| `sel_af6c1571268460f3` | `M257-1024` | 4 | flydsl 4 |
| `sel_fa433e3031e4b820` | `M257-1024` | 4 | flydsl 4 |
| `sel_210cfc092578b8a4` | `M1025-4096` | 2 | flydsl 2 |
| `sel_0631362708e8d12b` | `M1025-4096` | 2 | flydsl 2 |
| `sel_37d24ba76d7e044c` | `M1025-4096` | 4 | flydsl 4 |
| `sel_46286235a6faab2e` | `M1025-4096` | 4 | flydsl 4 |
| `sel_1ba2186a5c0e1bcd` | `M1025-4096` | 2 | flydsl 2 |
| `sel_68d27aaebb958d3d` | `M1025-4096` | 2 | flydsl 2 |
| `sel_a73c49daa0297477` | `M1025-4096` | 4 | flydsl 4 |
| `sel_8c4fdebec8988201` | `M1025-4096` | 4 | flydsl 4 |
| `sel_32745b928aea6042` | `M1025-4096` | 4 | flydsl 4 |
| `sel_0a7752ae120337e7` | `M1025-4096` | 4 | flydsl 4 |
| `sel_61ab7af709aa4b91` | `M>4096` | 6 | flydsl 6 |
| `sel_b407ec04b7f2b722` | `M>4096` | 6 | flydsl 6 |
| `sel_fbcb87b8a511ace8` | `M>4096` | 1 | flydsl 1 |
| `sel_57514ea584216554` | `M>4096` | 1 | flydsl 1 |
| `sel_983a2a0d4e3461b8` | `M>4096` | 3 | flydsl 3 |
| `sel_38e70a7eeb764cbb` | `M>4096` | 3 | flydsl 3 |
| `sel_e55cf949ff7b84c8` | `M>4096` | 1 | flydsl 1 |
| `sel_0cb13f5cf09549f8` | `M>4096` | 1 | flydsl 1 |
| `sel_57b438f8438882a7` | `M>4096` | 8 | flydsl 8 |
| `sel_7a91442fb8466771` | `M>4096` | 8 | flydsl 8 |
| `sel_85817b02d7893a31` | `M>4096` | 3 | flydsl 3 |
| `sel_7ebac85ed8525144` | `M>4096` | 3 | flydsl 3 |
| `sel_f977ba1e66ffa743` | `M>4096` | 12 | flydsl 12 |
| `sel_61458c81ffb37794` | `M>4096` | 12 | flydsl 12 |
| `sel_bcf6c7cdfae5b803` | `M>4096` | 12 | flydsl 12 |
| `sel_cc511d9367f34fc7` | `M>4096` | 12 | flydsl 12 |

### `gfx1201` · 32 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_206fdf9e83df645b` | `M<=16` | 2 | ck 2 |
| `sel_3e9b55af83f0b1f6` | `M<=16` | 2 | ck 2 |
| `sel_3bd56747a037c5e3` | `M<=16` | 2 | ck 2 |
| `sel_f11bc0c8009b8a15` | `M<=16` | 2 | ck 2 |
| `sel_76beab2c43ab2c44` | `M17-64` | 2 | ck 2 |
| `sel_24d536e8602db095` | `M17-64` | 2 | ck 2 |
| `sel_dcf194452af080be` | `M17-64` | 2 | ck 2 |
| `sel_f40a0e37a7e3dc8a` | `M17-64` | 2 | ck 2 |
| `sel_77313d16fac3bdf9` | `M65-256` | 3 | ck 3 |
| `sel_bbd97d15a6b08695` | `M65-256` | 3 | ck 3 |
| `sel_cbfee2b49ca9a641` | `M65-256` | 3 | ck 3 |
| `sel_6bf129cd5d12cd19` | `M65-256` | 3 | ck 3 |
| `sel_26171f96a828ccfa` | `M257-1024` | 4 | ck 4 |
| `sel_d27975e11c690594` | `M257-1024` | 4 | ck 4 |
| `sel_8e113ace2e5d507c` | `M257-1024` | 4 | ck 4 |
| `sel_37b674a9a4e5431a` | `M257-1024` | 4 | ck 4 |
| `sel_55c8e21c5e34fa33` | `M1025-4096` | 4 | ck 4 |
| `sel_cbe758879f159059` | `M1025-4096` | 4 | ck 4 |
| `sel_afe0f915d3816df5` | `M1025-4096` | 4 | ck 4 |
| `sel_b93f3edb76afb07a` | `M1025-4096` | 4 | ck 4 |
| `sel_a089903862221342` | `M>4096` | 3 | ck 3 |
| `sel_3eaf950f8bcfde07` | `M>4096` | 3 | ck 3 |
| `sel_87a2589902804f3a` | `M>4096` | 3 | ck 3 |
| `sel_a1301ba571d76a84` | `M>4096` | 3 | ck 3 |

### `` · 80 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_ffce9441cf275c19` | `M<=16` | 8 | ck 8 |
| `sel_00cfbd329b8b89f5` | `M<=16` | 8 | ck 8 |
| `sel_e123d7a6984a6e46` | `M<=16` | 2 | asm 2 |
| `sel_9e07559e27908297` | `M<=16` | 2 | ck 2 |
| `sel_dfe09ee8c19321a9` | `M<=16` | 10 | ck 10 |
| `sel_ad02af4af6fde2aa` | `M<=16` | 10 | ck 10 |
| `sel_fc02dfd0076e2dbf` | `M<=16` | 30 | ck 24 · asm 6 |
| `sel_9cbdb48bd6ca11aa` | `M<=16` | 30 | ck 30 |
| `sel_4d44e0adaba2e1a5` | `M<=16` | 2 | asm 2 |
| `sel_4761a12b96963dcb` | `M<=16` | 1 | ck 1 |
| `sel_b3cb08ae23f729a0` | `M<=16` | 32 | ck 28 · asm 4 |
| `sel_3090e4f9c3426aeb` | `M<=16` | 32 | ck 32 |
| `sel_78fb9e1ad04bce39` | `M<=16` | 1 | ck 1 |
| `sel_ab494237d5b534da` | `M<=16` | 1 | ck 1 |
| `sel_3af97f5a15af4a5e` | `M<=16` | 32 | ck 28 · asm 4 |
| `sel_be0362b5252924ac` | `M<=16` | 32 | ck 32 |
| `sel_50d5f6fa64da7c55` | `M17-64` | 8 | ck 8 |
| `sel_90a1aedec144cc2f` | `M17-64` | 8 | ck 8 |
| `sel_8c3c54280acef3dc` | `M17-64` | 10 | ck 10 |
| `sel_27150c66c18e9eb2` | `M17-64` | 10 | ck 10 |
| `sel_d8d3fb4dd4601a86` | `M17-64` | 30 | ck 24 · asm 6 |
| `sel_0eb8ca7808c6879c` | `M17-64` | 30 | ck 30 |
| `sel_7f831f4f9a8662af` | `M17-64` | 4 | asm 4 |
| `sel_5dc2aba4177f95de` | `M17-64` | 2 | ck 2 |
| `sel_bb6382dd9be6e335` | `M17-64` | 32 | ck 28 · asm 4 |
| `sel_2463ae70086f63da` | `M17-64` | 32 | ck 32 |
| `sel_78dd1e6c68f02761` | `M17-64` | 2 | ck 2 |
| `sel_4872fe3d76dd87ec` | `M17-64` | 2 | ck 2 |
| `sel_2b332de3d026e506` | `M17-64` | 32 | ck 28 · asm 4 |
| `sel_d8572e153db0bddb` | `M17-64` | 32 | ck 32 |
| `sel_7793bb78ecba4982` | `M65-256` | 8 | ck 8 |
| `sel_f13b8eda330b4a5d` | `M65-256` | 8 | ck 8 |
| `sel_f5c854c02051c0fc` | `M65-256` | 10 | ck 9 · asm 1 |
| `sel_8687f23ae10c7f01` | `M65-256` | 9 | ck 9 |
| `sel_029efa456f6f856c` | `M65-256` | 30 | ck 25 · asm 5 |
| `sel_d768c076397c8dfc` | `M65-256` | 30 | ck 30 |
| `sel_2946b2aabd136781` | `M65-256` | 1 | asm 1 |
| `sel_2494d16dbe391241` | `M65-256` | 4 | asm 4 |
| `sel_cfa61f8d65896d32` | `M65-256` | 2 | ck 2 |
| `sel_3fb2ba7c962e6324` | `M65-256` | 32 | ck 29 · asm 3 |
| `sel_20d6964ca6ccb41c` | `M65-256` | 32 | ck 32 |
| `sel_1e214a6ca674cf7f` | `M65-256` | 2 | ck 2 |
| `sel_d77bfe663effc243` | `M65-256` | 2 | ck 2 |
| `sel_2d24e14e4d6bcc56` | `M65-256` | 32 | ck 26 · asm 6 |
| `sel_78d4a44093551979` | `M65-256` | 32 | ck 32 |
| `sel_525055c7cc0433ff` | `M257-1024` | 8 | ck 8 |
| `sel_eb27adadc7354f17` | `M257-1024` | 8 | ck 8 |
| `sel_508b616d6d01a25b` | `M257-1024` | 11 | ck 8 · asm 3 |
| `sel_4f74770e936baf7e` | `M257-1024` | 8 | ck 8 |
| `sel_0a132ff754e18d5d` | `M257-1024` | 30 | ck 23 · asm 7 |
| `sel_0ef02bd3ace6e5dd` | `M257-1024` | 30 | ck 30 |
| `sel_1f1c4fa19a3f4ab9` | `M257-1024` | 3 | asm 2 · ck 1 |
| `sel_41526059035f7ff9` | `M257-1024` | 3 | ck 3 |
| `sel_4622690b10b7b13e` | `M257-1024` | 4 | asm 3 · ck 1 |
| `sel_e88ef9c17db225cc` | `M257-1024` | 1 | ck 1 |
| `sel_71dea6cb3f184618` | `M257-1024` | 31 | ck 20 · asm 11 |
| `sel_7c3fabce6b1963c2` | `M257-1024` | 30 | ck 30 |
| `sel_2c3fbea2b04f7979` | `M257-1024` | 2 | asm 1 · ck 1 |
| `sel_5267b21d56a1acac` | `M257-1024` | 1 | ck 1 |
| `sel_738404ef36c698d5` | `M257-1024` | 32 | ck 18 · asm 14 |
| `sel_d42bb4ff0eaeafd8` | `M257-1024` | 32 | ck 32 |
| `sel_9d8d0f6a651fc5d5` | `M1025-4096` | 8 | ck 8 |
| `sel_278d14431883d661` | `M1025-4096` | 8 | ck 8 |
| `sel_a06cbd9f2bac98bb` | `M1025-4096` | 10 | ck 8 · asm 2 |
| `sel_4db50fd138436c46` | `M1025-4096` | 8 | ck 8 |
| `sel_819ea17aa78f25b0` | `M1025-4096` | 30 | ck 22 · asm 8 |
| `sel_2e88a560f7470f3f` | `M1025-4096` | 30 | ck 30 |
| `sel_3c41d6206ff33a19` | `M1025-4096` | 4 | asm 4 |
| `sel_e4e3578e6eb0367d` | `M1025-4096` | 31 | ck 21 · asm 10 |
| `sel_871c81673c05271f` | `M1025-4096` | 31 | ck 31 |
| `sel_07e6493d83b557ac` | `M1025-4096` | 2 | ck 2 |
| `sel_7c621405ce48351f` | `M1025-4096` | 2 | ck 2 |
| `sel_186d37fa1ede065e` | `M1025-4096` | 28 | ck 16 · asm 12 |
| `sel_ce74955ee1497be9` | `M1025-4096` | 28 | ck 28 |
| `sel_863bddb78a9bde5c` | `M>4096` | 24 | ck 24 |
| `sel_be9e414632b08dcb` | `M>4096` | 24 | ck 24 |
| `sel_80374a396690ecaf` | `M>4096` | 28 | ck 24 · asm 4 |
| `sel_1aaec56f4cf0617c` | `M>4096` | 24 | ck 24 |
| `sel_0631d84e4d1825fd` | `M>4096` | 90 | ck 78 · asm 12 |
| `sel_33374a519982647b` | `M>4096` | 90 | ck 90 |
| `sel_bb8ee57e1c74eac7` | `M>4096` | 14 | asm 14 |
| `sel_bc90c58c4437d685` | `M>4096` | 96 | ck 60 · asm 36 |
| `sel_526b5ca6059e33c2` | `M>4096` | 90 | ck 90 |
| `sel_670fe6bda698c018` | `M>4096` | 6 | asm 6 |
| `sel_a18381e49a945886` | `M>4096` | 96 | asm 48 · ck 48 |
| `sel_58cdbeeca85fa6c5` | `M>4096` | 96 | ck 96 |

### `` · 256 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_e16c97cf9830e0c3` | `M<=16` | 2 | asm 2 |
| `sel_f1bb7148763944ce` | `M<=16` | 1 | asm 1 |
| `sel_1f01ee02c38c551c` | `M<=16` | 2 | asm 2 |
| `sel_6cae8f6b500888e5` | `M<=16` | 2 | asm 2 |
| `sel_d6eb8b3f7f9da394` | `M<=16` | 2 | asm 2 |
| `sel_025928fbbf591619` | `M<=16` | 1 | asm 1 |
| `sel_263b6c7bb299fe1d` | `M<=16` | 4 | asm 2 · flydsl 2 |
| `sel_ff5e78003d4154e8` | `M<=16` | 2 | ck 2 |
| `sel_246c96be2bea9b4b` | `M<=16` | 2 | asm 2 |
| `sel_c1acfdc747269605` | `M<=16` | 2 | flydsl 2 |
| `sel_20445ec848650764` | `M<=16` | 2 | ck 2 |
| `sel_4dfd17383dfd7553` | `M<=16` | 4 | ck 2 · flydsl 2 |
| `sel_237d87a5fd5ef88f` | `M<=16` | 4 | ck 2 · flydsl 2 |
| `sel_5f3c9b94108385c6` | `M<=16` | 4 | asm 2 · flydsl 2 |
| `sel_cab3aa00f33c97d5` | `M<=16` | 2 | ck 2 |
| `sel_02a05dbf594825fd` | `M<=16` | 2 | asm 2 |
| `sel_e07aad01433c685d` | `M<=16` | 1 | asm 1 |
| `sel_186637e8a8600e2c` | `M<=16` | 6 | ck 6 |
| `sel_0cd8bfe837025a32` | `M<=16` | 6 | ck 6 |
| `sel_37fce3386a8f68b2` | `M<=16` | 5 | asm 5 |
| `sel_d369351cadba6c8b` | `M<=16` | 2 | asm 1 · ck 1 |
| `sel_af35403c173e0fa0` | `M<=16` | 2 | ck 2 |
| `sel_2f11507e492f0a87` | `M<=16` | 1 | ck 1 |
| `sel_4027cf6e2972c97f` | `M<=16` | 1 | ck 1 |
| `sel_a45def2f04426e95` | `M<=16` | 1 | asm 1 |
| `sel_21e4a264c16b504a` | `M<=16` | 1 | asm 1 |
| `sel_76363813e2a57691` | `M<=16` | 1 | ck 1 |
| `sel_6374f5245c378976` | `M<=16` | 1 | ck 1 |
| `sel_ed5e8f17d1f95211` | `M<=16` | 1 | ck 1 |
| `sel_b0f5796bc5ddb56e` | `M<=16` | 4 | flydsl 4 |
| `sel_7e7e30af1f9e2161` | `M<=16` | 4 | flydsl 4 |
| `sel_4b0afa2b08ee334a` | `M<=16` | 2 | flydsl 2 |
| `sel_9d5a389638c1f089` | `M<=16` | 2 | flydsl 2 |
| `sel_91ba40bc67cda055` | `M<=16` | 2 | flydsl 2 |
| `sel_b3b3183ff944987a` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_34f78b78658e7e60` | `M<=16` | 2 | flydsl 2 |
| `sel_8954ccb2569f4ba3` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_17a16137c2c33f93` | `M<=16` | 2 | flydsl 2 |
| `sel_e278a74bce683182` | `M<=16` | 2 | ck 2 |
| `sel_68f0a6d58aba1125` | `M<=16` | 2 | asm 1 · flydsl 1 |
| `sel_e8ae8593dccd0342` | `M<=16` | 1 | ck 1 |
| `sel_9b2b37eb892ca687` | `M<=16` | 6 | flydsl 6 |
| `sel_1749674b09f84985` | `M<=16` | 6 | flydsl 4 · ck 2 |
| `sel_5a2fb4c4ea0c9ce9` | `M<=16` | 2 | flydsl 2 |
| `sel_9e2ee12e2ec5fbbe` | `M<=16` | 2 | flydsl 2 |
| `sel_8b97aa8d558cc24e` | `M<=16` | 2 | flydsl 2 |
| `sel_4adda54e23ed329b` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_48a7eeebf7f6eeba` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_ba0c9a0313271d61` | `M<=16` | 2 | ck 2 |
| `sel_c47a5025c17988ac` | `M<=16` | 6 | flydsl 6 |
| `sel_113aff628656eae4` | `M<=16` | 6 | flydsl 6 |
| `sel_a02f63176ec336cb` | `M<=16` | 2 | flydsl 2 |
| `sel_147237b35c3e8956` | `M<=16` | 2 | flydsl 2 |
| `sel_93a1cfbd03e78c93` | `M<=16` | 1 | asm 1 |
| `sel_8a9dfc38e58362d4` | `M<=16` | 1 | asm 1 |
| `sel_aff0543f7f2ea163` | `M<=16` | 2 | ck 2 |
| `sel_246a47f46508e372` | `M<=16` | 2 | ck 2 |
| `sel_237031481a18c414` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_e9b9ce103ebfc6ec` | `M<=16` | 2 | ck 1 · flydsl 1 |
| `sel_f772788a4bb4c44f` | `M<=16` | 5 | ck 5 |
| `sel_7e07b9528297430b` | `M17-64` | 4 | asm 4 |
| `sel_01b510ee9516fc8a` | `M17-64` | 4 | asm 4 |
| `sel_8941afc5afb8e705` | `M17-64` | 3 | asm 3 |
| `sel_95f71932ae5d6ed3` | `M17-64` | 2 | asm 2 |
| `sel_2eccc0fd7bca2da9` | `M17-64` | 3 | asm 3 |
| `sel_c8dfd1cb797e2d02` | `M17-64` | 4 | asm 4 |
| `sel_9383ce470ca723e2` | `M17-64` | 4 | asm 2 · flydsl 2 |
| `sel_4aae3ee06fb512be` | `M17-64` | 2 | flydsl 2 |
| `sel_387e3be43ad27968` | `M17-64` | 2 | asm 2 |
| `sel_afce0378777d9600` | `M17-64` | 2 | flydsl 2 |
| `sel_e6f53cfceceb3677` | `M17-64` | 2 | ck 2 |
| `sel_22d0854d28b67578` | `M17-64` | 4 | asm 2 · flydsl 2 |
| `sel_6ed2bf8847426eca` | `M17-64` | 2 | flydsl 2 |
| `sel_df31a626c3f6e401` | `M17-64` | 4 | asm 2 · flydsl 2 |
| `sel_1cb32d39df700b66` | `M17-64` | 2 | flydsl 2 |
| `sel_1911c5e170c182a0` | `M17-64` | 2 | asm 2 |
| `sel_8e280d6ffd020a3b` | `M17-64` | 2 | asm 2 |
| `sel_1065ca00f3e56ac7` | `M17-64` | 6 | ck 4 · asm 2 |
| `sel_3a86fb70bad19722` | `M17-64` | 4 | ck 4 |
| `sel_35f1e25615925d8b` | `M17-64` | 2 | asm 2 |
| `sel_9d5df5c39087ba14` | `M17-64` | 5 | asm 5 |
| `sel_88f218185e76ad6f` | `M17-64` | 5 | ck 5 |
| `sel_17f57f1a6a25b554` | `M17-64` | 2 | asm 2 |
| `sel_9addc24bdc595544` | `M17-64` | 2 | asm 2 |
| `sel_559e4a6d519c0d57` | `M17-64` | 2 | ck 2 |
| `sel_9ff62a8f315a04bc` | `M17-64` | 2 | ck 2 |
| `sel_a978df88963ffed6` | `M17-64` | 2 | ck 2 |
| `sel_0e1891e4200789ab` | `M17-64` | 4 | flydsl 4 |
| `sel_4ba4e3d26442dd3a` | `M17-64` | 4 | flydsl 4 |
| `sel_13b6c2c7aa8a5a3a` | `M17-64` | 4 | flydsl 4 |
| `sel_dd9cb490a728bc3b` | `M17-64` | 4 | flydsl 4 |
| `sel_e4e30bc3653df14b` | `M17-64` | 2 | flydsl 2 |
| `sel_93087bc26cd7f5f5` | `M17-64` | 2 | flydsl 2 |
| `sel_4f9e6d831dc55c8c` | `M17-64` | 2 | flydsl 2 |
| `sel_334d30d595b3132d` | `M17-64` | 2 | flydsl 2 |
| `sel_e5cedfcb4ab2af4c` | `M17-64` | 2 | flydsl 2 |
| `sel_898a0e45163906c2` | `M17-64` | 2 | flydsl 2 |
| `sel_2138e415df44e03a` | `M17-64` | 4 | asm 2 · flydsl 2 |
| `sel_5d89d7a987693ca4` | `M17-64` | 2 | ck 1 · flydsl 1 |
| `sel_9b11cb721e68c072` | `M17-64` | 6 | flydsl 6 |
| `sel_d39c730225fb687f` | `M17-64` | 6 | flydsl 6 |
| `sel_849de8129865fe2c` | `M17-64` | 4 | flydsl 4 |
| `sel_91aaaafc21d0ebb2` | `M17-64` | 4 | flydsl 4 |
| `sel_b799df9b31e76499` | `M17-64` | 2 | flydsl 2 |
| `sel_58ef9ffe3f6b2016` | `M17-64` | 2 | flydsl 2 |
| `sel_1de6a4460182e3cc` | `M17-64` | 4 | asm 2 · flydsl 2 |
| `sel_216a9b050d689488` | `M17-64` | 2 | flydsl 2 |
| `sel_850bfdc3771f392c` | `M17-64` | 6 | flydsl 6 |
| `sel_591e742478aec43b` | `M17-64` | 6 | flydsl 6 |
| `sel_819dddc3fe8129a9` | `M17-64` | 4 | flydsl 4 |
| `sel_4cc4ee1556580b14` | `M17-64` | 4 | flydsl 4 |
| `sel_f588fa1bdffcbb28` | `M17-64` | 2 | asm 2 |
| `sel_b531788a3ff5d58a` | `M17-64` | 2 | asm 2 |
| `sel_0160e9e767075b0a` | `M17-64` | 2 | ck 2 |
| `sel_7878d6afc6aa876e` | `M17-64` | 2 | ck 2 |
| `sel_4d80e1d7a6022400` | `M17-64` | 4 | ck 2 · flydsl 2 |
| `sel_cc196cb75c10982e` | `M17-64` | 4 | ck 2 · flydsl 2 |
| `sel_cd7a4479b4832ff4` | `M65-256` | 4 | asm 4 |
| `sel_d4bc91e94619f6ab` | `M65-256` | 3 | asm 3 |
| `sel_8721ed5386b6a8e3` | `M65-256` | 4 | asm 4 |
| `sel_4a3c412df84a82e1` | `M65-256` | 4 | asm 4 |
| `sel_8ce4d12eee10b8f5` | `M65-256` | 4 | asm 4 |
| `sel_9de9da160074d1ad` | `M65-256` | 3 | asm 3 |
| `sel_69d396ee6345c78a` | `M65-256` | 4 | flydsl 2 · asm 1 · ck 1 |
| `sel_acac36d38037d174` | `M65-256` | 3 | flydsl 2 · ck 1 |
| `sel_cd975e5f268e4b56` | `M65-256` | 2 | asm 2 |
| `sel_6772c75b8c179ff1` | `M65-256` | 2 | flydsl 2 |
| `sel_7744eb4afe27fc9a` | `M65-256` | 2 | flydsl 2 |
| `sel_ac33957773264334` | `M65-256` | 4 | asm 2 · flydsl 2 |
| `sel_998df9f8d285114e` | `M65-256` | 2 | flydsl 2 |
| `sel_38ee26f5dd543897` | `M65-256` | 4 | asm 2 · flydsl 2 |
| `sel_9deb6e9a3db73a79` | `M65-256` | 2 | flydsl 2 |
| `sel_7c2d594eadd2bcc7` | `M65-256` | 2 | ck 2 |
| `sel_5dfed84eda535260` | `M65-256` | 2 | ck 2 |
| `sel_5efcf81762bbed2e` | `M65-256` | 2 | asm 2 |
| `sel_fbba67ec124d02cf` | `M65-256` | 6 | ck 4 · asm 2 |
| `sel_377333eb93a059a3` | `M65-256` | 4 | ck 4 |
| `sel_c458eb6f43050898` | `M65-256` | 2 | asm 2 |
| `sel_2ea78aea09e33aea` | `M65-256` | 3 | asm 3 |
| `sel_7c6a3018256ef1fe` | `M65-256` | 3 | ck 3 |
| `sel_5d46cf46a35043b8` | `M65-256` | 2 | asm 2 |
| `sel_7298c366525ff454` | `M65-256` | 2 | asm 2 |
| `sel_39be6deaf80b2f43` | `M65-256` | 2 | asm 2 |
| `sel_88818697a596e6d4` | `M65-256` | 3 | flydsl 3 |
| `sel_cce5ff0fc84e99d9` | `M65-256` | 3 | flydsl 3 |
| `sel_6d772b1f4aeb0723` | `M65-256` | 4 | flydsl 4 |
| `sel_0631f30a3e98eec1` | `M65-256` | 4 | flydsl 4 |
| `sel_6383fe07a7fb74c9` | `M65-256` | 2 | flydsl 2 |
| `sel_814408e5a127d5c2` | `M65-256` | 2 | flydsl 2 |
| `sel_95ad66aa07240066` | `M65-256` | 2 | flydsl 2 |
| `sel_9bcefec2c4e7b4c3` | `M65-256` | 2 | flydsl 2 |
| `sel_b5ee635045e818a9` | `M65-256` | 2 | flydsl 2 |
| `sel_4d91ee0a6f5e8cf8` | `M65-256` | 2 | flydsl 2 |
| `sel_eb9b4c1bc4127f85` | `M65-256` | 4 | flydsl 2 · asm 1 · ck 1 |
| `sel_6475e1ec72d7d332` | `M65-256` | 3 | flydsl 2 · ck 1 |
| `sel_0f87da12127dc2ec` | `M65-256` | 6 | flydsl 6 |
| `sel_b799ac1f93b5c1b0` | `M65-256` | 6 | flydsl 6 |
| `sel_e4c08096fed3800d` | `M65-256` | 4 | flydsl 4 |
| `sel_22f43a508341b82c` | `M65-256` | 4 | flydsl 4 |
| `sel_cc834ad40bce1ccd` | `M65-256` | 2 | flydsl 2 |
| `sel_a26d08b346077373` | `M65-256` | 2 | flydsl 2 |
| `sel_c391890a9fa8b014` | `M65-256` | 4 | asm 2 · flydsl 2 |
| `sel_eb585756a288a021` | `M65-256` | 2 | flydsl 2 |
| `sel_10a3770d61646b0c` | `M65-256` | 6 | flydsl 6 |
| `sel_cf03731b36826be4` | `M65-256` | 6 | flydsl 6 |
| `sel_cb9310cb6a7e46a5` | `M65-256` | 4 | flydsl 4 |
| `sel_f745fbdf6dea6bcf` | `M65-256` | 4 | flydsl 4 |
| `sel_e7e410682191ed85` | `M65-256` | 2 | asm 2 |
| `sel_d539145885ce9ea5` | `M65-256` | 2 | asm 2 |
| `sel_1cd79191f65b9d1f` | `M65-256` | 2 | ck 2 |
| `sel_56637ba382bf73a2` | `M65-256` | 2 | ck 2 |
| `sel_65480dc10277c379` | `M65-256` | 4 | ck 2 · flydsl 2 |
| `sel_dd63fe6e4029c8ae` | `M65-256` | 4 | ck 2 · flydsl 2 |
| `sel_8432fbf2ea64a77f` | `M257-1024` | 2 | asm 2 |
| `sel_76ce6391a32b95cc` | `M257-1024` | 2 | asm 2 |
| `sel_2847318bc896190f` | `M257-1024` | 2 | asm 2 |
| `sel_78287b53296a1377` | `M257-1024` | 2 | asm 2 |
| `sel_7d907e37a62acc5f` | `M257-1024` | 2 | asm 2 |
| `sel_cb4a825700bba99c` | `M257-1024` | 2 | asm 2 |
| `sel_925373f5b04d7960` | `M257-1024` | 1 | flydsl 1 |
| `sel_d4805c3915ea8c77` | `M257-1024` | 1 | flydsl 1 |
| `sel_392c7e81639f37ed` | `M257-1024` | 4 | flydsl 2 · asm 1 · ck 1 |
| `sel_7fe760068a0c5814` | `M257-1024` | 3 | flydsl 2 · ck 1 |
| `sel_b4fd8f15dcaa394b` | `M257-1024` | 2 | asm 2 |
| `sel_1e738c14722bd0ee` | `M257-1024` | 2 | flydsl 2 |
| `sel_6e745e074d8b8e45` | `M257-1024` | 2 | flydsl 2 |
| `sel_c12f5143147db2b7` | `M257-1024` | 1 | flydsl 1 |
| `sel_2c3ba454d5dd062d` | `M257-1024` | 1 | flydsl 1 |
| `sel_1c552e0ac771bad6` | `M257-1024` | 4 | asm 2 · flydsl 2 |
| `sel_dbd146a825488ec0` | `M257-1024` | 2 | flydsl 2 |
| `sel_61165f3a4b083295` | `M257-1024` | 1 | flydsl 1 |
| `sel_3ac31f26a542816f` | `M257-1024` | 1 | flydsl 1 |
| `sel_107cef1d3945eb0a` | `M257-1024` | 4 | asm 2 · flydsl 2 |
| `sel_732e0f0cfccc17a9` | `M257-1024` | 2 | flydsl 2 |
| `sel_b69430f605aa39b9` | `M257-1024` | 2 | ck 2 |
| `sel_3cec9e141970f77f` | `M257-1024` | 2 | ck 2 |
| `sel_8d30dcfdaeb63c35` | `M257-1024` | 2 | asm 2 |
| `sel_092bacd707f960af` | `M257-1024` | 6 | ck 4 · asm 2 |
| `sel_1300de26387c81fd` | `M257-1024` | 4 | ck 4 |
| `sel_ddad7be60ad8c74e` | `M257-1024` | 2 | asm 2 |
| `sel_60d074b3bc96cb17` | `M257-1024` | 2 | asm 2 |
| `sel_ea99590da2dd46bd` | `M257-1024` | 2 | ck 2 |
| `sel_33f59913ae4c1e85` | `M257-1024` | 2 | asm 2 |
| `sel_8209af8d1a9917b0` | `M257-1024` | 2 | asm 2 |
| `sel_57b889543f8307b2` | `M257-1024` | 2 | asm 2 |
| `sel_690bf8848f26dc8b` | `M257-1024` | 3 | flydsl 3 |
| `sel_6a8e7a0351dd1769` | `M257-1024` | 3 | flydsl 3 |
| `sel_26d88ea005293a9a` | `M257-1024` | 4 | flydsl 4 |
| `sel_ff7f345731058c2d` | `M257-1024` | 4 | flydsl 4 |
| `sel_abf5608a91c2eb6a` | `M257-1024` | 2 | flydsl 2 |
| `sel_c2d5d8f3ff3f60d7` | `M257-1024` | 2 | flydsl 2 |
| `sel_0f184aa55fede47f` | `M257-1024` | 2 | flydsl 2 |
| `sel_6e6e06cb54ca5016` | `M257-1024` | 2 | flydsl 2 |
| `sel_91f0d35f6f585910` | `M257-1024` | 2 | flydsl 2 |
| `sel_5dec679d51f551da` | `M257-1024` | 2 | ck 2 |
| `sel_8e7fdfc60dda5eaa` | `M257-1024` | 4 | asm 2 · flydsl 2 |
| `sel_12dacd27ffe193ba` | `M257-1024` | 2 | ck 1 · flydsl 1 |
| `sel_2ba850c07126f582` | `M257-1024` | 6 | flydsl 6 |
| `sel_fc613a8d058b3be1` | `M257-1024` | 6 | flydsl 6 |
| `sel_012afec9233551f3` | `M257-1024` | 4 | flydsl 4 |
| `sel_e796912f1ec39c18` | `M257-1024` | 4 | flydsl 4 |
| `sel_8b75ef0c8bb54299` | `M257-1024` | 2 | flydsl 2 |
| `sel_84eb205891d2bb01` | `M257-1024` | 2 | flydsl 2 |
| `sel_0ddae4b41f418a0e` | `M257-1024` | 4 | ck 2 · flydsl 2 |
| `sel_e9ffd49a2f1e70df` | `M257-1024` | 4 | ck 2 · flydsl 2 |
| `sel_6e3ae771bda4b3ac` | `M257-1024` | 6 | flydsl 6 |
| `sel_a6f1590c94ae9235` | `M257-1024` | 6 | flydsl 6 |
| `sel_ee3ed6cb5bc4be16` | `M257-1024` | 4 | flydsl 4 |
| `sel_586ee152ab0df64c` | `M257-1024` | 4 | flydsl 4 |
| `sel_0e277ef736ffcbd9` | `M257-1024` | 2 | flydsl 2 |
| `sel_7a100e89284e9cf3` | `M257-1024` | 2 | flydsl 2 |
| `sel_7c0d1e5aa075e081` | `M257-1024` | 2 | flydsl 2 |
| `sel_9f9de064f07850ad` | `M257-1024` | 2 | flydsl 2 |
| `sel_fa47c88a527cbe88` | `M257-1024` | 2 | ck 2 |
| `sel_975d824ae37de5ea` | `M257-1024` | 2 | ck 2 |
| `sel_2cccd6627e68b9e4` | `M257-1024` | 4 | ck 2 · flydsl 2 |
| `sel_f1fcb282ae4c6195` | `M257-1024` | 4 | ck 2 · flydsl 2 |
| `sel_b203ca997382677f` | `M1025-4096` | 2 | asm 2 |
| `sel_7e5783ecdae7aef3` | `M1025-4096` | 2 | asm 2 |
| `sel_aef4c2643c5e5a26` | `M1025-4096` | 2 | asm 2 |
| `sel_beea95fffbc35218` | `M1025-4096` | 2 | asm 2 |
| `sel_bcdcaf1c443afbf6` | `M1025-4096` | 2 | asm 2 |
| `sel_1ec5689fe402a334` | `M1025-4096` | 2 | asm 2 |
| `sel_1ab9c3d22d331d96` | `M1025-4096` | 2 | flydsl 2 |
| `sel_36d621b4fff54156` | `M1025-4096` | 2 | flydsl 2 |
| `sel_fac6a7346ef5abd4` | `M1025-4096` | 4 | asm 2 · flydsl 2 |
| `sel_68e639ee1332caed` | `M1025-4096` | 2 | flydsl 2 |
| `sel_aeb30e7bf79c5406` | `M1025-4096` | 2 | asm 2 |
| `sel_506c6a2a451184d5` | `M1025-4096` | 2 | flydsl 2 |
| `sel_b0abd4f6d10eb300` | `M1025-4096` | 2 | flydsl 2 |
| `sel_0188ad03af04ef79` | `M1025-4096` | 2 | flydsl 2 |
| `sel_d40d47d1f18574d7` | `M1025-4096` | 2 | flydsl 2 |
| `sel_8417b7cb024d341e` | `M1025-4096` | 4 | flydsl 2 · asm 1 · ck 1 |
| `sel_dad6ad1c4bed8286` | `M1025-4096` | 3 | flydsl 2 · ck 1 |
| `sel_a9c852bfbdb63e3d` | `M1025-4096` | 2 | flydsl 2 |
| `sel_69e4326530b4279e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_c38b40892baa23b5` | `M1025-4096` | 4 | asm 2 · flydsl 2 |
| `sel_1b3eba53e61b1675` | `M1025-4096` | 2 | flydsl 2 |
| `sel_14901228c52a878a` | `M1025-4096` | 2 | asm 1 · ck 1 |
| `sel_b2af9efbf2dc47a3` | `M1025-4096` | 1 | ck 1 |
| `sel_e400785dc9688177` | `M1025-4096` | 2 | asm 2 |
| `sel_f2ce14fe15a516c1` | `M1025-4096` | 6 | ck 6 |
| `sel_fd3290e496513c70` | `M1025-4096` | 6 | ck 6 |
| `sel_70fd9e93074b41cf` | `M1025-4096` | 2 | asm 2 |
| `sel_b209d1b16658c630` | `M1025-4096` | 2 | ck 2 |
| `sel_9400c712c6e096c8` | `M1025-4096` | 2 | asm 2 |
| `sel_ec470dd513ac6566` | `M1025-4096` | 2 | ck 2 |
| `sel_64aa3976a86d436f` | `M1025-4096` | 2 | asm 2 |
| `sel_da6a88e65aa88522` | `M1025-4096` | 2 | asm 2 |
| `sel_63a9134ea6a0e589` | `M1025-4096` | 2 | asm 1 · ck 1 |
| `sel_4b0ef0a02e533a95` | `M1025-4096` | 1 | ck 1 |
| `sel_1b6e49cd04ed5888` | `M1025-4096` | 4 | flydsl 4 |
| `sel_c09070c10a3b3a9d` | `M1025-4096` | 4 | flydsl 4 |
| `sel_2ce36ad21693c033` | `M1025-4096` | 4 | flydsl 4 |
| `sel_a04898651f604129` | `M1025-4096` | 4 | flydsl 4 |
| `sel_887c079521a0c193` | `M1025-4096` | 2 | flydsl 2 |
| `sel_55d8541437595584` | `M1025-4096` | 2 | flydsl 2 |
| `sel_920be2cdfae294ed` | `M1025-4096` | 2 | flydsl 2 |
| `sel_9094b3ce06befc58` | `M1025-4096` | 2 | flydsl 2 |
| `sel_1b85872a793427ca` | `M1025-4096` | 2 | flydsl 2 |
| `sel_b8659fd4c202cdbf` | `M1025-4096` | 2 | flydsl 2 |
| `sel_8d5524a30238741e` | `M1025-4096` | 4 | asm 2 · flydsl 2 |
| `sel_b37709a51c4c4dc0` | `M1025-4096` | 2 | ck 1 · flydsl 1 |
| `sel_e3d6352e0995cf58` | `M1025-4096` | 6 | flydsl 6 |
| `sel_62e02d1e70065cf3` | `M1025-4096` | 6 | flydsl 6 |
| `sel_a19e44339e706d17` | `M1025-4096` | 4 | flydsl 4 |
| `sel_8019d39b60cb7bb7` | `M1025-4096` | 4 | flydsl 4 |
| `sel_0869262578c5fc0a` | `M1025-4096` | 2 | flydsl 2 |
| `sel_a28ddb9e7c2d430e` | `M1025-4096` | 2 | flydsl 2 |
| `sel_4ab07c7122e854c6` | `M1025-4096` | 4 | ck 2 · flydsl 2 |
| `sel_2cc03644abcb8da7` | `M1025-4096` | 4 | ck 2 · flydsl 2 |
| `sel_08970a44a74e0bad` | `M1025-4096` | 6 | flydsl 6 |
| `sel_52afae6455a3e253` | `M1025-4096` | 6 | flydsl 6 |
| `sel_a73e060e443f52cf` | `M1025-4096` | 4 | flydsl 4 |
| `sel_66ab5966d4d75a2e` | `M1025-4096` | 4 | flydsl 4 |
| `sel_a6e76278c6a92ef4` | `M1025-4096` | 2 | flydsl 2 |
| `sel_f2b59a8315a79567` | `M1025-4096` | 2 | flydsl 2 |
| `sel_7efda841dc3e3de9` | `M1025-4096` | 2 | flydsl 2 |
| `sel_8a53a29b2aeef3cf` | `M1025-4096` | 2 | flydsl 2 |
| `sel_cdc2c55fb5b774c1` | `M1025-4096` | 2 | ck 2 |
| `sel_392468a6fbdec9f6` | `M1025-4096` | 2 | ck 2 |
| `sel_fdb26d38150c76a8` | `M1025-4096` | 4 | ck 2 · flydsl 2 |
| `sel_21d8a5b44eea408d` | `M1025-4096` | 4 | ck 2 · flydsl 2 |
| `sel_397350694831cb90` | `M>4096` | 7 | asm 7 |
| `sel_f9e5960a64788d82` | `M>4096` | 7 | asm 7 |
| `sel_3b4caccd891057d2` | `M>4096` | 7 | asm 7 |
| `sel_bf7bc4b337c3b1cb` | `M>4096` | 7 | asm 6 · ck 1 |
| `sel_f1d553172b3b3da9` | `M>4096` | 1 | ck 1 |
| `sel_a97f0e27632d63d9` | `M>4096` | 7 | asm 7 |
| `sel_c179b73a963239a1` | `M>4096` | 7 | asm 6 · ck 1 |
| `sel_6850dbe6f8864587` | `M>4096` | 1 | ck 1 |
| `sel_30e9251855753ec9` | `M>4096` | 5 | flydsl 5 |
| `sel_e6765cc2d27889bc` | `M>4096` | 5 | flydsl 5 |
| `sel_cee2a3e9363bd46d` | `M>4096` | 12 | asm 6 · flydsl 6 |
| `sel_dd62f855f777eea3` | `M>4096` | 6 | flydsl 6 |
| `sel_961ac0242c835592` | `M>4096` | 6 | asm 6 |
| `sel_e839756fbc401ebb` | `M>4096` | 6 | flydsl 6 |
| `sel_f836d495635faa5d` | `M>4096` | 6 | flydsl 6 |
| `sel_45789edf377f0e65` | `M>4096` | 5 | flydsl 5 |
| `sel_3a33b0ae4fbad0b3` | `M>4096` | 5 | flydsl 5 |
| `sel_52c5fafc732e17cd` | `M>4096` | 12 | asm 6 · flydsl 6 |
| `sel_4b50ad9e5aa319c3` | `M>4096` | 6 | flydsl 6 |
| `sel_eb0af5abbdf6ac36` | `M>4096` | 5 | flydsl 5 |
| `sel_cb080f8edbb138c8` | `M>4096` | 5 | flydsl 5 |
| `sel_d0e03e6946e36dfc` | `M>4096` | 12 | asm 6 · flydsl 6 |
| `sel_bb86e051354c375a` | `M>4096` | 6 | flydsl 6 |
| `sel_d95caa4b80cf182f` | `M>4096` | 6 | asm 3 · ck 3 |
| `sel_d9ee3d79304ae924` | `M>4096` | 3 | ck 3 |
| `sel_6fdc6ed3bc71109c` | `M>4096` | 7 | asm 5 · ck 2 |
| `sel_3fe2ba31dc2b4d5b` | `M>4096` | 2 | ck 2 |
| `sel_180d323a30f9fdb8` | `M>4096` | 18 | ck 18 |
| `sel_c706b1d2a9b3e8b1` | `M>4096` | 18 | ck 18 |
| `sel_b6bb90010ef4f134` | `M>4096` | 6 | ck 6 |
| `sel_2f8dcb8e0a3fa74c` | `M>4096` | 6 | ck 6 |
| `sel_7a1bf46569ab58a3` | `M>4096` | 7 | ck 7 |
| `sel_4562c9e182ebf48c` | `M>4096` | 7 | ck 7 |
| `sel_8426bc617f4bebca` | `M>4096` | 7 | asm 7 |
| `sel_e8b911dd015625dc` | `M>4096` | 7 | asm 7 |
| `sel_97ff1e321f90a1d0` | `M>4096` | 2 | ck 2 |
| `sel_d9a6434310ffdad3` | `M>4096` | 7 | asm 6 · ck 1 |
| `sel_a52888b63c4f3bcb` | `M>4096` | 1 | ck 1 |
| `sel_61b64a296d39ebeb` | `M>4096` | 9 | flydsl 9 |
| `sel_c1e107bd29d2ac09` | `M>4096` | 9 | flydsl 9 |
| `sel_1e2f1ac547d3e44d` | `M>4096` | 12 | flydsl 12 |
| `sel_1c5f55f4e470c869` | `M>4096` | 12 | flydsl 12 |
| `sel_a82db104d051e490` | `M>4096` | 4 | flydsl 4 |
| `sel_905bef8bb3050f4f` | `M>4096` | 4 | flydsl 4 |
| `sel_652aef88d6548a82` | `M>4096` | 6 | flydsl 6 |
| `sel_6bc6814078ce0304` | `M>4096` | 6 | flydsl 6 |
| `sel_6395a1b764fee186` | `M>4096` | 1 | flydsl 1 |
| `sel_bf97fbabf89a62d0` | `M>4096` | 1 | flydsl 1 |
| `sel_e0deacfb9deae8c6` | `M>4096` | 6 | flydsl 6 |
| `sel_3204efa738caba8f` | `M>4096` | 6 | flydsl 6 |
| `sel_f91fc79b354d978f` | `M>4096` | 14 | asm 7 · flydsl 7 |
| `sel_e6b48ba2020a0425` | `M>4096` | 7 | flydsl 7 |
| `sel_fa16711bdb6af901` | `M>4096` | 19 | flydsl 19 |
| `sel_07d7c053dc5afeff` | `M>4096` | 19 | flydsl 19 |
| `sel_852b35bcd0e7f713` | `M>4096` | 20 | flydsl 20 |
| `sel_2b0efc09987bb63f` | `M>4096` | 20 | flydsl 20 |
| `sel_b1eecbaf027e9af2` | `M>4096` | 6 | flydsl 6 |
| `sel_4f9254e34b3bdd7e` | `M>4096` | 6 | flydsl 6 |
| `sel_26a1bb1fbf0f8ecc` | `M>4096` | 12 | flydsl 7 · asm 4 · ck 1 |
| `sel_c3ab402abd79fb5b` | `M>4096` | 8 | flydsl 7 · ck 1 |
| `sel_c17bdacdce14d60d` | `M>4096` | 19 | flydsl 19 |
| `sel_7d5d207a7dcffd60` | `M>4096` | 19 | flydsl 19 |
| `sel_758092c58c0afc98` | `M>4096` | 20 | flydsl 20 |
| `sel_449851dadef8e269` | `M>4096` | 20 | flydsl 20 |
| `sel_38a22c6bdfff0873` | `M>4096` | 7 | flydsl 7 |
| `sel_aac1e5d93ac907b0` | `M>4096` | 7 | flydsl 7 |
| `sel_0c364affb8e6ba4a` | `M>4096` | 7 | flydsl 7 |
| `sel_3252e0d4e7934d3c` | `M>4096` | 7 | flydsl 7 |
| `sel_090c42d1d4a606ec` | `M>4096` | 6 | ck 6 |
| `sel_a88bed8c73800206` | `M>4096` | 6 | ck 6 |
| `sel_d4561019b7a5d1e3` | `M>4096` | 14 | ck 7 · flydsl 7 |
| `sel_e99f13889a8ffd7c` | `M>4096` | 14 | ck 7 · flydsl 7 |

### `` · 304 CUs · `tuned_fmoe`

| selection ref | M bucket | rows | selected backend (rows) |
|---|---|---|---|
| `sel_84ee5bd2f76fae4e` | `M<=16` | 2 | ck 2 |
| `sel_7a0f26a969b3fc06` | `M<=16` | 2 | ck 2 |
| `sel_07f9c4dbe13ca198` | `M<=16` | 2 | ck 2 |
| `sel_a7cb39f7a59b02a2` | `M<=16` | 2 | ck 2 |
| `sel_684e00dfa28a25e5` | `M<=16` | 1 | asm 1 |
| `sel_a51f4fb93c35412b` | `M<=16` | 1 | ck 1 |
| `sel_2bbb93fbed01f799` | `M17-64` | 2 | ck 2 |
| `sel_a8cfbb5e62cd479b` | `M17-64` | 2 | ck 2 |
| `sel_c74422decb58bdf2` | `M17-64` | 2 | ck 2 |
| `sel_23796a3e38446f71` | `M17-64` | 2 | ck 2 |
| `sel_25c28edea53a3a93` | `M17-64` | 2 | asm 2 |
| `sel_127d5a4c57e58222` | `M17-64` | 2 | ck 2 |
| `sel_b0d1aa8e863c98cb` | `M65-256` | 2 | ck 2 |
| `sel_ed3f33437b4f930c` | `M65-256` | 2 | ck 2 |
| `sel_285705d2f16579f7` | `M65-256` | 2 | ck 2 |
| `sel_74e39ab3e3700df4` | `M65-256` | 2 | ck 2 |
| `sel_216462a2e542100f` | `M65-256` | 2 | asm 2 |
| `sel_c7892d8527ba8e5c` | `M65-256` | 2 | ck 2 |
| `sel_74a7c5da54176ba0` | `M257-1024` | 2 | ck 2 |
| `sel_a677628b8f161df9` | `M257-1024` | 2 | ck 2 |
| `sel_c996d9fc388504d2` | `M257-1024` | 2 | ck 2 |
| `sel_87b5893f1d34d098` | `M257-1024` | 2 | ck 2 |
| `sel_6a9b57acc91a85e5` | `M257-1024` | 2 | asm 2 |
| `sel_f98c1cd5c514e466` | `M257-1024` | 2 | ck 2 |
| `sel_eededbbddc298d2e` | `M1025-4096` | 2 | ck 2 |
| `sel_f4e19e3efdcaa866` | `M1025-4096` | 2 | ck 2 |
| `sel_212e39eb8d63cc02` | `M1025-4096` | 2 | ck 2 |
| `sel_766624fab73f2561` | `M1025-4096` | 2 | ck 2 |
| `sel_c64ced2e4880af9c` | `M1025-4096` | 2 | asm 2 |
| `sel_7258493f128afbe7` | `M1025-4096` | 2 | ck 2 |
| `sel_a3a9d2cea9ef35a3` | `M>4096` | 6 | ck 6 |
| `sel_c77ccf9d964f0c90` | `M>4096` | 6 | ck 6 |
| `sel_d5e658579e1c5fc5` | `M>4096` | 6 | ck 6 |
| `sel_d32bfd5ef633cad1` | `M>4096` | 6 | ck 6 |
| `sel_64279901a37a5657` | `M>4096` | 6 | asm 6 |
| `sel_a109f7028b9ca284` | `M>4096` | 6 | ck 6 |

Databases read but not counted as selections:

- `aiter/configs/a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/bf16_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/dsr1_a8w8_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/dsr1_a8w8_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/kimik3_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/q3vl_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/qwen38_27b_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv` — 48 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/bf16_tuned_gemm.csv` — 5 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5.2.csv` — 24 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_glm5_3_flash.csv` — 9 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_kimik3.csv` — 254 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_qwen3_5_397b_a17b_mxfp4_attnfp8.csv` — 136 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv` — 47 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/dsv41_bf16_tuned_gemm.csv` — 131 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/dsv4_bf16_tuned_gemm.csv` — 501 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/glm53_flash_bf16_tuned_gemm.csv` — 96 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv` — 197 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv` — 101 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/inkling_bf16_tuned_gemm.csv` — 2 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/kimi_bf16_tuned_gemm.csv` — 13 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv` — 90 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/kimik3_bf16_tuned_gemm.csv` — 245 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv` — 115 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv` — 98 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/minimax_m3_eagle_bf16_tuned_gemm.csv` — 121 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/minimax_m3_mxfp4_bf16_tuned_gemm.csv` — 98 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv` — 94 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/qwen3_5_397b_bf16_tuned_gemm.csv` — 94 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/qwen3_8_2p4t_a95b_bf16_tuned_gemm.csv` — 19 flydsl kernelName(s) in neither shipped naming form; skipped, not guessed
- `aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv` — 5 row(s) name a `cktile…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv` — 9 row(s) name a `opus…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/gptoss_a16w4_tuned_fmoe.csv` — 24 row(s) name a `swiglu…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv` — 14 row(s) name a `cktile…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/hunyuan3_fp8_per_tensor_tuned_fmoe.csv` — 17 row(s) name a `impl…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/kimik3_a8w4_tuned_fmoe.csv` — 2 row(s) name a `opus…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/minimax_m3_fp8_ptpc_tuned_fmoe.csv` — 25 row(s) name a `impl…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/qwen3_5_35b_fp8_ptpc_tuned_fmoe.csv` — 26 row(s) name a `impl…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/qwen3_5_397b_fp8_ptpc_tuned_fmoe.csv` — 15 row(s) name a `impl…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/a8w8_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/bf16_tuned_batched_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/dsv3_a4w4_blockscale_tuned_gemm.csv` — no `libtype` column: the selected backend is not recorded, so these rows are inventoried but not counted as a selection
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv` — 7 row(s) name a `cktile…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed
- `aiter/configs/model_configs/kimik2_fp8fp4_tuned_fmoe.csv` — 32 row(s) name a `cktile…` stage-1 kernel, which is none of the shipped naming forms; the backend is left unattributed

Databases inventoried: 156 (70394 rows).

## Other backends' shipped seeds

The 2042 Triton configuration groups AITER ships under `aiter/ops/triton/configs/gemm` are rendered on [`gemm_triton_seeds.md`](gemm_triton_seeds.md). They are Triton knob names; open that page when the Triton baseline's own config is the question, not when choosing FlyDSL knobs.

## Sources

- Curated cards: [`decisions/gemm.yaml`](decisions/gemm.yaml).
- Source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml).
- Shipped tuned databases (FlyDSL seeds and backend selection): [`evidence/gemm_csv_tuned_configs.yaml`](evidence/gemm_csv_tuned_configs.yaml).
- Shipped Triton configs: [`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml).
- Performance axes: [`performance_axes/gemm.yaml`](performance_axes/gemm.yaml).
- Generated by [`_render_decisions.py`](_render_decisions.py); edit the cards or evidence, never this file.
