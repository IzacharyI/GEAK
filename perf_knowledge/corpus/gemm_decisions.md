# GEMM development decision cards

This is the actionable layer. It does not dump regex matches and ask the reader to infer a recommendation. Every curated card states **when it applies, what to try, why, alternatives, evidence strength and limits**. The raw, reproducible source observations remain in [`gemm_source_evidence.md`](gemm_source_evidence.md).

Source baseline: AITER `a6bb499375849eec45d68c5ccaebc8865fd422c0` · 8 curated decision card(s) · 1083 source-evidence records · 306 shipped tuning groups.

Evidence levels:

- `source_observed`: implementation precedent only; add a candidate and measure it.
- `shipped_config`: parameter seed selected in AITER's source tree; alternatives and benchmark results are not attached, so vary and measure locally.
- Measured guidance is deliberately not copied here: it stays in learned cards/expert skills behind their existing feature switches.

## Which FlyDSL GEMM paths does gfx942 (MI300X) actually give you, and which does it silently switch or refuse?

**Card:** `flydsl-gfx942-path-gates` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- target architecture is gfx942 — read this before costing any other card on this page

### Try

- Read the architecture gates before the tuning tables. Whether a path exists on your box outranks every fact about how to tune it, and four separate gates in this library key on gfx942 alone.
- Do not fund async copy on gfx942. The library computes its async-copy flag as `architecture is not gfx942`, so the flag is already false here and the split-K kernel hard-codes async off on this branch. There is no knob to flip.
- Do not fund the small-M kernel family on gfx942. Its compile entry point raises, and its config registry returns nothing. Port the design if you want it; you cannot call it.
- Expect the m16n16k16 fragment with 4-byte DMA and two MFMA steps per warp-K, not the m16n16k32 / 16-byte / one-step bundle the other architectures get. The choice is made for you by one architecture test — see flydsl-half-mfma-call-forms.
- Note that the preshuffle family does run here, but loads A four bytes at a time instead of sixteen; price that into any preshuffle comparison rather than assuming the gfx950 behaviour — see flydsl-preshuffle-b-layout-contract.
- On any other architecture, invert all of the above: the wider fragment, the 16-byte DMA, async copy and the small-M family are all in play.

### Why this is a candidate

- Four independent gfx942 tests exist in the FlyDSL GEMM sources and they do different things: one switches the fragment/DMA/MFMA bundle, one disables async copy library-wide, two shut the small-M family, one narrows the preshuffle load width.
- This is the cheapest knowledge on the page. A gate costs nothing to check and a whole round to discover, and none of it is visible from the tuning tables, which record what was selected on the architectures where the path was open.

### Keep as alternatives

- On gfx942, spend the search on what remains genuinely open — the tile geometry, split_k, B staging and the epilogue — rather than on the gated paths.
- If a gated path is the real ceiling for your shape, vendor and port it deliberately as a structural change, and budget it as one.

### Evidence

- `src_c4c19187bcb7933c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/gemm_kernels.py:41`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331`
- `src_42146c2d073d655e` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/preshuffle_gemm.py:243`

### Limits

- These are the gates that name an architecture literally. A path can also be effectively closed by something this card does not see — a missing dtype, an LDS budget, a compiler version.
- Gates move between releases. Every citation carries its file and line; re-read them against the checkout you are building on before trusting the list.
- That a path is open says nothing about whether it is fast.

## Which legal MFMA call forms can seed a FlyDSL f16/bf16 GEMM?

**Card:** `flydsl-half-mfma-call-forms` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- input_dtype in {f16,bf16}

### Try

- For an m16n16k16 candidate, pair bf16 with mfma_f32_16x16x16bf16_1k and f16 with mfma_f32_16x16x16f16.
- Check the target architecture before treating the K shape as a free axis: the shipped split-K kernel does not let you pick it. On gfx942 it selects m16n16k16 with 4-byte DMA and two MFMA steps per warp-K; on every other architecture it selects m16n16k32 with 16-byte DMA and one. Benchmarking the K=32 form on gfx942 means rewriting that branch, not passing a flag.
- Also benchmark the m16n16k32 bf16/f16 forms where the architecture branch does select them and the fragment shape and register budget allow.

### Why this is a candidate

- AITER's FlyDSL split-K GEMM implements both K=16 and K=32 classes and branches explicitly on bf16 versus f16.
- The two classes are not offered side by side at runtime: one architecture test picks the WMMA implementation, the DMA width and the MFMA-per-warp-K count together, as one bundle.
- This establishes legal implementation precedents; it does not establish which form is faster for a new shape.

### Keep as alternatives

- Use the m16n16k16 form when the larger-K fragment does not fit the surrounding tile or register budget.
- Let a higher-level backend select the instruction when direct MFMA control is not required.

### Evidence

- `src_24517333d1cd6722` — `mfma_intrinsic` `mfma_f32_16x16x16bf16_1k` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:56`
- `src_0ed3b5780c589479` — `mfma_intrinsic` `mfma_f32_16x16x16f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:59`
- `src_387e95ce58e0449b` — `mfma_intrinsic` `mfma_f32_16x16x32_bf16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:77`
- `src_79cfb57473ed817f` — `mfma_intrinsic` `mfma_f32_16x16x32_f16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:80`
- `src_e689c0cbe1b4f8e8` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:128`

### Limits

- No latency or occupancy comparison is attached; treat both K shapes as candidates and re-measure.
- API spelling is version-sensitive; consult languages/flydsl/version_map.md before copying.

## How should a FlyDSL GEMM apply an XOR16 LDS layout without making stores and loads disagree?

**Card:** `flydsl-xor16-lds-layout-consistency` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- operands are staged through LDS
- XOR16 is included as a bank-conflict candidate

### Try

- Define the byte-column transform once as col_in_bytes XOR ((row modulo k_blocks16) times 16).
- Apply the identical transform at all four sites: the global-to-LDS store and the LDS-to-fragment load, for A and for B.

### Why this is a candidate

- The AITER implementation calls one helper at every write and read site; copying only one side changes addressing semantics rather than performance.
- The cited sites are A store, A fragment load, B store and B fragment load — the set that has to move together.

### Keep as alternatives

- Keep the unswizzled layout as the control arm.
- Benchmark a different padding or swizzle scheme when the tile geometry changes.

### Evidence

- `src_2e329cfe46a52fb9` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:27`
- `src_d1f233838e74f022` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:455`
- `src_6f222b94149ab249` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:530`
- `src_d2c808ad3cd9ce7c` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:625`
- `src_0438c6e24a955ea2` — `lds_swizzle` `swizzle_xor16` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:675`

### Limits

- The citations prove address consistency, not that XOR16 reduces bank conflicts on every tile.
- Retain the unswizzled implementation and choose by target-box measurement.

## Which tile and split-K values will the FlyDSL HGEMM family actually accept, and how does a shape narrow them?

**Card:** `flydsl-hgemm-config-space` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- a configuration is being chosen or searched for a known M, N, K

### Try

- Search tile_m within (16, 32, 48, 64, 80, 96, 112, 128, 160, 256), tile_n within (64, 128, 160, 192, 256) and tile_k within (64, 96, 128, 160, 256) rather than treating the 128x128x64 default as the space.
- Cap tile_m at max(96, align_up(2*M, 16)): a tile_m far above M pays for rows that do not exist, and the shipped derivation refuses those candidates outright.
- Take split_k from 1..32, keeping only values that divide K and leave K/split_k divisible by tile_k; every candidate must clear that filter, including 1, 2, 4, 8 and 16. Those five need nothing further; any other divisor also has to leave between 2 and 8 block-K loops per split.
- For a skinny M (a few rows against a deep K) start the search at small tile_m with a split_k above 1, not at the default.

### Why this is a candidate

- AITER states the space as module-level option tuples and two derivation helpers, not as one default; the default is one point inside it.
- The derivations encode the shape-dependence directly — tile_m is bounded by M, and split_k by the divisibility of K — so the narrowing rule is readable rather than guessed.
- This card says which candidates are legal and worth generating. It does not say which one wins; that is what the benchmark is for.

### Keep as alternatives

- Keep the library default 128x128x64 with split_k=1 as the control arm to measure against.
- Fix the tile and search only split_k when N or K constrain the tiling to one legal shape.

### Evidence

- `src_426b8960fb14a04f` — `config_space` `HGEMM_TILE_M_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:74`
- `src_62ce0ec6b152ecaa` — `config_space` `HGEMM_TILE_N_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:72`
- `src_3c72689d786ca8be` — `config_space` `HGEMM_TILE_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:73`
- `src_fce6be7eb6b247d6` — `config_space` `HGEMM_BASE_SPLIT_K_OPTIONS` at `aiter/ops/flydsl/gemm_kernels.py:75`
- `src_96320a5bab17a8f5` — `config_limit` `HGEMM_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/gemm_kernels.py:76`
- `src_562e9940a52a202f` — `config_space` `_hgemm_tile_m_options` at `aiter/ops/flydsl/gemm_kernels.py:222`
- `src_642476c9b693d5f1` — `config_space` `_hgemm_split_k_options` at `aiter/ops/flydsl/gemm_kernels.py:229`

### Limits

- These are the values the library is willing to compile, not values anybody measured as best; every candidate still has to be timed on the target box.
- The option tuples are the generic HGEMM family's. The small-M family has its own, wider, N space — see flydsl-small-m-hgemm-family.
- A candidate that is inside the space can still be rejected at compile time by the tiling and LDS rules — see flydsl-hgemm-tiling-validity.

## Which FlyDSL HGEMM configurations will fail to build at all, before any of them can be slow?

**Card:** `flydsl-hgemm-tiling-validity` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- candidate configurations are being generated automatically or swept

### Try

- Filter candidates before compiling them: N must satisfy N >= tile_n and N % tile_n == 0, and K/split_k must satisfy K/split_k >= tile_k and (K/split_k) % tile_k == 0.
- Require tile_k >= 32 and tile_k % 32 == 0, tile_m % (block_m_warps*16) == 0 and tile_n % (block_n_warps*16) == 0.
- Require each of tile_m*tile_k, tile_n*tile_k and tile_m*tile_n to be divisible by 8*64*block_m_warps*block_n_warps, the per-block vectorised load width.
- Estimate LDS as a_lds = max(stages*tile_m*tile_k*2, tile_m*tile_n*2) when B is not staged through LDS; when it is, the estimate is not a_lds plus something, it becomes align_up(a_lds,16) + stages*tile_n*tile_k*2. Drop candidates over the block budget.
- Do not spend a round on stages or pack_n in the generic family: the kernel compiles a fixed 2-stage pipeline and rejects any stages other than that, and rejects pack_n other than 1. They read like tuning knobs in the signature and are not.
- Check the split-K reduction counter capacity separately; the shipped counter is sized for a bounded number of output tiles.

### Why this is a candidate

- AITER performs all of these checks in host code before the kernel is built, so they are the stated contract rather than an inference from a crash.
- An optimizer that cannot separate 'illegal' from 'slow' spends its budget re-discovering the first, and a budgeted search has few rounds to spend.

### Keep as alternatives

- Let the compiler reject the configuration and catch the exception — acceptable only when the search is small, since each rejection still costs a compile.

### Evidence

- `src_9ae627bec0096537` — `config_validity` `_validate_hgemm_tiling` at `aiter/ops/flydsl/gemm_kernels.py:330`
- `src_d3f3cb90295b0c40` — `config_validity` `_estimate_hgemm_lds_bytes` at `aiter/ops/flydsl/gemm_kernels.py:249`
- `src_5c7e74ddb8a071de` — `config_validity` `_check_split_k_counter_capacity` at `aiter/ops/flydsl/gemm_kernels.py:695`
- `src_7c7ddd401ff4aaeb` — `config_limit` `SPLIT_K_COUNTER_MAX_LEN, 128` at `aiter/ops/flydsl/gemm_kernels.py:37`
- `src_c139f07a63569327` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MIN, 2` at `aiter/ops/flydsl/gemm_kernels.py:77`
- `src_539c4b7bf95dca07` — `config_limit` `HGEMM_EXTRA_BLOCK_K_LOOPS_MAX, 8` at `aiter/ops/flydsl/gemm_kernels.py:78`

### Limits

- The LDS formula is the library's own estimate for this kernel shape; a rewritten staging scheme changes it and the estimate stops applying.
- Passing every check means the configuration builds, not that it is fast, and not that it is correct in a modified kernel.

## What does AITER do differently when M is a handful of rows, and which knobs does that open?

**Card:** `flydsl-small-m-hgemm-family` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- M is small — the shipped routing threshold is M < 17
- dtype is bf16

### Try

- Check the architecture first, because on gfx942 this family is not available at all: the compile entry point raises outright, and the config registry returns an empty set before it even looks at the shape. On gfx942 read this card as a porting reference — the design is still the right one for a few rows against a deep K — but do not spend a round calling it, and do not read a shipped small-M config table as something you can pass to this box.
- Treat small M as a different kernel family rather than the generic kernel with a smaller tile: AITER routes it to a separate implementation selected by an explicit kernel_family value.
- In that family tile_m is fixed at 16 with block_m_warps=1 and 2 stages — the M dimension stops being a search axis, and the search moves entirely onto N and K.
- Search tile_n far wider than the generic family allows — the shipped set runs (32, 64, 96, 128, 160, 192, 224, 256, 384, 512, 768, 1024) — with tile_k in (32, 64, 96, 128, 160, 192, 256) and split_k up to 32.
- Consider the axes this family adds and the generic one does not: n_tile_repeat, persistent_n_tiles, waves_per_eu and a b_to_lds unroll factor.
- Respect the combination rules: n_tile_repeat > 1 only without B-in-LDS and only for the two shipped shapes; persistent_n_tiles > 1 only with B-in-LDS, tile_n >= 128, block_n_warps >= 2 and no more persistent tiles than N/tile_n.

### Why this is a candidate

- A skinny-M GEMM is bounded by how much of the machine the few M tiles can occupy, and every axis this family adds — repeat, persistent tiles, waves per EU — is an occupancy axis rather than a tiling one.
- The routing threshold, the fixed tile_m and the option sets are stated as named constants in the source, so the boundary between the two families is readable rather than folklore.

### Keep as alternatives

- Stay on the generic family with tile_m=16 when a second kernel cannot be written or maintained; that is inside the generic space and is the cheaper first move.
- Keep the generic default configuration as the control arm regardless of which family is chosen.

### Evidence

- `src_38699a8ce57faf0c` — `kernel_family` `KERNEL_FAMILY_HGEMM, hgemm` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:8`
- `src_8ffea30406cf0355` — `kernel_family` `KERNEL_FAMILY_SMALL_M, small_m` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:9`
- `src_bca963564a69909c` — `kernel_family` `KERNEL_FAMILY_SMALL_M` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:56`
- `src_b446c07f43b11542` — `config_limit` `SMALL_M_KERNEL_MAX, 17` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:69`
- `src_034b3cc6e3fd6aa8` — `tile_shape` `TILE_M, 16` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:70`
- `src_390ae8578e28ce34` — `config_space` `SMALL_M_TILE_N_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:83`
- `src_60717171485c58f5` — `config_space` `SMALL_M_TILE_K_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:81`
- `src_ee5e839c6374f38f` — `config_limit` `SMALL_M_MAX_SPLIT_K, 32` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:82`
- `src_a7c0de9870934fc0` — `config_space` `SMALL_M_N_TILE_REPEAT_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:102`
- `src_28936b493eeb6f6f` — `config_space` `SMALL_M_PERSISTENT_N_TILE_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:103`
- `src_1dcdf9c7b6794c6e` — `config_space` `SMALL_M_NON_B_TO_LDS_WAVES_PER_EU_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:97`
- `src_44b5602aacc7e7df` — `config_space` `SMALL_M_B_TO_LDS_UNROLL_OPTIONS` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:101`
- `src_cceaeb92fb86b540` — `config_validity` `_validate_small_m_registry_config` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:172`
- `src_bc6d04baf2182ba4` — `config_limit` `MAX_LDS_BYTES, 163840` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:76`
- `src_7eee3f6769348b4c` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:411`
- `src_e04716b6d760605b` — `arch_gate` `gfx942` at `aiter/ops/flydsl/kernels/small_m_hgemm.py:331`

### Limits

- On gfx942 none of this is callable; see the first action. The rest of the card describes the gfx950 path.
- That AITER maintains a separate family for small M is a design precedent, not a measurement that it beats a well-tuned generic configuration on your shape.
- The option sets are what the shipped registry enumerates; they are candidates to search, and nothing here says which member wins.
- The family is bf16-only and its LDS budget is its own; do not carry its constants into the generic kernel.

## What does enabling B preshuffle on a FlyDSL HGEMM actually require?

**Card:** `flydsl-preshuffle-b-layout-contract` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- B preshuffle is being considered as a candidate

### Try

- Treat b_preshuffle as a change of B's storage layout, not as a boolean tuning flag: the kernel expects B already permuted into the family's shuffle layout, and the host side must either shuffle it or be told it is pre-shuffled.
- Keep b_to_lds=false whenever b_preshuffle=true; the current kernel rejects the combination.
- Keep b_preshuffle=false on the small-M family, which rejects it outright.
- Count the one-off shuffle cost against the number of launches that reuse the same B before claiming a win; for a weight matrix reused across many calls it amortises, for a single call it does not.

### Why this is a candidate

- The generic HGEMM path supports preshuffled B and its dispatch default is even b_preshuffle=true, so this is a supported layout rather than a forbidden one — but the guard rails around it are all layout guard rails.
- The two rejections in the source are specific and different in kind: an unsupported combination (with b_to_lds) and an unsupported family (small-M). Neither is a statement about speed.

### Keep as alternatives

- Use the unshuffled B path as the correctness and performance control.
- Shuffle B once at load time outside the timed region when the deployment allows it, and measure both accounting rules explicitly.

### Evidence

- `src_93c0f2962c52fdfb` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/kernels/hgemm_dispatch.py:32`
- `src_4c74ced4671a1e5b` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:162`
- `src_3ee379ecc88d0ad1` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:767`
- `src_7b5432b56d9acd63` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:768`
- `src_520242be3bd7400e` — `layout_preshuffle` `b_preshuffle` at `aiter/ops/flydsl/gemm_kernels.py:886`
- `src_e7db1b030bf2e5c6` — `layout_preshuffle` `shuffle_weight` at `aiter/ops/flydsl/gemm_kernels.py:883`

### Limits

- This is an API/layout compatibility rule, not evidence that preshuffle is profitable on any shape.
- A benchmark that reuses one pre-shuffled B across iterations measures the steady state, not the first call; state which one you are reporting.
- The card is grounded in the vendored library's contract. A kernel copy that has been edited may have moved these checks.

## What explicit instruction-order bundle can seed a hand-scheduled FlyDSL split-K GEMM hot loop?

**Card:** `flydsl-splitk-hot-loop-schedule-bundle` · **evidence:** `source_observed` · **status:** `candidate_only`

Source-observed candidate — the cited implementation exists; no performance preference is implied.

### Use when

- operator_family=gemm
- target_language=flydsl
- kernel uses a split-K hot loop with explicit ROCm scheduling directives

### Try

- Copy the ORDER, not a set of magic numbers: A-fragment ds-reads, then B-fragment ds-reads, then the A and B global loads as vmem, then the MFMA issues, closed by a scheduling barrier.
- Derive each group's count from the loop geometry that produces it — warp K steps times warp M or N steps for the ds-reads, the per-block LDG register counts for the vmem groups, and the full MFMA count for the mfma group — rather than hard-coding the counts observed here.
- Move or retune the bundle as a unit against the exact load/MFMA loop, then compare with the unscheduled control.

### Why this is a candidate

- The directives are adjacent inside one scheduler function and describe a single ordering pattern; copying one directive in isolation loses that context.
- An earlier version of this card recorded the literal group sizes vmem(2)/dsrd(2)/dsrd(4)/mfma(8). Upstream now emits unit-count directives inside geometry-derived loops, so those numbers described one build of one tile and expired quietly. The order survived the rewrite; the constants did not, which is the reason this card states a rule and not a bundle of integers.

### Keep as alternatives

- Leave instruction ordering to the compiler; that is the control arm and it is often the right answer.
- Use the second shipped scheduler shape instead, which interleaves vmem and mfma through a running counter rather than emitting the groups back to back.

### Evidence

- `src_5da6811a509d9806` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:688`
- `src_b0da890c19bc671e` — `scheduling` `sched_dsrd` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:690`
- `src_ed1134f4e4cea08c` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:694`
- `src_86d7e5af1117454e` — `scheduling` `sched_vmem` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:698`
- `src_0f18d7f50ce33cbb` — `scheduling` `sched_mfma` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:702`
- `src_dca1b81d9d2b2edf` — `scheduling` `sched_barrier` at `aiter/ops/flydsl/kernels/splitk_hgemm.py:703`

### Limits

- Instruction scheduling is the most build-specific thing in this corpus; treat it as a candidate to try late, after tiling and split-K are settled.
- Changing tile geometry, LDS stages or access width changes the schedule that must be re-measured.

## Shipped configuration seeds

These rows are generated from AITER's shipped selected configs. Match all three condition columns before using a row. **Seed candidate** means every config in that group carries the value; **vary next** lists knobs that changed by shape. No benchmark archive or rejected alternatives are attached, so these are concrete candidates, not measured winners.

**These are Triton knobs.** The shipped selected configs live under `aiter/ops/triton/configs/gemm`, so the names are Triton's — `BLOCK_SIZE_M/N/K`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, `cache_modifier`. Only the block tile carries over to another backend more or less directly; the rest have no one-to-one FlyDSL equivalent and several have none at all. If you are authoring FlyDSL, the cards above are the part of this file that applies to you, and a row here is at best a hint about which tile shapes somebody found worth shipping for a given M bucket.

### `gfx950`

| decision ref | variant | M bucket | seed candidate | vary next | shipped support |
|---|---|---|---|---|---|
| `cfg_a57a5cb4f0a527d8` | `BATCHED_GEMM A16W16` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ecf596d3e577f90d` | `BATCHED_GEMM A16W16` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5261f7f2c5f25018` | `BATCHED_GEMM A8W8` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_cfdc51455df2e87e` | `BATCHED_GEMM A8W8` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5ced01ef02696086` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_128` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_bc0272cd92f3b185` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_16` | `BLOCK_SIZE_M=16`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_03bfe2299071dce1` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_256` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_fa1ae965db1a48b1` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_32` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_ed808d7d9d6d4ed7` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_64` | `BLOCK_SIZE_M=32`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_N`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_ca82327242a25cf4` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `any` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_f10f24372b10fb51` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `GROUP_SIZE_M`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_d127d80b9f3a14d5` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_16` | `BLOCK_SIZE_M=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=6` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_da8a6b59be55236a` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4` | `BLOCK_SIZE_K`, `cache_modifier`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_20d1053fe04f9788` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_32` | `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_66ee477043230066` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_5fef76ee7080c42f` | `BATCHED_GEMM AFP4WFP4` | `any` | `BLOCK_SIZE_M=256`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE_M` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_49e12dbc5c296ca9` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_128` | `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_f4db0923be0a6cec` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_16` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_d1b71235005b7c2f` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_256` | `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_502bb17ba8bb36a5` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_32` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_388edc5997d185a5` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_64` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_0ea6944810ece167` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `any` | `NUM_KSPLIT=1`, `kpack=1`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_93ea0c7b211220be` | `FF A16W16 fused` | `M_LEQ_4` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=4`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d6e7a9afe9f27192` | `FF A16W16 fused` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6a0984dbc07e813e` | `FF A16W16 fused` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d909e533e711175b` | `FF A16W16 fused` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_77f4c818e9104e3d` | `FUSED GEMM A8W8_BLOCKSCALE A16W16` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e2b51f0f74b70343` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_1024` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6fed383ad23068ae` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=7`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_0958f0b047f6019a` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=8` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_85a4bd4aa70e52e4` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_2048` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8586453d2d8fdfd8` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=7`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_24a6d65f93a3fc42` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_dbf1819a41a24b19` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=32`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=6` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e2436223b4756e89` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=8` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b2018685134dd493` | `FUSED GEMM A8W8_BLOCKSCALE A16W16 N8=512 N16=256` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=32`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ef4a35028b46292f` | `FUSED GEMM AFP4WFP4 A16W16` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9f9159b70258044c` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=32`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_34e2be0bf8b760e2` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_16` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=8`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_4a3eb8739f35e148` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_fbac7e4df70655a7` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=32`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=7`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=6` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_84d46ea7681b8200` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=7`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6a469dc9d8b91c3c` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e6144161244b0ddd` | `FUSED GEMM AFP4WFP4 A16W16 N4=512 N16=256` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b1b0bcb837a19333` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ffdbc0cd21ce9840` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b560da5877ad79ee` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6454fed977948ba6` | `GEMM A16W16` | `M_LEQ_1024` | `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `waves_per_eu` | 5 selected shape config(s) in 5 JSON file(s) |
| `cfg_8e171982124777b2` | `GEMM A16W16` | `M_LEQ_128` | `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_b7ed22d964ee5af3` | `GEMM A16W16` | `M_LEQ_16` | `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_b470f107eed8b9b2` | `GEMM A16W16` | `M_LEQ_2048` | `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_23f66c585b31f85d` | `GEMM A16W16` | `M_LEQ_256` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_24bdc7e2b99f7d28` | `GEMM A16W16` | `M_LEQ_32` | `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_99912b6b3b12007b` | `GEMM A16W16` | `M_LEQ_4` | `BLOCK_SIZE_K=512`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_5efe7b2ba9b56b2c` | `GEMM A16W16` | `M_LEQ_4096` | `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `waves_per_eu` | 5 selected shape config(s) in 5 JSON file(s) |
| `cfg_2ebaf2f982bc72d4` | `GEMM A16W16` | `M_LEQ_512` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_34329609765e0c35` | `GEMM A16W16` | `M_LEQ_64` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_f71fd98f79713f88` | `GEMM A16W16` | `M_LEQ_8` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_bf6f4e95db577390` | `GEMM A16W16` | `M_LEQ_8192` | `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `waves_per_eu` | 5 selected shape config(s) in 5 JSON file(s) |
| `cfg_7fa388e056024686` | `GEMM A16W16` | `any` | `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_50866abdcb6827b2` | `GEMM A16W16 ATOMIC` | `M_LEQ_1` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=4`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=16`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_fa50b53a7156f918` | `GEMM A16W16 ATOMIC` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_73d2d9863ba7bd0f` | `GEMM A16W16 ATOMIC` | `M_LEQ_16` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_b85e060af6adb8c4` | `GEMM A16W16 ATOMIC` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_27bdeabbfe157136` | `GEMM A16W16 ATOMIC` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_0d366e7f530ed847` | `GEMM A16W16 ATOMIC` | `M_LEQ_4` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=4`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=16`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_1fcd46d90df02104` | `GEMM A16W16 ATOMIC` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8` | `NUM_KSPLIT`, `cache_modifier`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_d26c0c32f269014a` | `GEMM A16W16 ATOMIC` | `M_LEQ_64` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_02a2a16c67568b8c` | `GEMM A16W16 ATOMIC` | `M_LEQ_8` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_b18154986acf87d3` | `GEMM A16W16 ATOMIC` | `any` | `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3`, `waves_per_eu=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `num_warps` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_6799a96dc1f19297` | `GEMM A16W16 gated` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_2304513d90ca89b0` | `GEMM A16W16 gated` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=8`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8241af392456da53` | `GEMM A16W16 gated` | `M_LEQ_2048` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_2fa1103ea4803a73` | `GEMM A16W16 gated` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a8bd629e52db1d73` | `GEMM A16W16 gated` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=2`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_34136ecbf144f9d9` | `GEMM A16W16 gated` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ece6aad4296100b9` | `GEMM A16W16 gated` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5e8838d9072d69f2` | `GEMM A16W16 gated` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=1`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_826b2cf674faaf4d` | `GEMM A16W16 gated` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=16`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9f86ee273c938082` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_5392684c57c2a71e` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_warps=2`, `waves_per_eu=4` | `NUM_KSPLIT`, `num_stages` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_e6d7c87de97c9096` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_33903460bdb7b29d` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_N`, `num_stages`, `num_warps`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_3659c9feaf03fae2` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | `GROUP_SIZE_M`, `cache_modifier` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_9698853d75892b65` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_N=64`, `matrix_instr_nonkdim=16`, `num_stages=3`, `waves_per_eu=1` | `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_fb75b9b6a2c9247d` | `GEMM A16W8_BLOCKSCALE` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_warps=2` | `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_4f2ea50f4e043d6c` | `GEMM A16W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `GROUP_SIZE_M`, `waves_per_eu` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_10824ac91ff89684` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_1d5e2d3bcc103c4f` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_d6db1faa1e24009c` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_c9882c8ccd7044d6` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `waves_per_eu=4` | `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_warps` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_464fce588c9e37ff` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8` | `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_61630e3de0c67328` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_84cece005985b9a6` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_cfec29501fb32c9c` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | `GROUP_SIZE_M` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_7f056f8d0efa8736` | `GEMM A16WFP4` | `M_LEQ_128` | `BLOCK_SIZE_N=128`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_57da2461b45a899b` | `GEMM A16WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_warps=4`, `waves_per_eu=2` | `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_20a0655c34d03c81` | `GEMM A16WFP4` | `M_LEQ_256` | `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_e1e5a9e6225dfb77` | `GEMM A16WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_warps=8`, `waves_per_eu=2` | `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_fbc741c4aad2ffc7` | `GEMM A16WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_6e65e8a166cd7d99` | `GEMM A16WFP4` | `M_LEQ_8` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_78dae4a4360b9563` | `GEMM A16WFP4` | `any` | `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_84c456a06bf7c756` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_128` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_eb3756eb470f0d73` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_16` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_051da91220159535` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_256` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5f58acb0b353e63c` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d6358554b873f2db` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_016d6f49837cf2e0` | `GEMM A16WFP4_PRESHUFFLED` | `M_LEQ_8` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=14`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9e7123d5f9e3f642` | `GEMM A16WFP4_PRESHUFFLED` | `any` | `BLOCK_SIZE_M=32`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `waves_per_eu=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_stages`, `num_warps` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_bfa02a7bfcc36471` | `GEMM A8W8` | `M_LEQ_1024` | `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_fa26e803c3619f91` | `GEMM A8W8` | `M_LEQ_128` | `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_97894ae631f67383` | `GEMM A8W8` | `M_LEQ_16` | `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `num_stages`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_e20824a4fa635628` | `GEMM A8W8` | `M_LEQ_2048` | `BLOCK_SIZE_N=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_d6d48aa61489bbbc` | `GEMM A8W8` | `M_LEQ_256` | `BLOCK_SIZE_N=64`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_57f169c7c8408d06` | `GEMM A8W8` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `cache_modifier`, `num_warps`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_beb9414a4d4edf58` | `GEMM A8W8` | `M_LEQ_4096` | `BLOCK_SIZE_K=128`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_b98a54f70ff68435` | `GEMM A8W8` | `M_LEQ_512` | `BLOCK_SIZE_N=64`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_b96f93f7eb025894` | `GEMM A8W8` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_5166e7f8cd2c14f4` | `GEMM A8W8` | `M_LEQ_8` | `BLOCK_SIZE_M=8`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_b7d40567cfa969ea` | `GEMM A8W8` | `any` | `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_stages`, `num_warps`, `waves_per_eu` | 4 selected shape config(s) in 4 JSON file(s) |
| `cfg_c94a55b233cad7ea` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_128` | `BLOCK_SIZE_K=128` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_ab4442fdd08d0ab3` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_6e04f3c060095c74` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_6511b85f28cc24a3` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_a80837d278172560` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_f3c8a2c89a3d1d60` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_d20ceac354b22aa4` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_cc3e0595aa865a63` | `GEMM A8W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `NUM_KSPLIT=1` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_deae58a34adb6258` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_9e274bd9b6c46eaf` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_N=16`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_3883cf64563aaa1b` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_2601a8a0628a409e` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_f93524c0330d99bb` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_855cb543440bd0a8` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_bf91dceaa41422ca` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 13 selected shape config(s) in 13 JSON file(s) |
| `cfg_8b23ae88a4a42049` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_stages`, `num_warps`, `waves_per_eu` | 14 selected shape config(s) in 14 JSON file(s) |
| `cfg_5585a593f8f102fe` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_128` | `cache_modifier=.cg`, `num_stages=3`, `num_warps=8` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `matrix_instr_nonkdim`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_17116d8b599bdad3` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_16` | `BLOCK_SIZE_M=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `waves_per_eu=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `num_warps` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_8771dd977b4889c6` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_256` | `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=32`, `num_stages=3`, `waves_per_eu=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_6b5fd44ad6ff823a` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_41820d4eec8d94a7` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=32`, `num_stages=3`, `num_warps=8` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_edf454706042f196` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_64` | `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `num_stages=3` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `matrix_instr_nonkdim`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_878d12505ef6a0b6` | `GEMM A8W8_PER_TOKEN_SCALE` | `M_LEQ_8` | `BLOCK_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_a689264922846e99` | `GEMM A8W8_PER_TOKEN_SCALE` | `any` | `NUM_KSPLIT=1` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 5 selected shape config(s) in 5 JSON file(s) |
| `cfg_96c6632485058e08` | `GEMM A8WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_42dcaab6d72effb2` | `GEMM A8WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a9c50f69675eb53f` | `GEMM A8WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=32`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ae81633725ac9ec9` | `GEMM A8WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f5900be3ed9c8516` | `GEMM A8WFP4` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_4db5e820c792ac7b` | `GEMM A8WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_1431b91346cbfdf9` | `GEMM A8WFP4` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_1851dab3fadb83ca` | `GEMM A8WFP4` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=16`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9d65f58c6f42e5bd` | `GEMM AFP4WFP4` | `M_LEQ_128` | `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_c22c7e6c6301717c` | `GEMM AFP4WFP4` | `M_LEQ_16` | `cache_modifier=.cg`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `num_warps`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_952efaab27dc2ced` | `GEMM AFP4WFP4` | `M_LEQ_2048` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_1b7fc4352a1aab65` | `GEMM AFP4WFP4` | `M_LEQ_256` | none shared by all configs | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_de9875158e9a48da` | `GEMM AFP4WFP4` | `M_LEQ_32` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_609d7834c43d62cd` | `GEMM AFP4WFP4` | `M_LEQ_4096` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d1d0a8dc138e0571` | `GEMM AFP4WFP4` | `M_LEQ_512` | `NUM_KSPLIT=1` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 18 selected shape config(s) in 18 JSON file(s) |
| `cfg_4c70fe56ffe6eaae` | `GEMM AFP4WFP4` | `M_LEQ_64` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_2362072de525ac82` | `GEMM AFP4WFP4` | `M_LEQ_8` | `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `waves_per_eu` | 25 selected shape config(s) in 25 JSON file(s) |
| `cfg_19c96d4f76752170` | `GEMM AFP4WFP4` | `any` | `NUM_KSPLIT=1` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 36 selected shape config(s) in 36 JSON file(s) |
| `cfg_f565e339602e6388` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_128` | none shared by all configs | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_32927746518772c2` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_256` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_60d953ac8627aebc` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_31` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_afa0e179ea27f2cb` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_32` | none shared by all configs | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_bbb13db433d0907a` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_512` | `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `num_stages`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_db9580db9d05c211` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_64` | none shared by all configs | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `matrix_instr_nonkdim`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_98e147da67fe19a2` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_8` | `matrix_instr_nonkdim=16` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_0f3bcc30969e9c45` | `GEMM AFP4WFP4_PRESHUFFLED` | `any` | `NUM_KSPLIT=1`, `num_stages=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `matrix_instr_nonkdim`, `num_warps`, `waves_per_eu` | 26 selected shape config(s) in 26 JSON file(s) |
| `cfg_7d6ab7c4c6aa8141` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | `NUM_KSPLIT` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_ddeca1e7c6324cb3` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | `NUM_KSPLIT` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_1bf618ef86d9fa83` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | `NUM_KSPLIT` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_ab4c50827ba292aa` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | `NUM_KSPLIT` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_30048a6f6b6fcf15` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_8` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=4`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | `NUM_KSPLIT` | 2 selected shape config(s) in 2 JSON file(s) |
| `cfg_378921749d27c94e` | `GEMM_PREQUANT AFP4WFP4` | `any` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | `NUM_KSPLIT`, `cache_modifier` | 2 selected shape config(s) in 2 JSON file(s) |

### `gfx942`

| decision ref | variant | M bucket | seed candidate | vary next | shipped support |
|---|---|---|---|---|---|
| `cfg_acd94d7e5a3e6b46` | `BATCHED_GEMM A16W16` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a8810d77e6fc8a68` | `BATCHED_GEMM A16W16` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_88838bf9ccdd67a2` | `BATCHED_GEMM A8W8` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5abe56239c8999b3` | `BATCHED_GEMM A8W8` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_7d811f098a5c7452` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_128` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_f03673de111b7c41` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_16` | `BLOCK_SIZE_M=16`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_8bc920e2d4097f5c` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_256` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_f3ab3dc0a42b1c5c` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_32` | `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_85d7cb720c1184e5` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_64` | `BLOCK_SIZE_M=32`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_warps=8` | `BLOCK_SIZE_N`, `num_stages`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_4bc50666b6e30c49` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `any` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16` | `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_a6553dc34c44e75d` | `FF A16W16 fused` | `M_LEQ_4` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=4`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_7ca3e74736e43722` | `FF A16W16 fused` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_73add493d896bfe1` | `FF A16W16 fused` | `M_LEQ_8` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=8`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e8a618536485341e` | `FF A16W16 fused` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e5d5b4c45523c9d9` | `FUSED GEMM A8W8_BLOCKSCALE A16W16` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c1f192871dca081b` | `GEMM A16W16` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_56040e8f9a8da05a` | `GEMM A16W16` | `M_LEQ_2048` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8326771c1eea75d9` | `GEMM A16W16` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_3478d40859900e02` | `GEMM A16W16` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_87219bb7e8370747` | `GEMM A16W16` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_fc28e4c29e869403` | `GEMM A16W16` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_01063c787c4b12ef` | `GEMM A16W16 ATOMIC` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_4ae551d9c194096f` | `GEMM A16W16 gated` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6f71200f98a37278` | `GEMM A16W16 gated` | `M_LEQ_2048` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9c88edfbbc242a78` | `GEMM A16W16 gated` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a7881c3f24826cb9` | `GEMM A16W16 gated` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=4`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f41edce947725f4e` | `GEMM A16W16 gated` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5e89def6b74b3ce0` | `GEMM A16W16 gated` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_26dc9deada5b95c6` | `GEMM A16W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_0139286957c7247b` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_2af8525659b81f6f` | `GEMM A8W8` | `M_LEQ_16` | `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_d1534465a7c2af96` | `GEMM A8W8` | `M_LEQ_64` | `matrix_instr_nonkdim=16`, `num_warps=4` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_stages`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_5787eebc48a66d7b` | `GEMM A8W8` | `any` | `cache_modifier=None`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_K`, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `num_warps`, `waves_per_eu` | 7 selected shape config(s) in 7 JSON file(s) |
| `cfg_157ffb19719c4b40` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_1` | `BLOCK_SIZE_K=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `num_warps`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_da4a49e152c50056` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_1024` | `BLOCK_SIZE_K=128`, `NUM_KSPLIT=1`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `num_warps`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_21e1a39bc7839a06` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `SPLITK_BLOCK_SIZE`, `cache_modifier`, `num_warps`, `waves_per_eu` | 8 selected shape config(s) in 8 JSON file(s) |
| `cfg_c23460f537b313ac` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_16` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `SPLITK_BLOCK_SIZE`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_1311810498769365` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_2048` | `BLOCK_SIZE_K=128`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `num_warps`, `waves_per_eu` | 6 selected shape config(s) in 6 JSON file(s) |
| `cfg_2e474d5a3267b122` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `kpack=2`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `SPLITK_BLOCK_SIZE`, `cache_modifier`, `num_stages`, `num_warps`, `waves_per_eu` | 8 selected shape config(s) in 8 JSON file(s) |
| `cfg_89a57b3ca5ee84c7` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_32` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `SPLITK_BLOCK_SIZE`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_d6c5b6bf7798a092` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_4` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `GROUP_SIZE_M=1`, `SPLITK_BLOCK_SIZE=512`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_658519ce57ddcf17` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_512` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `SPLITK_BLOCK_SIZE=1792`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 7 selected shape config(s) in 7 JSON file(s) |
| `cfg_cb120b4dcb86c8e7` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_64` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `GROUP_K=128`, `GROUP_N=128`, `cache_modifier=.cg`, `kpack=2`, `matrix_instr_nonkdim=16` | `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `SPLITK_BLOCK_SIZE`, `num_stages`, `num_warps`, `waves_per_eu` | 3 selected shape config(s) in 3 JSON file(s) |
| `cfg_d2642e1347a53ea1` | `GEMM A8W8_BLOCKSCALE` | `M_LEQ_8` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `SPLITK_BLOCK_SIZE=512`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `NUM_KSPLIT`, `cache_modifier`, `num_warps`, `waves_per_eu` | 9 selected shape config(s) in 9 JSON file(s) |
| `cfg_3eff28954f18b5fe` | `GEMM A8W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `GROUP_K=128`, `GROUP_N=128`, `NUM_KSPLIT=1`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2` | `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `GROUP_SIZE_M`, `cache_modifier`, `num_warps`, `waves_per_eu` | 10 selected shape config(s) in 10 JSON file(s) |
| `cfg_133b4f1fc4c41fa7` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ccd5e6f3adbfd648` | `GEMM A8W8_PER_TOKEN_SCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=2`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |

### `gfx1250`

| decision ref | variant | M bucket | seed candidate | vary next | shipped support |
|---|---|---|---|---|---|
| `cfg_386a49c8c892741c` | `BATCHED_GEMM A16W16` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b7ac0cfc5a126f01` | `BATCHED_GEMM A16W16` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_99b2a3d1c7e2c1a4` | `BATCHED_GEMM A8W8` | `M_GEQ_4096` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ae29d28a3ab9d9c8` | `BATCHED_GEMM A8W8` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6ec76b0ddc2237bb` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_128` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_853cb299168693db` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_16` | `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f52b6d2da0a23c44` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_256` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_43a39e20dd092899` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_32` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6a8e0d961351db46` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `M_LEQ_64` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8cd8b4e53e58fd04` | `BATCHED_GEMM A8W8 A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT` | `any` | `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_bb61d9cb5346223d` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_bacec2b0512b58a6` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=6` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5d2c98a673be40b0` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c8f0070dacae04a3` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e1b3b98fd95e2e91` | `BATCHED_GEMM AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e11165443e0440e0` | `BATCHED_GEMM AFP4WFP4` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=64`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c4f14fc34a887040` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_36eab773af82560f` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=6` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8a71de42cc0f3ab8` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_53f29d325ebcd889` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_77cc26807ff708c3` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_bca26da5f369990d` | `BATCHED_GEMM_PREQUANT AFP4WFP4` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=64`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_63961a5591bbc2ba` | `FF A16W16 fused` | `M_LEQ_4` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_24d9e8e7b095210d` | `FF A16W16 fused` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b64ef690a60248c3` | `FF A16W16 fused` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e34675a4e3782dbf` | `FF A16W16 fused` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_62cf630899d917b0` | `FUSED GEMM A8W8_BLOCKSCALE A16W16` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_81c75bce0eecc1c4` | `FUSED GEMM AFP4WFP4 A16W16` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9ffb3d7d0d0aa2ba` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_85391f4473e37a1a` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=16`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a98e77b59dcba1e4` | `FUSED GEMM AFP4WFP4_PRESHUFFLED A16W16` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5e0a9bd4d6dc72de` | `GEMM A16W16` | `M_LEQ_128` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_617feefabcbd607b` | `GEMM A16W16` | `M_LEQ_2048` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_6cfe90bbba903acf` | `GEMM A16W16` | `M_LEQ_256` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_036f72a5a7636519` | `GEMM A16W16` | `M_LEQ_512` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a5cc7d52db6f7603` | `GEMM A16W16` | `M_LEQ_64` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=6`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=4`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_2539ae636c9593f1` | `GEMM A16W16` | `any` | `BLOCK_SIZE_K=32`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=6`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=3`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_3aaedfe64964694b` | `GEMM A16W16 ATOMIC` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=32`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_54655a2470c042c1` | `GEMM A16W16 gated` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_bae27d0120648a3f` | `GEMM A16W16 gated` | `M_LEQ_2048` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_05d4ba49db87d3e6` | `GEMM A16W16 gated` | `M_LEQ_256` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a7e43724f9b855df` | `GEMM A16W16 gated` | `M_LEQ_512` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d4a5537c87466f3c` | `GEMM A16W16 gated` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_664ed39950b91d49` | `GEMM A16W16 gated` | `any` | `BLOCK_SIZE_K=64`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=16`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_afdb35d0ed517ff8` | `GEMM A16W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_4c00505baec4319c` | `GEMM A16W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=8`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_608dd72f9a981bfa` | `GEMM A16WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_e617eaaadcafeb40` | `GEMM A16WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_4ae8fc4c16fdf65b` | `GEMM A16WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_43716e9ab90df951` | `GEMM A16WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_eec399b9420e2417` | `GEMM A16WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c6c9ff3aa0bad012` | `GEMM A16WFP4` | `M_LEQ_8` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_b92dbf6075e0af63` | `GEMM A16WFP4` | `any` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_920af13bcc3b237f` | `GEMM A16WFP4_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d957c5be832f332b` | `GEMM A8W8` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_86933d1160956ed4` | `GEMM A8W8_BLOCKSCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.ca`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c96f3303869edd5a` | `GEMM A8W8_BLOCKSCALE_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_db863cb6589568e6` | `GEMM A8W8_PER_TOKEN_SCALE` | `any` | `BLOCK_SIZE_K=128`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d9a43a8c60e0a561` | `GEMM A8WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ae084844b611c1bc` | `GEMM A8WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=6` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_77fa97602bf7f604` | `GEMM A8WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_10c67bfc82830619` | `GEMM A8WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d8fa9714e0e6f662` | `GEMM A8WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `kpack=1`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=4` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5a73c72173bab183` | `GEMM A8WFP4` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=32`, `NUM_KSPLIT=1`, `cache_modifier=None`, `kpack=1`, `matrix_instr_nonkdim=32`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f807667140ab2ce2` | `GEMM AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=32`, `num_stages=3`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_87f367e46fd471b7` | `GEMM AFP4WFP4` | `M_LEQ_16` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=16`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_dc5e90588c940352` | `GEMM AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=128`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=32`, `num_stages=3`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_880dfe88b35e2782` | `GEMM AFP4WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=4`, `waves_per_eu=3` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_a5b9232500052e29` | `GEMM AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=64`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=32`, `num_stages=3`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_374b95ab2779bd57` | `GEMM AFP4WFP4` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=256`, `BLOCK_SIZE_N=256`, `GROUP_SIZE_M=2`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=32`, `num_stages=2`, `num_warps=8`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_fa89c9f2bcc73c86` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_128` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_7bfa418f36295d9b` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_256` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_9381d9d7ba07bac8` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_31` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8d489ef81303a783` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_32` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f9f935a26ca193fc` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_64` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_ced20d13c9524705` | `GEMM AFP4WFP4_PRESHUFFLED` | `M_LEQ_8` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_5d31249a8dbf2bd0` | `GEMM AFP4WFP4_PRESHUFFLED` | `any` | `BLOCK_SIZE_K=256`, `BLOCK_SIZE_M=32`, `BLOCK_SIZE_N=64`, `GROUP_SIZE_M=4`, `NUM_KSPLIT=1`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=2`, `num_warps=2`, `waves_per_eu=1` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_c9ddab412758d677` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_128` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_8b99925f83ae324a` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_256` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_f51f0ca714633a66` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_32` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_d5573e6cf86e1a86` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_64` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_7476f06bfb75b41c` | `GEMM_PREQUANT AFP4WFP4` | `M_LEQ_8` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=.cg`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=4`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |
| `cfg_aa6fb77979236a11` | `GEMM_PREQUANT AFP4WFP4` | `any` | `BLOCK_SIZE_K=512`, `BLOCK_SIZE_M=16`, `BLOCK_SIZE_N=128`, `GROUP_SIZE_M=1`, `NUM_KSPLIT=4`, `cache_modifier=None`, `matrix_instr_nonkdim=16`, `num_stages=1`, `num_warps=8`, `waves_per_eu=2` | all exposed knobs; one config cannot establish agreement | 1 selected shape config(s) in 1 JSON file(s) |

## Sources

- Curated cards: [`decisions/gemm.yaml`](decisions/gemm.yaml).
- Source evidence: [`evidence/gemm_source.yaml`](evidence/gemm_source.yaml).
- Shipped tuning evidence: [`evidence/gemm_tuned_configs.yaml`](evidence/gemm_tuned_configs.yaml).
- Generated by [`_render_decisions.py`](_render_decisions.py); edit the cards or evidence, never this file.
