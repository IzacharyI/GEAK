export const meta = {
  name: 'kernel-workflow',
  description: 'Multi-agent GPU kernel / e2e-model optimization (Director/TechLead/specialist Engineers) with budget-controlled rounds, independent verification, and integration. Optimizes a kernel or vLLM/SGLang model for inference speed on AMD Instinct MI-series GPUs (MI300X/300A/308X/325X on CDNA3 gfx942, MI350X/355X on CDNA4 gfx950 — the target card is auto-detected on-box).',
  whenToUse: 'Optimize the inference speed of a kernel directory (single kernel, fused kernels) or an end-to-end vLLM/SGLang model. Pass args.kernel_path (required), args.budget, args.gpu_ids, args.task.',
  phases: [
    { title: 'Setup', detail: 'director builds the isolated eval dir + canonical workspace' },
    { title: 'Author', detail: 'author_engineer writes a fresh optimize-loop seed (only when mode=author); speedup denominator stays the frozen online kernel' },
    { title: 'Analyze', detail: 'tech_lead analyzes kernel + writes roadmap' },
    { title: 'Benchmark', detail: 'benchmark_engineer builds the COMMANDMENT + baseline' },
    { title: 'Profile', detail: 'profile_engineer classifies; optional analysis_engineer validates Step-2 evidence' },
    { title: 'Optimize', detail: 'budget loop: tech_lead plans, specialist OR deep_explore engineers optimize, reprofile' },
    { title: 'Verify', detail: 'each candidate patch independently re-benchmarked' },
    { title: 'Merge', detail: 'integrator combines the round winners' },
    { title: 'Report', detail: 'tech_lead writes the final report + patch' },
    { title: 'Validate', detail: 'director independently validates vs the true baseline' },
  ],
};

// ---------------------------------------------------------------------------
// Args + defaults. (The script cannot touch the filesystem or read its own
// path; agents do all FS work, and every path is supplied/derived from args —
// nothing about the install location is hard-coded.)
// ---------------------------------------------------------------------------
const A = args || {};
if (!A.kernel_path) throw new Error('args.kernel_path is required (absolute path to the kernel/model directory)');

// WORKFLOW_DIR = the directory that holds this script + roles/ + knowledge/ + scripts/.
// A JS workflow script can't read its own path, so the caller passes it (it is just the
// dirname of the scriptPath used to launch the workflow).
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) {
  throw new Error('args.workflow_dir is required: absolute path to the directory containing ' +
    'kernel_workflow.js, roles/, knowledge/, scripts/ (i.e. the dirname of this script).');
}
// EXP_ROOT = where timestamped run dirs are written. Default: a sibling "exp/" next to the
// kernel_workflow dir (…/<parent>/kernel_workflow -> …/<parent>/exp). Override with args.exp_root.
const EXP_ROOT = String(A.exp_root || (WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/exp')).replace(/\/+$/, '');

const KERNEL_PATH_ORIG = A.kernel_path;
const BUDGET = parseInt(A.budget != null ? A.budget : 6, 10);
// POSITIVE CONTROL — a known-good change with an ALREADY-MEASURED effect, run once during Benchmark
// before any direction budget is spent. The null arm calibrates the NOISE floor; this calibrates the
// DETECTION floor, and nothing else in the workflow does. Without it a 1.000x report is
// unfalsifiable: "we found nothing" and "we cannot see anything" produce byte-identical output.
// This is not hypothetical — a run reported 1.000x on a tree where a +4.71% win was on the table.
//   {name, how, expected_pct_lo, expected_pct_hi, magnitude?, guard?, abort_on_fail?}
// `how` is prose the benchmark engineer executes (usually "flip env X, everything else identical").
// expected_pct_* bound the ALREADY-KNOWN delta; measuring outside that band means the harness is
// lying, so by default the run ABORTS rather than spending directions on an instrument that cannot
// read them. Set abort_on_fail:false to downgrade to a warning (use only when deliberately
// re-calibrating the expected band itself).
//
// `magnitude` says what the band IS: 'recorded' (default) means someone has measured this effect and
// the numbers are a fact, so reading under `lo` is the instrument's fault and aborts. 'constructed'
// means the control is a synthetic injection (benchmark_engineer.md 5b) and the band is a TARGET a
// knob was aimed at, never itself measured — there an under-read within a factor of two is a sizing
// miss, tolerated as PASS (UNDERSHOOT) if the effect still clears the null spread and every pair
// agrees in sign. Declare it truthfully: relabelling a recorded effect to clear the gate destroys the
// only evidence the run has that its own numbers mean anything.
const POSITIVE_CONTROL = (A.positive_control && typeof A.positive_control === 'object')
  ? A.positive_control : null;
const PC_ABORT = POSITIVE_CONTROL ? (A.positive_control.abort_on_fail !== false) : false;
// Per-card free-VRAM floor, in GiB, below which the pool counts as OCCUPIED and the round is planned
// GPU-less. 0 (the default) disables the sample entirely, so nothing changes for runs that do not set
// it. This is a FLOOR and not a window: on a shared box the correct threshold is well above what one
// launch needs at t=0, because an arm that starts with just enough dies when the tenant regrows.
const GPU_MIN_FREE_GIB = Number(A.gpu_min_free_gib || 0);
// Minimum verified geomean improvement over the cumulative best for a round winner to be COMMITTED
// into the canonical workspace (default 2%). Kept as a knob rather than a hard-coded constant so the
// gate is tunable per run (e.g. raise it on a noisy box, lower it to bank small compounding wins).
const MIN_IMPROVE = (() => {
  const v = parseFloat(A.min_improve != null ? A.min_improve : 0.02);
  return Number.isFinite(v) && v >= 0 ? v : 0.02;
})();
// Budget cost of ONE `deep_explore` direction. The deep-explore engineer does far more than a single
// specialist — broad rewrite authority, its own multi-iteration measure→profile→rewrite loop — so it
// is charged more than 1 against the direction budget (default 2). It also always runs in a DEDICATED
// round (no other directions that round), enforced below.
const DEEP_COST = (() => {
  const v = parseInt(A.deep_cost != null ? A.deep_cost : 2, 10);
  return Number.isFinite(v) && v >= 1 ? v : 2;
})();
// WHAT THIS WAVE IS FOR. Two values; the default is the historical behaviour.
//
//   'speedup'        (default) — the loop hunts a verified geomean win. Every rule below is unchanged.
//   'working_kernel'           — the loop hunts ONE artifact that RUNS. Speed is not scored.
//
// The second mode exists because the first one cannot deliver a fused megakernel, and the reason is
// arithmetic rather than judgement. Measured on wave 14: setup spends 57 GPU runs on the baseline and
// the positive control before a single direction is dispatched (~35 s/run, 33 min, 2.2 leases at the
// 900 s cap in scripts/gpu_group_lock.sh). A round then gets ONE lease — ~25 runs — split across the
// round's directions. Four directions is therefore ~6 hardware iterations per direction per round,
// and three rounds is ~18 for the whole wave. A cross-rank persistent kernel is not debuggable in 18
// iterations; waves 3 and 4 got one onto hardware, spent their rounds bisecting an illegal access,
// and the artifact was never picked up again.
//
// Three things have to change together, or changing any one of them does nothing:
//
//   1. ONE direction per round. Breadth buys nothing here: the directions contend for the same single
//      8-GPU lease, so N directions is N-way queueing, not N-way throughput. Enforced below, not
//      requested of the planner — the planner has produced 4-to-8-direction rounds under prose that
//      already said leases are scarce.
//   2. MAX_NO_IMPROVE cannot apply. A crash-debug round is non-improving BY CONSTRUCTION, so the
//      default stop-after-2 kills this mode on round 3 no matter how large the budget is. This is the
//      one that makes the other two pointless if it is missed.
//   3. The commit gate cannot be MIN_IMPROVE. Nothing clears "2% faster than cumulative" while it
//      still crashes, so nothing is committed, so round N+1 starts from the unfused tree and the
//      carry-over failure that produced waves 3->6 is reproduced exactly. In this mode the round's
//      candidate is committed on RUNNING — correctness pass, liveness not failing, activation
//      confirmed — which is the objective itself.
//
// What does NOT change: the five acceptance conditions, the leak scan, rank-max, paired reps. This
// mode sets the objective of one wave; it does not move the bar the wave is eventually judged
// against. And it deliberately makes the wave's timing numbers WORSE than useless rather than
// cheaper — see objectiveVerdict.
const OBJECTIVE = (() => {
  const v = String(A.objective || 'speedup').trim();
  if (v !== 'speedup' && v !== 'working_kernel') {
    throw new Error(`args.objective must be 'speedup' or 'working_kernel', got '${v}'`);
  }
  return v;
})();
const WORKING_KERNEL = OBJECTIVE === 'working_kernel';
function resolveGpuRequest(A) {
  const taskResource = (A.op_spec && A.op_spec.resource) || {};
  const parseIds = (name, value) => {
    const raw = String(value);
    const parts = raw.split(',').map(s => s.trim()).filter(Boolean);
    if (!parts.length) throw new Error(`${name} must contain at least one GPU id`);
    if (parts.some(id => !/^\d+$/.test(id))) throw new Error(`${name} contains a non-numeric GPU id`);
    const ids = parts.map(id => String(Number(id)));
    if (new Set(ids).size !== ids.length) throw new Error(`${name} contains duplicate GPU ids`);
    return ids;
  };
  const gpuList = parseIds('gpu_ids', A.gpu_ids != null ? A.gpu_ids : '0');
  const directFixed = String(A.job_gpu_ids || '').trim();
  const fixedValue = directFixed || taskResource.job_gpu_ids;
  const fixedIds = String(fixedValue || '').trim()
    ? parseIds('job_gpu_ids', fixedValue)
    : [];
  const inferredCount = fixedIds.length || 1;
  const countValue = A.gpus_per_job != null
    ? A.gpus_per_job
    : (taskResource.gpus_per_job != null ? taskResource.gpus_per_job : inferredCount);
  const countText = String(countValue).trim();
  if (!/^[1-9]\d*$/.test(countText)) {
    throw new Error('gpus_per_job must be a positive integer');
  }
  const gpusPerJob = Number(countText);
  if (!Number.isSafeInteger(gpusPerJob)) throw new Error('gpus_per_job is too large');
  if (fixedIds.length && fixedIds.length !== gpusPerJob) {
    throw new Error(`gpus_per_job=${gpusPerJob} does not match job_gpu_ids size ${fixedIds.length}`);
  }
  if (!fixedIds.length && gpusPerJob > gpuList.length) {
    throw new Error(`gpus_per_job=${gpusPerJob} exceeds gpu_ids pool size ${gpuList.length}`);
  }
  const pool = new Set(gpuList);
  if (fixedIds.some(id => !pool.has(id))) {
    throw new Error('job_gpu_ids must be a subset of gpu_ids');
  }
  const sharedSpec = fixedIds.length
    ? `group:${fixedIds.join(',')}`
    : (gpusPerJob > 1 ? `pool:${gpusPerJob}:${gpuList.join(',')}` : '');
  return {
    gpuList,
    gpusPerJob,
    fixedIds,
    sharedSpec,
    specForIndex: (index) => sharedSpec || gpuList[index % gpuList.length],
  };
}
const GPU_RESOURCE = resolveGpuRequest(A);
const GPU_LIST = GPU_RESOURCE.gpuList;
const GPU_IDS = GPU_LIST.join(',');
const TASK = A.task || '';
const EVAL_DIR_OVERRIDE = A.eval_dir || '';
const APPLY_TO_ORIGINAL = String(A.apply_to_original != null ? A.apply_to_original : 'false');
const KERNEL_NAME_HINT = KERNEL_PATH_ORIG.replace(/\/+$/, '').split('/').pop();

// --- author mode: when there is NO existing source, write a fresh from-scratch SEED first, then optimize it.
// mode=optimize (default) keeps the exact original behavior (backward compatible). mode=author seeds
// the workspace from an op task dir (immutable oracle + frozen online kernel in baseline_src/), the
// author_engineer writes a passing seed, then the SAME optimize loop runs — always timing against the
// frozen online kernel, never against the seed's own language. KERNEL_KNOWLEDGE_DIR is the AMD authoring
// knowledge base — REFERENCE ONLY (facts/how-to, never decisions; the author always measures regardless). Default:
// sibling perf_knowledge/ so standalone runs use it too; empty if WORKFLOW_DIR is unset (no behavior change).
const MODE = String(A.mode != null ? A.mode : 'optimize').trim() || 'optimize';
const TARGET_LANGUAGE = String(A.target_language != null ? A.target_language : 'triton').trim() || 'triton';
const OP_SPEC = A.op_spec || {};
// When the op will run on the CUDA/HIP-graph-captured decode path (e2e sets op_spec.cuda_graph_safe=true),
// the isolated oracle alone CANNOT catch a kernel that passes iso but host-syncs or lazily-compiles under
// graph capture — the "wins isolated, crashes serving" class (cuda_graph_capture_unsafe / NO_BINARY_FOR_GPU).
// This turns on an OPTIONAL capture+replay smoke in the verify step so that failure is caught at the cheap
// isolated stage. Unset (standalone single-kernel runs / non-graph ops) => byte-identical to before.
const REQUIRE_GRAPH_CAPTURE = !!(OP_SPEC && OP_SPEC.cuda_graph_safe === true);
// WORKLOAD ALIGNMENT (optional). When the caller supplies the real-workload shape/dtype case
// distribution, the benchmark harness benchmarks EXACTLY those (shape, dtype) cases, weights each
// by its total time contribution in the workload (weight = count * baseline_latency), and the
// optimization target becomes the time-weighted ratio-of-sums instead of an unweighted geomean.
//   workload_spec_path : path to a workload-v1 json (produced by parse_profile.py --workload-out,
//                        or hand-written). The benchmark_engineer reads it (JS can't touch FS).
//   op_spec.workload   : inline cases, same shape as a workload-v1 "kernels[].cases" list (or the
//                        full object). Takes precedence; weight_source becomes "caller".
// Both unset => unweighted behavior, byte-identical to before. Correctness ALWAYS stays on the
// frozen reference_io.pt oracle; this only shapes the PERFORMANCE measurement.
const WORKLOAD_SPEC_PATH = String(A.workload_spec_path || (OP_SPEC && OP_SPEC.workload_path) || '').trim();
const WORKLOAD_SPEC = (OP_SPEC && OP_SPEC.workload) || A.workload || null;
const HAS_WORKLOAD = !!(WORKLOAD_SPEC_PATH ||
  (Array.isArray(WORKLOAD_SPEC) && WORKLOAD_SPEC.length) ||
  (WORKLOAD_SPEC && Array.isArray(WORKLOAD_SPEC.kernels) && WORKLOAD_SPEC.kernels.length));
const argList = (value) => (Array.isArray(value) ? value : String(value || '').split(','))
  .map((s) => String(s).trim()).filter(Boolean);
// A scoped campaign must name the guards that can CREATE credit separately from guards that can
// only VETO a candidate. Without this split a skew-only win was folded into the global geomean and
// promoted as the current best while the requested uniform guard executed the baseline path.
const TARGET_GUARDS = argList(A.target_guards);
const REGRESSION_GUARDS = argList(A.regression_guards);
// `operator_e2e` is the only generally comparable metric across a launch-structure change. Per-stage
// timers collected in separate graphs/rank reductions are diagnostics; summing them is not a fused
// kernel time. `changed_kernel` remains available for tasks that provide a genuinely comparable
// same-timeline changed_us/replaced_sum_us measurement. `legacy` preserves the pre-feature global
// score and its attribution behavior for callers that did not opt into a scoped contract.
const PROMOTION_METRIC = String(A.promotion_metric || 'legacy').trim();
if (!['legacy', 'operator_e2e', 'changed_kernel'].includes(PROMOTION_METRIC)) {
  throw new Error(`args.promotion_metric must be 'legacy', 'operator_e2e' or 'changed_kernel', got ${PROMOTION_METRIC}`);
}
// Strict proof runs must state the metric; inheriting historical mixed behavior is not auditable.
if (String(A.strict_autonomy || 'false') === 'true' && PROMOTION_METRIC === 'legacy') {
  throw new Error('args.strict_autonomy requires an explicit promotion_metric');
}
// PRIMARY-metric selector: prefer the time-weighted number when a workload spec is in play and the
// agent reported one; otherwise fall back to the geomean (unweighted runs => unchanged behavior).
const primSpeedup = (o) => {
  if (!o) return 0;
  const w = o.verified_weighted != null ? o.verified_weighted
          : (o.speedup_weighted != null ? o.speedup_weighted : null);
  if (HAS_WORKLOAD && Number.isFinite(w)) return w;
  const g = o.verified_geomean != null ? o.verified_geomean : o.speedup_geomean;
  return Number.isFinite(g) ? g : 0;
};
const KERNEL_KNOWLEDGE_DIR = String(A.perf_knowledge_dir ||
  (WORKFLOW_DIR ? WORKFLOW_DIR.replace(/\/[^/]*$/, '') + '/perf_knowledge' : '')).replace(/\/+$/, '');
// Expert skills = human-authored, validated kernel recipes (perf_knowledge/expert_skills/). ADVISORY
// priors only: a matched `validated` skill is a HIGH-PRIOR author/optimize candidate the planning/author
// roles reproduce, then gate by the isolated A/B vs the oracle — it NEVER overrides measurement. Default
// OFF (opt-in: pass use_expert_skills="true"). When OFF (the default) NOTHING is injected -> byte-identical
// to a build without this feature. When invoked by the e2e layer the flag + dir are passed down.
const USE_EXPERT_SKILLS = String(A.use_expert_skills != null ? A.use_expert_skills : 'false') === 'true';
const EXPERT_SKILLS_DIR = String(A.expert_skills_dir ||
  (KERNEL_KNOWLEDGE_DIR ? KERNEL_KNOWLEDGE_DIR + '/expert_skills' : '')).replace(/\/+$/, '');
// Only planning + authoring roles consult skills; every other role gets no injection.
const EXPERT_SKILL_ROLES = new Set(['tech_lead', 'author_engineer', 'engineer', 'deep_engineer']);

// ---- Capability-evaluation mode (OPTIONAL, default OFF -> byte-identical behaviour) --------------
// In production, "this is already implemented next door, port it" is correct and the prior-art sweep
// exists to say exactly that. When the run is instead being used to measure whether the WORKFLOW can
// derive a result, that same doctrine imports the answer and the headline number stops meaning
// anything. Observed: a tech_lead published a reference path/branch/HEAD in roadmap.md §0, and an
// engineer's patch came back containing a 511-line file BYTE-IDENTICAL to that reference. The
// measured +4.3% was real; the run had demonstrated nothing.
//
// With capability_eval="true": prior art is reduced to CONCLUSIONS for the engineers (mechanism prose,
// no paths/hashes/source — see tech_lead.md 4d), and verify gains a byte-identity provenance check
// against known_reference_paths that returns status:"plagiarized" (verify_engineer.md 5b).
const CAPABILITY_EVAL = String(A.capability_eval != null ? A.capability_eval : 'false') === 'true';
// Strict autonomy is stronger than capability_eval. Capability mode hides known implementations;
// strict mode additionally refuses inherited code/state and turns the final-shape evidence into
// load-bearing gates. It is opt-in so ordinary kernel workflows remain backward compatible.
const STRICT_AUTONOMY = String(A.strict_autonomy != null ? A.strict_autonomy : 'false') === 'true';
if (STRICT_AUTONOMY && !CAPABILITY_EVAL) {
  throw new Error('args.strict_autonomy requires capability_eval=true');
}
if (STRICT_AUTONOMY && TARGET_GUARDS.length === 0) {
  throw new Error('args.strict_autonomy requires at least one explicit target_guards entry');
}
const positiveIntArg = (name, value, fallback) => {
  const n = Number(value == null ? fallback : value);
  if (Number.isSafeInteger(n) && n > 0) return n;
  if (STRICT_AUTONOMY) throw new Error(`args.${name} must be a positive integer`);
  return fallback;
};
const REQUIRED_REPLAYS = positiveIntArg('required_replays', A.required_replays, 1000);
const REQUIRED_PAIRS = positiveIntArg('required_pairs', A.required_pairs, 5);
const REQUIRED_PAIRS_BY_GUARD = (A.required_pairs_by_guard &&
  typeof A.required_pairs_by_guard === 'object') ? A.required_pairs_by_guard : {};
if (STRICT_AUTONOMY) {
  for (const [guard, count] of Object.entries(REQUIRED_PAIRS_BY_GUARD)) {
    if (!(Number.isSafeInteger(Number(count)) && Number(count) > 0)) {
      throw new Error(`args.required_pairs_by_guard[${guard}] must be a positive integer`);
    }
  }
}
const REQUIRE_OVERLAP = String(A.require_overlap != null ? A.require_overlap : 'false') === 'true';
const REQUIRE_ATTRIBUTION = String(A.require_attribution != null ? A.require_attribution : 'false') === 'true';
const REQUIRE_ARTIFACT_DISTINCT = String(
  A.require_artifact_distinct != null ? A.require_artifact_distinct : 'false') === 'true';
// ON-HARDWARE ACTIVATION. Opt-in (default OFF, so a wave that does not ask for it is byte-identical to
// before). When set, a switched / perf-bearing candidate may be committed or counted as an enabling
// step ONLY if its patched path was EXECUTED ON THE DEVICE this round — not merely compile-screened.
// This closes the exact hole that let a fused MegaMoE arm bank for 8 rounds behind a default-OFF flag,
// green on py_compile + a static ISA-distinctness hash, whose ON path crashed at JIT-trace time the
// first time it was ever run on hardware. `artifact_distinct: 'yes'` is satisfiable from a COMPILE_ONLY
// build (see roles/gfx950_lowering.md's lease-free method); it proves two binaries differ, NOT that the
// selected one ever traced+launched. `activation_confirmed: 'yes'` is likewise satisfiable from a host
// marker + static hash. Neither exercises the switch on a card. This flag makes on-device execution a
// hard, machine-checked field rather than prose the planner is free to defer.
const REQUIRE_HW_ACTIVATION = String(
  A.require_hardware_activation != null ? A.require_hardware_activation : 'false') === 'true';
const ACCURACY_METRIC = String(A.accuracy_metric || 'relL2');
const ACCURACY_THRESHOLD = Number(A.accuracy_threshold != null ? A.accuracy_threshold : 0.10);
if (STRICT_AUTONOMY && (!(ACCURACY_THRESHOLD > 0) || !Number.isFinite(ACCURACY_THRESHOLD))) {
  throw new Error('args.accuracy_threshold must be a positive finite number');
}
const launchTargetArg = Number(A.launch_target != null ? A.launch_target : 2);
if (STRICT_AUTONOMY && !(Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0)) {
  throw new Error('args.launch_target must be a positive integer');
}
const LAUNCH_TARGET = Number.isSafeInteger(launchTargetArg) && launchTargetArg > 0
  ? launchTargetArg : 2;
const CONTAINMENT_PREFLIGHT = (A.containment_preflight &&
  typeof A.containment_preflight === 'object') ? A.containment_preflight : {};
if (STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean !== true) {
  throw new Error(
    'args.strict_autonomy requires a clean containment_preflight emitted by trusted bootstrap_task.sh');
}
// Trees whose contents would constitute an imported answer. Given to VERIFY (to compare against) and
// deliberately NOT to engineers — handing them the list would be handing them the location.
const KNOWN_REFERENCE_PATHS = (Array.isArray(A.known_reference_paths) ? A.known_reference_paths
  : String(A.known_reference_paths || '').split(',')).map(s => String(s).trim()).filter(Boolean);
const KNOWN_REFERENCE_HASHES = Array.isArray(A.known_reference_hashes)
  ? A.known_reference_hashes.filter((r) => r && r.path_sha256 &&
    (r.sha256 || r.normalized_sha256)) : [];
if (STRICT_AUTONOMY && KNOWN_REFERENCE_PATHS.length) {
  throw new Error(
    'args.strict_autonomy refuses known_reference_paths in workflow args; supply ' +
    'known_reference_hashes generated out of band so no agent-readable address is published.');
}
if (CAPABILITY_EVAL) {
  log(`CAPABILITY EVAL mode: prior art is advisory CONCLUSIONS only; engineers get no reference ` +
      `paths, and verify rejects byte-identical files as status:"plagiarized". ` +
      `reference evidence=${KNOWN_REFERENCE_HASHES.length
        ? `${KNOWN_REFERENCE_HASHES.length} out-of-band file hash(es)`
        : (KNOWN_REFERENCE_PATHS.length ? `${KNOWN_REFERENCE_PATHS.length} hidden path(s)` :
          '(none given — provenance check DISABLED)')}`);
}

// REFERENCE CONTAINMENT. Withholding the location does not make the tree unreachable. Engineers run
// with a shell in a workspace that sits inside the project tree; one `ls ..` or one repo-wide grep
// walks straight into a sibling reference checkout, and both waves of this run did exactly that —
// wave 1 filed an 8-of-11-files byte-identical patch, and wave 2 repeated it at 11 of 12, differing
// only in the default of one env lookup. Both times the doctrine was present and correctly worded.
// Prose cannot fence off a directory that is one relative path away, so this is checked instead of
// asked for: under capability_eval a reference tree must live OUTSIDE the ancestor that contains the
// task, the workflow and the run artifacts. Move it (`git worktree move` if it is a worktree) rather
// than deleting the mention — a stale path in an old report is harmless once it resolves to nothing.
//
// Set by this block and consumed by the post-Setup enforcement gate, which sweeps the same ancestor.
let RUN_TREE_ANCESTOR = '';
if (CAPABILITY_EVAL) {
  // Normalized by hand: workflow scripts run without Node's `path`/`fs`, so require() would throw
  // here at runtime and turn a safety check into a crash.
  const abs = (p) => {
    const out = [];
    for (const seg of String(p).split('/')) {
      if (!seg || seg === '.') continue;
      if (seg === '..') out.pop(); else out.push(seg);
    }
    return '/' + out.join('/');
  };
  const parts = [KERNEL_PATH_ORIG, EXP_ROOT, WORKFLOW_DIR].filter(Boolean).map(abs);
  let ancestor = parts.length ? parts[0] : '';
  for (const p of parts.slice(1)) {                       // longest common directory prefix
    const a = ancestor.split('/'), b = p.split('/');
    let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++;
    ancestor = a.slice(0, i).join('/') || '/';
  }
  RUN_TREE_ANCESTOR = ancestor;
  const inside = KNOWN_REFERENCE_PATHS.filter(
    (p) => ancestor && (abs(p) === ancestor || abs(p).startsWith(ancestor + '/')));
  if (inside.length) {
    throw new Error(
      `REFERENCE NOT CONTAINED: capability_eval is on, but ${inside.length} reference tree(s) sit ` +
      `inside the run's own tree (${ancestor}): ${inside.join(' ')}. An engineer reaches these by ` +
      `walking up one directory, so the run would measure a copy and report it as a derivation — ` +
      `that has already happened twice here. Move each tree outside ${ancestor} (for a git worktree: ` +
      `git worktree move <old> <new>) and pass the new path in known_reference_paths, so verify can ` +
      `still compare against it. Set capability_eval=false if you WANT the reference ported.`);
  }
  log(`REFERENCE CONTAINMENT ok: ${KNOWN_REFERENCE_PATHS.length} reference tree(s) resolve outside ` +
      `the run tree ${ancestor}. NOTE this checks CONFIGURED PATHS ONLY, and that is a small fraction ` +
      `of the exposure. Sweeping this tree for reference-only markers afterwards found the answer in ` +
      `five more places, none of them a known_reference_path: the positive control's own applied ` +
      `workspace left behind under artifacts/, a previous wave's output tree, the SIBLING EVAL DIRS ` +
      `inside exp_root (one directory up from the dir every engineer is handed by name), the durable ` +
      `patches in state_dir, and a git bundle of the candidate branch. Run ` +
      `scripts/reference_leak_sweep.sh --tree ${ancestor} before trusting capability_eval, and note ` +
      `it still cannot see a reference reachable as a branch in another repository the run can read ` +
      `(JIT cache, sibling clone) — only verify's byte-identity check stands between that and a ` +
      `counted win.`);
}

// ---- Profile-analysis skill (OPTIONAL, pluggable; mirrors e2e_workflow's analysis_skill — see
// knowledge/analysis_skills/INDEX.md). After profile_engineer classifies the bottleneck, a separate
// analysis_engineer may run ONE analysis skill to enrich it with operator-specific structure (e.g. MoE
// route/expert imbalance, Stage1/Stage2/combine time-share) that the generic bottleneck labels
// (compute/memory/latency/lds/balanced/overhead) cannot express. ANALYSIS ONLY: emits findings,
// bounded hypotheses, constraints, bounds, unknowns and unranked references—never directions.
// `analysis_skill=none` (the default here) injects no input keys and no role-prompt text.
// Default OFF (unlike e2e's default-ON 'roofline'): kernel_workflow serves many non-MoE kernels, so an
// MoE-specific skill should not silently fire on every run — matches the USE_EXPERT_SKILLS precedent.
const ANALYSIS_SKILL = String(A.analysis_skill != null ? A.analysis_skill : 'none').trim();
const ANALYSIS_SKILL_ON = !!ANALYSIS_SKILL && ANALYSIS_SKILL !== 'none' && ANALYSIS_SKILL !== 'false';
if (ANALYSIS_SKILL_ON && !/^[a-z0-9][a-z0-9_-]*$/.test(ANALYSIS_SKILL)) {
  throw new Error(`args.analysis_skill must be a safe skill slug, got ${JSON.stringify(ANALYSIS_SKILL)}`);
}
const ANALYSIS_SKILL_INPUTS = ANALYSIS_SKILL_ON ? {
  ANALYSIS_SKILL: ANALYSIS_SKILL,
  ANALYSIS_SKILL_DIR: `${WORKFLOW_DIR}/knowledge/analysis_skills/${ANALYSIS_SKILL}`,
} : {};
if (ANALYSIS_SKILL_ON) log(`Profile-analysis skill: ${ANALYSIS_SKILL} (analysis only; Step-3 TechLead owns directions).`);

// ---------------------------------------------------------------------------
// DEEP-MODE continuation + cross-backend / e2e-feedback hooks. ALL OPTIONAL.
// When none are passed (every normal/fast e2e run, and every standalone run) these are '' / the
// current defaults and are NEVER threaded into a prompt — so behavior is byte-identical to the
// pre-feature build. They are set ONLY by e2e_workflow's deep_mode head scheduler.
//   STATE_DIR        a STABLE dir for THIS (kernel,backend) ACROSS deep waves. When set the run
//                    RESUMES: director seeds the canonical from STATE_DIR/best (the cumulative-best
//                    code) and returns prior_state (cumulative + history) so re-invocation CONTINUES
//                    instead of restarting (no lost experience, no re-explored directions). The frozen
//                    oracle baseline (immutable unittest.py/meta.json) stays the reference, so speedups
//                    remain comparable to the TRUE baseline across waves. update_memory writes STATE.json
//                    + syncs best/ each round.
//   SHARED_KB        cross-backend blackboard file (read by plan+engineers, appended by update_memory).
//   GLOBAL_KB        run-global cross-KERNEL technique blackboard (deep): techniques that generalize
//                    across head ops/backends. Optional; unset (default/fast) => byte-identical prompts.
//   E2E_FEEDBACK     path to the latest end-to-end A/B result + problems from e2e_workflow (engaged?,
//                    cudagraph behavior, mem footprint, decode regression, e2e delta) — steers planning.
//   HARNESS_ADDENDUM path to an e2e-refined harness addendum (timing-weight / cudagraph-capture / hard
//                    constraint gates). The IMMUTABLE oracle is NEVER touched; this only refines what the
//                    perf bench emphasizes so the isolated target aligns with e2e.
//   MAX_NO_IMPROVE   consecutive non-improving rounds before stopping (default 2 = current behavior).
const STATE_DIR = String(A.state_dir || '').replace(/\/+$/, '');
const SHARED_KB = String(A.shared_kb || '').trim();
const GLOBAL_KB = String(A.global_kb || '').trim();   // run-global cross-KERNEL technique blackboard (deep)
const E2E_FEEDBACK = String(A.e2e_feedback || '').trim();
const HARNESS_ADDENDUM = String(A.harness_addendum || '').trim();
// P2 (deep continuation): on a RESUMED wave (STATE_DIR holds prior work) the cold re-derivation of
// Analyze + baseline Profile is largely redundant. INCREMENTAL_RESUME tells those two ADVISORY agents to
// load the prior roadmap/profile and return it with only delta updates instead of re-deriving from
// scratch, so each burst spends its budget on optimization ROUNDS, not re-analysis. Benchmark is NEVER
// incremental (it re-pins a fresh in-window baseline every wave \u2014 the matched-A/B correctness rail).
// Unset (default/fast / first deep burst) => spreading {} adds nothing => byte-identical prompts.
const INCREMENTAL = !!STATE_DIR && String(A.incremental_analyze || '') === 'true';
const RESUME_INPUT = INCREMENTAL ? { INCREMENTAL_RESUME: '1' } : {};
const MAX_NO_IMPROVE = Math.max(1, parseInt(A.max_no_improve != null ? A.max_no_improve : 2, 10));
// Conditional inputs: spreading {} adds NOTHING to a prompt (byte-identical) when a hook is unset.
const KB_INPUTS = {
  ...(SHARED_KB ? { SHARED_KB } : {}),
  ...(GLOBAL_KB ? { GLOBAL_KB } : {}),
  ...(E2E_FEEDBACK ? { E2E_FEEDBACK } : {}),
  ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
};

// ---------------------------------------------------------------------------
// Reusable JSON-schema fragments.
// ---------------------------------------------------------------------------
const perCase = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      name: { type: 'string' },
      baseline_ms: { type: 'number' },
      optimized_ms: { type: 'number' },
      speedup: { type: 'number' },
      // Workload-alignment fields (present only when a WORKLOAD_SPEC drives the harness; absent
      // on a normal unweighted run). weight = this case's baseline time SHARE in the real workload;
      // it is the coefficient of the time-weighted metric Σweight / Σ(weight/speedup). count is
      // optional/informational (regime-attributed cases have no per-call count).
      weight: { type: 'number' },
      count: { type: 'number' },
      dims: { type: 'array', items: { type: 'array', items: { type: 'number' } } },
      dtypes: { type: 'array', items: { type: 'string' } },
      weight_source: { type: 'string' }, // trace | regime | regime_floor | prior | caller
    },
    required: ['name', 'speedup'],
  },
};
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
// Verdict vocabulary keeps clean/leak/skipped/unknown distinct. "The scanner could not run" can
// never be reported as clean — that collapse is the exact failure this subsystem exists to prevent.
const CONTAINMENT_GATE_SCHEMA = obj({
  verdict: { type: 'string', enum: ['clean', 'leak', 'skipped', 'unknown'] },
  findings: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

// The pool sample is three-valued for the same reason the containment verdict is: "we could not read
// the pool" must not collapse into "the pool is free". A round planned as if the hardware were
// available, when it is not, spends its entire clock inside flock and returns nothing.
const POOL_SCHEMA = obj({
  verdict: { type: 'string', enum: ['free', 'occupied', 'unknown'] },
  cards_total: { type: 'number' },
  cards_free: { type: 'number' },
  min_free_gib: { type: 'number' },
  foreign_pids: { type: 'array', items: { type: 'string' } },
  note: { type: 'string' },
}, ['verdict']);

const SETUP_SCHEMA = obj({
  eval_dir: { type: 'string' }, workspace: { type: 'string' }, baseline_dir: { type: 'string' },
  kernel_name: { type: 'string' }, source_files: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // Frozen-baseline verdict (BOTH modes). The unittest's timing + random-value parity baseline MUST be
  // the real online kernel — the immutable baseline_src/ dir OR an importable meta.baseline_callable —
  // never kernel_src/ (the candidate's own scaffold). The director sets baseline_frozen=true after it
  // copies baseline_src/ + confirms meta.baseline_callable; the script aborts the run if neither holds.
  baseline_frozen: { type: 'boolean' }, baseline_callable: { type: 'string' },
  // DEEP-MODE resume only: populated by the director ONLY when STATE_DIR was provided AND a prior best
  // exists there. Lets a continued wave restore its cumulative speedup + insight/ledger history so it
  // does not re-explore dead directions. Absent (undefined) on a fresh run -> no behavior change.
  resumed: { type: 'boolean' },
  workflow_revision: { type: 'string' },
  workflow_clean: { type: 'boolean' },
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    // Declared, because memoryMerge keys rows by `id` and dedupes on it, and the ladder is read back
  // out of `roadmap_rung`. A row missing `id` merges as a new row every round; a row missing
  // `verdict` is treated by the next plan_round as untried. Both read as normal ledger growth.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
    // The candidate shelf, declared for the same reason the ledger rows above are: the shelf is
    // keyed by `id`, aged by `base_round`, and offered or withheld on `files`. An entry missing
    // `files` is not "touches nothing" — shelfAdd marks it footprint=unknown and it is never
    // offered, so a dropped field costs an offer rather than producing a bad one.
    shelf: { type: 'array', items: obj({
      id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
      geomean: { type: 'number' }, patch: { type: 'string' },
      files: { type: 'array', items: { type: 'string' } },
      // The fusion chain this entry belongs to, identified by its terminal rung. Carried across
      // waves so a chain's own producer commit does not age its consumer half off the shelf. Empty
      // for a standalone optimisation. See the candidate_shelf region.
      chain_id: { type: 'string' },
      footprint: { type: 'string', enum: ['known', 'unknown'] },
      base_round: { type: 'number' }, shelved_round: { type: 'number' },
      absorbed: { type: 'boolean' },
    }, ['id', 'geomean']) },
    // round -> files CANONICAL took on that round. Keys are round numbers as strings.
    absorbed_files: { type: 'object' },
  }, []),
}, ['eval_dir', 'workspace', 'kernel_name']);

const AUTHOR_SCHEMA = obj({
  authored: { type: 'boolean' }, target_language: { type: 'string' }, correctness: { type: 'string' },
  baseline_ms: { type: 'number' }, kernel_src_path: { type: 'string' }, entry_point: { type: 'string' },
  build: { type: 'boolean' }, notes: { type: 'string' },
}, ['authored', 'correctness']);

const ANALYZE_SCHEMA = obj({
  kernel_type: { type: 'string' }, kernel_file: { type: 'string' }, entry_point: { type: 'string' },
  modifiable_files: { type: 'array', items: { type: 'string' } },
  bottleneck_guess: { type: 'string' }, roadmap_summary: { type: 'string' },
  // THE LADDER. Analyze's ranked candidate list — and the only place the run records what the
  // ORDER was and what each rung is CONDITIONAL on. Each entry should carry, beyond the free-form
  // fields: `id` (short, stable, quotable — `D0`, `D1`, …), `gated_on` (rung ids that must have
  // been *executed to spec* first, [] if none), `mandatory_arms` (arms without which the rung's
  // result cannot be interpreted, e.g. a publish-only arm that prices producer cost separately
  // from consumer benefit), and `is_positive_control` when the rung IS this run's control.
  //
  // Why the conditions have to be structured rather than prose. Wave 13's Analyze wrote a correct
  // four-rung ladder — bounding readout, then a positive control, then a readiness signal, then the
  // persistent fusion gated on the readiness signal — into `roadmap.md` and this field, and then
  // every round planned from the profile instead. D0 and D1 never ran; D2 ran first, in one
  // implementation that omitted its own mandatory publish-only arm; it read negative; and D3, the
  // acceptance-shape fusion, was never proposed because its gate was written against D2. The
  // ladder was right and it was structurally an orphan. Prose cannot be checked against what was
  // dispatched. A list of ids can. See roadmapLadderGate.
  //
  // So they are DECLARED, not just described. Left opaque, a ladder that omits `gated_on` validates
  // and every rung then looks unconditional — the gate finds no unmet prerequisite and reports OK
  // on exactly the run whose ordering was lost. `gated_on: []` is a real answer and must be given,
  // not inferred from absence.
  candidate_directions: { type: 'array', items: obj({
    id: { type: 'string' },              // short, stable, quotable: D0, D1, ...
    title: { type: 'string' },
    rationale: { type: 'string' },
    expected_speedup: { type: 'number' },
    gated_on: { type: 'array', items: { type: 'string' } },
    mandatory_arms: { type: 'array', items: { type: 'string' } },
    is_positive_control: { type: 'boolean' },
    // How this rung is CLOSED. `terminal` (default) closes on a measurement. `enabling` closes on
    // functional acceptance, because a prerequisite step has no standalone speedup to measure --
    // demanding one is what leaves a fusion permanently half-built. See stepRoleOf below.
    step_role: { type: 'string', enum: ['enabling', 'terminal'] },
    enables: { type: 'string' },
    cost_budget_pct: { type: 'number' },
    // A terminal rung must say what shape closes it. Without this a three-launch partial fusion and
    // the requested two-launch terminal are indistinguishable to the rung tracker.
    target_shape: {
      type: 'object',
      properties: {
        launches: { type: 'number' },
        stages_fused: { type: 'array', items: { type: 'string' } },
        require_overlap: { type: 'boolean' },
      },
      required: ['launches'],
      additionalProperties: true,
    },
  }, ['id', 'title', 'gated_on']) },
  // perf_knowledge resolution (REFERENCE ONLY): the operator/language this kernel maps to in the
  // AMD perf_knowledge base, plus the most relevant card paths, so engineers read focused context
  // instead of re-navigating the whole base. Empty string / [] / null when no card applies.
  kk_operator: { type: ['string', 'null'] }, kk_language: { type: ['string', 'null'] },
  kk_refs: { type: 'array', items: { type: 'string' } },
  // PRIOR ART IN-TREE: directions that are ALREADY IMPLEMENTED somewhere reachable (a sibling branch,
  // a worktree, an env-gated opt-in path in this very file) — even if not enabled in the tree under
  // optimization. Measuring an existing switch costs one A/B; re-deriving it costs a round and
  // usually fails. A run once optimized a tree from which ~1000 lines of already-written, already-
  // measured (+4.71%) fusion were absent, never noticed, and reported 1.000x.
  //   [{direction, implemented_at, how_to_enable, measured_effect, in_baseline}]
  prior_art: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // TILE-LEVEL DEPENDENCY GRAPH. Required when args.require_task_graph is set (see the gate below
  // the Analyze call); optional otherwise, because a single-op elementwise kernel has no interesting
  // graph and demanding one would only produce a filled-in form.
  //
  // Why this is an ARTIFACT and not prose. An Analyze phase that writes "the stages are serialized,
  // so fuse them" has restated the launch count, not analysed a dependency. The difference is not
  // stylistic: only the graph can say which orderings the DATA requires versus which are artifacts
  // of the current code, what the critical path is (the floor on any schedule — a proposal claiming
  // more than measured_e2e minus critical_path_us is arithmetically wrong), and which nodes have
  // slack (optimizing those changes nothing). None of that is answerable at kernel granularity, and
  // a run that cannot produce the graph has not done the analysis whatever prose it returns.
  //
  //   {nodes:      [{id, stage, tile, duration_us, source}],   // source: profile|derived|assumed
  //    edges:      [{from, to, scope, enforced_by, bytes}],
  //                //   scope:       register|lds|l2|hbm|cross_die|cross_rank
  //                //   enforced_by: launch_boundary|barrier|fence_flag|none_needed
  //    critical_path: [nodeId], critical_path_us, measured_e2e_us,
  //    zero_slack_nodes: [nodeId], false_edges: [{from,to,why}],
  //    unknowns:   [{what, why, what_would_settle_it}]}
  //
  // `unknowns` is load-bearing and not a confession: an edge whose scope could not be determined is
  // a FACT, and an estimate presented as a measurement is not. A short honest graph outranks a
  // complete invented one, and `source: 'assumed'` on every node is itself the finding.
  // See knowledge/tile_task_graph.md for the derivation method.
  //
  // The fields taskGraphGate READS are declared below rather than left to the comment above.
  // `additionalProperties: true` keeps the rest of the documented shape legal, but a field the gate
  // counts must be named in the schema: with the whole object opaque, a graph that calls the
  // enforcement column anything other than `enforced_by` validates, and the gate then reports
  // "0 edges enforced only by a launch boundary" — which reads as the finding "nothing to unfuse"
  // when the truth is "the column was never filled in". Those two must not look alike.
  task_graph: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      nodes: { type: 'array', items: obj({
        id: { type: 'string' }, stage: { type: 'string' }, duration_us: { type: 'number' },
        // measured | assumed. The gate counts assumed durations, because a graph whose every
        // duration is assumed is a hypothesis wearing a measurement's formatting.
        source: { type: 'string', enum: ['measured', 'assumed'] },
      }, ['id', 'source']) },
      edges: { type: 'array', items: obj({
        from: { type: 'string' }, to: { type: 'string' },
        // data | launch_boundary | barrier | assumed. `launch_boundary` is the load-bearing value:
        // it marks an ordering the DATA does not require, which is the only kind fusion can delete.
        enforced_by: { type: 'string' },
        scope: { type: 'string' },
        fan_in: { type: 'number' },
        // The operands the CONSUMER needs that the producer does not supply: weights, scales,
        // descriptors, index arrays. They depend on nothing, so they can be loaded while the
        // producer is still running. The graph's nodes are output tiles, so these have no node and
        // no edge and are otherwise unrankable — naming them here is the only place they appear.
        // An empty array is a real answer; a missing one is the gate's advisory note.
        producer_independent_operands: { type: 'array', items: { type: 'string' } },
      }, ['from', 'to', 'enforced_by']) },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
      critical_path_us: { type: 'number' },
      measured_e2e_us: { type: 'number' },
    },
  },
  // PER-PIPE RESOURCE TIMELINE. The other half of the analysis, required alongside task_graph.
  //
  // The graph answers whether two pieces of work MAY overlap. That is necessary and not sufficient,
  // and a program that only asks it proposes overlaps whose pipe was already saturated, or fuses to
  // reclaim a launch gap that measures zero. This artifact answers the complementary question:
  // which functional unit is idle, when, and what dependency-free work could be issued into it.
  //
  //   {pipes: [{stage, pipe, utilization_pct, source}],   // pipe: valu|mfma|vmem|lds|hbm|interconnect|scalar
  //    interkernel_gap_us: {median, max, n_boundaries},
  //    class: throughput_bound|latency_bound|launch_bound|mixed,
  //    stall_reason: [{stage, waiting_on, counter}],
  //    idle_pipe_opportunities: [{stage, idle_pipe, candidate_work, dag_edge_status, blocked_by}],
  //    closed_axes: [{axis, ruled_out_by}],
  //    unknowns: [{what, why, what_would_settle_it}]}
  //
  // `class` is what forecloses lever families before they cost a lease: all pipes low with a
  // near-zero inter-kernel gap is latency_bound, where raising occupancy is the wrong medicine
  // (adding waves creates no independent work inside a wave whose instruction stream has none, and
  // it spends the registers software pipelining needs) and fusing for launch overhead is dead on
  // arrival. `idle_pipe_opportunities` is what a fusion direction is actually FOR in that state: a
  // kernel boundary is a grid-wide barrier plus a pipeline drain, so the next stage's weight loads
  // and index preprocessing are not merely unscheduled, they are inexpressible.
  // See knowledge/pipe_occupancy.md.
  //
  // Same rule as task_graph: what the gate reads is declared. `utilization_pct` in particular is
  // the number every direction this run proposes will be priced against, and the gate DROPS any
  // pipe row whose value is not finite — so a row that omits it or ships it as a string vanishes
  // silently, and a table of five pipes with one saturated row can arrive at the gate as an empty
  // table reported as PIPE TABLE MISSING.
  resource_timeline: {
    type: ['object', 'null'],
    additionalProperties: true,
    properties: {
      pipes: { type: 'array', items: obj({
        stage: { type: 'string' },
        pipe: { type: 'string' },   // valu|mfma|vmem|lds|hbm|interconnect|scalar
        utilization_pct: { type: 'number' },
        source: { type: 'string' },
      }, ['stage', 'pipe', 'utilization_pct']) },
      interkernel_gap_us: obj({
        median: { type: 'number' }, max: { type: 'number' }, n_boundaries: { type: 'number' },
      }, ['median']),
      // The gate DERIVES the class from the numbers and compares it against this one; a mismatch is
      // itself reportable, so the stated value must be present to be contradicted.
      class: { type: 'string', enum: ['throughput_bound', 'latency_bound', 'launch_bound', 'mixed'] },
      // The gate reads `id`, the two size fields, and `rides_on`, so all four are named here rather
      // than left to `additionalProperties`. A hole with a size and no owner is reported; `rides_on`
      // is how Analyze says "another rung absorbs this" without the claim going unrecorded.
      idle_pipe_opportunities: { type: 'array', items: obj({
        id: { type: 'string' }, window: { type: 'string' }, idle_pipe: { type: 'string' },
        recoverable_us: { type: ['number', 'null'] }, pct_of_e2e: { type: ['number', 'null'] },
        rides_on: { type: 'string' },
      }, []) },
      closed_axes: { type: 'array', items: { type: 'object', additionalProperties: true } },
      unknowns: { type: 'array', items: { type: 'object', additionalProperties: true } },
    },
  },
}, ['kernel_type', 'roadmap_summary']);

const BENCH_SCHEMA = obj({
  commandment_path: { type: 'string' }, correctness_cmd: { type: 'string' },
  benchmark_cmd: { type: 'string' }, profile_cmd: { type: 'string' }, parse_hint: { type: 'string' },
  baseline_per_case: { type: 'array', items: { type: 'object', additionalProperties: true } },
  baseline_geomean_ms: { type: 'number' }, num_test_cases: { type: 'number' },
  // Workload-aligned outputs: present when a WORKLOAD_SPEC drove case selection + weights.
  // baseline_weighted_total_ms = the baseline time the weights represent (Σ weight_i in time units).
  // The metric is Σ weight_i / Σ (weight_i/speedup_i). workload_aligned flags weights are real (not 1).
  workload_aligned: { type: 'boolean' },
  baseline_weighted_total_ms: { type: 'number' },
  weights_provenance: { type: 'string' }, // e.g. "trace" | "regime" | "regime_floor" | "prior" | "caller" | "mixed"
  // Result of args.positive_control, when one was supplied. `measured_pct` is the delta the harness
  // actually recovered for a change whose effect is already known; `passed` is whether it landed in
  // the expected band. This is the run's evidence that its own measurement loop can SEE a real win.
  //   {ran, switch_present, switch_checked, measured_pct, expected_lo, expected_hi, passed, reps,
  //    null_arm_pct, note}
  // `switch_present` is the pre-flight: does the knob `how` names actually EXIST in the tree the
  // control will run in? It is separated from the reading because the failure it catches produces
  // a perfectly well-formed reading. A prescribed control named an env var absent from every tree
  // in the run; both arms therefore executed the same code; it measured -1.00%, a base-vs-base null
  // wearing a control's costume, and the only reason the wave did not proceed on a dead instrument
  // is that an engineer went looking. Absent switch => the control did not run, whatever number
  // came back.
  // `control_pairs_pct` and `null_pairs_pct` — the individual paired deltas, not just their medians —
  // are read by the gate: sign agreement across pairs and the WORST null pair are what separate a
  // small real effect from a small piece of drift, and a median hides both.
  // Declared, not merely described above. The gate reads seven fields off this object and every one
  // of them has a DEFAULT that reads as success when the field is absent: `ran` missing means it
  // ran (`pc.ran !== false`), `switch_present` missing means the pre-flight is inert, `null_arm_pct`
  // missing makes the overshoot check silently non-quiet. An opaque object lets a renamed or
  // dropped field validate and then be interpreted as a passing control.
  positive_control: obj({
    ran: { type: 'boolean' },
    switch_present: { type: 'boolean' },
    switch_checked: { type: 'string' },
    measured_pct: { type: 'number' },
    expected_lo: { type: 'number' }, expected_hi: { type: 'number' },
    passed: { type: 'boolean' }, reps: { type: 'number' },
    control_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_pairs_pct: { type: 'array', items: { type: 'number' } },
    null_arm_pct: { type: 'number' },
    note: { type: 'string' },
    // Required only when the object exists at all: a run with no control omits it entirely. When it
    // IS reported, these three are what separate a control from a number, so none may be defaulted.
  }, ['ran', 'switch_present', 'measured_pct']),
  reliable: { type: 'boolean' }, notes: { type: 'string' },
}, ['commandment_path', 'baseline_per_case', 'baseline_geomean_ms']);

const PROFILE_SCHEMA = obj({
  bottleneck: { type: 'string' }, profiler_used: { type: 'string' }, dispatch_count: { type: 'number' },
  // The accelerator detected on-box (e.g. "MI300X / gfx942 / CDNA3, 304 CU, ~5.3 TB/s"), so the
  // roofline ceiling + grid-sizing advice downstream use the real card instead of an assumed MI300X.
  device: { type: 'string' },
  key_metrics: { type: 'object', additionalProperties: true },
  top_kernels: { type: 'array', items: { type: 'object', additionalProperties: true } },
  top_opportunities: { type: 'array', items: { type: 'string' } },
  summary_path: { type: 'string' }, shift_note: { type: 'string' },
}, ['bottleneck', 'top_opportunities']);

const ANALYSIS_RESULT_SCHEMA = obj({
  status: { type: 'string', enum: ['ready', 'degraded'] },
  analysis_skill: { type: 'string' },
  analysis_schema_version: { type: 'string' },
  analysis_status: {
    type: 'string',
    enum: ['awaiting_measurement', 'evidence_complete', 'unavailable'],
  },
  analysis_json: { type: 'string' },
  failure_note: { type: 'string' },
}, [
  'status',
  'analysis_skill',
  'analysis_schema_version',
  'analysis_status',
  'analysis_json',
  'failure_note',
]);

const PLAN_SCHEMA = obj({
  stop: { type: 'boolean' }, reasoning: { type: 'string' },
  directions: {
    type: 'array',
    items: obj({
      id: { type: 'string' }, title: { type: 'string' },
      specialty: { type: 'string', enum: ['algorithm', 'memory', 'compute', 'host_runtime', 'distributed', 'deep_explore'] },
      focus_files: { type: 'array', items: { type: 'string' } },
      expected_speedup: { type: 'number' }, prompt: { type: 'string' },
      kk_refs: { type: 'array', items: { type: 'string' } }, // optional: perf_knowledge card paths for THIS direction (REFERENCE ONLY)
      // Binds the direction to the Analyze artifacts so the graph and the pipe table are READ when
      // the lease is allocated, not merely emitted. `fills_pipe` is the functional unit this
      // direction puts work into (or removes work from); `pipe_util_pct` is that pipe's measured
      // current utilization from `resource_timeline.pipes`; `headroom_basis` says which bound caps
      // the claim — the pipe's idle fraction or `measured_e2e_us - critical_path_us`. A direction
      // that cannot name a pipe is a hunch. See knowledge/pipe_occupancy.md.
      fills_pipe: { type: 'string' },
      pipe_util_pct: { type: 'number' },
      headroom_basis: { type: 'string' },
      graph_refs: { type: 'array', items: { type: 'string' } }, // task_graph node/edge ids this direction acts on
      // The `resource_timeline.idle_pipe_opportunities` id this direction fills. `fills_pipe` says
      // which functional unit is idle; this says WHICH sized window of idleness is being collected,
      // which is what makes a hole checkable as owned or unowned at the end of the round.
      fills_hole: { type: 'string' },
      // Binds the direction to Analyze's LADDER the way `fills_pipe` binds it to the pipe table.
      // `roadmap_rung` is the candidate_directions id this direction implements, or the literal
      // `off_ladder`. `rung_deviation` is REQUIRED whenever the rung is off_ladder, or is being
      // taken out of the ladder's order, or has unsatisfied `gated_on`: say what is being skipped
      // and what happens to it. Skipping a rung is allowed — the ladder is a plan, and the round's
      // evidence may be better than it. Skipping one silently is what is not: that is how a run
      // loses its own positive control and its own acceptance-shape direction in the same wave
      // without either loss appearing anywhere in the log.
      roadmap_rung: { type: 'string' },
      rung_deviation: { type: 'string' },
      // WHAT KIND OF STEP THIS IS, and therefore what it is judged on. `terminal` (the default) is
      // judged on speed. `enabling` is a prerequisite in a multi-step fusion and is judged on
      // function -- builds, path taken, correct, no deadlock -- because the producer half of a
      // fusion cannot be faster on its own: it adds signalling and buffering and has no consumer
      // yet. `enables` names the terminal rung it is a prerequisite for and is REQUIRED when the
      // role is enabling. `cost_budget_pct` is how much slower this step is expected to make the
      // operator; exceeding it is a design error rather than an expected temporary slowdown.
      step_role: { type: 'string', enum: ['enabling', 'terminal'] },
      enables: { type: 'string' },
      cost_budget_pct: { type: 'number' },
      gated_on: { type: 'array', items: { type: 'string' } },
      mandatory_arms: { type: 'array', items: { type: 'string' } },
      target_shape: {
        type: 'object',
        properties: {
          launches: { type: 'number' },
          stages_fused: { type: 'array', items: { type: 'string' } },
          require_overlap: { type: 'boolean' },
        },
        required: ['launches'],
        additionalProperties: true,
      },
    }, ['id', 'title', 'specialty', 'prompt']),
  },
}, ['stop', 'directions']);

const ENG_SCHEMA = obj({
  engineer_id: { type: 'string' }, specialty: { type: 'string' }, task: { type: 'string' },
  strategy: { type: 'string' }, speedup_geomean: { type: 'number' }, speedup_arithmetic: { type: 'number' },
  // Time-weighted ratio-of-sums vs the TRUE baseline (PRIMARY metric when workload_aligned).
  // = Σ weight_i / Σ (weight_i / speedup_i). Omitted on unweighted runs.
  speedup_weighted: { type: 'number' },
  per_case: perCase, status: { type: 'string' }, patch_file: { type: 'string' },
  // Concrete arms completed by the engineer. Verify independently reports its own list; a rung
  // whose mandatory control arm was omitted stays open even if the candidate produced a number.
  arms_run: { type: 'array', items: { type: 'string' } },
  strategies_tried: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' },
  // How the new code is TURNED ON, and how an independent party can prove it ran. A patch whose
  // fast path is gated behind an env var that nobody sets benchmarks as byte-identical to the
  // baseline and reads 1.000x — indistinguishable from "the idea did not work", which is the one
  // reading that stops the round. That has already happened here. `mode` is `default_on` (the
  // required default: the patch changes behavior with no switch at all) or `switch`, which REQUIRES
  // `switch_name`/`switch_value` and a `path_marker` verify can grep for.
  activation: obj({
    mode: { type: 'string' }, switch_name: { type: 'string' }, switch_value: { type: 'string' },
    path_marker: { type: 'string' }, marker_how: { type: 'string' },
  }, []),
}, ['status', 'speedup_geomean']);

const VERIFY_SCHEMA = obj({
  status: { type: 'string' }, correctness: { type: 'string' },
  verified_geomean: { type: 'number' }, verified_arithmetic: { type: 'number' },
  verified_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  per_case: perCase, variance_note: { type: 'string' }, notes: { type: 'string' },
  graph_safe: { type: 'string' },
  liveness: { type: 'string' }, // pass|fail|n/a — deadlock/stale-read stress (distributed specialty)
  replay_count: { type: 'number' },
  replay_results: { type: 'array', items: obj({
    guard: { type: 'string' }, count: { type: 'number' },
    status: { type: 'string' }, graph_safe: { type: 'string' },
  }, ['guard', 'count', 'status']) },
  arms_run: { type: 'array', items: { type: 'string' } },
  // ACCURACY. Acceptance criterion 4 is "relL2 < 0.10", and until now nothing carried the number: a
  // free-text `correctness` that the script only checks `startsWith('pass')` cannot be compared to a
  // threshold, so "precision is fine" was an executor's judgement with no figure travelling behind
  // it. Declared so functionalAcceptance can check `value < threshold` mechanically and the report
  // can print it. `metric` is the name of the error measure (relL2 here); `guard`/`route` is which
  // route it was measured on, because a relL2 read on one route says nothing about another. Absent
  // fields degrade to the old string check rather than crashing a run that never reports them.
  accuracy: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, []),
  accuracy_results: { type: 'array', items: obj({
    metric: { type: 'string' }, value: { type: 'number', minimum: 0 },
    threshold: { type: 'number', exclusiveMinimum: 0 },
    guard: { type: 'string' }, method: { type: 'string' },
  }, ['metric', 'value', 'threshold', 'guard']) },
  // LAUNCH SHAPE. Acceptance criterion 1 is "fully fused, TWO launches": one megakernel per EP rank
  // plus the one separate pre-dispatch quant launch. Nothing in the workflow counted launches, so a
  // candidate that fused dispatch+gemm1 but still launched combine separately (three launches) was
  // indistinguishable from a real two-launch fusion. `launches_base`/`launches_cand` are the kernel
  // launch counts per EP rank for one operator call, read PAIRED in the same collection; `target` is
  // the acceptance shape (2 for this campaign). `how_counted` is the evidence -- a trace record
  // count, a launch-marker tally -- because a claimed count with no method is a guess. Absent fields
  // do not crash a run; they leave criterion 1 UNJUDGED for that candidate, which the report says.
  launch_shape: obj({
    launches_base: { type: 'number' }, launches_cand: { type: 'number' },
    per_rank: { type: 'boolean' }, target: { type: 'number' },
    stages_fused: { type: 'array', items: { type: 'string' } }, how_counted: { type: 'string' },
  }, []),
  // How the number above was actually obtained. `reps` = interleaved A,B,A,B pairs (NOT total runs);
  // `null_arm_pct` = the delta measured by an arm doing byte-identical work, i.e. this run's own
  // noise floor. A claimed win smaller than |null_arm_pct| is unreadable no matter how it was
  // computed. Absent/low values do not void the result — they downgrade it to PROVISIONAL and are
  // surfaced in the log and report, because "1 rep, no null arm" has already produced a -0.44%
  // "win" that sat inside a 1.45% per-case spread.
  reps: { type: 'number' }, null_arm_pct: { type: 'number' },
  // The RAW interleaved readings behind `reps`, one row per pair, tagged with the guard/route each
  // was taken on. Declared so the driver can classify the bimodal 512 guards ARM-BLIND and condition
  // on the state instead of averaging over two discrete ones -- the doctrine in benchmark_engineer.md
  // that until now lived only as prose because nothing carried the raw pairs to the code that applies
  // it (modeSplit/pairedModeAware). It also lets the driver check the number came from a TARGET route
  // and not a skew rail: on this campaign the metric is the uniform route's own time, and a win read
  // off `512_rank-mixed-skew` is out of scope. `base`/`cand` are the paired rank-max timings (ms) for
  // one A,B pair. Absent leaves the aggregate `verified_geomean` in charge exactly as before.
  paired_readings: { type: 'array', items: obj({
    guard: { type: 'string' }, base: { type: 'number' }, cand: { type: 'number' },
  }, ['guard', 'base', 'cand']) },
  // Did the patched code path actually EXECUTE during the measured run? `yes` requires the
  // path_marker to have been observed; `no` means the arm labelled "candidate" ran baseline code,
  // which makes the whole comparison void rather than negative. `unknown` is treated as `no` — an
  // unproven activation is exactly the state that produced a 1.000x on an unexercised patch.
  activation_confirmed: { type: 'string' },  // yes|no|unknown
  activation_evidence: { type: 'string' },   // the command + the marker output that proves it
  // Did the switched/perf-bearing path run ON THE DEVICE this round? `yes` requires a gpu_lock-wrapped
  // benchmark/correctness invocation that reached the candidate with the switch SET — a real launch,
  // reported via a device-side observable (a nonzero mega_e2e reading, a rocprof/trace record for the
  // candidate kernel, or the on-device path marker captured from the torchrun run). A COMPILE_ONLY ISA
  // hash, a py_compile screen, or a CPU dry-run is `no`: they never touched a card, and a JIT-trace-time
  // crash is precisely the fault they cannot see. `n/a` is reserved for a candidate with no switched
  // path at all. When REQUIRE_HW_ACTIVATION is set the script holds a committable/enabling candidate to
  // `yes` here; `no`/`unknown` is void, not negative.
  activation_on_hardware: { type: 'string' }, // yes|no|n/a|unknown
  hardware_evidence: { type: 'string' },       // the gpu_lock cmd + the device-side observable it produced
  // A marker proves the HOST path ran. It does not prove the arms compiled to different code. Under
  // a JIT with a disk cache, two arms can print two different markers and execute one identical
  // cached binary, because the switch never entered the cache key. So for a JIT-compiled candidate
  // the verifier reports the arms' cache keys / IR-ISA hashes / binary paths, and the script checks
  // that they DIFFER. `artifact_distinct` is three-valued: `n/a` is the honest answer for a
  // non-JIT candidate and must stay available, or the verifier is forced to claim a proof it
  // could not run.
  artifact_distinct: { type: 'string' },     // yes|no|n/a|unknown
  artifact_hash_base: { type: 'string' },
  artifact_hash_candidate: { type: 'string' },
  // The files the patch actually touches. The verifier ALREADY computes this list — step 5 of
  // roles/verify_engineer.md diffs it against MODIFIABLE_FILES — and then threw it away, which is
  // the same defect this workflow has now fixed twice: an artifact produced, judged against, and
  // never handed to the role that needs it. The candidate shelf needs it to decide mechanically
  // whether a patch from round 2 still applies at round 5.
  //
  // REQUIRED, and required for the usual reason: omitted, an empty list reads as "touches nothing",
  // which is the maximally-combinable answer. Reporting it explicitly makes "no files" a claim
  // somebody made rather than a default nobody noticed.
  touched_files: { type: 'array', items: { type: 'string' } },
  // OVERLAP. See knowledge/overlap_instrument.md. Declared rather than left to prose because the
  // acceptance question "genuine overlap, not serialization disguised by kernel boundaries" cannot
  // be answered by any of the instruments that exist before fusion: once the stages are one kernel
  // there is one trace record, and both stage timers can RISE while the operator gets faster. The
  // doctrine "a latency win with no measured change in overlap is suspicious" has been in the task
  // text for several waves with nothing behind it, so the honest answer was always "not measured"
  // and the rule never bit.
  //
  // `measured` is three-valued and the three are not interchangeable: `no` means a meter was built
  // and reports no overlap (a finding), `unknown` means it could not be measured (a hole). Collapse
  // them and a fused kernel that was never instrumented reads exactly like one that was checked.
  overlap: obj({
    measured: { type: 'string', enum: ['yes', 'no', 'unknown'] },
    fraction: { type: 'number' },        // wall-clock, >=2 distinct roles active
    cu_fraction: { type: 'number' },     // CU-weighted; wall-clock alone is manufacturable
    method: { type: 'string' },
    // The meter's own controls. `scattered_reading` is the negative control — the unfused path,
    // whose true overlap is known to be zero. A meter that reads high there is broken and voids
    // everything after it. `forced_reading` is the positive control, because a meter that reads 0
    // on both is dead, not conservative, and the two are indistinguishable without it.
    scattered_reading: { type: 'number' },
    forced_reading: { type: 'number' },
    clock_skew_ns: { type: 'number' },   // s_memrealtime coherence across XCDs — assumed at your peril
    meter_overhead_pct: { type: 'number' },
    note: { type: 'string' },
  }, ['measured']),
  // ATTRIBUTION. See knowledge/fusion_preconditions.md, "Three ways a fusion result gets accepted
  // for the wrong reason". It explains a launch-structure win; PROMOTION_METRIC decides whether the
  // operator span or a genuinely comparable changed-kernel measurement creates credit.
  //
  // The two diverge exactly when a patch changes launch structure, and the divergence is not small.
  // Measured on this campaign: a fused candidate whose kernel ran 4878us against the 4774us of the
  // two kernels it replaced -- 2.18% SLOWER -- was promoted to current best on a +4.24% end-to-end
  // reading. The whole 223us lived in the gaps between kernels (residual moved +0.0799 -> -0.2178
  // ms), most likely because the fused kernel's grid-wide join incidentally aligned every rank's
  // consumer start and tightened the arrival window of the NEXT kernel. A real effect, worth a
  // barrier, not worth a megakernel -- and not a result this workflow may claim.
  //
  // Report it whenever the patch adds, removes or merges a launch. `changed_us` is the patched
  // kernel; `replaced_sum_us` is the sum of the kernels it stands in for, read PAIRED in the same
  // collection at the same guard. When the two disagree in sign with the end-to-end claim, the
  // mismatch rejects only when promotion_metric=changed_kernel; under operator_e2e it withholds the
  // mechanism claim while retaining the measured operator result.
  attribution: obj({
    changed_us: { type: 'number' },
    replaced_sum_us: { type: 'number' },
    guard: { type: 'string' },
    // e2e minus the sum of all kernel times, both arms. Where a gap-win hides.
    residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' },
    method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
    note: { type: 'string' },
  }, []),
}, ['status', 'verified_geomean', 'touched_files']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  // De-opaqued for the same reason the other five were: the script reads `patch_file`, `geomean`,
  // `weighted`, `arithmetic` and `per_case` out of this object, and under additionalProperties a
  // renamed or dropped field validated fine and then read as a default — a missing `patch_file`
  // makes the commit step apply nothing, a missing `geomean` scores the integration at 0 and the
  // merge silently loses. `touched_files` is the integrated patch's own footprint; when it is
  // absent the script does NOT assume the integration touched nothing (that would be the
  // maximally-combinable default again) but falls back to the union of every patch that was on the
  // table, which over-ages the shelf rather than offering a patch that no longer applies.
  best: obj({
    patch_file: { type: 'string' }, geomean: { type: 'number' },
    weighted: { type: ['number', 'null'] }, arithmetic: { type: 'number' },
    per_case: perCase, touched_files: { type: 'array', items: { type: 'string' } },
    // Which PATCHES / SHELF_PATCHES ids this combination actually contains. The role already
    // emitted this key; it was simply never declared, so a rename would have validated. Reported
    // for the log and for sizing K — the shelf's aging does NOT depend on it, precisely so that an
    // omitted list cannot leave a superseded candidate on offer.
    patches: { type: 'array', items: { type: 'string' } },
  }, []),
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  // Declared, because memoryMerge keys rows by `id` and dedupes on it, and the ladder is read back
  // out of `roadmap_rung`. A row missing `id` merges as a new row every round; a row missing
  // `verdict` is treated by the next plan_round as untried. Both read as normal ledger growth.
  ledger: { type: 'array', items: obj({
    id: { type: 'string' }, title: { type: 'string' }, specialty: { type: 'string' },
    // confirmed | partial | unresolved | dead_end — see tech_lead.md "Verdict by SPEC, not by
    // result". `unresolved` is what keeps a rung re-openable instead of closing everything above it.
    verdict: { type: 'string', enum: ['confirmed', 'partial', 'unresolved', 'dead_end'] },
    roadmap_rung: { type: ['string', 'null'] },
    expected: { type: 'number' }, actual: { type: 'number' },
    lesson: { type: 'string' },
    first_round: { type: 'number' }, last_round: { type: 'number' },
  }, ['id', 'verdict']) },
  // DURABLE FACTS ABOUT THE HARDWARE AND TOOLCHAIN, as opposed to insights about this kernel.
  // The two are different and the workflow had a home for only one of them. An insight ("the 512
  // guard is bimodal") is about the operator under study and dies with the wave that found it, which
  // is correct. A lowering ("the workgroup barrier does not drain vmcnt on this target", "the
  // profiler reports VGPR in units of two") is true of every kernel that will ever be built on this
  // box, and dying with the wave means the next wave pays a lease to rediscover it. That has already
  // happened: an engineer read four such facts out of emitted ISA in a round that never got a GPU,
  // wrote them into a worker_result, and nothing carried them anywhere. `knowledge/` is read-only to
  // this script by design — an unvalidated claim auto-appended to a card becomes doctrine — so this
  // field does not write a card. It surfaces the candidates, with their evidence, in the log and in
  // the report, where a human merges them. The bar is deliberately high; see tech_lead.md.
  knowledge_delta: { type: 'array', items: obj({
    fact: { type: 'string' },
    // Which card it belongs in, so the merge is a placement decision and not a search.
    card: { type: 'string' },
    // The artifact that establishes it: an ISA dump path, a counter, a sha256, a measured number.
    // A fact whose evidence is "the engineer said so" is an insight, not a knowledge delta.
    evidence: { type: 'string' },
    // What a reader would have believed WITHOUT this fact. Required, because a lowering only earns
    // a card when the source-level reading of it says something else — and when the wrong reading
    // is the benign one, that is the whole reason it costs a wave to find.
    contradicts: { type: 'string' },
    // true = holds for any kernel on this target; false = this operator's shape. Only the first
    // kind is a card edit; the second belongs in the report and nowhere else.
    generalizes: { type: 'boolean' },
  }, ['fact', 'evidence', 'generalizes']) },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
  head_before: { type: 'string' }, head_after: { type: 'string' },
  exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
}, ['committed']);

const REPORT_SCHEMA = obj({
  final_speedup_geomean: { type: 'number' }, final_speedup_arithmetic: { type: 'number' },
  final_speedup_weighted: { type: 'number' }, // time-weighted ratio-of-sums (PRIMARY when workload_aligned)
  rounds: { type: 'number' }, budget_used: { type: 'number' },
  report_path: { type: 'string' }, final_patch: { type: 'string' }, per_case: perCase,
}, ['final_speedup_geomean', 'report_path', 'final_patch']);

const VALIDATE_SCHEMA = obj({
  kernel_name: { type: 'string' },
  director_verified_speedup_geomean: { type: 'number' },
  director_verified_speedup_arithmetic: { type: 'number' },
  director_verified_speedup_weighted: { type: 'number' }, // PRIMARY when workload_aligned
  tech_lead_reported_speedup_geomean: { type: 'number' },
  validation_status: { type: 'string' }, correctness: { type: 'string' },
  autonomy_acceptance_confirmed: { type: 'boolean' },
  workflow_unchanged: { type: 'boolean' },
  workflow_revision_end: { type: 'string' },
  attribution: obj({
    changed_us: { type: 'number' }, replaced_sum_us: { type: 'number' },
    guard: { type: 'string' }, residual_ms_base: { type: 'number' },
    residual_ms_cand: { type: 'number' }, method: { type: 'string' },
    absolute_to_frozen: { type: 'boolean' },
  }, []),
  per_case: perCase, applied_to_original: { type: 'string' },
  // Count of GPU-lease jobs still alive for this EVAL_DIR when validation finished. Non-zero means
  // some direction backgrounded a lease and produced a measurement that is NOT in the report.
  orphan_leases_swept: { type: 'number' },
  arbitration_note: { type: 'string' }, final_patch: { type: 'string' },
}, ['director_verified_speedup_geomean', 'validation_status']);

// ---------------------------------------------------------------------------
// Prompt helpers. Every agent reads its role file from WORKFLOW_DIR and the
// relevant knowledge files itself; the script only passes paths + JSON inputs.
// ---------------------------------------------------------------------------
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

// --- Hung-agent guard ------------------------------------------------------
// An agent LLM call that HANGS (no response, no terminal error) blocks a
// parallel()/pipeline() round-barrier forever (observed: engineer agents frozen
// mid-turn wedged the whole optimize round for >30min). The harness resolves
// terminal API errors to null but NOT an indefinite hang. So bound every agent()
// call: if it has not returned after AGENT_TIMEOUT_MS, resolve it to null (which
// every .filter(Boolean)/null-check downstream already tolerates) and let the
// round proceed. VERY generous default (60min): a true hang never returns, so this only fires on a
// hang, NEVER on a legitimately-long agent. Inner agents include benchmark/profile/verify that build
// (hipcc/ninja) and run benches — minutes, well under 60min — plus the LLM-heavy optimize engineers
// (the ones observed hanging). A too-short bound would kill legit long agents (e.g. a slow rocprof or
// build), so keep it large. Cache keys (prompt, opts) are unchanged so resume still works. Falls back
// to raw agent() if setTimeout is unavailable. args.agent_timeout_ms=0 disables.
// API-FAULT TOLERANCE: a transient API failure (gateway 4xx/5xx, rate-limit, dropped connection, the
// model API going down mid-run) must NOT crash the whole workflow. agentT retries the call up to
// AGENT_RETRIES times on a thrown API/agent error, then resolves to null (every .filter(Boolean)/
// null-check downstream — incl. the Director validate + final report — already degrades on null rather
// than exiting). A timeout (hang) resolves null immediately and is NOT retried (a real hang would just
// burn another full timeout window). args.agent_retries tunes the count. If the failure is PERSISTENT
// (e.g. an auth/header requirement the client doesn't send), retries are exhausted then the run
// degrades — re-run with Workflow({resumeFromRunId}) once the client/API is fixed; cached agent results
// make resume cheap.
const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 3600000, 10);
const AGENT_RETRIES = Math.max(1, parseInt(A.agent_retries != null ? A.agent_retries : 4, 10));
async function agentT(p, o) {
  const label = (o && o.label) ? o.label : 'agent';
  for (let attempt = 1; attempt <= AGENT_RETRIES; attempt++) {
    try {
      if (typeof setTimeout !== 'function' || !(AGENT_TIMEOUT_MS > 0)) return await agent(p, o);
      let to;
      const guard = new Promise((resolve) => {
        to = setTimeout(() => {
          log(`  [hung-agent guard] ${label} exceeded ${Math.round(AGENT_TIMEOUT_MS / 60000)}min with no return — resolving null so the round proceeds.`);
          resolve(null);
        }, AGENT_TIMEOUT_MS);
      });
      // A timeout resolves null (returned as-is, no retry). An API/agent error rejects -> caught below.
      return await Promise.race([
        agent(p, o).then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
        guard,
      ]);
    } catch (e) {
      const msg = String(e && e.message ? e.message : e).slice(0, 200);
      if (attempt < AGENT_RETRIES) {
        log(`  [api-fault guard] ${label} attempt ${attempt}/${AGENT_RETRIES} hit an API/agent error (${msg}) — retrying so a transient outage doesn't kill the run.`);
        continue;
      }
      log(`  [api-fault guard] ${label} still failing after ${AGENT_RETRIES} attempts (${msg}) — resolving null so the workflow degrades gracefully instead of exiting.`);
      return null;
    }
  }
  return null;
}

// Expert-skills injection. PURELY ADDITIVE: '' when OFF or the role is not a skills consumer, so
// roleAgent is byte-identical to the pre-feature build in those cases. When ON, appends an advisory
// pointer telling the agent to Read the fragment + query the skills index (scripts have no fs access).
function expertSkillsBlock(role) {
  if (!USE_EXPERT_SKILLS || !EXPERT_SKILL_ROLES.has(role) || !EXPERT_SKILLS_DIR) return '';
  return `\n\n## Expert skills (ADVISORY — opt-in, enabled this run)\n` +
    `Also Read ${WORKFLOW_DIR}/roles/_fragments/expert_skills.md and follow it: query ` +
    `${EXPERT_SKILLS_DIR}/index.yaml for skills whose \`match\` fits this op (operator/dtype/regime, and ` +
    `from_backend->to_backend for migration skills) and whose validation_status is \`validated\`, and ` +
    `treat each as a HIGH-PRIOR candidate to reproduce — advisory only, never overriding your isolated ` +
    `A/B vs the oracle, never reducing a result below the measured baseline.` +
    // A skill is written for production, so it rightly cites its own reference implementation by
    // branch and commit — that is what makes it reproducible. Under capability_eval those citations
    // become a map to the answer, and the skill is read off disk by the agent, so the orchestrator
    // cannot redact them. Say what is usable instead. `reproduce` above means re-derive, not fetch.
    (CAPABILITY_EVAL
      ? ` **Capability eval: use a skill's MECHANISM and MEASUREMENTS, never its addresses.** Its ` +
        `Sources/Procedure sections may cite a branch, commit, or sibling checkout containing the ` +
        `finished implementation. Do not read, port, copy, or diff against any of them — write the ` +
        `code yourself from the described mechanism. A patch containing a file byte-identical to ` +
        `such a reference is rejected at verify as \`plagiarized\`, whatever it measures.`
      : '');
}

function analysisSkillBlock(role, phase) {
  if (!ANALYSIS_SKILL_ON) return '';
  if (role === 'analysis_engineer') {
    return `\n\n## Profile-analysis skill (ANALYSIS ONLY — explicitly enabled this run)\n` +
      `Read ${ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR}/SKILL.md, execute only its checked-in ` +
      `builder/runner/analyzer/validator chain, and return the validated artifact contract. ` +
      `Never improvise arithmetic or change the generic Profile result.`;
  }
  if (role === 'tech_lead' && phase === 'plan_round') {
    return `\n\n## Profile-analysis evidence (explicitly enabled this run)\n` +
      `Inspect PROFILE_SUMMARY.analysis_result. Only Read analysis_json when status=ready and ` +
      `analysis_schema_version is the Skill's current schema. If analysis_status=awaiting_measurement, ` +
      `use findings only at their declared confidence; do not treat hypotheses, bounds, or claims as ` +
      `root cause and do not rank a direction from them. If evidence_complete, use only resolved ` +
      `high/medium-confidence evidence alongside generic profile/per-case data. A degraded or unavailable ` +
      `analysis contributes nothing; continue from the generic Profile result.`;
  }
  return '';
}

function roleAgent(role, phase, intro, inputs) {
  const base = `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}

## Inputs
${cfg(inputs)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
  return base + expertSkillsBlock(role) + analysisSkillBlock(role, phase);
}

// ===========================================================================
// PHASE: Setup
// ===========================================================================
phase('Setup');
const setup = await agentT(
  roleAgent('director', 'setup', 'Build the isolated evaluation environment.', {
    KERNEL_PATH_ORIG, EXP_ROOT, EVAL_DIR_OVERRIDE, KERNEL_NAME_HINT, TASK, SKILL_DIR: WORKFLOW_DIR,
    MODE, TARGET_LANGUAGE, OP_SPEC,
    ...(STRICT_AUTONOMY ? { STRICT_AUTONOMY: '1' } : {}),
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);
if (STRICT_AUTONOMY && setup.resumed) {
  throw new Error(
    'STRICT AUTONOMY REFUSED: director seeded this run from STATE_DIR/best. A proof run must start ' +
    'from the frozen input tree with an empty/fresh state directory; otherwise inherited code and ' +
    'manual state edits cannot be distinguished from work derived in this run.');
}
if (STRICT_AUTONOMY && (setup.workflow_clean !== true ||
    !String(setup.workflow_revision || '').trim())) {
  throw new Error(
    'STRICT AUTONOMY REFUSED: GEAK workflow/knowledge must be a clean, committed revision before ' +
    'launch. Mid-wave or uncommitted instructions make the run non-reproducible.');
}
const WORKFLOW_REVISION_START = String(setup.workflow_revision || '');
if (STRICT_AUTONOMY) {
  const roots = new Set(Array.isArray(CONTAINMENT_PREFLIGHT.roots)
    ? CONTAINMENT_PREFLIGHT.roots.map((r) => String(r).replace(/\/+$/, '')) : []);
  const expectedRoots = [KERNEL_PATH_ORIG, EXP_ROOT, STATE_DIR, WORKFLOW_DIR]
    .filter(Boolean).map((r) => String(r).replace(/\/+$/, ''));
  const missingRoots = expectedRoots.filter((r) => !roots.has(r));
  if (CONTAINMENT_PREFLIGHT.content_scan !== 'clean' ||
      CONTAINMENT_PREFLIGHT.skill_address_scan !== 'clean' ||
      String(CONTAINMENT_PREFLIGHT.workflow_revision || '') !== WORKFLOW_REVISION_START ||
      Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0) !== KNOWN_REFERENCE_HASHES.length ||
      missingRoots.length) {
    throw new Error(
      `STRICT AUTONOMY REFUSED: containment attestation is not bound to this workflow/roots ` +
      `(missing roots: ${missingRoots.join(', ') || 'none'}).`);
  }
}

// CONTAINMENT ENFORCEMENT — the startup check above is a path filter and it missed the two leaks that
// actually happened. Both were CONTENT: a control workspace holding the applied reference patch left
// inside the tree, and a knowledge card that opened with the reference branch and commit. Neither is
// a known_reference_path, so neither could ever have failed the startup check. This gate runs the two
// scanners that do see them, and it runs AFTER Setup because Setup is what materialises the eval dir,
// the baseline copy and the inherited assets — sweeping before them scans an empty tree and passes.
// It is an agent because workflow scripts have no fs/child_process: the check has to be executed by
// something with a shell, and one cheap agent beats a comment asking a human to remember.
if (STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true) {
  log(`CONTAINMENT PREFLIGHT clean: trusted bootstrap scanned the run tree before agent startup ` +
      `(marker manifest ${String(CONTAINMENT_PREFLIGHT.marker_manifest_sha256 || '').slice(0, 12)}…, ` +
      `${Number(CONTAINMENT_PREFLIGHT.reference_hash_count || 0)} opaque reference hashes).`);
}
if (CAPABILITY_EVAL && RUN_TREE_ANCESTOR &&
    !(STRICT_AUTONOMY && CONTAINMENT_PREFLIGHT.clean === true)) {
  const gate = await agentT(
    `Run two containment scanners and report their exit codes verbatim. Do not fix anything, do not ` +
    `read any file they flag, do not summarise the flagged content — reporting the paths is the whole job.\n\n` +
    // REF_SCAN_MAX_TREES is raised from its default of 200 on purpose. At the default the sweep
    // prints "NOTE n repo(s) had more unique ref trees than REF_SCAN_MAX_TREES=200; the remainder
    // were NOT scanned", and its hit count is then a LOWER BOUND rather than a count. A gate whose
    // clean verdict means "clean in the part I looked at" is the exact shape of the two leaks this
    // gate exists to catch, both of which were reachable and both of which read as clean at the
    // time. The full sweep costs roughly five minutes once, before any lease is taken.
    `1. REF_SCAN_MAX_TREES=100000 bash ${WORKFLOW_DIR}/scripts/reference_leak_sweep.sh --tree ${RUN_TREE_ANCESTOR}\n` +
    `   If its output still contains a REF_SCAN_MAX_TREES coverage NOTE, the scan is PARTIAL: say so\n` +
    `   in your note and return verdict "unknown" rather than clean, whatever the exit code was.\n` +
    `2. bash ${WORKFLOW_DIR}/scripts/skill_address_scan.sh --skills-dir ${EXPERT_SKILLS_DIR || '(skip: expert skills off)'} ` +
    `--scan-root ${RUN_TREE_ANCESTOR}\n\n` +
    `If a script is missing or its --skills-dir is empty, record that step as "skipped" with the reason ` +
    `and do NOT call it clean. "The scanner did not run" and "the scanner found nothing" are different ` +
    `results and this run has already been damaged twice by conflating them.`,
    { phase: 'Setup', label: 'containment gate', schema: CONTAINMENT_GATE_SCHEMA });
  if (!gate || gate.verdict !== 'clean') {
    const verdict = gate ? gate.verdict : 'unknown';
    if (STRICT_AUTONOMY || verdict === 'leak') {
      throw new Error(
        `CONTAINMENT GATE FAILED (${verdict}): strict/capability evidence requires an explicit clean ` +
        `scan of ${RUN_TREE_ANCESTOR} before budget is spent.\n` +
        `${gate ? (gate.findings || []).join('\n') : 'gate agent returned nothing'}\n` +
        `${gate ? gate.note || '' : ''}`);
    }
    log(`CONTAINMENT GATE ${verdict}: scanner did not establish clean. ${gate ? gate.note || '' : ''}`);
  } else log(`CONTAINMENT GATE clean: ${gate.note || ''}`);
}

// ---------------------------------------------------------------------------
// POOL SAMPLE — read the hardware before planning a round that assumes it.
//
// tech_lead.md rule 3c tells the lead to sample the pool before budgeting a GPU direction. That rule
// is prose, and prose is advisory: on 2026-08-22 a whole wave (3 rounds, 20 agents, 1.95M tokens)
// planned every round as if 8 idle cards were waiting, while an external tenant held ~283 GB on all
// eight for the wave's entire life. Every lease request sat inside flock, one direction spent 95
// minutes there, and NOT ONE ARM WAS MEASURED. Nothing in any artifact showed it except sysfs, and
// the run reported final_speedup:1 as though the ideas had been tried.
//
// So the script takes the sample itself and hands the lead a fact rather than an instruction. It does
// NOT abort: that same wave produced its single most valuable finding (a JIT cache-key defect that
// voided an earlier closure) entirely without hardware. An occupied pool should redirect a round to
// lease-free work, not end the run. Sampling per round rather than once is deliberate — the tenant
// observed here cycles between ~287 GiB and ~25 GiB free inside two minutes.
async function samplePool(round) {
  if (!(GPU_MIN_FREE_GIB > 0)) return null;
  const p = await agentT(
    `Sample the GPU pool and report what you see. Do NOT acquire a lease, do NOT run any workload, ` +
    `do NOT try to free anything. Reading is the whole job.\n\n` +
    `1. Per-card free VRAM: read /sys/class/drm/card*/device/mem_info_vram_{total,used} (or ` +
    `\`rocm-smi --showmeminfo vram --csv\`) and compute free = total - used in GiB for every card.\n` +
    `2. Per-card utilisation: /sys/class/drm/card*/device/gpu_busy_percent.\n` +
    `3. \`rocm-smi --showpids\`. For each KFD PID it lists, check whether that PID exists in your own ` +
    `/proc. A PID that does NOT is in a FOREIGN NAMESPACE — that is an external tenant, not a stale ` +
    `lease of ours, and the two have opposite remedies. List those PIDs in foreign_pids.\n\n` +
    `verdict:"free" only if EVERY card has at least ${GPU_MIN_FREE_GIB} GiB free. If any card is ` +
    `below that floor, or a foreign tenant holds memory, verdict:"occupied". If you cannot read the ` +
    `pool at all, verdict:"unknown" — do not report an unreadable pool as free.`,
    { phase: 'Optimize', label: `pool sample r${round}`, schema: POOL_SCHEMA });
  if (!p) {
    log(`POOL SAMPLE r${round}: sampler returned nothing. Treating as UNKNOWN, not free.`);
    return { verdict: 'unknown', note: 'sampler agent returned nothing' };
  }
  const detail = `${p.cards_free != null ? `${p.cards_free}/${p.cards_total} cards free, ` : ''}` +
    `min ${p.min_free_gib != null ? p.min_free_gib : '?'} GiB vs a ${GPU_MIN_FREE_GIB} GiB floor` +
    `${(p.foreign_pids || []).length ? `, foreign PIDs ${p.foreign_pids.join(',')}` : ''}`;
  if (p.verdict === 'free') {
    log(`POOL SAMPLE r${round}: FREE — ${detail}.`);
  } else {
    log(`POOL ${p.verdict.toUpperCase()} r${round}: ${detail}. ${p.note || ''}\n` +
        `  Plan this round GPU-LESS. A GPU direction dispatched now will spend its clock inside flock ` +
        `and return nothing; an instrument authored now, with its driver, is a partial the next round ` +
        `can run in one lease.`);
  }
  return p;
}

async function runProfileAnalysis(profileSummary, round, label) {
  if (!ANALYSIS_SKILL_ON || !profileSummary) return null;
  const result = await agentT(
    roleAgent(
      'analysis_engineer',
      'analyze_profile',
      'Execute and validate the optional Step-2 analysis pipeline.',
      {
        WORKSPACE: CANONICAL,
        EVAL_DIR,
        ROUND: round,
        COMMANDMENT,
        PROFILE_SUMMARY: profileSummary,
        SKILL_DIR: WORKFLOW_DIR,
        ...ANALYSIS_SKILL_INPUTS,
      },
    ),
    {
      phase: 'Profile',
      label,
      schema: ANALYSIS_RESULT_SCHEMA,
    },
  );
  return result || {
    status: 'degraded',
    analysis_skill: ANALYSIS_SKILL,
    analysis_schema_version: '',
    analysis_status: 'unavailable',
    analysis_json: '',
    failure_note: 'analysis agent failed or returned invalid structured output',
  };
}

// ---------------------------------------------------------------------------
// Enforce a FROZEN REAL-ONLINE BASELINE in BOTH modes (author AND same-language
// optimize). The immutable unittest times + parity-checks the candidate against
// baseline_src/ / meta.baseline_callable (the live online kernel); if neither
// exists it would silently fall back to timing kernel_src/ against itself — the
// "optimized-HIP vs naive-HIP = fake 15.7×" bug this harness exists to prevent.
// The script has no FS access, so we trust the director's structured verdict
// (it copied baseline_src/ + confirmed the callable). Missing -> abort/re-extract.
// ---------------------------------------------------------------------------
const hasBaseline = setup.baseline_frozen === true ||
  (typeof setup.baseline_callable === 'string' && setup.baseline_callable.trim().length > 0);
if (!hasBaseline) {
  const reason = `no frozen baseline (baseline_src/ or meta.baseline_callable) for ${KERNEL_NAME} — ` +
    `re-extract; refusing to time the candidate against kernel_src/ (fake-win risk)`;
  log(`Setup ABORT: ${reason}`);
  return {
    mode: MODE, authored: false, target_language: TARGET_LANGUAGE,
    eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
    final_geomean: 0, final_patch: '', validation_status: 'no_baseline', reason,
  };
}

// ===========================================================================
// PHASE: Author (mode=author only) — write a fresh from-scratch impl as the
// optimize loop's CODE SEED. On success, HEAD of CANONICAL becomes that seed
// (what the optimize loop diffs its edits against) and the rest of the pipeline
// (Analyze/Benchmark/Profile/optimize loop) runs UNCHANGED on it. The SPEEDUP
// denominator is NEVER the seed — it is the frozen REAL ONLINE kernel in
// baseline_src/ (meta.baseline_callable), regardless of TARGET_LANGUAGE.
// On failure (no correct seed), abort early with a structured result so the
// e2e caller drops this language.
// ===========================================================================
if (MODE === 'author') {
  phase('Author');
  const authored = await agentT(
    roleAgent('author_engineer', 'author', 'Write the simplest correct baseline in the target language.', {
      TARGET_LANGUAGE, OP_SPEC, WORKSPACE: CANONICAL, TASK_DIR: KERNEL_PATH_ORIG,
      GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, KERNEL_KNOWLEDGE_DIR,
      GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    }),
    { phase: 'Author', label: `author:${TARGET_LANGUAGE}`, schema: AUTHOR_SCHEMA });
  if (!authored || !authored.authored || authored.correctness !== 'pass') {
    log(`Author mode FAILED for ${TARGET_LANGUAGE}: ${authored ? authored.notes || authored.correctness : 'no result'}. Aborting (no seed to optimize).`);
    return {
      mode: 'author', authored: false, target_language: TARGET_LANGUAGE,
      eval_dir: EVAL_DIR, kernel_name: KERNEL_NAME,
      final_geomean: 0, final_patch: '', validation_status: 'author_failed',
      reason: authored ? authored.notes || 'author produced no correct baseline' : 'author returned nothing',
    };
  }
  log(`Author mode: ${TARGET_LANGUAGE} seed written (correct, seed ${authored.baseline_ms || '?'} ms; denominator = frozen online kernel). Optimizing it now.`);
}

// ===========================================================================
// PHASE: Analyze + Roadmap (TechLead)
// ===========================================================================
phase('Analyze');
// The two analyze call sites below repeat their inputs verbatim instead of sharing one object, and
// that is deliberate: tests/test_input_contract.js reads THIS FILE and checks that every input a
// role declares is actually threaded to it. A spread of a named object hides the inputs from that
// check, which then reports the role as under-supplied. Factoring these eight lines out would trade
// a real static guard for a cosmetic saving. If you add an input, add it in both places.
let analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR,
    // Authoritative resolved rank count (from gpus_per_job | op_spec.resource | job_gpu_ids).
    // >1 is what makes the `distributed` specialty eligible; OP_SPEC.resource may be absent.
    GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
    ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
    ...(STRICT_AUTONOMY ? {
      STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
      PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
      REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    } : {}),
    // The resume flag says "a prior wave already built the roadmap"; STATE_DIR is the only place
    // that roadmap and the open-rung list actually survive between waves, because EVAL_DIR is
    // rebuilt from scratch by bootstrap_task.sh. roles/tech_lead.md's fast path already instructs
    // analyze to read `STATE_DIR` and `STATE.json` — until now it was never given either, so the
    // instruction could not be followed and the fast path had nothing to resume from.
    ...(STATE_DIR ? { STATE_DIR } : {}),
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });

// <<REPLAY:analyze_resume_fallback>>
// THE RESUMED WAVE THAT ANALYSED NOTHING.
//
// roles/tech_lead.md gives analyze a fast path when INCREMENTAL_RESUME is set: read the roadmap a
// prior wave persisted instead of re-deriving it, and "do a full analysis only if no prior roadmap
// exists". That last clause is the whole safety of the fast path, and it is unenforceable from here
// -- workflow scripts have no filesystem, so this file cannot see whether EVAL_DIR/roadmap.md was
// found or whether the phase quietly returned an empty shell.
//
// Wave 15 is what that costs. bootstrap_task.sh assembles a FRESH EVAL_DIR, so the prior wave's
// roadmap was not in it; analyze took the fast path, found nothing to read, and returned a valid
// schema with no candidate_directions and no task_graph. Three rounds then ran with an empty ladder.
// The engineers carried the D0..D3 rung ids forward from wave 14 by hand, out of their own memory,
// and D2 went unspent for three waves because nothing on disk was tracking that it was owed. Round
// 3's engineer eventually re-materialised roadmap.md himself, at 01:08, unprompted.
//
// The LADDER MISSING caveat below did fire -- every round, unchanged, changing nothing. A warning
// that repeats and is never acted on is not a guard, it is a log line. So act on it here: an empty
// ladder out of a RESUMED analyze is not a fact about the kernel, it is the fast path failing to
// find its input, and the remedy is the one the role file already prescribes. Re-run once with the
// resume flag off. One extra analyze call against three wasted rounds.
//
// Deliberately narrow: only when INCREMENTAL was on, only on an empty ladder (a resume with no
// ladder is a contradiction in terms -- the ladder IS what is being resumed), and only once.
function analyzeResumeDegenerate(incremental, ver) {
  if (!incremental) return { retry: false, reason: '' };
  const rungs = (ver && Array.isArray(ver.candidate_directions) ? ver.candidate_directions : [])
    .filter((c) => c && (c.id || c.title));
  if (rungs.length) return { retry: false, reason: '' };
  return { retry: true, reason:
    'ANALYZE RESUME DEGENERATE: the fast path ran with INCREMENTAL_RESUME and returned no ' +
    'candidate_directions. A resumed wave with no ladder has nothing to resume, which means the ' +
    'prior roadmap was not reachable from EVAL_DIR rather than that the ladder is empty. ' +
    'Re-running analyze once WITHOUT the resume flag, per roles/tech_lead.md ("do a full analysis ' +
    'only if no prior roadmap exists").' };
}
// <</REPLAY:analyze_resume_fallback>>

{
  const d = analyzeResumeDegenerate(INCREMENTAL, analysis);
  if (d.retry) {
    log(d.reason);
    // Identical to the call above except that RESUME_INPUT is absent — that omission IS the fix.
    const full = await agentT(
      roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
        WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
        KERNEL_KNOWLEDGE_DIR,
        GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
        ...(A.require_task_graph ? { REQUIRE_TASK_GRAPH: '1' } : {}),
        ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
        ...(STRICT_AUTONOMY ? {
          STRICT_AUTONOMY: '1', TARGET_GUARDS, REGRESSION_GUARDS,
          PROMOTION_METRIC, LAUNCH_TARGET: Number(A.launch_target || 2),
          REQUIRE_OVERLAP, REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        } : {}),
        // Kept on the recovery call too: the full analysis re-derives the ladder from the source,
        // but the rungs a prior wave already spent, and the ones it left owed, exist only here.
        ...(STATE_DIR ? { STATE_DIR } : {}),
      }),
      { phase: 'Analyze', label: 'tech_lead:analyze:full', schema: ANALYZE_SCHEMA });
    const got = (full && Array.isArray(full.candidate_directions) ? full.candidate_directions : [])
      .filter((c) => c && (c.id || c.title));
    if (got.length) {
      log(`ANALYZE RE-RUN recovered a ladder of ${got.length} rung(s). Using the full analysis.`);
      analysis = full;
    } else {
      log('ANALYZE RE-RUN also returned no ladder. This run genuinely has no recorded ordering; ' +
          'the LADDER MISSING caveat below is a real finding, not a resume artifact.');
      if (full) analysis = full;
    }
  }
}
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

// ---------------------------------------------------------------------------------------------
// TASK-GRAPH GATE (args.require_task_graph)
//
// Opt-in, and off by default: most kernels have no interesting dependency graph and requiring one
// everywhere would produce filled-in forms, which are worse than nothing because they look like
// evidence. Turn it on for multi-stage / multi-rank operators where the whole question is which
// orderings are real.
//
// Why a GATE and not a nudge. The failure this catches is not laziness, it is that the roadmap
// prose and the roadmap artifact can disagree without anyone noticing: a run can assert "the stages
// are serialized and must be fused" in `roadmap_summary`, have that assertion accepted by every
// downstream phase, and never once have enumerated an edge. Asking for the object makes the claim
// checkable. If the graph cannot be built, THAT is the analysis result and it should be said out
// loud, not routed around.
//
// Deliberately NOT a hard abort. An analysis that returns a partial graph plus honest `unknowns`
// is more useful than a retry loop that eventually returns a complete-looking invention, and this
// gate must not create pressure toward the second. It reports; the caveat travels to the report so
// a reader can discount every downstream ranking accordingly.
// The decision is a pure function of the returned graph, and it is lifted VERBATIM by
// tests/test_task_graph_gate.js between these markers — so the test exercises the shipped code
// rather than a paraphrase of it that can drift. Keep it pure: no log(), no closure over run state.
// <<REPLAY:task_graph_gate>>
function taskGraphGate(tg) {
  const nodes = tg && Array.isArray(tg.nodes) ? tg.nodes.length : 0;
  const edges = tg && Array.isArray(tg.edges) ? tg.edges.length : 0;
  if (!tg || (!nodes && !edges)) {
    return { verdict: 'MISSING', summary: '', caveat:
      'TASK GRAPH MISSING: this run required a tile-level dependency graph from Analyze and none ' +
      'was returned. Every direction ranked below is therefore asserted, not derived — there is no ' +
      'record of which orderings are required by the data and which are artifacts of the current ' +
      'code, no critical path to bound what any change can win, and no slack to say which nodes ' +
      'are worth touching. Read the rankings as hypotheses.' };
  }
  // A graph with no `launch_boundary` edge is a real and reportable finding (nothing to unfuse);
  // a graph where EVERY edge is one is almost always a graph built at kernel granularity, where the
  // column was never really filled in. Both are visible only if the count is printed, so print it.
  const eArr = Array.isArray(tg.edges) ? tg.edges : [];
  const launchEdges = eArr.filter((e) => e && e.enforced_by === 'launch_boundary').length;
  const unknowns = Array.isArray(tg.unknowns) ? tg.unknowns.length : 0;
  const assumed = (Array.isArray(tg.nodes) ? tg.nodes : [])
    .filter((n) => n && n.source === 'assumed').length;
  const cp = Number(tg.critical_path_us);
  const e2e = Number(tg.measured_e2e_us);
  const quantified = Number.isFinite(cp) && Number.isFinite(e2e) && e2e > 0;
  const summary = `Task graph: ${nodes} nodes, ${edges} edges, ${launchEdges} enforced only by a ` +
    `launch boundary, ${unknowns} declared unknown, ${assumed} node duration(s) assumed. ` +
    (quantified
      // NOT "addressable ceiling". 1 - cp/e2e is the share of e2e that the DEPENDENCE CHAIN does not
      // explain, and calling that a ceiling on gains is wrong in the common case: on a throughput-bound
      // operator the critical path is short precisely because the binding floor is arithmetic, not
      // ordering, and no amount of overlap deletes arithmetic. The 2026-08-23 run made this concrete --
      // it reported cp=130µs against e2e=4580.8µs and then had to spend a prose note correcting the
      // implication, because ~97% of that gap was ~6.5 TFLOP of GEMM per rank. A gate that has to be
      // talked out of its own headline number is miscalibrated; say what the quantity IS and let the
      // graph's bubble set carry the claim about what is actually recoverable.
      ? `critical_path=${cp.toFixed(1)}µs vs e2e=${e2e.toFixed(1)}µs ` +
        `(dependence explains ${(100 * (cp / e2e)).toFixed(1)}% of e2e; the remaining ` +
        `${(100 * (1 - cp / e2e)).toFixed(1)}% is resource-bound work plus bubbles, and only the ` +
        'bubble set is addressable by reordering)'
      : 'critical path NOT quantified');

  if (quantified && cp > e2e) {
    return { verdict: 'INCONSISTENT', summary, caveat:
      `TASK GRAPH INCONSISTENT: critical_path_us (${cp}) exceeds measured_e2e_us (${e2e}). The ` +
      'longest path through the graph cannot be longer than the thing it is a path through, so ' +
      'either a node duration is wrong or an edge is spurious. Do not use this graph\'s headroom ' +
      'number to size any proposal.' };
  }
  // Every edge a launch boundary AND no edge at a narrower scope = the kernel-granularity tell.
  // Not fatal (a genuinely four-launch operator can look like this), but the whole point of the
  // artifact is the scope column, and a graph without it cannot distinguish "must be a barrier"
  // from "happens to be a barrier" — which is the only question the graph was built to answer.
  const narrow = eArr.filter((e) => e && e.scope && e.scope !== 'cross_rank' && e.scope !== 'hbm').length;
  if (edges && launchEdges === edges && narrow === 0) {
    return { verdict: 'KERNEL_GRANULARITY', summary, caveat:
      `TASK GRAPH IS KERNEL-GRANULAR: all ${edges} edges came back enforced_by=launch_boundary with ` +
      'no edge at a narrower scope, which is what a graph whose nodes are KERNELS looks like. Such ' +
      'a graph restates the launch order and cannot separate an ordering the data requires from one ' +
      'the current code imposes. Treat any fusion direction ranked from it as unsupported.' };
  }
  // WHAT THE CONSUMER NEEDS THAT THE PRODUCER DOES NOT SUPPLY.
  //
  // The graph's nodes are output tiles, so its edges only ever describe the operand that flows along
  // the edge. Every other operand the consumer needs -- the second weight matrix, its scales, the
  // descriptors, the index arrays -- has no node, therefore no edge, therefore can never surface as
  // an opportunity. That is a blind spot of the representation, not of the analyst: a real graph on
  // this operator listed six nodes and five edges, correctly found both fusable edges, and never
  // mentioned that GEMM2's weights depend on nothing at all and could be pulled into L2 during
  // GEMM1 -- with both stages measured at ~30% HBM, the one lever the table most obviously funds.
  //
  // Asking per edge rather than per node keeps it cheap: the answer is a list of names, it is static
  // (a read of the consumer's signature), and an empty list is a legitimate answer that means
  // something. Advisory: a missing answer makes prefetch unrankable, it does not invalidate the graph.
  const fanInEdges = eArr.filter((e) => e && Number(e.fan_in) > 0);
  const unanswered = fanInEdges
    .filter((e) => !Array.isArray(e.producer_independent_operands))
    .map((e) => `${e.from || '?'}->${e.to || '?'}`);
  if (unanswered.length) {
    return { verdict: 'OK', summary, caveat:
      `EDGES THAT DID NOT SAY WHAT THE CONSUMER LOADS ANYWAY (${unanswered.length}): ` +
      `${unanswered.join(', ')}. Each names an operand the consumer waits for; none names the ` +
      'operands it needs that the producer does not supply — weights, scales, descriptors, index ' +
      'arrays. Those depend on nothing and can be loaded during the producer\'s own execution, ' +
      'which is a different and usually cheaper lever than moving the dependent operand earlier. ' +
      'It cannot be ranked from this graph because it has no node here.' };
  }
  return { verdict: 'OK', summary, caveat: '' };
}
// <</REPLAY:task_graph_gate>>

// <<REPLAY:pipe_occupancy_gate>>
// THE PIPE TABLE, AND WHETHER THE DIRECTIONS WERE PRICED AGAINST IT.
//
// Why this is a separate gate from taskGraphGate rather than more fields on it: they answer
// different questions and they fail independently. The graph can be perfect and the plan still
// worthless, which is exactly what happened. A program spent four waves on occupancy, launch-count
// and communication-overlap directions, every one of them derivable from a correct dependency
// graph, and then collected hardware counters for the first time and found: no pipe above 41%, HBM
// at 7-8%, inter-kernel gap 0.00us over 72 boundaries, and occupancy kill-gated at BOTH ends. Every
// lever it had tried was foreclosed by a measurement nobody had made. The graph was not wrong; it
// was answering "may these overlap", and nothing was answering "is that pipe even busy".
//
// The gate is deliberately cheap to satisfy and expensive to fake: it prints the table, derives the
// class from the numbers rather than trusting the stated one, and names the directions that claimed
// more than their pipe's idle fraction can pay.
const PIPE_SATURATED_PCT = 80;   // at or above this, the pipe is the constraint
const PIPE_LOW_PCT = 50;         // below this for EVERY pipe, nothing is throughput-bound
const GAP_NEGLIGIBLE_US = 0.5;   // inter-kernel gap below this cannot fund a launch-count direction

function pipeOccupancyGate(rt, directions) {
  const dirs = Array.isArray(directions) ? directions : [];
  const pipes = rt && Array.isArray(rt.pipes) ? rt.pipes.filter((p) => p && Number.isFinite(Number(p.utilization_pct))) : [];
  if (!pipes.length) {
    return { verdict: 'MISSING', summary: '', class_derived: null, caveat:
      'PIPE TABLE MISSING: this run required a per-pipe resource timeline from Analyze and none was ' +
      'returned. Nothing below is priced against what the machine was actually doing — there is no ' +
      'record of which functional unit was the constraint, so a direction targeting an already-' +
      'saturated pipe and one filling an idle pipe are indistinguishable here, and the classic ' +
      'reflexes (raise occupancy, fuse launches) cannot be ruled out before they cost a lease. ' +
      'Read every expected_speedup as unbounded above.' };
  }
  const maxUtil = Math.max(...pipes.map((p) => Number(p.utilization_pct)));
  const satur = pipes.filter((p) => Number(p.utilization_pct) >= PIPE_SATURATED_PCT);
  const gap = rt.interkernel_gap_us && Number.isFinite(Number(rt.interkernel_gap_us.median))
    ? Number(rt.interkernel_gap_us.median) : null;
  const opps = Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities.length : 0;
  const closed = Array.isArray(rt.closed_axes) ? rt.closed_axes.length : 0;

  // Derive the class from the numbers. A stated class that the table does not support is the
  // failure this catches: `class` drives which levers are even candidates, so a wrong one is worse
  // than an absent one.
  let derived = 'mixed';
  // One saturated engine does not make every producer/consumer overlap throughput-bound. If MFMA
  // is full while VMEM/interconnect are idle, the operator is mixed and cross-engine work may still
  // fit. Call it throughput-bound only when every measured binding engine is saturated (or only one
  // engine was measured); otherwise retain the vector nature in `mixed`.
  if (satur.length && satur.length === pipes.length) derived = 'throughput_bound';
  else if (maxUtil < PIPE_LOW_PCT && gap !== null && gap > GAP_NEGLIGIBLE_US) derived = 'launch_bound';
  else if (maxUtil < PIPE_LOW_PCT) derived = 'latency_bound';

  const table = pipes
    .map((p) => `${p.stage || '?'}.${p.pipe || '?'}=${Number(p.utilization_pct).toFixed(1)}%`)
    .join(' ');
  const summary = `Pipe table: ${table}; max=${maxUtil.toFixed(1)}% ` +
    `(${satur.length} pipe(s) at/above ${PIPE_SATURATED_PCT}%); ` +
    `inter-kernel gap median=${gap === null ? 'UNMEASURED' : gap.toFixed(3) + 'us'}; ` +
    `class stated=${rt.class || 'none'} derived=${derived}; ` +
    `${opps} idle-pipe opportunit(y/ies), ${closed} closed axis/axes.`;

  const notes = [];
  if (rt.class && rt.class !== derived) {
    notes.push(`CLASS CONTRADICTS THE TABLE: Analyze stated class="${rt.class}" but the numbers it ` +
      `reported derive "${derived}" (max pipe ${maxUtil.toFixed(1)}%, gap ` +
      `${gap === null ? 'unmeasured' : gap.toFixed(3) + 'us'}). The class decides which lever ` +
      'families are candidates at all, so take the derived one and re-read the rankings against it.');
  }
  if (derived === 'latency_bound') {
    notes.push('LATENCY-BOUND: every pipe is below ' + PIPE_LOW_PCT + '% and the launch gap is ' +
      'negligible, so the machine is stalled on dependences, not on any resource. Two standard ' +
      'levers are dead here and should not be given a lease: raising OCCUPANCY (adding waves adds ' +
      'stalled waves when each wave\'s instruction stream has no independent work, and it spends ' +
      'the registers and LDS that software pipelining needs), and fusing to reclaim LAUNCH ' +
      'OVERHEAD (the gap is already zero). What does pay is making dependency-free work of the ' +
      'NEXT stage issuable in this one\'s shadow — a kernel boundary is a grid-wide barrier plus a ' +
      'pipeline drain, so that work is not unscheduled, it is inexpressible.');
    if (!opps) {
      notes.push('NO IDLE-PIPE OPPORTUNITIES LISTED: the table says the pipes are idle and the ' +
        'artifact names nothing that could be issued into them. That is the one question this ' +
        'analysis existed to answer; without it the directions below cannot have been derived from it.');
    }
  }
  // Directions priced above what their own pipe can pay. Checked here rather than trusted, because
  // expected_speedup is the field most likely to be a wish.
  const overclaimed = [];
  const unpriced = [];
  for (const d of dirs) {
    if (!d || d.specialty === 'deep_explore') continue;
    if (!d.fills_pipe) { unpriced.push(d.id || d.title || '?'); continue; }
    const util = Number.isFinite(Number(d.pipe_util_pct)) ? Number(d.pipe_util_pct) : null;
    const exp = Number(d.expected_speedup);
    if (util === null || !Number.isFinite(exp)) continue;
    const ceiling = 1 / Math.max(util / 100, 1e-6); // filling an idle pipe completely, the optimistic bound
    if (exp > ceiling) {
      overclaimed.push(`${d.id || d.title}: claims ${exp.toFixed(3)}x on ${d.fills_pipe} at ` +
        `${util.toFixed(1)}% utilization, whose perfect-fill bound is ${ceiling.toFixed(3)}x`);
    }
  }
  if (unpriced.length) {
    notes.push(`DIRECTIONS WITH NO PIPE NAMED (${unpriced.length}): ${unpriced.join(', ')}. A ` +
      'direction that cannot say which functional unit it fills has not been derived from the ' +
      'table and its expected_speedup is unbounded by anything. Treat as hypotheses.');
  }
  if (overclaimed.length) {
    notes.push('DIRECTION OVERCLAIMS ITS PIPE: ' + overclaimed.join('; ') +
      '. These are arithmetically impossible against the reported utilization and can be rejected ' +
      'without a run; if the utilization is wrong, that is the finding.');
  }
  // A SIZED HOLE THAT NO DIRECTION CLAIMS.
  //
  // `overclaimed` catches a direction that promises more than its pipe can pay. This catches the
  // opposite and more expensive error: a hole that Analyze located, sized in microseconds, and then
  // nobody planned against. It is invisible per-direction, because the evidence of the miss is in a
  // list the per-direction check never reads.
  //
  // Measured cost of not having it: a run reported a Stage2 tail round at 18.75% occupancy, 115us,
  // 2.5% of e2e -- one of only two quantified holes in the whole operator -- and then issued four
  // directions, none of which mentioned it. The hole was assumed to be collected as a side effect of
  // the fusion rung. That rung never ran, so nothing collected it, and the 2.5% stayed on the table
  // for four waves without ever being declined.
  //
  // Only checked once directions exist (the post-Analyze call passes none), and `rides_on` is a
  // legitimate answer: a hole that another rung will absorb is planned for, it just is not planned
  // for separately. What is refused is silence.
  if (dirs.length) {
    const claimed = new Set();
    for (const d of dirs) {
      if (!d) continue;
      if (d.fills_hole) claimed.add(String(d.fills_hole));
      for (const g of Array.isArray(d.graph_refs) ? d.graph_refs : []) claimed.add(String(g));
    }
    const unowned = (Array.isArray(rt.idle_pipe_opportunities) ? rt.idle_pipe_opportunities : [])
      .filter((o) => o && (Number(o.recoverable_us) > 0 || Number(o.pct_of_e2e) > 0))
      .filter((o) => !o.rides_on && !claimed.has(String(o.id)))
      .map((o) => `${o.id || o.window || '?'} (${Number(o.recoverable_us) > 0 ? Number(o.recoverable_us).toFixed(0) + 'us' : ''}` +
        `${Number(o.pct_of_e2e) > 0 ? (Number(o.recoverable_us) > 0 ? ', ' : '') + Number(o.pct_of_e2e).toFixed(1) + '% of e2e' : ''})`);
    if (unowned.length) {
      notes.push(`SIZED HOLE WITH NO DIRECTION (${unowned.length}): ${unowned.join('; ')}. Analyze ` +
        'located these, measured them and put a number on them, and no direction in this round ' +
        'names one as the thing it fills. A hole nobody claims is not a hole nobody can reach — it ' +
        'is a hole nobody looked at. Either give it a direction, or set `rides_on` to the rung that ' +
        'will absorb it so the claim is recorded and can be checked when that rung reports.');
    }
  }
  const verdict = notes.length ? (overclaimed.length || (rt.class && rt.class !== derived) ? 'INCONSISTENT' : 'ADVISORY') : 'OK';
  return { verdict, summary, class_derived: derived, caveat: notes.join(' ') };
}
// <</REPLAY:pipe_occupancy_gate>>

// <<REPLAY:roadmap_ladder_gate>>
// DID THE ROUND PLAN AGAINST THE LADDER ANALYZE BUILT, OR AROUND IT?
//
// The third of the three "was the artifact READ" gates, and the one that catches a failure the
// other two structurally cannot. taskGraphGate asks whether the dependency graph exists.
// pipeOccupancyGate asks whether each direction was priced against a busy or idle pipe. Both look
// at the directions that WERE issued. Neither can see the direction that was never issued.
//
// The cost of not having it, measured: a wave's Analyze produced a correct four-rung ladder ending
// in the exact acceptance shape the whole program existed to reach, wrote it to `roadmap.md` and to
// `analysis.json:candidate_directions`, and then `plan_round` was never handed either artifact and
// never mentioned a rung again — `grep -c 'D0\|D1\|D2\|D3\|roadmap'` over the run's report and
// insight log returned 0. Six directions were dispatched. The bounding readout never ran. The rung
// the ladder explicitly designated as THIS RUN'S POSITIVE CONTROL never ran, so the run had no
// control and had to improvise a substitute mid-flight from an engineer's scratch workspace. The
// fusion rung, gated on a rung that ran out of order and without its own mandatory arm, was never
// proposed at all. Every one of those is invisible in a per-direction check; all of them are
// obvious the moment you diff dispatched-rungs against the ladder.
//
// <<REPLAY:open_rungs>>
// A RUNG IS DONE WHEN IT PRODUCED A NUMBER, NOT WHEN IT WAS PLANNED.
//
// `dispatchedRungs` is filled from `lg.planned`, i.e. from what the round INTENDED to take. That is
// the right input for the ordering check — a rung whose prerequisite was attempted and failed should
// not silently satisfy the prerequisite, but neither should the run pretend the attempt never
// happened. It is the wrong input for "what is still owed", and until now it was the only record.
//
// What that cost, in full: one wave's ladder ended in the fusion rung the whole program existed to
// reach. The rung below it was planned, its device arm hit an illegal access, and it was retired as
// a bring-up failure — but it counted as dispatched, so nothing was owed. The fusion rung's producer
// side was then written in a later round and measured by nobody. The wave ended with the ladder
// nominally clear, a handover note inside one engineer's round directory, and no measurement. The
// next wave inherited none of it.
//
// So tally an OUTCOME per rung and carry the unfinished ones forward:
//   never_planned  no round has taken it
//   faulted        a direction took it and the code did not run
//   unmeasured     a direction took it, ran, and returned no verified number
//   measured       a verified number exists — and only this one closes the rung
const RUNG_OUTCOMES = ['never_planned', 'faulted', 'unmeasured', 'measured'];
function rungIdOf(c) {
  return String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
}
// Grade one round's result for the rung it declared. `verified` is the verified geomean; a direction
// that never reached verify has none.
// `role` is 'enabling' or 'terminal'. An ENABLING rung is closed by functional acceptance, not by a
// number: it has no standalone speedup to measure and demanding one is what leaves a fusion
// permanently half-built. A TERMINAL rung is closed by a measurement -- including a losing one.
function rungOutcomeOf(res, role) {
  if (!res) return 'unmeasured';
  const engFailed = !res.eng || /fail|error|crash/i.test(String(res.eng.status || ''));
  if (engFailed) return 'faulted';
  if (res.inactive || res.unbacked || res.same_artifact) return 'unmeasured';
  if (role === 'enabling') {
    return functionalAcceptance(res.ver).pass ? 'measured' : 'unmeasured';
  }
  return verifyCompleted(res.ver) && Number(res.ver && res.ver.verified_geomean) > 0
    ? 'measured' : 'unmeasured';
}
// tally: Map<rungId, {attempts, last_outcome}>. Returns the entries the next wave still owes,
// strongest-evidence-last so `measured` can never be reintroduced by a later weaker grade.
function openRungs(ladder, tally) {
  const t = tally instanceof Map ? tally : new Map(Object.entries(tally || {}));
  return (Array.isArray(ladder) ? ladder : [])
    .filter((c) => c && (c.id || c.title))
    .map((c) => {
      const id = rungIdOf(c);
      const e = t.get(id) || {};
      const outcome = RUNG_OUTCOMES.includes(e.last_outcome) ? e.last_outcome : 'never_planned';
      return { id, title: c.title || id, gated_on: Array.isArray(c.gated_on) ? c.gated_on : [],
        is_positive_control: !!c.is_positive_control,
        // Carried so the next wave knows what closing this rung would even mean. A prerequisite
        // re-planned as a speed experiment is a prerequisite that gets rejected again.
        step_role: stepRoleOf(c), enables: String(c.enables || '') || undefined,
        mandatory_arms: Array.isArray(c.mandatory_arms) ? c.mandatory_arms : [],
        target_shape: c.target_shape || undefined,
        attempts: Number(e.attempts) || 0, last_outcome: outcome };
    })
    .filter((c) => c.last_outcome !== 'measured');
}
// <</REPLAY:open_rungs>>

// <<REPLAY:direction_contract>>
// Join the planner's direction back to Analyze's rung. A planner routinely returns only the fields
// it needs to describe the immediate experiment; silently dropping step_role/mandatory_arms here
// changes how the experiment is judged and is how a prerequisite becomes a terminal speed test.
function enrichDirection(direction, ladder) {
  const d = direction || {};
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rungId = String(d.roadmap_rung || '').trim();
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId) || {};
  const pickArray = (a, b) => Array.isArray(a) ? a : (Array.isArray(b) ? b : []);
  return {
    ...d,
    gated_on: pickArray(d.gated_on, rung.gated_on),
    mandatory_arms: pickArray(d.mandatory_arms, rung.mandatory_arms),
    step_role: d.step_role || rung.step_role || 'terminal',
    enables: d.enables || rung.enables || undefined,
    cost_budget_pct: d.cost_budget_pct != null ? d.cost_budget_pct : rung.cost_budget_pct,
    target_shape: d.target_shape || rung.target_shape || undefined,
  };
}

function mandatoryArmsVerdict(direction, ver, strict) {
  const expected = Array.isArray(direction && direction.mandatory_arms)
    ? [...new Set(direction.mandatory_arms.map(String).filter(Boolean))] : [];
  if (!expected.length) return { pass: true, expected, actual: [], missing: [] };
  const actual = Array.isArray(ver && ver.arms_run)
    ? [...new Set(ver.arms_run.map(String).filter(Boolean))] : [];
  const have = new Set(actual);
  const missing = expected.filter((a) => !have.has(a));
  return {
    pass: missing.length === 0 && (!strict || actual.length > 0),
    expected, actual, missing,
    caveat: missing.length
      ? `mandatory arm(s) not independently verified: ${missing.join(', ')}`
      : '',
  };
}

function shouldVerifyDirection(direction, eng, forceVerify, score) {
  if (!eng || /fail|error|crash/i.test(String(eng.status || ''))) return false;
  if (forceVerify || String((direction && direction.step_role) || '').toLowerCase() === 'enabling') {
    return true;
  }
  return Number(score) > 1.0;
}

function strictDirectionVerdict(direction, ladder, completedRungs) {
  const d = direction || {};
  const rungId = String(d.roadmap_rung || '').trim();
  if (!rungId || rungId === 'off_ladder') {
    return { pass: false, reason: 'strict autonomy direction must name a ladder rung' };
  }
  const idOf = (c) => String((c && (c.id || c.title)) || '').trim().split(/\s+/)[0];
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId);
  if (!rung) return { pass: false, reason: `unknown roadmap rung ${rungId}` };
  const done = new Set(Array.isArray(completedRungs) ? completedRungs
    : (completedRungs instanceof Set ? [...completedRungs] : []));
  const unmet = (Array.isArray(rung.gated_on) ? rung.gated_on : []).filter((g) => !done.has(String(g)));
  if (unmet.length) {
    return { pass: false, reason: `${rungId} prerequisite(s) not completed to spec: ${unmet.join(', ')}` };
  }
  if (String(d.step_role || '') === 'enabling' && !String(d.enables || '').trim()) {
    return { pass: false, reason: `${rungId} is enabling but names no terminal rung` };
  }
  const rungRole = String(rung.step_role || 'terminal');
  if (String(d.step_role || 'terminal') !== rungRole ||
      (rungRole === 'enabling' && String(d.enables || '') !== String(rung.enables || ''))) {
    return { pass: false, reason: `${rungId} changed its Analyze step_role/enables contract` };
  }
  const expected = Array.isArray(rung.mandatory_arms) ? rung.mandatory_arms : [];
  const supplied = new Set(Array.isArray(d.mandatory_arms) ? d.mandatory_arms.map(String) : []);
  const missing = expected.filter((a) => !supplied.has(String(a)));
  if (missing.length) {
    return { pass: false, reason: `${rungId} dropped mandatory arm(s): ${missing.join(', ')}` };
  }
  if (rung.target_shape) {
    const planned = d.target_shape || {};
    const expectedStages = [...new Set((rung.target_shape.stages_fused || []).map(String))].sort();
    const plannedStages = [...new Set((planned.stages_fused || []).map(String))].sort();
    if (Number(planned.launches) !== Number(rung.target_shape.launches) ||
        (!!rung.target_shape.require_overlap && planned.require_overlap !== true) ||
        expectedStages.join('\0') !== plannedStages.join('\0')) {
      return { pass: false, reason: `${rungId} changed its Analyze target_shape` };
    }
  }
  return { pass: true, reason: '' };
}
// <</REPLAY:direction_contract>>

// <<REPLAY:autonomy_transition>>
function winnerCommitTransition(requested, commitResult, winner, winnerResult, strict) {
  const strictProof = !strict || (commitResult && commitResult.transaction_valid === true &&
    commitResult.exact_patch === true && String(commitResult.head_before || '').trim() &&
    String(commitResult.head_after || '').trim() &&
    commitResult.head_before !== commitResult.head_after);
  const committed = !!(requested && commitResult && commitResult.committed === true && winner && strictProof);
  const acceptanceReached = !!(committed && winnerResult && winnerResult.autonomy_contract &&
    winnerResult.autonomy_contract.accepted === true);
  return { committed, acceptanceReached };
}

function landedRungOutcome(baseOutcome, role, hasTargetShape, evidencePass,
                           winnerCommitted, winnerMatches, enablingLanded) {
  if (role === 'enabling') return enablingLanded ? baseOutcome : 'unmeasured';
  if (hasTargetShape) {
    return evidencePass && winnerCommitted && winnerMatches ? baseOutcome : 'unmeasured';
  }
  return baseOutcome;
}

function finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure) {
  if (directorStatus === 'flagged' || scopedGuardFailure) return 'flagged';
  if (strictProofFailure) return 'autonomy_incomplete';
  return directorStatus || 'unknown';
}

function finalMetricFailure(metric, targetGuards, primary, guards, attribution) {
  const scoped = Array.isArray(targetGuards) && targetGuards.length > 0;
  if ((scoped || metric === 'changed_kernel') && !(Number(primary) > 1.0)) return true;
  if (scoped && (!guards || !guards.complete || !guards.regression_pass)) return true;
  if (metric === 'changed_kernel') {
    const a = attribution || {};
    if (!(a.absolute_to_frozen === true && Number(a.changed_us) > 0 &&
          Number(a.replaced_sum_us) > 0)) return true;
  }
  return false;
}

function integrationEligible(integrate, score, bestIndividual, guards) {
  return !!(integrate && integrate.conclusion === 'improved' && integrate.best &&
    guards && guards.complete && guards.regression_pass && Number(score) > Number(bestIndividual));
}
// <</REPLAY:autonomy_transition>>

// Deliberately advisory, like its siblings. The ladder is Analyze's plan, and a round with better
// evidence SHOULD leave it — `rung_deviation` exists to make that a statement rather than a
// silence. The gate never blocks a direction; it refuses to let a skip be invisible.
function roadmapLadderGate(ladder, directions, completedRungs) {
  const rungs = (Array.isArray(ladder) ? ladder : []).filter((c) => c && (c.id || c.title));
  const dirs = Array.isArray(directions) ? directions : [];
  const done = new Set(Array.isArray(completedRungs) ? completedRungs : completedRungs instanceof Set ? [...completedRungs] : []);
  if (!rungs.length) {
    return { verdict: 'MISSING', summary: '', planned: [], caveat:
      'LADDER MISSING: Analyze returned no candidate_directions, so there is no recorded ordering ' +
      'for this run and nothing below can be checked against a plan. Each round is then free to ' +
      'start from the profile, which is exactly the state in which the most expensive direction — ' +
      'the one that is conditional on two cheaper ones — is never reached, because nothing is ' +
      'tracking that it was owed.' };
  }
  // `id` is what a plan can quote; fall back to the leading token of the title, which is how these
  // are written in practice ("D2 Per-token ... readiness").
  const idOf = (c) => String(c.id || String(c.title).trim().split(/\s+/)[0] || '').trim();
  const byId = new Map(rungs.map((c) => [idOf(c), c]));

  const planned = [];
  const undeclared = [];
  const offLadder = [];
  const gateViolations = [];
  for (const d of dirs) {
    if (!d) continue;
    const label = d.id || d.title || '?';
    const rung = (d.roadmap_rung || '').trim();
    if (!rung) { undeclared.push(label); continue; }
    if (rung === 'off_ladder') {
      if (!String(d.rung_deviation || '').trim()) offLadder.push(label);
      continue;
    }
    planned.push(rung);
    const c = byId.get(rung);
    if (!c) { offLadder.push(`${label} (names rung "${rung}", which is not on the ladder)`); continue; }
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    const unmet = gates.filter((g) => !done.has(g));
    if (unmet.length && !String(d.rung_deviation || '').trim()) {
      gateViolations.push(`${label} takes ${rung}, whose gated_on [${unmet.join(', ')}] ` +
        'has not been dispatched, with no rung_deviation stated');
    }
  }

  const seen = new Set([...done, ...planned]);
  const never = rungs.filter((c) => !seen.has(idOf(c)));
  const controls = never.filter((c) => c.is_positive_control);
  const blocked = rungs.filter((c) => {
    const gates = Array.isArray(c.gated_on) ? c.gated_on.map(String) : [];
    return gates.length && !seen.has(idOf(c)) && gates.some((g) => !seen.has(g));
  });

  const summary = `Ladder: ${rungs.length} rung(s) [${rungs.map(idOf).join(', ')}]; ` +
    `completed to spec [${[...done].join(', ') || 'none'}]; ` +
    `this round [${planned.join(', ') || 'none on ladder'}]; ` +
    `${never.length} never reached.`;

  const notes = [];
  if (undeclared.length) {
    notes.push(`DIRECTIONS WITH NO RUNG NAMED (${undeclared.length}): ${undeclared.join(', ')}. ` +
      'A direction that cannot say which rung it is — including "off_ladder" — has not been ' +
      'planned against the ladder, and the ladder cannot then be used to tell what is still owed.');
  }
  if (offLadder.length) {
    notes.push(`OFF-LADDER WITHOUT A STATED DEVIATION: ${offLadder.join('; ')}. Leaving the ladder ` +
      'is allowed and often right; leaving it silently means the rung it displaced is not recorded ' +
      'anywhere as still open.');
  }
  if (gateViolations.length) {
    notes.push('RUNG TAKEN OUT OF ORDER: ' + gateViolations.join('; ') + '. The prerequisite is ' +
      'usually what makes the result INTERPRETABLE, not merely what makes it likelier to win — a ' +
      'readiness rung measured before its bounding readout can read negative for a reason the ' +
      'bounding readout would have priced in advance.');
  }
  if (controls.length) {
    notes.push('THE LADDER\'S OWN POSITIVE CONTROL HAS NOT BEEN DISPATCHED: ' +
      controls.map(idOf).join(', ') + '. Analyze designated this rung as the change with a ' +
      'predicted direction that proves the harness can detect a win at all. Until it runs, every ' +
      'null below is ambiguous between "no effect" and "no instrument".');
  }
  if (blocked.length) {
    notes.push(`RUNGS STILL BLOCKED BY UNDISPATCHED PREREQUISITES (${blocked.length}): ` +
      blocked.map((c) => `${idOf(c)} gated_on [${(c.gated_on || []).join(', ')}]`).join('; ') +
      '. These cannot be reached by continuing to plan around the ladder. If a gate is being ' +
      'abandoned, say so; a gate that is simply never satisfied silently deletes everything above it.');
  }
  const verdict = notes.length
    ? (gateViolations.length || controls.length ? 'INCONSISTENT' : 'ADVISORY')
    : 'OK';
  return { verdict, summary, planned, caveat: notes.join(' ') };
}
// <</REPLAY:roadmap_ladder_gate>>

// <<REPLAY:evidence_stop>>
// WHAT THE NO-IMPROVE COUNTER IS ALLOWED TO COUNT.
//
// `MAX_NO_IMPROVE` stops a search that has stopped finding anything. To conclude that, the round
// has to have LOOKED — some direction must have produced a number that survived verification. A
// round whose directions never got the lease, never executed the patched path, or handed back a
// claim with no patch behind it has said nothing about the search space, and charging it to the
// stopping criterion ends the run on the strength of experiments that were never performed.
//
// This is not hypothetical. Measured on wave 14: budget 8, max_no_improve 3, three rounds, seven
// directions, one GPU lease left in the budget and the plan naming round 4 as the round that had
// to spend it. Round 1 dispatched three directions and produced zero measurements (two returned
// static analysis without a lease, one copied the baseline latencies into `optimized_ms`). Round 2
// dispatched two and produced zero the same way. Round 3 produced exactly one real reading. The
// counter reached 3 and the loop exited at the top of round 4 with the budget not exhausted. Of
// the three rounds it counted, one was an experiment that ran and lost and two were rounds in
// which nothing was measured at all.
//
// An earlier, narrower version of this rule existed — it exempted a round only when EVERY
// direction was `inactive`. It missed wave 14 twice over: a single direction that returned nothing
// (rather than returning an unexecuted patch) breaks the `every`, and "never got the lease" is not
// `inactive` in the first place.
//
// Admissibility is deliberately the same standard the rest of the round already applies, so a
// reading that is VOID for promotion cannot be counted as evidence for stopping:
//   - `inactive`      the patched path did not run, or was not proven to run
//   - `unbacked`      a claim whose declared patch does not exist
//   - `same_artifact` both arms resolved to one compiled binary
//   - outcome         `measured` per rungOutcomeOf — a verified number for a terminal step, or
//                     functional acceptance for an enabling one
const evidenceGap = (r, roleOf, outcomeOf) => {
  if (!r) return 'no result was returned';
  if (r.inactive) return `the patched path was ${r.inactive === 'no' ? 'not executed' : 'not proven to execute'}`;
  if (r.unbacked) return 'the claim has no patch behind it';
  if (r.same_artifact) return 'both arms were the same compiled binary';
  const outcome = outcomeOf(r, roleOf(r.d));
  return outcome === 'measured' ? null : `the direction ${outcome === 'faulted' ? 'faulted' : 'returned no verified number'}`;
};
function roundEvidence(clean, roleOf, outcomeOf) {
  const arr = Array.isArray(clean) ? clean : [];
  let measured = 0;
  const gaps = [];
  for (const r of arr) {
    const gap = evidenceGap(r, roleOf, outcomeOf);
    if (gap === null) measured++;
    else gaps.push(`${(r && r.d && r.d.id) || '?'}: ${gap}`);
  }
  return { measured, total: arr.length, gaps };
}
// <</REPLAY:evidence_stop>>

// <<REPLAY:claim_boundary>>
// THE CLAIM BOUNDARY. Three decisions that together answer one question: did a measurement that
// happened on hardware actually reach the scoring harness?
//
// They are lifted out as pure predicates rather than left inline because of what they cost when they
// were inline. On 2026-08-23 a real, reproducible, bit-identical +20.6% was measured three separate
// times and entered the harness zero times — once because the engineer's declared patch did not exist
// (the effect lived only in bench CLI flags), once because the engineer wrote a correct claim to disk
// and then kept measuring past the round's deadline, so `eng` was null. Neither failure was at the
// measurement boundary; the instrument worked every time. Both were here.
//
// The predicates are deliberately dumb. `needsRecovery` and `recovered` share one definition of
// "usable" so that recovery can never accept something the caller would have rejected, and `unbacked`
// requires BOTH that verify could not apply the patch AND that the engineer's own numbers claim a win
// — an unapplied patch under a null claim is just a null result, and calling it a reporting failure
// would train readers to ignore the label.
function claimBoundary(speedupOf) {
  const usable = (c) => !!c && Array.isArray(c.per_case) && c.per_case.length > 0;
  return {
    // Go looking on disk for a claim the engineer never handed back?
    needsRecovery: (eng) => !usable(eng),
    // Is what came off disk a claim, or an empty acknowledgement that there was nothing there?
    recovered: (onDisk) => usable(onDisk),
    // A win the engineer cannot hand over. Not "tried and lost" — unmeasured, and re-dispatchable.
    unbacked: (r) => !!r && !!r.ver && r.ver.status === 'apply_failed' && speedupOf(r.eng) > 1.0,
  };
}
// <</REPLAY:claim_boundary>>

// <<REPLAY:objective_gate>>
// A WAVE THAT SKIPS THE POSITIVE CONTROL MUST NOT SHIP A NUMBER.
//
// `objective: 'working_kernel'` is allowed to skip the control, and skipping it is most of what makes
// the mode affordable — the control batch is 2.2 of the wave's leases (see OBJECTIVE above). The
// hazard is obvious and has already happened twice in this project's history: a wave collects timings
// anyway, someone reads one, and it enters a report as a result. The comment on POSITIVE_CONTROL
// states the reason it cannot be read — without a control, "we found nothing" and "we cannot see
// anything" produce byte-identical output — and that reason does not weaken just because the wave
// was not hunting a win.
//
// So the trade is stated explicitly rather than left to discipline: this mode does not make timing
// numbers cheaper, it makes them INADMISSIBLE. Every finite geomean from an uncontrolled wave is
// VOID, 1.000x included. Voiding 1.000x is not pedantry — an uncontrolled 1.000x is precisely the
// reading a blind harness emits, and it is the one that has been believed before.
//
// A wave that wants a number pays for the control. `pcRan` is the caller's honest answer to "did the
// positive control actually run in this wave", so a run that keeps the control keeps its scoring.
function objectiveVerdict(objective, pcRan, geomean) {
  if (objective !== 'working_kernel') return { state: 'scored', caveat: '' };
  if (pcRan) return { state: 'scored', caveat: '' };
  if (!Number.isFinite(Number(geomean))) return { state: 'no_number', caveat: '' };
  return {
    state: 'void_no_control',
    caveat: 'VOID AS A TIMING RESULT: this wave ran with objective=working_kernel and no positive ' +
      'control, so nothing established that its harness can resolve an effect at all. The reading ' +
      'is withdrawn rather than reported with a warning, 1.000x included — an uncontrolled 1.000x ' +
      'is what a blind harness emits. The artifact is still judged on whether it RUNS. To obtain a ' +
      'timing result, re-measure it in a wave that pays for the control.',
  };
}
// A round's candidate is committed on RUNNING, not on being faster. Deliberately strict on
// activation for the usual reason: an unexercised patch reads as a clean pass on every other field.
// `liveness: 'n/a'` is accepted because it is the honest answer for a non-distributed candidate,
// while 'fail' is a timeout, which this workflow has always counted as a failure and never a skip.
function runsCleanly(ver, requirements) {
  return functionalAcceptance(ver, requirements).pass;
}
// <</REPLAY:objective_gate>>

// <<REPLAY:enabling_step>>
// A STEP THAT ENABLES A FUSION IS NOT A STEP THAT SPEEDS ONE UP.
//
// The round filter that decides what survives is `primSpeedup(ver) > 1.0`. Every direction is judged
// by whether it made the operator faster THIS ROUND, and everything else is discarded. For a fusion
// built in stages that filter is not a quality bar, it is a structural block, because the producer
// half of a fusion cannot be faster on its own by construction: it adds completion signalling and a
// second buffer, it has no consumer yet to hand the work to, and the only thing it can possibly
// measure is its own overhead. It is supposed to be slower. It gets rejected for being exactly what
// it is, does not enter the next round's canonical tree, and the consumer half is then written
// against a tree where the producer half no longer exists — so it is never written at all.
//
// That is how this project stopped at half a fusion. The producer side of the combine fold was
// authored, compiled, and never carried forward; the consumer side was never begun; the two-launch
// shape that was the entire acceptance criterion was never reached.
//
// So a direction declares which of two things it is, and is judged accordingly:
//
//   terminal  — it closes a fusion chain (or it is a standalone optimisation). Judged on SPEED, with
//               the full protocol: all four guards, rank-max, base/candidate/blank-control
//               interleaved, and the overlap fraction on the edge it claims to have fused.
//   enabling  — it is a prerequisite. Judged on FUNCTION: it builds, its path is actually taken,
//               the results are correct, and it does not deadlock. Its timing is RECORDED AS A COST,
//               not used to reject it. It is committed to the canonical tree so the next step has
//               something to build on.
//
// Three things keep `enabling` from being a way to commit anything at all:
//
//   1. It must name the terminal rung it enables. A prerequisite to nothing is a regression.
//   2. Its cost is bounded. `cost_budget_pct` is what the direction predicted it would cost; blowing
//      through that is a design error rather than an expected temporary slowdown, and it is rejected.
//   3. The cost is DEBT, and the debt is tracked by name until the terminal step pays it. A chain
//      that never closes leaves the tree slower than it found it, and that has to be on the record
//      as an outstanding balance rather than absorbed into the baseline.
//
// (3) is also what stops the chain from laundering its own overhead. Every committed enabling step
// makes the canonical tree slower, and the terminal step is measured against the canonical tree. Left
// alone, a chain that costs 3% and then recovers 3% reads as +3%. So the baseline is PINNED when the
// first enabling step of a chain is committed, and the terminal step's claim is against the pin.
const ENABLING_DEFAULT_BUDGET_PCT = 5.0;   // a prerequisite that costs more than this is a design error
const CHAIN_DEBT_MAX_ROUNDS = 2;           // rounds an unpaid chain may stay open before it is called out

const round2 = (x) => Math.round(Number(x) * 100) / 100;

function stepRoleOf(d) {
  const s = String((d && d.step_role) || '').trim().toLowerCase();
  return s === 'enabling' ? 'enabling' : 'terminal';
}

// The four functional conditions, each reported by name. "It failed acceptance" and "it failed
// acceptance because its path was never taken" are different findings and only the second is
// actionable, so the caller gets the list rather than a boolean.
// A verification that actually COMPLETED: the verifier ran and produced a correctness+timing result.
// Shared by functionalAcceptance (the enabling path) and the candidate `verified` filter (the
// terminal path) so the two cannot disagree about whether a run counts -- and they did. A terminal
// fusion is CORRECT but SLOWER the first time it lands, which the verifier reports as
// status:'regression' (see verify_engineer.md: "shows a regression ... Report it as
// status:'regression' with the numbers anyway"). functionalAcceptance accepted that; the candidate
// filter demanded status==='verified' exactly and dropped it -- so the consumer half of a fusion was
// admitted on the enabling path and rejected on the terminal path for being exactly what it is.
// Excludes only the states where the comparison never happened (apply_failed) or was tainted
// (plagiarized, harness_modified, inactive/correctness handled separately by the callers).
function verifyCompleted(ver) {
  if (!ver) return false;
  const s = String(ver.status == null ? '' : ver.status).toLowerCase();
  return ['verified', 'regression', 'slower'].includes(s) || s.startsWith('verif');
}

function functionalAcceptance(ver, requirements) {
  const req = requirements || {};
  const s = (v) => String(v == null ? '' : v).toLowerCase();
  const missing = [];
  if (!ver) return { pass: false, missing: ['no verify result at all'] };
  if (!verifyCompleted(ver)) {
    missing.push(`verify did not complete (status=${ver.status || 'none'})`);
  }
  if (!s(ver.correctness).startsWith('pass')) missing.push('correctness did not pass');
  // Acceptance criterion 4 carries a NUMBER (relL2 < 0.10), and a free-text `correctness: 'pass'`
  // cannot be compared to it. When the verifier reports `accuracy`, hold the candidate to the
  // threshold mechanically -- a "pass" string sitting on top of a relL2 of 0.4 is not a pass. When
  // Outside strict mode, absence preserves the historical string check. Under strict autonomy the
  // numeric evidence is required and absence fails explicitly.
  const acc = (ver && ver.accuracy) || {};
  const accRows = Array.isArray(ver && ver.accuracy_results) && ver.accuracy_results.length
    ? ver.accuracy_results : (Object.keys(acc).length ? [acc] : []);
  const accuracyGuards = Array.isArray(req.accuracyGuards) ? req.accuracyGuards.map(String) : [];
  const validateAccuracy = (row, guard) => {
    const val = Number(row && row.value), threshold = Number(row && row.threshold);
    if (!row || String(row.metric || '') !== String(req.accuracyMetric || 'relL2')) {
      return `${guard || 'accuracy'} missing metric ${req.accuracyMetric || 'relL2'}`;
    }
    if (!(Number.isFinite(val) && val >= 0 && Number.isFinite(threshold) && threshold > 0 &&
          threshold <= Number(req.accuracyThreshold || threshold) &&
          val < threshold && val < Number(req.accuracyThreshold || threshold) &&
          String(row.method || '').trim())) {
      return `${guard || row.guard || 'accuracy'} has invalid value/threshold/method`;
    }
    return '';
  };
  if (req.requireAccuracy) {
    const guards = accuracyGuards.length ? accuracyGuards : [''];
    for (const guard of guards) {
      const row = guard ? accRows.find((r) => String(r.guard || '') === guard) : accRows[0];
      const gap = validateAccuracy(row, guard);
      if (gap) missing.push(gap);
    }
  } else {
    const accVal = Number(acc.value), accThr = Number(acc.threshold);
    if (Number.isFinite(accVal) && Number.isFinite(accThr) && !(accVal < accThr)) {
      missing.push(`accuracy ${acc.metric || 'error'}=${accVal} is not < threshold ${accThr}`);
    }
  }
  if (s(ver.activation_confirmed) !== 'yes') missing.push('the new path was not confirmed to run');
  // A compile screen is not a readout. `activation_confirmed:'yes'` and `artifact_distinct:'yes'` are
  // both satisfiable without a card — a host marker and a static ISA hash respectively — so on a wave
  // that opts in, the switched path must additionally be shown to have EXECUTED ON THE DEVICE this round.
  // Absent/`no`/`unknown` here is void, not negative: it is the state a JIT-trace-time crash lives in.
  if (req.requireHardwareActivation && s(ver.activation_on_hardware) !== 'yes') {
    missing.push('the switched path was not confirmed to run ON HARDWARE this round (a compile-only / ' +
      'static-ISA / CPU-dry-run proof does not exercise it; require a gpu_lock device run, with the ' +
      `switch set, that reached the candidate — got activation_on_hardware=${ver.activation_on_hardware || 'none'})`);
  }
  if (req.requireLiveness) {
    if (s(ver.liveness) !== 'pass') missing.push('liveness pass is required and was not reported');
    const requiredReplays = Number(req.requiredReplays || 0);
    const replayGuards = Array.isArray(req.replayGuards) ? req.replayGuards.map(String) : [];
    const replayRows = Array.isArray(ver.replay_results) ? ver.replay_results : [];
    if (replayGuards.length) {
      for (const guard of replayGuards) {
        const row = replayRows.find((r) => String(r.guard || '') === guard);
        if (!row || String(row.status || '').toLowerCase() !== 'pass' ||
            Number(row.count) < requiredReplays ||
            (req.requireGraphSafe && String(row.graph_safe || '').toLowerCase() !== 'pass')) {
          missing.push(`${guard} replay evidence is missing or below ${requiredReplays}`);
        }
      }
    } else {
      const replayCount = Number(ver.replay_count);
      if (requiredReplays > 0 && (!Number.isFinite(replayCount) || replayCount < requiredReplays)) {
        missing.push(`replay_count ${Number.isFinite(replayCount) ? replayCount : 'missing'} < required ${requiredReplays}`);
      }
    }
  } else if (!['pass', 'n/a', ''].includes(s(ver.liveness))) {
    missing.push('liveness failed (hang or timeout)');
  }
  // Graph capture/replay safety is part of the same "does it run correctly under the real harness"
  // bar as liveness: criterion 4 wants 1000 CUDA-Graph replays per route with no deadlock, and a
  // kernel that captures then replays wrong has failed FUNCTION, not speed. Reported-and-bad fails;
  // `n/a`/empty is the honest answer off the graph-captured path and stays a pass.
  if (req.requireGraphSafe) {
    if (s(ver.graph_safe) !== 'pass') missing.push('graph capture/replay pass is required and was not reported');
  } else if (!['pass', 'n/a', ''].includes(s(ver.graph_safe))) {
    missing.push('graph capture/replay is not safe');
  }
  if (req.requireArtifactDistinct && s(ver.artifact_distinct) !== 'yes') {
    missing.push('JIT artifact distinctness proof is required and was not reported');
  } else if (req.requireArtifactDistinct &&
      (!String(ver.artifact_hash_base || '').trim() ||
       !String(ver.artifact_hash_candidate || '').trim() ||
       String(ver.artifact_hash_base) === String(ver.artifact_hash_candidate))) {
    missing.push('JIT artifact hashes are missing or identical');
  }
  const requiredArms = Array.isArray(req.requiredArms) ? req.requiredArms.map(String).filter(Boolean) : [];
  if (requiredArms.length) {
    const actualArms = new Set(Array.isArray(ver.arms_run) ? ver.arms_run.map(String) : []);
    const missingArms = requiredArms.filter((a) => !actualArms.has(a));
    if (missingArms.length) missing.push(`mandatory arm(s) not independently verified: ${missingArms.join(', ')}`);
  }
  return { pass: missing.length === 0, missing };
}

// The verdict for ONE enabling direction. `geomean` is its measured speedup, which is expected to be
// below 1 and is not a reason to reject.
function enablingVerdict(d, ver, geomean, requirements) {
  const fa = functionalAcceptance(ver, requirements);
  const enables = String((d && d.enables) || '').trim();
  const budget = Number((d && d.cost_budget_pct) != null ? d.cost_budget_pct : ENABLING_DEFAULT_BUDGET_PCT);
  const g = Number(geomean);
  const costPct = Number.isFinite(g) && g > 0 ? round2((1 / g - 1) * 100) : null;
  if (!enables) {
    return { commit: false, cost_pct: costPct, reason:
      'declared step_role=enabling but named no terminal rung in `enables`. A step that is a ' +
      'prerequisite to nothing is not a prerequisite, it is a change that made the operator slower.' };
  }
  if (!fa.pass) {
    return { commit: false, cost_pct: costPct, reason:
      `failed functional acceptance: ${fa.missing.join('; ')}. An enabling step is exempt from the ` +
      'speed bar and from nothing else — these four conditions are the whole bar it is held to.' };
  }
  if (costPct != null && Number.isFinite(budget) && costPct > budget) {
    return { commit: false, cost_pct: costPct, reason:
      `costs ${costPct}%, over its own declared budget of ${budget}%. A prerequisite is allowed to ` +
      'be slower than what it replaces; it is not allowed to cost more than the fusion it enables ' +
      'can return. Over budget is a design error, not a temporary slowdown.' };
  }
  return { commit: true, cost_pct: costPct, enables, reason:
    `functional acceptance passed (builds, path taken, correct, no hang) and it enables ${enables}. ` +
    `Committed on FUNCTION at ${costPct == null ? 'an unread cost' : costPct + '% cost'}, carried as ` +
    'debt against that rung. Not scored as a win and cumulative is unchanged.' };
}

// The outstanding balance. `debt` is a list of {round, id, enables, cost_pct}.
function chainDebtReport(debt, roundNow, ladderMeasured) {
  const open = (Array.isArray(debt) ? debt : []).filter((e) => e && !ladderMeasured.has(e.enables));
  if (!open.length) return { open: [], overdue: [], caveat: '' };
  const byRung = new Map();
  for (const e of open) {
    const k = e.enables;
    const cur = byRung.get(k) || { enables: k, steps: [], cost_pct: 0, oldest_round: e.round };
    cur.steps.push(e.id);
    cur.cost_pct = round2(cur.cost_pct + (Number(e.cost_pct) || 0));
    cur.oldest_round = Math.min(cur.oldest_round, e.round);
    byRung.set(k, cur);
  }
  const rows = [...byRung.values()];
  const overdue = rows.filter((r) => roundNow - r.oldest_round >= CHAIN_DEBT_MAX_ROUNDS);
  const caveat = 'CHAIN DEBT OUTSTANDING: ' + rows.map((r) =>
    `${r.enables} owes ${r.cost_pct}% from [${r.steps.join(', ')}] since round ${r.oldest_round}`).join('; ') +
    '. These steps were committed on function, not on speed, so the canonical tree is currently ' +
    'that much slower and none of it has been paid back yet. The terminal rung is the only thing ' +
    'that can pay it.' +
    (overdue.length ? ` OVERDUE (${CHAIN_DEBT_MAX_ROUNDS}+ rounds unpaid): ${overdue.map((r) => r.enables).join(', ')}. ` +
      'Plan the terminal rung this round or state what happens to the debt — a chain abandoned ' +
      'half-built leaves the tree slower than it was found, which is the worst of the three outcomes.' : '');
  return { open: rows, overdue, caveat };
}
// <</REPLAY:enabling_step>>

// <<REPLAY:terminal_forcing>>
// NEAR THE END OF THE BUDGET, FORCING REPLACES ADVISING.
//
// roadmapLadderGate is advisory: it prints "the terminal rung was not planned" and lets the round
// proceed. That is correct while there is budget to both explore a lever AND still come back and
// close the chain. It is wrong the moment there is not. A wave that commits three enabling steps,
// leaves their terminal fusion rung unplanned every round because there is "one more round to
// spare", and then runs out of budget ships a tree that is SLOWER than it found it and a fusion that
// was never closed -- the exact half-fusion this project reached. The advisory never becomes an
// action, so at the end nobody DECIDED to abandon the chain; each round independently decided it
// could wait, and the sum was abandonment.
//
// So this converts the advisory into an action at the point it stops being safe to defer. Force the
// round onto an owed terminal rung when EITHER a chain is overdue (a prerequisite has sat unpaid for
// CHAIN_DEBT_MAX_ROUNDS) OR there is no longer slack to both explore and still pay every open chain
// (remaining <= owed count + 1). Returns null when nothing is owed, or when the planner is already
// taking one of the owed rungs -- forcing is the exception, and a planner closing the chain on its
// own is never overridden.
function terminalRungToForce(debtReport, ladder, plannedRungs, remaining) {
  const open = (debtReport && Array.isArray(debtReport.open)) ? debtReport.open : [];
  if (!open.length) return null;
  const idOf = (c) => String((c && c.id) || String((c && c.title) || '').trim().split(/\s+/)[0] || '').trim();
  const planned = new Set((Array.isArray(plannedRungs) ? plannedRungs : []).map((r) => String(r).trim()));
  const owed = open
    .map((r) => ({ rung: String(r.enables || '').trim(), oldest: Number(r.oldest_round) }))
    .filter((r) => r.rung && !planned.has(r.rung));
  if (!owed.length) return null;
  const overdue = new Set(((debtReport && Array.isArray(debtReport.overdue)) ? debtReport.overdue : [])
    .map((r) => String(r.enables || '').trim()));
  const budgetTight = Number(remaining) <= owed.length + 1;
  const anyOverdue = owed.some((r) => overdue.has(r.rung));
  if (!budgetTight && !anyOverdue) return null;
  // Oldest-owed (and overdue-first) is the chain closest to being abandoned, so close it first.
  owed.sort((a, b) => {
    const ao = overdue.has(a.rung) ? 0 : 1, bo = overdue.has(b.rung) ? 0 : 1;
    if (ao !== bo) return ao - bo;
    return (a.oldest || Infinity) - (b.oldest || Infinity);
  });
  const rungId = owed[0].rung;
  const rung = (Array.isArray(ladder) ? ladder : []).find((c) => idOf(c) === rungId) || null;
  return { rungId, rung, reason: overdue.has(rungId) ? 'overdue' : 'budget_tight' };
}
// <</REPLAY:terminal_forcing>>

const CLAIM = claimBoundary(primSpeedup);

const REQUIRE_TASK_GRAPH = !!A.require_task_graph;
// Acceptance criterion 1's target launch count per EP rank: the fused megakernel plus the one
// separate pre-dispatch quant launch = 2. Configurable so a campaign with a different fusion target
// can set it, but the default is the shape this project is judged against. See launchVerdict.
function functionalRequirementsFor(direction, purpose) {
  const distributed = String((direction && direction.specialty) || '') === 'distributed';
  const shapeChanging = !!(direction && direction.target_shape);
  const strictPath = STRICT_AUTONOMY && (distributed || shapeChanging);
  const runtimeCommit = purpose === 'commit' && stepRoleOf(direction) === 'terminal';
  return {
    requireAccuracy: strictPath,
    requireLiveness: strictPath,
    requireGraphSafe: strictPath,
    requireArtifactDistinct: strictPath && REQUIRE_ARTIFACT_DISTINCT,
    // Deliberately NOT gated on strictPath and NOT emptied by runtimeCommit: "did the switched path run
    // on a card this round" is the one requirement a still-slow terminal commit must keep, because it is
    // exactly what the commit-on-running gate means by "runs". Waiving it at commit (as requiredArms is
    // waived) is how a patch whose ON path never executed gets preserved into the next round's tree.
    requireHardwareActivation: REQUIRE_HW_ACTIVATION,
    requiredReplays: strictPath ? REQUIRED_REPLAYS : 0,
    replayGuards: strictPath ? [...new Set([...TARGET_GUARDS, ...REGRESSION_GUARDS])] : [],
    accuracyMetric: ACCURACY_METRIC,
    accuracyThreshold: ACCURACY_THRESHOLD,
    accuracyGuards: strictPath ? TARGET_GUARDS : [],
    // Null/attribution/overlap arms decide final acceptance, not whether a correct running terminal
    // artifact is preserved for the next round. Enabling arms remain part of functional acceptance.
    requiredArms: runtimeCommit ? [] : ((direction && direction.mandatory_arms) || []),
    strict: STRICT_AUTONOMY,
  };
}
// Overlap caveats, accumulated across rounds. They travel to the report for the same reason the
// task-graph caveat does: the thing they qualify — whether a fused win was ever shown to BE an
// overlap win — is invisible in the final number, and by report time nobody re-reads the log.
// Set true the first time any verified candidate reports a launch count at or below the target: the
// acceptance shape (criterion 1) was actually reached by something this wave, as opposed to merely
// never checked. It only ratchets true; a later three-launch candidate does not unset it.
let TWO_LAUNCH_REACHED = false;
let AUTONOMY_ACCEPTANCE_REACHED = false;
const ACCEPTANCE_CAVEATS = [];
const OVERLAP_CAVEATS = [];
// Launch-shape caveats: how many launches each candidate reached, and whether the two-launch fused
// shape (acceptance criterion 1) was actually met. Carried for the same reason as the overlap
// caveats -- an unfused candidate that reads like a fused one is invisible unless the count travels.
const LAUNCH_CAVEATS = [];
// Guard-conditioning caveats: a target-route reading that was under-sampled for its bimodal state, or
// a wave whose only readings were on off-target (skew) rails. Carried to the report because "the win
// was on a skew guard" is invisible in a single aggregate number.
const GUARD_CAVEATS = [];
// Withdrawn timing readings, same rationale: a number that was voided at round 2 is invisible in a
// report that only prints what survived, and "no number" and "a number we refuse to read" are the
// two states this project has most often confused.
const OBJECTIVE_CAVEATS = [];
// Attribution caveats, and the rejections. A candidate excluded because its win was outside the
// kernel it changed leaves NO trace in a report that prints only survivors, and "we had no
// candidate" and "we had one and refused it, for this reason" are exactly the two states a reader
// must be able to tell apart -- the second is a finding about the mechanism, and it is the more
// useful of the two.
const ATTRIBUTION_CAVEATS = [];
// Chain caveats: enabling steps kept on function, enabling steps refused, and the outstanding debt
// they leave behind. Surfaced for the same reason the others are -- a tree that is 3% slower because
// two prerequisites landed and their terminal step never did is a tree whose report says 1.00x with
// no indication that the number is a half-built fusion rather than a null result.
const CHAIN_CAVEATS = [];
// Enabling steps committed on function, each {round, id, enables, cost_pct}. Cleared per rung when
// that rung is finally measured.
const CHAIN_DEBT = [];
// Rung ids that have produced a measurement. Kept alongside rungTally because the debt report needs
// the same fact and must not depend on the tally's grading order.
const LADDER_MEASURED = new Set();
// The cumulative speedup at the moment the FIRST enabling step of any open chain was committed, and
// the canonical commit it was pinned at. Every committed enabling step makes the canonical tree
// slower, and the terminal step is measured against the canonical tree -- so without this pin a
// chain that costs 3% and then recovers 3% reads as +3%. The pin is the blank control the terminal
// step's claim has to be stated against.
let CHAIN_BASELINE = null;
// Hardware/toolchain facts this wave established, for a human to merge into knowledge/. See
// MEMORY_SCHEMA.knowledge_delta for why this is surfaced rather than written.
const KNOWLEDGE_DELTA = [];
let TASK_GRAPH_CAVEAT = '';
let PIPE_TABLE_CAVEAT = '';
if (REQUIRE_TASK_GRAPH) {
  const g = taskGraphGate(analysis && analysis.task_graph);
  if (g.summary) log(g.summary);
  TASK_GRAPH_CAVEAT = g.caveat;
  if (g.caveat) log(g.caveat);
  // The graph says which orderings are real; the pipe table says whether any of them is worth
  // touching. They fail independently — a correct graph with no table is exactly the state that
  // funded four waves of occupancy and launch-count directions on a machine whose busiest pipe was
  // at 41%. Print both, always, and carry both to the report.
  const pg = pipeOccupancyGate(analysis && analysis.resource_timeline, []);
  if (pg.summary) log(pg.summary);
  PIPE_TABLE_CAVEAT = pg.caveat;
  if (pg.caveat) log(pg.caveat);
}

// The ladder Analyze ranked, and the record of what has been taken off it. Both are threaded into
// every plan_round call: an artifact a phase is not handed is an artifact that phase does not use,
// and this one was orphaned for an entire wave — correct ladder, zero mentions downstream, the
// acceptance-shape rung never proposed. Not gated on require_task_graph; every run has a roadmap.
const LADDER = (analysis && Array.isArray(analysis.candidate_directions)) ? analysis.candidate_directions : [];
const dispatchedRungs = new Set();
// Separate from the set above and it has to be: that one answers "was this taken", this one answers
// "did taking it produce a number". Only the second can say what the next wave still owes.
const rungTally = new Map();   // rungId -> { attempts, last_outcome }
function recordRungOutcome(id, outcome) {
  const key = String(id || '').trim();
  if (!key) return;
  const e = rungTally.get(key) || { attempts: 0, last_outcome: 'never_planned' };
  // `measured` is absorbing. A rung that produced a number once is closed; a later round that takes
  // it again and faults must not reopen it, or the ladder never terminates.
  if (e.last_outcome !== 'measured') e.last_outcome = outcome;
  rungTally.set(key, e);
}
if (STRICT_AUTONOMY) {
  if (!LADDER.length) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: Analyze returned no candidate_directions. A proof run with no ' +
      'machine-readable ladder cannot show that the two-launch terminal was derived.');
  }
  const terminalTargets = LADDER.filter((c) => c && c.target_shape &&
    Number(c.target_shape.launches) <= LAUNCH_TARGET);
  if (!terminalTargets.length) {
    throw new Error(
      `STRICT AUTONOMY REFUSED: no ladder rung declares target_shape.launches <= ${LAUNCH_TARGET}. ` +
      'A prose promise to fuse the operator cannot close the machine-tracked acceptance shape.');
  }
  const incompleteTargets = terminalTargets.filter((c) =>
    !Array.isArray(c.mandatory_arms) || c.mandatory_arms.length === 0 ||
    !Array.isArray(c.target_shape.stages_fused) || c.target_shape.stages_fused.length === 0 ||
    (REQUIRE_OVERLAP && c.target_shape.require_overlap !== true));
  if (incompleteTargets.length) {
    throw new Error(
      'STRICT AUTONOMY REFUSED: terminal target rung(s) must declare mandatory_arms and the required ' +
      `overlap shape: ${incompleteTargets.map((c) => c.id || c.title).join(', ')}`);
  }
}
{
  const lg0 = roadmapLadderGate(LADDER, [], new Set());
  if (lg0.summary) log(lg0.summary);
  if (lg0.caveat) log(lg0.caveat);
}
// Under the fusion campaign (working_kernel) an empty ladder is not merely advisory: it makes the
// rung / chain-debt / terminal-forcing machinery inert, so a staged fusion cannot be tracked to its
// terminal rung and the run degrades into independent one-shot rounds -- the state this project
// reached and then reported as 1.00x. It is not hard-failed here (a degraded Analyze may legitimately
// return no directions and be re-requested by analyze_resume_fallback), but it is escalated into the
// report so the absence is a decision on the record rather than a silence.
if (WORKING_KERNEL && !LADDER.length) {
  const msg = 'LADDER EMPTY under objective=working_kernel: Analyze returned no candidate_directions, ' +
    'so there is no staged path to a fused terminal rung. The chain-debt and terminal-forcing ' +
    'machinery has nothing to act on and every round will start from the profile. Re-run Analyze so ' +
    'it emits the fusion ladder, or this wave cannot converge on the two-launch shape by construction.';
  log(msg);
  CHAIN_CAVEATS.push(`setup: ${msg}`);
}

// Surface prior art loudly. A direction that already exists somewhere is a measurement, not a round
// of engineering — and one that exists but is MISSING from the tree under optimization is a silent
// ceiling on everything this run can achieve. Neither is visible unless it is printed here.
// An OMITTED key and an EMPTY array are different findings and must not collapse into each other.
// `[]` means the sweep ran and found nothing; absent means nothing is known either way — and a
// report is then free to assert "all prior art was in_baseline: true" with no record behind it.
// That happened: a run whose analysis.json had no prior_art key at all filed a report claiming the
// fused path existed in the baseline as an env-gated opt-in. It did not. Nobody could check,
// because there was nothing to check against.
const PRIOR_ART_RECORDED = !!(analysis && Array.isArray(analysis.prior_art));
const PRIOR_ART = PRIOR_ART_RECORDED ? analysis.prior_art : [];
if (!PRIOR_ART_RECORDED) {
  log(`PRIOR ART UNRECORDED: analysis.json has no "prior_art" key, so the step-4d sweep is not on ` +
      `the record — this is NOT the same as finding none. Every later statement about what does or ` +
      `does not exist in the baseline is UNSOURCED and must be read as such.`);
} else if (PRIOR_ART.length === 0) {
  log(`PRIOR ART: none — the sweep ran and reported an empty result.`);
}
for (const pa of PRIOR_ART) {
  const where = CAPABILITY_EVAL ? 'HIDDEN_REFERENCE' : (pa.implemented_at || '?');
  const eff = pa.measured_effect ? ` (measured ${pa.measured_effect})` : '';
  if (pa.in_baseline === false && CAPABILITY_EVAL) {
    // Same finding, opposite instruction. Here the absent implementation is the ANSWER KEY: it
    // bounds what a successful run should reach, and copying it makes the run unreadable.
    log(`PRIOR ART NOT IN BASELINE (capability eval): "${pa.direction}" exists at ${where}${eff} but ` +
        `is ABSENT from the tree being optimized. Treat it as the answer key, NOT as a source: it is ` +
        `the bar this run must reach on its own. Do not port it, and do not surface its location to ` +
        `engineers.`);
  } else if (pa.in_baseline === false) {
    log(`PRIOR ART NOT IN BASELINE: "${pa.direction}" is already implemented at ${where}${eff} but ` +
        `is ABSENT from the tree being optimized. Port or measure it before re-deriving it; if the ` +
        `intent was to optimize the tree that HAS it, this run is pointed at the wrong target.`);
  } else {
    log(`PRIOR ART: "${pa.direction}" already implemented at ${where}${eff}; ` +
        `enable via ${pa.how_to_enable || '?'}. Measure it, do not re-derive it.`);
  }
  // `in_baseline` decides whether a direction is one A/B or a whole round, and under capability_eval
  // it decides whether the answer key is inside the tree. An unevidenced boolean is a guess wearing
  // a schema field.
  if (pa.in_baseline != null && !pa.evidence) {
    log(`PRIOR ART UNEVIDENCED: "${pa.direction}" asserts in_baseline=${pa.in_baseline} with no ` +
        `"evidence" field naming the check that established it. Treat the flag as unverified.`);
  }
}

// Leak scrub. tech_lead.md 4d already forbids writing reference locations into engineer-visible
// files under capability_eval, but prose instructions are exactly what failed the first time: the
// roadmap led with the reference path, branch and HEAD, and an engineer copied a whole file from it.
// So check rather than trust. This warns instead of aborting — the leak may be a bare mention in a
// sentence the human reader wants, and killing a run at Analyze over a substring match trades one
// silent failure for another. The provenance gate in verify is the enforcing half.
if (CAPABILITY_EVAL && KNOWN_REFERENCE_PATHS.length) {
  const needles = KNOWN_REFERENCE_PATHS.flatMap((p) => {
    const base = p.replace(/\/+$/, '').split('/').pop();
    return [p.replace(/\/+$/, ''), ...(base && base.length > 6 ? [base] : [])];
  });
  // THIS CHECK WAS DEAD CODE, AND ITS DEATH WAS SILENT.
  //
  // `fs` is never defined in this file -- workflow scripts run without Node's fs/path, which the note
  // at the top of this file says in as many words. So `fs.readFileSync` below threw ReferenceError on
  // every iteration, the `catch { continue }` swallowed it, and the loop did nothing. It has never run.
  // The cost is on the record: wave 7's analysis.json shipped four reference paths and was found by a
  // human running grep, not by this. A guard that cannot fail is worse than no guard, because the
  // absence of a warning gets read as a clean result. Say so, once, loudly, and route to the tool that
  // actually walks the tree -- do not leave a reader thinking this ran.
  if (typeof fs === 'undefined') {
    log('CONTAINMENT CHECK INERT: the in-workflow reference-path scan of EVAL_DIR cannot run — ' +
        'workflow scripts have no fs. This is not a clean result, it is no result. Run ' +
        'scripts/reference_leak_sweep.sh --tree <run tree> and scripts/skill_address_scan.sh ' +
        'out of band; only those actually read the tree.');
  }
  for (const f of (typeof fs === 'undefined' ? [] : ['roadmap.md', 'codebase_context.md', 'analysis.json'])) {
    let txt = '';
    try { txt = fs.readFileSync(`${EVAL_DIR}/${f}`, 'utf8'); } catch { continue; }
    // analysis.json legitimately carries locations in prior_art[].implemented_at (that field is for
    // the orchestrator and the human, and is not forwarded to engineers). Scan the rest of it only.
    if (f === 'analysis.json') {
      try {
        const j = JSON.parse(txt); delete j.prior_art; txt = JSON.stringify(j);
      } catch { /* unparseable -> scan verbatim, a false positive here is cheap */ }
    }
    const hit = needles.find((n) => txt.includes(n));
    if (hit) {
      log(`LEAK WARNING: ${f} mentions the reference location "${hit}" while capability_eval is on. ` +
          `Engineers read this file. Prior art must reach them as MECHANISM, never as an address — ` +
          `a path in the roadmap is how the last run ended up with a byte-identical copy of the ` +
          `answer. Verify's provenance check will reject any patch that imports from it.`);
    }
  }
}

// perf_knowledge pointers resolved by the TechLead in analyze (REFERENCE ONLY; threaded to the
// planner + engineers so they read focused op/language cards instead of the whole base). Empty when
// no operator card applies (e.g. point-cloud HIP ops) or KERNEL_KNOWLEDGE_DIR is unset → no change.
const KK_OPERATOR = (analysis && analysis.kk_operator) || '';
const KK_LANGUAGE = (analysis && analysis.kk_language) || '';
const KK_REFS = (analysis && Array.isArray(analysis.kk_refs)) ? analysis.kk_refs : [];

// ===========================================================================
// PHASE: Benchmark setup (Benchmark Engineer)
// ===========================================================================
phase('Benchmark');
const bench = await agentT(
  roleAgent('benchmark_engineer', 'setup', 'Build the COMMANDMENT and record a reliable baseline.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
    ANALYSIS: analysis,
    TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
    STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
    ...(WORKLOAD_SPEC_PATH ? { WORKLOAD_SPEC_PATH } : {}),
    ...(WORKLOAD_SPEC ? { WORKLOAD_SPEC } : {}),
    ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}),
  }),
  { phase: 'Benchmark', label: 'benchmark_engineer', schema: BENCH_SCHEMA });
// A bench agent that dies without returning has usually NOT failed to measure — it has failed to
// report. Workflow scripts cannot read the filesystem, so an unreturned baseline that is sitting in
// EVAL_DIR is invisible here and used to abort the run outright. On 2026-08-21 that discarded 40
// baseline runs plus a COMPLETE 6-pair positive control (~70 min of an 8-card lease) because the
// agent was interrupted mid-correctness, after every number was already written to setup_ab_*.json.
// So before throwing, spend one cheap agent asking whether the measurements exist on disk. It may
// only RECOVER — re-measuring here would silently double the phase's cost and hide the failure.
let benchR = bench;
if (!benchR || !benchR.baseline_per_case) {
  log('Benchmark setup returned nothing. Attempting RECOVERY from EVAL_DIR before aborting — an ' +
      'agent that died mid-phase usually left its measurements on disk.');
  benchR = await agentT(
    roleAgent('benchmark_engineer', 'recover',
      'The previous benchmark_engineer for this EVAL_DIR did not return. RECOVER ONLY: read ' +
      `${EVAL_DIR} — baseline_timing.json, COMMANDMENT.md, setup_ab_*.json, ab_logs/ — and ` +
      'reconstruct the return JSON from what is already there. Do NOT run any GPU command and do ' +
      'NOT take a lease: this step exists to avoid re-measuring, so a fresh measurement here is a ' +
      'failure, not a fallback. Write baseline_timing.json / COMMANDMENT.md if measurements exist ' +
      'but the file does not. A setup_ab_control*.json with claim_complete:true IS the positive ' +
      'control result — report its measured median and null arm rather than re-running it. If the ' +
      'measurements genuinely are not on disk, return baseline_per_case: [] and say so in notes.',
      { WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, ANALYSIS: analysis,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
        STRICT_AUTONOMY, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
        ...(POSITIVE_CONTROL ? { POSITIVE_CONTROL } : {}) }),
    { phase: 'Benchmark', label: 'benchmark_recover', schema: BENCH_SCHEMA });
  if (!benchR || !Array.isArray(benchR.baseline_per_case) || !benchR.baseline_per_case.length) {
    throw new Error('Benchmark setup failed: no baseline recorded, and none recoverable from EVAL_DIR');
  }
  log(`Benchmark RECOVERED from disk: ${benchR.baseline_per_case.length} cases. ${benchR.notes || ''}`);
}
const BASELINE_PER_CASE = benchR.baseline_per_case;
const BASELINE_GEOMEAN_MS = benchR.baseline_geomean_ms;
log(`Benchmark done. ${benchR.num_test_cases || BASELINE_PER_CASE.length} cases, baseline geomean ${BASELINE_GEOMEAN_MS} ms, reliable=${benchR.reliable}`);

// Positive-control gate. Runs BEFORE any direction budget is spent, because the thing it can prove
// false — "this harness can detect a real win" — invalidates every measurement taken after it.
let PC_OVERSHOOT = '';   // set when the control passed but read HIGH; travels into the report
let PC_UNDERSHOOT = '';  // set when a CONSTRUCTED control passed but read LOW; ditto
// Did a control actually run in THIS wave? Not "was one configured" — a configured control whose
// switch was absent ran two identical arms. objectiveVerdict reads this to decide whether an
// uncontrolled wave's timings are admissible at all.
let PC_RAN = false;
if (POSITIVE_CONTROL) {
  const pc = benchR.positive_control || {};
  const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
  const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
  const got = Number(pc.measured_pct);
  // The two bounds are NOT symmetric, because they fail for opposite reasons.
  //
  //   below `lo`  -> the instrument CANNOT SEE the effect. This is the failure the gate was built
  //                  for: it makes every later number, 1.000x included, uninterpretable. HARD ABORT.
  //   above `hi`  -> the instrument saw the effect and read it BIGGER than recorded. That is a scale
  //                  concern, not a blindness one, and a run that can over-read a 4.7% win can still
  //                  resolve a 1.5% one. Treat as PASS WITH OVERSHOOT and carry the caveat forward.
  //   absurd/wrong sign -> the instrument is measuring something else entirely. HARD ABORT.
  //
  // This distinction was learned the expensive way. A control reproduced a known +4.71% effect at
  // +5.13%, 5/5 pairs favourable, every pair in [4.26, 6.01], against a +0.38% null arm inside the
  // guard's 0.66% noise threshold — i.e. it demonstrated exactly the sensitivity the step exists to
  // demonstrate — and a symmetric band killed the whole run over 0.20pp. The old `hi` had been set
  // to max-observed + 0.2pp from three same-week measurements (4.57/4.71/4.73), which is a
  // reproduction interval, not a sanity bound. A fourth measurement on a different day landing
  // 0.4pp out is ordinary; a band that tight was measuring the weather.
  //
  // Everything below is computed on MAGNITUDE plus an expected SIGN, not on the raw number, so that
  // a control can be a deliberate SLOWDOWN. A control whose `how` points at a finished optimization
  // is the convenient case, not the general one: most runs have no known win lying around, and in a
  // capability evaluation one that does is a hazard, because the control workspace is the answer
  // applied. The gate's question — "can this loop resolve an effect of the size we are hunting?" —
  // is answered just as well by an injected known cost, and an instrument blind to a deliberate
  // slowdown is blind to a real speedup. So `expected_pct_lo/hi` may both be negative; the caller
  // states the direction by their sign and this code stops caring which way it points.
  // (See benchmark_engineer.md 5b, "When no known-good change exists".)
  // <<REPLAY:pc_gate>>  scripts/replay_runs.js lifts everything between these markers verbatim and
  // re-decides recorded controls with it. Keep the region PURE: it may read only `lo`, `hi`, `got`,
  // `pc` and `POSITIVE_CONTROL`, and it must not log, throw, or touch anything outside itself.
  const mLo = Math.min(Math.abs(lo), Math.abs(hi));       // smallest effect the loop must resolve
  const mHi = Math.max(Math.abs(lo), Math.abs(hi));
  const wantSign = Math.sign(lo + hi) || 1;               // +1 = control is faster, -1 = slower
  const mGot = Math.abs(got);
  const ABSURD = Number.isFinite(Number(POSITIVE_CONTROL.implausible_pct))
    ? Math.abs(Number(POSITIVE_CONTROL.implausible_pct)) : (Number.isFinite(mHi) ? 2 * mHi : NaN);
  const nullArm = Number(pc.null_arm_pct);
  // An overshoot is only benign if the null arm is quiet. If the byte-identical arm is ALSO reading
  // high, the interleave is drifting and the overshoot is the drift, not the effect.
  const nullQuiet = Number.isFinite(nullArm) && Math.abs(nullArm) <= mLo / 2;
  // PRE-FLIGHT, AND IT OUTRANKS THE READING. If the knob `how` names is not in the tree the control
  // ran in, both arms executed the same code and `measured_pct` is a null, not a control — a
  // well-formed number with nothing behind it, which is strictly worse than a missing one because
  // it passes every check that looks only at the number. `switch_present: false` therefore means
  // the control DID NOT RUN, whatever came back. Reported explicitly rather than inferred from a
  // small reading, because a real control can legitimately read small and an absent one can
  // legitimately read large (drift). Undefined keeps the old behaviour: runs that never reported
  // the field are decided exactly as before, so replay of historical controls is unaffected.
  const switchAbsent = pc.switch_present === false;
  const ran = pc.ran !== false && !switchAbsent &&
    Number.isFinite(got) && Number.isFinite(lo) && Number.isFinite(hi);

  // A CONTROL'S EXPECTED MAGNITUDE IS EITHER A MEASUREMENT OR A GUESS, AND THE TWO CANNOT BE GATED
  // THE SAME WAY.
  //
  // The overshoot rule above already concedes half of this: a band set from three same-week
  // reproductions is "a reproduction interval, not a sanity bound". The same is true underneath, and
  // it is MORE true, because the floor is where a band gets set by arithmetic on a workload nobody
  // has ever run. `magnitude: 'constructed'` marks that case — a synthetic control (benchmark_
  // engineer.md 5b), where `expected_pct_lo/hi` is a TARGET the engineer aimed a knob at, not an
  // effect anyone has recorded. A recorded 4.7% win that reads 2.3% is an instrument problem. An
  // injected cost aimed at 3.4% that lands at 2.3% is a KNOB-SIZING problem, and the instrument that
  // measured it is the one piece of the experiment that demonstrably worked.
  //
  // This was learned on 2026-08-22, wave 7. An engineer with no known-good change available built the
  // synthetic slowdown the task asks for, calibrated it on a dose ladder (spin 50/200/800 -> +6.8 /
  // +36 / +175%, monotone), extrapolated linearly to spin=25 for ~3.4%, and measured -2.30%: 6 of 6
  // pairs negative, range -1.58..-3.07, against a -0.04% null arm whose worst pair was 0.35%. Then
  // they reported the 0.2pp shortfall in plain text instead of retrying or widening the band — the
  // exact behaviour this workflow asks for everywhere else — and the gate killed the run for it.
  // s_sleep is sublinear at small counts; the extrapolation was optimistic. Nothing about that says
  // the loop cannot see 2.5%. It had just seen 2.30% at ~7x its own null spread.
  //
  // So a constructed control that UNDERSHOOTS its target is admissible, but only on evidence that
  // the reading is an effect and not noise, which is the question the gate actually asks:
  //   - it must still clear a floor, so an injection that never took effect (wrong binary, JIT cache
  //     collision -- the failure knowledge/jit_arm_isolation.md exists for) still hard-aborts;
  //   - it must be RESOLVED: at least RESOLVE_K times the worst null pair, not the null median. A
  //     median hides the spread, and the spread is what an effect has to beat;
  //   - every pair must agree in sign. A median can come out of pairs that disagree; a real injected
  //     cost does not sometimes make the kernel faster.
  // A recorded control gets none of this latitude: its magnitude is a fact, and reading half of a
  // fact is the instrument's fault.
  const constructed = String(POSITIVE_CONTROL.magnitude || 'recorded') === 'constructed';
  const UNDERSHOOT_FRAC = 0.5;   // below half the target, assume the injection did not take
  const RESOLVE_K = 3;           // effect must beat the worst null pair by this factor
  const nullPairs = Array.isArray(pc.null_pairs_pct) ? pc.null_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const nullWorst = nullPairs.length ? Math.max(...nullPairs.map(Math.abs))
    : (Number.isFinite(nullArm) ? Math.abs(nullArm) : NaN);
  const ctrlPairs = Array.isArray(pc.control_pairs_pct) ? pc.control_pairs_pct.map(Number).filter(Number.isFinite) : [];
  const signUnanimous = ctrlPairs.length >= 3 && ctrlPairs.every((d) => Math.sign(d) === wantSign);
  const resolvedByScale = Number.isFinite(nullWorst) && mGot >= RESOLVE_K * nullWorst;

  // RESOLUTION ON A NULL THAT IS NOT UNIMODAL.
  //
  // `RESOLVE_K x the worst null pair` is the right test when the null scatters around a centre: the
  // worst of a handful of pairs is then a fair stand-in for the spread. It is the WRONG test when
  // the null is a mixture. Some guards sit in one of two discrete states run to run, and one
  // excursion into the slow state -- an additive host-side cost that lands on whichever arm happens
  // to draw it -- sets `nullWorst` by itself. The rule then asks the effect to beat three times a
  // single draw from the tail of a distribution it is not competing with, and a large, perfectly
  // clean effect fails.
  //
  // Measured on 2026-08-23: the 512_rank-mixed-skew guard, same tree against itself, 10 pairs. Nine
  // pairs inside 3pp, one at 9.02pp. A candidate measured over the same 10 interleaved reps read
  // +17.34% median, 10/10 pairs positive, range +14.30..+18.65 -- and 17.34/9.02 = 1.9x, a FAIL.
  // Yet the two arms' raw readings did not overlap at all (base 0.8364..0.8832 ms, candidate
  // 0.7014..0.7537 ms). The one slow-state excursion that did land, landed on the candidate arm and
  // made the effect look SMALLER. There is no reading of that data in which the effect is drift.
  //
  // So add a second, distribution-free way to be resolved: the effect pairs and the null pairs, by
  // magnitude, do not overlap. Under the null hypothesis that all n+m pairs are draws from one
  // distribution, the chance of the two groups separating completely is 2/C(n+m, n) -- 7.9e-3 at
  // 5-vs-5, 1.1e-5 at 10-vs-10. That is a rank test, so a single fat-tailed draw cannot break it,
  // and it needs nothing the engineer is not already required to report.
  //
  // This is deliberately NOT "sign unanimity is enough". The A/B driver measured a ~0.6% ordering
  // bias on this operator -- the arm that runs second reads slow -- and a bias like that produces
  // unanimous signs with no effect at all. Separation is immune to it in a way unanimity is not:
  // a 0.6pp bias cannot lift every effect pair above every null pair when the null itself spans
  // several pp. Both routes still have to clear `mLo * UNDERSHOOT_FRAC`, so an injection that never
  // took effect is caught either way.
  const absCtrl = ctrlPairs.map(Math.abs);
  const absNull = nullPairs.map(Math.abs);
  const resolvedBySeparation = absCtrl.length >= 5 && absNull.length >= 5 &&
    Math.min(...absCtrl) > Math.max(...absNull);
  const resolved = resolvedByScale || resolvedBySeparation;
  const undershoot = ran && constructed && Math.sign(got) === wantSign &&
    mGot < mLo && mGot >= mLo * UNDERSHOOT_FRAC && resolved && signUnanimous;

  // Wrong sign is an insensitivity failure, not an overshoot: the loop did not see the change it was
  // handed, whatever else it saw.
  const tooSmall = ran && (Math.sign(got) !== wantSign || (mGot < mLo && !undershoot));
  const absurd = ran && Number.isFinite(ABSURD) && mGot > ABSURD;
  const overshoot = ran && mGot > mHi && !absurd;
  const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
  // <</REPLAY:pc_gate>>
  PC_RAN = ran;
  if (switchAbsent) {
    log(`POSITIVE CONTROL SWITCH ABSENT: "${POSITIVE_CONTROL.name || 'unnamed'}" names a knob that ` +
        `is not in the tree it was run against, so both arms executed identical code and the ` +
        `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'reported'} reading is a base-vs-base null. ` +
        `This is a LAUNCH-ARGS defect, not an instrument failure — fix \`positive_control.how\` to ` +
        `name a switch that exists, or have the benchmark engineer construct a synthetic control ` +
        `(benchmark_engineer.md 5b) and declare magnitude:'constructed'.`);
  }
  log(`Positive control "${POSITIVE_CONTROL.name || 'unnamed'}": ` +
      `measured ${Number.isFinite(got) ? got.toFixed(2) + '%' : 'NOT RUN'}, ` +
      `expected ${lo}..${hi}% (absurd above ${Number.isFinite(ABSURD) ? ABSURD.toFixed(2) + '% in magnitude' : '?'}) ` +
      `-> ${ok ? (overshoot ? 'PASS (OVERSHOOT)' : undershoot ? 'PASS (UNDERSHOOT)' : 'PASS') : 'FAIL'}` +
      (Number.isFinite(nullArm) ? ` (null arm ${nullArm.toFixed(2)}%)` : ' (null arm UNREPORTED)'));
  if (ok && undershoot) {
    // Same treatment as an overshoot, opposite direction: the run may proceed, and every number it
    // reports carries the caveat. A loop that under-reads a known cost by this much may under-read a
    // real win by the same factor — which matters most at the margin, where a win just inside a
    // guard's noise floor is the difference between a result and nothing.
    PC_UNDERSHOOT = `the positive control read ${got.toFixed(2)}% for a CONSTRUCTED effect targeted at ` +
      `${lo}..${hi}%, with a ${Number.isFinite(nullArm) ? nullArm.toFixed(2) : '?'}% null arm ` +
      `(worst null pair ${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp) and ` +
      `${ctrlPairs.length}/${ctrlPairs.length} pairs agreeing in sign. The effect is ` +
      `${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '>>'} the worst ` +
      `null pair, so the loop resolves it; the shortfall against the target is a sizing miss in the ` +
      `injected knob, which was never measured before this run. Treat every effect below as possibly ` +
      `reading LOW by roughly ${(mLo - mGot).toFixed(2)}pp at this size — a marginal win may be real, ` +
      `and a marginal loss may be worse than it looks.`;
    log(`POSITIVE CONTROL UNDERSHOOT: ${PC_UNDERSHOOT}`);
  }
  if (ok && overshoot) {
    // Not a failure, but it must not vanish either: it is a standing caveat on every number the run
    // goes on to report, and the report has to say so.
    PC_OVERSHOOT = `the positive control read ${got.toFixed(2)}% for a change recorded at ` +
      `${lo}..${hi}%, with a ${nullArm.toFixed(2)}% null arm. The harness resolves the known effect ` +
      `with the right sign and clean separation, so it is sensitive enough to trust; but it reads ` +
      `HIGH by roughly ${(mGot - mHi).toFixed(2)}pp at this effect size, so treat every speedup below ` +
      `as possibly carrying the same upward scale bias.`;
    log(`POSITIVE CONTROL OVERSHOOT: ${PC_OVERSHOOT}`);
  }
  if (!ok) {
    const why = `Positive control FAILED: a change with a known effect of ${lo}..${hi}% measured ` +
      `${Number.isFinite(got) ? got.toFixed(2) + '%' : 'nothing (control did not run)'}. ` +
      (absurd
        ? `That is beyond the ${ABSURD.toFixed(2)}% plausibility ceiling — at that size the harness ` +
          `is not measuring this change, it is measuring something else (wrong arm, wrong guard, ` +
          `wrong binary). `
        : overshoot
          ? `It overshot the expected ceiling AND the null arm is ` +
            `${Number.isFinite(nullArm) ? Math.abs(nullArm).toFixed(2) + '%, too large relative to the ' + mLo + '% effect' : 'UNREPORTED'}, ` +
            `so the excess cannot be told apart from interleave drift. `
          : (Math.sign(got) !== wantSign && Number.isFinite(got)
              ? `It came back with the WRONG SIGN, so the loop is not tracking the change it was ` +
                `handed — a wiring error (arms swapped, wrong guard, wrong binary) before it is a ` +
                `sensitivity one. `
              : ``) +
            `The measurement loop cannot resolve an effect of the size this run is looking for, so ` +
            `every number it reports — including 1.000x — is uninterpretable. Fix the harness, not ` +
            `the kernel. `) +
      // WHICH THING TO FIX. An under-reading control has two possible causes and they call for
      // opposite work, so the abort must not leave the reader to guess. Say which of the undershoot
      // conditions was missed, in the reader's own numbers.
      (tooSmall && Math.sign(got) === wantSign && mGot < mLo
        ? `This is an UNDER-read, so before touching the harness, establish which of the two it is. ` +
          (!constructed
            ? `This control is declared \`magnitude: 'recorded'\`, meaning ${lo}..${hi}% is an effect ` +
              `someone has actually measured — so reading ${mGot.toFixed(2)}% IS the instrument's fault. ` +
              `If instead that band was a target you aimed a synthetic knob at and never measured, the ` +
              `control is \`magnitude: 'constructed'\` and should be declared as such; do NOT relabel a ` +
              `recorded effect to get past this gate, that falsifies the experiment. `
            : mGot < mLo * UNDERSHOOT_FRAC
              ? `It read under half the target (${(mLo * UNDERSHOOT_FRAC).toFixed(2)}%), which is the ` +
                `signature of an injection that never took effect — arms sharing a JIT cache entry ` +
                `(knowledge/jit_arm_isolation.md), the env var unread, the gated code not on the hot ` +
                `path. Verify the two arms have distinct disk_keys before blaming the loop. `
              : !signUnanimous
                ? `Its ${ctrlPairs.length} pair(s) do not all agree in sign, so the median is coming out ` +
                  `of a disagreement rather than an effect — that IS a resolution failure. `
                : !resolved
                  ? `It is only ${Number.isFinite(nullWorst) && nullWorst > 0 ? (mGot / nullWorst).toFixed(1) + 'x' : '?'} ` +
                    `the worst null pair (${Number.isFinite(nullWorst) ? nullWorst.toFixed(2) : '?'}pp), under the ` +
                    `${RESOLVE_K}x an effect must clear to be told apart from the interleave, and its ` +
                    `pairs do not separate from the null's either ` +
                    `(smallest control pair ${absCtrl.length ? Math.min(...absCtrl).toFixed(2) : '?'}pp ` +
                    `vs largest null pair ${absNull.length ? Math.max(...absNull).toFixed(2) : '?'}pp` +
                    `${absCtrl.length >= 5 && absNull.length >= 5 ? '' : '; separation needs >=5 pairs on BOTH arms'}), ` +
                    `so neither route to resolution is open. Quiet the interleave or raise the injected ` +
                    `cost. If the null is fat-tailed rather than noisy -- a few readings well above an ` +
                    `otherwise tight cluster -- collect more null pairs and check whether the two arms ` +
                    `separate; that route does not care about the tail. `
                  : ``)
        : ``) +
      `Note from benchmark engineer: ${pc.note || '(none)'}`;
    if (PC_ABORT) throw new Error(why);
    log(`WARNING: ${why}`);
  }
}

// ===========================================================================
// PHASE: Baseline profile (Profile Engineer)
// ===========================================================================
phase('Profile');
let profileSummary = await agentT(
  roleAgent('profile_engineer', 'baseline', 'Profile the baseline and classify the bottleneck.', {
    WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0), ROUND: 0,
    COMMANDMENT,
    ...RESUME_INPUT,
  }),
  { phase: 'Profile', label: 'profile_engineer:baseline', schema: PROFILE_SCHEMA });
const baselineAnalysisResult = await runProfileAnalysis(
  profileSummary,
  0,
  'analysis_engineer:baseline',
);
if (profileSummary) {
  profileSummary = {
    ...profileSummary,
    analysis_result: baselineAnalysisResult,
  };
}
log(`Baseline bottleneck: ${profileSummary ? profileSummary.bottleneck : '?'} (dispatch_count=${profileSummary ? profileSummary.dispatch_count : '?'})`);

// ===========================================================================
// PHASE: Optimization loop (budget-controlled)
// ===========================================================================
let dispatched = 0;          // counts ONLY optimization-direction engineers (the budget)
let round = 0;
let cumulative = 1.0;        // best verified geomean speedup vs the TRUE baseline
let noImprove = 0;
// Consecutive rounds that produced no admissible measurement. Separate from `noImprove` because it
// means the opposite thing: `noImprove` says the search is not finding wins, `noEvidence` says the
// instrument is not returning readings. Both stop the loop at MAX_NO_IMPROVE, and they must not be
// summed — see roundEvidence.
let noEvidence = 0;
// Consecutive rounds in which the switched/perf path never reached a card. ORTHOGONAL to both counters
// above and armed under EVERY objective (including working_kernel), because the thing it catches is not
// "no win" and not "no reading" but "no on-device execution at all". working_kernel deliberately exempts
// itself from the no-improve/no-evidence stops — a debug round is non-improving by construction and a
// crashing-on-device terminal round is real progress that produces no speed number — but a round whose
// candidate never traced+launched on hardware is neither of those; it is scaffolding banked on static
// screens, which is exactly the pathology that let a fused arm defer its ON-path crash for 8 rounds. So
// this stop stays live in working_kernel. A round that reached the card (even to crash at runtime) resets
// it; only a run genuinely stuck never-reaching-hardware trips it. Inert unless REQUIRE_HW_ACTIVATION.
let noHardware = 0;
// Decoupled from MAX_NO_IMPROVE and independently overridable via `max_no_hardware`. Default stays
// MAX_NO_IMPROVE so every existing campaign is unchanged. The reason for the knob: authoring ONE large
// graph-safe fused megakernel as a single unit (whole-fusion-first, not an incremental single-edge
// ladder) legitimately spends several early rounds on trace-time / deadlock crash-debug that never
// reach launch, and a cap of 2-3 hard-stops that authoring before the terminal ON-path ever lands.
// Raising the cap does not defeat the stop's purpose — it still terminates a run that is genuinely
// stuck never-reaching-hardware, just with room for a big authoring push. Any round that reaches the
// card (even to crash at runtime) still resets it to 0.
const MAX_NO_HARDWARE = Math.max(1, parseInt(A.max_no_hardware != null ? A.max_no_hardware : MAX_NO_IMPROVE, 10));
// Why the round loop ended. Written down because nothing used to write it down: reconstructing the
// stop condition of one finished wave took a full session of reading code against artifacts, and
// the answer ("the counter reached its cap while a lease was still in the budget") was a defect
// that had been invisible for four waves.
let stopReason = null;
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// The blackboard behind `history.insights`. Kept as records rather than strings so an insight can
// carry the round it came from and whether that round produced evidence at all.
let insightBook = [];
// The candidate shelf: verified non-winners, kept as offers. `absorbedByRound[r]` is the file set
// CANONICAL took on in round r, which is what ages a shelved patch out. See the candidate_shelf
// region below.
let shelf = [];
let absorbedByRound = {};
const SHELF_MAX = 24;
// K, the number of historical candidates offered to the integrator alongside this round's own.
// DELIBERATELY SMALL, and the reason is cost, not caution: one 8-GPU collective lease per round
// means every extra offer the integrator takes seriously is verification time that came out of the
// round's own directions. Start at 2 and size it off the measured hit rate — if shelved candidates
// are offered for several waves and never successfully combined, that null result is the finding
// and K should go to 0, not up. Guessing K high converts a cheap experiment into an expensive one
// before anyone has seen it work once.
const SHELF_OFFER_K = Number(A.shelf_offer_k) > 0 ? Number(A.shelf_offer_k) : 2;
// The evidence K gets resized from. See the report dispatch for why it is reported even when it is
// all zeros.
const SHELF_STATS = { k: SHELF_OFFER_K, shelved: 0, withheld_unknown_footprint: 0,
  rounds_offered: 0, offers: 0, hits: 0, hit_ids: [] };

// <<REPLAY:memory_merge>>
// CROSS-ROUND MEMORY. This used to be one line — `history.insights = mem.insights` — and that line
// was the largest source of context loss in the loop. `update_memory` asks an agent to "distill
// durable insights"; an agent handed one round's results returns that round's insights, and the
// assignment threw away everything earlier that it happened not to restate. Round 3's engineers
// therefore re-proposed round 1's dead directions on a board that no longer remembered them, and
// the loop paid for the same experiment twice out of a budget counted in leases.
//
// So the board is APPEND-AND-AGE, never replaced. Three properties it has to have:
//
//   * an insight the summariser stops repeating SURVIVES. Silence from a summariser is not evidence
//     that a finding expired; it is evidence the summariser was looking at a different round.
//   * every insight carries its ORIGIN ROUND, and re-stating it cannot launder that away. A claim
//     distilled from a round where nothing executed is not the same kind of object as one measured
//     against the baseline, and the two must not become indistinguishable through re-summarisation.
//   * eviction, when the board is full, is REPORTED. A blackboard that silently drops its oldest
//     entries reproduces the original bug more slowly.
function normInsight(s) {
  return String(s == null ? '' : s)
    .replace(/^\[r\d+[^\]]*\]\s*/, '')     // strip a provenance tag if the agent echoed one back
    .replace(/\s+/g, ' ').trim().toLowerCase();
}
function mergeInsights(book, incoming, round, roundVoid, max) {
  const MAX = Number.isFinite(max) ? max : 40;
  const out = book.map((e) => Object.assign({}, e));
  const index = new Map(out.map((e, i) => [normInsight(e.text), i]));
  for (const raw of (Array.isArray(incoming) ? incoming : [])) {
    const text = String(raw == null ? '' : raw).replace(/^\[r\d+[^\]]*\]\s*/, '').trim();
    if (!text) continue;
    const k = normInsight(text);
    if (index.has(k)) {
      // Restating an insight refreshes it and is what keeps a still-relevant finding alive under
      // ageing — but it cannot upgrade a finding that came out of a round with no evidence.
      const e = out[index.get(k)];
      e.last_round = round; e.restated = (e.restated || 0) + 1;
      if (!roundVoid) e.void_round = e.void_round && false;
      continue;
    }
    out.push({ text, first_round: round, last_round: round, restated: 0, void_round: !!roundVoid });
    index.set(k, out.length - 1);
  }
  if (out.length <= MAX) return { book: out, evicted: [] };
  // Age out by last-seen round, then by how often the finding kept coming back. Ties break toward
  // the more recent entry.
  const ranked = out.map((e, i) => ({ e, i })).sort((a, b) =>
    (b.e.last_round - a.e.last_round) || ((b.e.restated || 0) - (a.e.restated || 0)) || (b.i - a.i));
  const keep = new Set(ranked.slice(0, MAX).map((x) => x.i));
  return { book: out.filter((_, i) => keep.has(i)), evicted: out.filter((_, i) => !keep.has(i)) };
}
function renderInsights(book) {
  return book.map((e) => `[r${e.first_round}` +
    (e.last_round !== e.first_round ? `-r${e.last_round}` : '') +
    (e.void_round ? ' FROM-VOID-ROUND' : '') + `] ${e.text}`);
}
// What has already been tried, derived from the round log rather than from an agent's recollection
// of it. Engineers were never shown this: they got the insight list, which says what was LEARNED,
// and nothing that says what was ATTEMPTED. Those differ exactly where it matters — a direction that
// failed leaves an insight only if someone bothered to write one.
//
// A VOID direction is listed separately and in the opposite sense. An inactive patch was never
// tried, so filing it as a dead end would suppress the one experiment the round still owes.
function deadEnds(rounds) {
  const tried = [], untried = [];
  for (const r of (Array.isArray(rounds) ? rounds : [])) {
    const titles = new Map((r.directions || []).map((d) => [d.id, d]));
    for (const res of (r.results || [])) {
      const d = titles.get(res.id) || {};
      const label = `r${r.round} "${d.title || res.id}" (${d.specialty || '?'})`;
      if (res.inactive) {
        untried.push(`${label}: VOID — the patched path was ${res.inactive === 'no' ? 'not executed' : 'not proven to execute'}. ` +
          `NOT a dead end: this direction has not actually been tested. Re-run it with activation proven.`);
      } else if (res.status === 'apply_failed') {
        untried.push(`${label}: patch did not apply — untested, not disproven.`);
      } else if (Number.isFinite(res.verified) && res.verified > 0) {
        tried.push(`${label}: verified ${res.verified.toFixed(3)}x${res.verified < 1.005 ? ' — no gain' : ''}`);
      } else {
        tried.push(`${label}: ${res.status || 'no result'}`);
      }
    }
  }
  return { tried, untried };
}
// <</REPLAY:memory_merge>>

// <<REPLAY:candidate_shelf>>
// THE SHELF — verified candidates that did not win their round, kept as offers instead of as numbers.
//
// Before this, a round kept exactly one thing: the winner, committed into CANONICAL. Every other
// candidate that passed INDEPENDENT verification survived only as `{id, claimed, verified, status}`
// in the round record — a number, not a patch. So a direction that verified at 1.03x in round 2 and
// lost to a 1.09x was gone: round 5 could not combine with it, could not build on it, and would
// cheerfully re-derive it. On this task the budget is 8 rounds of an 8-GPU collective, one lease per
// round, so re-deriving a finished result is the most expensive mistake on the menu.
//
// WHAT MAKES A SHELVED PATCH STILL APPLICABLE is the whole difficulty. A patch is a diff against the
// CANONICAL that existed when it was cut, and every round that commits a winner moves CANONICAL out
// from under everything on the shelf. So each entry records the round it was cut against, and
// eligibility is decided by FILE SETS: if nothing absorbed since then touches a file this patch
// touches, it still applies and can be offered for combination.
//
// That test is mechanical ON PURPOSE. The alternative — asking the agent "does your patch conflict
// with that one?" — is the failure this workflow keeps rediscovering: a question whose unanswered
// form has a benign default. Anything derivable from the artifacts is derived, not asked.
//
// AN EMPTY FILE SET IS NOT ORTHOGONALITY. A verified candidate reporting no touched files has an
// UNKNOWN footprint, and unknown must not read as "conflicts with nothing", which is precisely the
// maximally-combinable answer. Those are held back and named, never silently offered.
const shelfFile = (s) => String(s == null ? '' : s).replace(/^\.\//, '').replace(/^\/+/, '').trim();
const shelfFiles = (c) => [...new Set((Array.isArray(c.touched_files) ? c.touched_files : [])
  .map(shelfFile).filter(Boolean))];

function shelfAdd(shelf, entries, round, max) {
  const out = shelf.slice();
  for (const e of entries) {
    const files = shelfFiles(e);
    const row = {
      id: e.id, title: e.title || '', specialty: e.specialty || '',
      geomean: Number(e.geomean) || 0, patch: e.patch || '', files,
      // The fusion chain this entry belongs to, identified by the TERMINAL rung it closes on (both
      // halves of a chain share it: an enabling step names it in `enables`, a terminal step in
      // `roadmap_rung`). Carried so shelfEligible can tell "your own producer touched these files"
      // -- which is expected, because the two halves of a fusion edit the same files by construction
      // -- apart from "an unrelated winner touched these files", which is what actually staleness a
      // patch. Empty for a standalone optimisation, which then ages against every absorption as before.
      chain_id: String(e.chain_id || '').trim(),
      // Three-valued on purpose, same reason `artifact_distinct` is: "we could not tell" has to stay
      // sayable, or the only expressible answer is the one that flatters the candidate.
      footprint: files.length ? 'known' : 'unknown',
      base_round: round, shelved_round: round, absorbed: false,
    };
    const at = out.findIndex((x) => x.id === e.id);
    if (at >= 0) out[at] = { ...out[at], ...row }; else out.push(row);
  }
  out.sort((a, b) => b.geomean - a.geomean);
  return { shelf: out.slice(0, max), evicted: out.slice(max) };
}

// `absorbedByRound` maps round number -> what CANONICAL took on in that round. Each value is EITHER a
// plain file array (a standalone commit, no chain) OR `{files, chain_id}` (a commit that was part of
// a fusion chain). The object form is what lets a chain's own producer commit stop aging the chain's
// own consumer half off the shelf.
function shelfEligible(shelf, absorbedByRound, k) {
  const norm = (v) => Array.isArray(v)
    ? { files: v, chain_id: '' }
    : { files: (v && v.files) || [], chain_id: String((v && v.chain_id) || '').trim() };
  const eligible = [], stale = [], unknown = [];
  for (const e of shelf) {
    if (e.absorbed) continue;
    if (e.footprint !== 'known') { unknown.push(e); continue; }
    const moved = new Set();
    for (const r of Object.keys(absorbedByRound || {})) {
      if (Number(r) <= e.base_round) continue;
      const a = norm(absorbedByRound[r]);
      // A commit on the SAME chain as this shelf entry does not age it. The producer and consumer
      // halves of a fusion touch the same files by construction, so file-overlap with your own
      // chain's earlier half is the intended relationship, not a conflict -- aging on it is exactly
      // what left the consumer half permanently stale and the fusion permanently half-built.
      if (a.chain_id && e.chain_id && a.chain_id === e.chain_id) continue;
      for (const f of a.files) moved.add(shelfFile(f));
    }
    const clash = e.files.filter((f) => moved.has(f));
    if (clash.length) stale.push({ ...e, clash }); else eligible.push(e);
  }
  eligible.sort((a, b) => b.geomean - a.geomean);
  return { offer: eligible.slice(0, Math.max(0, k)), eligible, stale, unknown };
}
// <</REPLAY:candidate_shelf>>

// <<REPLAY:overlap_gate>>
// IS THE OVERLAP CLAIM BACKED BY AN INSTRUMENT? See knowledge/overlap_instrument.md.
//
// The acceptance bar asks for genuine compute/communication overlap rather than serialization
// disguised by kernel boundaries. Nothing that exists before fusion can answer that: four launches
// give four trace records and the intervals settle it for free, but ONE fused kernel gives one
// record, and the per-stage timers move the wrong way (a stage allowed to start early charges its
// own waiting to itself, so both can rise while the operator gets faster).
//
// So the fused shape destroys every instrument that would have judged it, and a faster fused kernel
// is evidence of *something* — removed launch overhead, L2 residency across a boundary, a grid the
// scheduler can pack better — of which overlap is only one candidate. This gate exists so that
// "nobody measured it" cannot arrive looking like "measured and fine".
//
// It NEVER fails a candidate. A real latency win with an unmeasured overlap claim is a good result
// with a named hole; downgrading it to a failure would teach the loop to stop reporting the hole.
const OVERLAP_SCATTERED_MAX = 0.05;  // the unfused path's true overlap is 0; 5% is instrument slop.
function overlapVerdict(ver, opts) {
  const o = (ver && ver.overlap) || {};
  const won = Number((opts && opts.geomean) || 0) > 1.0;
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const frac = num(o.fraction), scat = num(o.scattered_reading), forced = num(o.forced_reading);
  const m = o.measured === 'yes' || o.measured === 'no' ? o.measured : 'unknown';

  // The negative control first, because it invalidates everything downstream. A meter that finds
  // overlap on the four-launch path is measuring its own artefacts.
  if (m !== 'unknown' && scat != null && scat > OVERLAP_SCATTERED_MAX) {
    return { state: 'meter_broken', caveat:
      `The overlap meter reads ${(scat * 100).toFixed(1)}% on the SCATTERED path, whose true overlap ` +
      `is zero by construction (four launches, disjoint intervals). The meter is measuring its own ` +
      `artefacts, so its ${frac != null ? `${(frac * 100).toFixed(1)}% ` : ''}reading on the fused ` +
      `path carries no information. Overlap is UNMEASURED for this candidate.` };
  }
  // A meter that has never read a known non-zero is indistinguishable from a dead one.
  if (m !== 'unknown' && scat == null) {
    return { state: 'meter_unvalidated', caveat:
      `Overlap was reported as "${m}" but the meter's negative control is missing: nobody ran it ` +
      `against the unfused path, where the answer is known to be zero. An uncontrolled meter reading ` +
      `is not evidence — the same rule the workflow applies to a benchmark with no positive control.` };
  }
  if (m === 'no' && forced == null) {
    return { state: 'meter_unvalidated', caveat:
      `The meter reports NO overlap and has never read a known non-zero (no forced-concurrency ` +
      `control). A dead meter and a correct "there is no overlap here" produce the same output, and ` +
      `this run cannot tell them apart. Treat the negative finding as unconfirmed.` };
  }
  if (m === 'unknown') {
    return { state: 'unmeasured', caveat: won
      ? `This candidate improves latency but overlap was NOT measured` +
        `${o.note ? ` (${String(o.note).slice(0, 200)})` : ''}. In the fused shape latency cannot ` +
        `distinguish overlap from removed launch overhead or better L2 residency, so this result ` +
        `must not be reported as demonstrating overlap. Name the collection experiment that is missing.`
      : `Overlap was not measured for this candidate; no overlap claim may be made from it either way.` };
  }
  // Measured, controlled. What is left is whether the number and the latency agree.
  if (m === 'yes' && frac != null && frac > 0.05 && !won) {
    return { state: 'contention', caveat:
      `The meter reports ${(frac * 100).toFixed(1)}% overlap but the operator did not get faster. ` +
      `Concurrency without a latency win is CONTENTION — two roles co-resident and competing for the ` +
      `same CUs, issue slots or LDS. That is a finding about the partition, not a partial success.` };
  }
  if (m === 'yes' && (frac == null || frac <= 0.05) && won) {
    return { state: 'win_without_overlap', caveat:
      `A latency win with essentially no measured overlap (${frac == null ? 'fraction unreported' : `${(frac * 100).toFixed(1)}%`}). ` +
      `The win is probably real but it is NOT an overlap result — look for the mechanism (launch ` +
      `overhead, L2 residency, grid shape) and attribute it, or the next round will build on the ` +
      `wrong cause.` };
  }
  if (m === 'no') {
    return { state: 'measured_none', caveat:
      `The meter was built, controlled, and reports no overlap. That is a result about this edge, ` +
      `not a missing measurement.` };
  }
  return { state: 'measured', caveat: '' };
}
// <</REPLAY:overlap_gate>>

// <<REPLAY:attribution_gate>>
// IS THE WIN INSIDE THE KERNEL THAT WAS CHANGED? See knowledge/fusion_preconditions.md.
//
// A launch-structure change can move both the fused body and the waits around it. The task declares
// which one is the score. `operator_e2e` owns the complete operator span and uses attribution to
// explain it; `changed_kernel` is valid only when both arms provide a comparable same-timeline
// kernel measure. Independently rank-reduced stage timers are never silently promoted into that role.
//
// Whether a mismatch rejects depends on the declared promotion metric. For `operator_e2e`, launch
// alignment and downstream wait changes are part of the operator result; attribution is mandatory
// evidence about WHY it won, not a replacement score. For `changed_kernel`, the task explicitly
// owns a comparable same-timeline kernel measure and the mismatch rejects.
const ATTRIBUTION_EPS_PCT = 0.5;  // below this the two readings are the same number

function attributionVerdict(ver, opts) {
  const a = (ver && ver.attribution) || {};
  const won = Number((opts && opts.geomean) || 0) > 1.0;
  const metric = String((opts && opts.metric) || 'changed_kernel');
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const chg = num(a.changed_us), rep = num(a.replaced_sum_us);
  const e2ePct = ((Number((opts && opts.geomean) || 1)) - 1) * 100;

  if (chg == null || rep == null || rep <= 0) {
    if (!won) return { state: 'not_applicable', reject: false, caveat: '' };
    return { state: 'unattributed', reject: false, caveat:
      `This candidate claims ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% end-to-end but did not ` +
      `report the changed kernel's own time against the kernels it replaced. End-to-end moves for ` +
      `reasons that live BETWEEN launches, so on its own it cannot say whether the kernel got ` +
      `faster. Report attribution.changed_us / replaced_sum_us, read paired in one collection, or ` +
      `the number stays unattributed.` };
  }

  // Positive delta = the changed kernel is FASTER than what it replaced.
  const kPct = ((rep - chg) / rep) * 100;
  const resB = num(a.residual_ms_base), resC = num(a.residual_ms_cand);
  const gapNote = (resB != null && resC != null)
    ? ` The residual (end-to-end minus the sum of kernel times) moved ${resB >= 0 ? '+' : ''}` +
      `${resB.toFixed(4)} -> ${resC >= 0 ? '+' : ''}${resC.toFixed(4)} ms, i.e. ` +
      `${Math.abs((resB - resC) * 1000).toFixed(0)}us of the claim is in the gaps.`
    : '';

  if (won && kPct < -ATTRIBUTION_EPS_PCT) {
    const reject = metric === 'changed_kernel';
    return { state: reject ? 'gap_win' : 'mechanism_mismatch', reject, caveat:
      `${reject ? 'REJECTED as a changed-kernel win.' : 'Operator win retained, mechanism claim withheld.'} ` +
      `End-to-end reads ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% but the ` +
      `kernel that was changed is SLOWER than the kernels it replaced: ${chg.toFixed(1)}us against ` +
      `${rep.toFixed(1)}us, ${kPct.toFixed(2)}%.${gapNote} The win is not inside the code under ` +
      `test. Find and measure the mechanism before calling it overlap. Under operator_e2e that ` +
      `cross-launch/cross-rank effect remains part of the product metric; under changed_kernel it is ` +
      `outside the declared score.` };
  }
  if (!won && kPct > ATTRIBUTION_EPS_PCT) {
    return { state: 'kernel_win_e2e_loss', reject: false, caveat:
      `The changed kernel is ${kPct.toFixed(2)}% faster than what it replaced, but end-to-end did ` +
      `not improve.${gapNote} The cost moved rather than disappeared -- into launch structure, ` +
      `arrival skew, or a downstream kernel's wait. This is a real kernel result with a live ` +
      `regression attached; locate it before shipping.` };
  }
  if (won && Math.abs(kPct) <= ATTRIBUTION_EPS_PCT) {
    const reject = metric === 'changed_kernel';
    return { state: reject ? 'gap_win' : 'mechanism_mismatch', reject, caveat:
      `${reject ? 'REJECTED as a changed-kernel win.' : 'Operator win retained, mechanism claim withheld.'} ` +
      `End-to-end reads ${e2ePct >= 0 ? '+' : ''}${e2ePct.toFixed(2)}% while the ` +
      `changed kernel is flat against the kernels it replaced (${chg.toFixed(1)}us vs ` +
      `${rep.toFixed(1)}us, ${kPct >= 0 ? '+' : ''}${kPct.toFixed(2)}%).${gapNote} A flat kernel ` +
      `cannot explain the end-to-end win; report the operator result and withhold the claimed mechanism.` };
  }
  return { state: 'attributed', reject: false, caveat: won
    ? '' : `Kernel time ${kPct >= 0 ? '+' : ''}${kPct.toFixed(2)}%, consistent with the end-to-end reading.` };
}
// <</REPLAY:attribution_gate>>

// <<REPLAY:launch_gate>>
// IS THE FUSION ACTUALLY FUSED? See acceptance criterion 1: "fully fused, TWO launches" — one
// megakernel per EP rank plus the one separate pre-dispatch quant launch. This is the criterion the
// whole campaign exists to reach, and until now nothing in the workflow could see it: a candidate
// that fused dispatch+gemm1 but still launched combine on its own ran three launches and read
// exactly like a real two-launch fusion, because no field counted launches.
//
// Like the overlap gate and unlike the attribution gate, this NEVER rejects. A candidate can be a
// correct, faster kernel and not yet be at the two-launch shape — that is a partial rung, not a
// wrong result, and failing it would teach the loop to stop reporting how many launches it is at.
// What it does is make the count a stated number that travels into the round log and the report, and
// tell the ladder whether the TERMINAL fusion shape has actually been reached (`shape_met`), so a
// wave cannot report "fusion rung closed" on a candidate that is still three launches.
//
// Three-valued on the same principle as overlap.measured: `unjudged` (no count reported) is a HOLE,
// not a pass. Collapsing it into "met" is exactly how an unfused candidate would read as accepted.
function launchVerdict(ver, target, expectedStages) {
  const ls = (ver && ver.launch_shape) || {};
  const num = (x) => (Number.isFinite(Number(x)) ? Number(x) : null);
  const cand = num(ls.launches_cand), base = num(ls.launches_base);
  // The orchestrator/ladder owns the target. A verifier reports what ran; it cannot move the goal
  // from 2 to 3 and make the same observation pass.
  const tgt = num(target) != null ? num(target) : (num(ls.target) != null ? num(ls.target) : 2);
  const stages = Array.isArray(ls.stages_fused) ? ls.stages_fused.map(String) : [];
  const expected = Array.isArray(expectedStages) ? expectedStages.map(String) : [];
  const missingStages = expected.filter((s) => !stages.includes(s));
  const proofMissing = ls.per_rank !== true || !String(ls.how_counted || '').trim() ||
    missingStages.length > 0;
  if (cand == null || !Number.isInteger(cand) || cand <= 0) {
    return { state: 'unjudged', shape_met: false, caveat:
      `Launch count NOT reported, so acceptance criterion 1 (two launches) is UNJUDGED for this ` +
      `candidate. A fusion that collapsed dispatch+gemm1 but still launches combine separately is ` +
      `three launches and reads identical to a real two-launch fusion here. Report ` +
      `launch_shape.launches_cand (per EP rank, one operator call) with how_counted.` };
  }
  if (expected.length && proofMissing) {
    return { state: 'unjudged', shape_met: false, caveat:
      `Launch count ${cand} was reported without a complete proof: per_rank must be true, ` +
      `how_counted must be non-empty, and stages_fused is missing [${missingStages.join(', ') || 'none'}].` };
  }
  if (cand <= tgt) {
    return { state: 'met', shape_met: true, caveat: base != null && base !== cand
      ? `Launch count ${base} -> ${cand} (target ${tgt}); the two-launch fused shape is reached.` : '' };
  }
  return { state: 'above_target', shape_met: false, caveat:
    `Launch count is ${cand}${base != null ? ` (from ${base})` : ''}, target ${tgt}. The stages are ` +
    `not all fused yet -- this is a partial rung, not a wrong result, but the TERMINAL two-launch ` +
    `shape has NOT been reached and the fusion rung must not be reported as closed on it.` };
}
// <</REPLAY:launch_gate>>

// <<REPLAY:bimodal_split>>
// THE 512 GUARDS ARE NOT NOISY, THEY ARE BIMODAL — and the escape hatch the doctrine prescribes for
// them does not work on them.
//
// Measured on this box, unmodified tree against itself, 10 runs per guard: `8192_uniform` unimodal,
// worst pair 1.09%; `512_uniform` worst pair 6.21%; `512_rank-mixed-skew` worst pair 9.30% with 2 of
// 10 runs sitting ~7-8% above an otherwise tight cluster. Those high runs reproduce to four digits,
// which drift does not do. So the guard occupies one of two discrete states per run, and the excess
// (~0.07-0.10 ms, 9-13% of a 512-token iteration) is additive.
//
// The standing rule is "deep sample, judge by SEPARATION" — two arms whose raw readings do not
// overlap at all are separated in a way no tail draw can undo (10-vs-10, p=1.1e-5 under the null).
// That rule is right, and on a bimodal guard it is UNREACHABLE: both arms draw slow runs, the ranges
// interleave, and separation never happens however deep you sample. The escape hatch is unavailable
// on precisely the two guards it was written for, which is why they have been demoted to "regression
// guards only" for several waves while the effect being hunted (5-6% by Analyze's own envelope) sits
// underneath a 9.30% worst pair.
//
// The way out is not more runs, it is conditioning on the state. Classify each reading, then compare
// like with like, and report the state occupancy separately instead of averaging over it.
//
// TWO RULES KEEP THIS FROM BECOMING A KNOB THAT MANUFACTURES WINS:
//  1. Classification is ARM-BLIND. The split is computed on the POOLED readings with no knowledge of
//     which arm each came from. A per-arm threshold is a free parameter and would let an engineer
//     choose the boundary that flatters the candidate.
//  2. Refusing to classify stays available and is not a failure. If the pooled readings show no
//     clean gap, this is not a bimodal guard today and the analysis falls back to all pairs. A
//     classifier that always finds two clusters would find them in pure noise.
const BIMODAL_MIN_GAP_PCT = 3.0;  // a gap smaller than this is a spread, not two states.
function modeSplit(readings, minGapPct) {
  const gapMin = Number.isFinite(minGapPct) ? minGapPct : BIMODAL_MIN_GAP_PCT;
  const v = readings.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length < 6) return { classified: false, reason: `only ${v.length} readings; need >=6 to see a gap` };
  // The widest relative gap between consecutive sorted readings. Anything else (k-means, a fixed
  // threshold) needs a parameter that is not in the data.
  let at = -1, best = 0;
  for (let i = 0; i < v.length - 1; i++) {
    const g = v[i] > 0 ? ((v[i + 1] - v[i]) / v[i]) * 100 : 0;
    if (g > best) { best = g; at = i; }
  }
  if (best < gapMin) {
    return { classified: false, reason: `widest gap ${best.toFixed(2)}% < ${gapMin}% — one state, not two`,
             gap_pct: best };
  }
  const cut = (v[at] + v[at + 1]) / 2;
  // The slow state must be the minority, or "slow" is just the operating point and the few fast
  // readings are the anomaly. Naming the wrong cluster inverts every conclusion below it.
  const nSlow = v.length - at - 1;
  if (nSlow > v.length / 2) {
    return { classified: false, reason: `the upper cluster holds ${nSlow}/${v.length} readings — that is ` +
             `the operating point, not a tail`, gap_pct: best };
  }
  return { classified: true, cut, gap_pct: best, n_fast: at + 1, n_slow: nSlow };
}

// `pairs` = [{base, cand}, ...] collected interleaved. Returns the mode-aware paired analysis.
function pairedModeAware(pairs, minGapPct) {
  const ok = pairs.filter((p) => Number.isFinite(Number(p.base)) && Number.isFinite(Number(p.cand)));
  const split = modeSplit([...ok.map((p) => p.base), ...ok.map((p) => p.cand)], minGapPct);
  const pct = (p) => ((Number(p.base) - Number(p.cand)) / Number(p.base)) * 100;  // >0 = candidate faster
  const all = ok.map(pct);
  const signTest = (xs) => {
    if (!xs.length) return { n: 0, agree: 0, p: null };
    const pos = xs.filter((x) => x > 0).length, neg = xs.filter((x) => x < 0).length;
    const agree = Math.max(pos, neg);
    // Two-sided sign test at unanimity is what the existing doctrine leans on; report it for any
    // level of agreement so a 9-of-10 is not silently rounded to unanimous.
    return { n: xs.length, agree, p: agree === xs.length ? Math.pow(2, -xs.length + 1) : null };
  };
  const med = (xs) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);

  if (!split.classified) {
    return { conditioned: false, reason: split.reason, gap_pct: split.gap_pct,
             all: { deltas: all, median: med(all), ...signTest(all) } };
  }
  const isSlow = (x) => Number(x) > split.cut;
  const fast = ok.filter((p) => !isSlow(p.base) && !isSlow(p.cand));
  const mixed = ok.filter((p) => isSlow(p.base) !== isSlow(p.cand));
  const bothSlow = ok.filter((p) => isSlow(p.base) && isSlow(p.cand));
  const fastD = fast.map(pct);
  // Occupancy is NOT a nuisance to be divided out. If the candidate lands in the slow state less
  // often than the baseline does, that IS an effect, and on a guard where the slow state costs
  // 9-13% of the iteration it can be the largest effect present. Discarding the slow runs and
  // reporting only the fast-mode delta would throw away a real win as if it were noise.
  const occ = (rows, pick) => {
    const n = rows.length; const k = rows.filter((p) => isSlow(pick(p))).length;
    return { n, slow: k, pct: n ? (k / n) * 100 : null };
  };
  return {
    conditioned: true, cut: split.cut, gap_pct: split.gap_pct,
    fast: { deltas: fastD, median: med(fastD), ...signTest(fastD) },
    all: { deltas: all, median: med(all), ...signTest(all) },
    dropped: { mixed_mode_pairs: mixed.length, both_slow_pairs: bothSlow.length },
    occupancy: { base: occ(ok, (p) => p.base), cand: occ(ok, (p) => p.cand) },
  };
}

// How many pairs to collect so that enough of them land with BOTH arms in the fast state. At the
// measured ~20% slow rate a pair is usable with probability 0.8^2 = 0.64, so asking for 10 fast
// pairs means collecting ~16 — which is why "10 pairs" as written has been quietly delivering ~6.
function pairsNeeded(wantFastPairs, slowRatePct) {
  const s = Math.min(0.9, Math.max(0, (Number(slowRatePct) || 0) / 100));
  const usable = (1 - s) * (1 - s);
  return usable <= 0 ? null : Math.ceil(wantFastPairs / usable);
}

// THE CALL SITE the three functions above never had. It groups the verifier's raw interleaved
// readings by guard, classifies each guard arm-blind, and reports the conditioned median plus whether
// enough BOTH-fast pairs were collected -- turning the bimodal doctrine from prose the agent might
// follow into arithmetic the driver applies. It also enforces the campaign's TARGET-route scope: the
// promotion metric is the uniform route's own time, and a reading on a skew rail is reported for
// do-no-harm but is not a number a win may be claimed on. Returns { available, guards, target_median,
// off_target, caveats }; empty/absent readings leave the aggregate geomean in charge, unchanged.
function analyzeGuards(pairedReadings, targetGuards, wantFastPairs) {
  const rows = Array.isArray(pairedReadings) ? pairedReadings : [];
  if (!rows.length) return { available: false, guards: [], target_median: null, off_target: [], caveats: [] };
  const byGuard = new Map();
  for (const r of rows) {
    const g = String((r && r.guard) || '').trim() || '(unnamed)';
    if (!byGuard.has(g)) byGuard.set(g, []);
    byGuard.get(g).push({ base: Number(r.base), cand: Number(r.cand) });
  }
  // Target scope: the explicit set if configured; else any guard whose name says "uniform" (this
  // campaign's target route); else, if none match, every guard is target and nothing is filtered.
  let targets = new Set((Array.isArray(targetGuards) ? targetGuards : []).map((g) => String(g).trim()).filter(Boolean));
  if (!targets.size) {
    const uni = [...byGuard.keys()].filter((g) => /uniform/i.test(g));
    if (uni.length) targets = new Set(uni);
  }
  const allTarget = targets.size === 0;
  const want = Number.isFinite(Number(wantFastPairs)) && Number(wantFastPairs) > 0 ? Number(wantFastPairs) : 10;
  const guards = [], caveats = [];
  let targetMedian = null;
  for (const [g, pairs] of byGuard) {
    const analysis = pairedModeAware(pairs, undefined);
    const isTarget = allTarget || targets.has(g);
    const slow = analysis.conditioned && analysis.occupancy
      ? Math.max(analysis.occupancy.base.pct || 0, analysis.occupancy.cand.pct || 0)
      : 0;
    const needed = pairsNeeded(want, slow);
    const usableFast = analysis.conditioned ? (analysis.fast.n || 0) : (analysis.all.n || 0);
    const median = analysis.conditioned ? analysis.fast.median : analysis.all.median;
    guards.push({ guard: g, is_target: isTarget, conditioned: analysis.conditioned,
      median, fast_pairs: usableFast, pairs_needed: needed, occupancy: analysis.occupancy || null });
    if (isTarget && targetMedian == null && Number.isFinite(Number(median))) targetMedian = Number(median);
    if (isTarget && needed != null && usableFast < want && Number(analysis.all.n) < needed) {
      caveats.push(`${g}: ${usableFast} usable fast-state pair(s) of the ~${needed} needed at a ` +
        `${Math.round(slow)}% slow-state rate; the mode-aware median is UNDER-SAMPLED, so any win on ` +
        `this guard is PROVISIONAL until more pairs are collected.`);
    }
  }
  const offTarget = guards.filter((r) => !r.is_target).map((r) => r.guard);
  if (!allTarget && targetMedian == null && offTarget.length) {
    caveats.push(`No reading on a TARGET route [${[...targets].join(', ')}]; the only readings are on ` +
      `off-target guard(s) [${offTarget.join(', ')}], which are do-no-harm rails and NOT the metric. ` +
      `This measured nothing it is allowed to claim a win on -- the exact state a wave once shipped a ` +
      `number from a skew guard in.`);
  }
  return { available: true, guards, target_median: targetMedian, off_target: offTarget, caveats };
}

// The score that can create credit is computed only from explicit TARGET guards. Regression guards
// are vetoes, never positive weight. This is intentionally based on per_case (the verifier's folded,
// noise-aware result); paired_readings are separately required by the strict terminal contract.
function guardContract(result, targetGuards, regressionGuards, fallbackScore) {
  const rows = Array.isArray(result && result.per_case) ? result.per_case : [];
  const byName = new Map(rows.map((r) => [String(r.name || '').trim(), r]));
  const targets = (Array.isArray(targetGuards) ? targetGuards : []).map(String).filter(Boolean);
  const regressions = (Array.isArray(regressionGuards) ? regressionGuards : []).map(String).filter(Boolean);
  const missingTargets = targets.filter((g) => !byName.has(g));
  const missingRegressions = regressions.filter((g) => !byName.has(g));
  const targetValues = targets.map((g) => Number((byName.get(g) || {}).speedup))
    .filter((v) => Number.isFinite(v) && v > 0);
  const score = targets.length
    ? (targetValues.length === targets.length
      ? Math.exp(targetValues.reduce((s, v) => s + Math.log(v), 0) / targetValues.length)
      : 0)
    : Number(fallbackScore || 0);
  const regressed = regressions.filter((g) => {
    const v = Number((byName.get(g) || {}).speedup);
    return Number.isFinite(v) && v < 1.0;
  });
  return {
    score,
    complete: missingTargets.length === 0 && missingRegressions.length === 0,
    regression_pass: missingRegressions.length === 0 && regressed.length === 0,
    missing_targets: missingTargets,
    missing_regressions: missingRegressions,
    regressed,
  };
}

function promotionScore(result, metric, targetGuards, regressionGuards, fallbackScore) {
  if (!result) return 0;
  if (metric === 'changed_kernel') {
    const a = result.attribution || {};
    const changed = Number(a.changed_us), replaced = Number(a.replaced_sum_us);
    return a.absolute_to_frozen === true &&
      Number.isFinite(changed) && changed > 0 && Number.isFinite(replaced) && replaced > 0
      ? replaced / changed : 0;
  }
  return guardContract(result, targetGuards, regressionGuards, fallbackScore).score;
}
// <</REPLAY:bimodal_split>>

const metricScore = (result) =>
  promotionScore(result, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(result));
const ATTRIBUTION_METRIC = PROMOTION_METRIC === 'legacy' ? 'changed_kernel' : PROMOTION_METRIC;

// <<REPLAY:autonomy_acceptance>>
// Completion and acceptance are deliberately separate. A correct two-launch artifact that is still
// slow is progress under objective=working_kernel and must be commit-able, but it is not proof that
// the autonomous optimization succeeded.
function autonomyAcceptanceVerdict(direction, ver, opts) {
  const o = opts || {};
  const shape = (direction && direction.target_shape) || null;
  if (!shape) return { applicable: false, complete: true, accepted: false, reasons: [] };
  const reasons = [];
  const scoreReasons = [];
  const fa = functionalAcceptance(ver, o.functionalRequirements || {});
  if (!fa.pass) reasons.push(...fa.missing);

  const lv = launchVerdict(ver, Number(shape.launches || o.launchTarget || 2), shape.stages_fused || []);
  if (!lv.shape_met) reasons.push(lv.caveat || 'launch target was not met');

  const gc = guardContract(ver, o.targetGuards || [], o.regressionGuards || [], o.fallbackScore);
  const declaredScore = promotionScore(ver, o.promotionMetric || 'operator_e2e',
    o.targetGuards || [], o.regressionGuards || [], gc.score);
  if (!gc.complete) {
    if (gc.missing_targets.length) reasons.push(`missing target guard(s): ${gc.missing_targets.join(', ')}`);
    if (gc.missing_regressions.length) reasons.push(`missing regression guard(s): ${gc.missing_regressions.join(', ')}`);
  }
  if (!gc.regression_pass && gc.regressed.length) {
    reasons.push(`regression guard(s) below baseline: ${gc.regressed.join(', ')}`);
  }

  const allGuards = [...new Set([...(o.targetGuards || []), ...(o.regressionGuards || [])])];
  if (o.requirePaired && allGuards.length) {
    const pairs = Array.isArray(ver && ver.paired_readings) ? ver.paired_readings : [];
    if (!Number.isFinite(Number(ver && ver.null_arm_pct))) {
      reasons.push('same-session byte-identical null_arm_pct is required');
    } else if ((declaredScore - 1) * 100 <= Math.abs(Number(ver.null_arm_pct))) {
      scoreReasons.push(`target claim ${((declaredScore - 1) * 100).toFixed(3)}% does not clear null ` +
        `${Math.abs(Number(ver.null_arm_pct)).toFixed(3)}%`);
    }
    for (const guard of allGuards) {
      const n = pairs.filter((p) => String(p.guard || '') === String(guard)).length;
      const needed = Number((o.requiredPairsByGuard || {})[guard] || o.requiredPairs || 1);
      if (n < needed) {
        reasons.push(`${guard} has ${n} paired reading(s), requires ${needed}`);
      }
    }
    const targetNeed = Math.max(Number(o.requiredPairs || 1),
      ...(o.targetGuards || []).map((g) => Number((o.requiredPairsByGuard || {})[g] || 0)));
    const ga = analyzeGuards(pairs, o.targetGuards || [], targetNeed);
    if (!ga.available || !Number.isFinite(Number(ga.target_median))) {
      reasons.push('target guard has no readable paired result');
    }
    reasons.push(...ga.caveats);
    const choose = (n, k) => {
      let out = 1;
      for (let i = 1; i <= k; i++) out = out * (n - k + i) / i;
      return out;
    };
    for (const guard of (o.targetGuards || [])) {
      const gp = pairs.filter((p) => String(p.guard || '') === String(guard));
      const wins = gp.filter((p) => Number(p.base) > Number(p.cand)).length;
      let tail = 0;
      for (let k = wins; k <= gp.length; k++) tail += choose(gp.length, k);
      const p = gp.length ? tail / Math.pow(2, gp.length) : 1;
      if (p > 0.05) scoreReasons.push(`${guard} paired sign test p=${p.toFixed(4)} > 0.05`);
    }
  }

  const needsOverlap = !!(shape.require_overlap || o.requireOverlap);
  if (needsOverlap) {
    const ov = overlapVerdict(ver, { geomean: gc.score });
    const frac = Number(ver && ver.overlap && ver.overlap.fraction);
    const cuFrac = Number(ver && ver.overlap && ver.overlap.cu_fraction);
    const scattered = Number(ver && ver.overlap && ver.overlap.scattered_reading);
    const forced = Number(ver && ver.overlap && ver.overlap.forced_reading);
    if (ver && ver.overlap && ver.overlap.measured !== 'yes') {
      reasons.push('controlled overlap evidence measured=yes is required');
    } else if (!Number.isFinite(scattered) || scattered > 0.05 ||
               !Number.isFinite(forced) || forced <= 0.05) {
      reasons.push('overlap meter requires passing scattered and forced-concurrency controls');
    } else if (!['measured', 'contention'].includes(ov.state)) {
      reasons.push(ov.caveat || `overlap meter state is ${ov.state}`);
    } else if (!(frac > 0.05) || !(cuFrac > 0.05)) {
      reasons.push('on-edge wall-clock and CU-weighted overlap must both exceed instrument slop');
    }
  }

  if (o.requireAttribution) {
    const a = (ver && ver.attribution) || {};
    if (!(Number(a.changed_us) > 0 && Number(a.replaced_sum_us) > 0 &&
          (o.targetGuards || []).includes(String(a.guard || '')) &&
          String(a.method || '').trim() &&
          Number.isFinite(Number(a.residual_ms_base)) &&
          Number.isFinite(Number(a.residual_ms_cand)) &&
          (o.promotionMetric !== 'changed_kernel' || a.absolute_to_frozen === true))) {
      reasons.push('launch-changing terminal has invalid attribution values/guard/method/residuals');
    } else {
      const at = attributionVerdict(ver, {
        geomean: declaredScore, metric: o.promotionMetric || 'operator_e2e',
      });
      if (at.reject) reasons.push(at.caveat);
    }
  }

  const complete = reasons.length === 0;
  const accepted = complete && scoreReasons.length === 0 && declaredScore > 1.0 && !!o.positiveControlRan;
  const acceptanceReasons = [...reasons, ...scoreReasons];
  if (complete && !(declaredScore > 1.0)) {
    acceptanceReasons.push(`declared target score ${declaredScore.toFixed(4)}x is not > 1.0`);
  }
  if (complete && !o.positiveControlRan) acceptanceReasons.push('positive control did not pass in this wave');
  return { applicable: true, complete, accepted, score: declaredScore, launch: lv, guards: gc,
    reasons: acceptanceReasons };
}
// <</REPLAY:autonomy_acceptance>>

// DEEP-MODE resume: restore cumulative speedup + insight/ledger history from the prior wave so this
// continuation builds ON the cumulative best (canonical was already seeded from STATE_DIR/best by the
// director) and does not re-explore dead directions. No-op on a fresh run (prior_state undefined).
if (setup.resumed && setup.prior_state) {
  const ps = setup.prior_state;
  if (Number.isFinite(ps.cumulative) && ps.cumulative > cumulative) cumulative = ps.cumulative;
  if (Array.isArray(ps.insights)) {
    // A resumed wave inherits the previous wave's board as round-0 entries, so continuity across
    // waves works the same way continuity across rounds does.
    insightBook = mergeInsights(insightBook, ps.insights, 0, false).book;
    history.insights = renderInsights(insightBook);
  }
  if (Array.isArray(ps.ledger)) history.ledger = ps.ledger;
  if (ps.bottleneck_now) history.bottleneck_now = ps.bottleneck_now;
  if (Array.isArray(ps.best_per_case) && ps.best_per_case.length) bestPerCase = ps.best_per_case;
  // The shelf crosses waves for the same reason the ledger does: a candidate that verified in wave
  // 12 round 2 is exactly the thing wave 13 should not re-derive. Rounds restart at 1 each wave, so
  // a carried entry's base_round is rewritten to 0 — older than anything this wave will absorb, and
  // therefore aged against every absorption rather than none. The absorbed sets come with it, so a
  // patch the previous wave's winners already invalidated stays invalid instead of being reoffered.
  if (Array.isArray(ps.shelf)) {
    shelf = ps.shelf.filter((e) => e && e.id && !e.absorbed)
      .map((e) => ({ ...e, base_round: 0, files: shelfFiles(e),
        footprint: shelfFiles(e).length ? 'known' : 'unknown' }));
  }
  if (ps.absorbed_files && typeof ps.absorbed_files === 'object') {
    // Everything the previous wave absorbed, collapsed to round 0.5: after a carried entry's
    // base_round of 0, before this wave's round 1.
    const prior = [];
    for (const k of Object.keys(ps.absorbed_files)) for (const f of (ps.absorbed_files[k] || [])) prior.push(f);
    if (prior.length) absorbedByRound[0.5] = [...new Set(prior.map(shelfFile))];
  }
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
  if (shelf.length) log(`  shelf: ${shelf.length} verified non-winner(s) carried forward, ` +
    `${shelf.filter((e) => e.footprint === 'known').length} with a known file footprint.`);
}

// The no-improve stop is a SPEEDUP stop. Under objective=working_kernel every round before the last
// one is non-improving by construction, so leaving it armed would end the wave on round 3 with a
// 15-lease budget untouched — see OBJECTIVE, item 2. The remaining stops are the budget and the
// TechLead's own decision, both of which still apply.
//
// `noEvidence` is the same cap applied to a different failure. A round that measures nothing does
// not advance `noImprove` (roundEvidence explains why), so without a second bound a run whose
// harness is broken would keep planning rounds until the budget ran out. Three consecutive rounds
// with no reading is not a search that has run out of ideas, it is an instrument that is not
// working, and the answer to that is not another lease.
while (dispatched < BUDGET && (WORKING_KERNEL || (noImprove < MAX_NO_IMPROVE && noEvidence < MAX_NO_IMPROVE))) {
  round++;
  const remaining = BUDGET - dispatched;
  phase('Optimize');

  // --- (a) Plan the round (TechLead) ------------------------------------
  // Sample the pool FIRST, so the lead plans against the hardware that exists rather than the
  // hardware the plan assumes. GPU_POOL is omitted entirely when the sample is disabled, which keeps
  // the prompt byte-identical for runs that never set gpu_min_free_gib.
  const pool = await samplePool(round);
  const plan = await agentT(
    roleAgent('tech_lead', 'plan_round', 'Decide this round\'s orthogonal directions (or stop).', {
      ...(pool ? { GPU_POOL: pool, GPU_MIN_FREE_GIB } : {}),
      EVAL_DIR, ROUND: round, BUDGET_REMAINING: remaining, CUMULATIVE_SPEEDUP: cumulative,
      BASELINE_GEOMEAN_MS, SKILL_DIR: WORKFLOW_DIR, PROFILE_SUMMARY: profileSummary,
      CURRENT_BEST_PER_CASE: bestPerCase, HISTORY: history,
      // The ladder Analyze built, and what has been taken off it. Passed as DATA, not as a path to
      // maybe-read: a plan phase that is only told where roadmap.md lives plans from the profile.
      ROADMAP: `${EVAL_DIR}/roadmap.md`,
      ROADMAP_LADDER: LADDER,
      LADDER_DISPATCHED: [...dispatchedRungs],
      LADDER_COMPLETED: [...LADDER_MEASURED],
      OPEN_RUNGS: openRungs(LADDER, rungTally),
      TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, LAUNCH_TARGET,
      STRICT_AUTONOMY, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
      REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
      // What the tree currently owes. Enabling steps are committed on function and make the tree
      // slower; only their terminal rung can pay that back. A planner that is not shown the balance
      // plans the next round as if the tree were where it started, and the half-built chain stays
      // half-built -- which is the exact state this project reached and then reported as 1.00x.
      ...(CHAIN_DEBT.length ? {
        CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
        CHAIN_BASELINE,
      } : {}),
      // The dependency graph and the pipe table, for the SAME reason. pipeOccupancyGate (below)
      // rejects a direction whose claim exceeds what its pipe's idle fraction can pay — judging the
      // planner against a table the planner was never shown. And the graph is the whole basis for
      // "these two stages have no edge between them, so they can overlap": a plan phase without it
      // can only reorder work it already knows about, never fuse.
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
        ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
      ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
        ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      ...KB_INPUTS,
    }),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    stopReason = `the TechLead planned no directions for round ${round}` +
      `${plan && plan.reasoning ? `: ${plan.reasoning}` : '.'} ` +
      `${BUDGET - dispatched} of ${BUDGET} budget unit(s) were left unspent.`;
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => {
    const enriched = enrichDirection(d, LADDER);
    return {
      ...enriched,
      idx: i,
      id: enriched.id || `r${round}_d${i}`,
      gpu_id: GPU_RESOURCE.specForIndex(i),
      out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
    };
  });
  // Near the end of the budget, forcing replaces advising: if enabling steps are committed and their
  // terminal fusion rung is still open, the remaining lease is spent CLOSING it rather than exploring
  // a new lever. Runs before the deep/working clamps so the forced terminal is the direction that
  // gets the lease. See terminalRungToForce and roadmapLadderGate (the advisory this makes actionable).
  {
    const debtReport = chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED);
    const plannedRungs = directions.map((d) => String(d.roadmap_rung || '').trim()).filter(Boolean);
    const force = terminalRungToForce(debtReport, LADDER, plannedRungs, remaining);
    if (force) {
      const c = force.rung || {};
      const forced = {
        idx: 0,
        id: `r${round}_close_${force.rungId}`,
        title: c.title || `close fusion rung ${force.rungId}`,
        specialty: 'distributed',
        focus_files: Array.isArray(c.focus_files) ? c.focus_files : [],
        expected_speedup: Number(c.expected_speedup) || undefined,
        roadmap_rung: force.rungId,
        step_role: 'terminal',
        gated_on: Array.isArray(c.gated_on) ? c.gated_on : [],
        mandatory_arms: Array.isArray(c.mandatory_arms) ? c.mandatory_arms : [],
        target_shape: c.target_shape || { launches: LAUNCH_TARGET, require_overlap: REQUIRE_OVERLAP },
        rung_deviation: `forced onto terminal rung ${force.rungId} (${force.reason}): its enabling ` +
          `steps are committed and this rung is the only thing that pays their debt, so the remaining ` +
          `budget is spent closing the fusion rather than exploring.`,
        prompt: `Close the fusion. Enabling/producer steps for terminal rung "${force.rungId}"` +
          `${c.title ? ` (${c.title})` : ''} are already committed to the canonical tree: their ` +
          `completion signalling and buffers exist but nothing consumes them yet. Write the CONSUMER ` +
          `half that collapses the launches into the two-launch fused shape, and report launch_shape, ` +
          `accuracy (relL2), overlap and attribution. This is step_role=terminal: it is judged on ` +
          `FUNCTION and MEASUREMENT -- it may be slower than the tree the first time it lands and is ` +
          `still committed on RUNNING -- not on beating the current best. ${c.rationale || c.prompt || ''}`,
        gpu_id: GPU_RESOURCE.specForIndex(0),
        out_dir: `${EVAL_DIR}/round_${round}/engineer_0`,
      };
      log(`Round ${round}: FORCING terminal rung ${force.rungId} (${force.reason}); ` +
          `${directions.length} planned direction(s) deferred. The budget no longer has slack to both ` +
          `explore and close every open chain, and an unclosed chain ships a tree slower than it started.`);
      directions = [forced];
    }
  }
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
  // ONE direction per round under objective=working_kernel. Clamped here rather than asked of the
  // planner: the planner is told leases are scarce in prose today and still plans 4-8 directions,
  // and a direction that is planned but never gets the lease returns static analysis, which is what
  // round 1 of wave 14 returned from two of its three engineers.
  if (WORKING_KERNEL && directions.length > 1) {
    const dropped = directions.slice(1).map(d => d.id);
    directions = directions.slice(0, 1);
    log(`Round ${round}: objective=working_kernel — one direction gets the lease. Kept ${directions[0].id}, ` +
        `deferred ${dropped.join(', ')}. They are not dead ends; they were not run.`);
  }
  // Price this round's directions against the pipe table BEFORE the lease is spent. The table was
  // printed once after Analyze; this is the call that makes it load-bearing, because a direction
  // claiming more than its pipe's idle fraction can pay is rejectable by arithmetic alone.
  if (REQUIRE_TASK_GRAPH) {
    const pg = pipeOccupancyGate(analysis && analysis.resource_timeline, directions);
    if (pg.caveat) log(`Round ${round}: ${pg.caveat}`);
  }
  // And against the ladder. Unlike the pipe check this one is NOT gated on require_task_graph:
  // every run has a roadmap, and "the rung that was never reached" is not a multi-stage-operator
  // problem. Runs BEFORE the directions are charged to the budget, so the skip is on the record at
  // the moment it is still cheap to change.
  {
    const lg = roadmapLadderGate(LADDER, directions, LADDER_MEASURED);
    log(`Round ${round}: ${lg.summary}`);
    if (lg.caveat) log(`Round ${round}: ${lg.caveat}`);
    if (STRICT_AUTONOMY) {
      const invalid = directions.map((d) => ({ d, v: strictDirectionVerdict(d, LADDER, LADDER_MEASURED) }))
        .filter((x) => !x.v.pass);
      if (invalid.length) {
        throw new Error(
          'STRICT AUTONOMY PLAN REFUSED before budget charge: ' +
          invalid.map((x) => `${x.d.id}: ${x.v.reason}`).join('; '));
      }
    }
    for (const r of lg.planned) {
      dispatchedRungs.add(r);
      const e = rungTally.get(r) || { attempts: 0, last_outcome: 'never_planned' };
      e.attempts += 1;
      // Provisional: a rung that is taken counts as unmeasured until a verified number arrives.
      // If the round dies before the grading below runs, that is exactly the state to carry forward.
      if (e.last_outcome !== 'measured') e.last_outcome = 'unmeasured';
      rungTally.set(r, e);
    }
  }
  const roundCost = directions.reduce((s, d) => s + (d.specialty === 'deep_explore' ? DEEP_COST : 1), 0);
  dispatched += roundCost;
  log(`Round ${round}: ${directions.length} direction(s) [${directions.map(d => d.specialty).join(', ')}], cost ${roundCost}, budget ${dispatched}/${BUDGET}`);

  // --- (b,c) Optimize -> Verify, pipelined per direction ----------------
  const results = await pipeline(
    directions,
    (d) => {
      const isDeep = d.specialty === 'deep_explore';
      // deep_explore reads its own role (broad authority + own iteration loop); specialists read engineer.md.
      const readLine = isDeep
        ? `Then Read ${WORKFLOW_DIR}/roles/deep_engineer.md and ALL knowledge files under ${WORKFLOW_DIR}/knowledge/ ` +
          `(you have broad authority — combine algorithm + memory + compute + host_runtime levers in one ` +
          `coherent rewrite), and follow them. You MAY edit ANY modifiable source (kernel + Python wrapper ` +
          `+ C++ binding), not just focus_files. Run your OWN multi-iteration measure→(self-)profile→rewrite ` +
          `loop and push to the TARGET; keep the best correct version.`
        : `Then Read ${WORKFLOW_DIR}/roles/engineer.md and ${WORKFLOW_DIR}/knowledge/self_monitoring.md and the ` +
          `knowledge files for your specialty, and follow them.`;
      return agentT(
      `You are Engineer ${d.id} (specialty=${d.specialty}) for round ${round}.
First create YOUR private workspace, then optimize.
\`\`\`bash
# Fresh, ISOLATED workspace via tar-copy that EXCLUDES build artifacts (.git/build/__pycache__/.torch_ext/
# *.so/*.o) — no 'rm' anywhere. Each engineer's out_dir is unique per (round,engineer), so the workspace
# is clean on creation; the tar excludes mean no stale build cache is ever inherited (torch .torch_ext
# stores ABSOLUTE paths, so excluding it forces each workspace to build its own fresh).
mkdir -p ${d.out_dir}/workspace
( cd ${CANONICAL} && tar --exclude=./.git --exclude='*/.git' --exclude=./build --exclude='*/build' \\
    --exclude=./__pycache__ --exclude='*/__pycache__' --exclude=./.torch_ext --exclude='*/.torch_ext' \\
    --exclude='*.so' --exclude='*.o' -cf - . ) | ( cd ${d.out_dir}/workspace && tar -xf - )
# The tar excluded .git ON PURPOSE (the source history must never be readable from a workspace), so
# make a FRESH one-commit repo here instead. Without this \`git diff\` has no HEAD to diff against and
# best_patch.diff has to be hand-maintained — which is how a wave shipped patches nobody could
# regenerate. A fresh repo carries no history, so nothing leaks and nothing extra is copied.
cd ${d.out_dir}/workspace
printf '%s\\n' 'build/' '__pycache__/' '*.so' '.torch_ext/' '.rocprofv3/' '*.o' > .gitignore
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
git init -q
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "workspace baseline"
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git add -A && git diff HEAD > ${d.out_dir}/best_patch.diff\`
whenever your implementation BUILDS and passes correctness -- NOT only when geomean>1.0. A step that
is a prerequisite for a later fusion (step_role=enabling) and the final consumer half of a fusion are
both SLOWER than the tree on their own, by construction, and the workflow decides their fate from the
saved patch: an enabling step that leaves no patch cannot be committed on function, and a terminal
fusion that leaves no patch cannot be committed on RUNNING. Withholding the diff because it is not yet
faster is how this project shipped half a fusion nobody could carry forward. (Stage first: a plain
\`git diff\` omits files you CREATED, and a patch missing a new file applies cleanly and then fails at
import.) Report your measured geomean honestly whatever it is; do not suppress a patch to avoid a
number below 1.0.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: {
          id: d.id, title: d.title, focus_files: d.focus_files || [],
          expected_speedup: d.expected_speedup, prompt: d.prompt,
          roadmap_rung: d.roadmap_rung, gated_on: d.gated_on || [],
          mandatory_arms: d.mandatory_arms || [], step_role: d.step_role,
          enables: d.enables, cost_budget_pct: d.cost_budget_pct,
          target_shape: d.target_shape,
          target_guards: TARGET_GUARDS, regression_guards: REGRESSION_GUARDS,
          promotion_metric: PROMOTION_METRIC, required_pairs: REQUIRED_PAIRS,
          required_pairs_by_guard: REQUIRED_PAIRS_BY_GUARD,
        },
        ...(isDeep ? { TARGET: d.expected_speedup ? `reach ${d.expected_speedup}x (or ~90% of the roofline ceiling), whichever is the harder bar` : 'reach ~90% of the roofline ceiling' } : {}),
        KERNEL_PATH: `${d.out_dir}/workspace`,
        OUTPUT_DIR: d.out_dir,
        CANONICAL, GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT,
        codebase_context: `${EVAL_DIR}/codebase_context.md`,
        profiling_summary: profileSummary ? profileSummary.summary_path : '',
        baseline_per_case: BASELINE_PER_CASE,
        INSIGHTS: history.insights,
        // What earlier rounds already spent budget on. Without this an engineer sees only what was
        // learned, never what was attempted, and re-proposes a direction the loop already bought.
        ...(function () { const de = deadEnds(history.rounds); return {
          ...(de.tried.length ? { ALREADY_TRIED: de.tried } : {}),
          ...(de.untried.length ? { NOT_YET_ACTUALLY_TESTED: de.untried } : {}) }; })(),
        KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE,
        KK_REFS: (d.kk_refs && d.kk_refs.length ? d.kk_refs : KK_REFS),
        ...KB_INPUTS,
      })}

Return ONLY the worker_result.json structure as StructuredOutput.`,
      { phase: 'Optimize', label: `${isDeep ? 'deep' : 'eng'} ${d.id}:${d.specialty}`, schema: ENG_SCHEMA }
    ).then((eng) => ({ d, eng }));
    },

    // RECOVER AN ENGINEER'S CLAIM FROM DISK BEFORE DROPPING IT.
    //
    // The Benchmark phase has done this since wave 1 (see the RECOVERY block above); the Optimize
    // phase did not, and on 2026-08-23 that asymmetry cost a wave its only result. Three rounds
    // produced the SAME physical win -- payload_chunk_rows gated at the 512 bucket, +20.6% rank-max,
    // 3/3 pairs, rel-L2 0.0 against the default, i.e. bit-identical (the ratio-to-null it was first
    // reported at, ~10x, came from a 5-pair null that missed that guard's bimodal tail; deeper
    // sampling the next day put the worst null pair at 9.30pp, so the ratio is the open question --
    // the point here is that the win was never SCORED, which is a different failure entirely) --
    // and it was scored ZERO all three times, never for a measurement failure and always at the claim
    // boundary. The decisive one: the engineer wrote a complete worker_result.json to disk at 08:05
    // (populated per_case, speedup_geomean 1.0594, a best_patch.diff that `git apply --check` accepts)
    // and then KEPT MEASURING until 08:12, so the round closed with no StructuredOutput and `eng` was
    // null. The result existed, on disk, in the right shape, at the right path, and the harness stepped
    // over it.
    //
    // Telling engineers "emit the claim last" is the right instruction and it is now in the role file,
    // but an instruction is not a mechanism: the failure mode is an agent running out of time, and an
    // agent that runs out of time cannot be relied on to have followed the instruction about what to do
    // before running out of time. So read the file. No agent, no GPU, no re-measurement -- this is a
    // parse of bytes the engineer already produced, and it can only recover a claim, never invent one.
    async (prev) => {
      const { d } = prev;
      let { eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      if (CLAIM.needsRecovery(eng)) {
        // Workflow scripts run without `fs` (see the note at the top of this file), so recovery is a
        // cheap agent that reads bytes -- exactly what the Benchmark phase already does. It is told to
        // RECOVER ONLY: no GPU, no lease, no re-measurement. This can surface a claim the engineer
        // already produced; it cannot manufacture one.
        const onDisk = await agentT(
          // `engineer`, not `optimize_engineer` — the latter has never existed. roleAgent tells the
          // agent to Read roles/<role>.md, so a wrong name sends it to a missing file and it
          // improvises the procedure for the one path whose whole point is NOT to improvise.
          // Wave 13 hit this in three consecutive rounds before the report named it.
          roleAgent('engineer', 'recover',
            `The engineer for ${d.out_dir} did not return a usable claim. RECOVER ONLY: read ` +
            `${d.out_dir}/worker_result.json AND compare it with the mtimes/claim_complete fields of ` +
            'ab_*.json aggregates beside it. If a completed aggregate is newer, the declaration is stale: ' +
            'recover from that aggregate and name it. Use raw logs only if both are absent/truncated. ' +
            'Return the claim already on disk. Do NOT run any GPU ' +
            'command, do NOT take a lease, do NOT re-measure and do NOT improve the result: a fresh ' +
            'measurement here is a failure, not a fallback. Report per_case EXACTLY as recorded, ' +
            'including guards the engineer marked UNRESOLVED. If no claim is on disk, return ' +
            'per_case: [] and say so in notes.',
            { OUT_DIR: d.out_dir, SKILL_DIR: WORKFLOW_DIR, BASELINE_PER_CASE, COMMANDMENT }),
          { phase: 'Optimize', label: `recover ${d.id}`, schema: ENG_SCHEMA });
        if (CLAIM.recovered(onDisk)) {
          log(`${d.id}: no usable StructuredOutput, but a claim was on disk — ` +
              `${onDisk.per_case.length} case(s), geomean ${onDisk.speedup_geomean}. RECOVERED. ` +
              'The claim boundary is not allowed to delete a measurement.');
          eng = onDisk;
        } else {
          log(`${d.id}: no StructuredOutput and nothing recoverable from ${d.out_dir}.`);
        }
      }
      // Verification is the trust boundary. A working-kernel round and an enabling prerequisite
      // must reach it even when the engineer honestly reports a regression; filtering them here
      // made every later commit-on-function/commit-on-running rule unreachable.
      if (!shouldVerifyDirection(
        d, eng, WORKING_KERNEL || PROMOTION_METRIC === 'changed_kernel', metricScore(eng))) {
        return { d, eng, ver: null };
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          FROZEN_KERNEL_PATH: KERNEL_PATH_ORIG,
          // The file whitelist Analyze declared. verify_engineer.md step 5 says to diff the patch's
          // file list against it and fail anything outside — a patch that edits the instrument and
          // the subject in the same diff has no readable result. Analyze has produced
          // `modifiable_files` all along and nothing consumed it, so that check was asking verify to
          // compare against a list it did not have; it either skipped the check or invented the list.
          // Absent, it says UNDECLARED rather than being omitted — the same rule as ACTIVATION
          // below. An omitted whitelist and an empty one read identically to the agent, and the
          // silent reading is "no check to do".
          MODIFIABLE_FILES: (analysis && Array.isArray(analysis.modifiable_files)
            && analysis.modifiable_files.length)
            ? analysis.modifiable_files : 'UNDECLARED',
          // Verify applies specialty-specific gates (see verify_engineer.md step 4c: a
          // `distributed` patch can be numerically correct and still deadlock).
          ...(d.specialty ? { SPECIALTY: d.specialty } : {}),
          MANDATORY_ARMS: d.mandatory_arms || [],
          ENGINEER_ARMS_RUN: (eng && eng.arms_run) || [],
          TARGET_SHAPE: d.target_shape || null,
          TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC,
          STRICT_AUTONOMY, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
          REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD, LAUNCH_TARGET,
          ACCURACY_METRIC, ACCURACY_THRESHOLD,
          // What the engineer says turns its code ON, verbatim. If it declared nothing, verify is
          // told so explicitly rather than being left to assume default-ON — the assumption is the
          // bug: an undeclared switch and a genuinely default-ON patch look identical from here.
          ACTIVATION: (eng && eng.activation) ? JSON.stringify(eng.activation) : 'UNDECLARED',
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          // Strict Verify sees only out-of-band file hashes, never a filesystem address. This still
          // catches an exact/comment-only copy before it can be changed in a later round and
          // laundered past final comparison. Non-strict capability mode keeps the legacy path input.
          ...(CAPABILITY_EVAL && KNOWN_REFERENCE_HASHES.length
            ? { KNOWN_REFERENCE_HASHES } : {}),
          ...(CAPABILITY_EVAL && !STRICT_AUTONOMY && KNOWN_REFERENCE_PATHS.length
            ? { KNOWN_REFERENCE_PATHS: KNOWN_REFERENCE_PATHS.join(' ') } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);

  // --- UNBACKED CLAIM: the engineer measured a win it cannot hand over ----
  // The other half of the 2026-08-23 claim-boundary loss. An engineer returned a well-formed claim
  // whose declared patch did not exist on disk, because the effect lived only in bench CLI flags and
  // no patch had ever been written. Verify reports that as `apply_failed`, and `apply_failed` then
  // flows into the round exactly like a candidate that was tried and lost. It is not the same thing,
  // and the difference is expensive: that direction had produced the run's ONLY win (+20.6% at one
  // guard, bit-identical), and reading it as "did not work" is what let two later rounds re-derive it
  // from scratch. A measurement that cannot be handed over is a reporting defect. Name it as one.
  for (const r of clean) {
    if (!CLAIM.unbacked(r)) continue;
    r.unbacked = true;
    log(`${r.d.id}: CLAIM NOT BACKED BY A PATCH — the engineer claims ` +
        `${primSpeedup(r.eng).toFixed(4)}x but ${r.patch} does not apply. This is a REPORTING ` +
        'failure, not a null result: if the effect was real it came from flags or an unsaved edit ' +
        'and has been lost. Do NOT record this direction as "tried, did not work" — record it as ' +
        'unmeasured, and re-dispatch it with the patch written first.');
  }

  // --- ACTIVATION: did the candidate's code actually RUN? ----------------
  // A patch whose fast path is gated behind a switch nobody sets measures byte-identical to the
  // baseline. The harness then reports 1.000x, the round reads "the idea did not work", and the
  // direction is dropped — on evidence that was never collected. That is what happened in wave 1's
  // first round. So an unexercised patch gets its OWN outcome, loudly, and is never allowed to be
  // filed as a null result. `unknown` counts as `no`: the whole point is that silence here is
  // indistinguishable from the failure, so silence must not be the cheap answer.
  for (const r of clean) {
    if (!r.ver || r.ver.status === 'apply_failed') continue;
    const act = String(r.ver.activation_confirmed || 'unknown').toLowerCase();
    if (act === 'yes') continue;
    r.inactive = act;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`INACTIVE ${r.d.id}: the patched code path was ${act === 'no' ? 'NOT executed' : 'NOT PROVEN to execute'} ` +
        `during the measured run, so its ${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'result'} ` +
        `is VOID — not a negative result. Do NOT record this direction as tried-and-failed; it has ` +
        `not been tried. Evidence: ${r.ver.activation_evidence || '(none supplied)'}`);
  }

  // --- ARTIFACT DISTINCTNESS: were the two arms the same binary? ---------
  // The activation loop above catches an arm that never ran. This catches the arm that ran and was
  // byte-identical to the one it was measured against — a strictly harder failure, because every
  // gate in this workflow passes it: the patch is real, the marker fires, the null arm behaves, and
  // the number is a clean 1.000. On 2026-08-22 a retroactive audit found a FlyDSL switch that never
  // entered the compiler's disk-cache key (only the OUTER co_names are walked, and the cache root is
  // machine-global, so building the arms in separate checkouts did not isolate them either). The
  // all-ON arm, the all-OFF arm and the canonical unpatched tree all resolved to one disk_key. A
  // "well-powered null" from that comparison had already closed an entire optimization axis, and the
  // axis had to be reopened a wave later. Identical artifacts are a VOID experiment, never a null.
  //
  // Only equality is fatal. `n/a` (not a JIT candidate) and `unknown` (the proof could not be run)
  // are reported and left alone: unlike activation, where silence is the cheap answer that hides the
  // failure, here the fatal state is directly observable and demanding a proof the verifier has no
  // way to produce would just push it to guess `yes`.
  for (const r of clean) {
    if (!r.ver || r.ver.status === 'apply_failed' || r.inactive) continue;
    const ad = String(r.ver.artifact_distinct || '').toLowerCase();
    const hb = String(r.ver.artifact_hash_base || '').trim();
    const hc = String(r.ver.artifact_hash_candidate || '').trim();
    const sameHash = hb && hc && hb === hc;
    if (ad !== 'no' && !sameHash) {
      if (ad === 'unknown' || (!ad && (hb || hc))) {
        log(`ARTIFACT DISTINCTNESS UNPROVEN ${r.d.id}: base=${hb || '(none)'} candidate=${hc || '(none)'}. ` +
            `The result stands, but a JIT candidate whose arms were never shown to differ cannot ` +
            `close an axis — do not record a 1.000 here as a null result.`);
      }
      continue;
    }
    r.inactive = 'no';
    r.same_artifact = true;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`SAME BINARY ${r.d.id}: base and candidate resolved to the SAME compiled artifact` +
        `${sameHash ? ` (${hb})` : ''}, so its ` +
        `${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'result'} measures ` +
        `nothing about the idea — it is VOID, not a null. The switch did not enter the compiler's ` +
        `cache key. Re-run with the switch anchored in the OUTER traced scope before this direction ` +
        `is called tried.`);
  }

  // Attribution runs BEFORE the verified filter, because unlike the overlap gate it can reject, and
  // a rejected candidate must not reach the candidate list at all. Decided for every direction that
  // passed verification, not only the ones that describe themselves as fusion: a direction's own
  // account of what it did is exactly the thing that must not decide whether the evidence is checked.
  for (const r of clean) {
    if (!(r.ver && verifyCompleted(r.ver) &&
      String(r.ver.correctness || '').toLowerCase().startsWith('pass') && !r.inactive)) continue;
    // An enabling step makes no win claim, so there is no win to attribute to the wrong kernel. Its
    // bar is functional acceptance plus its cost budget, both applied below.
    if (stepRoleOf(r.d) === 'enabling') continue;
    const at = attributionVerdict(r.ver, { geomean: metricScore(r.ver), metric: ATTRIBUTION_METRIC });
    r.attribution_state = at.state;
    if (at.caveat) r.attribution_caveat = at.caveat;
    if (at.reject) r.attribution_rejected = at.state;
  }

  // Evaluate the complete terminal contract after activation/artifact/attribution processing and
  // before promotion. In working-kernel mode an incomplete-but-functional artifact may still be
  // committed so the next round can build on it; it cannot close its rung or satisfy autonomy.
  for (const r of clean) {
    if (stepRoleOf(r.d) !== 'terminal' || !r.d.target_shape) continue;
    const av = autonomyAcceptanceVerdict(r.d, r.ver, {
      functionalRequirements: functionalRequirementsFor(r.d),
      launchTarget: LAUNCH_TARGET,
      targetGuards: TARGET_GUARDS,
      regressionGuards: REGRESSION_GUARDS,
      promotionMetric: PROMOTION_METRIC,
      fallbackScore: metricScore(r.ver),
      requirePaired: STRICT_AUTONOMY,
      requiredPairs: REQUIRED_PAIRS, requiredPairsByGuard: REQUIRED_PAIRS_BY_GUARD,
      requireOverlap: REQUIRE_OVERLAP,
      requireAttribution: REQUIRE_ATTRIBUTION,
      positiveControlRan: PC_RAN,
    });
    r.autonomy_contract = av;
    if (av.accepted) {
      log(`AUTONOMY CANDIDATE ${r.d.id}: evidence clears the contract at ${av.score.toFixed(4)}x; ` +
          `acceptance remains pending until this exact patch is committed to canonical.`);
    } else if (av.applicable) {
      const msg = `${r.d.id}: ${av.reasons.join('; ') || 'terminal did not clear the acceptance score'}`;
      ACCEPTANCE_CAVEATS.push(msg);
      log(`AUTONOMY INCOMPLETE ${msg}`);
    }
  }

  // ENABLING STEPS ARE JUDGED BEFORE THE SPEED FILTER, because the speed filter is what they cannot
  // pass. An enabling step is a prerequisite half of a fusion: it adds completion signalling and a
  // second buffer, it has no consumer yet, and its only measurable effect is its own overhead. See
  // <<REPLAY:enabling_step>>. Judged on function, committed on function, cost carried as debt.
  const enablingKeeps = [];
  for (const r of clean) {
    if (stepRoleOf(r.d) !== 'enabling' || r.inactive) continue;
    const ev = enablingVerdict(r.d, r.ver, primSpeedup(r.ver), functionalRequirementsFor(r.d));
    r.enabling_verdict = ev;
    if (ev.commit && r.patch) {
      enablingKeeps.push(r);
      log(`ENABLING KEEP ${r.d.id}: ${ev.reason}`);
    } else {
      log(`ENABLING REJECT ${r.d.id}: ${ev.reason}${ev.commit && !r.patch ? ' (and it produced no patch to keep)' : ''}`);
      CHAIN_CAVEATS.push(`${r.d.id} [enabling, not kept]: ${ev.reason}`);
    }
  }

  const verified = clean.filter(r => r.ver && verifyCompleted(r.ver) &&
    functionalAcceptance(r.ver, functionalRequirementsFor(
      r.d, WORKING_KERNEL ? 'commit' : 'acceptance')).pass &&
    // Under objective=working_kernel the commit gate is "does it RUN", not "is it faster" (see the
    // commit block below and OBJECTIVE item 3). The TERMINAL step of a staged fusion is SLOWER than
    // the tree the first time it is written -- it has only just acquired its consumer and still
    // carries the signalling and the second buffer the enabling steps installed -- so gating
    // `verified` on >1.0 here deletes the fused megakernel before it can ever be committed on
    // RUNNING. That is precisely how this project stopped at half a fusion: the producer half was
    // enabling (exempt from this filter) and the consumer half hit exactly this line. In speedup
    // mode the >1.0 bar stays, because there a candidate that is not faster has nothing to offer.
    (WORKING_KERNEL || metricScore(r.ver) > 1.0) && !r.inactive &&
    !r.attribution_rejected &&
    (WORKING_KERNEL || (guardContract(r.ver, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(r.ver)).complete &&
      guardContract(r.ver, TARGET_GUARDS, REGRESSION_GUARDS, primSpeedup(r.ver)).regression_pass)) &&
    // A strict speedup wave may not promote an acceptance-terminal whose launch/overlap/guard
    // contract is incomplete. A working-kernel wave still keeps a functional partial artifact.
    (WORKING_KERNEL || !STRICT_AUTONOMY || !r.d.target_shape ||
      (r.autonomy_contract && r.autonomy_contract.complete)) &&
    // An enabling step never competes for the round win: it is expected to be below 1.0 and it has
    // its own commit path below. One that happens to measure above 1.0 is still not a win claim --
    // the fusion it enables has not been closed yet, so there is nothing yet to attribute.
    stepRoleOf(r.d) !== 'enabling');

  // Rejection by exclusion is invisible -- the direction just quietly stops existing, which reads
  // like "it didn't work". Print it with its number attached, for the same reason PLAGIARIZED is
  // printed with its number: the speedup is exactly what makes it tempting to accept.
  for (const r of clean) {
    if (!r.attribution_rejected) continue;
    log(`ATTRIBUTION ${r.attribution_rejected.toUpperCase()} ${r.d.id}: ${r.attribution_caveat}`);
    ATTRIBUTION_CAVEATS.push(`${r.d.id} [${r.attribution_rejected}]: ${r.attribution_caveat}`);
  }
  for (const r of clean) {
    if (r.attribution_rejected || !r.attribution_caveat) continue;
    log(`ATTRIBUTION ${String(r.attribution_state).toUpperCase()} ${r.d.id}: ${r.attribution_caveat}`);
    if (r.attribution_state !== 'attributed') {
      ATTRIBUTION_CAVEATS.push(`${r.d.id} [${r.attribution_state}]: ${r.attribution_caveat}`);
    }
  }

  // `plagiarized` and `harness_modified` are already excluded by the filter above, but exclusion is
  // invisible — the direction just quietly stops existing, which reads like "it didn't work". Both
  // are load-bearing findings about the RUN rather than about the kernel, and the speedup that came
  // with them is exactly the thing that makes them tempting to accept, so print it alongside.
  for (const r of clean) {
    const st = r.ver && r.ver.status;
    if (st !== 'plagiarized' && st !== 'harness_modified') continue;
    const pct = ((metricScore(r.ver) - 1) * 100);
    log(`${st.toUpperCase()} ${r.d.id}: REJECTED as a win despite ` +
        `${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'an unreadable number'}. ` +
        `${st === 'plagiarized'
          ? 'The patch contains a file byte-identical to a known reference — the result was imported, not derived.'
          : 'The patch edited the measurement harness, so subject and instrument changed together.'} ` +
        `${r.ver.notes || ''}`);
  }

  // Flag wins whose provenance cannot support them. This does NOT reject the result — a real win
  // measured sloppily is still probably a win, and silently discarding it would repeat the mistake
  // this check exists to catch. It marks the result PROVISIONAL so the number travels with the
  // conditions it was taken under, all the way into the report.
  for (const r of verified) {
    const claimPct = (metricScore(r.ver) - 1) * 100;
    const reps = Number(r.ver.reps);
    const nullArm = Number(r.ver.null_arm_pct);
    const reasons = [];
    if (!Number.isFinite(reps) || reps < 5) reasons.push(`reps=${Number.isFinite(reps) ? reps : 'unreported'} (<5)`);
    if (!Number.isFinite(nullArm)) reasons.push('no null arm');
    else if (claimPct <= Math.abs(nullArm)) reasons.push(`claim ${claimPct.toFixed(2)}% <= null arm ${Math.abs(nullArm).toFixed(2)}%`);
    // Mode-aware, target-scoped conditioning of the raw readings, when the verifier returned them.
    // This is where modeSplit/pairedModeAware/pairsNeeded finally decide something: an under-sampled
    // bimodal target guard, or a win read only off a skew rail, becomes a stated PROVISIONAL reason
    // rather than a silent average. Never rejects -- like the rest of the provisional block, it marks.
    const ga = analyzeGuards(r.ver.paired_readings, TARGET_GUARDS, 10);
    if (ga.available) {
      r.guard_analysis = ga;
      for (const c of ga.caveats) { reasons.push(c); GUARD_CAVEATS.push(`${r.d.id}: ${c}`); }
    }
    if (reasons.length) {
      r.provisional = reasons.join('; ');
      log(`PROVISIONAL ${r.d.id}: +${claimPct.toFixed(2)}% claimed but ${r.provisional}. ` +
          `Carried forward, but it must be re-measured with >=5 interleaved reps and a null arm ` +
          `before it is reported as a result.`);
    }
    // Overlap. Judged for every verified direction, not only the ones that claim fusion: a
    // direction's own account of what it did is exactly the thing that must not decide whether the
    // evidence is checked. Never rejects — see overlapVerdict.
    const ov = overlapVerdict(r.ver, { geomean: metricScore(r.ver) });
    r.overlap_state = ov.state;
    if (ov.caveat) {
      r.overlap_caveat = ov.caveat;
      log(`OVERLAP ${ov.state.toUpperCase()} ${r.d.id}: ${ov.caveat}`);
      if (ov.state !== 'measured_none') OVERLAP_CAVEATS.push(`${r.d.id} [${ov.state}]: ${ov.caveat}`);
    }
    // And the objective. In a controlled wave this is a no-op on every result.
    const oj = objectiveVerdict(OBJECTIVE, PC_RAN, metricScore(r.ver));
    r.objective_state = oj.state;
    if (oj.caveat) {
      r.objective_void = true;
      r.objective_caveat = oj.caveat;
      log(`TIMING VOID ${r.d.id}: ${oj.caveat}`);
      OBJECTIVE_CAVEATS.push(`${r.d.id}: ${oj.caveat}`);
    }
    // Launch shape — acceptance criterion 1. Judged for every verified direction; never rejects, but
    // records the count and whether the two-launch fused shape was reached, so a fusion rung cannot
    // be reported as closed on a candidate that is still three launches. See launchVerdict.
    const directionLaunchTarget = Number(r.d && r.d.target_shape && r.d.target_shape.launches);
    const lv = launchVerdict(r.ver,
      Number.isFinite(directionLaunchTarget) ? directionLaunchTarget : LAUNCH_TARGET,
      (r.d && r.d.target_shape && r.d.target_shape.stages_fused) || []);
    r.launch_state = lv.state;
    r.launch_shape_met = lv.shape_met;
    if (Number(r.ver && r.ver.launch_shape && r.ver.launch_shape.launches_cand) <= LAUNCH_TARGET) {
      TWO_LAUNCH_REACHED = true;
    }
    if (lv.caveat) {
      r.launch_caveat = lv.caveat;
      log(`LAUNCH ${lv.state.toUpperCase()} ${r.d.id}: ${lv.caveat}`);
      if (lv.state !== 'met') LAUNCH_CAVEATS.push(`${r.d.id} [${lv.state}]: ${lv.caveat}`);
    }
  }

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: metricScore(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
    touched_files: Array.isArray(r.ver.touched_files) ? r.ver.touched_files : [],
    // The fusion chain this candidate belongs to, identified by its terminal rung (a terminal step's
    // own rung, or the rung an enabling step enables). Carried onto the shelf so a later commit on
    // the SAME chain does not age this candidate off it. Empty for a standalone optimisation.
    chain_id: String((r.d && (r.d.enables || r.d.roadmap_rung)) || '').trim(),
  }));

  // What the shelf can offer this round, decided before the integrator is dispatched. `stale` and
  // `unknown` are LOGGED rather than dropped silently: a shelf that offers nothing for five rounds
  // looks identical to a shelf that was never consulted, and the two want opposite responses.
  const shelfPick = shelfEligible(shelf, absorbedByRound, SHELF_OFFER_K);
  if (shelfPick.offer.length) { SHELF_STATS.rounds_offered++; SHELF_STATS.offers += shelfPick.offer.length; }
  if (shelf.length) {
    log(`SHELF r${round}: ${shelf.length} shelved, ${shelfPick.eligible.length} still apply, ` +
        `offering ${shelfPick.offer.length} (K=${SHELF_OFFER_K})` +
        (shelfPick.offer.length ? `: ${shelfPick.offer.map((e) => `${e.id}@${e.geomean.toFixed(3)}x`).join(', ')}` : '') +
        (shelfPick.stale.length ? `; ${shelfPick.stale.length} aged out (a later winner touched their files: ` +
          `${shelfPick.stale.map((e) => `${e.id}[${e.clash.join(',')}]`).join('; ')})` : '') +
        (shelfPick.unknown.length ? `; ${shelfPick.unknown.length} withheld for UNKNOWN footprint ` +
          `(${shelfPick.unknown.map((e) => e.id).join(', ')}) — their verify reported no touched_files, ` +
          `so whether they conflict is unknown, and unknown is not orthogonal` : ''));
  }

  let integrate = null;
  // One verified direction plus a shelved offer is a real merge opportunity — historically this
  // round would have skipped the merge phase entirely and the shelved work would age out unused.
  if (!STRICT_AUTONOMY &&
      (verified.length >= 2 || (verified.length >= 1 && shelfPick.offer.length > 0))) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        TARGET_GUARDS, REGRESSION_GUARDS, PROMOTION_METRIC, STRICT_AUTONOMY,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        // Historical candidates that verified in an earlier round, lost, and whose files nothing
        // absorbed since has touched. They are OFFERS, not obligations: the integrator is expected
        // to say so when one does not combine, and that answer is what sizes K.
        SHELF_PATCHES: shelfPick.offer.map((e) => ({
          id: e.id, title: e.title, specialty: e.specialty, patch: e.patch,
          files: e.files, verified_geomean_when_cut: e.geomean,
          cut_against_round: e.base_round, shelved_round: e.shelved_round,
        })),
        SHELF_NOTE: shelfPick.offer.length
          ? 'SHELF_PATCHES come from earlier rounds. Their `verified_geomean_when_cut` was measured ' +
            'against the CANONICAL of round `cut_against_round`, NOT against the CANONICAL you have ' +
            'now, so it is a reason to try one, never a number you may carry into your result. The ' +
            'file sets were checked mechanically: nothing absorbed since each patch was cut touches ' +
            'a file it touches, so it should still apply. Take one only if you can show it composes; ' +
            'reporting "offered N, none composed, here is why" is a real answer and is how the size ' +
            'of this offer gets tuned.'
          : '',
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? promotionScore({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
      per_case: integrate.best.per_case || [],
    }, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    primSpeedup({ verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean })) : 0;
    const integGuards = integrate && integrate.best
      ? guardContract(integrate.best, TARGET_GUARDS, REGRESSION_GUARDS, integPrim)
      : { complete: false, regression_pass: false };
    if (integrationEligible(integrate, integPrim,
      Math.max(...candidates.map(c => c.geomean)), integGuards)) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
        // An integration's footprint, if it reported one; otherwise everything that was on the
        // table. See INTEGRATE_SCHEMA.best — the fallback deliberately over-states.
        touched_files: Array.isArray(integrate.best.touched_files) && integrate.best.touched_files.length
          ? integrate.best.touched_files
          : [...new Set([...candidates.flatMap((c) => c.touched_files || []),
                         ...shelfPick.offer.flatMap((e) => e.files)])],
        integrated_from: Array.isArray(integrate.best.patches) ? integrate.best.patches : [],
      });
      const took = shelfPick.offer.filter((e) => (integrate.best.patches || []).includes(e.id));
      SHELF_STATS.hits += took.length;
      SHELF_STATS.hit_ids.push(...took.map((e) => e.id));
      if (took.length) log(`SHELF HIT r${round}: the integration took ${took.map((e) => e.id).join(', ')} ` +
        `from the shelf — cross-round combination, not a re-derivation.`);
      else if (shelfPick.offer.length) log(`SHELF MISS r${round}: ${shelfPick.offer.length} offered, ` +
        `none reported as included. ${integrate.notes ? `Integrator: ${String(integrate.notes).slice(0, 300)}` : ''}`);
    } else if (shelfPick.offer.length) {
      log(`SHELF MISS r${round}: ${shelfPick.offer.length} offered, the integration did not improve ` +
          `on the best individual, so nothing from the shelf was taken.`);
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  // Under objective=working_kernel the commit gate is the objective, not the speed. MIN_IMPROVE
  // would commit nothing — a kernel that still crashes is never 2% faster than cumulative — and a
  // round that commits nothing hands round N+1 the unfused tree, which is exactly how the only fused
  // megakernel this project has ever produced (wave 3, 727 lines, on hardware, crashing) came to be
  // dropped after wave 4 and to no longer exist on disk. Committing on RUNNING is what makes the
  // debug loop cumulative instead of 15 independent first attempts.
  const winnerResult = winner ? (verified.find(r => r.d.id === winner.id) || null) : null;
  const winnerVer = winnerResult && winnerResult.ver;
  const winnerRuns = !!(winner && runsCleanly(
    winnerVer, functionalRequirementsFor(winnerResult && winnerResult.d, 'commit')));
  let improved = WORKING_KERNEL
    ? winnerRuns
    : !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));
  let winnerCommitted = false;
  if (WORKING_KERNEL) {
    log(`Round ${round}: objective=working_kernel — commit gate is "does it run", not speed. ` +
        `${winner ? (winnerRuns ? `${winner.id} RUNS (correctness pass, activation confirmed, liveness not failing) ` +
          `-> committed.` : `${winner.id} does not yet run -> not committed; next round continues from the ` +
          `current canonical tree.`) : 'no verified candidate this round.'}`);
  }

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    let commitResult = await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
git rev-parse HEAD > ${EVAL_DIR}/round_${round}/winner_precommit_head.txt
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
${STRICT_AUTONOMY
  ? 'If the patch does not apply exactly, return committed=false. Do NOT hand-merge in strict autonomy.'
  : 'If BOTH `git apply` and `git apply --3way` fail, inspect the patch and apply it manually (edit the'}
${STRICT_AUTONOMY ? '' : `
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.)`}
Return JSON {committed, current_best_diff, head_before, head_after, exact_patch,
 transaction_valid, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    if (STRICT_AUTONOMY) {
      commitResult = await agentT(
        `Independently verify the canonical commit transaction for round ${round}. Do not edit files.
Read ${EVAL_DIR}/round_${round}/winner_precommit_head.txt as head_before and run
\`git -C ${CANONICAL} rev-parse HEAD\` as head_after. Require exactly one commit in
\`head_before..head_after\`. Compute stable patch ids for ${winner.patch} and for
\`git -C ${CANONICAL} diff head_before..head_after\`; they must match. Also require the commit
subject to start with "round ${round} winner:". Return committed=true and exact_patch=true only if
all checks pass. If HEAD is unchanged, return transaction_valid=true, committed=false and
exact_patch=false only when \`git status --porcelain\` is also empty. A missing pre-head, dirty
working tree, multi-commit HEAD, or patch mismatch is
transaction_valid=false. Return JSON
{committed, head_before, head_after, exact_patch, transaction_valid, current_best_diff, note}.`,
        { phase: 'Merge', label: `commit transaction check r${round}`, schema: COMMIT_SCHEMA });
      if (!(commitResult && commitResult.transaction_valid === true &&
            commitResult.committed === true && commitResult.exact_patch === true &&
            String(commitResult.head_before || '').trim() &&
            String(commitResult.head_after || '').trim() &&
            commitResult.head_before !== commitResult.head_after)) {
        throw new Error(
          `STRICT AUTONOMY ABORT: winner commit transaction for round ${round} could not be ` +
          'independently matched to the verified patch. Canonical may have changed; do not continue.');
      }
    }
    const transition = winnerCommitTransition(
      improved, commitResult, winner, winnerResult, STRICT_AUTONOMY);
    winnerCommitted = transition.committed;
    if (!winnerCommitted) {
      improved = false;
      const msg = `WINNER NOT COMMITTED r${round}: ${winner.id} passed selection but the canonical commit ` +
        `did not report committed=true. Cumulative, rung completion and autonomy acceptance stay unchanged.`;
      log(msg);
      if (winnerResult && winnerResult.autonomy_contract && winnerResult.autonomy_contract.accepted) {
        ACCEPTANCE_CAVEATS.push(msg);
      }
    }
    if (winnerCommitted) {
    // A voided reading must not become the number the next round is measured against. Under
    // objective=working_kernel with no control there is no admissible speedup, so `cumulative` and
    // `bestPerCase` stay where they were and only the ARTIFACT advances.
    const winnerVoid = WORKING_KERNEL && !!(verified.find(r => r.d.id === winner.id) || {}).objective_void;
    // In working_kernel mode the commit gate is RUNNING, so a committed winner can be SLOWER than the
    // tree -- a terminal fusion is, the first time it lands (see the `verified` filter above). The
    // ARTIFACT must advance so the next round builds on the fused shape, but the HEADLINE number must
    // not ratchet DOWN: a regression installed as `cumulative` lets the following round "win" by
    // reverting, and reports a not-yet-optimised fusion as if it were the deliverable. So under
    // working_kernel cumulative only moves up. In speedup mode a winner cleared MIN_IMPROVE by
    // construction, so `winner.geomean > cumulative` already holds and this guard is a no-op there.
    if (!winnerVoid && (!WORKING_KERNEL || winner.geomean > cumulative)) {
      cumulative = winner.geomean;
      bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    }
    finalWinner = winner;
    if (transition.acceptanceReached) {
      AUTONOMY_ACCEPTANCE_REACHED = true;
      log(`AUTONOMY ACCEPTANCE COMMITTED ${winner.id}: the exact contract-clearing patch now ` +
          `exists in canonical at target score ${winner.geomean.toFixed(4)}x.`);
    }
    noImprove = 0;

    // --- (f) Re-profile the new best ------------------------------------
    profileSummary = await agentT(
      roleAgent('profile_engineer', 'reprofile', 'Re-profile the new best and explain the bottleneck shift.', {
        WORKSPACE: CANONICAL, EVAL_DIR, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0), ROUND: round,
        COMMANDMENT, PREVIOUS_METRICS: profileSummary,
      }),
      { phase: 'Optimize', label: `reprofile r${round}`, schema: PROFILE_SCHEMA });
    const reprofileAnalysisResult = await runProfileAnalysis(
      profileSummary,
      round,
      `analysis_engineer:reprofile r${round}`,
    );
    if (profileSummary) {
      profileSummary = {
        ...profileSummary,
        analysis_result: reprofileAnalysisResult,
      };
    }
    }
  }

  // --- (e2) Commit the enabling steps ------------------------------------
  // Separate from the winner commit above and it has to be. The winner is committed because it is
  // FASTER; these are committed because they WORK and something else is going to be built on top of
  // them. Committing them is the whole point of the distinction: a prerequisite that passes on
  // function and is then left out of the canonical tree is a prerequisite the next round cannot see,
  // and the step that was supposed to consume it gets written against a tree where it does not
  // exist. Applied AFTER the winner so the winner's patch still applies against the tree it was cut
  // from; one that then conflicts is logged and shelved rather than forced.
  const enablingUnlanded = [];
  const enablingLandedIds = new Set();
  if (enablingKeeps.length) {
    let res = await agentT(
      `You are the TechLead committing round ${round}'s ENABLING steps into the canonical workspace.

These are NOT round winners. Each one passed functional acceptance — it builds, its path is
confirmed to run, correctness passes, and it does not hang — and each is a prerequisite for a
fusion step that has not been written yet. They are expected to be SLOWER on their own. Commit
them so the next round has something to build the consumer half against.

\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
git rev-parse HEAD > ${EVAL_DIR}/round_${round}/enabling_precommit_head.txt
\`\`\`
Then, for EACH patch below IN ORDER, run \`git apply <patch> || git apply --3way <patch>\` and commit
it on its own with the message shown. If a patch fails BOTH plain and --3way apply, do NOT hand-merge
it and do NOT force it: skip it, leave the tree as it was before that patch, and report it in
\`skipped\`. A prerequisite that no longer applies is a prerequisite whose assumptions changed, and
silently reconciling it is how a chain ends up half-applied in a way nobody can see.

${enablingKeeps.map((r) => `- patch: ${r.patch}\n  message: "round ${round} enabling: ${r.d.id} ` +
  `(enables ${r.enabling_verdict.enables}, cost ${r.enabling_verdict.cost_pct}%)"`).join('\n')}

Finally: \`git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff\`
and re-run the COMMANDMENT CORRECTNESS check on ${CANONICAL} via gpu_lock. Report JSON
{committed: [<patch paths that landed>], skipped: [{patch, reason}], correctness_after,
 head_before, head_after, exact_patch, transaction_valid, note}.`,
      { phase: 'Merge', label: `commit enabling r${round}`, schema: obj({
        committed: { type: 'array', items: { type: 'string' } },
        skipped: { type: 'array', items: { type: 'object', additionalProperties: true } },
        correctness_after: { type: 'string' }, note: { type: 'string' },
        head_before: { type: 'string' }, head_after: { type: 'string' },
        exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
      }, ['committed']) });
    if (STRICT_AUTONOMY) {
      res = await agentT(
        `Independently verify the enabling commit transaction for round ${round}. Do not edit files.
Read ${EVAL_DIR}/round_${round}/enabling_precommit_head.txt and compare it with
\`git -C ${CANONICAL} rev-parse HEAD\`. If unchanged, report committed:[]; the orchestrator stops so
the patch can be retried. If changed, strict mode has exactly one direction: require exactly one
commit, subject starting "round ${round} enabling:", and a stable patch-id equal to
${enablingKeeps[0].patch}. Then run the COMMANDMENT correctness entry on ${CANONICAL}; only a pass
may return transaction_valid=true, exact_patch=true and
committed:["${enablingKeeps[0].patch}"]. Any missing pre-head,
dirty working tree, unexpected/multiple commit, patch mismatch or correctness failure is transaction_valid=false.
Return {committed, skipped:[], correctness_after, head_before, head_after,
exact_patch, transaction_valid, note}.`,
        { phase: 'Merge', label: `enabling transaction check r${round}`, schema: obj({
          committed: { type: 'array', items: { type: 'string' } },
          skipped: { type: 'array', items: { type: 'object', additionalProperties: true } },
          correctness_after: { type: 'string' }, note: { type: 'string' },
          head_before: { type: 'string' }, head_after: { type: 'string' },
          exact_patch: { type: 'boolean' }, transaction_valid: { type: 'boolean' },
        }, ['committed', 'transaction_valid']) });
      if (!(res && res.transaction_valid === true && res.exact_patch === true &&
            Array.isArray(res.committed) && res.committed.includes(enablingKeeps[0].patch) &&
            String(res.head_before || '').trim() && String(res.head_after || '').trim() &&
            res.head_before !== res.head_after)) {
        throw new Error(
          `STRICT AUTONOMY ABORT: enabling commit transaction for round ${round} is not ` +
          'independently accounted. Canonical may have changed; do not continue.');
      }
    }
    const reportedLanded = new Set(Array.isArray(res && res.committed) ? res.committed : []);
    const enablingCommitCorrect = !!(res && String(res.correctness_after || '').toLowerCase().startsWith('pass'));
    const landed = enablingCommitCorrect ? reportedLanded : new Set();
    if (reportedLanded.size && !enablingCommitCorrect) {
      const msg = `r${round} enabling commits changed canonical but correctness_after did not pass`;
      CHAIN_CAVEATS.push(msg);
      if (STRICT_AUTONOMY) throw new Error(`STRICT AUTONOMY ABORT: ${msg}`);
    }
    if (landed.size && CHAIN_BASELINE == null) {
      CHAIN_BASELINE = { round, cumulative, note:
        'Pinned at the first enabling commit. Every enabling step below makes the canonical tree ' +
        'slower, so the terminal step must state its claim against THIS number, not against the ' +
        'canonical it will actually be measured on.' };
      log(`CHAIN BASELINE PINNED at round ${round}, cumulative=${cumulative.toFixed(4)}x. ` +
          CHAIN_BASELINE.note);
    }
    const enablingAbsorbed = [];
    let enablingChain = '';
    for (const r of enablingKeeps) {
      if (!landed.has(r.patch)) {
        log(`ENABLING NOT LANDED ${r.d.id}: the patch did not apply to the canonical tree. It stays ` +
            'on the shelf; the rung it enables remains owed.');
        CHAIN_CAVEATS.push(`${r.d.id} [enabling, did not apply]: ${r.enabling_verdict.enables} still owed.`);
        enablingUnlanded.push(r);
        continue;
      }
      enablingLandedIds.add(r.d.id);
      CHAIN_DEBT.push({ round, id: r.d.id, enables: r.enabling_verdict.enables,
        cost_pct: r.enabling_verdict.cost_pct });
      CHAIN_CAVEATS.push(`${r.d.id} [enabling, committed r${round}]: ${r.enabling_verdict.reason}`);
      for (const f of (Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [])) {
        const nf = shelfFile(f); if (nf) enablingAbsorbed.push(nf);
      }
      if (!enablingChain) enablingChain = String(r.enabling_verdict.enables || '').trim();
    }
    // Record what the enabling commits took on, TAGGED with the chain they belong to. This closes the
    // blind spot the winner-only recording left -- a shelf entry on an UNRELATED chain is now aged
    // against these files too -- while the chain tag keeps the SAME chain's consumer half from being
    // aged off by its own producer. Merged with any winner absorption already recorded this round.
    if (enablingAbsorbed.length) {
      const prev = absorbedByRound[round];
      const prevFiles = Array.isArray(prev) ? prev : (prev && prev.files) || [];
      const prevChain = (prev && !Array.isArray(prev) && prev.chain_id) || enablingChain;
      absorbedByRound[round] = { files: [...new Set([...prevFiles, ...enablingAbsorbed])], chain_id: prevChain };
    }
    // Committing these does NOT advance cumulative. They are known to be slower; recording them as
    // progress would make the run's headline number a measure of how much overhead it has installed.
    log(`Round ${round}: ${landed.size} enabling step(s) committed on function. cumulative unchanged ` +
        `at ${cumulative.toFixed(4)}x — an enabling step is not a win and must not move the headline.`);
    if (res && res.correctness_after && !String(res.correctness_after).toLowerCase().startsWith('pass')) {
      log(`ENABLING COMMIT WARNING r${round}: correctness after the enabling commits reads ` +
          `"${res.correctness_after}". The canonical tree is the input to every later round; a chain ` +
          'that lands broken is worse than a chain that never lands.');
      CHAIN_CAVEATS.push(`r${round} enabling commit: correctness_after=${res.correctness_after}`);
    }
  }

  // Only landed state can close a staged rung. Evidence is evaluated earlier, but a commit agent
  // can fail or skip the patch; marking the rung before that transaction is how the workflow
  // previously declared success on code absent from canonical.
  for (const r of clean) {
    if (!(r.d && r.d.roadmap_rung) || r.d.roadmap_rung === 'off_ladder') continue;
    const role = stepRoleOf(r.d);
    let outcome = rungOutcomeOf(r, role);
    const contract = r.autonomy_contract;
    const evidencePass = !r.d.target_shape ? true
      : (STRICT_AUTONOMY ? !!(contract && contract.accepted) : !!(contract && contract.complete));
    outcome = landedRungOutcome(outcome, role, !!r.d.target_shape, evidencePass,
      winnerCommitted, !!(winner && winner.id === r.d.id), enablingLandedIds.has(r.d.id));
    r.rung_outcome = outcome;
    recordRungOutcome(r.d.roadmap_rung, outcome);
    if (outcome === 'measured') LADDER_MEASURED.add(r.d.roadmap_rung);
  }
  {
    const owed = openRungs(LADDER, rungTally);
    if (owed.length) {
      log(`Round ${round}: rungs still owed after evidence + commit gates: ` +
        owed.map((c) => `${c.id}(${c.last_outcome}, ${c.attempts} attempt(s))`).join(', ') +
        '. A staged rung closes only when its exact verified patch exists in canonical.');
    }
  }

  // The outstanding balance, recomputed every round against what has actually been measured.
  {
    const dr = chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED);
    if (dr.caveat) {
      log(`Round ${round}: ${dr.caveat}`);
      if (dr.overdue.length) CHAIN_CAVEATS.push(`r${round} OVERDUE: ${dr.caveat}`);
    }
  }

  const enablingLanded = CHAIN_DEBT.filter((e) => e.round === round).length;
  // Did this round measure anything at all? Decided before the counters move, by the same
  // admissibility standard promotion uses, so a reading that is VOID for a candidate cannot be
  // evidence for stopping. See roundEvidence.
  const ev = roundEvidence(clean, stepRoleOf, rungOutcomeOf);
  if (ev.measured) noEvidence = 0;
  else noEvidence++;
  // ON-HARDWARE accounting. A round "touched hardware" iff some clean direction reported its switched
  // path executed on the device this round (activation_on_hardware=yes). Reset on touch; else increment.
  // Live under EVERY objective when REQUIRE_HW_ACTIVATION is set — unlike noImprove/noEvidence, this is
  // not a speed or reading stop, so working_kernel does not exempt it. A round that reached the card even
  // to crash at runtime resets it; only a run stuck never-reaching-hardware (the static-scaffold pathology,
  // and the trace-time crash that hides there) trips it. Hard-stop + surface, per the operator's choice.
  if (REQUIRE_HW_ACTIVATION) {
    const touched = clean.some((r) => r && r.ver &&
      String(r.ver.activation_on_hardware || '').toLowerCase() === 'yes');
    if (touched) noHardware = 0;
    else noHardware++;
    if (!touched) {
      log(`Round ${round}: NO candidate reached the device this round (none reported ` +
        `activation_on_hardware=yes). noHardware=${noHardware}/${MAX_NO_HARDWARE} consecutive such rounds. ` +
        'A round banked entirely on compile/static/CPU screens is not on-device progress — it is exactly ' +
        'how a fused ON-path deferred its JIT-trace crash across rounds. At the cap the wave hard-stops so ' +
        'a human fixes the ON-path or the lease, rather than spending another lease on OFF scaffolding.');
    }
    if (noHardware >= MAX_NO_HARDWARE) {
      stopReason = `${noHardware} consecutive round(s) never executed the switched path on hardware ` +
        `(the no-hardware cap, ${MAX_NO_HARDWARE}). ${BUDGET - dispatched} of ${BUDGET} budget unit(s) ` +
        'were left unspent. This is not an exhausted search and not a slow kernel: the ON path never ' +
        'reached a card — it may not even trace/launch there. Fix the activation or the lease before ' +
        'planning another direction; do not bank more compile-only scaffolding.';
      log(`Round ${round}: hard-stop — ${stopReason}`);
      break;
    }
  }
  if (improved) {
    // noImprove was already reset at the commit above.
  } else if (enablingLanded) {
    // A round that landed a prerequisite produced verified, carried-forward progress; it just was
    // not progress the headline number can express. Charging it to the stopping criterion is how a
    // run gives up two rounds before the step that pays for all of them. MAX_NO_IMPROVE is a guard
    // against a search that has stopped finding anything, and this search has not.
    log(`Round ${round}: NOT counted toward noImprove — ${enablingLanded} enabling step(s) landed. ` +
        'A prerequisite that passed functional acceptance and entered the canonical tree is progress ' +
        'the cumulative speedup cannot show.');
  } else if (!ev.measured) {
    // No direction this round produced a reading the round is allowed to believe. That is a
    // harness, lease or reporting fault, not a search-space fault, and charging it to the stopping
    // criterion ends the run on the strength of experiments that were never performed.
    log(`Round ${round}: NOT counted toward noImprove — none of the ${ev.total} direction(s) ` +
        `produced an admissible measurement, so the round produced no evidence about the kernel ` +
        `at all. ${ev.gaps.join('; ')}. noEvidence=${noEvidence}/${MAX_NO_IMPROVE} consecutive ` +
        `such rounds; at the cap the run stops, because the thing to fix is the instrument, not ` +
        `the direction. Fix the lease or the activation before spending more budget.`);
  } else {
    noImprove++;
  }

  // --- (e3) Shelve this round's verified non-winners --------------------
  // Everything that PASSED independent verification and merely lost keeps its patch instead of
  // collapsing to a number in history.rounds. A 1.03x that lost to a 1.09x is a finished, verified
  // piece of work; on an 8-round budget with one collective lease per round, re-deriving it later
  // is the most expensive thing the loop can do.
  //
  // The integrated candidate is NOT shelved: it is not an independent direction, it is a
  // combination of things already on the shelf, and shelving it would offer the same work twice.
  {
    // An enabling step that passed on function but did not APPLY is verified work with a patch and
    // no home. The shelf is exactly the place for that: the rung it enables is still owed, and the
    // next round should be offered the work rather than made to re-derive it.
    const unlandedEntries = enablingUnlanded.map((r) => ({
      source: `enabling ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
      geomean: metricScore(r.ver) || 1.0, per_case: (r.ver && r.ver.per_case) || [], patch: r.patch,
      touched_files: Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [],
      // Chain tag: the terminal rung this producer half enables. On the shelf, the consumer half that
      // eventually lands must not age this entry off -- they are two halves of the same fusion.
      chain_id: String((r.enabling_verdict && r.enabling_verdict.enables) || (r.d && r.d.enables) || '').trim(),
    }));
    // Correct, activated patches that verifyCompleted but were kept OUT of `candidates` only by the
    // speed gate -- i.e. correct-but-slower results in speedup mode (a regression that composes with
    // another patch is exactly what the shelf is for). Excluded FOR CAUSE stay excluded: an
    // attribution-rejected gap-win must not be re-offered (the doctrine is to implement the barrier
    // directly, not resurface a megakernel), an inactive patch never ran, and an enabling step has
    // its own path. These would otherwise vanish -- correct work with a patch, lost to nothing.
    const inVerified = new Set(verified.map((r) => r.d && r.d.id));
    const slowLosers = clean.filter((r) => r.patch && !r.inactive && !r.attribution_rejected &&
      stepRoleOf(r.d) !== 'enabling' && verifyCompleted(r.ver) &&
      String(r.ver.correctness || '').toLowerCase().startsWith('pass') &&
      !inVerified.has(r.d && r.d.id) && metricScore(r.ver) <= 1.0)
      .map((r) => ({
        source: `slow ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        geomean: metricScore(r.ver) || 1.0, per_case: (r.ver && r.ver.per_case) || [], patch: r.patch,
        touched_files: Array.isArray(r.ver && r.ver.touched_files) ? r.ver.touched_files : [],
        chain_id: String((r.d && (r.d.enables || r.d.roadmap_rung)) || '').trim(),
      }));
    if (slowLosers.length) log(`SHELF r${round}: keeping ${slowLosers.length} correct-but-slower ` +
      `patch(es) [${slowLosers.map((c) => `${c.id}@${c.geomean.toFixed(3)}x`).join(', ')}] that lost only ` +
      `on speed -- they may compose with a later patch, and re-deriving them costs a lease.`);
    const losers = candidates.filter((c) => c !== winner && c.source !== 'integrated' && c.patch)
      .concat(unlandedEntries).concat(slowLosers);
    if (losers.length) {
      const added = shelfAdd(shelf, losers, round, SHELF_MAX);
      shelf = added.shelf;
      const noFiles = losers.filter((c) => !(c.touched_files || []).length);
      SHELF_STATS.shelved += losers.length;
      SHELF_STATS.withheld_unknown_footprint += noFiles.length;
      log(`SHELVED r${round}: ${losers.map((c) => `${c.id}@${c.geomean.toFixed(3)}x`).join(', ')} ` +
          `(${shelf.length}/${SHELF_MAX} on the shelf)` +
          (noFiles.length ? `. ${noFiles.map((c) => c.id).join(', ')} reported no touched_files, so ` +
            `their footprint is UNKNOWN and they will never be offered — unknown is not orthogonal.` : '') +
          (added.evicted.length ? ` Evicted the weakest: ${added.evicted.map((c) => c.id).join(', ')}.` : ''));
    }
    // What CANONICAL took on. Recorded only when the winner was actually committed — a winner that
    // did not clear MIN_IMPROVE changed nothing, so nothing aged.
    if (improved && winner) {
      const took = (winner.touched_files || []).map(shelfFile).filter(Boolean);
      // Object form carries the winner's chain, so a shelved consumer half of the SAME chain is not
      // aged off by its own producer/earlier-stage landing. A standalone winner has no chain_id and
      // ages every unrelated shelf entry exactly as before. MERGED with any enabling-commit
      // absorption already recorded this round (the enabling commit block runs first); if the two
      // belong to different chains the tag is dropped, so the union ages conservatively.
      if (took.length) {
        const prev = absorbedByRound[round];
        const prevFiles = Array.isArray(prev) ? prev : (prev && prev.files) || [];
        const prevChain = (prev && !Array.isArray(prev) && prev.chain_id) || '';
        const wChain = String(winner.chain_id || '').trim();
        const chain_id = !prevChain ? wChain : (prevChain === wChain ? wChain : '');
        absorbedByRound[round] = { files: [...new Set([...prevFiles, ...took])], chain_id };
      } else log(`SHELF WARN r${round}: the committed winner ${winner.id} reported no touched_files, so ` +
        `nothing was recorded as absorbed and shelved patches will NOT be aged against it. Every ` +
        `offer from here on may be stale in a way this check cannot see.`);
    }
  }

  // A round in which every direction was INACTIVE produced no evidence about the kernel. Its
  // distilled insights are tagged accordingly rather than being trusted as measurements.
  const allInactive = clean.length > 0 && clean.every(r => r.inactive);

  // --- update cross-round memory (insight blackboard + hypothesis ledger)
  const mem = await agentT(
    roleAgent('tech_lead', 'update_memory', 'Distill durable insights + update the hypothesis ledger.', {
      EVAL_DIR, ROUND: round, SKILL_DIR: WORKFLOW_DIR,
      ROUND_RESULTS: clean.map(r => ({ id: r.d.id, title: r.d.title, specialty: r.d.specialty,
        expected: r.d.expected_speedup, claimed: r.eng ? r.eng.speedup_geomean : 0,
        verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
        // The rung this direction was dispatched under. The ledger has a `roadmap_rung` column, so
        // without this the TechLead has to guess which rung each row settled — and a rung graded
        // dead_end by guess is a rung nothing above it will ever be built on.
        roadmap_rung: r.d.roadmap_rung || null, rung_deviation: r.d.rung_deviation || null,
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      // The ladder and what is still unspent on it. update_memory writes the memory the NEXT wave
      // reads; an unreached rung that is not written down here is a rung the next wave re-derives
      // from scratch or never sees. This is the only phase positioned to carry it across waves.
      ROADMAP_LADDER: LADDER, LADDER_DISPATCHED: [...dispatchedRungs],
      LADDER_COMPLETED: [...LADDER_MEASURED],
      // The ladder minus what actually produced a number, with each rung's attempt count and last
      // outcome. This is the entry the role writes into STATE.json as `open_rungs` and the next
      // wave's analyze fast path rebuilds its ladder from. LADDER_DISPATCHED cannot serve: it says
      // a rung was taken, not that taking it produced anything.
      OPEN_RUNGS: openRungs(LADDER, rungTally),
      // The unpaid balance travels to the next wave with everything else, for the same reason the
      // shelf does: a canonical tree carrying committed prerequisites is not the same artifact as a
      // clean one, and the next wave has to know which it inherited.
      ...(CHAIN_DEBT.length ? {
        CHAIN_DEBT: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
        CHAIN_BASELINE,
      } : {}),
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase,
        // The shelf travels to the next wave on the same road as the ledger: written into STATE.json
        // here, read back as setup.prior_state.shelf there. It is passed through as data the role
        // must copy VERBATIM, not as something to summarise — a shelf entry that loses its `patch`
        // path or its `files` list is not a smaller shelf entry, it is an offer that can no longer
        // be made or can no longer be checked for conflict.
        SHELF: shelf, ABSORBED_FILES: absorbedByRound } : {}),
      ...(SHARED_KB ? { SHARED_KB, TARGET_LANGUAGE } : {}),
    }),
    { phase: 'Optimize', label: `tech_lead:memory r${round}`, schema: MEMORY_SCHEMA });
  if (mem) {
    if (mem.insights) {
      const before = insightBook.length;
      const merged = mergeInsights(insightBook, mem.insights, round, allInactive, 40);
      insightBook = merged.book;
      history.insights = renderInsights(insightBook);
      const added = insightBook.filter((e) => e.first_round === round).length;
      log(`Round ${round} memory: ${added} new insight(s), ${before} carried forward, ` +
          `${insightBook.length} on the board` + (allInactive ? ' (this round produced NO evidence — its entries are tagged FROM-VOID-ROUND)' : ''));
      // Never silent. A board that drops findings without saying so is the bug this replaced.
      for (const e of merged.evicted) log(`INSIGHT EVICTED (board full, last seen r${e.last_round}): ${e.text}`);
    }
    // Logged one line per fact, and logged LOUDLY, because the whole failure being fixed is that
    // these were written down somewhere nobody read. A delta that does not generalize is kept too
    // but marked: the report wants it, `knowledge/` does not.
    if (Array.isArray(mem.knowledge_delta)) {
      for (const k of mem.knowledge_delta) {
        if (!k || !k.fact || !k.evidence) continue;
        KNOWLEDGE_DELTA.push({ round, ...k });
        log(`KNOWLEDGE DELTA r${round} [${k.generalizes ? `card: ${k.card || 'UNPLACED'}` : 'this operator only'}]: ` +
            `${k.fact} | evidence: ${k.evidence}` +
            (k.contradicts ? ` | would otherwise have been read as: ${k.contradicts}` : ''));
      }
    }
    if (mem.ledger) history.ledger = history.ledger.concat(mem.ledger);
    if (mem.bottleneck_now) history.bottleneck_now = mem.bottleneck_now;
    if (mem.suggest_next) history.suggest_next = mem.suggest_next;
  }
  history.rounds.push({
    round,
    directions: directions.map(d => ({ id: d.id, title: d.title, specialty: d.specialty })),
    results: clean.map(r => ({ id: r.d.id, claimed: r.eng ? r.eng.speedup_geomean : 0,
      verified: r.ver ? r.ver.verified_geomean : 0, status: r.ver ? r.ver.status : (r.eng ? r.eng.status : 'none'),
      // Carried so the next round can tell "tried and failed" from "never actually ran".
      inactive: r.inactive || null })),
    integrate: integrate ? { conclusion: integrate.conclusion, geomean: integrate.best ? integrate.best.geomean : 0 } : null,
    winner: winner ? { source: winner.source, geomean: winner.geomean } : null,
    improved, cumulative,
  });
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, ` +
      `cumulative=${cumulative.toFixed(2)}x, measured ${ev.measured}/${ev.total} direction(s), ` +
      `noImprove=${noImprove}/${MAX_NO_IMPROVE}, noEvidence=${noEvidence}/${MAX_NO_IMPROVE}, ` +
      `budget=${dispatched}/${BUDGET}`);
  if (STRICT_AUTONOMY && AUTONOMY_ACCEPTANCE_REACHED) {
    stopReason = `strict autonomy acceptance was reached in round ${round}; the accepted terminal ` +
      `artifact is committed and ${BUDGET - dispatched} of ${BUDGET} budget unit(s) remain unspent.`;
    break;
  }
}

// Which of the four stops fired, in the loop's own order of evaluation. Recorded rather than
// inferred: the budget is denominated in DIRECTIONS while the scarce resource is GPU leases, so
// "the budget was not exhausted" and "there was still hardware to spend" are the same sentence
// only by accident, and a reader looking at the artifacts a day later cannot tell which stop it
// was. One wave ended with a lease in hand and the next round already named in its own plan.
if (!stopReason) {
  const left = BUDGET - dispatched;
  if (dispatched >= BUDGET) {
    stopReason = `the direction budget was exhausted (${dispatched}/${BUDGET} charged over ${round} round(s)). ` +
      'The budget counts directions, not GPU leases; a round of directions that never reached the ' +
      'hardware costs the same as a round that did.';
  } else if (!WORKING_KERNEL && noEvidence >= MAX_NO_IMPROVE) {
    stopReason = `${noEvidence} consecutive round(s) produced no admissible measurement, which is the ` +
      `max_no_improve cap (${MAX_NO_IMPROVE}). ${left} of ${BUDGET} budget unit(s) were left unspent. ` +
      'This is an instrument failure, not an exhausted search: the next wave should fix the lease ' +
      'or the activation path before it plans another direction.';
  } else if (!WORKING_KERNEL && noImprove >= MAX_NO_IMPROVE) {
    stopReason = `${noImprove} round(s) in a row measured a candidate and none beat the current best by ` +
      `more than ${(MIN_IMPROVE * 100).toFixed(0)}%, which is the max_no_improve cap (${MAX_NO_IMPROVE}). ` +
      `${left} of ${BUDGET} budget unit(s) were left unspent.`;
  } else {
    stopReason = `the loop ended after ${round} round(s) with ${left} of ${BUDGET} budget unit(s) unspent ` +
      'and no stop condition recorded.';
  }
}
log(`Rounds ended: ${stopReason}`);

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    ...(ACCEPTANCE_CAVEATS.length ? { ACCEPTANCE_CAVEATS } : {}),
    // Why the loop ended, and what was left on the table when it did. The report is where the next
    // wave reads its starting conditions from; "the run finished" and "the run ran out of things
    // to try" are different facts and only one of them is usually true.
    STOP_REASON: stopReason,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    PROFILE_SUMMARY: profileSummary,
    // The report's provenance paragraph must cite this, not recollection. UNRECORDED means the
    // report may not assert anything about what is or is not in the baseline.
    PRIOR_ART_SWEEP: PRIOR_ART_RECORDED ? JSON.stringify(PRIOR_ART) : 'UNRECORDED',
    // A control that passed but read HIGH is a standing caveat on every number below it. It is
    // supplied here so the Measurement-confidence section states it instead of quietly dropping it.
    ...(PC_OVERSHOOT ? { POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT } : {}),
    ...(PC_UNDERSHOOT ? { POSITIVE_CONTROL_UNDERSHOOT: PC_UNDERSHOOT } : {}),
    // Travels to the report because the thing it qualifies — whether the directions were derived
    // or asserted — is invisible in the final number, and by report time nobody re-reads the log.
    // What the shelf actually did this wave. Reported unconditionally once anything was shelved,
    // because "offered 6 across 4 rounds, 0 composed" and "never consulted" are opposite findings
    // that produce the same silence, and K is supposed to be sized off the first one. A wave that
    // shelves and never hits is a reason to lower K or drop the mechanism — but only if somebody
    // can see it happened.
    ...(SHELF_STATS.rounds_offered || SHELF_STATS.shelved ? { SHELF_ACTIVITY: SHELF_STATS } : {}),
    ...(OVERLAP_CAVEATS.length ? { OVERLAP_CAVEATS } : {}),
    // Acceptance criterion 1 (two launches). Present only when a candidate was NOT at the fused
    // shape or did not report its count, so a wave that reached two launches on every measured
    // candidate is byte-identical to before. TWO_LAUNCH_REACHED says whether any verified candidate
    // this wave actually hit the two-launch shape, because "criterion 1 met" and "criterion 1 never
    // checked" are the two states a reader must not confuse.
    ...(LAUNCH_CAVEATS.length ? { LAUNCH_CAVEATS, TWO_LAUNCH_REACHED } : {}),
    // Guard-scope caveats: an under-sampled bimodal target route, or a win read only off a skew rail.
    // Present only when something was off, so a wave measured cleanly on target is unchanged.
    ...(GUARD_CAVEATS.length ? { GUARD_CAVEATS } : {}),
    ...(ATTRIBUTION_CAVEATS.length ? { ATTRIBUTION_CAVEATS } : {}),
    // The half-built fusion, if there is one. A wave that committed prerequisites and never closed
    // the chain reports a cumulative speedup that is BELOW where it started, and the reason is not
    // in the number: the tree is carrying overhead it has not been paid back for. Both facts have to
    // reach the report or the wave reads as a failed search rather than an unfinished build.
    ...(CHAIN_CAVEATS.length ? { CHAIN_CAVEATS } : {}),
    ...(CHAIN_DEBT.length ? {
      CHAIN_DEBT_FINAL: chainDebtReport(CHAIN_DEBT, round, LADDER_MEASURED).open,
      CHAIN_BASELINE,
      CHAIN_NOTE:
        'Enabling steps are committed on FUNCTION (builds, path taken, correct, no deadlock), not ' +
        'on speed, because the producer half of a fusion cannot be faster on its own. Any rung ' +
        'still listed in CHAIN_DEBT_FINAL is a fusion that was started and not closed: state the ' +
        'cost it left in the tree, and state that the speedup for that chain is NOT YET MEASURED ' +
        'rather than reporting the degraded number as the result. CHAIN_BASELINE is what the ' +
        'terminal step must be compared against — the canonical tree already contains the ' +
        'overhead, so measuring against the canonical would credit the chain with removing its own ' +
        'cost.',
    } : {}),
    // Only present on a non-default objective, so a speedup wave's report is byte-identical to
    // before. On a working_kernel wave it is the first thing a reader needs: the wave's own numbers
    // are withdrawn, and the deliverable is whether an artifact ran.
    ...(WORKING_KERNEL ? { OBJECTIVE, OBJECTIVE_NOTE:
      'This wave was run for a WORKING FUSED ARTIFACT, not for speed. One direction per round, the ' +
      'commit gate was "does it run", and the no-improve stop was disabled because a debug round is ' +
      'non-improving by construction. Do NOT read a speedup out of this wave' +
      (PC_RAN ? '.' : ' — it ran no positive control, so every timing reading in it is void.') } : {}),
    ...(OBJECTIVE_CAVEATS.length ? { OBJECTIVE_CAVEATS } : {}),
    // The report is the only artifact a human reliably reads, so it is where the durable facts have
    // to surface. Its section must separate the ones that generalize (a card edit is owed) from the
    // ones that do not, and must carry the evidence with each — a fact quoted without the ISA dump
    // or sha that established it is not mergeable and will be rediscovered anyway.
    ...(KNOWLEDGE_DELTA.length ? { KNOWLEDGE_DELTA, KNOWLEDGE_DELTA_NOTE:
      'Facts this wave established about the HARDWARE/TOOLCHAIN, not about this kernel. Give them ' +
      'their own section titled "Knowledge deltas — merge into knowledge/". List the generalizing ' +
      'ones first with their target card and their evidence verbatim; list the operator-specific ' +
      'ones under a separate heading and say plainly that they are not card material. This workflow ' +
      'does not edit knowledge/ itself.' } : {}),
    ...(TASK_GRAPH_CAVEAT ? { TASK_GRAPH_CAVEAT } : {}),
    ...(PIPE_TABLE_CAVEAT ? { PIPE_TABLE_CAVEAT } : {}),
    // Recomputed against the FINAL dispatch record, not the round-0 one: the question the report
    // has to answer is which rungs the wave never reached, and that is only knowable at the end.
    // A wave that stops with its highest rung unproposed has produced a number and not an answer,
    // and that fact belongs next to the number rather than in a log nobody re-reads.
    ...(LADDER.length ? { ROADMAP_LADDER_FINAL: JSON.stringify(
      roadmapLadderGate(LADDER, [], LADDER_MEASURED)).slice(0, 3000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.task_graph
      ? { TASK_GRAPH: JSON.stringify(analysis.task_graph).slice(0, 4000) } : {}),
    ...(REQUIRE_TASK_GRAPH && analysis && analysis.resource_timeline
      ? { RESOURCE_TIMELINE: JSON.stringify(analysis.resource_timeline).slice(0, 3000) } : {}),
  }),
  { phase: 'Report', label: 'tech_lead:report', schema: REPORT_SCHEMA });

// ===========================================================================
// PHASE: Director validation + arbitration
// ===========================================================================
phase('Validate');
const validation = await agentT(
  roleAgent('director', 'validate', 'Independently validate the final patch vs the TRUE baseline.', {
    KERNEL_PATH_ORIG, EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR, GPU_ID: GPU_RESOURCE.specForIndex(0),
    APPLY_TO_ORIGINAL, COMMANDMENT,
    FINAL_PATCH: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
    TECH_LEAD_REPORTED_GEOMEAN: report ? report.final_speedup_geomean : cumulative,
    STRICT_AUTONOMY, PROMOTION_METRIC, TARGET_GUARDS, REGRESSION_GUARDS,
    LAUNCH_TARGET, REQUIRE_OVERLAP, REQUIRE_ATTRIBUTION, REQUIRE_ARTIFACT_DISTINCT,
    REQUIRED_REPLAYS, REQUIRED_PAIRS, REQUIRED_PAIRS_BY_GUARD,
    ACCURACY_METRIC, ACCURACY_THRESHOLD, AUTONOMY_ACCEPTANCE_REACHED,
    WORKFLOW_REVISION_START,
    ...(STRICT_AUTONOMY && KNOWN_REFERENCE_HASHES.length
      ? { KNOWN_REFERENCE_HASHES } : {}),
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const aggregatePrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
const finalPrimary = (TARGET_GUARDS.length || PROMOTION_METRIC === 'changed_kernel')
  ? promotionScore(validation || {}, PROMOTION_METRIC, TARGET_GUARDS,
    REGRESSION_GUARDS, aggregatePrimary)
  : aggregatePrimary;
const finalGuards = guardContract(validation || {}, TARGET_GUARDS, REGRESSION_GUARDS, aggregatePrimary);
const directorStatus = validation ? validation.validation_status : 'unknown';
const finalAttribution = (validation && validation.attribution) || {};
const scopedGuardFailure = finalMetricFailure(
  PROMOTION_METRIC, TARGET_GUARDS, finalPrimary, finalGuards, finalAttribution);
const strictProofFailure = STRICT_AUTONOMY &&
  (!AUTONOMY_ACCEPTANCE_REACHED || !(validation && validation.autonomy_acceptance_confirmed === true) ||
   !(validation && validation.workflow_unchanged === true));
// Correctness/apply failures outrank the more specific autonomy label; never soften `flagged` into
// `autonomy_incomplete`. Scoped regression guards are vetoes in every mode, not only strict mode.
const finalValidationStatus = finalStatusVerdict(directorStatus, scopedGuardFailure, strictProofFailure);
// A leaked lease is not a cosmetic problem: the orphan runs unread AFTER the report is filed, and
// while it runs it contends with whatever measured the numbers we just accepted. Surface it in the
// completion line so it is not buried in the validation JSON.
if (validation && validation.orphan_leases_swept > 0) {
  log(`WARNING: ${validation.orphan_leases_swept} orphaned GPU lease(s) were still alive at closeout ` +
      `and have been killed. Some direction backgrounded a lease job — its measurement is NOT in ` +
      `the report, and it may have contended with the validated numbers.`);
}

log(`COMPLETE. ${KERNEL_NAME}: verified ${TARGET_GUARDS.length ? `target-guard ${PROMOTION_METRIC}` :
    (HAS_WORKLOAD ? 'time-weighted' : 'geomean')} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${finalValidationStatus}). Results in ${EVAL_DIR}`);

return {
  mode: MODE,
  target_language: MODE === 'author' ? TARGET_LANGUAGE : undefined,
  authored: MODE === 'author' ? true : undefined,
  eval_dir: EVAL_DIR,
  kernel_name: KERNEL_NAME,
  workload_aligned: HAS_WORKLOAD,
  final_speedup: finalPrimary,                 // PRIMARY metric (weighted when workload-aligned)
  final_weighted: finalWeighted,
  final_geomean: finalGeomean,
  final_arithmetic: validation ? validation.director_verified_speedup_arithmetic : null,
  tech_lead_reported_geomean: report ? report.final_speedup_geomean : cumulative,
  validation_status: finalValidationStatus,
  autonomy_acceptance_reached: AUTONOMY_ACCEPTANCE_REACHED,
  autonomy_acceptance_confirmed: !!(validation && validation.autonomy_acceptance_confirmed === true &&
    validation.workflow_unchanged === true && directorStatus !== 'flagged' &&
    !scopedGuardFailure && !strictProofFailure),
  acceptance_caveats: ACCEPTANCE_CAVEATS,
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  stop_reason: stopReason,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
};
