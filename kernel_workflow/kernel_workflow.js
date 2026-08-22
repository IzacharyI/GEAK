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
// Trees whose contents would constitute an imported answer. Given to VERIFY (to compare against) and
// deliberately NOT to engineers — handing them the list would be handing them the location.
const KNOWN_REFERENCE_PATHS = (Array.isArray(A.known_reference_paths) ? A.known_reference_paths
  : String(A.known_reference_paths || '').split(',')).map(s => String(s).trim()).filter(Boolean);
if (CAPABILITY_EVAL) {
  log(`CAPABILITY EVAL mode: prior art is advisory CONCLUSIONS only; engineers get no reference ` +
      `paths, and verify rejects byte-identical files as status:"plagiarized". ` +
      `known_reference_paths=${KNOWN_REFERENCE_PATHS.length ? KNOWN_REFERENCE_PATHS.join(' ') : '(none given — provenance check DISABLED)'}`);
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
if (CAPABILITY_EVAL && KNOWN_REFERENCE_PATHS.length) {
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

// Verdict vocabulary is three-valued on purpose. `skipped` exists so that "the scanner could not run"
// can never be reported as `clean` — that collapse is the exact failure this whole subsystem exists
// to prevent, and a two-valued schema would force the agent to lie.
const CONTAINMENT_GATE_SCHEMA = obj({
  verdict: { type: 'string', enum: ['clean', 'leak', 'skipped'] },
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
  prior_state: obj({
    cumulative: { type: 'number' }, insights: { type: 'array', items: { type: 'string' } },
    ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
    bottleneck_now: { type: 'string' }, best_per_case: perCase,
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
  candidate_directions: { type: 'array', items: { type: 'object', additionalProperties: true } },
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
  //   {ran, measured_pct, expected_lo, expected_hi, passed, reps, null_arm_pct, note}
  // `control_pairs_pct` and `null_pairs_pct` — the individual paired deltas, not just their medians —
  // are read by the gate: sign agreement across pairs and the WORST null pair are what separate a
  // small real effect from a small piece of drift, and a median hides both.
  positive_control: { type: 'object', additionalProperties: true },
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
  // How the number above was actually obtained. `reps` = interleaved A,B,A,B pairs (NOT total runs);
  // `null_arm_pct` = the delta measured by an arm doing byte-identical work, i.e. this run's own
  // noise floor. A claimed win smaller than |null_arm_pct| is unreadable no matter how it was
  // computed. Absent/low values do not void the result — they downgrade it to PROVISIONAL and are
  // surfaced in the log and report, because "1 rep, no null arm" has already produced a -0.44%
  // "win" that sat inside a 1.45% per-case spread.
  reps: { type: 'number' }, null_arm_pct: { type: 'number' },
  // Did the patched code path actually EXECUTE during the measured run? `yes` requires the
  // path_marker to have been observed; `no` means the arm labelled "candidate" ran baseline code,
  // which makes the whole comparison void rather than negative. `unknown` is treated as `no` — an
  // unproven activation is exactly the state that produced a 1.000x on an unexercised patch.
  activation_confirmed: { type: 'string' },  // yes|no|unknown
  activation_evidence: { type: 'string' },   // the command + the marker output that proves it
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
}, ['status', 'verified_geomean']);

const INTEGRATE_SCHEMA = obj({
  attempted: { type: 'boolean' },
  combos_tried: { type: 'array', items: { type: 'object', additionalProperties: true } },
  best: { type: 'object', additionalProperties: true },
  improved_over_best_individual: { type: 'boolean' },
  conclusion: { type: 'string' }, notes: { type: 'string' },
}, ['attempted', 'conclusion']);

const MEMORY_SCHEMA = obj({
  insights: { type: 'array', items: { type: 'string' } },
  ledger: { type: 'array', items: { type: 'object', additionalProperties: true } },
  bottleneck_now: { type: 'string' }, suggest_next: { type: 'string' },
}, ['insights']);

const COMMIT_SCHEMA = obj({
  committed: { type: 'boolean' }, current_best_diff: { type: 'string' }, note: { type: 'string' },
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
    ...(STATE_DIR ? { STATE_DIR } : {}),
  }),
  { phase: 'Setup', label: 'director:setup', schema: SETUP_SCHEMA });
if (!setup || !setup.eval_dir) throw new Error('Setup failed: director did not return an eval_dir');
const EVAL_DIR = setup.eval_dir;
const CANONICAL = setup.workspace;       // canonical current-best workspace (advances each round)
const KERNEL_NAME = setup.kernel_name;
const COMMANDMENT = `${EVAL_DIR}/COMMANDMENT.md`;
log(`Setup done. EVAL_DIR=${EVAL_DIR}`);

// CONTAINMENT ENFORCEMENT — the startup check above is a path filter and it missed the two leaks that
// actually happened. Both were CONTENT: a control workspace holding the applied reference patch left
// inside the tree, and a knowledge card that opened with the reference branch and commit. Neither is
// a known_reference_path, so neither could ever have failed the startup check. This gate runs the two
// scanners that do see them, and it runs AFTER Setup because Setup is what materialises the eval dir,
// the baseline copy and the inherited assets — sweeping before them scans an empty tree and passes.
// It is an agent because workflow scripts have no fs/child_process: the check has to be executed by
// something with a shell, and one cheap agent beats a comment asking a human to remember.
if (CAPABILITY_EVAL && RUN_TREE_ANCESTOR) {
  const gate = await agentT(
    `Run two containment scanners and report their exit codes verbatim. Do not fix anything, do not ` +
    `read any file they flag, do not summarise the flagged content — reporting the paths is the whole job.\n\n` +
    `1. bash ${WORKFLOW_DIR}/scripts/reference_leak_sweep.sh --tree ${RUN_TREE_ANCESTOR}\n` +
    `2. bash ${WORKFLOW_DIR}/scripts/skill_address_scan.sh --skills-dir ${EXPERT_SKILLS_DIR || '(skip: expert skills off)'} ` +
    `--scan-root ${RUN_TREE_ANCESTOR}\n\n` +
    `If a script is missing or its --skills-dir is empty, record that step as "skipped" with the reason ` +
    `and do NOT call it clean. "The scanner did not run" and "the scanner found nothing" are different ` +
    `results and this run has already been damaged twice by conflating them.`,
    { phase: 'Setup', label: 'containment gate', schema: CONTAINMENT_GATE_SCHEMA });
  if (!gate) {
    log('CONTAINMENT GATE: the gate agent returned nothing. Treating as UNKNOWN, not clean — the run ' +
        'continues, but any capability claim from it is unsupported until the scanners are run by hand.');
  } else if (gate.verdict === 'leak') {
    throw new Error(
      `CONTAINMENT GATE FAILED: a copy of the answer is reachable inside ${RUN_TREE_ANCESTOR}.\n` +
      `${(gate.findings || []).join('\n')}\n` +
      `The run is stopped before any budget is spent, because a candidate derived here cannot be ` +
      `distinguished from a candidate copied here. Move the offending artifact outside the tree (mv, ` +
      `not rm, so it stays available to verify) and relaunch. Set capability_eval=false if the port ` +
      `is what you actually want.`);
  } else {
    log(`CONTAINMENT GATE ${gate.verdict}: ${gate.note || ''}`);
  }
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
const analysis = await agentT(
  roleAgent('tech_lead', 'analyze', 'Analyze the kernel and write the roadmap.', {
    WORKSPACE: CANONICAL, EVAL_DIR, TASK, SKILL_DIR: WORKFLOW_DIR,
    KERNEL_KNOWLEDGE_DIR,
    // Authoritative resolved rank count (from gpus_per_job | op_spec.resource | job_gpu_ids).
    // >1 is what makes the `distributed` specialty eligible; OP_SPEC.resource may be absent.
    GPUS_PER_JOB: String(GPU_RESOURCE.gpusPerJob),
    ...(CAPABILITY_EVAL ? { CAPABILITY_EVAL: '1' } : {}),
    ...RESUME_INPUT,
  }),
  { phase: 'Analyze', label: 'tech_lead:analyze', schema: ANALYZE_SCHEMA });
log(`Analyze done. kernel_type=${analysis ? analysis.kernel_type : '?'}`);

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
  const where = pa.implemented_at || '?';
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
  for (const f of ['roadmap.md', 'codebase_context.md', 'analysis.json']) {
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
  const ran = pc.ran !== false && Number.isFinite(got) && Number.isFinite(lo) && Number.isFinite(hi);

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
  const resolved = Number.isFinite(nullWorst) && mGot >= RESOLVE_K * nullWorst;
  const undershoot = ran && constructed && Math.sign(got) === wantSign &&
    mGot < mLo && mGot >= mLo * UNDERSHOOT_FRAC && resolved && signUnanimous;

  // Wrong sign is an insensitivity failure, not an overshoot: the loop did not see the change it was
  // handed, whatever else it saw.
  const tooSmall = ran && (Math.sign(got) !== wantSign || (mGot < mLo && !undershoot));
  const absurd = ran && Number.isFinite(ABSURD) && mGot > ABSURD;
  const overshoot = ran && mGot > mHi && !absurd;
  const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
  // <</REPLAY:pc_gate>>
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
                    `${RESOLVE_K}x an effect must clear to be told apart from the interleave. Quiet the ` +
                    `interleave or raise the injected cost. `
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
let bestPerCase = BASELINE_PER_CASE;
let finalWinner = null;      // {geomean, arithmetic, per_case, patch, source}
const history = { insights: [], ledger: [], rounds: [], bottleneck_now: profileSummary ? profileSummary.bottleneck : 'unknown', suggest_next: '' };
// The blackboard behind `history.insights`. Kept as records rather than strings so an insight can
// carry the round it came from and whether that round produced evidence at all.
let insightBook = [];

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
  log(`RESUMED from STATE_DIR: cumulative=${cumulative.toFixed(3)}x, ${history.insights.length} insights, ${history.ledger.length} ledger entries carried forward.`);
}

while (dispatched < BUDGET && noImprove < MAX_NO_IMPROVE) {
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
      KERNEL_KNOWLEDGE_DIR, KK_OPERATOR, KK_LANGUAGE, KK_REFS,
      ...KB_INPUTS,
    }),
    { phase: 'Optimize', label: `tech_lead:plan r${round}`, schema: PLAN_SCHEMA });

  if (!plan || plan.stop || !plan.directions || plan.directions.length === 0) {
    log(`Round ${round}: TechLead chose to stop. ${plan ? plan.reasoning || '' : ''}`);
    break;
  }

  let directions = plan.directions.slice(0, remaining).map((d, i) => ({
    ...d,
    idx: i,
    id: d.id || `r${round}_d${i}`,
    gpu_id: GPU_RESOURCE.specForIndex(i),
    out_dir: `${EVAL_DIR}/round_${round}/engineer_${i}`,
  }));
  // deep_explore is a DEDICATED-ROUND, heavyweight mandate: if the plan includes one, run ONLY it this
  // round (its broad ground-up rewrite touches many files and can't be merged with specialist patches),
  // and charge DEEP_COST against the budget. Otherwise each specialist direction costs 1.
  const deepDir = directions.find(d => d.specialty === 'deep_explore');
  if (deepDir) directions = [deepDir];
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
\`\`\`
${readLine} If KK_OPERATOR is non-empty, also consult the operator/language SOTA cards under
KERNEL_KNOWLEDGE_DIR per your role's "operator/language SOTA knowledge (REFERENCE ONLY)" section
(facts/how-to only; measure everything; never go below baseline).
Save best_patch.diff via \`cd <KERNEL_PATH> && git diff > ${d.out_dir}/best_patch.diff\` when geomean>1.0.

## Inputs
${cfg({
        SPECIALTY: d.specialty,
        DIRECTION: { id: d.id, title: d.title, focus_files: d.focus_files || [], expected_speedup: d.expected_speedup, prompt: d.prompt },
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

    (prev) => {
      const { d, eng } = prev;
      const patch = `${d.out_dir}/best_patch.diff`;
      if (!eng || eng.status === 'failed' || !(primSpeedup(eng) > 1.0)) {
        return { d, eng, ver: null };
      }
      return agentT(
        roleAgent('verify_engineer', 'verify', 'Independently re-measure this candidate patch.', {
          CANONICAL, PATCH: patch, VERIFY_DIR: `${d.out_dir}/verify`,
          GPU_ID: d.gpu_id, SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
          // Verify applies specialty-specific gates (see verify_engineer.md step 4c: a
          // `distributed` patch can be numerically correct and still deadlock).
          ...(d.specialty ? { SPECIALTY: d.specialty } : {}),
          // What the engineer says turns its code ON, verbatim. If it declared nothing, verify is
          // told so explicitly rather than being left to assume default-ON — the assumption is the
          // bug: an undeclared switch and a genuinely default-ON patch look identical from here.
          ACTIVATION: (eng && eng.activation) ? JSON.stringify(eng.activation) : 'UNDECLARED',
          ...(HARNESS_ADDENDUM ? { HARNESS_ADDENDUM } : {}),
          ...(REQUIRE_GRAPH_CAPTURE ? { REQUIRE_GRAPH_CAPTURE: '1' } : {}),
          // Verify is the ONLY role that learns where the reference trees are. It needs the
          // locations to compare against them; engineers must not have them at all.
          ...(CAPABILITY_EVAL && KNOWN_REFERENCE_PATHS.length
            ? { KNOWN_REFERENCE_PATHS: KNOWN_REFERENCE_PATHS.join(' ') } : {}),
        }),
        { phase: 'Verify', label: `verify ${d.id}`, schema: VERIFY_SCHEMA }
      ).then((ver) => ({ d, eng, ver, patch }));
    }
  );

  const clean = results.filter(Boolean);

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
    const pct = ((primSpeedup(r.ver) - 1) * 100);
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
    const pct = ((primSpeedup(r.ver) - 1) * 100);
    log(`SAME BINARY ${r.d.id}: base and candidate resolved to the SAME compiled artifact` +
        `${sameHash ? ` (${hb})` : ''}, so its ` +
        `${Number.isFinite(pct) ? (pct >= 0 ? '+' : '') + pct.toFixed(2) + '%' : 'result'} measures ` +
        `nothing about the idea — it is VOID, not a null. The switch did not enter the compiler's ` +
        `cache key. Re-run with the switch anchored in the OUTER traced scope before this direction ` +
        `is called tried.`);
  }

  const verified = clean.filter(r => r.ver && r.ver.status === 'verified' &&
    r.ver.correctness === 'pass' && primSpeedup(r.ver) > 1.0 && !r.inactive);

  // `plagiarized` and `harness_modified` are already excluded by the filter above, but exclusion is
  // invisible — the direction just quietly stops existing, which reads like "it didn't work". Both
  // are load-bearing findings about the RUN rather than about the kernel, and the speedup that came
  // with them is exactly the thing that makes them tempting to accept, so print it alongside.
  for (const r of clean) {
    const st = r.ver && r.ver.status;
    if (st !== 'plagiarized' && st !== 'harness_modified') continue;
    const pct = ((primSpeedup(r.ver) - 1) * 100);
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
    const claimPct = (primSpeedup(r.ver) - 1) * 100;
    const reps = Number(r.ver.reps);
    const nullArm = Number(r.ver.null_arm_pct);
    const reasons = [];
    if (!Number.isFinite(reps) || reps < 5) reasons.push(`reps=${Number.isFinite(reps) ? reps : 'unreported'} (<5)`);
    if (!Number.isFinite(nullArm)) reasons.push('no null arm');
    else if (claimPct <= Math.abs(nullArm)) reasons.push(`claim ${claimPct.toFixed(2)}% <= null arm ${Math.abs(nullArm).toFixed(2)}%`);
    if (reasons.length) {
      r.provisional = reasons.join('; ');
      log(`PROVISIONAL ${r.d.id}: +${claimPct.toFixed(2)}% claimed but ${r.provisional}. ` +
          `Carried forward, but it must be re-measured with >=5 interleaved reps and a null arm ` +
          `before it is reported as a result.`);
    }
  }

  // --- (d) Build candidate list; integrate if >=2 verified --------------
  // `geomean` here is the PRIMARY metric used for sorting/gating/cumulative: the time-weighted
  // ratio-of-sums when a workload spec is active, else the unweighted geomean (unchanged behavior).
  // The raw unweighted geomean is retained separately for the report.
  let candidates = verified.map(r => ({
    source: `engineer ${r.d.id}`, id: r.d.id, title: r.d.title, specialty: r.d.specialty,
    geomean: primSpeedup(r.ver), geomean_unweighted: r.ver.verified_geomean,
    weighted: r.ver.verified_weighted != null ? r.ver.verified_weighted : null,
    arithmetic: r.ver.verified_arithmetic || r.ver.verified_geomean,
    per_case: r.ver.per_case || [], patch: r.patch,
  }));

  let integrate = null;
  if (verified.length >= 2) {
    phase('Merge');
    integrate = await agentT(
      roleAgent('integrator', 'integrate', 'Combine this round\'s verified patches into one best implementation.', {
        CANONICAL, INTEGRATE_DIR: `${EVAL_DIR}/round_${round}/integrate`,
        GPU_ID: GPU_RESOURCE.specForIndex(0), SKILL_DIR: WORKFLOW_DIR, COMMANDMENT, BASELINE_PER_CASE,
        BEST_INDIVIDUAL: Math.max(...candidates.map(c => c.geomean)),
        PATCHES: verified.map(r => ({ id: r.d.id, specialty: r.d.specialty, title: r.d.title,
          strategy: r.eng ? r.eng.strategy : '', verified_geomean: r.ver.verified_geomean,
          files: r.d.focus_files || [], patch: r.patch })),
        INSIGHTS: history.insights,
      }),
      { phase: 'Merge', label: `integrate r${round}`, schema: INTEGRATE_SCHEMA });
    const integPrim = integrate && integrate.best ? primSpeedup({
      verified_weighted: integrate.best.weighted, verified_geomean: integrate.best.geomean,
    }) : 0;
    if (integrate && integrate.conclusion === 'improved' && integrate.best &&
      integPrim > Math.max(...candidates.map(c => c.geomean))) {
      candidates.push({
        source: 'integrated', id: `r${round}_integrated`, title: 'integrated', specialty: 'integrate',
        geomean: integPrim, geomean_unweighted: integrate.best.geomean,
        weighted: integrate.best.weighted != null ? integrate.best.weighted : null,
        arithmetic: integrate.best.arithmetic || integrate.best.geomean,
        per_case: integrate.best.per_case || [], patch: integrate.best.patch_file,
      });
    }
  }

  candidates.sort((a, b) => b.geomean - a.geomean);
  const winner = candidates[0] || null;
  const improved = !!(winner && winner.geomean > cumulative * (1 + MIN_IMPROVE));

  // --- (e) Commit the winner into the canonical workspace ---------------
  if (improved) {
    await agentT(
      `You are the TechLead committing round ${round}'s winning patch into the canonical workspace.
\`\`\`bash
export GIT_PAGER=cat GIT_TERMINAL_PROMPT=0 GIT_EDITOR=true
cd ${CANONICAL}
git checkout -- .
# Try a plain apply first, then a 3-way apply (auto-reconciles context-line drift against the blobs)
# before falling back to a manual reconstruction. --3way resolves most "patch does not apply" cases
# that are just context offsets, so the manual path is only hit on a genuine semantic conflict.
git apply ${winner.patch} || git apply --3way ${winner.patch}
git -c user.email=team@workflow -c user.name=team add -A
git -c user.email=team@workflow -c user.name=team commit -q -m "round ${round} winner: ${winner.source} (${winner.geomean.toFixed(2)}x)"
git --no-pager diff "$(git rev-list --max-parents=0 HEAD)..HEAD" > ${EVAL_DIR}/current_best.diff
\`\`\`
If BOTH \`git apply\` and \`git apply --3way\` fail, inspect the patch and apply it manually (edit the
files to match the patch's intent), then \`add -A\` + commit. The applied source is NOT guaranteed to
match the patch verbatim after a hand-merge, so after committing, RE-RUN the COMMANDMENT correctness
check (cd ${CANONICAL} && the COMMANDMENT CORRECTNESS cmd via gpu_lock); only report committed=true if
it still passes. (When a clean \`git apply\`/\`--3way\` succeeds, correctness was already verified and a
re-check is not required.) Return JSON {committed, current_best_diff, note}.`,
      { phase: 'Merge', label: `commit r${round}`, schema: COMMIT_SCHEMA });
    cumulative = winner.geomean;
    bestPerCase = winner.per_case && winner.per_case.length ? winner.per_case : bestPerCase;
    finalWinner = winner;
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
  } else if (clean.length && clean.every(r => r.inactive)) {
    // Every direction this round measured code that never ran. That is a harness/activation fault,
    // not a search-space fault, and charging it to the stopping criterion would end the run on the
    // strength of experiments that were never performed.
    log(`Round ${round}: NOT counted toward noImprove — every direction was INACTIVE, so the round ` +
        `produced no evidence about the kernel at all. Fix activation before spending more budget.`);
  } else {
    noImprove++;
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
        notes: r.eng ? r.eng.notes : '' })),
      INTEGRATE: integrate, WINNER: winner ? { source: winner.source, geomean: winner.geomean } : null,
      IMPROVED: improved, REPROFILE_SHIFT: profileSummary ? profileSummary.shift_note : '',
      PRIOR_HISTORY: history,
      ...(STATE_DIR ? { STATE_DIR, CANONICAL, CUMULATIVE_SPEEDUP: cumulative, BEST_PER_CASE: bestPerCase } : {}),
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
  log(`Round ${round} done. winner=${winner ? winner.source + ' ' + winner.geomean.toFixed(2) + 'x' : 'none'}, cumulative=${cumulative.toFixed(2)}x, noImprove=${noImprove}`);
}

// ===========================================================================
// PHASE: Final report (TechLead)
// ===========================================================================
phase('Report');
const report = await agentT(
  roleAgent('tech_lead', 'report', 'Write the final report and the cumulative final patch.', {
    EVAL_DIR, WORKSPACE: CANONICAL, SKILL_DIR: WORKFLOW_DIR,
    HISTORY: history, FINAL_WINNER: finalWinner, BASELINE_PER_CASE,
    BASELINE_GEOMEAN_MS, CUMULATIVE_SPEEDUP: cumulative,
    PROFILE_SUMMARY: profileSummary,
    // The report's provenance paragraph must cite this, not recollection. UNRECORDED means the
    // report may not assert anything about what is or is not in the baseline.
    PRIOR_ART_SWEEP: PRIOR_ART_RECORDED ? JSON.stringify(PRIOR_ART) : 'UNRECORDED',
    // A control that passed but read HIGH is a standing caveat on every number below it. It is
    // supplied here so the Measurement-confidence section states it instead of quietly dropping it.
    ...(PC_OVERSHOOT ? { POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT } : {}),
    ...(PC_UNDERSHOOT ? { POSITIVE_CONTROL_UNDERSHOOT: PC_UNDERSHOOT } : {}),
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
    ...(HAS_WORKLOAD && report && report.final_speedup_weighted != null
        ? { TECH_LEAD_REPORTED_WEIGHTED: report.final_speedup_weighted } : {}),
    BASELINE_TIMING: BASELINE_PER_CASE,
  }),
  { phase: 'Validate', label: 'director:validate', schema: VALIDATE_SCHEMA });

const finalGeomean = validation ? validation.director_verified_speedup_geomean : cumulative;
// PRIMARY headline: the time-weighted speedup when workload-aligned, else the geomean (unchanged).
const finalWeighted = validation && validation.director_verified_speedup_weighted != null
  ? validation.director_verified_speedup_weighted : null;
const finalPrimary = HAS_WORKLOAD && Number.isFinite(finalWeighted) ? finalWeighted : finalGeomean;
// A leaked lease is not a cosmetic problem: the orphan runs unread AFTER the report is filed, and
// while it runs it contends with whatever measured the numbers we just accepted. Surface it in the
// completion line so it is not buried in the validation JSON.
if (validation && validation.orphan_leases_swept > 0) {
  log(`WARNING: ${validation.orphan_leases_swept} orphaned GPU lease(s) were still alive at closeout ` +
      `and have been killed. Some direction backgrounded a lease job — its measurement is NOT in ` +
      `the report, and it may have contended with the validated numbers.`);
}

log(`COMPLETE. ${KERNEL_NAME}: verified ${HAS_WORKLOAD ? 'time-weighted' : 'geomean'} ${finalPrimary ? finalPrimary.toFixed(2) : '?'}x` +
    `${HAS_WORKLOAD && Number.isFinite(finalGeomean) ? ` (unweighted geomean ${finalGeomean.toFixed(2)}x)` : ''}` +
    ` (status ${validation ? validation.validation_status : '?'}). Results in ${EVAL_DIR}`);

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
  validation_status: validation ? validation.validation_status : 'unknown',
  rounds: report ? report.rounds : round,
  budget_used: dispatched,
  budget_total: BUDGET,
  report_path: report ? report.report_path : `${EVAL_DIR}/tech_lead_report.md`,
  final_patch: report ? report.final_patch : `${EVAL_DIR}/final_patch.diff`,
};
