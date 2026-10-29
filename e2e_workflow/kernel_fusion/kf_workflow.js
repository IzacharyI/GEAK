export const meta = {
  name: 'kernel-fusion',
  description: 'Standalone KernelFusion stage that runs BEFORE the GEAK e2e workflow: AgentX (or fixed-shape) baseline -> sampled traces + shapes (kernel-extract launch/capture method) -> top-1 GPU-time trace -> semantics -> fusion screen -> observed-workload weighted unit tests -> combined apply-back A/B. Returns the fused overlay the e2e workflow then starts from.',
  whenToUse: 'Run through interface/run_kf_e2e.py, which chains this stage into the unchanged e2e workflow. Direct use: pass the same model/backend/tp/gpu_ids/workload args run_e2e.py would, plus eval_dir.',
  phases: [
    { title: 'Baseline', detail: 'one measured serving leg on the deployed stack (AgentX trace replay, or the fixed ISL/OSL/CONC sweep); its server log is the workload evidence' },
    { title: 'Sample', detail: 'kernel-extract method: log -> observed workload -> sampling points -> one owned probe server, one bounded profile window per point (traces + shapes); the top-1 decode point by GPU-time mass is the analysis trace' },
    { title: 'Semantics', detail: 'Pattern/Phase/Layer/Kernel tables from the analysis trace, decode boundaries/shapes from the graph-construction replay' },
    { title: 'Screen', detail: 'fusion_screen.py: chains -> initial benefit rank -> two Score passes -> adaptive Top-K' },
    { title: 'UnitSide', detail: 'every Top-K entry, alternative rung and pool item unit-tested; the gate is the observed-workload weighted speedup over every sampling point' },
    { title: 'StageFusion', detail: 'L3, opt-in (fusion_l3_mega): each expert-parallel MoE stage the screen listed goes to GEAK mode=mega as one multi-rank operator on the serving GPUs; a delivered candidate is packaged as one overlay for the combined A/B' },
    { title: 'ApplyBack', detail: 'author the unit-side passes as overlays, gate the stack with ONE combined A/B (base server vs stacked server, throughput + gsm8k)' },
    { title: 'Report', detail: 'publish 00_INDEX.md + kf_return.json for interface/run_kf_e2e.py' },
  ],
};

// ---------------------------------------------------------------------------
// RUNTIME SANDBOX CONTRACT (same as e2e_workflow.js): no filesystem, no child_process, no
// dynamic import, no wall-clock reads. Every file and every command is an agent's job; this script only
// sequences agents and carries their structured returns.
// ---------------------------------------------------------------------------
let _rawArgs = args;
if (typeof _rawArgs === 'string') {
  let _s = _rawArgs.trim();
  for (let _i = 0; _i < 8 && _s.length > 1; _i++) {
    try { _rawArgs = JSON.parse(_s); break; } catch (_e) { _s = _s.slice(0, -1).trim(); }
  }
}
const A = _rawArgs || {};
const WORKFLOW_DIR = String(A.workflow_dir || '').replace(/\/+$/, '');
if (!WORKFLOW_DIR) throw new Error('kernel-fusion: args.workflow_dir is required (the e2e_workflow directory)');
const KF_DIR = `${WORKFLOW_DIR}/kernel_fusion`;
const MODEL_PATH = String(A.model_path || '');
if (!MODEL_PATH) throw new Error('kernel-fusion: args.model_path is required');
const MODEL_NAME = String(A.model_name || MODEL_PATH.replace(/\/+$/, '').split('/').pop() || 'model');
const EXP_ROOT = String(A.exp_root || '');
const EVAL_DIR = String(A.eval_dir || (EXP_ROOT ? `${EXP_ROOT}/kernel_fusion` : '')).replace(/\/+$/, '');
if (!EVAL_DIR) throw new Error('kernel-fusion: args.eval_dir (or exp_root) is required');
const BACKEND = String(A.backend || 'sglang').trim().toLowerCase();
const GPU_IDS = String(A.gpu_ids != null ? A.gpu_ids : '0');
const GPU_LIST = GPU_IDS.split(',').map((s) => s.trim()).filter(Boolean);
const SERVING_TP = parseInt(A.tp != null ? A.tp : 1, 10);
const SERVING_GPU = String(A.serving_gpu != null ? A.serving_gpu
  : GPU_LIST.slice(0, Math.max(1, SERVING_TP)).join(',') || '0');
const WORKLOAD_SPEC = (A.workload_spec && typeof A.workload_spec === 'object') ? A.workload_spec : {};
const IS_AGENTX = String(WORKLOAD_SPEC.kind || '') === 'agentx_trace_replay';
const CONC = parseInt(IS_AGENTX && WORKLOAD_SPEC.concurrency != null ? WORKLOAD_SPEC.concurrency
  : (A.conc != null ? A.conc : 64), 10);
const ISL = parseInt(A.isl != null ? A.isl : 1024, 10);
const OSL = parseInt(A.osl != null ? A.osl : 1024, 10);
const WORKLOAD = IS_AGENTX
  ? { kind: 'agentx_trace_replay', conc: CONC, isl: ISL, osl: OSL,
      note: 'ISL/OSL are a kernel-targeting placeholder; the AgentX client owns the load', spec: WORKLOAD_SPEC }
  : { kind: 'fixed_shape', isl: ISL, osl: OSL, conc: CONC };
const INIT_FLAGS = String(A.initial_extra_server_args || '');
const INIT_ENV = String(A.initial_extra_env || '');
const INIT_OVERLAY = String(A.initial_overlay_pythonpath || '');
const EXEC_PREFIX = String(A.exec_prefix || '');
const FUSION_RUNTIME_INPUTS = EXEC_PREFIX ? { EXEC_PREFIX } : {};
const FAST_MODE = false;
const ACCURACY_INPUTS = {};

// ---- ELAPSED CLOCK (as in e2e_workflow.js: no wall-clock API in the sandbox, so timers) ----
const TIME_BUDGET_MS = A.time_budget_s != null ? parseInt(A.time_budget_s, 10) * 1000 : null;
let ELAPSED_MS = 0;
if (TIME_BUDGET_MS != null && typeof setTimeout === 'function') {
  const step = Math.max(60000, Math.ceil(TIME_BUDGET_MS / 2048));
  for (let at = step; at <= TIME_BUDGET_MS + step; at += step) {
    const mark = at;
    const t = setTimeout(() => { if (mark > ELAPSED_MS) ELAPSED_MS = mark; }, mark);
    if (t && t.unref) t.unref();
  }
}
const remainingMs = () => (TIME_BUDGET_MS == null ? Infinity : Math.max(0, TIME_BUDGET_MS - ELAPSED_MS));
const remainingMin = () => (TIME_BUDGET_MS == null ? Infinity : Math.round(remainingMs() / 60000));
// Wall time held back for the report + return: nothing new starts inside it.
const KF_RESERVE_MS = 1000 * parseInt(A.kf_reserve_s != null ? A.kf_reserve_s : 1200, 10);
const TIME_HEAD_DEADLINE_MS = TIME_BUDGET_MS == null ? null : Math.max(0, TIME_BUDGET_MS - KF_RESERVE_MS);
// Minimum time a stage needs before it is worth starting (minutes). A stage that cannot finish
// before the deadline is skipped with a recorded reason instead of being cut down mid-flight.
const STAGE_MIN = {
  sample: parseInt(A.kf_min_sample_min != null ? A.kf_min_sample_min : 60, 10),
  semantics: parseInt(A.kf_min_semantics_min != null ? A.kf_min_semantics_min : 60, 10),
  unitside: parseInt(A.kf_min_unitside_min != null ? A.kf_min_unitside_min : 45, 10),
  applyback: parseInt(A.kf_min_applyback_min != null ? A.kf_min_applyback_min
    : (IS_AGENTX ? 150 : 60), 10),
  l3: parseInt(A.kf_min_l3_min != null ? A.kf_min_l3_min : 240, 10),
};
function stageFits(stage) {
  if (TIME_HEAD_DEADLINE_MS == null) return true;
  const left = Math.max(0, TIME_HEAD_DEADLINE_MS - ELAPSED_MS) / 60000;
  if (left >= STAGE_MIN[stage]) return true;
  log(`KernelFusion: ${stage} needs >= ${STAGE_MIN[stage]}min, only ${Math.round(left)}min left before ` +
      'the KernelFusion deadline; not starting it.');
  return false;
}

// ---- Fusion semantic mapping ----
const SEMANTICS_MAPPING_ON = String(A.semantics_mapping != null ? A.semantics_mapping : 'true') === 'true';
const FUSION_SHAPE_CAPTURE_ON = String(A.semantics_shape_capture != null ? A.semantics_shape_capture : 'true') === 'true';
const FUSION_COMPLETE_TABLE_ON = FUSION_SHAPE_CAPTURE_ON;
const FUSION_COMPLETE_TABLE_SWITCH = 'semantics_shape_capture';
const FUSION_COMPLETE_TABLE_TIMEOUT_MS = A.semantics_completion_timeout_ms != null
  ? parseInt(A.semantics_completion_timeout_ms, 10) : null;
const VLLM_PHASE_ANNOTATE = false;

// ---- Kernel-fusion discovery knobs (same names and defaults as the e2e-embedded KernelFusion) ----
const FUSION_DISCOVERY_ON = String(A.fusion_discovery != null ? A.fusion_discovery : 'true') === 'true';
const FUSION_REQUIRED = String(A.fusion_required != null ? A.fusion_required : 'true') === 'true';
const FUSION_TOP_K = parseInt(A.fusion_top_k != null ? A.fusion_top_k : 8, 10);
const FUSION_SCREEN_PASSES = 2;
const FUSION_UNIT_PARALLEL = parseInt(A.fusion_unit_parallel != null ? A.fusion_unit_parallel : 8, 10);
const FUSION_GSM8K_THINKING = String(A.fusion_gsm8k_thinking != null ? A.fusion_gsm8k_thinking : 'false') === 'true';
const FUSION_APPLYBACK_MODE = String(A.fusion_applyback_mode || 'combined').trim() === 'serial' ? 'serial' : 'combined';
const FUSION_BUDGET = parseInt(A.fusion_budget != null ? A.fusion_budget
  : (FUSION_APPLYBACK_MODE === 'combined' ? 10 : 6), 10);
// An AgentX leg owns its own warm-up and runs one measured window, so the A/B is one leg per side
// (fusion_combined_ab.sh AB_REPEAT_MODE=replay_once). A fixed-shape A/B keeps the warm_server
// lifecycle with three timed rounds.
const FUSION_AB_REPEAT_MODE = String(A.fusion_ab_repeat_mode || (IS_AGENTX ? 'replay_once' : 'warm_server'));
const FUSION_COMBINED_AB_REPEATS = parseInt(
  A.fusion_combined_ab_repeats != null ? A.fusion_combined_ab_repeats : (IS_AGENTX ? 1 : 3), 10);
const FUSION_UNITSIDE_ON = String(A.fusion_unitside != null ? A.fusion_unitside : 'true') === 'true';
const FUSION_LEASE_WAIT_S = 540;
const FUSION_LEASE_WAIT_ROUNDS = Math.max(1, parseInt(
  A.fusion_lease_wait_rounds != null ? A.fusion_lease_wait_rounds : 2, 10));
const FUSION_UNIT_ALL_RUNGS = String(A.fusion_unit_all_rungs != null ? A.fusion_unit_all_rungs : 'false') === 'true';
// Ladder descent: when no rung-0 job of a Top-K entry passes (it may be waived, blocked or
// unmeasurable -- a whole-attention or MoE megafusion cannot be isolated from its paged-KV /
// distributed state), unit-test that entry's narrower rungs in a second wave. Without it a
// waived top rung hid every rung below it: on DeepSeek-V4 all 8 entries were waived and nothing
// reached apply-back although their 2-3-kernel rungs are ordinary single-card unit tests.
const FUSION_UNIT_DESCEND = String(A.fusion_unit_descend != null ? A.fusion_unit_descend : 'true') === 'true';
// Compact rungs (at most this many kernels) are unit-tested with rung 0 in the first wave instead
// of waiting for ladder descent: they are the fusions most likely to be authored correctly, and
// on DeepSeek-V4 the wide rungs ahead of them came back not isolable round after round.
const FUSION_COMPACT_RUNG_MAX = Math.max(0, parseInt(
  A.fusion_compact_rung_max != null ? A.fusion_compact_rung_max : 4, 10) || 0);
// Best-of-N authoring. One authoring pass is a draw, not a property of the window: the same rung
// (DeepSeek-V4 e01.r4) measured 1.50x in one round and 0.97x in another. A unit job whose verdict
// is a parity pass below FUSION_UNIT_RETRY_BELOW, or that did not engage, is re-authored until it
// has FUSION_UNIT_ATTEMPTS attempts, each in its own directory, and the fastest parity-passing
// verdict is the one the gate reads (fusion_reuse_verdicts.py --prefer-best).
const FUSION_UNIT_ATTEMPTS = Math.max(1, parseInt(
  A.fusion_unit_attempts != null ? A.fusion_unit_attempts : 3, 10) || 1);
const FUSION_UNIT_RETRY_BELOW = parseFloat(
  A.fusion_unit_retry_below != null ? A.fusion_unit_retry_below : 1.2);
// Serving-side unit test (fusion_serving_unit.py) for windows the single-card validator could not
// isolate (paged-KV / compressed-KV / SWA state of the serving attention backend): the overlay is
// authored on one card, then timed on the live stack at every decode sampling point against a base
// server, with gsm8k on both. One window per Top-K entry with no passing rung, best expected gain
// first, at most FUSION_SERVING_UNIT_MAX of them (each costs one ~15 min server session).
// Engagement preflight + one repair round in apply-back, before the combined A/B.
const FUSION_PREFLIGHT_REPAIR = String(A.fusion_preflight_repair != null ? A.fusion_preflight_repair : 'true') === 'true';
const FUSION_SERVING_UNIT = String(A.fusion_serving_unit != null ? A.fusion_serving_unit : 'true') === 'true';
const FUSION_SERVING_UNIT_MAX = Math.max(0, parseInt(
  A.fusion_serving_unit_max != null ? A.fusion_serving_unit_max : 5, 10) || 0);
const FUSION_BUDGET_S_ARG = A.fusion_budget_s != null ? parseInt(A.fusion_budget_s, 10) : null;
const FUSION_BUDGET_FRAC = parseFloat(A.fusion_budget_frac != null ? A.fusion_budget_frac : 0.6);
const FUSION_BUDGET_DEFAULT_S = 28800;
const FUSION_BUDGET_MIN_S = 1800;
const FUSION_ITEM_MINUTES_DEFAULT = 45;
const FUSION_POOL_MAX = Math.max(0, parseInt(A.fusion_pool_max != null ? A.fusion_pool_max : 8, 10));
const FUSION_GFX = String(A.gfx || A.expected_gfx || 'gfx942');
const FUSION_APPLY_TIMEOUT_MS = parseInt(A.fusion_apply_timeout_ms != null ? A.fusion_apply_timeout_ms : 0, 10);
// Sampling (kernel-extract method). The analysis trace is the top-1 point by GPU-time mass of each
// phase listed here: decode only by default, so the fusion analysis reads the single decode batch
// size that carries the most device time; every sampling point still weights the unit tests.
const FUSION_SAMPLING_POINTS = Math.max(2, parseInt(A.fusion_sampling_points != null ? A.fusion_sampling_points : 6, 10) || 6);
// AgentX defaults to prefill AND decode: on DeepSeek-V4 the trace replay spends ~61% of its GPU
// time in prefill (80k-token inputs, ~800-token outputs), which a decode-only analysis never saw.
const FUSION_ANALYSIS_PHASES = String(A.fusion_analysis_phases || (IS_AGENTX ? 'prefill,decode' : 'decode')).split(',')
  .map((s) => s.trim().toLowerCase()).filter((s) => s === 'prefill' || s === 'decode').join(',') || 'decode';
const FUSION_NOISE_BAND_PCT = parseFloat(A.fusion_noise_band_pct != null ? A.fusion_noise_band_pct
  : (IS_AGENTX ? 1.0 : 0.5));
// The baseline leg: on AgentX the CI-faithful parity window, one replay (the client warms itself).
const BASELINE_PURPOSE = String(A.fusion_baseline_purpose || 'parity');
// Resume from an earlier KernelFusion eval dir of the SAME deployment: its measured baseline and its
// verified sampling (traces, manifest, observed workload) are reused in place, and the stage starts
// at Semantics. A changed stack, flag set or workload invalidates them; the reuse agent checks.
const REUSE_DIR = String(A.fusion_reuse_eval_dir || '').replace(/\/+$/, '');
// Earlier KernelFusion eval dirs of the same deployment and Top-K whose PASSING unit-side
// verdicts (and the kernels they point at) seed this round: a candidate with one is not
// re-authored. Authoring is not repeatable -- e01.r4 measured 1.50x in one round and 0.97x in
// another -- so without this a re-run loses earlier wins at random.
const REUSE_UNIT_DIRS = (Array.isArray(A.fusion_reuse_unitside_dirs)
  ? A.fusion_reuse_unitside_dirs : String(A.fusion_reuse_unitside_dirs || '').split(','))
  .map((d) => String(d).trim().replace(/\/+$/, '')).filter(Boolean);
// L3 stage-wise fusion (opt-in). L1 elementwise and L2 op-wise fusions are the Screen windows,
// which stop at expert-parallel MoE kernels. With fusion_l3_mega=true the screen also lists each
// EP MoE stage, and StageFusion hands the largest ones to GEAK mode=mega (kernel_mega.js, one
// workflow() level) on the serving GPUs; a delivered candidate joins the combined A/B as one
// overlay. Off by default: no StageFusion phase and no --stage-l3 screen.
const FUSION_L3_ON = String(A.fusion_l3_mega != null ? A.fusion_l3_mega : 'false') === 'true';
const FUSION_L3_GEAK_DIR = String(A.fusion_l3_geak_dir ||
  `${WORKFLOW_DIR.replace(/\/[^/]*$/, '')}/kernel_workflow`).replace(/\/+$/, '');
const FUSION_L3_MAX_STAGES = Math.max(1, parseInt(A.fusion_l3_max_stages != null ? A.fusion_l3_max_stages : 1, 10) || 1);
// A stage below this share of request kernel time is not worth an hours-long mega search.
const FUSION_L3_MIN_SHARE_PCT = parseFloat(A.fusion_l3_min_share_pct != null ? A.fusion_l3_min_share_pct : 2);
const FUSION_L3_BUDGET_S = parseInt(A.fusion_l3_budget_s != null ? A.fusion_l3_budget_s : 43200, 10);
const FUSION_L3_ROUNDS = parseInt(A.fusion_l3_rounds != null ? A.fusion_l3_rounds : 12, 10);
// Caller overrides merged last into the mega args (e.g. use_expert_skills, target_guards, task).
const FUSION_L3_MEGA_ARGS = (A.fusion_l3_mega_args && typeof A.fusion_l3_mega_args === 'object')
  ? A.fusion_l3_mega_args : {};
// An earlier KernelFusion eval dir: each stage packages its fusion/l3/<stage_id>/mega_result.json
// instead of running mega again.
const FUSION_L3_REUSE_DIR = String(A.fusion_l3_reuse_dir || '').replace(/\/+$/, '');
// What the caller already knows about the stage's operator (any of kernel_path, benchmark_harness,
// call_site, package, ep_size, mega_args): prepare verifies it instead of discovering it.
const FUSION_L3_OPERATOR = (A.fusion_l3_operator && typeof A.fusion_l3_operator === 'object')
  ? A.fusion_l3_operator : null;
// Extra kernel-name regex for EP MoE kernels the built-in pattern misses (another EP library).
const FUSION_L3_EP_REGEX = String(A.fusion_l3_ep_kernel_regex || '');
const FUSION_L3_EP_DEFAULT = parseInt(A.ep != null ? A.ep : SERVING_TP, 10) || SERVING_TP;
let FUSION_INPUTS = { FUSION_TOPK_JSON: '', FUSION_CANDIDATES_JSON: '', FUSION_VALIDATION_JSON: '', FUSION_UNITSIDE_JSON: '' };

// ---- run state ----
let BASELINE_TPUT = 0;
let NOISE_BAND = FUSION_NOISE_BAND_PCT;
let curFlags = INIT_FLAGS;
let curEnv = INIT_ENV;
let curOverlay = INIT_OVERLAY;
let curTput = 0;
const acceptedFusions = [];
let fusionAccuracyRef = null;
let semantics = null;
let fusionCapture = null;
let fusionDisposition = null;
let fusionCoverage = null;
let fusionObservedWorkload = '';
let fusionCompletionOutcome = 'not_run';
let fusionExecutionList = [];
let fusionCandidatePool = [];
let fusionBoardK = null;
let fusionBoardNoiseBand = null;
let fusionBoardPhaseBlind = [];
let fusionUnitStatus = null;
let fusionUnitEligibleIds = null;
let fusionSemanticsAttempted = false;
let fusionL3Stages = [];
let fusionL3 = null;           // StageFusion outcome, reported as kf_return.l3
const fusionL3Authored = [];   // packaged L3 overlays, FUSION_AUTHOR_SCHEMA rows
let baseline = null;
const profile = null;
const TRACELENS_INPUTS = {};

// ---- schemas ----
const obj = (props, required) => ({ type: 'object', properties: props, required: required || [], additionalProperties: true });
const arrObj = { type: 'array', items: { type: 'object', additionalProperties: true } };
const arrStr = { type: 'array', items: { type: 'string' } };
const cfg = (o) => Object.entries(o).map(([k, v]) =>
  `- ${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join('\n');

const BASELINE_SCHEMA = obj({
  status: { type: 'string' }, eval_dir: { type: 'string' },
  baseline_throughput_tok_s: { type: 'number' }, metric_basis: { type: 'string' },
  bench_summary_json: { type: 'string' }, server_log: { type: 'string' }, bench_log: { type: 'string' },
  bench_script: { type: 'string' }, reused_from: { type: 'string' }, notes: { type: 'string' },
}, ['status', 'baseline_throughput_tok_s', 'server_log']);
const CAPTURE_SCHEMA = obj({
  round: { type: ['number', 'string'] },
  trace_dir: { type: 'string' }, trace_files: arrStr,
  analysis_rank_trace: { type: 'string' }, trace_manifest_json: { type: 'string' },
  phase_evidence_status: { type: 'string' }, source: { type: 'string' },
  observed_workload_json: { type: 'string' }, sampling_summary_json: { type: 'string' },
  analysis_points: { type: 'object', additionalProperties: true },
  notes: { type: 'string' },
}, ['trace_manifest_json']);
const SEMANTICS_SCHEMA = obj({
  status: { type: 'string' }, round: { type: ['number', 'string'] },
  trace_manifest_json: { type: 'string' }, structural_patterns_json: { type: 'string' },
  semantic_event_audit_jsonl: { type: 'string' }, layer_instance_audit_json: { type: 'string' },
  semantic_table_json: { type: 'string' }, semantic_table_md: { type: 'string' },
  semantic_report_md: { type: 'string' }, semantic_report_json: { type: 'string' },
  shape_capture_plan_json: { type: 'string' }, quality_json: { type: 'string' },
  shape_log_jsonl: { type: 'string' }, op_coverage_manifest: { type: 'string' },
  kernel_semantic_evidence_jsonl: { type: 'string' },
  shape_type_verification_json: { type: 'string' },
  phases_in_tables: arrStr, missing_required_phases: arrStr,
  notes: { type: 'string' },
}, ['status']);
const FUSION_RANK_SCHEMA = obj({
  status: { type: 'string' }, round: { type: ['number', 'string'] },
  fusion_topk_json: { type: 'string' }, fusion_topk_md: { type: 'string' },
  execution_list: arrObj, config_levers: arrObj, notes: { type: 'string' },
  candidate_pool: arrObj, k_effective: { type: 'number' }, noise_band_pct: { type: 'number' },
  phase_blind: arrStr, l3_stages: arrObj,
}, ['fusion_topk_json', 'execution_list']);
const FUSION_UNIT_SCHEMA = obj({
  candidate_id: { type: 'string' }, verdict_path: { type: 'string' },
  parity: { type: 'string' }, isolated_speedup: { type: 'number' },
  engaged: { type: 'boolean' }, notes: { type: 'string' },
}, ['candidate_id', 'verdict_path']);
const FUSION_UNIT_AGG_SCHEMA = obj({
  status: { type: 'string' }, fusion_unitside_json: { type: 'string' },
  fusion_unitside_md: { type: 'string' }, validated_count: { type: 'number' },
  applyback_eligible_ids: arrStr,
  unit_status_by_candidate: { type: 'object', additionalProperties: true },
  waived: arrObj, deferred: arrObj, notes: { type: 'string' },
}, ['status', 'fusion_unitside_json']);
const FUSION_LEASE_SCHEMA = obj({
  ok: { type: 'boolean' }, action: { type: 'string' }, lease_id: { type: 'string' },
  alive_count: { type: 'number' }, busy_holders: arrObj, serving_lock_held: { type: 'boolean' },
  leases: arrObj, signalled: arrObj, refused: arrObj, released: arrStr,
  lease_path: { type: 'string' }, reason: { type: 'string' },
}, ['ok']);
const FUSION_APPLY_SCHEMA = obj({
  accepted_fusions: arrObj, final_overlay: { type: 'string' },
  e2e_throughput_tok_s: { type: 'number' }, accepted_flags: { type: 'string' }, accepted_env: { type: 'string' },
  rejected: arrObj, deferred: arrObj, deferred_author_count: { type: 'number' },
  applyback_gate_json: { type: 'string' }, applyback_report_md: { type: 'string' },
  learned_cards: arrObj, accuracy_reference: obj({}), accepted_accuracy: obj({}),
  combined_decision: { type: 'string' }, combined_ab: obj({}),
  notes: { type: 'string' },
}, ['accepted_fusions']);
const FUSION_AUTHOR_SCHEMA = obj({
  exec_id: { type: 'string' }, status: { type: 'string', enum: ['authored', 'blocked'] },
  fusion: { type: 'string' }, overlay_dir: { type: 'string' }, banner_tag: { type: 'string' },
  candidate_ids: { type: 'array', items: { type: 'string' } }, parity: { type: 'string' },
  reason: { type: 'string' },
}, ['exec_id', 'status']);
const FUSION_REUSE_SCHEMA = obj({
  ok: { type: 'boolean' }, reused: { type: 'object', additionalProperties: true },
  reason: { type: 'string' },
}, ['ok']);
const FUSION_SERVING_AUTHOR_SCHEMA = obj({
  candidate_id: { type: 'string' }, status: { type: 'string', enum: ['authored', 'blocked'] },
  loader_dir: { type: 'string' }, banner_tag: { type: 'string' }, kernel_source: { type: 'string' },
  parity: { type: 'string' }, reason: { type: 'string' },
}, ['candidate_id', 'status']);
const FUSION_SERVING_MEASURE_SCHEMA = obj({
  ok: { type: 'boolean' }, base_ok: { type: 'boolean' },
  verdicts: { type: 'object', additionalProperties: true }, reason: { type: 'string' },
}, ['ok']);
const FUSION_PREFLIGHT_SCHEMA = obj({
  server_ok: { type: 'boolean' }, tp: { type: 'number' },
  engaged: { type: 'object', additionalProperties: true }, missing: arrStr,
  server_log: { type: 'string' }, reason: { type: 'string' },
}, ['server_ok']);
const FUSION_LADDERS_SCHEMA = obj({
  ok: { type: 'boolean' }, ladders: { type: 'object', additionalProperties: true },
  reason: { type: 'string' },
}, ['ok']);
const FUSION_L3_PREP_SCHEMA = obj({
  status: { type: 'string', enum: ['ready', 'blocked'] }, stage_id: { type: 'string' },
  kernel_path: { type: 'string' }, benchmark_harness: { type: 'string' },
  ep_size: { type: 'number' }, package: { type: 'string' }, call_site: { type: 'string' },
  cases: arrObj, mega_args: { type: 'object', additionalProperties: true },
  reason: { type: 'string' }, notes: { type: 'string' },
}, ['status', 'stage_id']);
const REPORT_SCHEMA = obj({
  kf_return_json: { type: 'string' }, report_md: { type: 'string' }, index_md: { type: 'string' },
  notes: { type: 'string' },
}, ['kf_return_json']);

// ---- agents ----
const PROCESS_SAFETY = `## PROCESS SAFETY (a violation can kill the caller's orchestrator, failing the whole task)
This container's PID 1 is the CALLER's orchestrator process, not yours, and it is NOT restartable.
NEVER run global or pattern-matched process cleanup: no \`pkill -f\` / \`pgrep -f ... | xargs kill\` /
\`killall\` / \`ps aux | grep ... | xargs kill\`, and never \`kill -- -PGID\` for a group you did not
create. Manage ONLY processes you started, by the pid you captured at launch. A script of yours that
launches a server sources the shared teardown contract:
    source ${WORKFLOW_DIR}/scripts/server_teardown.sh
    trap server_teardown EXIT
    \${SERVER_LAUNCH_PREFIX:-} <launch server> &   # own session => pgid == pid
    SERVER_PID=\$!; server_record_identity "\$SERVER_PID"
KernelFusion's own launcher, ${WORKFLOW_DIR}/scripts/fusion_serving_session.sh, already does this.

## LONG MEASUREMENTS OUTLIVE YOUR TOOL CALL
A serving leg (an AgentX replay is ~1-3h with its boot) is longer than any Bash call. Launch it
detached, then WAIT for it with the blocking waiter, one call per slice, until it prints LEG_DONE:
    setsid nohup bash run_leg.sh > run_leg.out 2>&1 < /dev/null & echo "LEG_PID=\$!"
    bash ${WORKFLOW_DIR}/scripts/fusion_leg_wait.sh <file that marks completion> <LEG_PID> run_leg.out 540
LEG_RUNNING => call it again; LEG_EXITED => the leg ended without its result. Your harness may refuse
a bare foreground \`sleep\`; do NOT switch to a Monitor / background-notification tool and stop calling
tools: the moment your turn ends, your reply IS your final answer and nothing wakes you again. Never
return before the leg you were asked to run has its result (or has demonstrably died).
`;
function withProcessSafety(prompt) { return typeof prompt === 'string' ? PROCESS_SAFETY + prompt : prompt; }

const AGENT_TIMEOUT_MS = parseInt(A.agent_timeout_ms != null ? A.agent_timeout_ms : 21600000, 10);
function agentTimeoutFor() {
  if (TIME_BUDGET_MS == null) return AGENT_TIMEOUT_MS;
  // Never let one agent outlive the stage deadline by more than the reserve.
  const left = Math.max(120000, TIME_BUDGET_MS - ELAPSED_MS);
  return AGENT_TIMEOUT_MS > 0 ? Math.min(AGENT_TIMEOUT_MS, left) : left;
}
function agentBounded(rawPrompt, opts) {
  const prompt = withProcessSafety(rawPrompt);
  const timeoutMs = (opts && opts.timeoutMs > 0) ? opts.timeoutMs : agentTimeoutFor();
  if (typeof setTimeout !== 'function' || !(timeoutMs > 0)) return agent(prompt, opts);
  let to;
  const run = agent(prompt, opts);
  const guard = new Promise((resolve) => {
    to = setTimeout(() => {
      log(`  [hung-agent guard] ${(opts && opts.label) || 'agent'} exceeded ${Math.round(timeoutMs / 60000)}min with no return.`);
      resolve(null);
    }, timeoutMs);
  });
  return Promise.race([
    run.then((r) => { clearTimeout(to); return r; }, (e) => { clearTimeout(to); throw e; }),
    guard,
  ]);
}
async function safeAgent(prompt, opts, tries = 3) {
  let lastErr = 'unknown';
  for (let i = 0; i < tries; i++) {
    try {
      const r = await agentBounded(prompt, opts);
      if (r) return r;
      lastErr = 'null/empty result';
    } catch (e) { lastErr = String(e); }
    log(`agent[${(opts && opts.label) || '?'}] attempt ${i + 1}/${tries} failed: ${String(lastErr).slice(0, 160)}`);
  }
  log(`agent[${(opts && opts.label) || '?'}] DEGRADED to null after ${tries} tries (${String(lastErr).slice(0, 120)})`);
  return null;
}

function workloadIdentityBlock() {
  if (!IS_AGENTX) return '';
  return `
## WORKLOAD IDENTITY — this run measures an AgentX TRACE REPLAY, not a fixed ISL/OSL sweep
The served load is the AgentX corpus replayed by aiperf (concurrency ${CONC}). The CLIENT owns the
load, its warm-up and its measured window; BENCH_CLIENT, E2E_METRIC and the AGENTX_* variables are
already exported into your environment by interface/run_kf_e2e.py: never set or override them for a
measured leg. ISL=${ISL} / OSL=${OSL} on a bench line is a placeholder the trace client ignores.
Sampling probes and unit tests use shapes taken from the baseline's own server log
(OBSERVED_WORKLOAD_JSON), never from these placeholders.
`;
}

function roleAgent(role, phase, intro, inputs) {
  const inall = {
    BACKEND, SERVING_TP, SERVING_GPU, GPU_IDS, MODEL_NAME, WORKFLOW_DIR,
    ...(IS_AGENTX ? { WORKLOAD_KIND: 'agentx_trace_replay' } : {}),
    ...inputs,
  };
  return `You are the ${role}. PHASE=${phase}.
First Read ${WORKFLOW_DIR}/roles/${role}.md and follow its instructions for PHASE=${phase}.
Read any knowledge files it points you to under ${WORKFLOW_DIR}/knowledge/.
This is the STANDALONE KernelFusion stage (${KF_DIR}/README.md): it runs before the GEAK e2e
workflow, so there is no formal Profile, no Strategize and no director-staged bench script. Where the
role file mentions EVAL_DIR/bench_e2e.sh, that file is a wrapper the Baseline phase wrote around
${WORKFLOW_DIR}/scripts/bench_e2e.sh.
Do all filesystem/shell work yourself (Bash/Read/Write). ${intro}
When you invoke bench_e2e.sh, pass BACKEND=${BACKEND} in its env so the correct serving adapter is used.

## SERVING CONFIG INVARIANT (do not violate — all e2e numbers must be comparable)
Every serving measurement in this stage (baseline, sampling probes, A/B legs) uses tensor-parallel
TP=${SERVING_TP} on the GPU set GPU=${SERVING_GPU}: pass BACKEND=${BACKEND} TP=${SERVING_TP}
GPU=${SERVING_GPU} to bench_e2e.sh / fusion_serving_session.sh. GPU_IDS=${GPU_IDS} is the pool for
single-GPU isolated work (unit tests, authoring parity checks), one id per task.
${workloadIdentityBlock()}
## Inputs
${cfg(inall)}

Return ONLY the structured JSON the role file specifies (a StructuredOutput tool is forced).`;
}

function makeSem(ids) {
  const free = ids.slice(); const waiters = [];
  const pump = () => { while (waiters.length && waiters[0].n <= free.length) {
    const w = waiters.shift(); w.resolve(free.splice(0, w.n)); } };
  return {
    size: ids.length,
    acquire(n = 1) { if (n <= free.length) return Promise.resolve(free.splice(0, n));
      return new Promise((resolve) => { waiters.push({ n, resolve }); }); },
    release(got) { free.push(...got); pump(); },
    async with(n, fn) { const g = await this.acquire(n); try { return await fn(g); } finally { this.release(g); } },
  };
}

// Concurrency for the apply-back gsm8k gate: the server's CUDA-graph batch
// (sglang --cuda-graph-max-bs), else the workload concurrency. Above that batch the
// server decodes eagerly, so most tokens would bypass a decode-path fusion.
function fusionGsm8kConcurrency(flags) {
  const found = [...String(flags || '').matchAll(/--cuda-graph-max-bs(?:=|\s+)(\d+)/g)];
  const bs = found.length ? parseInt(found[found.length - 1][1], 10) : NaN;
  return Math.max(1, bs > 0 ? bs : CONC);
}

// Identity of the stack a gsm8k base score was measured on. Apply-back rewrites
// the combined overlay in place, so the accepted fusion ids are part of the key,
// not just the overlay path.
function fusionStackKey() {
  const ids = acceptedFusions.map((r) => String((r && (r.exec_id || r.fusion)) || '')).sort();
  return JSON.stringify({ fusions: ids, overlay: curOverlay || '', flags: curFlags || '', env: curEnv || '' });
}

// ---- KernelFusion: serving-GPU lease + process registry (root cause B.9) --------------------
// Qwen3.5-397B (20261010): e03's integrator started its A/B driver with nohup/setsid and returned
// "not finished"; the driver was reparented to PID 1 and held all eight serving GPUs. e04..e08
// each saw the GPUs occupied, correctly refused (PROCESS SAFETY) to kill a process they had not
// started, and returned "blocked" with no measurement. The agent was asked to judge GPU
// ownership it could not prove. Ownership now belongs to the WORKFLOW:
//   * before each apply_one / stacking step / accuracy calibration / collective unit test the
//     workflow ACQUIRES the serving lease (scripts/fusion_gpu_lease.py acquire);
//   * the agent REGISTERS every process it starts (pid, pgid, /proc start time) in
//     $EVAL_DIR/fusion/gpu_lease/<lease_id>.json -- or launches it through the helper;
//   * after the agent returns the workflow WAITS (bounded) for registered processes and then
//     REAPS only those, by recorded identity. They are processes this workflow's own call
//     started, so this is exactly what PROCESS SAFETY allows: never a pattern kill, never PID 1,
//     never a group containing PID 1 or one of our ancestors, never a recycled pid;
//   * an acquire that finds an earlier lease still alive reaps THAT lease (also ours) first.
// Workflow scripts have no shell, so each lease operation is ONE command run by a narrowly
// scoped agent that returns the script's JSON verbatim -- the script, not the agent, decides.
const FUSION_LEASE_SCRIPT = `${WORKFLOW_DIR}/scripts/fusion_gpu_lease.py`;
function fusionLeaseDir() { return `${EVAL_DIR}/fusion/gpu_lease`; }
function fusionLeaseInputs(leaseId) {
  return {
    GPU_LEASE_ID: leaseId, GPU_LEASE_DIR: fusionLeaseDir(), GPU_LEASE_SCRIPT: FUSION_LEASE_SCRIPT,
    GPU_LEASE_FILE: `${fusionLeaseDir()}/${leaseId}.json`,
  };
}
// Shared prompt text for every agent that runs serving work under a lease. It restates the
// lease contract in the call itself (the role files carry the detail) because the failure it
// prevents -- backgrounding an A/B and returning -- is invisible until the NEXT call.
function fusionLeaseRules(leaseId, serving = true) {
  const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
  const helper = `${ex}python3 ${FUSION_LEASE_SCRIPT}`;
  const sel = `--lease-dir ${fusionLeaseDir()} --lease-id ${leaseId}`;
  const head = serving
    ? `## SERVING-GPU LEASE ${leaseId} (owned by the workflow, not by you)
The workflow already holds the serving-GPU lease for this call. You do not judge whether the GPUs
are "occupied": if bench_e2e.sh waits on its serving lock, let it wait.`
    : `## PROCESS REGISTRY ${leaseId} (owned by the workflow, not by you)
This is isolated single-card work: use only the GPU_IDS card(s) you were given; never start a server.`;
  return `

${head}
- Register EVERY process you start (A/B driver, bench_e2e.sh leg, server, gsm8k/probe client).
  Launch long work through the helper, which starts it in its own session and registers it:
      ${helper} launch ${sel} --log <log file> --role ab_driver -- bash <driver.sh>
  or register a pid you started yourself, immediately after starting it:
      ${helper} register ${sel} --pid <pid> --role server
- NEVER background an A/B, bench or server and return: no nohup, no setsid, no trailing &, no
  disown outside the helper's launch subcommand. A long A/B runs in the FOREGROUND of this call:
  after launching, poll with
      ${helper} wait ${sel} --max-s 540
  (one call fits the Bash tool's 10-minute limit) and repeat it until it reports alive_count 0.
  Never return while it reports anything alive -- a returned call with live processes is reaped.
- GPU busy / measurement not finished / not reached is NOT a disposition. If you could not
  measure, return the entry in \`retry\` with reason_code gpu_busy|not_finished|not_reached and
  measured:false -- never in \`rejected\`. The workflow re-runs it after the lease is free.
- \`rejected\` (blocked) requires measured evidence: measured:true and blocked_evidence one of
  unit_fail, unit_parity_fail, e2e_regression, e2e_no_gain, ab_reject, accuracy_fail,
  not_engaged, wire_failure, kernel_unavailable, stack_reject.
- Never kill, signal or wait on a process you did not start; the workflow reaps registered
  processes after you return.`;
}
// Run ONE deterministic script command through a narrowly scoped agent and return its JSON.
// The workflow has no shell; the script, not the agent, decides.
function fusionScriptOp(cmd, label, schema) {
  return safeAgent(
    'You are the KernelFusion script runner. Run EXACTLY this one shell command with the Bash ' +
    'tool (timeout 600000 ms), once, and do nothing else:\n\n' +
    `    ${cmd}\n\n` +
    'It prints ONE JSON object on its last stdout line. Its exit code (0 = clear, 2 = busy / still ' +
    'alive) is part of the answer, not a failure. Return that JSON object verbatim as the structured ' +
    'output. Do NOT run any other command, do NOT signal, kill, inspect or clean up any process ' +
    'yourself, and do NOT retry with different arguments: the script owns process identity and ' +
    'safety. If the command cannot run at all, return {"ok": false, "reason": "<the error text>"}.',
    { phase: 'KernelFusion', label, schema }, 2);
}
async function fusionLeaseOp(op, args, label) {
  const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
  const cmd = `${ex}python3 ${FUSION_LEASE_SCRIPT} ${op} --lease-dir ${fusionLeaseDir()} ${args}`;
  return fusionScriptOp(cmd, `fusion-lease:${op}:${label}`, FUSION_LEASE_SCHEMA);
}
// Bounded wait for a lease's registered processes, then reap exactly those. `leaseId` '*'
// settles every held lease (the barrier after the parallel unit-side wave).
async function fusionLeaseSettle(leaseId, why) {
  const scope = leaseId === '*' ? '--all' : `--lease-id ${leaseId}`;
  for (let round = 0; round < FUSION_LEASE_WAIT_ROUNDS; round++) {
    const st = await fusionLeaseOp('wait', `${scope} --max-s ${FUSION_LEASE_WAIT_S} --release`, leaseId);
    if (!st) break;
    if (Number(st.alive_count) === 0) return { clean: true, reaped: false };
    log(`KernelFusion lease ${leaseId} (${why}): ${st.alive_count} registered process(es) still ` +
        `alive after wait round ${round + 1}/${FUSION_LEASE_WAIT_ROUNDS}.`);
  }
  const reaped = await fusionLeaseOp('reap', `${scope} --grace-s 15`, leaseId);
  const left = reaped ? Number(reaped.alive_count) || 0 : null;
  log(`KernelFusion lease ${leaseId} (${why}): reaped the registered process groups the call ` +
      `left behind (signalled ${reaped && Array.isArray(reaped.signalled) ? reaped.signalled.length : '?'}, ` +
      `still alive ${left == null ? '?' : left}). Unregistered processes were not touched.`);
  return { clean: false, reaped: true, alive_after: left };
}
// Acquire the serving lease. Holders reported busy are workflow-owned leases (an earlier call
// that left work behind): settle them -- wait, then reap by registered identity -- and retry
// once. A serving lock held by something OUTSIDE the registry is never touched: the caller gets
// {ok:false} and defers the entry to its retry queue (gpu_busy).
async function fusionLeaseAcquire(leaseId) {
  const acq = () => fusionLeaseOp('acquire',
    `--lease-id ${leaseId} --serving-gpu ${SERVING_GPU} --wait-s ${FUSION_LEASE_WAIT_S}`, leaseId);
  let res = await acq();
  if (res && res.ok) return { ok: true };
  const holders = (res && Array.isArray(res.busy_holders))
    ? res.busy_holders.map((h) => h && h.lease_id).filter(Boolean) : [];
  for (const holder of holders) {
    log(`KernelFusion lease ${leaseId}: earlier workflow lease ${holder} still has live registered ` +
        'processes; settling it before this call.');
    await fusionLeaseSettle(holder, `stale holder before ${leaseId}`);
  }
  if (holders.length) {
    res = await acq();
    if (res && res.ok) return { ok: true };
  }
  return { ok: false, reason: (res && res.reason) || 'lease runner returned no result' };
}
// Run fn(leaseInputs) under the serving lease; always settle the lease afterwards.
async function withFusionServingLease(leaseId, fn) {
  const got = await fusionLeaseAcquire(leaseId);
  if (!got.ok) {
    log(`KernelFusion lease ${leaseId}: not acquired (${got.reason}); deferring, not blocking.`);
    return { leaseBusy: true, reason: got.reason };
  }
  try {
    return { value: await fn(fusionLeaseInputs(leaseId)) };
  } finally {
    await fusionLeaseSettle(leaseId, 'post-call');
  }
}

// Expected e2e gain of one execution entry, in percent of request time, or null when the board
// does not carry enough to say. Used for the small-gain route: a fusion whose expected gain is
// below the measured noise band cannot win a solo A/B (Qwen3.5-397B, CONC=4/NUM_PROMPTS=40:
// one config spans 263-268 tok/s, +-1%; AR+Norm+Quant is worth ~+0.66%), so after unit pass +
// engagement + no regression it goes straight to stack_pending instead of a solo accept/reject.
function fusionExpectedGainPct(entry) {
  if (!entry) return null;
  for (const k of ['expected_e2e_gain_pct', 'expected_gain_pct']) {
    // Number(null) is 0: a board that carries the key without a value must fall through to
    // the share x speedup estimate, not read as "0% expected" and route every entry to stack.
    if (entry[k] == null || entry[k] === '') continue;
    const v = Number(entry[k]);
    if (Number.isFinite(v)) return v;
  }
  const share = Number.isFinite(Number(entry.e2e_share)) ? Number(entry.e2e_share)
    : (Number.isFinite(Number(entry.workload_forward_pct)) ? Number(entry.workload_forward_pct) / 100 : NaN);
  const sp = Number(entry.estimated_speedup || entry.speedup);
  if (!(share > 0) || !(sp > 1)) return null;
  return Number((share * (1 - 1 / sp) * 100).toFixed(4));
}

// ---- KernelFusion execution helpers (GEAK_structure 二.10) ---------------------------------
// Candidate id of one alternative-group rung. Rung 0 is the execution entry itself; a deeper
// rung is unit-tested and applied under this id. Mirrors
// fusion_unitside_harness.alt_rung_candidate_id.
function fusionAltRungCandidateId(execId, item) {
  return item && item.candidate_id ? String(item.candidate_id) : `${execId}.r${item ? item.rung : '?'}`;
}
// The ladder of one execution slot, best first. An alternative group (alternative windows that
// eliminate the same kernel instance) holds ONE Top-K slot; apply-back walks its rungs in order
// and the first rung that lands (accepted or stack_pending) ends the slot. Every rung entry keeps
// the slot's exec_id so the apply-back gate credits the slot; `alt_rung` names the rung. An older
// board without alt_ladder is a one-rung ladder.
function fusionAltLadder(entry) {
  if (!entry) return [];
  const ladder = Array.isArray(entry.alt_ladder) ? entry.alt_ladder.filter((r) => r && typeof r === 'object') : [];
  const base = { ...entry, alt_rung: 0 };
  if (ladder.length <= 1) return [base];
  const rungs = ladder.slice().sort((a, b) => (Number(a.rung) || 0) - (Number(b.rung) || 0));
  const out = [base];
  for (const item of rungs) {
    if (Number(item.rung) === 0) continue;
    const cid = fusionAltRungCandidateId(entry.exec_id, item);
    const rung = { ...entry, ...item, exec_id: entry.exec_id, alt_group: entry.alt_group,
      alt_rung: item.rung, candidate_id: cid, candidate_ids: [cid] };
    delete rung.alt_ladder;
    // Keep `rung`: validators name the rung's verdict from it (a deleted one produced
    // `e01.rundefined__self.json`, which the harness then counts as never measured).
    rung.rung = item.rung;
    out.push(rung);
  }
  return out;
}
// Apply-back order: expected e2e gain, highest first (unknown last); board order breaks ties.
function fusionOrderByExpectedGain(entries) {
  return (entries || []).map((e, i) => ({ e, i, g: fusionExpectedGainPct(e) }))
    .sort((a, b) => {
      if ((a.g == null) !== (b.g == null)) return a.g == null ? 1 : -1;
      if (a.g != null && b.g !== a.g) return b.g - a.g;
      return a.i - b.i;
    }).map((x) => x.e);
}
// 单侧 status derived from a validator's own return, used only for alt-ladder rungs and pool items
// the aggregate harness did not report (they sit outside its Top-K denominator).
function fusionUnitStatusFromResult(res) {
  if (!res) return '';
  const parity = String(res.parity || '').toLowerCase();
  const sp = Number(res.isolated_speedup);
  if (parity === 'pass') return Number.isFinite(sp) ? (sp > 1 ? 'pass' : 'fail') : '';
  if (parity === 'fail') return 'needs_diagnosis';
  return '';
}
// Needs the whole serving TP group to unit-test (an AllReduce fusion): such entries go through
// the serving lease one at a time; everything else runs in parallel on idle single cards.
// Mirrors fusion_unitside_harness.is_collective_candidate.
function fusionEntryIsCollective(entry) {
  const tags = ((entry && entry.op_tags) || []).map((t) => String(t).toLowerCase());
  if (tags.some((t) => ['allreduce', 'all_reduce', 'communication', 'collective'].includes(t))) return true;
  const names = `${((entry && entry.kernel_names) || []).join(' ')} ${(entry && entry.family) || ''}`;
  return /all_?reduce|cross_device_reduce|reduce_scatter|all_?gather|custom_ar|^collective/i.test(names);
}

// A window over an expert-parallel MoE kernel needs the whole EP group (cross-GPU IPC,
// EP-partitioned experts): no single-card unit test can run it. Mirrors fusion_screen._EP_MOE_KERNEL,
// which rejects such windows; this keeps an older board's rungs out of the unit waves.
const FUSION_EP_EXTRA = FUSION_L3_EP_REGEX ? new RegExp(FUSION_L3_EP_REGEX, 'i') : null;
function fusionRungCrossesEp(entry) {
  return ((entry && entry.kernel_names) || []).some((n) =>
    /megamoe|ep_combine|ep_dispatch|deepep|mori_|all_?to_?all|a2a_/i.test(String(n)) ||
    (FUSION_EP_EXTRA !== null && FUSION_EP_EXTRA.test(String(n))));
}
function fusionRungIsCompact(entry) {
  const n = ((entry && entry.kernel_names) || []).length;
  return n > 0 && n <= FUSION_COMPACT_RUNG_MAX && String((entry && entry.gate) || 'pass') === 'pass';
}

// Execution-list entries the apply-back loop calls. The budget counts only entries with a
// 单侧 apply-back-eligible candidate: an entry the unit-side gate already blocked costs no
// serving A/B, so it must not take a slot from a later passing entry. Ineligible entries
// before the cut are still sent (the integrator records their disposition in a minute or
// two). Without the eligible-id list this is the plain first-N slice. Must match
// fusion_applyback_harness._budget_cutoff_rank.
// Each execution-list entry as the widest rung that passed its unit test: the entry itself when
// rung 0 is eligible, else its first eligible narrower rung, else the entry unchanged.
// The ladder lists compact rungs first (fusion_screen._compact_first), so "first eligible" would
// prefer a 2-kernel window over a passing 10-kernel one: take the eligible rung with the highest
// expected gain, ladder order breaking ties.
function fusionApplyRungs(list, eligibleIds) {
  if (!Array.isArray(eligibleIds)) return list.slice();
  const eligible = new Set(eligibleIds.map(String));
  const ok = (e) => ((Array.isArray(e.candidate_ids) && e.candidate_ids.length)
    ? e.candidate_ids : [e.candidate_id || e.exec_id]).some((cid) => eligible.has(String(cid)));
  return list.map((entry) => {
    if (!entry || ok(entry)) return entry;
    const passing = fusionAltLadder(entry).slice(1).filter(ok);
    if (!passing.length) return entry;
    const gain = (e) => { const g = fusionExpectedGainPct(e); return g == null ? -1 : g; };
    return passing.reduce((best, e) => (gain(e) > gain(best) ? e : best), passing[0]);
  });
}
// The candidate apply-back wires for one (rung) entry: its apply-back-eligible candidate with the
// fastest measured unit-side verdict. An entry merges several candidates (patterns / sides), and
// left to choose, the integrator wired one that never passed (DeepSeek-V4 run 9: e01 shipped c02's
// re-authored act+quant at 1.0x instead of c01's measured 1.42x window).
function fusionPickApplyCandidate(entry, eligibleIds, measured) {
  if (!entry) return null;
  const eligible = Array.isArray(eligibleIds) ? new Set(eligibleIds.map(String)) : null;
  const cids = (Array.isArray(entry.candidate_ids) && entry.candidate_ids.length)
    ? entry.candidate_ids : [entry.candidate_id || entry.exec_id];
  let best = null;
  for (const cid of cids.map(String)) {
    if (eligible && !eligible.has(cid)) continue;
    const m = measured[cid];
    if (!m || !m.verdict_path) continue;
    const sp = Number.isFinite(m.speedup) ? m.speedup : 0;
    if (!best || sp > best.speedup) best = { cid, speedup: sp, verdict_path: m.verdict_path };
  }
  return best;
}
function fusionApplyBudgetEntries(list, eligibleIds, budget) {
  const n = Math.max(0, budget);
  if (!Array.isArray(eligibleIds)) return list.slice(0, n);
  const eligible = new Set(eligibleIds.map(String));
  const out = [];
  let used = 0;
  for (const entry of list) {
    if (used >= n) break;
    out.push(entry);
    if ((entry.candidate_ids || []).some((cid) => eligible.has(String(cid)))) used += 1;
  }
  return out;
}

// The part of a kernel_mega.js return that packaging and the report need: the verdict and the
// selected candidate's registry row (tree, patch, activation, per-case scores).
function fusionMegaSummary(ret) {
  if (!ret || typeof ret !== 'object') return null;
  const sel = String(ret.mega_selected_candidate || '');
  const reg = Array.isArray(ret.mega_candidate_registry) ? ret.mega_candidate_registry : [];
  const row = reg.find((c) => c && String(c.id || '') === sel) || null;
  const pick = (c) => (c ? {
    id: c.id, source: c.source, tree: c.tree, head: c.head, final_tree: c.final_tree,
    final_head: c.final_head, patch: c.patch, activation: c.activation,
    absolute_score: c.absolute_score, correctness: c.correctness, per_case: c.per_case,
  } : null);
  return {
    deliverable: ret.mega_deliverable === true, selected_candidate: sel,
    selected_source: String(ret.mega_selected_source || ''), selected: pick(row),
    final_speedup: ret.final_speedup != null && Number.isFinite(Number(ret.final_speedup))
      ? Number(ret.final_speedup) : null,
    validation_status: String(ret.validation_status || ''), stop_reason: String(ret.stop_reason || ''),
    eval_dir: String(ret.eval_dir || ''), report_path: String(ret.report_path || ''),
    final_patch: String(ret.final_patch || ''), note: String(ret.mega_note || ''),
  };
}

// Phase coverage of a semantics_mapper result, from the mapper's own phase_coverage verdict
// (`phases_in_tables` / `missing_required_phases`, copied from quality_json). A failed or absent
// table covers nothing; a result that does not report coverage is `null` (unknown), never covered.
function semanticsPhaseCoverage(sem, completionEnabled) {
  const strs = (v) => (Array.isArray(v) ? v.map(String) : null);
  const failed = !sem || !sem.semantic_table_json ||
    (sem.status !== 'pass' && sem.status !== 'partial');
  const inTables = failed ? [] : strs(sem.phases_in_tables);
  const missing = failed ? null : strs(sem.missing_required_phases);
  let decodeCovered = null;
  if (failed) decodeCovered = false;
  else if (missing) decodeCovered = !missing.includes('decode');
  else if (inTables) decodeCovered = inTables.includes('decode');
  return {
    semantics_status: sem ? String(sem.status || '') : 'missing',
    phases_in_tables: inTables,
    missing_required_phases: missing,
    decode_covered: decodeCovered,
    decode_completion_enabled: !!completionEnabled,
    decode_completion_switch: FUSION_COMPLETE_TABLE_SWITCH,
  };
}

// ===========================================================================
// PHASE: Baseline. ONE measured leg on the deployed stack. On AgentX it is the CI-faithful replay
// (parity window, the client's own warm-up, no outer warm-up round); its server log is what the
// Sample phase reads the workload distribution from, and its number is KernelFusion's reference.
// ===========================================================================
phase('Baseline');
log(`KernelFusion stage: EVAL_DIR=${EVAL_DIR}, backend=${BACKEND}, TP=${SERVING_TP} on GPU=${SERVING_GPU}, ` +
    `workload=${IS_AGENTX ? `AgentX trace replay (conc ${CONC})` : `ISL ${ISL} / OSL ${OSL} / CONC ${CONC}`}, ` +
    `budget ${TIME_BUDGET_MS == null ? 'unbounded' : `${Math.round(TIME_BUDGET_MS / 60000)}min`}.`);
if (FUSION_L3_ON && TIME_BUDGET_MS != null) {
  const needMin = STAGE_MIN.sample + STAGE_MIN.semantics + STAGE_MIN.unitside + STAGE_MIN.l3 + STAGE_MIN.applyback;
  if (TIME_HEAD_DEADLINE_MS / 60000 < needMin) {
    log(`WARNING: L3 stage fusion is on but the KernelFusion budget (${Math.round(TIME_HEAD_DEADLINE_MS / 60000)}min ` +
        `before the reserve) is below the ~${needMin}min its stages need; StageFusion will likely be skipped as ` +
        'out_of_time. Raise kf_budget_s.');
  }
}
if (FUSION_DISCOVERY_ON && REUSE_DIR) {
  baseline = await safeAgent(
    roleAgent('fusion_trace_collector', 'reuse_baseline',
      'Reuse the measured baseline of REUSE_EVAL_DIR for this stage: verify it was measured on this exact ' +
      'deployment, write EVAL_DIR\'s bench_e2e.sh wrapper, and return that baseline. Run nothing on the GPUs.', {
        EVAL_DIR, REUSE_EVAL_DIR: REUSE_DIR, MODEL_PATH, WORKLOAD,
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv,
        SKILL_DIR: WORKFLOW_DIR, ...FUSION_RUNTIME_INPUTS,
      }),
    { phase: 'Baseline', label: 'fusion-trace-collector:reuse_baseline', schema: BASELINE_SCHEMA }, 2);
  if (baseline && Number(baseline.baseline_throughput_tok_s) > 0) {
    baseline = { ...baseline, reused: true };
    log(`KernelFusion: reusing the baseline of ${REUSE_DIR}.`);
  } else {
    log(`KernelFusion: REUSE_EVAL_DIR ${REUSE_DIR} has no reusable baseline ` +
        `(${(baseline && baseline.notes) || 'no result'}); measuring a fresh one.`);
    baseline = null;
  }
}
if (FUSION_DISCOVERY_ON && !(baseline && Number(baseline.baseline_throughput_tok_s) > 0)) {
  baseline = await safeAgent(
    roleAgent('fusion_trace_collector', 'baseline',
      'Create EVAL_DIR, write its bench_e2e.sh wrapper, run ONE measured baseline leg on the deployed stack ' +
      'and return its throughput and the leg\'s server/bench logs. Do not profile, sample or tune.', {
        EVAL_DIR, MODEL_PATH, WORKLOAD, BASELINE_DIR: `${EVAL_DIR}/baseline`,
        MEASUREMENT_PURPOSE: BASELINE_PURPOSE,
        OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv,
        SKILL_DIR: WORKFLOW_DIR, ...FUSION_RUNTIME_INPUTS,
      }),
    { phase: 'Baseline', label: 'fusion-trace-collector:baseline', schema: BASELINE_SCHEMA }, 2);
  // A collector that returns while its detached leg is still running hands back 0: send it back to
  // finish THAT leg rather than starting the stage on no baseline.
  for (let attempt = 1; !(baseline && Number(baseline.baseline_throughput_tok_s) > 0) && attempt <= 2; attempt++) {
    log(`KernelFusion baseline returned ${baseline ? baseline.baseline_throughput_tok_s : 'nothing'} -- ` +
        `re-dispatching the collector to finish the baseline leg (${attempt}/2).`);
    baseline = await safeAgent(
      roleAgent('fusion_trace_collector', 'baseline',
        'RESUME the baseline in the EXISTING EVAL_DIR: if its leg is still alive, wait on it with ' +
        'scripts/fusion_leg_wait.sh until bench_summary.json lands; if it died without a summary, read why, ' +
        'fix what is in your power and relaunch it once. Return only a measured positive throughput.', {
          EVAL_DIR, MODEL_PATH, WORKLOAD, BASELINE_DIR: `${EVAL_DIR}/baseline`,
          MEASUREMENT_PURPOSE: BASELINE_PURPOSE,
          OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv,
          SKILL_DIR: WORKFLOW_DIR, ...FUSION_RUNTIME_INPUTS,
        }),
      { phase: 'Baseline', label: `fusion-trace-collector:baseline resume ${attempt}`, schema: BASELINE_SCHEMA }, 1);
  }
}
if (FUSION_DISCOVERY_ON) {
  BASELINE_TPUT = baseline ? Number(baseline.baseline_throughput_tok_s) || 0 : 0;
  curTput = BASELINE_TPUT;
  if (BASELINE_TPUT > 0) {
    log(`KernelFusion baseline: ${BASELINE_TPUT} (${baseline.metric_basis || 'throughput'}); ` +
        `server log ${baseline.server_log}; noise band ${NOISE_BAND}%.`);
  } else {
    log('ERROR: KernelFusion baseline produced no measured throughput; the stage cannot sample or gate ' +
        'anything and hands the unchanged stack to the e2e workflow.');
    fusionDisposition = {
      status: 'baseline_failed', applied: [], blocked: [], deferred: [], failed_stage: 'baseline',
      notes: (baseline && baseline.notes) || 'baseline agent returned no measured throughput',
    };
  }
}

// Semantics may narrow its two-phase default to the analysis phases (run_semantics_1_2.py reads
// GEAK_SEMANTICS_REQUIRE_PHASES); the role exports it into every script it runs.
const SEMANTICS_PHASE_INPUTS = {
  SEMANTICS_REQUIRE_PHASES: FUSION_ANALYSIS_PHASES,
  ANALYSIS_PHASES: FUSION_ANALYSIS_PHASES,
};
// Default graph-construction Shape replay setup for this host (sglang, local execution). The
// caller's args.semantics_shape_capture_setup wins key by key. The replay drives a short native
// synthetic load through fusion_shape_replay_bench.sh (bench_e2e.sh without the run's AgentX metric
// identity): it only has to make the server run forwards while graph capture records shapes.
const SEMANTICS_SHAPE_CAPTURE_SETUP = Object.assign({
  backend: BACKEND, execution_mode: 'local', model: MODEL_PATH,
  benchmark: `${WORKFLOW_DIR}/scripts/fusion_shape_replay_bench.sh`, benchmark_repository: EVAL_DIR,
  bench_client: 'native', port: 0, tensor_parallel_size: SERVING_TP, gpu_ids: SERVING_GPU,
  extra_server_args: curFlags, extra_env: curEnv,
  workload: { concurrency: CONC, input_length: Math.min(ISL, 2048), output_length: 64,
    random_range_ratio: 0.8, num_prompts: Math.max(CONC, 8) },
  repeats: 1, profile_num_steps: 1,
}, (A.semantics_shape_capture_setup && typeof A.semantics_shape_capture_setup === 'object')
  ? A.semantics_shape_capture_setup : {});

// The standalone stage samples through SGLang's /generate, /metrics and /start_profile and parses
// its scheduler log, so it is SGLang-only; any other backend is skipped with a recorded reason.
const FUSION_BACKENDS = new Set(['sglang']);
if (!FAST_MODE && FUSION_DISCOVERY_ON && !FUSION_BACKENDS.has(BACKEND)) {
  log(`KernelFusion skipped: BACKEND=${BACKEND} is not supported by the standalone stage (sglang).`);
  fusionDisposition = {
    status: 'skipped_backend', applied: [], blocked: [], deferred: [],
    notes: `The standalone KernelFusion stage supports BACKEND=sglang; this run uses ${BACKEND}.`,
  };
}
if (!FAST_MODE && FUSION_DISCOVERY_ON && FUSION_BACKENDS.has(BACKEND) && BASELINE_TPUT > 0) {
  phase('Sample');
  fusionSemanticsAttempted = SEMANTICS_MAPPING_ON;
  // KernelFusion budget: explicit args.fusion_budget_s, else a share of the time left before the
  // run's dispatch deadline, else the default. Never below FUSION_BUDGET_MIN_S.
  const FUSION_BUDGET_MS = 1000 * Math.max(FUSION_BUDGET_MIN_S, FUSION_BUDGET_S_ARG != null
    ? FUSION_BUDGET_S_ARG
    : (TIME_HEAD_DEADLINE_MS != null
      ? Math.floor(Math.max(0, TIME_HEAD_DEADLINE_MS - ELAPSED_MS) * FUSION_BUDGET_FRAC / 1000)
      : FUSION_BUDGET_DEFAULT_S));
  // Minutes per apply-back item (the screen's t_item): args.fusion_item_minutes, else the default.
  const FUSION_ITEM_MINUTES = A.fusion_item_minutes != null ? Number(A.fusion_item_minutes)
    : FUSION_ITEM_MINUTES_DEFAULT;
  log(`KernelFusion budget ${Math.round(FUSION_BUDGET_MS / 60000)}min, ` +
      `t_item ${FUSION_ITEM_MINUTES}min (sizes the screen's adaptive K).`);
  const fusionEntryTput = curTput;
  const fusionRecoveryState = {
    overlay: curOverlay,
    flags: curFlags,
    env: curEnv,
    throughput: curTput,
    acceptedFusionCount: acceptedFusions.length,
    inputs: { ...FUSION_INPUTS },
    applyResult: null,
  };
  let fusionFailureStage = 'capture';
  try {
  // Scope all run-local discovery artifacts together. Nothing outside this
  // block may seed Top-K or Unit-side inputs before the current capture.
  {
    const fusionRound = 'fusion_capture';
    const fusionCaptureDir = `${EVAL_DIR}/${fusionRound}`;
    const expectedFusionManifest = `${fusionCaptureDir}/profile_trace_manifest.json`;
    // KernelFusion evidence after the kernel-extract skill's launch-and-capture method
    // (knowledge/kernel_extraction_skills/kernel-extract/references/dynamic-workloads.md, capture.md):
    // the completed baseline's server log -> observed workload -> N sampling points -> ONE owned
    // probe server (fusion_serving_session.sh) with one bounded window per point, record_shapes +
    // with_stack on. The analysis trace is the top-1 point by GPU-time mass of each analysis phase
    // (decode by default); every point becomes an observed-workload UT case weighted by the forwards
    // it represents. A fixed-shape workload plans its two points from ISL/OSL/CONC instead of a log.
    const fusionSamplingDir = `${fusionCaptureDir}/sampling`;
    const captureTask = (IS_AGENTX
      ? 'Sample the completed AgentX baseline: '
      : 'Sample the fixed ISL/OSL/CONC workload (SYNTHETIC_SHAPE, no log): ') +
      'start ONE probe server with scripts/fusion_serving_session.sh and run ' +
      'scripts/fusion_agentx_sampling.py run on it, then publish TRACE_MANIFEST_JSON (the top-1 GPU-time ' +
      'point of every ANALYSIS_PHASES phase) and OBSERVED_WORKLOAD_JSON. Do not run another replay leg. ' +
      'Do not build Top-N.';
    const captureInputs = {
      EVAL_DIR, MODEL_PATH, WORKLOAD, ROUND: fusionRound,
      CAPTURE_DIR: fusionCaptureDir, TRACE_MANIFEST_JSON: expectedFusionManifest,
      SAMPLING_DIR: fusionSamplingDir, SAMPLING_POINTS: FUSION_SAMPLING_POINTS,
      ANALYSIS_PHASES: FUSION_ANALYSIS_PHASES,
      OBSERVED_WORKLOAD_JSON: `${fusionSamplingDir}/fusion_observed_workload.json`,
      ...(IS_AGENTX
        ? { BASELINE_LOG: baseline.server_log, BASELINE_BENCH_LOG: baseline.bench_log || '' }
        : { SYNTHETIC_SHAPE: `${ISL},${OSL},${CONC}` }),
      OVERLAY_PYTHONPATH: curOverlay, EXTRA_SERVER_ARGS: curFlags, EXTRA_ENV: curEnv,
      SKILL_DIR: WORKFLOW_DIR, ...FUSION_RUNTIME_INPUTS,
    };
    let samplingStarted = false;
    if (REUSE_DIR && baseline && baseline.reused) {
      samplingStarted = true;
      fusionCapture = await safeAgent(
        roleAgent('fusion_trace_collector', 'reuse_capture',
          'Reuse the verified sampling of REUSE_EVAL_DIR (its manifest, traces and observed workload) for this ' +
          'stage. Run nothing on the GPUs; return the capture result exactly as PHASE=reuse_capture specifies.', {
            EVAL_DIR, REUSE_EVAL_DIR: REUSE_DIR, ANALYSIS_PHASES: FUSION_ANALYSIS_PHASES,
            SKILL_DIR: WORKFLOW_DIR, ...FUSION_RUNTIME_INPUTS,
          }),
        { phase: 'Sample', label: 'fusion-trace-collector:reuse_capture', schema: CAPTURE_SCHEMA }, 2);
      if (!(fusionCapture && fusionCapture.trace_manifest_json &&
            !/fail/i.test(String(fusionCapture.phase_evidence_status || '')))) {
        log(`KernelFusion: REUSE_EVAL_DIR ${REUSE_DIR} has no reusable sampling ` +
            `(${(fusionCapture && fusionCapture.notes) || 'no result'}); sampling afresh.`);
        fusionCapture = null;
        samplingStarted = false;
      }
    }
    if (!fusionCapture && stageFits('sample')) {
      samplingStarted = true;
      fusionCapture = await safeAgent(
        roleAgent('fusion_trace_collector', 'capture', captureTask, captureInputs),
        { phase: 'Sample', label: 'fusion-trace-collector:capture', schema: CAPTURE_SCHEMA }, 1);
      // A required KernelFusion must not fall through to an empty discovery because the collector
      // returned before its probe server finished: send it back to finish (or redo) the capture.
      const captureFailed = (c) => !(c && c.trace_manifest_json) ||
        /fail/i.test(String(c.phase_evidence_status || ''));
      for (let attempt = 1; FUSION_REQUIRED && captureFailed(fusionCapture) && attempt <= 2 && stageFits('sample'); attempt++) {
        log(`KernelFusion is required and the capture returned no usable manifest ` +
          `(${(fusionCapture && fusionCapture.phase_evidence_status) || 'none'}) -- re-dispatching the collector (${attempt}/2).`);
        fusionCapture = await safeAgent(
          roleAgent('fusion_trace_collector', 'capture',
            'RETRY: the previous capture attempt returned no usable TRACE_MANIFEST_JSON (missing, or its status ' +
            'failed). Read its probe journals and session/server logs under CAPTURE_DIR and fix the cause before ' +
            're-probing. If its probe session is still alive, wait for it with scripts/fusion_leg_wait.sh instead ' +
            'of relaunching; if it died, redo the capture. Return only once TRACE_MANIFEST_JSON exists with ' +
            'verified traces. ' + captureTask, captureInputs),
          { phase: 'Sample', label: `fusion-trace-collector:capture retry ${attempt}`, schema: CAPTURE_SCHEMA }, 1);
      }
    }
    fusionObservedWorkload = (fusionCapture && fusionCapture.observed_workload_json)
      || captureInputs.OBSERVED_WORKLOAD_JSON;
    log(`KernelFusion sampling: ${FUSION_SAMPLING_POINTS} point(s) planned; analysis ` +
        `${JSON.stringify((fusionCapture && fusionCapture.analysis_points) || {})} (top-1 GPU-time point per ` +
        `${FUSION_ANALYSIS_PHASES}); UT weights ${fusionObservedWorkload}` +
        (fusionCapture && fusionCapture.sampling_summary_json ? `; summary ${fusionCapture.sampling_summary_json}` : ''));
    // fusion_agentx_sampling.py writes this artifact deterministically.
    // Keep the agent return as the preferred path, but do not discard a valid
    // capture merely because the agent timed out or omitted the path field.
    const fusionTraceManifest = (fusionCapture && fusionCapture.trace_manifest_json)
      ? fusionCapture.trace_manifest_json : expectedFusionManifest;
    if (!fusionCapture || !fusionCapture.trace_manifest_json) {
      log(`KernelFusion capture agent returned no manifest; trying deterministic capture artifact ${expectedFusionManifest}.`);
    }
    if (SEMANTICS_MAPPING_ON && samplingStarted && fusionTraceManifest && stageFits('semantics')) {
      phase('Semantics');
      fusionFailureStage = 'semantic';
      semantics = await safeAgent(
        roleAgent('semantics_mapper', 'build_table',
          'Build fusion semantics from the clean production graph trace. Eager/shape evidence may only fill semantic gaps. If EXEC_PREFIX is set, use it as the literal command prefix.', {
            EVAL_DIR, MODEL_PATH, MODEL_NAME, WORKLOAD, ROUND: fusionRound,
            TRACE_MANIFEST_JSON: fusionTraceManifest,
            PROFILE_TOPN_JSON: '', PROFILE_WORKLOAD_JSON: '', SKILL_DIR: WORKFLOW_DIR,
            ...SEMANTICS_PHASE_INPUTS, ...FUSION_RUNTIME_INPUTS,
          }),
        { phase: 'KernelFusion', label: 'semantics-mapper:fusion', schema: SEMANTICS_SCHEMA }, 1);
      if (FUSION_COMPLETE_TABLE_ON && semantics &&
          semantics.status !== 'failed' && semantics.status !== 'fail') {
        fusionFailureStage = 'shape_completion';
        const completed = await safeAgent(
          roleAgent('semantics_mapper', 'complete_table',
            'Complete unresolved shapes without replacing production-trace timing or phase attribution. If EXEC_PREFIX is set, use it as the literal command prefix.', {
              EVAL_DIR, MODEL_PATH, MODEL_NAME, WORKLOAD, ROUND: fusionRound,
              TRACE_MANIFEST_JSON: fusionTraceManifest,
              STRUCTURAL_PATTERNS_JSON: semantics.structural_patterns_json || '',
              SEMANTIC_TABLE_JSON: semantics.semantic_table_json || '',
              SHAPE_CAPTURE_PLAN_JSON: semantics.shape_capture_plan_json || '',
              SHAPE_CAPTURE_SETUP: SEMANTICS_SHAPE_CAPTURE_SETUP, SKILL_DIR: WORKFLOW_DIR,
              ...SEMANTICS_PHASE_INPUTS, ...FUSION_RUNTIME_INPUTS,
            }),
          { phase: 'KernelFusion', label: 'semantics-mapper:shape-completion', schema: SEMANTICS_SCHEMA,
            ...(FUSION_COMPLETE_TABLE_TIMEOUT_MS != null ? { timeoutMs: FUSION_COMPLETE_TABLE_TIMEOUT_MS } : {}) }, 1);
        fusionCompletionOutcome = completed ? 'returned' : 'no_result';
        if (completed) semantics = completed;
      }
    } else {
      semantics = { status: 'failed', notes: samplingStarted
        ? 'semantics not started (disabled or out of KernelFusion time)'
        : 'sampling not started (out of KernelFusion time)' };
    }

    // A partial table still carries every phase it resolved (e.g. prefill complete,
    // decode boundaries missing). Screen what exists and say which phases are blind,
    // instead of dropping the whole fusion track.
    const semanticsUsable = semantics && semantics.semantic_table_json &&
      (semantics.status === 'pass' || semantics.status === 'partial');
    if (semanticsUsable && semantics.status === 'partial') {
      log('KernelFusion: semantic table is partial; screening the phases it resolved. ' +
          'Chains in unresolved phases cannot be discovered this round (see 01_SEMANTIC.md).');
    }
    // Which phases the fusion tables actually hold. A missing decode is a coverage gap of the
    // whole track (decode is most of a serving cycle), so it travels in the disposition to the
    // report instead of living only in the run log.
    fusionCoverage = semanticsPhaseCoverage(semantics, FUSION_COMPLETE_TABLE_ON);
    fusionCoverage.decode_completion_outcome = fusionCompletionOutcome;
    if (fusionCoverage.decode_covered === false) {
      log(`WARNING: KernelFusion decode coverage MISSING — the fusion tables hold ` +
          `[${(fusionCoverage.phases_in_tables || []).join(', ') || 'none'}] only, so no decode fusion ` +
          'can be discovered this round. ' + (!FUSION_COMPLETE_TABLE_ON
            ? `Decode boundary completion is disabled (${FUSION_COMPLETE_TABLE_SWITCH}=false).`
            : (fusionCompletionOutcome === 'no_result'
              ? 'Decode boundary completion returned no result (timeout or failure); its agent may ' +
                'still be writing a table this round will not read.'
              : 'Decode boundary completion ran and did not resolve it (see 01_SEMANTIC.md).')));
    } else if (fusionCoverage.decode_covered == null) {
      log('WARNING: the semantics mapper did not report phase coverage; KernelFusion decode ' +
          'coverage is unknown (check quality_json phase_coverage).');
    }
    if (semanticsUsable) {
      // Policy after the captured trace has been turned into a kernel table:
      // screen chains, rank by estimated time saved, then two Score passes
      // merged into the final Top-K.
      phase('Screen');
      fusionFailureStage = 'screen';
      const screened = await safeAgent(
        roleAgent('kernel_fusion_analyst', 'screen',
          'Run scripts/fusion_screen.py and return its JSON unchanged. Do not invent or rerank candidates. ' +
          'Pass --fusion-budget-s FUSION_BUDGET_S and --item-minutes ITEM_MINUTES (they size the adaptive K). ' +
          'When OBSERVED_WORKLOAD_JSON is set, pass --observed-workload OBSERVED_WORKLOAD_JSON (phase weights = measured forwards). ' +
          (FUSION_L3_ON ? 'STAGE_L3 is true: also pass --stage-l3 (and --ep-kernel-regex EP_KERNEL_REGEX when set) and ' +
            'copy its l3_stages into your return. ' : '') +
          'If EXEC_PREFIX is set, use it as the literal command prefix.', {
            EVAL_DIR, MODEL_NAME, MODEL_PATH, ROUND: fusionRound,
            SEMANTIC_TABLE_JSON: semantics.semantic_table_json,
            WORKLOAD_ISL: ISL, WORKLOAD_OSL: OSL, WORKLOAD_CONC: CONC,
            ...(fusionObservedWorkload ? { OBSERVED_WORKLOAD_JSON: fusionObservedWorkload } : {}),
            TOP_K: FUSION_TOP_K,
            FUSION_BUDGET_S: Math.round(FUSION_BUDGET_MS / 1000),
            ITEM_MINUTES: FUSION_ITEM_MINUTES,
            PASSES: FUSION_SCREEN_PASSES,
            GFX: FUSION_GFX,
            ...(FUSION_L3_ON ? { STAGE_L3: true } : {}),
            ...(FUSION_L3_ON && FUSION_L3_EP_REGEX ? { EP_KERNEL_REGEX: FUSION_L3_EP_REGEX } : {}),
            SKILL_DIR: WORKFLOW_DIR,
            ...FUSION_RUNTIME_INPUTS,
          }),
        { phase: 'KernelFusion', label: 'fusion-screen', schema: FUSION_RANK_SCHEMA }, 1);
      if (screened && screened.status !== 'failed' && screened.fusion_topk_json) {
        FUSION_INPUTS.FUSION_CANDIDATES_JSON = screened.fusion_candidates_json || '';
        FUSION_INPUTS.FUSION_TOPK_JSON = screened.fusion_topk_json;
        fusionExecutionList = Array.isArray(screened.execution_list)
          ? screened.execution_list.slice() : [];
        // Older boards carry none of these; every consumer treats them as optional.
        fusionCandidatePool = Array.isArray(screened.candidate_pool) ? screened.candidate_pool.slice() : [];
        fusionBoardK = Number.isFinite(Number(screened.k_effective)) ? Number(screened.k_effective) : null;
        fusionBoardNoiseBand = Number(screened.noise_band_pct) > 0 ? Number(screened.noise_band_pct) : null;
        fusionBoardPhaseBlind = Array.isArray(screened.phase_blind) ? screened.phase_blind.map(String) : [];
        if (FUSION_L3_ON) {
          fusionL3Stages = (Array.isArray(screened.l3_stages) ? screened.l3_stages : [])
            .filter((st) => st && st.stage_id);
          log(`KernelFusion screen: ${fusionL3Stages.length} L3 stage(s) (expert-parallel MoE): ` +
              (fusionL3Stages.map((st) => `${st.stage_id} ${st.phase} ${st.request_share_pct}%`).join(', ') || '-'));
        }
        log(`KernelFusion screen: ${fusionExecutionList.length} final candidate(s)` +
            (fusionBoardK != null ? ` (K=${fusionBoardK})` : '') +
            `, ${fusionCandidatePool.length} in the candidate pool; applying every final candidate.`);
      }
    }
  }

  // ---- Phase 3.0: 单侧 gate (unit-side), MANDATORY before apply-back (GEAK_structure 二.10.1) ----
  // Every Top-K entry is unit-tested on its captured shapes -- accuracy against the unfused live
  // chain plus an isolated microbench, for the existing kernel (when the board matched one) AND
  // the self-authored kernel -- and 04_FUSION_UNITSIDE.md / fusion_unitside.json are published
  // before any serving A/B, so apply-back reuses verdicts and kernels instead of redoing them.
  // An alternative group (one Top-K slot, several windows that eliminate the same kernel
  // instance) has only rung 0 unit-tested here (FUSION_UNIT_ALL_RUNGS tests every rung up front).
  // candidate_pool items (below the Top-K bar) are unit-tested here too and reported in
  // 04_FUSION_UNITSIDE.md; apply-back takes only execution-list entries.
  // Scheduling: single-GPU work (elementwise / norm / quant / small GEMM) runs in parallel, one
  // exclusive idle card each (makeSem over GPU_LIST); a hard barrier settles their leases;
  // collective work (AR fusions need the whole TP group) then runs one at a time under the
  // serving-GPU lease. A failed or missing Top-K verdict is passed to the harness as
  // never-measured (--budget-skipped), so the report shows it red instead of silently passing.
  // Pool items stay out of the serving lease (a collective pool item is not unit-tested).
  const fusionPoolEntries = fusionOrderByExpectedGain((fusionCandidatePool || [])
    .filter((e) => e && (e.exec_id || e.candidate_id))
    .map((e) => (e.exec_id ? e : { ...e, exec_id: String(e.candidate_id) })))
    .filter((e) => !fusionEntryIsCollective(e))
    .slice(0, FUSION_POOL_MAX);
  // Validator returns for rungs / pool items, keyed by candidate id (fallback status source).
  const fusionExtraUnitResults = {};
  // Every measured parity pass, by candidate id: {speedup, verdict_path}. Apply-back hands the
  // integrator the exact passing candidate and its verdict (kernel_source) to wire.
  const fusionUnitMeasured = {};
  if (FUSION_INPUTS.FUSION_TOPK_JSON && fusionExecutionList.length && stageFits('unitside')) {
    phase('UnitSide');
    if (!FUSION_UNITSIDE_ON) {
      log('KernelFusion Phase 3.0 (单侧 gate) SKIPPED by explicit args.fusion_unitside=false: ' +
          '04_FUSION_UNITSIDE.md will not exist, so apply-back has no unit-side result to act on.');
    } else {
      fusionFailureStage = 'unitside';
      const semanticTableJson = (semantics && semantics.semantic_table_json) || '';
      // The token range the server can hand each patched seam (graph capture sizes, prefill
      // chunk budget). Workflow scripts have no filesystem / child_process access, so the agents
      // derive it: each validator and the aggregate run scripts/serving_shape_envelope.py on
      // SERVING_BACKEND + SERVING_FLAGS into SHAPE_ENVELOPE_JSON (same flags -> same file), every
      // verdict runs its overlay at both ends (integration_checks.shape_envelope), and the harness
      // fails a candidate that did not.
      const shapeEnvelopeInputs = {
        SHAPE_ENVELOPE_JSON: `${EVAL_DIR}/fusion/serving_shape_envelope.json`,
        SERVING_BACKEND: BACKEND, SERVING_FLAGS: curFlags || '',
        // AgentX: time every sampling point of the candidate's phase (verdict.weighted_cases);
        // the harness gates on the observed-workload weighted speedup.
        ...(fusionObservedWorkload ? { OBSERVED_WORKLOAD_JSON: fusionObservedWorkload } : {}),
      };
      const unitJobs = [];
      const seenCids = new Set();
      const epSkipped = [];   // [{cid, kind}] windows over the EP MoE: no single-card unit test
      const addJob = (entry, cid, kind) => {
        if (!cid || seenCids.has(String(cid))) return;
        seenCids.add(String(cid));
        if (fusionRungCrossesEp(entry)) { epSkipped.push({ cid: String(cid), kind }); return; }
        unitJobs.push({ entry, cid: String(cid), kind, collective: fusionEntryIsCollective(entry),
          result: null, why: '' });
      };
      // The screen agent's echo of execution_list may drop alt_ladder (run 7 did); the first wave
      // needs every entry's compact rungs, so read the missing ladders back from FUSION_TOPK_JSON.
      const recoverLadders = async (entries) => {
        const noLadder = entries.filter((e) => e && e.exec_id &&
          !(Array.isArray(e.alt_ladder) && e.alt_ladder.length > 1));
        if (!noLadder.length) return;
        const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
        const got = await fusionScriptOp(
          `${ex}python3 ${WORKFLOW_DIR}/scripts/fusion_ladders.py ${FUSION_INPUTS.FUSION_TOPK_JSON} ` +
          noLadder.map((e) => `--exec ${e.exec_id}`).join(' '),
          'fusion-ladders', FUSION_LADDERS_SCHEMA);
        const ladders = (got && got.ladders && typeof got.ladders === 'object') ? got.ladders : {};
        for (const entry of noLadder) {
          if (Array.isArray(ladders[entry.exec_id]) && ladders[entry.exec_id].length > 1) {
            entry.alt_ladder = ladders[entry.exec_id];
          }
        }
        log(`KernelFusion Phase 3.0: recovered the ladders of ` +
            `${noLadder.filter((e) => Array.isArray(e.alt_ladder) && e.alt_ladder.length > 1).length}/` +
            `${noLadder.length} entr(ies) the screen echo dropped, from FUSION_TOPK_JSON.`);
      };
      if (FUSION_COMPACT_RUNG_MAX > 0 || FUSION_UNIT_ALL_RUNGS) await recoverLadders(fusionExecutionList);
      let onDemandRungs = 0;
      for (const entry of fusionExecutionList) {
        if (!entry || !entry.exec_id) continue;
        for (const rungEntry of fusionAltLadder(entry)) {
          if (Number(rungEntry.alt_rung) > 0 && !FUSION_UNIT_ALL_RUNGS && !fusionRungIsCompact(rungEntry)) {
            onDemandRungs += 1; continue;
          }
          const cids = (Array.isArray(rungEntry.candidate_ids) && rungEntry.candidate_ids.length)
            ? rungEntry.candidate_ids : [rungEntry.candidate_id || rungEntry.exec_id];
          for (const cid of cids) addJob(rungEntry, cid, Number(rungEntry.alt_rung) > 0 ? 'alt' : 'topk');
        }
      }
      for (const entry of fusionPoolEntries) {
        const cids = (Array.isArray(entry.candidate_ids) && entry.candidate_ids.length)
          ? entry.candidate_ids : [entry.candidate_id || entry.exec_id];
        for (const cid of cids) addJob(entry, cid, 'pool');
      }
      const validateUnit = (job, gpuIds, extraInputs) => safeAgent(
        roleAgent('fusion_unit_validator', 'validate_one',
          'Unit-test exactly CANDIDATE_ID of TARGET_EXECUTION on its captured shapes: the existing kernel side ' +
          '(each TARGET_EXECUTION.matched_existing_kernels entry that fits, first) and the self-authored ' +
          'fused kernel side, each against the unfused LIVE reference chain (accuracy + isolated speedup). ' +
          'Write one verdict per side under VERDICT_DIR and keep the authored kernel under KERNEL_DIR so ' +
          (fusionObservedWorkload ? 'OBSERVED_WORKLOAD_JSON is set: also time EVERY sampling point of the ' +
            "candidate's phase and record them as weighted_cases [{sig, ref_ms, cand_ms}]. " : '') +
          'apply-back reuses it. Use ONLY the card(s) in the GPU_IDS input. Never touch the serving stack. ' +
          (job.kind === 'alt' ? `This is rung ${job.entry.alt_rung} of alternative group ` +
            `${job.entry.alt_group || job.entry.exec_id} (UNIT_JOB_KIND=alt): unit-test this rung's window. ` : '') +
          (job.kind === 'pool' ? 'This is a candidate_pool item (UNIT_JOB_KIND=pool). ' : '') +
          (extraInputs && extraInputs.AUTHOR_ATTEMPT
            ? `AUTHORING ATTEMPT ${extraInputs.AUTHOR_ATTEMPT} of ${extraInputs.AUTHOR_ATTEMPTS}: PRIOR_VERDICT is an ` +
              'earlier attempt at this same candidate (a parity pass that was too slow, or a kernel that did not ' +
              'engage). Read it, then author an INDEPENDENT design that addresses why it was slow (different fusion ' +
              'boundary, tiling, launch configuration, or the existing kernel side) -- do not resubmit that kernel. ' +
              'Write ONLY under the VERDICT_DIR and KERNEL_DIR of this call; the workflow keeps the fastest attempt. '
            : '') +
          'Return FUSION_UNIT_SCHEMA.' + (extraInputs && extraInputs.GPU_LEASE_ID
            ? fusionLeaseRules(extraInputs.GPU_LEASE_ID, job.collective) : ''), {
            EVAL_DIR, MODEL_PATH, MODEL_NAME, EXEC_ID: job.entry.exec_id, CANDIDATE_ID: job.cid,
            TARGET_EXECUTION: job.entry, UNIT_JOB_KIND: job.kind,
            FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
            FUSION_CANDIDATES_JSON: FUSION_INPUTS.FUSION_CANDIDATES_JSON,
            SEMANTIC_TABLE_JSON: semanticTableJson, ...shapeEnvelopeInputs,
            GPU_IDS: gpuIds, TP: job.collective ? SERVING_TP : 1,
            VERDICT_DIR: `${EVAL_DIR}/verdict`, KERNEL_DIR: `${EVAL_DIR}/fusion/unitside/${job.cid}`,
            FUSION_DIR: `${EVAL_DIR}/fusion`,
            ...(extraInputs || {}), ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
          }),
        { phase: 'KernelFusion', label: `fusion_unit_validator:${job.entry.exec_id}:${job.cid}`,
          schema: FUSION_UNIT_SCHEMA }, 1);
      let reusedUnit = {};
      if (REUSE_UNIT_DIRS.length) {
        const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
        const got = await fusionScriptOp(
          `${ex}python3 ${WORKFLOW_DIR}/scripts/fusion_reuse_verdicts.py --to ${EVAL_DIR}/verdict ` +
          `--to-topk ${FUSION_INPUTS.FUSION_TOPK_JSON} ` +
          REUSE_UNIT_DIRS.map((d) => `--from ${d}`).join(' '),
          'fusion-reuse-verdicts', FUSION_REUSE_SCHEMA);
        reusedUnit = (got && got.reused && typeof got.reused === 'object') ? got.reused : {};
        log(`KernelFusion Phase 3.0: reused ${Object.keys(reusedUnit).length} passing verdict(s) from ` +
            `${REUSE_UNIT_DIRS.length} earlier round(s): ` +
            Object.entries(reusedUnit).map(([c, r]) => `${c} ${r.speedup}x`).join(', '));
      }
      // A reused pass stands in for the validator: same verdict file, same kernel. One below the
      // best-of-N retry bar is still re-authored (it stays retryable; --prefer-best keeps it
      // unless an attempt beats it).
      const takeReused = (job) => {
        const r = reusedUnit[job.cid];
        if (!r) return false;
        job.result = { candidate_id: job.cid, verdict_path: `${EVAL_DIR}/verdict/${job.cid}__${r.side}.json`,
          parity: 'pass', isolated_speedup: Number(r.speedup), engaged: true,
          notes: `reused from ${r.from}` };
        job.reused = Number(r.speedup) >= FUSION_UNIT_RETRY_BELOW;
        return true;
      };
      const singleJobs = unitJobs.filter((j) => !j.collective && !takeReused(j));
      const collectiveJobs = unitJobs.filter((j) => j.collective && !j.result);
      const kinds = (k) => unitJobs.filter((j) => j.kind === k).length;
      log(`KernelFusion Phase 3.0: ${unitJobs.length} unit test(s) over ${fusionExecutionList.length} ` +
          `Top-K entr(ies) (${kinds('topk')} Top-K, ${kinds('alt')} alternative rung(s) incl. compact ones, ${kinds('pool')} pool` +
          (epSkipped.length ? `; ${epSkipped.length} window(s) over the EP MoE not unit-testable: ` +
            epSkipped.map((x) => x.cid).join(', ') : '') +
          (onDemandRungs ? `; ${onDemandRungs} deeper alternative rung(s) not unit-tested (args.fusion_unit_all_rungs=true tests them)` : '') +
          '): ' +
          `${singleJobs.length} single-GPU in parallel on {${GPU_LIST.join(',')}}, ` +
          `${collectiveJobs.length} collective serially under the serving-GPU lease.`);
      if (singleJobs.length) {
        const UNIT_ISO = makeSem(GPU_LIST);
        await parallel(singleJobs.map((job) => async () => UNIT_ISO.with(1, async (cards) => {
          // A per-job registry (no serving acquire: one isolated card) so the barrier below can
          // reap anything a microbench left behind before the serving work starts.
          job.result = await validateUnit(job, cards.join(','), fusionLeaseInputs(`unit_${job.cid}`));
          if (!job.result) job.why = 'unit validator returned no verdict';
        })));
        await fusionLeaseSettle('*', 'Phase 3.0 single-GPU barrier');
      }
      for (const job of collectiveJobs) {
        const leased = await withFusionServingLease(`unit_${job.cid}`,
          (leaseIn) => validateUnit(job, SERVING_GPU, leaseIn));
        job.result = leased.value || null;
        if (!job.result) {
          job.why = leased.leaseBusy ? `serving-GPU lease busy: ${leased.reason}`
            : 'unit validator returned no verdict';
        }
      }
      if (FUSION_UNIT_DESCEND && !FUSION_UNIT_ALL_RUNGS && stageFits('unitside')) {
        const passed = (job) => fusionUnitStatusFromResult(job.result) === 'pass';
        // Stuck = rung 0 did not pass (or could not be tested). A passing compact rung does not
        // end the descent: a wider rung of the same slot may be worth several times more.
        const isStuck = (entry) => {
          const top = unitJobs.filter((j) => j.kind === 'topk' && j.entry.exec_id === entry.exec_id);
          return !top.some(passed);
        };
        await recoverLadders(fusionExecutionList.filter((e) => e && e.exec_id && isStuck(e)));
        const before = unitJobs.length;
        const stuck = [];
        for (const entry of fusionExecutionList) {
          if (!entry || !entry.exec_id || !isStuck(entry)) continue;
          stuck.push(entry.exec_id);
          for (const rungEntry of fusionAltLadder(entry).slice(1)) {
            const cids = (Array.isArray(rungEntry.candidate_ids) && rungEntry.candidate_ids.length)
              ? rungEntry.candidate_ids : [rungEntry.candidate_id || rungEntry.exec_id];
            for (const cid of cids) addJob(rungEntry, cid, 'alt');
          }
        }
        const descend = unitJobs.slice(before).filter((j) => !j.collective && !takeReused(j));
        log(`KernelFusion Phase 3.0 ladder descent: ${stuck.length} Top-K entr(ies) without a passing ` +
            `rung 0 (${stuck.join(', ') || '-'}); unit-testing ${descend.length} narrower rung(s).`);
        if (descend.length) {
          const UNIT_ISO2 = makeSem(GPU_LIST);
          await parallel(descend.map((job) => async () => UNIT_ISO2.with(1, async (cards) => {
            job.result = await validateUnit(job, cards.join(','), fusionLeaseInputs(`unit_${job.cid}`));
            if (!job.result) job.why = 'unit validator returned no verdict';
          })));
          await fusionLeaseSettle('*', 'Phase 3.0 ladder-descent barrier');
        }
      }
      // ---- best-of-N authoring: re-author the slow / non-engaging parity results ----
      if (FUSION_UNIT_ATTEMPTS > 1 && stageFits('unitside')) {
        const retryable = unitJobs.filter((j) => {
          if (j.collective || j.reused || !j.result) return false;
          const parity = String(j.result.parity || '').toLowerCase();
          const sp = Number(j.result.isolated_speedup);
          return parity === 'not_engaged' ||
            (parity === 'pass' && (j.result.engaged === false || !Number.isFinite(sp) || sp < FUSION_UNIT_RETRY_BELOW));
        });
        const attempts = [];
        for (const job of retryable) {
          for (let k = 2; k <= FUSION_UNIT_ATTEMPTS; k += 1) attempts.push({ job, k });
        }
        log(`KernelFusion Phase 3.0 best-of-${FUSION_UNIT_ATTEMPTS}: ${retryable.length} candidate(s) passed parity ` +
            `below ${FUSION_UNIT_RETRY_BELOW}x or did not engage (${retryable.map((j) => j.cid).join(', ') || '-'}); ` +
            `${attempts.length} more authoring attempt(s).`);
        if (attempts.length) {
          const UNIT_ISO3 = makeSem(GPU_LIST);
          await parallel(attempts.map((a) => async () => UNIT_ISO3.with(1, async (cards) => {
            const dir = `${EVAL_DIR}/fusion/attempts/a${a.k}`;
            await validateUnit(a.job, cards.join(','), {
              ...fusionLeaseInputs(`unit_${a.job.cid}_a${a.k}`),
              VERDICT_DIR: `${dir}/verdict`, KERNEL_DIR: `${dir}/unitside/${a.job.cid}`,
              AUTHOR_ATTEMPT: a.k, AUTHOR_ATTEMPTS: FUSION_UNIT_ATTEMPTS,
              PRIOR_VERDICT: a.job.result.verdict_path || '',
            });
          })));
          await fusionLeaseSettle('*', 'Phase 3.0 best-of-N barrier');
          const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
          const dirs = [];
          for (let k = 2; k <= FUSION_UNIT_ATTEMPTS; k += 1) dirs.push(`${EVAL_DIR}/fusion/attempts/a${k}`);
          const got = await fusionScriptOp(
            `${ex}python3 ${WORKFLOW_DIR}/scripts/fusion_reuse_verdicts.py --prefer-best --to ${EVAL_DIR}/verdict ` +
            dirs.map((d) => `--from ${d}`).join(' '), 'fusion-best-of-n', FUSION_REUSE_SCHEMA);
          const better = (got && got.reused && typeof got.reused === 'object') ? got.reused : {};
          for (const job of retryable) {
            const r = better[job.cid];
            if (!r) continue;
            job.result = { candidate_id: job.cid, verdict_path: `${EVAL_DIR}/verdict/${job.cid}__${r.side}.json`,
              parity: 'pass', isolated_speedup: Number(r.speedup), engaged: true,
              notes: `best of ${FUSION_UNIT_ATTEMPTS} authoring attempts: ${r.from}` };
          }
          log(`KernelFusion Phase 3.0 best-of-${FUSION_UNIT_ATTEMPTS}: a later attempt beat the first for ` +
              `${Object.keys(better).length} candidate(s): ` +
              Object.entries(better).map(([c, r]) => `${c} ${r.speedup}x`).join(', '));
        }
      }

      // ---- serving-side unit test for windows a single card cannot isolate ----
      if (FUSION_SERVING_UNIT && FUSION_SERVING_UNIT_MAX > 0 && fusionObservedWorkload && stageFits('unitside')) {
        const passedJob = (j) => fusionUnitStatusFromResult(j.result) === 'pass';
        const picks = [];
        for (const entry of fusionOrderByExpectedGain(fusionExecutionList)) {
          if (picks.length >= FUSION_SERVING_UNIT_MAX) break;
          if (!entry || !entry.exec_id) continue;
          const mine = unitJobs.filter((j) => j.entry.exec_id === entry.exec_id && !j.collective);
          if (mine.some(passedJob)) continue;
          // The validator answered (not isolable / not engaged / unmeasured) but nothing passed.
          const tried = mine.filter((j) => j.result);
          if (!tried.length) continue;
          const gain = (j) => { const g = fusionExpectedGainPct(j.entry); return g == null ? -1 : g; };
          picks.push(tried.reduce((best, j) => (gain(j) > gain(best) ? j : best), tried[0]));
        }
        log(`KernelFusion Phase 3.0 serving-side unit test: ${picks.length} window(s) with no passing rung ` +
            `(${picks.map((j) => j.cid).join(', ') || '-'}).`);
        if (picks.length) {
          const tagOf = (cid) => `su_${String(cid).replace(/[^A-Za-z0-9]+/g, '_')}`;
          const UNIT_ISO4 = makeSem(GPU_LIST);
          const authored = [];
          await parallel(picks.map((job) => async () => UNIT_ISO4.with(1, async (cards) => {
            const dir = `${EVAL_DIR}/fusion/serving_unit/${job.cid}`;
            const res = await safeAgent(
              roleAgent('fusion_unit_validator', 'serving_author',
                'Author the serving overlay for CANDIDATE_ID of TARGET_EXECUTION so it can be unit-tested ON THE ' +
                'LIVE STACK (the single-card validator could not isolate it: PRIOR_VERDICT says why and may hold a ' +
                'seed kernel under its KERNEL_DIR). Follow PHASE=serving_author in your role: the fused kernel under ' +
                'KERNEL_DIR, a lazy overlay that patches the window seam in the installed serving code and prints the ' +
                'BANNER_TAG banner, a combined-loader dir LOADER_DIR, and a kernel-level parity self-test on the card ' +
                'in GPU_IDS. Never start a server. Return FUSION_SERVING_AUTHOR_SCHEMA.' +
                fusionLeaseRules(`suauthor_${job.cid}`, false), {
                  EVAL_DIR, MODEL_PATH, MODEL_NAME, EXEC_ID: job.entry.exec_id, CANDIDATE_ID: job.cid,
                  TARGET_EXECUTION: job.entry, FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
                  SEMANTIC_TABLE_JSON: semanticTableJson, PRIOR_VERDICT: job.result.verdict_path || '',
                  KERNEL_DIR: `${dir}/kernel`, LOADER_DIR: `${dir}/loader`, BANNER_TAG: tagOf(job.cid),
                  SERVING_BACKEND: BACKEND, SERVING_FLAGS: curFlags || '', GPU_IDS: cards.join(','),
                  ...fusionLeaseInputs(`suauthor_${job.cid}`), ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
                }),
              { phase: 'KernelFusion', label: `fusion_serving_author:${job.cid}`,
                schema: FUSION_SERVING_AUTHOR_SCHEMA }, 1);
            if (res && res.status === 'authored' && res.loader_dir && res.kernel_source) {
              authored.push({ job, loader: res.loader_dir, banner: res.banner_tag || tagOf(job.cid),
                kernel: res.kernel_source });
            } else {
              log(`KernelFusion serving-side unit test: ${job.cid} not authored: ${(res && res.reason) || 'no result'}`);
            }
          })));
          await fusionLeaseSettle('*', 'Phase 3.0 serving-author barrier');
          if (authored.length) {
            const ex = EXEC_PREFIX ? `${EXEC_PREFIX} ` : '';
            const cmd = `${ex}python3 ${WORKFLOW_DIR}/scripts/fusion_serving_unit.py batch ` +
              `--root ${EVAL_DIR}/fusion/serving_unit --observed-workload ${fusionObservedWorkload} ` +
              `--tp ${SERVING_TP} --verdict-dir ${EVAL_DIR}/verdict ` +
              authored.map((a) => `--cand ${a.job.cid}::${a.loader}::${a.banner}::${a.kernel}`).join(' ');
            const leased = await withFusionServingLease('serving_unit', (leaseIn) => safeAgent(
              roleAgent('fusion_unit_validator', 'serving_measure',
                'Run SERVING_UNIT_CMD on the serving GPUs (PHASE=serving_measure in your role): export the serving ' +
                'config (BACKEND, MODEL, TP, GPU, EXTRA_SERVER_ARGS=SERVING_FLAGS, EXTRA_ENV=SERVING_ENV) and launch ' +
                'it DETACHED, then wait for it with fusion_leg_wait.sh until it exits (one base session plus one ' +
                'session per candidate, ~15 min each). Return its final JSON line as FUSION_SERVING_MEASURE_SCHEMA.' +
                fusionLeaseRules('serving_unit', true), {
                  EVAL_DIR, SERVING_UNIT_CMD: cmd, SERVING_UNIT_ROOT: `${EVAL_DIR}/fusion/serving_unit`,
                  BACKEND, MODEL: MODEL_PATH, TP: SERVING_TP, GPU: SERVING_GPU,
                  SERVING_FLAGS: curFlags || '', SERVING_ENV: curEnv || '',
                  ...leaseIn, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
                }),
              { phase: 'KernelFusion', label: 'fusion_serving_measure', schema: FUSION_SERVING_MEASURE_SCHEMA }, 1));
            const measured = (leased.value && leased.value.verdicts) || {};
            for (const a of authored) {
              const v = measured[a.job.cid];
              if (!v) continue;
              // The serving verdict stands next to the single-card one; the harness keeps the best side.
              a.job.servingResult = { candidate_id: a.job.cid, verdict_path: v.verdict_path,
                parity: v.parity, isolated_speedup: Number(v.speedup), engaged: !!v.engaged };
              if (fusionUnitStatusFromResult(a.job.servingResult) === 'pass') a.job.result = a.job.servingResult;
            }
            log(`KernelFusion Phase 3.0 serving-side unit test: ` + (leased.value
              ? authored.map((a) => `${a.job.cid} ${(measured[a.job.cid] || {}).parity || 'no verdict'} ` +
                `${(measured[a.job.cid] || {}).speedup || ''}`).join(', ')
              : `not run (${leased.leaseBusy ? `serving lease busy: ${leased.reason}` : 'measure agent failed'})`));
          }
        }
      }

      for (const job of unitJobs) {
        if (job.kind !== 'topk' && job.result) fusionExtraUnitResults[job.cid] = job.result;
        if (fusionUnitStatusFromResult(job.result) === 'pass' && job.result.verdict_path) {
          fusionUnitMeasured[job.cid] = { speedup: Number(job.result.isolated_speedup),
            verdict_path: job.result.verdict_path };
        }
      }
      // Only Top-K candidates are the harness's coverage denominator; a rung or pool item that
      // returned nothing is simply untested (it cannot land without a unit pass).
      const neverMeasured = unitJobs.filter((j) => !j.result && j.kind === 'topk').map((j) => `${j.cid}=${j.why}`)
        .concat(epSkipped.filter((x) => x.kind === 'topk').map((x) =>
          `${x.cid}=window over the expert-parallel MoE (megamoe/ep_combine): needs the whole EP group, no single-card unit test`));
      const aggregateOnce = (extra) => safeAgent(
        roleAgent('fusion_unit_validator', 'aggregate',
          (extra || '') +
          'Never move, rename or delete anything in VERDICT_DIR, and never --waive a candidate that has a ' +
          'verdict file there: a waiver is only for a candidate nobody could measure. ' +
          'Run scripts/fusion_unitside_harness.py over VERDICT_DIR for the Top-K denominator (pass ' +
          '--semantic-table SEMANTIC_TABLE_JSON when set, --observed-workload OBSERVED_WORKLOAD_JSON when set, ' +
          '--shape-envelope SHAPE_ENVELOPE_JSON after generating it ' +
          'with scripts/serving_shape_envelope.py from SERVING_BACKEND/SERVING_FLAGS, and every BUDGET_SKIPPED item ' +
          'as --budget-skipped), ' +
          'publish EVAL_DIR/04_FUSION_UNITSIDE.md even when the gate fails, run report_index.py, and return ' +
          'FUSION_UNIT_AGG_SCHEMA with unit_status_by_candidate copied from fusion_unitside.json ' +
          '(alternative-rung and candidate-pool ids included).', {
            EVAL_DIR, FUSION_DIR: `${EVAL_DIR}/fusion`, VERDICT_DIR: `${EVAL_DIR}/verdict`,
            FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
            FUSION_CANDIDATES_JSON: FUSION_INPUTS.FUSION_CANDIDATES_JSON,
            SEMANTIC_TABLE_JSON: semanticTableJson, ...shapeEnvelopeInputs,
            FUSION_UNITSIDE_BUDGET: kinds('topk'),
            EXTRA_UNIT_CANDIDATES: unitJobs.filter((j) => j.kind !== 'topk').map((j) => j.cid),
            EQUIVALENT_COVERED: [], SUBSUMED_COVERED: [], BUDGET_SKIPPED: neverMeasured,
            ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
          }),
        { phase: 'KernelFusion', label: 'fusion_unit_validator:aggregate', schema: FUSION_UNIT_AGG_SCHEMA }, 2);
      let unitAgg = await aggregateOnce('');
      // A validator pass the aggregate dropped (waived, moved away, missing) is re-aggregated once,
      // naming those candidates: run 7's aggregate moved every verdict out of VERDICT_DIR and
      // waived all eight, including a measured parity-pass at 1.04x.
      const passedCids = unitJobs.filter((j) => fusionUnitStatusFromResult(j.result) === 'pass').map((j) => j.cid);
      const aggStatus = (unitAgg && unitAgg.unit_status_by_candidate) || {};
      const dropped = passedCids.filter((cid) =>
        !['pass', 'equivalent_pass', 'subsumed_pass', 'fail', 'needs_diagnosis', 'blocked']
          .includes(String(aggStatus[cid] || '')));
      if (dropped.length) {
        log(`KernelFusion Phase 3.0: the aggregate reported no measured status for validator pass(es) ` +
            `${dropped.join(', ')} (${dropped.map((c) => aggStatus[c] || 'missing').join(', ')}); re-aggregating once.`);
        const again = await aggregateOnce(
          `RE-AGGREGATE: the previous aggregate dropped ${dropped.join(', ')}, whose validators returned ` +
          'parity pass with a measured speedup. Restore any verdict file that was moved out of VERDICT_DIR ' +
          '(e.g. from a subdirectory) to VERDICT_DIR, do not waive those candidates, and let the harness judge ' +
          'their verdicts. ');
        if (again) unitAgg = again;
      }
      // Only a path the aggregate reported: the final apply-back gate passes it to the harness,
      // which must not be pointed at a file that was never written.
      FUSION_INPUTS.FUSION_UNITSIDE_JSON = (unitAgg && unitAgg.fusion_unitside_json) || '';
      fusionUnitStatus = (unitAgg && unitAgg.unit_status_by_candidate
        && typeof unitAgg.unit_status_by_candidate === 'object') ? { ...unitAgg.unit_status_by_candidate } : null;
      // A rung / pool status the harness did not report falls back to the validator's own numbers.
      for (const [cid, res] of Object.entries(fusionExtraUnitResults)) {
        const st = fusionUnitStatusFromResult(res);
        if (!st) continue;
        if (!fusionUnitStatus) fusionUnitStatus = {};
        if (!fusionUnitStatus[cid]) fusionUnitStatus[cid] = st;
      }
      // Apply-back reads the eligible ids; the harness writes them, else derive them from the
      // per-candidate statuses (same pass set the harness uses).
      fusionUnitEligibleIds = (unitAgg && Array.isArray(unitAgg.applyback_eligible_ids))
        ? unitAgg.applyback_eligible_ids
        : (fusionUnitStatus ? Object.keys(fusionUnitStatus).filter((cid) =>
          ['pass', 'equivalent_pass', 'subsumed_pass'].includes(String(fusionUnitStatus[cid]))) : null);
      log(`KernelFusion Phase 3.0: ${unitAgg ? (unitAgg.status || '?') : 'aggregate agent failed'}; ` +
          `${unitJobs.length - unitJobs.filter((j) => !j.result).length}/${unitJobs.length} unit test(s) returned a verdict` +
          (neverMeasured.length ? `; never measured: ${neverMeasured.join('; ')}` : '') +
          `. Report: ${(unitAgg && unitAgg.fusion_unitside_md) || `${EVAL_DIR}/04_FUSION_UNITSIDE.md`}`);
    }
  }

  // ---- StageFusion (L3): one EP MoE stage -> GEAK mode=mega -> one overlay ----
  // Serial on the serving GPUs, every step under the serving lease so no KernelFusion server shares
  // them with mega's own group leases: prepare (the served operator, its frozen source tree, a
  // multi-rank UT checked on the serving GPUs) -> kernel_mega.js -> package the delivered candidate
  // as an overlay that joins the combined A/B.
  if (FUSION_L3_ON && fusionL3Stages.length) {
    phase('StageFusion');
    fusionFailureStage = 'stage_fusion';
    const picked = fusionL3Stages
      .filter((st) => !(Number(st.request_share_pct) < FUSION_L3_MIN_SHARE_PCT))
      .slice(0, FUSION_L3_MAX_STAGES);
    fusionL3 = { stages: fusionL3Stages.map((st) => ({
      stage_id: st.stage_id, phase: st.phase || '', request_share_pct: st.request_share_pct,
      subsumes: st.subsumes || [],
      status: picked.includes(st) ? 'pending' : 'not_picked',
      reason: picked.includes(st) ? '' : `below fusion_l3_min_share_pct ${FUSION_L3_MIN_SHARE_PCT}% ` +
        `or past fusion_l3_max_stages ${FUSION_L3_MAX_STAGES}` })) };
    const l3Row = (id) => fusionL3.stages.find((r) => r.stage_id === id);
    const skipAll = (status, why) => {
      for (const r of fusionL3.stages) if (r.status === 'pending') { r.status = status; r.reason = why; }
      log(`KernelFusion StageFusion: ${why}`);
    };
    if (FUSION_APPLYBACK_MODE !== 'combined') {
      skipAll('skipped', 'L3 overlays join only the combined A/B (fusion_applyback_mode=combined); not run.');
    } else if (!FUSION_L3_REUSE_DIR && typeof workflow !== 'function') {
      skipAll('skipped', 'this runtime has no workflow() to call GEAK mode=mega; not run.');
    }
    for (const stage of picked) {
      const row = l3Row(stage.stage_id);
      if (row.status !== 'pending') continue;
      const id = String(stage.stage_id);
      const l3Dir = `${EVAL_DIR}/fusion/l3/${id}`;
      const reuseDir = FUSION_L3_REUSE_DIR ? `${FUSION_L3_REUSE_DIR}/fusion/l3/${id}` : '';
      let prep = null;
      let mega = null;
      if (!reuseDir) {
        const leftS = TIME_HEAD_DEADLINE_MS == null ? Infinity
          : Math.floor((TIME_HEAD_DEADLINE_MS - ELAPSED_MS) / 1000) - 60 * STAGE_MIN.applyback;
        const budgetS = Math.min(FUSION_L3_BUDGET_S, leftS);
        if (!(budgetS >= 60 * STAGE_MIN.l3)) {
          row.status = 'out_of_time';
          row.reason = `needs >= ${STAGE_MIN.l3}min before apply-back, ${Math.max(0, Math.round(leftS / 60))}min left`;
          log(`KernelFusion StageFusion ${id}: ${row.reason}; not starting it.`);
          continue;
        }
        const prepareOnce = (attempt, prior) => {
          const prepLease = `l3_prepare_${id}${attempt > 1 ? `_a${attempt}` : ''}`;
          return withFusionServingLease(prepLease, (leaseIn) => safeAgent(
            roleAgent('fusion_stage_l3', 'prepare',
              (prior ? `RETRY: the previous prepare of this stage returned blocked (${prior}). Read what it left under ` +
                'L3_DIR, fix what is in your power (a different call site, an existing benchmark to adapt, a UT bug) ' +
                'and try once more; return blocked again only with the evidence. ' : '') +
              'Prepare L3_STAGE for GEAK mode=mega (PHASE=prepare in your role): find the operator the served ' +
              'model runs for this stage, freeze its source tree under L3_DIR, write the multi-rank UT, check it on ' +
              'SERVING_GPU under a GEAK group lease, write L3_DIR/prepared.json and return FUSION_L3_PREP_SCHEMA.' +
              fusionLeaseRules(prepLease, true), {
                EVAL_DIR, MODEL_PATH, MODEL_NAME, L3_STAGE: stage, L3_DIR: l3Dir, EP_DEFAULT: FUSION_L3_EP_DEFAULT,
                ...(FUSION_L3_OPERATOR ? { OPERATOR_HINT: FUSION_L3_OPERATOR } : {}),
                SEMANTIC_TABLE_JSON: (semantics && semantics.semantic_table_json) || '',
                ...(fusionObservedWorkload ? { OBSERVED_WORKLOAD_JSON: fusionObservedWorkload } : {}),
                SERVING_FLAGS: curFlags || '', SERVING_ENV: curEnv || '', CURRENT_OVERLAY: curOverlay || '',
                GEAK_KERNEL_WORKFLOW_DIR: FUSION_L3_GEAK_DIR,
                ...leaseIn, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
              }),
            { phase: 'KernelFusion', label: `fusion_stage_l3:prepare:${id}${attempt > 1 ? ':retry' : ''}`,
              schema: FUSION_L3_PREP_SCHEMA }, 1));
        };
        const ready = (p) => !!(p && p.status === 'ready' && p.kernel_path && p.benchmark_harness);
        let prepared = await prepareOnce(1, null);
        prep = prepared.value || null;
        if (!prepared.leaseBusy && !ready(prep)) {
          prepared = await prepareOnce(2, (prep && prep.reason) || 'no result');
          prep = prepared.value || null;
        }
        if (!ready(prep)) {
          row.status = prepared.leaseBusy ? 'gpu_busy' : 'prepare_blocked';
          row.reason = prepared.leaseBusy ? prepared.reason : ((prep && prep.reason) || 'prepare returned no result');
          log(`KernelFusion StageFusion ${id}: not prepared (${row.reason}).`);
          continue;
        }
        const ep = Math.min(GPU_LIST.length || SERVING_TP,
          Number(prep.ep_size) > 0 ? Number(prep.ep_size) : FUSION_L3_EP_DEFAULT);
        const megaArgs = {
          ...(prep.mega_args && typeof prep.mega_args === 'object' ? prep.mega_args : {}),
          mode: 'mega', workflow_dir: FUSION_L3_GEAK_DIR,
          kernel_path: prep.kernel_path, benchmark_harness: prep.benchmark_harness,
          gpu_ids: SERVING_GPU, gpus_per_job: ep,
          exp_root: `${l3Dir}/mega_runs`, state_dir: `${l3Dir}/mega_state`,
          budget: FUSION_L3_ROUNDS, mega_time_budget_s: Number.isFinite(budgetS) ? budgetS : FUSION_L3_BUDGET_S,
          ...FUSION_L3_MEGA_ARGS,
        };
        row.mega_args = { kernel_path: megaArgs.kernel_path, benchmark_harness: megaArgs.benchmark_harness,
          gpu_ids: megaArgs.gpu_ids, gpus_per_job: megaArgs.gpus_per_job, budget: megaArgs.budget,
          mega_time_budget_s: megaArgs.mega_time_budget_s, state_dir: megaArgs.state_dir };
        log(`KernelFusion StageFusion ${id}: GEAK mode=mega on GPU ${megaArgs.gpu_ids} (gpus_per_job ` +
            `${megaArgs.gpus_per_job}), ${megaArgs.budget} rounds, ${Math.round(megaArgs.mega_time_budget_s / 60)}min; ` +
            `operator ${prep.call_site || prep.package || '?'}.`);
        const ran = await withFusionServingLease(`l3_mega_${id}`, async () => {
          try {
            return { ret: await workflow({ scriptPath: `${FUSION_L3_GEAK_DIR}/kernel_mega.js` }, megaArgs) };
          } catch (e) {
            return { error: (e && e.message) ? e.message : String(e) };
          }
        });
        const out = ran.value || {};
        mega = fusionMegaSummary(out.ret);
        row.mega = mega;
        if (ran.leaseBusy || out.error || !(mega && mega.deliverable)) {
          row.status = ran.leaseBusy ? 'gpu_busy' : (out.error ? 'mega_error' : 'mega_no_deliverable');
          row.reason = ran.leaseBusy ? ran.reason
            : (out.error || (mega && mega.note) || 'mega returned no result');
          log(`KernelFusion StageFusion ${id}: ${row.status} (${row.reason}).`);
          continue;
        }
        log(`KernelFusion StageFusion ${id}: mega delivered ${mega.selected_candidate} at ` +
            `${mega.final_speedup}x vs its frozen baseline.`);
      }
      const pkgLease = `l3_package_${id}`;
      const packaged = await withFusionServingLease(pkgLease, (leaseIn) => safeAgent(
        roleAgent('fusion_stage_l3', 'package',
          'Package the delivered mega candidate of L3_STAGE as one KernelFusion overlay (PHASE=package in your ' +
          'role): the serving process must import the candidate\'s code with its activation, and print the ' +
          'BANNER_TAG banner from inside the stage\'s call site. Check it with the UT on SERVING_GPU, then return ' +
          'FUSION_AUTHOR_SCHEMA. Do not start a serving server.' + fusionLeaseRules(pkgLease, true), {
            EVAL_DIR, MODEL_PATH, MODEL_NAME, L3_STAGE: stage, L3_DIR: l3Dir, BANNER_TAG: id,
            ...(reuseDir
              ? { MEGA_RESULT_JSON: `${reuseDir}/mega_result.json`, PREPARED_JSON: `${reuseDir}/prepared.json` }
              : { MEGA_RESULT: mega, PREPARED: prep }),
            FUSION_OVERLAYS_DIR: `${EVAL_DIR}/fusion/fusion_overlays`,
            CURRENT_OVERLAY: curOverlay || '', CURRENT_FLAGS: curFlags || '', CURRENT_ENV: curEnv || '',
            GEAK_KERNEL_WORKFLOW_DIR: FUSION_L3_GEAK_DIR,
            ...leaseIn, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
          }),
        { phase: 'KernelFusion', label: `fusion_stage_l3:package:${id}`, schema: FUSION_AUTHOR_SCHEMA }, 1));
      const pkg = packaged.value || null;
      if (pkg && pkg.status === 'authored' && pkg.overlay_dir) {
        fusionL3Authored.push({ ...pkg, exec_id: id, level: 'L3', banner_tag: pkg.banner_tag || id,
          candidate_ids: [id], subsumes: stage.subsumes || [] });
        row.status = 'packaged';
        row.overlay_dir = pkg.overlay_dir;
      } else {
        row.status = packaged.leaseBusy ? 'gpu_busy' : 'package_blocked';
        row.reason = packaged.leaseBusy ? packaged.reason : ((pkg && pkg.reason) || 'package returned no result');
      }
      log(`KernelFusion StageFusion ${id}: ${row.status}` + (row.reason ? ` (${row.reason})` : '') +
          (reuseDir ? `; mega result reused from ${reuseDir}` : '') + '.');
    }
  }

  // Apply-back is best-effort. A discovery/validation failure leaves the inputs
  // empty and simply hands the pre-Fusion stack to the e2e workflow. Run one
  // execution-list entry (including its degrade ladder) per agent call so a later
  // failure cannot erase terminal results already returned by earlier calls.
  let fapply = null;
  let fusionApplyFailedExec = '';
  let fusionApplyUnprocessed = [];
  let combinedFailure = '';   // set when the combined A/B fails KernelFusion (no serial fallback)
  if (FUSION_INPUTS.FUSION_TOPK_JSON && (FUSION_INPUTS.FUSION_UNITSIDE_JSON || fusionL3Authored.length) &&
      stageFits('applyback')) {
    phase('ApplyBack');
    const fusionApplyCandidate = (entry) =>
      fusionPickApplyCandidate(entry, fusionUnitEligibleIds, fusionUnitMeasured);
    const fusionApplyEntries = fusionApplyBudgetEntries(
      fusionApplyRungs(fusionExecutionList, fusionUnitEligibleIds), fusionUnitEligibleIds, FUSION_BUDGET);
    let applyState = {
      accepted_fusions: [], final_overlay: curOverlay,
      e2e_throughput_tok_s: curTput, accepted_flags: curFlags, accepted_env: curEnv,
      rejected: [], deferred: [], deferred_author_count: 0,
      applyback_gate_json: '', applyback_report_md: '', learned_cards: [], notes: '',
    };
    const rowKey = (r) => String((r && (r.exec_id || r.candidate_id || r.fusion)) || '');
    const mergeRows = (before, after) => {
      const out = []; const at = new Map();
      for (const row of (before || []).concat(after || [])) {
        const key = rowKey(row);
        if (!key) { out.push(row); continue; }
        if (at.has(key)) out[at.get(key)] = row;
        else { at.set(key, out.length); out.push(row); }
      }
      return out;
    };
    const mergeCards = (before, after) => {
      const out = []; const at = new Map();
      for (const card of (before || []).concat(after || [])) {
        const key = String((card && (card.key || card.card)) || '');
        if (!key) { out.push(card); continue; }
        if (at.has(key)) out[at.get(key)] = card;
        else { at.set(key, out.length); out.push(card); }
      }
      return out;
    };
    const dispositionIds = () => new Set(
      (applyState.accepted_fusions || []).concat(applyState.rejected || [], applyState.deferred || [])
        .map(rowKey).filter(Boolean));

    // Merge one apply-back call's aggregate into applyState and, when it accepted
    // fusions, make its stack the current one. Shared by combined and serial mode.
    const commitStep = (step, accuracyKey, label) => {
      const acceptedBefore = new Set((applyState.accepted_fusions || []).map(rowKey).filter(Boolean));
      const mergedAccepted = mergeRows(applyState.accepted_fusions, step.accepted_fusions);
      const acceptedIds = new Set(mergedAccepted.map(rowKey).filter(Boolean));
      const mergedRejected = mergeRows(applyState.rejected, step.rejected)
        .filter((r) => !acceptedIds.has(rowKey(r)));
      const rejectedIds = new Set(mergedRejected.map(rowKey).filter(Boolean));
      const mergedDeferred = mergeRows(applyState.deferred, step.deferred)
        .filter((r) => !acceptedIds.has(rowKey(r)) && !rejectedIds.has(rowKey(r)));
      applyState = {
        ...applyState, ...step,
        accepted_fusions: mergedAccepted,
        rejected: mergedRejected,
        deferred: mergedDeferred,
        learned_cards: mergeCards(applyState.learned_cards, step.learned_cards),
        deferred_author_count: Math.max(
          Number(applyState.deferred_author_count) || 0,
          Number(step.deferred_author_count) || 0),
      };
      fapply = applyState;

      const newlyAccepted = mergedAccepted.filter((r) => !acceptedBefore.has(rowKey(r)));
      const measuredRef = step.accuracy_reference;
      if (measuredRef && measuredRef.stack_key === accuracyKey && Number(measuredRef.exact_match) >= 0) {
        fusionAccuracyRef = { ...measuredRef };
      }
      if (newlyAccepted.length) {
        curOverlay = step.final_overlay || curOverlay;
        curFlags = step.accepted_flags || curFlags;
        curEnv = step.accepted_env || curEnv;
        if (step.e2e_throughput_tok_s && step.e2e_throughput_tok_s > curTput) {
          curTput = step.e2e_throughput_tok_s;
        }
        for (const accepted of newlyAccepted) acceptedFusions.push(accepted);
        // The accepted candidate's score was measured on exactly the stack that is
        // now current, so it is the next call's base; without it the cache resets.
        const acc = step.accepted_accuracy;
        fusionAccuracyRef = (acc && Number(acc.exact_match) >= 0)
          ? { ...acc, stack_key: fusionStackKey() } : null;
        log(`KernelFusion apply-back ${label}: committed ${newlyAccepted.length} ` +
            `terminal fusion win(s); e2e now ${curTput} tok/s.`);
      }
      // The outer catch restores the latest terminal checkpoint, not the state from
      // before every Apply-back call. Earlier proven wins survive a later JS exception.
      fusionRecoveryState.overlay = curOverlay;
      fusionRecoveryState.flags = curFlags;
      fusionRecoveryState.env = curEnv;
      fusionRecoveryState.throughput = curTput;
      fusionRecoveryState.acceptedFusionCount = acceptedFusions.length;
      fusionRecoveryState.inputs = { ...FUSION_INPUTS };
      fusionRecoveryState.applyResult = fapply;
    };

    // ---- combined mode: author every pass, then ONE A/B for the whole stack ----
    // Authoring needs no server, so in-budget unit-side passes are authored in waves,
    // one GPU each (for the overlay's parity self-check). The stack is then gated by
    // one base server vs one stacked server. A rejected (or failed) combined A/B fails
    // KernelFusion with its reason and the run moves on; nothing falls back to serial.
    const authoredByExec = new Map();
    let combinedAccepted = false;
    if (FUSION_APPLYBACK_MODE === 'combined') {
      const eligible = new Set((fusionUnitEligibleIds || []).map(String));
      const toAuthor = fusionApplyEntries.filter((e) => e && e.exec_id && FUSION_INPUTS.FUSION_UNITSIDE_JSON &&
        (!Array.isArray(fusionUnitEligibleIds) ||
         (e.candidate_ids || []).some((cid) => eligible.has(String(cid)))));
      const authorWidth = Math.max(1, Math.min(FUSION_UNIT_PARALLEL, GPU_LIST.length || 1));
      fusionFailureStage = 'apply_back:author';
      log(`KernelFusion apply-back (combined): authoring ${toAuthor.length} overlay(s), ` +
          `${authorWidth} at a time; budget ${FUSION_BUDGET}.`);
      const authored = [];
      const authorOne = (entry, slot, repair) => safeAgent(
          roleAgent('fusion_integrator', 'author_one',
            (repair ? 'REPAIR: your overlay for TARGET_EXEC_ID loaded on the stacked server but its ENGAGED banner ' +
              'never printed (PREFLIGHT_EVIDENCE): the function it patches is not on the served call path. Follow ' +
              '"Repairing a seam that never engaged" in your role, re-target the overlay at the call site that ' +
              'actually runs, and write it to the same overlay dir. ' : '') +
            'Author the overlay for exactly TARGET_EXEC_ID and check its parity on GPU_ID only. ' +
            'Do not launch a serving server; other entries are being authored on the other cards. ' +
            (fusionApplyCandidate(entry)
              ? `Apply EXACTLY APPLY_CANDIDATE_ID: wire the kernel its unit-side verdict APPLY_VERDICT measured ` +
                '(its kernel_source) at that candidate\'s own window. Do not re-author a different kernel or switch to ' +
                'another candidate or rung of this entry: the unit test is the evidence for that kernel only. ' +
                'Print the ENGAGED banner from inside the patched call path (the first time it runs), not at patch time. '
              : '') +
            'Return candidate_ids = [the candidate the overlay wires].', {
              EVAL_DIR, MODEL_PATH, GPU_ID: String(GPU_LIST[slot % GPU_LIST.length] || GPU_LIST[0] || '0'),
              TP: SERVING_TP, TARGET_EXEC_ID: entry.exec_id, TARGET_EXECUTION: entry,
              ...(fusionApplyCandidate(entry) ? { APPLY_CANDIDATE_ID: fusionApplyCandidate(entry).cid,
                APPLY_VERDICT: fusionApplyCandidate(entry).verdict_path,
                APPLY_UNIT_SPEEDUP: fusionApplyCandidate(entry).speedup } : {}),
              FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
              FUSION_CANDIDATES_JSON: FUSION_INPUTS.FUSION_CANDIDATES_JSON,
              FUSION_UNITSIDE_JSON: FUSION_INPUTS.FUSION_UNITSIDE_JSON,
              CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
              FUSION_OVERLAYS_DIR: `${EVAL_DIR}/fusion/fusion_overlays`,
              ...(repair ? { PREFLIGHT_EVIDENCE: repair, PRIOR_AUTHOR_RESULT: repair.prior } : {}),
              ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
            }),
          { phase: 'KernelFusion',
            label: `fusion_integrator:author_one:${entry.exec_id}${repair ? ':repair' : ''}`,
            schema: FUSION_AUTHOR_SCHEMA }, 1);
      for (let i = 0; i < toAuthor.length; i += authorWidth) {
        const chunk = toAuthor.slice(i, i + authorWidth);
        const rs = await Promise.all(chunk.map((entry, slot) => authorOne(entry, slot, null)));
        chunk.forEach((entry, k) => {
          const r = rs[k] || { exec_id: entry.exec_id, status: 'blocked',
            reason: 'author_one returned no result' };
          authored.push(r);
          if (r.status === 'authored' && r.overlay_dir) authoredByExec.set(String(entry.exec_id), r);
        });
      }
      for (const r of fusionL3Authored) {
        authored.push(r);
        authoredByExec.set(String(r.exec_id), r);
      }
      let preflightDark = null;   // banner tags the stacked server never printed (null = not checked)
      // ---- engagement preflight + one repair round (before the hours-long A/B) ----
      // An overlay can load cleanly and still patch a function the served path never calls
      // (DeepSeek-V4 run 10: 4 of 7 -- a base-class method its subclass overrides, a name the
      // caller imported into its own module, a branch the serving flags never take). Boot the
      // stacked server once, collect which banners never printed, and send only those overlays
      // back to their authors with that evidence; the A/B's own preflight then drops what is
      // still dark.
      if (FUSION_PREFLIGHT_REPAIR && authoredByExec.size && stageFits('applyback')) {
        fusionFailureStage = 'apply_back:preflight';
        const leased = await withFusionServingLease('preflight', (leaseIn) => safeAgent(
          roleAgent('fusion_integrator', 'preflight',
            'Run the engagement preflight over every AUTHORED overlay (PHASE=preflight in your role): build the ' +
            'combined loader, run COMBINED_AB_SCRIPT with PREFLIGHT_ONLY=1, and return its ' +
            'combined_ab/preflight/engagement.json as FUSION_PREFLIGHT_SCHEMA.' +
            fusionLeaseRules('preflight', true), {
              EVAL_DIR, MODEL_PATH, SERVING_GPU, TP: SERVING_TP, AUTHORED: authored,
              CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
              FUSION_OVERLAYS_DIR: `${EVAL_DIR}/fusion/fusion_overlays`,
              COMBINED_AB_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_combined_ab.sh`,
              GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_gsm8k_eval.py`,
              ...leaseIn, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
            }),
          { phase: 'KernelFusion', label: 'fusion_integrator:preflight', schema: FUSION_PREFLIGHT_SCHEMA }, 1));
        const pf = leased.value || null;
        // Authors return the tag with or without the banner's "overlay-" prefix (run 14: e01 came
        // back as "overlay-e01_..." while the preflight reports "e01_..." and was never repaired).
        const bareTag = (t) => String(t || '').replace(/^\[?overlay-/, '').replace(/\]$/, '');
        const dark = (pf && pf.server_ok && Array.isArray(pf.missing)) ? pf.missing.map(bareTag) : [];
        if (pf && pf.server_ok) preflightDark = dark;
        const darkEntries = toAuthor.filter((e) => {
          const r = authoredByExec.get(String(e.exec_id));
          return r && dark.includes(bareTag(r.banner_tag));
        });
        log(`KernelFusion apply-back preflight: ` + (pf
          ? `server ${pf.server_ok ? 'up' : 'DID NOT START'}; engaged ${JSON.stringify(pf.engaged || {})}; ` +
            `repairing ${darkEntries.length} overlay(s) that never engaged (${dark.join(', ') || '-'})`
          : `not run (${leased.leaseBusy ? `serving lease busy: ${leased.reason}` : 'preflight agent failed'})`));
        if (darkEntries.length) {
          const fixed = await Promise.all(darkEntries.map((entry, slot) => {
            const prior = authoredByExec.get(String(entry.exec_id));
            return authorOne(entry, slot, { banner_tag: prior.banner_tag, engaged_lines: 0,
              engagement_json: `${EVAL_DIR}/fusion/combined_ab/preflight/engagement.json`,
              server_log: pf.server_log || `${EVAL_DIR}/fusion/combined_ab/preflight/server.log`,
              prior });
          }));
          darkEntries.forEach((entry, k) => {
            const r = fixed[k];
            if (!r || r.status !== 'authored' || !r.overlay_dir) return;  // keep the prior row
            const idx = authored.findIndex((a) => String(a.exec_id) === String(entry.exec_id));
            if (idx >= 0) authored[idx] = r;
            authoredByExec.set(String(entry.exec_id), r);
          });
        }
      }
      // An L3 overlay that engaged replaces the stage's own L1/L2 overlays: they patch kernels the
      // mega path no longer launches. The integrator records them as subsumed.
      const l3Subsumed = {};
      const l3SubsumedRows = [];   // restored if the stack has to be gated without L3
      for (const r of fusionL3Authored) {
        const tag = String(r.banner_tag || '').replace(/^\[?overlay-/, '').replace(/\]$/, '');
        if (preflightDark && preflightDark.includes(tag)) continue;
        for (const execId of r.subsumes || []) {
          if (!l3Subsumed[execId] && authoredByExec.has(String(execId))) l3Subsumed[execId] = r.exec_id;
        }
      }
      if (Object.keys(l3Subsumed).length) {
        for (const execId of Object.keys(l3Subsumed)) {
          authoredByExec.delete(String(execId));
          const idx = authored.findIndex((a) => String(a.exec_id) === String(execId));
          if (idx >= 0) l3SubsumedRows.push(authored.splice(idx, 1)[0]);
        }
        log(`KernelFusion apply-back: L3 overlay(s) subsume ${Object.entries(l3Subsumed)
          .map(([e, st]) => `${e}->${st}`).join(', ')}; those overlays leave the stack.`);
      }
      // One combined A/B over `authored`. `withoutL3` = the L3 rows that a first, rejected A/B
      // carried: the L1/L2 stack is then gated alone, so an L3 overlay that loses end to end
      // cannot take the L1/L2 wins down with it.
      const combinedAb = (authored, accuracyKey, withoutL3) => safeAgent(
        roleAgent('fusion_integrator', 'combined_ab',
          (withoutL3 ? `RETRY WITHOUT L3: the stack with the L3 overlay(s) was rejected (${withoutL3.reason}). ` +
            'Gate the L1/L2 stack in AUTHORED alone, writing to OUT_DIR=$EVAL_DIR/fusion/combined_ab_without_l3 ' +
            'and the combined loader FUSION_OVERLAYS_DIR/<model>/combined_without_l3/; record every L3_REJECTED row ' +
            'in rejected with its reason. ' : '') +
          'Stack every AUTHORED overlay that does not conflict with a better-ranked one, gate the stack with ' +
          'ONE run of COMBINED_AB_SCRIPT (one base server vs one stacked server), and return the full aggregate ' +
          'FUSION_APPLY_SCHEMA with combined_decision. On reject, record no dispositions: KernelFusion fails with the decision reason.' +
          (fusionL3Authored.length && !withoutL3 ? ' AUTHORED includes L3 stage overlays (level L3, GEAK mode=mega): stack them like any ' +
            'other row; record every L3_SUBSUMED entry in rejected with reason "subsumed by L3 <stage_id>".' : ''), {
            EVAL_DIR, MODEL_PATH, SERVING_GPU, TP: SERVING_TP, WORKLOAD,
            AUTHORED: authored, PRIOR_APPLY_RESULT: applyState,
            FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
            FUSION_CANDIDATES_JSON: FUSION_INPUTS.FUSION_CANDIDATES_JSON,
            FUSION_UNITSIDE_JSON: FUSION_INPUTS.FUSION_UNITSIDE_JSON,
            CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
            CURRENT_THROUGHPUT: curTput, BASELINE_THROUGHPUT: BASELINE_TPUT,
            NOISE_BAND_PCT: NOISE_BAND, FUSION_BUDGET,
            FUSION_OVERLAYS_DIR: `${EVAL_DIR}/fusion/fusion_overlays`,
            COMBINED_AB_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_combined_ab.sh`,
            ...(fusionObservedWorkload ? { OBSERVED_WORKLOAD_JSON: fusionObservedWorkload } : {}),
            AB_REPEATS: FUSION_COMBINED_AB_REPEATS, AB_REPEAT_MODE: FUSION_AB_REPEAT_MODE,
            GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_gsm8k_eval.py`,
            GSM8K_CONCURRENCY: fusionGsm8kConcurrency(curFlags), GSM8K_THINKING: FUSION_GSM8K_THINKING,
            ACCURACY_STACK_KEY: accuracyKey,
            ...(fusionL3Authored.length && !withoutL3 ? { L3_SUBSUMED: l3Subsumed } : {}),
            ...(withoutL3 ? { L3_REJECTED: withoutL3.rows } : {}),
            ...ACCURACY_INPUTS, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
          }),
        { phase: 'KernelFusion', label: `fusion_integrator:combined_ab${withoutL3 ? ':without_l3' : ''}`,
          schema: FUSION_APPLY_SCHEMA, timeoutMs: FUSION_APPLY_TIMEOUT_MS }, 1);
      const rejectReason = (step) => (step
        ? `combined A/B rejected: ${((step.combined_ab || {}).reason) || step.notes || 'see fusion/combined_ab/combined_decision.json'}`
        : 'combined A/B returned no result');
      if (authoredByExec.size) {
        fusionFailureStage = 'apply_back:combined_ab';
        let accuracyKey = fusionStackKey();
        let step = await combinedAb(authored, accuracyKey, null);
        const l3InStack = authored.filter((r) => r.level === 'L3');
        if (step && step.combined_decision !== 'accept' && l3InStack.length) {
          const reason = rejectReason(step);
          for (const r of l3InStack) {
            const st = fusionL3 && fusionL3.stages.find((x) => x.stage_id === r.exec_id);
            if (st) { st.status = 'ab_rejected'; st.reason = reason; }
          }
          const rest = authored.filter((r) => r.level !== 'L3').concat(l3SubsumedRows);
          log(`KernelFusion apply-back (combined): ${reason} with L3 ${l3InStack.map((r) => r.exec_id).join(', ')} ` +
              `in the stack; ` + (rest.length && stageFits('applyback')
              ? `gating the ${rest.length} L1/L2 overlay(s) alone.` : 'no L1/L2 overlay left to gate alone.'));
          if (rest.length && stageFits('applyback')) {
            accuracyKey = fusionStackKey();
            step = await combinedAb(rest, accuracyKey, { reason,
              rows: l3InStack.map((r) => ({ exec_id: r.exec_id, reason: `L3 overlay: ${reason}` })) });
          }
        }
        if (step && step.combined_decision === 'accept') {
          commitStep(step, accuracyKey, 'combined');
          combinedAccepted = true;
        } else {
          combinedFailure = rejectReason(step);
          log(`KernelFusion apply-back (combined): ${combinedFailure}; KernelFusion fails here (no per-entry fallback).`);
        }
        for (const st of (fusionL3 ? fusionL3.stages : [])) {
          if (st.status !== 'packaged') continue;
          const kept = combinedAccepted && (step.accepted_fusions || []).some((r) => r && r.exec_id === st.stage_id);
          st.status = kept ? 'applied' : 'ab_rejected';
          if (!kept) st.reason = combinedFailure || 'not kept in the accepted stack';
        }
      } else {
        combinedFailure = 'no unit-side pass could be authored into an overlay';
        log(`KernelFusion apply-back (combined): ${combinedFailure}; KernelFusion fails here (no per-entry fallback).`);
      }
    }

    if (!combinedAccepted && FUSION_APPLYBACK_MODE === 'serial') {
      log(`KernelFusion apply-back: ${fusionApplyEntries.length} execution-list entry/ladder call(s), ` +
          `budget ${FUSION_BUDGET}; terminal results are committed after each call.`);
      for (let applyIndex = 0; applyIndex < fusionApplyEntries.length; applyIndex++) {
        const applyEntry = fusionApplyEntries[applyIndex];
        if (!applyEntry || !applyEntry.exec_id || dispositionIds().has(String(applyEntry.exec_id))) continue;
        fusionFailureStage = `apply_back:${applyEntry.exec_id}`;
        const accuracyKey = fusionStackKey();
        const step = await safeAgent(
          roleAgent('fusion_integrator', 'apply_one',
            'Apply exactly TARGET_EXEC_ID and its declared degrade ladder; do not loop unrelated execution-list entries. ' +
            'Read FUSION_TOPK_JSON and FUSION_UNITSIDE_JSON, and act only when the selected tier-A/B row has ' +
            'unit_side_status pass/equivalent_pass/subsumed_pass. Start from CURRENT_OVERLAY/FLAGS/ENV/THROUGHPUT, ' +
            'which already include earlier terminal wins. For the selected ladder, author a reversible lazy-load overlay, ' +
            'prove ENGAGED on every TP rank, then gate it ONCE with COMBINED_AB_SCRIPT (one base server and one candidate ' +
            'server, each AB_REPEATS timed rounds plus gsm8k on the same server); its decision covers throughput and ' +
            'accuracy, so do not run a separate gsm8k or re-run the A/B, and descend its ' +
            'declared ladder only when the wider rung fails. Preserve PRIOR_APPLY_RESULT dispositions verbatim and merge ' +
            'only this call\'s terminal result into it. Write the full aggregate apply_result.json, run ' +
            'fusion_applyback_harness.py with --allow-partial-coverage (later calls still have legitimate unprocessed rows), ' +
            'and return the full aggregate FUSION_APPLY_SCHEMA. Curate knowledge only for a fusion newly accepted by this call.', {
              EVAL_DIR, MODEL_PATH, SERVING_GPU, TP: SERVING_TP, WORKLOAD,
              TARGET_EXEC_ID: applyEntry.exec_id, TARGET_EXECUTION: applyEntry,
              PRIOR_APPLY_RESULT: applyState,
              FUSION_TOPK_JSON: FUSION_INPUTS.FUSION_TOPK_JSON,
              FUSION_CANDIDATES_JSON: FUSION_INPUTS.FUSION_CANDIDATES_JSON,
              FUSION_UNITSIDE_JSON: FUSION_INPUTS.FUSION_UNITSIDE_JSON,
              CURRENT_OVERLAY: curOverlay, CURRENT_FLAGS: curFlags, CURRENT_ENV: curEnv,
              CURRENT_THROUGHPUT: curTput, BASELINE_THROUGHPUT: BASELINE_TPUT,
              NOISE_BAND_PCT: NOISE_BAND, FUSION_BUDGET,
              FUSION_OVERLAYS_DIR: `${EVAL_DIR}/fusion/fusion_overlays`,
              COMBINED_AB_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_combined_ab.sh`, AB_REPEATS: FUSION_COMBINED_AB_REPEATS,
              AB_REPEAT_MODE: FUSION_AB_REPEAT_MODE,
              GSM8K_EVAL_SCRIPT: `${WORKFLOW_DIR}/scripts/fusion_gsm8k_eval.py`,
              GSM8K_CONCURRENCY: fusionGsm8kConcurrency(curFlags), GSM8K_THINKING: FUSION_GSM8K_THINKING,
              ACCURACY_STACK_KEY: accuracyKey,
              ACCURACY_REFERENCE: (fusionAccuracyRef && fusionAccuracyRef.stack_key === accuracyKey)
                ? fusionAccuracyRef : null,
              AUTHORED_OVERLAY: (authoredByExec.get(String(applyEntry.exec_id)) || null),
              ...ACCURACY_INPUTS, ...FUSION_RUNTIME_INPUTS, SKILL_DIR: WORKFLOW_DIR,
            }),
          { phase: 'KernelFusion', label: `fusion_integrator:apply_one:${applyEntry.exec_id}`,
            schema: FUSION_APPLY_SCHEMA, timeoutMs: FUSION_APPLY_TIMEOUT_MS }, 1);
        if (!step) {
          fusionApplyFailedExec = String(applyEntry.exec_id);
          fusionApplyUnprocessed = fusionApplyEntries.slice(applyIndex).map((e) => e.exec_id).filter(Boolean);
          log(`KernelFusion apply-back ${applyEntry.exec_id} failed or returned no terminal result; ` +
              `stopping the remaining ${Math.max(0, fusionApplyUnprocessed.length - 1)} apply-back call(s) ` +
              'and handing the pre-Fusion stack to the e2e workflow with earlier terminal wins preserved.');
          break;
        }
        commitStep(step, accuracyKey, applyEntry.exec_id);
      }
    }
  // KernelFusion is an optional optimization track. If apply-back fails to return a
  // terminal result, preserve the exact pre-Fusion runtime state and hand it to the e2e workflow. Record the failure explicitly so reports do not confuse "Fusion
  // failed" with "Fusion ran and found no win".
  if (!fapply) {
    log('KernelFusion apply-back failed or returned no terminal result; preserving the ' +
        'pre-Fusion overlay/flags/env/throughput and handing the pre-Fusion stack to the e2e workflow.');
    fusionDisposition = {
      applied: [], blocked: [], deferred: [], dispositioned: 0,
      deferred_author_count: 0,
      applyback_report_md: '', applyback_gate_json: '', learned_cards: [],
      failed_stage: combinedFailure ? 'apply_back:combined_ab'
        : (fusionApplyFailedExec ? `apply_back:${fusionApplyFailedExec}` : 'apply_back'),
      unprocessed_exec_ids: fusionApplyUnprocessed,
      notes: combinedFailure
        ? `KernelFusion failed: ${combinedFailure}. No fusion was applied; ` +
          'the pre-Fusion runtime state was preserved for the e2e workflow.'
        : 'KernelFusion apply-back failed before producing a terminal result; ' +
          'the pre-Fusion runtime state was preserved for the e2e workflow.',
    };
  }
  const acc = (fapply && Array.isArray(fapply.accepted_fusions)) ? fapply.accepted_fusions : [];
  // Surface the coverage verdict in the run log next to the win count. Two accepted
  // fusions out of a twelve-row board is a real result AND an incomplete one; the log
  // should not report only the half that looks like success.
  if (fapply) {
    const dispositioned = acc.length
      + (Array.isArray(fapply.rejected) ? fapply.rejected.length : 0)
      + (Array.isArray(fapply.deferred) ? fapply.deferred.length : 0);
    log(`Fusion apply-back coverage: ${dispositioned} execution-list row(s) given an explicit ` +
        `disposition (applied ${acc.length} / blocked ${(fapply.rejected || []).length} / ` +
        `deferred ${(fapply.deferred || []).length}); gate report: ` +
        `${fapply.applyback_report_md || '(not run — fusion_applyback_harness.py was skipped)'}`);
    // Lift the dispositions out of this block so PHASE=report can attribute the fusion track.
    // Without this, only `accepted_fusions` survives and a fully-blocked fusion run is
    // indistinguishable in final_report.md from one that never ran.
    fusionDisposition = {
      applied: acc.slice(),
      blocked: Array.isArray(fapply.rejected) ? fapply.rejected.slice() : [],
      deferred: Array.isArray(fapply.deferred) ? fapply.deferred.slice() : [],
      dispositioned,
      deferred_author_count: fapply.deferred_author_count || 0,
      applyback_report_md: fapply.applyback_report_md || '',
      applyback_gate_json: fapply.applyback_gate_json || '',
      learned_cards: Array.isArray(fapply.learned_cards) ? fapply.learned_cards.slice() : [],
      ...(fusionApplyFailedExec ? {
        failed_stage: `apply_back:${fusionApplyFailedExec}`,
        unprocessed_exec_ids: fusionApplyUnprocessed,
      } : {}),
      notes: fusionApplyFailedExec
        ? `${fapply.notes || ''} Apply-back stopped at ${fusionApplyFailedExec}; earlier terminal wins were preserved.`.trim()
        : (fapply.notes || ''),
    };
    // The reproducibility half. An apply-back that banks wins but writes no card leaves the
    // next run to rediscover them, and rediscovery is not deterministic — that is exactly how
    // the same trace produced a different fusion set run to run.
    const cards = Array.isArray(fapply.learned_cards) ? fapply.learned_cards : [];
    log(cards.length
      ? `Fusion knowledge: ${cards.length} learned card(s) curated (${cards.map(c => `${c.card}:${c.action}`).join(', ')}).`
      : 'Fusion knowledge: NO learned cards written — the next run will rediscover these fusions from scratch. ' +
        'Expected only if nothing cleared the ★★ bar this round.');
  }
  if (acc.length) {
    log(`KernelFusion: accepted ${acc.length} fusion(s); e2e now ${curTput} tok/s.`);
  } else {
    log(`KernelFusion: no fusion accepted (${fapply ? (fapply.notes || 'none passed the gate') : 'agent null/degraded'}).`);
  }
  } else {
    log('KernelFusion discovery/validation produced no apply-back-ready Top-K; handing the pre-Fusion stack to the e2e workflow.');
    if (FUSION_REQUIRED) {
      const requiredFailureStage = (!semantics || semantics.status !== 'pass')
        ? 'semantic'
        : (!FUSION_INPUTS.FUSION_TOPK_JSON ? 'discovery' : 'unit_validation');
      const requiredFailure =
        'KernelFusion was explicitly required but produced no apply-back-ready Top-K. ' +
        `Expected deterministic capture manifest: ${EVAL_DIR}/fusion_capture/profile_trace_manifest.json`;
      log(`ERROR: ${requiredFailure} Recording failure and handing the pre-Fusion stack to the e2e workflow.`);
      fusionDisposition = {
        applied: [], blocked: [], deferred: [],
        failed_stage: requiredFailureStage,
        notes: requiredFailure,
      };
    }
  }
  // Checkpoint only — not a new runtime baseline. Later stages keep using curTput;
  // Finalize/Director still score against BASELINE_TPUT.
  {
    fusionFailureStage = 'disposition';
    const appliedN = (fusionDisposition && Array.isArray(fusionDisposition.applied))
      ? fusionDisposition.applied.length : 0;
    let fusionStatus = 'no_win';
    if (fusionApplyFailedExec) fusionStatus = appliedN > 0 ? 'applyback_partial_failure' : 'applyback_failed';
    else if (appliedN > 0) fusionStatus = 'applied';
    else if (!FUSION_INPUTS.FUSION_TOPK_JSON) fusionStatus = 'discovery_failed';
    else if (!FUSION_INPUTS.FUSION_UNITSIDE_JSON) fusionStatus = 'validation_failed';
    else if (combinedFailure) fusionStatus = 'combined_ab_failed';
    else if (!fapply) fusionStatus = 'applyback_failed';
    const fusionDeltaPct = fusionEntryTput > 0
      ? Number((((curTput - fusionEntryTput) / fusionEntryTput) * 100).toFixed(3)) : 0;
    fusionDisposition = Object.assign({
      applied: [], blocked: [], deferred: [],
    }, fusionDisposition || {}, {
      status: fusionStatus,
      entry_throughput_tok_s: fusionEntryTput,
      exit_throughput_tok_s: curTput,
      delta_pct: fusionDeltaPct,
    });
  }
  } catch (e) {
    const message = (e && e.message) ? e.message : String(e);
    const recoveredApply = fusionRecoveryState.applyResult;
    const recoveredApplied = recoveredApply && Array.isArray(recoveredApply.accepted_fusions)
      ? recoveredApply.accepted_fusions.slice() : [];
    const recoveredBlocked = recoveredApply && Array.isArray(recoveredApply.rejected)
      ? recoveredApply.rejected.slice() : [];
    const recoveredDeferred = recoveredApply && Array.isArray(recoveredApply.deferred)
      ? recoveredApply.deferred.slice() : [];
    curOverlay = fusionRecoveryState.overlay;
    curFlags = fusionRecoveryState.flags;
    curEnv = fusionRecoveryState.env;
    curTput = fusionRecoveryState.throughput;
    acceptedFusions.length = fusionRecoveryState.acceptedFusionCount;
    FUSION_INPUTS = { ...fusionRecoveryState.inputs };
    fusionDisposition = {
      status: 'unexpected_exception',
      failed_stage: fusionFailureStage,
      applied: recoveredApplied, blocked: recoveredBlocked, deferred: recoveredDeferred,
      dispositioned: recoveredApplied.length + recoveredBlocked.length + recoveredDeferred.length,
      entry_throughput_tok_s: fusionEntryTput,
      exit_throughput_tok_s: fusionRecoveryState.throughput,
      delta_pct: fusionEntryTput > 0
        ? Number((((fusionRecoveryState.throughput - fusionEntryTput) / fusionEntryTput) * 100).toFixed(3)) : 0,
      notes: `KernelFusion ${fusionFailureStage} raised an unexpected exception: ${message}. ` +
        `${recoveredApplied.length} earlier terminal win(s) were preserved.`,
    };
    log(`ERROR: KernelFusion ${fusionFailureStage} raised an unexpected exception: ${message}. ` +
        'Restored the latest terminal Fusion checkpoint and handing the pre-Fusion stack to the e2e workflow.');
  }
  // Every exit path above rebuilds fusionDisposition, so the coverage verdict is attached here,
  // once. coverage_gaps is what the report reads: a phase listed there had no fusion candidates.
  if (fusionObservedWorkload) {
    // The unit-side improvement is the observed-workload weighted speedup; the e2e A/B
    // above stays the acceptance gate.
    fusionDisposition = Object.assign({}, fusionDisposition || {}, {
      agentx_sampling: { observed_workload_json: fusionObservedWorkload,
        unit_speedup_basis: 'observed_workload_weighted' },
    });
  }
  if (fusionCoverage) {
    fusionDisposition = Object.assign({}, fusionDisposition || {}, {
      phase_coverage: fusionCoverage,
      coverage_gaps: fusionCoverage.decode_covered === false ? ['decode']
        : (fusionCoverage.decode_covered == null ? ['decode:unknown'] : []),
    });
  }
}

// ===========================================================================
// PHASE: Report. The workflow has no filesystem, so one agent persists the return value as
// kf_return.json (interface/run_kf_e2e.py reads it) and indexes the phase reports.
// ===========================================================================
phase('Report');
const kfApplied = (fusionDisposition && Array.isArray(fusionDisposition.applied)) ? fusionDisposition.applied : [];
const kfStatus = (fusionDisposition && fusionDisposition.status)
  || (!FUSION_DISCOVERY_ON ? 'disabled' : (BASELINE_TPUT > 0 ? 'no_win' : 'baseline_failed'));
// The fused stack the e2e workflow starts from: only a stack that passed the combined A/B moved
// curOverlay/curFlags/curEnv, so an unaccepted run hands back exactly what it was given.
const kfReturn = {
  schema: 'geak.kernel_fusion_stage/1',
  eval_dir: EVAL_DIR,
  status: kfStatus,
  backend: BACKEND, tp: SERVING_TP, serving_gpu: SERVING_GPU,
  workload_kind: IS_AGENTX ? 'agentx_trace_replay' : 'fixed_shape',
  baseline_throughput_tok_s: BASELINE_TPUT,
  baseline_metric_basis: (baseline && baseline.metric_basis) || '',
  baseline_bench_summary_json: (baseline && baseline.bench_summary_json) || '',
  final_throughput_tok_s: curTput,
  delta_pct: (fusionDisposition && fusionDisposition.delta_pct != null) ? fusionDisposition.delta_pct
    : (BASELINE_TPUT > 0 ? Number((((curTput - BASELINE_TPUT) / BASELINE_TPUT) * 100).toFixed(3)) : 0),
  applied_count: kfApplied.length,
  initial_overlay_pythonpath: INIT_OVERLAY,
  final_overlay: kfApplied.length ? (curOverlay || '') : INIT_OVERLAY,
  accepted_flags: kfApplied.length ? curFlags : INIT_FLAGS,
  accepted_env: kfApplied.length ? curEnv : INIT_ENV,
  analysis_phases: FUSION_ANALYSIS_PHASES,
  analysis_points: (fusionCapture && fusionCapture.analysis_points) || {},
  trace_manifest_json: (fusionCapture && fusionCapture.trace_manifest_json) || '',
  observed_workload_json: fusionObservedWorkload || '',
  sampling_summary_json: (fusionCapture && fusionCapture.sampling_summary_json) || '',
  semantic_table_json: (semantics && semantics.semantic_table_json) || '',
  fusion_topk_json: FUSION_INPUTS.FUSION_TOPK_JSON || '',
  fusion_unitside_json: FUSION_INPUTS.FUSION_UNITSIDE_JSON || '',
  fusion_disposition: fusionDisposition || {},
  ...(FUSION_L3_ON ? { l3: fusionL3 || { stages: [], notes: 'the screen listed no L3 stage' } } : {}),
  elapsed_min: Math.round(ELAPSED_MS / 60000),
};
const reported = await safeAgent(
  `You are the KernelFusion reporter. Do exactly this, with the Bash/Write tools, and nothing else:
1. Write the JSON object below, verbatim and pretty-printed, to ${EVAL_DIR}/kf_return.json, and
   the same content to ${EVAL_DIR}/workflow_return.json (the runner's completion marker); create
   ${EVAL_DIR} if it does not exist. Write workflow_return.json LAST, after steps 2 and 3.
2. Write ${EVAL_DIR}/KF_REPORT.md: a short human report of the stage from that JSON -- status, baseline
   and final throughput with the delta, the analysis trace (analysis_points: the top-1 GPU-time point)
   and the sampling summary it came from, the Top-K / unit-side / apply-back outcome with every applied,
   blocked and deferred row and its reason, ${FUSION_L3_ON ? 'the L3 stage fusions (l3: each stage, its GEAK mode=mega outcome and overlay), ' : ''}and the coverage gaps. Link the phase reports that exist
   under ${EVAL_DIR} (01_SEMANTIC.md, 02*/03*/04_FUSION_UNITSIDE.md, 05*).
3. Run: python3 ${WORKFLOW_DIR}/scripts/report_index.py --eval-dir ${EVAL_DIR}
   (if its CLI differs, read --help and index ${EVAL_DIR}; a failure here is a note, not a stop).
Return {"kf_return_json": "<path>", "report_md": "<path>", "index_md": "<path or empty>", "notes": "..."}.

kf_return:
${JSON.stringify(kfReturn, null, 1)}`,
  { phase: 'Report', label: 'kernel-fusion:report', schema: REPORT_SCHEMA }, 2);
log(`KernelFusion stage done: status=${kfStatus}, applied ${kfApplied.length}, ` +
    `${BASELINE_TPUT} -> ${curTput}; return ${(reported && reported.kf_return_json) || `${EVAL_DIR}/kf_return.json (writer failed)`}.`);
return kfReturn;
