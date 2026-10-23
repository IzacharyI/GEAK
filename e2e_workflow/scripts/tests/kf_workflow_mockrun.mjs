// Mock Workflow runtime for kernel_fusion/kf_workflow.js (tests/test_fusion_capture_manifest_contract.py).
// Stubs every agent by its label, records phases/prompts, prints one JSON summary line.
// usage: node kf_workflow_mockrun.mjs <kf_workflow.js> <accept|reject|baseline_fail|tight|fixed_shape|vllm>
import fs from 'fs';
const [,, script, scenario] = process.argv;
let src = fs.readFileSync(script, 'utf8').replace('export const meta', 'const meta');
const calls = [];
const phases = [];
const logs = [];
const respond = (prompt, opts) => {
  const l = opts.label || '';
  calls.push({ label: l, prompt });
  if (scenario === 'baseline_fail' && l.startsWith('fusion-trace-collector:baseline')) return { status: 'failed', baseline_throughput_tok_s: 0, server_log: '' };
  if (l.startsWith('fusion-trace-collector:baseline')) return { status: 'pass', baseline_throughput_tok_s: 1000, server_log: '/e/baseline/server.log', bench_log: '/e/baseline/bench.log', metric_basis: 'total' };
  if (l.startsWith('fusion-trace-collector:capture')) return { trace_manifest_json: '/e/fusion_capture/profile_trace_manifest.json', observed_workload_json: '/e/fusion_capture/sampling/fusion_observed_workload.json', analysis_points: { decode: 'de_16' }, phase_evidence_status: 'measured_annotation', sampling_summary_json: '/e/s.json' };
  if (l === 'semantics-mapper:fusion') return { status: 'partial', semantic_table_json: '/e/t.json', phases_in_tables: [], missing_required_phases: ['decode'], structural_patterns_json: '/e/p.json', shape_capture_plan_json: '/e/plan.json' };
  if (l === 'semantics-mapper:shape-completion') return { status: 'pass', semantic_table_json: '/e/t2.json', phases_in_tables: ['decode'], missing_required_phases: [] };
  if (l === 'fusion-screen') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], expected_gain_pct: 2 }, { exec_id: 'e2', candidate_ids: ['c2'] }], candidate_pool: [], k_effective: 8, noise_band_pct: 1 };
  if (l.startsWith('fusion_unit_validator:aggregate')) return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['c1'], unit_status_by_candidate: { c1: 'pass', c2: 'unit_fail' } };
  if (l.startsWith('fusion_unit_validator:')) return { candidate_id: l.split(':').pop(), verdict_path: '/e/v.json', parity: 'pass', isolated_speedup: 1.1 };
  if (l.startsWith('fusion-lease:')) return { ok: true, alive_count: 0 };
  if (l.startsWith('fusion_integrator:author_one')) return { exec_id: l.split(':').pop(), status: 'authored', overlay_dir: '/e/ovl/' + l.split(':').pop(), banner_tag: 'x' };
  if (l === 'fusion_integrator:combined_ab') return scenario === 'reject'
    ? { accepted_fusions: [], combined_decision: 'reject', combined_ab: { reason: 'no gain' }, rejected: [], deferred: [] }
    : { accepted_fusions: [{ exec_id: 'e1' }], final_overlay: '/e/stack', e2e_throughput_tok_s: 1050, combined_decision: 'accept', rejected: [], deferred: [{ exec_id: 'e2', reason: 'unit_fail' }], accepted_flags: '--x', accepted_env: 'A=1', learned_cards: [] };
  if (l === 'kernel-fusion:report') return { kf_return_json: '/e/kf_return.json' };
  throw new Error('unexpected agent label ' + l);
};
const agent = async (prompt, opts) => respond(prompt, opts || {});
const parallel = async (thunks) => Promise.all(thunks.map((t) => t()));
const phase = (p) => phases.push(p);
const log = (m) => logs.push(m);
const args = {
  workflow_dir: '/w/e2e_workflow', model_path: '/data/models/DeepSeek-V4-Pro-0813',
  backend: scenario === 'vllm' ? 'vllm' : 'sglang', tp: 8,
  gpu_ids: '0,1,2,3,4,5,6,7', eval_dir: '/e', exp_root: '/x', time_budget_s: scenario === 'tight' ? 3600 : 43200,
  workload_spec: scenario === 'fixed_shape' ? {} : { kind: 'agentx_trace_replay', concurrency: 128 },
  isl: 1024, osl: 1024, conc: 64,
  initial_extra_server_args: '--tp 8', initial_extra_env: 'E=1', initial_overlay_pythonpath: '',
};
const fn = new Function('args', 'agent', 'parallel', 'phase', 'log', 'return (async () => {\n' + src + '\n})();');
const ret = await fn(args, agent, parallel, phase, log);
const bad = calls.filter((c) => /undefined|\[object Object\]|NaN/.test(c.prompt.replace(/JSON\.stringify|undefined_/g, '')));
const prompts = {};
for (const c of calls) prompts[c.label] = c.prompt;
console.log(JSON.stringify({ phases, labels: calls.map((c) => c.label), ret, bad: bad.map((b) => b.label), prompts, logs }));
