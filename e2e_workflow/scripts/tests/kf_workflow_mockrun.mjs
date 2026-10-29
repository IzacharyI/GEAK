// Mock Workflow runtime for kernel_fusion/kf_workflow.js (tests/test_fusion_capture_manifest_contract.py).
// Stubs every agent by its label, records phases/prompts, prints one JSON summary line.
// usage: node kf_workflow_mockrun.mjs <kf_workflow.js> <accept|reject|baseline_fail|tight|fixed_shape|vllm|l3|...>
// l3* scenarios turn on StageFusion (fusion_l3_mega) and stub workflow() as kernel_mega.js.
import fs from 'fs';
const [,, script, scenario] = process.argv;
let src = fs.readFileSync(script, 'utf8').replace('export const meta', 'const meta');
const calls = [];
const workflowCalls = [];
const L3 = scenario.startsWith('l3');
const L3_STAGES = [
  { stage_id: 'l3_01', level: 'L3', phase: 'decode', request_share_pct: 12.5, subsumes: ['e2'], ep_kernels: ['megamoe_stage1'] },
  { stage_id: 'l3_02', level: 'L3', phase: 'prefill', request_share_pct: 1.0, subsumes: [] },
];
let aggCalls = 0;
const phases = [];
const logs = [];
const respond = (prompt, opts) => {
  const l = opts.label || '';
  calls.push({ label: l, prompt });
  if (scenario === 'baseline_fail' && l.startsWith('fusion-trace-collector:baseline')) return { status: 'failed', baseline_throughput_tok_s: 0, server_log: '' };
  if (l === 'fusion-trace-collector:reuse_baseline') return { status: 'pass', baseline_throughput_tok_s: 799.42, server_log: '/old/baseline/server.log', reused_from: '/old' };
  if (l === 'fusion-trace-collector:reuse_capture') return { trace_manifest_json: '/old/fusion_capture/profile_trace_manifest.json', observed_workload_json: '/old/fusion_capture/sampling/fusion_observed_workload.json', analysis_points: { decode: 'de_5' }, phase_evidence_status: 'measured_annotation', source: 'reused-sampling' };
  if (l.startsWith('fusion-trace-collector:baseline')) return { status: 'pass', baseline_throughput_tok_s: 1000, server_log: '/e/baseline/server.log', bench_log: '/e/baseline/bench.log', metric_basis: 'total' };
  if (l.startsWith('fusion-trace-collector:capture')) return { trace_manifest_json: '/e/fusion_capture/profile_trace_manifest.json', observed_workload_json: '/e/fusion_capture/sampling/fusion_observed_workload.json', analysis_points: { decode: 'de_16' }, phase_evidence_status: 'measured_annotation', sampling_summary_json: '/e/s.json' };
  if (l === 'semantics-mapper:fusion') return { status: 'partial', semantic_table_json: '/e/t.json', phases_in_tables: [], missing_required_phases: ['decode'], structural_patterns_json: '/e/p.json', shape_capture_plan_json: '/e/plan.json' };
  if (l === 'semantics-mapper:shape-completion') return { status: 'pass', semantic_table_json: '/e/t2.json', phases_in_tables: ['decode'], missing_required_phases: [] };
  if (l === 'fusion-reuse-verdicts') return { ok: true, reused: { c1: { side: 'self', speedup: 1.43, from: '/old/verdict/c1__self.json' } } };
  if (l === 'fusion-best-of-n') return { ok: true, reused: scenario === 'best_of_n' ? { c1: { side: 'self', speedup: 1.4, from: '/e/fusion/attempts/a3/verdict/c1__self.json' } } : {} };
  if (l.startsWith('fusion_serving_author:')) { const cid = l.split(':').pop(); return { candidate_id: cid, status: 'authored', loader_dir: `/e/su/${cid}/loader`, banner_tag: `su_${cid}`, kernel_source: `/e/su/${cid}/k.py` }; }
  if (l === 'fusion_serving_measure') { const v = {}; for (const m of prompt.matchAll(/--cand ([^:\s]+)::/g)) v[m[1]] = { verdict_path: `/e/verdict/${m[1]}__serving.json`, parity: 'pass', speedup: 1.01, engaged: true }; return { ok: true, base_ok: true, verdicts: v }; }
  if (l === 'fusion-screen' && scenario === 'ep') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], alt_group: 'g1', kernel_names: ['quant', 'megamoe_stage1_compact', 'ep_combine_intranode_0'], alt_ladder: [{ rung: 0 }, { rung: 1, kernel_names: ['quant', 'gemm'] }, { rung: 2, kernel_names: ['topk', 'megamoe_prepare'] }] }, { exec_id: 'e2', candidate_ids: ['c2'] }], candidate_pool: [] };
  if (l === 'fusion-screen' && scenario === 'apply_pick') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1', 'c2'] }], candidate_pool: [] };
  if (l.startsWith('fusion_unit_validator:aggregate') && scenario === 'apply_pick') return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['c1'], unit_status_by_candidate: { c1: 'pass', c2: 'fail' } };
  if (scenario === 'apply_pick' && l.startsWith('fusion_unit_validator:')) { const cid = l.split(':').pop(); return { candidate_id: cid, verdict_path: `/e/verdict/${cid}__self.json`, parity: 'pass', isolated_speedup: cid === 'c1' ? 1.42 : 1.62 }; }
  if (l === 'fusion-screen' && scenario === 'apply_gain') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], alt_group: 'g1', expected_e2e_gain_pct: 3, alt_ladder: [{ rung: 0 }, { rung: 1, kernel_names: ['a', 'b'], expected_e2e_gain_pct: 0.5 }, { rung: 2, kernel_names: ['a', 'b', 'c', 'd', 'e', 'f'], expected_e2e_gain_pct: 2 }] }], candidate_pool: [] };
  if (l.startsWith('fusion_unit_validator:aggregate') && scenario === 'apply_gain') return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['e1.r1', 'e1.r2'], unit_status_by_candidate: { c1: 'fail', 'e1.r1': 'pass', 'e1.r2': 'pass' } };
  if (scenario === 'apply_gain' && l.startsWith('fusion_unit_validator:')) { const cid = l.split(':').pop(); return cid === 'c1' ? { candidate_id: cid, verdict_path: '/e/v.json', parity: 'fail' } : { candidate_id: cid, verdict_path: '/e/v.json', parity: 'pass', isolated_speedup: 1.3 }; }
  if (l === 'fusion_integrator:preflight') return { server_ok: true, tp: 8, engaged: { x: scenario === 'repair' ? 0 : 8 }, missing: scenario === 'repair' ? ['x'] : [], server_log: '/e/fusion/combined_ab/preflight/server.log' };
  if (l === 'fusion-ladders') return { ok: true, ladders: { e1: [{ rung: 0 }, { rung: 1, kernel_names: ['a', 'b'] }, { rung: 2 }] } };
  if (l === 'fusion-screen' && scenario === 'descend_noladder') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], alt_group: 'g1' }], candidate_pool: [] };
  if (l === 'fusion-screen' && scenario === 'descend') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], alt_group: 'g1', alt_ladder: [{ rung: 0 }, { rung: 1, kernel_names: ['a', 'b'] }, { rung: 2 }] }, { exec_id: 'e2', candidate_ids: ['c2'] }], candidate_pool: [] };
  if (l === 'fusion-screen' && scenario === 'l3_only') return { status: 'no_candidate', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [], candidate_pool: [], l3_stages: L3_STAGES };
  if (l === 'fusion-screen' && L3) return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], expected_gain_pct: 2 }, { exec_id: 'e2', candidate_ids: ['c2'] }], candidate_pool: [], l3_stages: L3_STAGES };
  if (l.startsWith('fusion_unit_validator:aggregate') && L3) return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['c1', 'c2'], unit_status_by_candidate: { c1: 'pass', c2: 'pass' } };
  if (l.startsWith('fusion_stage_l3:prepare:')) return { status: 'ready', stage_id: l.split(':').pop(), kernel_path: '/e/fusion/l3/l3_01/baseline_tree', benchmark_harness: '/e/fusion/l3/l3_01/baseline_tree/moe_ut.py', ep_size: 8, package: 'aiter', call_site: 'aiter.megamoe:mega_moe', mega_args: { target_guards: ['64_uniform'], task: 'fuse the stage', budget: 99 } };
  if (l.startsWith('fusion_stage_l3:package:')) { const id = l.split(':').pop(); return { exec_id: id, status: 'authored', overlay_dir: `/e/ovl/${id}`, banner_tag: id, parity: 'pass relL2=0.03' }; }
  if (l === 'fusion_integrator:combined_ab' && L3) return { accepted_fusions: [{ exec_id: 'e1' }, { exec_id: 'l3_01', level: 'L3' }], final_overlay: '/e/stack', e2e_throughput_tok_s: 1100, combined_decision: 'accept', rejected: [{ exec_id: 'e2', reason: 'subsumed by L3 l3_01' }], deferred: [], learned_cards: [] };
  if (l === 'fusion-screen') return { status: 'pass', fusion_topk_json: '/e/topk.json', fusion_candidates_json: '/e/c.json', execution_list: [{ exec_id: 'e1', candidate_ids: ['c1'], expected_gain_pct: 2 }, { exec_id: 'e2', candidate_ids: ['c2'] }], candidate_pool: [], k_effective: 8, noise_band_pct: 1 };
  if (l.startsWith('fusion_unit_validator:aggregate') && scenario === 'agg_drop') {
    aggCalls += 1;
    return aggCalls === 1 ? { status: 'pass', fusion_unitside_json: '/e/u.json', unit_status_by_candidate: { c1: 'waived', c2: 'waived' } }
      : { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['c1'], unit_status_by_candidate: { c1: 'pass', c2: 'waived' } };
  }
  if (l.startsWith('fusion_unit_validator:aggregate') && (scenario === 'descend' || scenario === 'descend_noladder')) return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['e1.r1'], unit_status_by_candidate: { c1: 'waived', 'e1.r1': 'pass', 'e1.r2': 'fail', c2: 'waived' } };
  if ((scenario === 'descend' || scenario === 'descend_noladder') && l.startsWith('fusion_unit_validator:')) { const cid = l.split(':').pop(); return cid === 'e1.r1' ? { candidate_id: cid, verdict_path: '/e/v.json', parity: 'pass', isolated_speedup: 1.3 } : { candidate_id: cid, verdict_path: '/e/v.json', parity: 'unmeasured' }; }
  if (l.startsWith('fusion_unit_validator:aggregate')) return { status: 'pass', fusion_unitside_json: '/e/u.json', applyback_eligible_ids: ['c1'], unit_status_by_candidate: { c1: 'pass', c2: 'unit_fail' } };
  if (l.startsWith('fusion_unit_validator:')) return { candidate_id: l.split(':').pop(), verdict_path: '/e/v.json', parity: 'pass', isolated_speedup: 1.1 };
  if (l.startsWith('fusion-lease:')) return { ok: true, alive_count: 0 };
  if (l.startsWith('fusion_integrator:author_one')) return { exec_id: l.split(':')[2], status: 'authored', overlay_dir: '/e/ovl/' + l.split(':')[2], banner_tag: scenario === 'repair' ? 'overlay-x' : 'x' };
  if (l === 'fusion_integrator:combined_ab') return scenario === 'reject'
    ? { accepted_fusions: [], combined_decision: 'reject', combined_ab: { reason: 'no gain' }, rejected: [], deferred: [] }
    : { accepted_fusions: [{ exec_id: 'e1' }], final_overlay: '/e/stack', e2e_throughput_tok_s: 1050, combined_decision: 'accept', rejected: [], deferred: [{ exec_id: 'e2', reason: 'unit_fail' }], accepted_flags: '--x', accepted_env: 'A=1', learned_cards: [] };
  if (l === 'kernel-fusion:report') return { kf_return_json: '/e/kf_return.json' };
  throw new Error('unexpected agent label ' + l);
};
const agent = async (prompt, opts) => respond(prompt, opts || {});
const workflow = async (ref, wfArgs) => {
  workflowCalls.push({ ref, args: wfArgs });
  if (!L3) throw new Error('unexpected workflow() call');
  if (scenario === 'l3_throw') throw new Error('Setup failed: director did not return an eval_dir');
  if (scenario === 'l3_nodeliver') return { mode: 'mega', mega_deliverable: false, mega_selected_candidate: '', mega_candidate_registry: [], final_speedup: null, mega_note: 'mega: no fully verified finalist' };
  return { mode: 'mega', mega_deliverable: true, mega_selected_candidate: 'm3', mega_selected_source: 'workflow',
    mega_candidate_registry: [{ id: 'm1', tree: '/m/t1' }, { id: 'm3', tree: '/m/t3', final_tree: '/m/f3', patch: '/m/p3.diff', activation: { switch_name: 'X', switch_value: '1' }, absolute_score: 1.15, per_case: [] }],
    final_speedup: 1.15, validation_status: 'accepted', eval_dir: '/m/eval', report_path: '/m/eval/tech_lead_report.md', final_patch: '/m/eval/final_patch.diff', mega_note: 'mega: selected m3' };
};
const parallel = async (thunks) => Promise.all(thunks.map((t) => t()));
const phase = (p) => phases.push(p);
const log = (m) => logs.push(m);
const args = {
  workflow_dir: '/w/e2e_workflow', model_path: '/data/models/DeepSeek-V4-Pro-0813',
  backend: scenario === 'vllm' ? 'vllm' : 'sglang', tp: 8,
  gpu_ids: '0,1,2,3,4,5,6,7', eval_dir: '/e', exp_root: '/x', time_budget_s: scenario === 'tight' ? 3600 : 43200,
  workload_spec: scenario === 'fixed_shape' ? {} : { kind: 'agentx_trace_replay', concurrency: 128 },
  isl: 1024, osl: 1024, conc: 64,
  ...(scenario === 'reuse' ? { fusion_reuse_eval_dir: '/old/' } : {}),
  ...(scenario === 'reuse_unit' ? { fusion_reuse_unitside_dirs: ['/old1', '/old2/'] } : {}),
  initial_extra_server_args: '--tp 8', initial_extra_env: 'E=1', initial_overlay_pythonpath: '',
  ...(L3 ? { fusion_l3_mega: 'true', fusion_l3_mega_args: { use_expert_skills: 'true' } } : {}),
  ...(scenario === 'l3_reuse' ? { fusion_l3_reuse_dir: '/old/' } : {}),
};
const fn = new Function('args', 'agent', 'parallel', 'phase', 'log', 'workflow', 'return (async () => {\n' + src + '\n})();');
const ret = await fn(args, agent, parallel, phase, log, workflow);
const bad = calls.filter((c) => /undefined|\[object Object\]|NaN/.test(c.prompt.replace(/JSON\.stringify|undefined_/g, '')));
const prompts = {};
for (const c of calls) prompts[c.label] = c.prompt;
console.log(JSON.stringify({ phases, labels: calls.map((c) => c.label), ret, bad: bad.map((b) => b.label), prompts, logs, workflowCalls }));
