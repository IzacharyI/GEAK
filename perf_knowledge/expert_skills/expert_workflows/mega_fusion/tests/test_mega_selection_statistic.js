#!/usr/bin/env node
'use strict';

// The final selection verdict checks that a finalist row's per-case speedups and headline score
// summarize its own paired readings. A task-supplied A/B tool (every operator other than MegaMoE)
// reports median(base) / median(cand); the verdict used to accept only the median of per-pair ratios
// within 0.002, so the Muse attention-block run (team_attnblk_20261106) rejected both finalists
// (F56, S56, S32 and the headline) although they had passed every measured gate. The fixture holds that
// run's real per_case table and raw pairs.

const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(path.resolve(__dirname, '..', 'workflow.js'), 'utf8');
const fx = JSON.parse(fs.readFileSync(
  path.resolve(__dirname, 'fixtures', 'mega_attnblk_finalist_pairs.json'), 'utf8'));

const region = (name) => {
  const m = src.match(new RegExp(`// <<REPLAY:${name}>>([\\s\\S]*?)// <</REPLAY:${name}>>`));
  if (!m) throw new Error(`missing ${name} replay region`);
  return m[1];
};

const api = new Function(`
  const BIMODAL_GUARDS = [];
  function guardContract() { return { complete: true, regression_pass: true }; }
  function promotionScore(v, metric, targets) {
    const row = (v.per_case || []).find((r) => r.name === targets[0]);
    return row ? Number(row.speedup) : 0;
  }
  ${region('mega_topology_contract')}
  ${region('mega_candidate_registry')}
  return { megaFinalSelectionVerdict, pairedGuardReadout };
`)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const WORLD = 4;
const SELECTED = '/eval/attnblk/mega_selected';
const guards = [...fx.target_guards, ...fx.regression_guards];
const finalist = {
  id: 'mega_hip', source: 'search', status: 'scored',
  tree: '/state/candidates/mega_hip/tree', head: '997b0d626cbda5c1e4fb951f5bef081f622111eb',
};

const geomean = (v) => Math.exp(v.reduce((a, x) => a + Math.log(x), 0) / v.length);
const medianRatio = (g) => api.pairedGuardReadout(fx.paired_readings, g).score;

const row = (perCase, score) => ({
  candidate_id: finalist.id, source: finalist.source, tree: finalist.tree, head: finalist.head,
  claim_complete: true, status: 'verified', score,
  correctness: 'pass', guards_pass: true,
  path_marker: '[op] path=MEGA1', path_marker_count: WORLD, world_size: WORLD,
  launches: 1, launches_base: 8, graph_safe: 'pass',
  artifact_distinct: 'yes', artifact_hash_base: 'base-co', artifact_hash_candidate: 'mega-co',
  accuracy_metric: 'relL2', accuracy_value: 0.036, liveness_replays: 256,
  replay_results: guards.map((guard) => ({
    guard, count: 256, status: 'pass', graph_safe: 'pass', arrival_jitter: false })),
  paired_readings: fx.paired_readings, null_arm_pct: 1.343, per_case: perCase,
});
const selection = (r) => ({
  claim_complete: true, attempt_id: 'select_mega_1793983235',
  evidence_manifest: '/eval/attnblk/select_mega_1793983235/selection_evidence.json',
  selected_candidate_id: finalist.id, selected_source: finalist.source,
  selected_tree: SELECTED, selected_head: '1ab0cb58315a37d92bbe81aa3f02a6e893627bb8',
  materialized_from_head: finalist.head, selected_score: r.score, candidates: [r],
});
const verdict = (r) => api.megaFinalSelectionVerdict(selection(r), [finalist], {
  selectedWorkspace: SELECTED, launchTarget: 1, worldSize: WORLD, allowPartialFusion: true,
  baselineLaunches: 8, targetGuards: fx.target_guards, regressionGuards: fx.regression_guards,
  requiredReplays: 256, requiredPairs: 5, accuracyMetric: 'relL2', accuracyThreshold: 0.06,
  requireArrivalJitter: false,
});

console.log('\n# the recorded attention-block finalist is internally consistent');
const recorded = verdict(row(fx.per_case, fx.score));
ok(recorded.pass, `ratio-of-medians per_case and headline pass (${recorded.reasons.join('; ') || 'no reasons'})`);
ok(Math.abs(fx.per_case.find((r) => r.name === 'S56').speedup - medianRatio('S56')) > 0.002,
  'the fixture does exercise the gap: S56 per_case differs from the paired-ratio median by > 0.002');

console.log('\n# the per-pair-ratio median remains accepted');
const canonicalCases = fx.per_case.map((r) => ({ ...r, speedup: medianRatio(r.name) }));
const canonical = verdict(row(canonicalCases, geomean(fx.target_guards.map(medianRatio))));
ok(canonical.pass, `median-of-ratios per_case and headline pass (${canonical.reasons.join('; ') || 'no reasons'})`);

console.log('\n# a number that summarizes neither statistic is still rejected');
const inflated = fx.per_case.map((r) => (r.name === 'S56' ? { ...r, speedup: 1.2 } : r));
const v1 = verdict(row(inflated, fx.score));
ok(!v1.pass && v1.reasons.includes('selected finalist: target guard S56'),
  'an inflated target per_case fails its guard');
const v2 = verdict(row(fx.per_case, fx.score * 1.02));
ok(!v2.pass && v2.reasons.includes('selected finalist: headline score does not match paired target readings'),
  'an inflated headline fails');
const lowered = fx.per_case.map((r) => (r.name === 'S32' ? { ...r, speedup: 1.30 } : r));
ok(verdict(row(lowered, fx.score)).reasons.includes('selected finalist: regression guard S32'),
  'a regression per_case that matches neither summary fails');

console.log('\n# path marker is counted per rank, not per marker line');
const lines = verdict({ ...row(fx.per_case, fx.score), path_marker_count: 2 * WORLD });
ok(lines.reasons.includes('selected finalist: candidate path marker'),
  'reporting 8 marker lines (4 ranks x 2 layers) as the count is rejected');
ok(/distinct ranks that printed it, never the number of marker lines/.test(
  fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'director.md'), 'utf8')),
  'director.md tells select_mega to count distinct ranks');

console.log(failures === 0
  ? '\nPASS: final selection accepts either summary of the finalist\'s own pairs and nothing else.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
