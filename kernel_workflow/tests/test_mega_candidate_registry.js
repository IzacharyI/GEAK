#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_workflow.js'), 'utf8');
const match = src.match(
  /\/\/ <<REPLAY:mega_candidate_registry>>([\s\S]*?)\/\/ <<\/REPLAY:mega_candidate_registry>>/,
);
if (!match) throw new Error('missing mega_candidate_registry replay region');

const api = new Function(`
  function guardContract(v, targets, regressions) {
    const rows = new Map((v.per_case || []).map((r) => [r.name, r]));
    const missing = regressions.filter((g) => !rows.has(g));
    return {
      complete: targets.every((g) => rows.has(g)) && missing.length === 0,
      regression_pass: missing.length === 0 &&
        regressions.every((g) => Number(rows.get(g).speedup) >= 1),
    };
  }
  function promotionScore(v, metric, targets, regressions, fallback) {
    if (!targets.length) return Number(fallback || 0);
    const row = (v.per_case || []).find((r) => r.name === targets[0]);
    return row ? Number(row.speedup) : 0;
  }
  ${match[1]}
  return {
    validMegaCandidateId, normalizeMegaCandidate, upsertMegaCandidate,
    megaCandidateHardPass, selectMegaCandidate, selectMegaSearchParent,
    megaCalibrationClaimPass, pairedGuardReadout, megaCandidateFromVerification,
  };
`)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const complete = (id, score) => ({
  id, source: 'search', status: 'scored', tree: `/state/${id}`, head: `${id}-head`,
  claim_complete: true, attempt_id: `${id}:1`, evidence_manifest: `${id}.json`,
  target_guard: 'target_case',
  absolute_score: score, per_case: [{ name: '8192_uniform', speedup: score }],
  paired_readings: Array.from({ length: 5 }, (_, i) => ({
    guard: 'target_case', base: 4.6 + i * 0.001,
    cand: (4.6 + i * 0.001) / score,
  })),
  null_arm_pct: 0.1, reps: 5,
  artifact_hash_base: 'base-hash', artifact_hash_candidate: `${id}-hash`,
  correctness_pass: true, activation_pass: true, head_pass: true, launch_pass: true,
  liveness_pass: true, graph_pass: true, guards_pass: true, artifact_distinct: true,
  measurement_pass: true,
});

console.log('\n# one source contract for skill-on and skill-off');
ok(api.normalizeMegaCandidate({ id: 'guided', source: 'search' }).source_valid,
  'ordinary search source is accepted');
ok(api.normalizeMegaCandidate({ id: 'integrated', source: 'integrated' }).source_valid,
  'integrated source is accepted');
ok(api.normalizeMegaCandidate({ id: 'legacy', source: 'validated_skill' }).status === 'rejected',
  'validated_skill is not a separate candidate source');
ok(api.normalizeMegaCandidate({ id: 'guided_fusion_v1', source: 'search' }).source_id_valid,
  'candidate ids are not reserved for a reproduction lane');
ok(!api.validMegaCandidateId('../escape') && !api.validMegaCandidateId('a/b'),
  'candidate ids cannot escape their state directory');
{
  const staged = api.normalizeMegaCandidate({
    id: 'staged', source: 'search', checkpoint_complete: true,
    structural_verified: true, runtime_verified: false, score_complete: false,
    structural_report: '/eval/structure.json',
  });
  ok(staged.checkpoint_complete && staged.structural_verified &&
     !staged.runtime_verified && !staged.score_complete,
  'checkpoint/structural/runtime/score states remain independent');
}

console.log('\n# WIP cannot overwrite verified evidence');
{
  let registry = [complete('guided', 1.0448)];
  registry = api.upsertMegaCandidate(registry, {
    id: 'guided', source: 'search', status: 'authoring', claim_complete: false,
    head: 'new-head', tree: '/state/guided', attempt_id: 'guided:2',
  });
  ok(registry[0].absolute_score === 1.0448 && registry[0].head === 'guided-head',
    'an incomplete turn preserves the last verified score/head');
  ok(registry[0].working_head === 'new-head',
    'new authoring progress is retained separately');
}
{
  const structural = {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'sealed-head', claim_complete: false,
    checkpoint_complete: true, structural_verified: true,
    structural_report: '/eval/structure.json',
    structural_skill_id: 'skill',
    structural_candidate_head: 'sealed-head',
    structural_candidate_tree_digest: 'tree-digest',
    structural_skill_bundle_sha256: 'bundle-digest',
    structural_planner_extension_sha256: 'planner-digest',
    structural_contract_revision: 'v1',
    structural_contract_sha256: 'contract-digest',
  };
  let registry = [structural];
  registry = api.upsertMegaCandidate(registry, {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'sealed-head', claim_complete: false,
    next_blocker: 'device runtime fault',
  });
  ok(registry[0].structural_verified &&
     registry[0].structural_candidate_head === 'sealed-head' &&
     registry[0].structural_contract_sha256 === 'contract-digest',
  'an incomplete runtime attempt preserves exact-HEAD structural evidence');
  registry = api.upsertMegaCandidate(registry, {
    id: 'structural', source: 'search', status: 'authoring',
    tree: '/state/structural', head: 'changed-head', claim_complete: false,
  });
  ok(!registry[0].structural_verified,
    'a changed source HEAD still revokes structural evidence');
}

console.log('\n# selection is knowledge-blind and absolute-to-frozen');
{
  const selected = api.selectMegaCandidate([
    complete('guided', 1.0448),
    complete('autonomous', 1.06),
    complete('slow', 0.99),
  ], 1.45);
  ok(selected.selected && selected.selected.id === 'autonomous',
    'the numerically fastest passing candidate wins regardless of knowledge source');
  ok(!api.selectMegaCandidate([complete('slow', 0.99)], 0).selected,
    'a correct sub-baseline candidate cannot be final output');
  ok(api.selectMegaSearchParent([
    complete('guided', 1.0448), complete('autonomous', 1.06),
  ]).id === 'autonomous', 'continuation parent is the fastest verified candidate');
}

console.log('\n# search may score a complete partial fusion');
{
  const score = 1.02;
  const meta = {
    id: 'partial', source: 'search', status: 'authoring',
    tree: '/state/partial', head: 'partial-head', attempt_id: 'partial:1',
  };
  const ver = {
    status: 'verified', claim_complete: true, attempt_id: 'partial:1',
    evidence_manifest: '/eval/partial.json', candidate_head: 'partial-head',
    verified_geomean: score, per_case: [{ name: '8192_uniform', speedup: score }],
    paired_readings: Array.from({ length: 5 }, (_, i) => ({
      guard: '8192_uniform', base: 4.6 + i * 0.001,
      cand: (4.6 + i * 0.001) / score,
    })),
    null_arm_pct: 0.1, reps: 5,
    accuracy_results: [{ guard: '8192_uniform', metric: 'relL2', value: 0.05, threshold: 0.1 }],
    correctness: 'pass', activation_confirmed: 'yes', activation_on_hardware: 'yes',
    launch_shape: { launches_base: 4, launches_cand: 3 },
    liveness: 'pass', replay_count: 30, graph_safe: 'pass',
    artifact_distinct: 'yes', artifact_hash_base: 'base', artifact_hash_candidate: 'partial',
  };
  const opts = {
    targetGuards: ['8192_uniform'], regressionGuards: [], launchTarget: 2,
    promotionMetric: 'operator_e2e', accuracyMetric: 'relL2', accuracyThreshold: 0.1,
    requiredReplays: 30, requiredPairs: 5, allowPartialFusion: true,
  };
  const partial = api.megaCandidateFromVerification(meta, ver, opts);
  ok(partial.status === 'scored' && partial.launch_pass,
    'a faster 3-launch fusion scores against a 4-launch frozen baseline');
  ok(partial.runtime_verified && partial.score_complete,
    'hardware runtime and score completion are recorded separately from source structure');
  const fullOnly = api.megaCandidateFromVerification(meta, ver, {
    ...opts, allowPartialFusion: false,
  });
  ok(fullOnly.status !== 'scored' && !fullOnly.launch_pass,
    'exact two-launch enforcement remains available for a full-fusion target');
}

console.log('\n# calibration remains fail-closed');
{
  const pc = {
    claim_complete: true, ran: true, switch_present: true, passed: true,
    attempt_id: 'pc:1', evidence_manifest: '/eval/pc.json', measured_pct: -5,
    control_pairs_pct: [-5, -5.1, -4.9, -5.2, -5],
    null_pairs_pct: [0.1, -0.1, 0.2, -0.2, 0.1, 0, -0.1, 0.1],
  };
  ok(api.megaCalibrationClaimPass(pc, {
    expected_pct_lo: -9, expected_pct_hi: -4,
    required_control_pairs: 5, required_null_pairs: 8,
  }), 'complete resolved control calibrates the portfolio');
  ok(!api.megaCalibrationClaimPass({ ...pc, evidence_manifest: '' }, {
    expected_pct_lo: -9, expected_pct_hi: -4,
  }), 'missing evidence manifest fails calibration');
}

console.log(failures === 0
  ? '\nPASS: one Mega candidate registry works with Expert Skills on or off.'
  : `\nFAIL: ${failures} assertion(s)`);
process.exit(failures === 0 ? 0 : 1);
