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
  function guardContract(v, targets, regressions, fallback) {
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
    validMegaCandidateId, normalizeMegaCandidate, upsertMegaCandidate, megaCandidateHardPass,
    selectMegaCandidate, megaSkillLaneDue, megaCalibrationClaimPass,
    pairedGuardReadout, megaFinalSelectionVerdict, megaCandidateFromVerification,
  };
`)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const complete = (id, source, score) => ({
  id, source, status: 'scored', tree: `/state/${id}`, head: `${id}-head`,
  claim_complete: true, attempt_id: `${id}:1`, evidence_manifest: `${id}.json`,
  absolute_score: score, per_case: [{ name: '8192_uniform', speedup: score }],
  paired_readings: Array.from({ length: 5 }, (_, i) => ({
    guard: '8192_uniform', base: 4.6 + i * 0.001,
    cand: (4.6 + i * 0.001) / score,
  })),
  null_arm_pct: 0.1, reps: 5,
  artifact_hash_base: 'base-hash', artifact_hash_candidate: `${id}-hash`,
  correctness_pass: true, activation_pass: true, head_pass: true, launch_pass: true,
  liveness_pass: true, graph_pass: true, guards_pass: true, artifact_distinct: true,
  measurement_pass: true,
});

console.log('\n# no hand-authored artifact source exists in the portfolio contract');
{
  const c = api.normalizeMegaCandidate({ id: 'bad', source: 'validated_artifact' });
  ok(c.status === 'rejected' && !c.source_valid,
    'an imported artifact source is rejected rather than relabeled as search');
  ok(!api.validMegaCandidateId('../escape') && !api.validMegaCandidateId('a/b'),
    'candidate ids cannot escape their state directory');
  ok(api.normalizeMegaCandidate({
    id: 'm25_skill', source: 'search', status: 'scored',
  }).status === 'rejected',
  'ordinary search cannot take over the reserved skill lane');
}

console.log('\n# incomplete claims cannot erase completed evidence');
{
  let registry = api.upsertMegaCandidate([], complete('m25_skill', 'validated_skill', 1.0448));
  registry = api.upsertMegaCandidate(registry, {
    id: 'm25_skill', source: 'validated_skill', status: 'authoring',
    claim_complete: false, absolute_score: 0, attempts: 2, head: 'new-unverified-head',
  });
  ok(registry[0].claim_complete && registry[0].absolute_score === 1.0448,
    'a force-emitted partial does not replace the last complete score');
  ok(registry[0].head === 'm25_skill-head' &&
     registry[0].working_head === 'new-unverified-head',
  'new WIP HEAD is recorded separately and never paired with old verified evidence');
}

console.log('\n# slow candidates remain WIP and never become final output');
{
  const selected = api.selectMegaCandidate([
    complete('slow', 'search', 0.99),
    complete('fast', 'search', 1.02),
  ], 1.45);
  ok(selected.selected && selected.selected.id === 'fast',
    'selection requires absolute speedup above frozen MegaMoE V2');
  ok(!api.selectMegaCandidate([complete('slow', 'search', 0.99)], 1.45).selected,
    'a correct sub-baseline megakernel cannot be final output');
}

console.log('\n# validated-skill candidate is the stable tie-breaker, not a ceiling');
{
  const tied = api.selectMegaCandidate([
    complete('m25_skill', 'validated_skill', 1.0448),
    complete('search', 'search', 1.055),
  ], 1.45);
  ok(tied.selected.id === 'm25_skill' && tied.tie_kept_skill,
    'a challenger inside the measured noise keeps the stable skill candidate');
  const win = api.selectMegaCandidate([
    complete('m25_skill', 'validated_skill', 1.0448),
    complete('search', 'search', 1.07),
  ], 1.45);
  ok(win.selected.id === 'search' && !win.tie_kept_skill,
    'a significant faster result supersedes the skill candidate');
}

console.log('\n# skill lane is reserved but non-blocking');
{
  const skill = { id: 'm25_skill', source: 'validated_skill', status: 'authoring', attempts: 1 };
  ok(!api.megaSkillLaneDue([skill], 2, 'm25_skill', 6, 3),
    'ordinary search gets the rounds between skill attempts');
  ok(api.megaSkillLaneDue([skill], 4, 'm25_skill', 6, 3),
    'the persistent skill lane returns on its configured interval');
  ok(!api.megaSkillLaneDue([{ ...skill, status: 'scored' }], 4, 'm25_skill', 6, 3),
    'a scored skill candidate stops consuming authoring attempts');
  const due = (round, attempts) => api.megaSkillLaneDue([
    { ...skill, attempts },
  ], round, 'm25_skill', 4, 2, 2);
  ok(due(1, 0) && due(2, 1) && !due(3, 2) && due(4, 2) &&
     !due(5, 3) && due(6, 3),
  'production cold start schedules skill,skill,search,skill,search,skill');
}

console.log('\n# calibration requires an atomic complete claim');
{
  const pc = {
    claim_complete: true, ran: true, switch_present: true, passed: true,
    attempt_id: 'control:1', evidence_manifest: '/eval/control/evidence.json',
    measured_pct: -5, control_pairs_pct: [-5, -4.8, -5.2, -4.9, -5.1],
    null_pairs_pct: [0.1, -0.2, 0.15, -0.1, 0.08, -0.05, 0.11, -0.09],
  };
  ok(api.megaCalibrationClaimPass(pc, { expected_pct_lo: -9, expected_pct_hi: -4 }),
    'a completed resolved control enables scoring');
  ok(!api.megaCalibrationClaimPass({ ...pc, claim_complete: false },
    { expected_pct_lo: -9, expected_pct_hi: -4 }),
  'a partial claim never calibrates the portfolio');
  ok(!api.megaCalibrationClaimPass({
    claim_complete: true, ran: true, switch_present: true, passed: true, measured_pct: -5,
  }, { expected_pct_lo: -9, expected_pct_hi: -4 }),
  'an in-band scalar without control/null pairs cannot calibrate the portfolio');
}

console.log('\n# final selection is bound to supplied finalists and controlled output');
{
  const finalists = [
    complete('m25_skill', 'validated_skill', 1.0448),
    complete('search', 'search', 1.07),
  ];
  const row = (c) => ({
    candidate_id: c.id, source: c.source, tree: c.tree, head: c.head, score: c.absolute_score,
    claim_complete: true, correctness: 'pass', guards_pass: true,
    path_marker: 'MEGA==8', path_marker_count: 8, launches: 2, graph_safe: 'pass',
    artifact_distinct: 'yes', overlap_measured: 'yes', overlap_fraction: 0.2,
    overlap_cu_fraction: 0.15,
    artifact_hash_base: 'base-hash', artifact_hash_candidate: `${c.id}-hash`,
    overlap_scattered_reading: 0.0, overlap_forced_reading: 0.8,
    attribution_complete: true, accuracy_metric: 'relL2', accuracy_value: 0.05,
    liveness_replays: 256,
    attribution: {
      changed_us: 4400, replaced_sum_us: 4600,
      residual_ms_base: 0.1, residual_ms_cand: 0.05,
      guard: '8192_uniform', method: 'same-timeline rank-max', absolute_to_frozen: true,
    },
    replay_results: [
      '8192_uniform', '8192_rank-mixed-skew', '512_uniform', '512_rank-mixed-skew',
    ].map((guard) => ({
      guard, count: 256, status: 'pass', graph_safe: 'pass', arrival_jitter: true,
    })),
    paired_readings: [
      ...Array.from({ length: 5 }, (_, i) => ({
        guard: '8192_uniform', base: 4.6 + i * 0.001,
        cand: (4.6 + i * 0.001) / c.absolute_score,
      })),
      ...Array.from({ length: 5 }, (_, i) => ({
        guard: '8192_rank-mixed-skew', base: 5.3 + i * 0.001, cand: 5.3 + i * 0.001,
      })),
      ...Array.from({ length: 16 }, (_, i) => ({
        guard: '512_uniform', base: 0.74 + i * 0.0001, cand: 0.74 + i * 0.0001,
      })),
      ...Array.from({ length: 16 }, (_, i) => ({
        guard: '512_rank-mixed-skew', base: 0.86 + i * 0.0001, cand: 0.86 + i * 0.0001,
      })),
    ],
    null_arm_pct: 0.1,
    per_case: [
      { name: '8192_uniform', speedup: c.absolute_score },
      { name: '8192_rank-mixed-skew', speedup: 1.0 },
      { name: '512_uniform', speedup: 1.0 },
      { name: '512_rank-mixed-skew', speedup: 1.0 },
    ],
  });
  const selection = {
    claim_complete: true, attempt_id: 'final:1', evidence_manifest: '/eval/final/evidence.json',
    selected_candidate_id: 'search', selected_source: 'search',
    selected_tree: '/eval/mega_selected', selected_head: 'materialized-head',
    materialized_from_head: 'search-head', selected_score: 1.07,
    candidates: finalists.map(row),
  };
  const opts = {
    selectedWorkspace: '/eval/mega_selected', launchTarget: 2, requiredReplays: 256,
    worldSize: 8,
    targetGuards: ['8192_uniform'],
    regressionGuards: ['8192_rank-mixed-skew', '512_uniform', '512_rank-mixed-skew'],
    requiredPairs: 5,
    requiredPairsByGuard: {
      '8192_uniform': 5, '8192_rank-mixed-skew': 5,
      '512_uniform': 16, '512_rank-mixed-skew': 16,
    },
    accuracyMetric: 'relL2', accuracyThreshold: 0.10,
    requireOverlap: true, requireAttribution: true, tieNoisePct: 1.45,
  };
  ok(api.megaFinalSelectionVerdict(selection, finalists, opts).pass,
    'fastest fully passing finalist in the controlled workspace is accepted');
  ok(!api.megaFinalSelectionVerdict({
    ...selection, selected_tree: '/outside/hand-tree',
  }, finalists, opts).pass,
  'an arbitrary Director-returned tree cannot become final workspace');
  ok(!api.megaFinalSelectionVerdict({
    ...selection, selected_candidate_id: 'unknown',
  }, finalists, opts).pass,
  'a selected id outside the supplied registry is rejected');
  ok(!api.megaFinalSelectionVerdict({
    ...selection,
    selected_candidate_id: 'm25_skill',
    selected_source: 'validated_skill',
    selected_score: 1.0448,
    candidates: [row(finalists[0])],
  }, finalists, opts).pass,
  'omitting the faster supplied finalist cannot make a slower selection pass');
  ok(!api.megaFinalSelectionVerdict({
    ...selection,
    candidates: finalists.map(row).map((r) =>
      r.candidate_id === 'search' ? { ...r, overlap_measured: 'unknown' } : r),
  }, finalists, opts).pass,
  'missing finalist overlap evidence cannot be hidden by a headline score');
  ok(!api.megaFinalSelectionVerdict({
    ...selection,
    candidates: finalists.map(row).map((r) =>
      r.candidate_id === 'm25_skill' ? { ...r, claim_complete: false } : r),
  }, finalists, opts).pass,
  'a partial Top-K finalist blocks selection instead of disappearing from comparison');
  ok(!api.megaFinalSelectionVerdict({
    ...selection,
    candidates: finalists.map(row).map((r) =>
      r.candidate_id === 'search' ? {
        ...r,
        paired_readings: r.paired_readings.map((p) =>
          p.guard === '8192_uniform' ? { ...p, cand: p.base * 1.1 } : p),
      } : r),
  }, finalists, opts).pass,
  'a positive headline cannot override raw pairs that show the candidate is slower');
}

console.log('\n# 512 bimodality is conditioned arm-blind without hiding a real arm effect');
{
  const bimodal = Array.from({ length: 8 }, (_, i) => ({
    guard: '512_uniform',
    base: i < 2 ? 1.10 : 1.00,
    cand: i < 2 ? 1.10 : 1.00,
  }));
  const split = api.pairedGuardReadout(bimodal, '512_uniform');
  ok(split.bimodal && split.raw_count === 8 && split.count === 6,
    'both arms drawing the slow state triggers fast-state conditioning');
  const largeEffect = Array.from({ length: 8 }, () => ({
    guard: '512_uniform', base: 1.10, cand: 1.00,
  }));
  ok(!api.pairedGuardReadout(largeEffect, '512_uniform').bimodal,
    'clusters containing only one arm are treated as a real effect, not a slow state');
}

console.log(failures === 0
  ? '\nPASS: Mega keeps independent workflow-authored candidates and selects only a real speedup.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
