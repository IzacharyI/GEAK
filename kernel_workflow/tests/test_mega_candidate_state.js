#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_workflow.js'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'director.md'), 'utf8');
const lead = fs.readFileSync(path.join(wf, 'roles', 'tech_lead.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# candidate registry is a first-class resume artifact');
ok(/candidate_registry: \{ type: 'array'/.test(src) &&
   /measurement_calibration: \{ type: 'object'/.test(src),
  'prior-state schema declares candidates and calibration');
ok(/Array\.isArray\(ps\.candidate_registry\)/.test(src) &&
   /for \(const c of ps\.candidate_registry\)/.test(src) &&
   /upsertMegaCandidate\(megaCandidateRegistry, \{/.test(src),
  'resume merges every prior candidate rather than replacing the registry');
ok(/const MEGA_RESUME_STATE = MODE === 'mega'/.test(src) &&
   /STATE_DIR, MEGA_RESUME_STATE/.test(src) &&
   /MEGA_RESUME_STATE=true[\s\S]{0,320}MUST[\s\S]{0,180}prior_state/.test(director),
  'a machine-readable continuation flag outranks stale fresh-run task prose');
ok(/Calibration is intentionally NOT restored as authority/.test(src) &&
   !/megaMeasurementCalibration = \{ \.\.\.megaMeasurementCalibration, \.\.\.ps\.measurement_calibration \}/.test(src),
  'prior calibration is audit data and cannot authorize a new wave');
ok(/megaStateSequenceBase = Math\.max\(0, Number\(ps\.state_sequence\)\)/.test(src) &&
   /megaStateSequenceBase \+ Number\(currentRound\) \* 10/.test(src),
  'state sequence advances from the prior wave instead of resetting with round 1');

console.log('\n# each lane owns persistent source state');
ok(/STATE_DIR}\/candidates\/\$\{candidateId\}\/tree/.test(src),
  'candidate workspaces live under stable per-id state paths');
ok(/If .*\/\.git does not exist[\s\S]*If it exists, continue from its HEAD/.test(src),
  'an existing lane is continued and never recreated');
ok(/candidate_registry: <CANDIDATE_REGISTRY>/.test(lead) &&
   /never collapse them into `\$STATE_DIR\/best`/.test(lead),
  'state writer preserves independent lineages instead of one mutable best tree');
ok(/Mega candidate-registry exception/.test(director) &&
   /resumed:true` plus `prior_state`/.test(director) &&
   /even when[\s\S]*`STATE_DIR\/best` is absent/.test(director),
  'Director restores candidate state without requiring a legacy global best');
ok(/MEGA STATE PERSIST FAILED/.test(src) &&
   /return false;/.test(src) &&
   /candidate state for round \$\{currentRound\} could not be persisted/.test(src),
  'a round cannot advance without a confirmed monotonic state write');
ok(/working_snapshot: \{/.test(src) &&
   /activation: next\.activation/.test(src) &&
   /topology: next\.topology/.test(src) &&
   /changed_files: next\.changed_files/.test(src),
  'unverified candidate progress is persisted as a structured working snapshot');

console.log(failures === 0
  ? '\nPASS: Mega candidate lanes and calibration resume independently.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
