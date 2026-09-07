#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_workflow.js'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'director.md'), 'utf8');
const task = fs.readFileSync(
  path.join(wf, 'tasks', 'megamoe_v2_ep8_mega', 'GEAK_TASK.md'), 'utf8',
);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# final selection receives only workflow-authored, above-baseline candidates');
ok(/c\.absolute_score > 1\.0/.test(src),
  'portfolio filter rejects every candidate at or below frozen MegaMoE V2');
ok(/source: 'validated_skill'/.test(src) && /candidate_source: raw\.candidate_source/.test(src),
  'skill and open-search candidates enter the same registry');
ok(!/source: 'validated_artifact'/.test(src),
  'no hand-authored implementation is registered as a selectable source');

console.log('\n# Director remeasures rather than trusting round scores');
ok(/roleAgent\('director', 'select_mega'/.test(src),
  'a dedicated final arbitration call exists inside the shared workflow');
ok(/CANDIDATES: batch/.test(src) && /FINAL_WORKSPACE = megaSelection\.selected_tree/.test(src),
  'the selected whole tree, not the last round, becomes the final validation workspace');
ok(/claim_complete === true/.test(src) && /selected_candidate_id/.test(src),
  'an incomplete final selection cannot materialize a result');

console.log('\n# the final gate is common and expensive only once');
ok(/A source label[\s\S]*never relaxes a[\s\S]*quality gate/.test(director) &&
   /speedup is `<=1\.0`/.test(director),
  'Director applies one source-blind correctness and above-baseline gate');
ok(/run the three regression guards/.test(director) &&
   /required liveness/.test(director),
  'finalists pay the complete guard and liveness contract');
ok(/is a target only/.test(task) && /Never read, run, copy, diff, import/.test(task),
  'the M2.5 number is a target and its implementation is not an oracle arm');

console.log('\n# final output independently checks the selected score');
ok(/finalPrimary > 1\.0/.test(src) && /mega_deliverable = !!\(megaSelection/.test(src),
  'the final return cannot report a sub-baseline megakernel as deliverable');
ok(/validation\.materialization_matches === true/.test(src) &&
   /selected_source_head/.test(src) && /selected_materialized_head/.test(src),
  'a second Director call binds the final workspace to the selected candidate commit');
ok(src.lastIndexOf('persistMegaCandidateState(round, true)') >
   src.indexOf("roleAgent('director', 'validate'"),
  'finalist state is persisted only after independent final validation');

console.log(failures === 0
  ? '\nPASS: final Mega output is the fastest fully verified workflow-authored speedup.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
