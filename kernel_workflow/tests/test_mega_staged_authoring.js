#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
const wf = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js'), 'utf8');
const skill = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'skill.md',
), 'utf8');
const reference = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'recipe_v1.md',
), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# the M2.5 package is matched normative knowledge');
ok(/role: normative_prior/.test(skill) &&
   /does not create a reproduction mode/.test(skill),
  'skill declares normative constraints inside the common Mega lifecycle');
ok(/reference_file: recipe_v1\.md/.test(skill) &&
   /detailed validated reference/.test(skill),
  'skill points to the detailed M2.5 design reference');
ok(/normative: true/.test(reference) &&
   /normative_scope: semantic_and_compiler_shape/.test(reference) &&
   /does not create a reproduction mode,[\s\S]{0,80}reserved lane/.test(reference),
  'normative implementation constraints do not create a reproduction sub-mode');
ok(/CTA.*local shard drains[\s\S]*without a global stage barrier/i.test(skill),
  'expert knowledge carries the tile-pipeline mechanism, not merely launch fusion');

console.log('\n# runtime uses the same roles and candidate source');
ok(/MEGA EXPERT SKILL — NORMATIVE KNOWLEDGE IN THE COMMON LIFECYCLE/.test(wf),
  'enabled skill is appended to the common planner/engineer prompt');
ok(/Candidate source remains search\/integrated/.test(wf),
  'expert guidance does not introduce a validated-skill candidate');
ok(!/role = source === 'validated_skill' \? 'mega_engineer'/.test(wf) &&
   /const role = 'engineer'/.test(wf),
  'the special recipe executor is absent from dispatch');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(wf),
  'no reserved reproduction lane is registered');

console.log(failures === 0
  ? '\nPASS: M2.5 knowledge guides the ordinary Mega lifecycle.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
