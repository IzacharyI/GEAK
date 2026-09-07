#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_workflow.js'), 'utf8');
const bootstrap = fs.readFileSync(
  path.resolve(__dirname, '..', 'scripts', 'bootstrap_task.sh'), 'utf8',
);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const fn = src.match(/function expertSkillsBlock\(role\) \{[\s\S]*?\n\}\n\nfunction analysisSkillBlock/);
if (!fn) throw new Error('cannot lift expertSkillsBlock');
const functionText = fn[0].replace(/\n\nfunction analysisSkillBlock$/, '');
const expertSkillsBlock = new Function(`
  const USE_EXPERT_SKILLS = true;
  const EXPERT_SKILL_ROLES = new Set(['tech_lead','engineer','deep_engineer','mega_engineer']);
  const EXPERT_SKILLS_DIR = '/skills';
  const WORKFLOW_DIR = '/workflow';
  const MODE = 'mega';
  const MEGA_SKILL_ID = 'megamoe_ep_mega_fusion';
  const CAPABILITY_EVAL = false;
  ${functionText}
  return expertSkillsBlock;
`)();

console.log('\n# exact M2.5 recipe is isolated to its candidate lane');
ok(expertSkillsBlock('mega_engineer').includes('/skills/skills/megamoe_ep_mega_fusion/skill.md'),
  'mega_engineer receives the validated M2.5 deconstruction');
ok(expertSkillsBlock('engineer') === '' && expertSkillsBlock('tech_lead') === '',
  'ordinary search and planning do not receive the exact M2.5 recipe');

console.log('\n# Mega uses the shared budget loop, not a blocking pre-loop gate');
ok(/if \(MODE === 'mega'\) \{[\s\S]{0,800}const turn = await runMegaCandidateTurn/.test(src),
  'the shared optimization loop dispatches one Mega candidate turn');
ok(/if \(false && MODE === 'mega'\)/.test(src) === false,
  'there is no executable false-gated Reproduce branch masquerading as the design');
ok(!/validated_artifact|TRUSTED_CANDIDATES|m25_artifact/.test(src),
  'hand-authored artifacts are absent from the candidate contract');
ok(/mega mode refuses trusted_candidates/.test(bootstrap),
  'bootstrap refuses attempts to register executable M2.5 source');

console.log('\n# final selection is a speedup, not correctness-only admission');
ok(/c\.absolute_score > 1\.0/.test(src),
  'candidate selection requires speedup above frozen MegaMoE V2');
ok(/finalPrimary > 1\.0/.test(src),
  'final deliverable independently rechecks the above-baseline condition');

console.log(failures === 0
  ? '\nPASS: Mega skill production is isolated while all candidates share final selection.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
