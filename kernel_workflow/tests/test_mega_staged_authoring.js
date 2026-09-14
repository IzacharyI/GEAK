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
const playbook = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'playbook.md',
), 'utf8');
const contract = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'contract.yaml',
), 'utf8');
const validation = fs.readFileSync(path.join(
  ROOT, 'perf_knowledge', 'expert_skills', 'skills',
  'megamoe_ep_mega_fusion', 'validation.yaml',
), 'utf8');
const engineer = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'engineer.md',
), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# the validated fusion package is matched normative knowledge');
ok(/role: normative_prior/.test(skill) &&
   /does not create a reproduction mode/.test(skill),
  'skill declares normative constraints inside the common Mega lifecycle');
ok(/playbook_file: playbook\.md/.test(skill) &&
   /contract_file: contract\.yaml/.test(skill) &&
   /validation_file: validation\.yaml/.test(skill),
  'skill points to separate human, machine and evidence artifacts');
ok(/normative: true/.test(playbook) &&
   /normative_scope: semantic_and_compiler_shape/.test(playbook) &&
   /does not create a reproduction mode,[\s\S]{0,80}reserved lane/.test(playbook),
  'normative implementation constraints do not create a reproduction sub-mode');
ok(/schema_version: expert-skill-contract-v1/.test(contract) &&
   /host_ready_pointer_identity/.test(contract) &&
   /combine_transport_and_work_domain/.test(contract),
  'the generic checker receives declarative Skill invariants');
ok(/schema_version: expert-skill-validation-v1/.test(validation) &&
   /status: validated/.test(validation),
  'measured validation status is separate from prose');
ok(/CTA.*local shard drains[\s\S]*without a global stage barrier/i.test(skill),
  'expert knowledge carries the tile-pipeline mechanism, not merely launch fusion');

console.log('\n# runtime uses the same roles and candidate source');
ok(/MEGA EXPERT SKILL — NORMATIVE KNOWLEDGE IN THE COMMON LIFECYCLE/.test(wf),
  'enabled skill is appended to the common planner/engineer prompt');
ok(/Candidate source remains search\/integrated/.test(wf),
  'expert guidance does not introduce a validated-skill candidate');
ok(!/role = source === 'validated_skill' \? 'mega_engineer'/.test(wf) &&
   /const role = 'engineer'/.test(wf),
  'a special Skill executor is absent from dispatch');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(wf),
  'no reserved reproduction lane is registered');
ok(/authoringContractNeedsPreflight/.test(wf) &&
   /STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU/.test(wf) &&
   /AUTHORING_CONTRACT_PREFLIGHT_REQUIRED/.test(wf),
  'a full-target author receives no GPU until exact-HEAD structural verification');
ok(/fail-closed GPU prohibition/.test(engineer) &&
   /If you edit any production[\s\S]*GPU authorization is revoked/.test(engineer),
  'Engineer cannot use stale structural evidence or edit and benchmark in one turn');

console.log(failures === 0
  ? '\nPASS: validated fusion knowledge guides the ordinary Mega lifecycle.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
