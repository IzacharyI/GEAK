#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_workflow.js'), 'utf8');
const verifier = fs.readFileSync(path.join(wf, 'roles', 'verify_engineer.md'), 'utf8');
const director = fs.readFileSync(path.join(wf, 'roles', 'director.md'), 'utf8');
const argsPath = path.resolve(
  __dirname, '..', '..', '..', 'geak_launch', 'megamoe_generic_gpu_v1', 'launch_args.json',
);
const args = JSON.parse(fs.readFileSync(argsPath, 'utf8'));

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# Expert-off Mega remains a complete generic lifecycle');
ok(/const PINNED_MEGA_SKILL = MODE === 'mega' && USE_EXPERT_SKILLS &&/.test(src),
  'disabled Expert Skills cannot select pinned Skill tools or files');
ok(/const CHECK_EXPERT_SKILL_CONTRACT = MODE === 'mega' && USE_EXPERT_SKILLS &&/.test(src),
  'disabled Expert Skills cannot schedule Skill structural verification');
ok(/const REQUIRED_ACCURACY_CASES = Object\.freeze\(argList\([\s\S]{0,240}A\.required_accuracy_cases/.test(src),
  'accuracy coverage has a run-owned input independent of Expert Skills');
ok(/requiredAccuracyCases: REQUIRED_ACCURACY_CASES/.test(src) &&
   /REQUIRED_ACCURACY_CASES,\s*\n\s*MEGA_PLAN_IR/.test(src) &&
   /REQUIRED_ACCURACY_CASES,\s*\n\s*REQUIRE_OVERLAP/.test(src),
  'score-tier and finalist verification receive the run-owned accuracy cases');
ok(/phase\('Analyze'\)/.test(src) && /async function planMegaCandidateTurn/.test(src) &&
   /async function runMegaCandidateTurn/.test(src) && /'director', 'select_mega'/.test(src),
  'generic Analyze, Plan, Author, Verify and final selection remain present');
ok(verifier.includes('When `GRAPH_CONTRACT_TOOL` is empty') &&
   director.includes('When the optional\n     tool is empty') &&
   verifier.includes('never\n   waives direct graph accuracy or liveness evidence'),
  'an absent Skill runtime tool cannot waive generic graph accuracy or liveness');

console.log('\n# Generic launch contract is fresh and Skill-independent');
ok(args.mode === 'mega' && String(args.use_expert_skills) === 'false',
  'launch enables Mega and disables Expert Skills');
ok(args.expert_skill_id === '' && String(args.require_expert_skill_bundle_identity) === 'false' &&
   String(args.verify_expert_skill_contract) === 'false' &&
   String(args.require_expert_skill_contract) === 'false' &&
   String(args.block_on_expert_skill_contract) === 'false',
  'all Expert Skill pinning and contract switches are disabled');
ok(String(args.mega_resume_state) === 'false' &&
   /megamoe_generic_gpu_v1$/.test(String(args.state_dir)),
  'generic control starts from a fresh isolated state');
ok(JSON.stringify(args.required_accuracy_cases) === JSON.stringify(['128', '512', '8192']) &&
   args.accuracy_metric === 'relL2' && Number(args.accuracy_threshold) === 0.1,
  '128/512/8192 direct accuracy remains an explicit run contract');
ok(Number(args.launch_target) === 2 && String(args.allow_partial_fusion) === 'false' &&
   Number(args.required_replays) === 256 && args.target_guards.includes('8192_uniform'),
  'two-launch, replay and target performance gates remain explicit');
ok(args.task.includes('path=MEGA on all 8 ranks') &&
   args.task.includes('frozen BASELINE_PER_CASE') &&
   args.task.includes('same-tree activation switch comparison is diagnostic only'),
  'task preserves activation and frozen-denominator acceptance semantics');

if (failures) process.exitCode = 1;
else console.log('\nPASS: Expert-off Mega keeps the generic lifecycle and run-owned acceptance gates.');
