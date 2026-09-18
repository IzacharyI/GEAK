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
const playbook = skill;
const contract = skill;
const validation = skill;
const engineer = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'engineer.md',
), 'utf8');
const verifier = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'verify_engineer.md',
), 'utf8');
const director = fs.readFileSync(path.join(
  ROOT, 'kernel_workflow', 'roles', 'director.md',
), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# the experimental transfer is explicit and workflow-owned');
ok(/role: experimental_authoring_prior/.test(skill) &&
   /does not create a reproduction mode/.test(skill),
  'skill declares explicitly pinned constraints inside the common Mega lifecycle');
ok(/embedded_components: \[planner_extension, contract, runtime_validation\]/.test(skill) &&
   /```expert-skill-planner-extension/.test(skill) &&
   /```expert-skill-contract/.test(skill) &&
   /```expert-skill-runtime-python/.test(skill),
  'one skill file embeds human, machine, evidence, and runtime components');
ok(/normative: true/.test(playbook) &&
   /normative_scope: explicitly_pinned_authoring/.test(playbook) &&
   /does not create a reproduction mode,[\s\S]{0,120}reserved candidate source/.test(playbook),
  'normative implementation constraints do not create a reproduction sub-mode');
ok(/schema_version: expert-skill-contract-v1/.test(contract) &&
   /unified_semantic_loop/.test(contract) &&
   /direct_runtime_combine_item/.test(contract) &&
   /names_are_normative: false/.test(contract) &&
   /severity: advisory/.test(contract),
  'the generic checker receives semantic claims plus non-normative adapter hints');
ok(/schema_version: expert-skill-validation-v1/.test(validation) &&
   /status: experimental/.test(validation) &&
   /auto_apply: false/.test(validation),
  'reference evidence, transfer status, and auto-application are separate');
ok(/G1 and G2 are chosen inside the same loop[\s\S]*no grid barrier/i.test(skill),
  'expert knowledge carries the tile-pipeline mechanism, not merely launch fusion');

console.log('\n# runtime uses the same roles and candidate source');
ok(/EXPERIMENTAL TRANSFER, EXPLICIT PIN/.test(wf),
  'enabled skill is appended to the common planner/engineer prompt');
ok(/pinned authoring usage requires mega_structural_only=true/.test(wf) &&
   /pinned candidate_validation requires the structural contract/.test(wf),
  'experimental usage is mechanically limited to explicit authoring or validation');
ok(/expert_skill_id must be a safe slug/.test(wf) &&
   /PINNED_MEGA_SKILL/.test(wf) &&
   /explicitly pinned Expert Skill requires validation status/.test(wf),
  'explicit Skill status and component paths cannot float independently');
ok(/pinned candidate_validation missing gates/.test(wf) &&
   /required_replays>=256/.test(wf) &&
   /direct_graph_accuracy/.test(wf),
  'experimental candidate validation requires the full runtime evidence profile');
ok(/Candidate source remains search\/integrated/.test(wf),
  'expert guidance does not introduce a validated-skill candidate');
ok(!/role = source === 'validated_skill' \? 'mega_engineer'/.test(wf) &&
   /const role = 'engineer'/.test(wf),
  'a special Skill executor is absent from dispatch');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(wf),
  'no reserved reproduction lane is registered');
ok(/authoringContractNeedsPreflight/.test(wf) &&
   /const stagedGpuAuthoring/.test(wf) &&
   /STAGED_GPU_AUTHORING/.test(wf) &&
   /STRUCTURAL_PREFLIGHT_REQUIRED_NO_GPU/.test(wf) &&
   /AUTHORING_CONTRACT_PREFLIGHT_REQUIRED/.test(wf),
  'candidate validation permits staged GPU smoke while structural-only remains GPU-forbidden');
ok(/fail-closed GPU prohibition unless/.test(engineer) &&
   /earliest construction\/JIT\/small-correctness smoke/.test(engineer) &&
   /temporary compile-time[\s\S]*earns no correctness\/performance credit/.test(engineer) &&
   /Once you apply a production fix[\s\S]*GPU authorization is revoked/.test(engineer),
  'Engineer may bisect reversibly but cannot use stale evidence after a production fix');
ok(/preserve every failing contract check ID as its own entry/.test(engineer) &&
   /never collapse them into a summary/.test(engineer) &&
   /GPU absence is not a reason to defer source authoring/.test(engineer),
  'structural-only authoring keeps exact failures and commits coherent GPU-free source progress');
ok(/postAuthoringVerify/.test(wf) &&
   /priorLaneHead/.test(wf) &&
   /structuralPassThisTurn/.test(wf) &&
   /meta\.structural_candidate_head === expectedHead/.test(wf) &&
   /postAuthoringVerify \|\|/.test(wf),
  'a new structurally verified authoring HEAD reaches independent on-card Verify in the same turn');
ok(/required_changed_files_missing/.test(verifier) &&
   /Never synthesize failures from informational/.test(verifier) &&
   /resources\.parameters/.test(verifier),
  'structural verifier preserves exact PlanIR and does not promote reference parity advisories');
ok(/CANDIDATE_IMPORT_MODULES/.test(wf) &&
   /CANDIDATE_PYTHONPATH/.test(wf) &&
   /every module in \$\{JSON\.stringify\(CANDIDATE_IMPORT_MODULES\)\}/.test(wf),
  'the caller supplies a generic candidate import-identity contract');
ok(/IMPORT_IDENTITY_VOID/.test(engineer) &&
   /IMPORT_IDENTITY_VOID/.test(verifier) &&
   /IMPORT_IDENTITY_VOID/.test(director),
  'author, verifier and final arbiter reject installed/global module resolution');

console.log(failures === 0
  ? '\nPASS: experimental transfer knowledge is explicit and workflow-owned.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
