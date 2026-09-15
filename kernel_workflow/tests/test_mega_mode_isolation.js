#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_workflow.js'), 'utf8');
const searchLead = fs.readFileSync(
  path.resolve(__dirname, '..', 'roles', 'mega_search_lead.md'), 'utf8',
);
const engineer = fs.readFileSync(
  path.resolve(__dirname, '..', 'roles', 'engineer.md'), 'utf8',
);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const fn = src.match(
  /function expertSkillsBlock\(role\) \{[\s\S]*?\n\}\n\nfunction analysisSkillBlock/,
);
if (!fn) throw new Error('cannot lift expertSkillsBlock');
const functionText = fn[0].replace(/\n\nfunction analysisSkillBlock$/, '');
const makeBlock = (enabled) => new Function(`
  const USE_EXPERT_SKILLS = ${enabled};
  const EXPERT_SKILL_ROLES = new Set([
    'tech_lead','author_engineer','engineer','deep_engineer','mega_search_lead'
  ]);
  const EXPERT_SKILLS_DIR = '/skills';
  const EXPERT_SKILL_ID = 'megamoe_ep_mega_fusion';
  const EXPERT_SKILL_DIR = '/skills/skills/megamoe_ep_mega_fusion';
  const EXPERT_SKILL_PLAYBOOK_FILE = '/skills/skills/megamoe_ep_mega_fusion/skill.md';
  const EXPERT_SKILL_PLANNER_EXTENSION_FILE =
    '/skills/skills/megamoe_ep_mega_fusion/skill.md';
  const EXPERT_SKILL_CONTRACT_FILE = '/skills/skills/megamoe_ep_mega_fusion/skill.md';
  const EXPERT_SKILL_REVISION = 'v1';
  const EXPERT_SKILL_BUNDLE_TOOL = '/skills/_contribute/validate_skill.py';
  const EXPERT_SKILL_BUNDLE_SHA256 = 'b'.repeat(64);
  const EXPERT_SKILL_PLANNER_EXTENSION_SHA256 = 'p'.repeat(64);
  const EXPERT_SKILL_CONTRACT_SHA256 = 'c'.repeat(64);
  const EXPERT_SKILL_VALIDATION_STATUS = 'experimental';
  const EXPERT_SKILL_USAGE = 'authoring';
  const REQUIRE_EXPERT_SKILL_BUNDLE_IDENTITY = true;
  const WORKFLOW_DIR = '/workflow';
  const MODE = 'mega';
  const CAPABILITY_EVAL = false;
  ${functionText}
  return expertSkillsBlock;
`)();

console.log('\n# Expert Skills change knowledge, not Mega mode');
const enabled = makeBlock(true);
ok(enabled('mega_search_lead').includes('skill.md') &&
   enabled('mega_search_lead').includes('machine-readable Planner Extension') &&
   enabled('mega_search_lead').includes('structural contract'),
  'the common Mega planner receives the single embedded Skill when enabled');
ok(enabled('engineer').includes('EXPERIMENTAL TRANSFER, EXPLICIT PIN'),
  'the common Engineer receives the explicitly pinned transfer knowledge');
ok(/Candidate source remains search\/integrated/.test(enabled('engineer')),
  'skill injection cannot create a reproduction candidate source');
const disabled = makeBlock(false);
ok(disabled('mega_search_lead') === '' && disabled('engineer') === '',
  'with Expert Skills off the same roles receive no matched fusion knowledge');

console.log('\n# no reproduction scheduler or reserved skill lane');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(src),
  'Mega startup does not register a special skill candidate');
ok(!/if \(USE_EXPERT_SKILLS && \(!MEGA_SEARCH_ENABLED/.test(src) &&
   !/megaSkillLaneDue\(megaCandidateRegistry, currentRound/.test(src),
  'candidate dispatch never branches into a reproduction schedule');
ok(/const role = 'engineer';\s*\n\s*const roleFile = 'engineer\.md';/.test(src),
  'all Mega candidates use the same authoring role');
ok(/MEGA unified mode: Expert Skills are/.test(src),
  'runtime reports a single lifecycle with a knowledge toggle');

console.log('\n# autonomous candidates remain measurable and flexible');
ok(/search plan rejected diagnostic-only direction/.test(src),
  'diagnostic-only work cannot consume a candidate turn');
ok(/Full and profitable partial[\s\S]*fusion are both legal/.test(searchLead) &&
   /partial terminal[\s\S]*complete runnable operator/.test(searchLead),
  'full and partial fusion may both become runnable performance candidates');
ok(/complete runnable operator/.test(searchLead) &&
   /external operator is complete, correctness passes/.test(engineer),
  'partial fusion means a complete operator, not half-implemented source');
ok(/allowPartialFusion: true/.test(src) &&
   /launches >= targetLaunches && launches < baseLaunches/.test(src),
  'search scoring accepts a measured launch reduction while full fusion remains the target');
ok(/let analysis = await agentT\(/.test(src) &&
   /roleAgent\(MODE === 'mega' \? 'mega_search_lead' : 'tech_lead'/.test(src),
  'Mega still performs its real Analyze phase');
ok(/const MEGA_ANALYZE_SCHEMA = \{[\s\S]*'candidate_directions'[\s\S]*'task_graph'[\s\S]*'resource_timeline'/.test(src) &&
   /schema: MODE === 'mega' \? MEGA_ANALYZE_SCHEMA : ANALYZE_SCHEMA/.test(src),
  'Mega Analyze cannot return roadmap prose while omitting its structured pipeline artifacts');
ok(/const MEGA_ANALYZE_SCHEMA = \{[\s\S]*candidate_directions:[\s\S]*'candidate_id'[\s\S]*'target_topology'/.test(src),
  'Mega Analyze preserves Skill candidate identity and topology through StructuredOutput');
ok(/'task_graph', 'resource_timeline', 'mega_plan_ir'/.test(src) &&
   /required mega_plan_ir invalid/.test(src),
  'Mega Analyze requires a lowerable typed plan, not only a roadmap');
ok(/plan_version/.test(src) &&
   /work_domains/.test(src) &&
   /buffers/.test(src) &&
   /counters/.test(src) &&
   /local_memory/.test(src) &&
   /primary_loop/.test(src) &&
   /compiler_constraints/.test(src) &&
   /evidence_requirements/.test(src),
  'MegaPlanIR v2 strongly types operator-neutral dataflow, resources and evidence');
ok(!/stage2_pointer_count|combine_pointer_count|g2_chunk|combine_third_queue/.test(src) &&
   !/MegaMoE|GEMM1|GEMM2|g2_chunk/.test(searchLead) &&
   !/MegaMoE|8192_uniform|AITER_MEGAMOE|SCATTERED|g2_/.test(src),
  'operator-specific identifiers are Skill data, not workflow or planner schema');
ok(/contract_failures/.test(src) &&
   /correctness.*abi.*lifecycle.*resource\/compiler.*schedule.*performance/s.test(searchLead),
  'Planner receives categorized verifier failures instead of prose-only blockers');
ok(/TASK_GRAPH: analysis\.task_graph/.test(src) &&
   /RESOURCE_TIMELINE: analysis\.resource_timeline/.test(src) &&
   /MEGA_PLAN_IR: analysis\.mega_plan_ir/.test(src) &&
   /EXPERT_SKILL_PLANNER_EXTENSION: EXPERT_SKILL_PLANNER_EXTENSION_FILE/.test(src) &&
   !/MEGA_PLAN_IR: JSON\.stringify/.test(src),
  'the planner receives complete graph/resource/PlanIR objects plus its Skill Extension');
ok(!/MEGA_TOPOLOGY_LEVERS/.test(src) &&
   /target_topology: targetTopology/.test(src),
  'target_topology is no longer deleted by an experimental kill switch');
ok(/Mega round \$\{currentRound\}: \$\{pg\.summary\}/.test(src) &&
   /roadmapLadderGate\(LADDER, \[d\], LADDER_MEASURED\)/.test(src) &&
   /strict Mega direction refused before authoring/.test(src),
  'Mega runs pipe, ladder and strict gates before authoring');
ok(/verify_structure/.test(src) &&
   /structural_verified/.test(src) &&
   /runtime_verified/.test(src) &&
   /score_complete/.test(src),
  'checkpoint, structural, runtime and scored completion are distinct');
ok(/EXPERT_SKILL_REFERENCE_PATH,/.test(src) &&
   !/Advance candidate lane \$\{candidateId\}[\s\S]{0,2500}EXPERT_SKILL_REFERENCE_PATH/.test(src),
  'an optional reference is passed only to post-authoring Verify, never Author');
ok(/EXPERT_SKILL_CONTRACT_TOOL/.test(src) &&
   !/megamoe_ep_mega_fusion|mega_moe_stage1/.test(src),
  'the core workflow exposes a generic Expert Skill contract interface');
ok(/candidateClaimsSkillTarget/.test(src) &&
   /contractBlocksRuntime/.test(src) &&
   /REQUIRE_EXPERT_SKILL_CONTRACT/.test(src),
  'a full Skill target is preflight-gated while partial fallbacks remain measurable');
ok(/const MEGA_STRUCTURAL_ONLY = MODE === 'mega'/.test(src) &&
   /STRUCTURAL_ONLY: '1'/.test(src) &&
   /!MEGA_STRUCTURAL_ONLY/.test(src) &&
   /structural-only acceptance reached/.test(src),
  'structural-only mode forbids runtime Verify and stops on a complete source contract');
ok(/const benchCache = MEGA_STRUCTURAL_ONLY \? null/.test(src) &&
   /const bench = MEGA_STRUCTURAL_ONLY \? structuralBench/.test(src) &&
   /STRUCTURAL_ONLY: no GPU benchmark/.test(src),
  'structural-only mode cannot fall through to a GPU benchmark');

console.log(failures === 0
  ? '\nPASS: mode=mega has one lifecycle; Expert Skills are optional normative knowledge.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
