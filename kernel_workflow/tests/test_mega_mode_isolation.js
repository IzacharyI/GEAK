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
  /function expertSkillsBlock\(role, phase\) \{[\s\S]*?\n\}\n\nfunction analysisSkillBlock/,
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
  const EXPERT_SKILL_PLANNER_GUIDE_FILE =
    '/skills/skills/megamoe_ep_mega_fusion/planner_guide.md';
  const EXPERT_SKILL_AUTHOR_GUIDE_FILE =
    '/skills/skills/megamoe_ep_mega_fusion/author_guide.md';
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
ok(enabled('mega_search_lead', 'analyze').includes('skill.md') &&
   enabled('mega_search_lead', 'analyze').includes('canonical Planner Extension'),
  'Mega Analyze receives the canonical Skill extension');
ok(enabled('mega_search_lead', 'plan_round').includes('planner_guide.md') &&
   !enabled('mega_search_lead', 'plan_round').includes('skill.md once'),
  'round Planner receives only the compact routing guide');
ok(enabled('engineer', 'optimize').includes('author_guide.md') &&
   enabled('engineer', 'optimize').includes('independent Verify owns it'),
  'Author receives only the compact implementation guide');
ok(/Candidate source remains search\/integrated/.test(enabled('engineer', 'optimize')),
  'skill injection cannot create a reproduction candidate source');
const disabled = makeBlock(false);
ok(disabled('mega_search_lead', 'plan_round') === '' && disabled('engineer', 'optimize') === '',
  'with Expert Skills off the same roles receive no matched fusion knowledge');

console.log('\n# no reproduction scheduler or reserved skill lane');
ok(!/id: MEGA_SKILL_CANDIDATE_ID, source: 'validated_skill'/.test(src),
  'Mega startup does not register a special skill candidate');
ok(!/if \(USE_EXPERT_SKILLS && \(!MEGA_SEARCH_ENABLED/.test(src) &&
   !/megaSkillLaneDue\(megaCandidateRegistry, currentRound/.test(src),
  'candidate dispatch never branches into a reproduction schedule');
const leakFn = src.match(
  /function freshMegaDirectionLeak\(direction\) \{[\s\S]*?\n\}\n\nasync function planMegaCandidateTurn/,
);
if (!leakFn) throw new Error('cannot lift freshMegaDirectionLeak');
const freshMegaDirectionLeak = new Function(
  `${leakFn[0].replace(/\n\nasync function planMegaCandidateTurn$/, '')}
   return freshMegaDirectionLeak;`,
)();
ok(freshMegaDirectionLeak({
  prompt: 'CONTINUE the existing candidate lane at /tmp/geak_state/old/candidates/x',
}),
  'fresh-state planner contamination detects an external prior lane');
ok(!freshMegaDirectionLeak({
  prompt: 'Author the complete candidate from the frozen source and current MegaPlanIR.',
}),
  'a clean frozen-baseline Analyze direction remains eligible');
const authorLeakFn = src.match(
  /function freshMegaAuthorLeak\(result\) \{[\s\S]*?\n\}\n\nasync function planMegaCandidateTurn/,
);
if (!authorLeakFn) throw new Error('cannot lift freshMegaAuthorLeak');
const freshMegaAuthorLeak = new Function(
  `${authorLeakFn[0].replace(/\n\nasync function planMegaCandidateTurn$/, '')}
   return freshMegaAuthorLeak;`,
)();
ok(freshMegaAuthorLeak({
  notes: 'Reproduced the same-candidate v3 authored source and copied its plan IR.',
}),
  'fresh-state Author disclosure detects copied prior source and plan');
ok(!freshMegaAuthorLeak({
  notes: 'Authored from the frozen baseline and current plan.',
}),
  'fresh source authoring does not trigger the disclosure gate');
ok(!freshMegaAuthorLeak({
  notes: 'Lane setup copied frozen_baseline into candidate tree before authoring.',
}),
  'required frozen-baseline lane initialization is not mistaken for prior reuse');
ok(/freshLane && freshMegaDirectionLeak\(raw\)/.test(src) &&
   /restored the ' \+\s*'clean Analyze direction from the frozen baseline/.test(src) &&
   /base_candidate_id: 'frozen_baseline'/.test(src),
  'fresh runs programmatically replace contaminated Planner output before Author');
ok(/fresh candidate Author disclosed reuse/.test(src) &&
   /fresh_candidate_tree_reuse/.test(src) &&
   /FORBIDDEN_CANDIDATE_TREE_DIGESTS\.has/.test(src),
  'fresh Author disclosure and exact prior tree digest both block structural Verify');
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
ok(/allowPartialFusion: ALLOW_PARTIAL_FUSION/.test(src) &&
   /launches >= targetLaunches && launches < baseLaunches/.test(src),
  'caller policy controls whether a measured partial launch reduction may score');
ok(/let analysis = await agentT\(/.test(src) &&
   /roleAgent\(MODE === 'mega' \? 'mega_search_lead' : 'tech_lead'/.test(src),
  'Mega still performs its real Analyze phase');
ok(/const MEGA_ANALYZE_SCHEMA = \{[\s\S]*'candidate_directions'[\s\S]*'task_graph'[\s\S]*'resource_timeline'/.test(src) &&
   /schema: MODE === 'mega' \? MEGA_ANALYZE_SCHEMA : ANALYZE_SCHEMA/.test(src),
  'Mega Analyze cannot return roadmap prose while omitting its structured pipeline artifacts');
ok(/const MEGA_ANALYZE_SCHEMA = \{[\s\S]*candidate_directions:[\s\S]*'candidate_id'[\s\S]*'target_topology'/.test(src),
  'Mega Analyze preserves Skill candidate identity and topology through StructuredOutput');
ok(/baseline_operator_map/.test(src) &&
   /function applyBaselineOperatorMap/.test(src) &&
   /direction\.focus_files = \[\.\.\.new Set/.test(src) &&
   /value\.modifiable_files = \[\.\.\.new Set/.test(src),
  'Analyze-discovered baseline transformations automatically enter Author scope');
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
const timeoutSafeFn = src.match(
  /function safeVerifyTimeoutRecovery\(result\) \{[\s\S]*?\n\}\n\nasync function planMegaCandidateTurn/,
);
if (!timeoutSafeFn) throw new Error('cannot lift safeVerifyTimeoutRecovery');
const safeVerifyTimeoutRecovery = new Function(
  `${timeoutSafeFn[0].replace(/\n\nasync function planMegaCandidateTurn$/, '')}
   return safeVerifyTimeoutRecovery;`,
)();
ok(/check twice at least 10s apart/.test(src) &&
   safeVerifyTimeoutRecovery({
     timeout_recovery_safe: true, active_gpu_processes: 0, lane_lock_free: true,
   }) &&
   !safeVerifyTimeoutRecovery({
     timeout_recovery_safe: true, active_gpu_processes: 1, lane_lock_free: true,
   }) &&
   !safeVerifyTimeoutRecovery({
     timeout_recovery_safe: true, active_gpu_processes: 0, lane_lock_free: false,
   }),
  'timed-out Verify retries only after process and lane quiescence');
ok(/verify_timeout_retry/.test(src) &&
   /VERIFY_TIER: 'timeout_recovery_smoke'/.test(src) &&
   /TIMEOUT_RECOVERY_ATTEMPT: '1'/.test(src) &&
   /max_retries: 1/.test(src),
  'timeout recovery uses one isolated same-HEAD smoke retry');
ok(/const noGpuFrontMatter = MEGA_STRUCTURAL_ONLY \|\| MEGA_ROUTE_ONLY/.test(src) &&
   /const benchCache = noGpuFrontMatter \? null/.test(src) &&
   /const bench = noGpuFrontMatter \? structuralBench/.test(src) &&
   /STRUCTURAL_ONLY: no GPU benchmark/.test(src) &&
   /const pool = \(MEGA_STRUCTURAL_ONLY \|\| MEGA_ROUTE_ONLY\)/.test(src) &&
   /\? \(MEGA_STRUCTURAL_ONLY\s*\? 'STRUCTURAL_ONLY_NO_GPU'/.test(src),
  'structural-only mode cannot sample, lease, or benchmark a GPU');
ok(/const MEGA_ROUTE_ONLY = MODE === 'mega'/.test(src) &&
   /MEGA_STRUCTURAL_ONLY \|\| MEGA_ROUTE_ONLY\)\s*\? null : await samplePool/.test(src) &&
   /if \(MEGA_ROUTE_ONLY\) \{[\s\S]{0,500}route-only PASS/.test(src) &&
   /no candidate source or GPU was touched/.test(src) &&
   /!noGpuFrontMatter && MEGA_FAST_TEST/.test(src) &&
   /mega_route_only: true[\s\S]{0,500}validation_status: 'route_only'/.test(src),
  'route-only mode stops after topology derivation without GPU, source, state, or cache mutation');
ok(/requiredAccuracyCases: EXPERT_SKILL_ACCURACY_CASES/.test(src) &&
   /EXPERT_SKILL_ACCURACY_CASES, \.\.\.TARGET_GUARDS/.test(src),
  'Expert Skill accuracy cases are mandatory functional and scoring gates');
ok(/EXPERT_SKILL_ACCURACY_CASES\.length > 0, 'expert_skill_accuracy_cases'/.test(src),
  'pinned candidate validation fails closed without declared accuracy cases');
ok(/label: `mega:structure:\$\{candidateId\}`[\s\S]{0,120}timeout_ms: 600000/.test(src) &&
   /if \(shouldStructuralVerify\) megaAdvanceMs\(600000\)/.test(src),
  'deep structural verification has a replay-consistent ten-minute budget');
ok(engineer.includes('Every staged smoke') &&
   engineer.includes('candidate-only harness mode') &&
   engineer.includes('path marker'),
  'staged GPU authoring cannot count fallback-path smoke evidence');

console.log(failures === 0
  ? '\nPASS: mode=mega has one lifecycle; Expert Skills are optional normative knowledge.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
