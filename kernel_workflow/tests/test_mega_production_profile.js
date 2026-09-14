#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const wf = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(wf, 'kernel_workflow.js'), 'utf8');
const args = JSON.parse(fs.readFileSync(
  path.join(wf, 'tasks', 'megamoe_v2_ep8_mega', 'launch_args.json'), 'utf8',
));
const director = fs.readFileSync(path.join(wf, 'roles', 'director.md'), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# production is the bounded default');
ok(args.mega_profile === 'production' && args.budget === 6,
  'default profile uses six candidate turns');
ok(args.mega_time_budget_s === 28800 && args.mega_final_reserve_s === 7200 &&
   args.mega_closeout_reserve_s === 900,
  'modeled target is eight hours with two hours reserved for finalist validation');
ok(args.mega_candidate_timeout_s === 3600 && args.mega_final_timeout_s === 7200,
  'candidate and finalist calls have separate bounded timeouts');
ok(args.use_expert_skills === 'true' &&
   args.expert_skill_id === 'megamoe_ep_mega_fusion' &&
   args.expert_skill_playbook.endsWith('/playbook.md') &&
   args.expert_skill_contract.endsWith('/contract.yaml') &&
   args.expert_skill_validation.endsWith('/validation.yaml') &&
   args.mega_skill_candidate_id == null && args.mega_search_enabled == null,
  'Expert Skills use playbook/contract/validation in the common lifecycle');
ok(6 * args.mega_candidate_timeout_s <=
   args.mega_time_budget_s - args.mega_final_reserve_s,
  'modeled dispatch window can reach all six common candidate turns');

console.log('\n# valid output triggers early convergence');
ok(/MEGA_STOP_ON_DELIVERABLE && turn\.selected/.test(src) &&
   /expertTargetReached/.test(src) &&
   /EXPERT_SKILL_RECORDED_LOW/.test(src) &&
   /searchAttempts >= MEGA_MIN_SEARCH_ATTEMPTS/.test(src),
  'Skill-on search stops early only after reaching its full-fusion target');
ok(/dispatch deadline reached/.test(src) && /MEGA_TIME_BUDGET_S - MEGA_FINAL_RESERVE_S/.test(src),
  'the workflow preserves final-validation time instead of spending the full wall budget on search');

console.log('\n# production validates best-first and pays fallback only on failure');
ok(/for \(let i = 0; i < finalistOrder\.length; i\+\+\)/.test(src) &&
   /Trying the next scored fallback/.test(src),
  'final candidates are validated sequentially in score order');
ok(/roleAgent\('director', 'recover_mega'/.test(src),
  'a completed finalist manifest is recovered without repaying GPU validation');
ok(args.mega_final_top_k === 1 && args.mega_final_fallback_k === 2,
  'one primary finalist is normal; a second is only a failure fallback');
ok(args.mega_tie_noise_pct === 0,
  'production validates the numerically fastest scored candidate first');
ok(/A\.mega_tie_noise_pct != null \? A\.mega_tie_noise_pct : 1\.45/.test(src),
  'runtime preserves an explicit zero instead of restoring the audit tie band');
ok(/VALIDATION_TIER: MEGA_PRODUCTION \? 'identity_only' : 'full'/.test(src),
  'production second Director call checks identity instead of rerunning the GPU suite');
ok(/Production identity-only tier/.test(director),
  'Director documents the no-duplicate-GPU validation contract');
ok(/timeout_marker: true/.test(src) && /MEGA SAFE STOP/.test(src),
  'a timed-out candidate/finalist stops safely instead of overlapping recovery or fallback GPU work');
ok(/const role = 'engineer';\s*\n\s*const roleFile = 'engineer\.md';/.test(src) &&
   /MEGA EXPERT SKILL — NORMATIVE KNOWLEDGE IN THE COMMON LIFECYCLE/.test(src),
  'skill-enabled and skill-disabled Mega candidates use the same Engineer lifecycle');
ok(/max_retries: 1/.test(src),
  'bounded production calls cannot multiply their wall budget through API retries');
ok(/LANE_MANIFEST: laneManifest/.test(src) && /lane\.json/.test(director),
  'a first-attempt timeout leaves a discoverable lane for the next invocation');
ok(/reportReady && !!finalWinner && megaSelectionContractPassed/.test(src),
  'no finalist means no redundant full validation call');
ok(/validation\.final_patch_verified === true/.test(src),
  'identity-only validation still proves the final patch reproduces the selected tree');
ok(/APPLY_TO_ORIGINAL !== 'true'[\s\S]{0,120}validation\.applied_to_original/.test(src),
  'a requested apply-back must be confirmed before production reports delivery');

console.log('\n# audit-only mechanism gates are not production blockers');
ok(args.require_overlap === 'false' && args.require_attribution === 'false',
  'overlap and attribution are diagnostics in production');
ok(/mega_profile=audit/.test(args._audit_profile_overrides) &&
   /require_overlap=true/.test(args._audit_profile_overrides),
  'the exhaustive audit profile remains explicitly available');

console.log(failures === 0
  ? '\nPASS: production is bounded for delivery while audit remains opt-in.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
