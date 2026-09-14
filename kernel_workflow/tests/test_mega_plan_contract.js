#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_workflow.js'), 'utf8');

const match = src.match(
  /\/\/ <<REPLAY:mega_topology_contract>>([\s\S]*?)\/\/ <<\/REPLAY:mega_topology_contract>>/,
);
if (!match) throw new Error('megaTopologyVerdict not found');
// eslint-disable-next-line no-new-func
const { verdict, claimsTarget, needsPreflight } = new Function(
  `${match[1]}\nreturn { verdict: megaTopologyVerdict, ` +
  `claimsTarget: candidateClaimsPlanTarget, needsPreflight: authoringContractNeedsPreflight };`,
)();

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const plan = {
  target: {
    launch_count: 2,
    required_regions: ['producer', 'consumer'],
    required_queues: ['upstream', 'downstream'],
    required_capabilities: ['item_overlap'],
  },
};

ok(claimsTarget({ target_topology: { launch_count: 2 } }, plan),
  'v2 launch_count marks an exact full-target candidate');
ok(claimsTarget({ target_topology: { launches: 2 } }, plan),
  'the historical launches spelling remains a compatibility input');
ok(!claimsTarget({ target_topology: { launch_count: 3 } }, plan),
  'a partial target does not bypass the full-target structural contract');
ok(needsPreflight(
  { target_topology: { launch_count: 2 } }, plan, null, true,
), 'a new full-target candidate is GPU-forbidden before structural Verify');
ok(needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  { head: 'new', structural_candidate_head: 'old', structural_verified: true },
  true,
), 'a structural pass for another HEAD cannot authorize authoring GPU use');
ok(!needsPreflight(
  { target_topology: { launch_count: 2 } },
  plan,
  { head: 'same', structural_candidate_head: 'same', structural_verified: true },
  true,
), 'the exact independently verified full-target HEAD may use a GPU');
ok(!needsPreflight(
  { target_topology: { launch_count: 3 } }, plan, null, true,
), 'a complete partial fallback is not forced through the full-target Skill contract');

ok(!verdict({}, plan).pass,
  'a Mega direction cannot omit target_topology');
ok(!verdict({
  topology_inferred: true,
  target_topology: {
    launch_count: 2,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'a compatibility fallback does not satisfy a strict planner contract');
ok(verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 2,
    included_regions: ['producer', 'consumer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan).pass,
'an operator-neutral terminal topology matching MegaPlanIR passes');
ok(!verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 2,
    included_regions: ['producer'],
    included_queues: ['upstream', 'downstream'],
    capabilities: ['item_overlap'],
  },
}, plan).pass,
'a terminal cannot omit a required region');
ok(!verdict({
  step_role: 'terminal',
  target_topology: {
    launch_count: 3,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'an undeclared partial terminal cannot silently replace the full target');
ok(verdict({
  step_role: 'terminal',
  rung_deviation: 'measured partial fusion is intentionally evaluated first',
  target_topology: {
    launch_count: 3,
    included_regions: ['producer', 'consumer'],
  },
}, plan).pass,
'an explicit partial-terminal deviation remains legal');

console.log(failures ? `\nFAILED: ${failures}` :
  '\nPASS: Mega topology is an operator-neutral typed plan contract.');
process.exit(failures ? 1 : 0);
