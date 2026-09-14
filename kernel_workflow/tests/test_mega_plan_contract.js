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
const verdict = new Function(`${match[1]}\nreturn megaTopologyVerdict;`)();

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
