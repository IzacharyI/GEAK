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

ok(!verdict({}, { target_launches: 2 }).pass,
  'a Mega direction cannot omit target_topology');
ok(!verdict({
  topology_inferred: true,
  target_topology: { launches: 2, fused_stages: ['gemm1', 'gemm2', 'combine'] },
}, { target_launches: 2 }).pass,
'a compatibility fallback does not satisfy a strict planner contract');
ok(verdict({
  step_role: 'terminal',
  target_topology: {
    launches: 2,
    fused_stages: ['dispatch', 'gemm1', 'gemm2', 'p2p', 'combine'],
    combine_mode: 'queue',
  },
}, {
  target_launches: 2,
  resource_contract: { num_waves: 8 },
  schedule_contract: { work_shards: 4 },
}).pass,
'the terminal topology matching MegaPlanIR passes');
ok(!verdict({
  step_role: 'terminal',
  target_topology: { launches: 3, fused_stages: ['gemm1', 'gemm2'] },
}, { target_launches: 2 }).pass,
'an undeclared partial terminal cannot silently replace the full target');
ok(verdict({
  step_role: 'terminal',
  rung_deviation: 'measured partial fusion is intentionally evaluated first',
  target_topology: { launches: 3, fused_stages: ['gemm1', 'gemm2'] },
}, { target_launches: 2 }).pass,
'an explicit partial-terminal deviation remains legal');

console.log(failures ? `\nFAILED: ${failures}` : '\nPASS: Mega topology is a typed plan contract.');
process.exit(failures ? 1 : 0);
