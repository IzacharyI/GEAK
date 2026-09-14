#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

const task = fs.readFileSync(path.resolve(
  __dirname, '..', 'tasks', 'megamoe_v2_ep8_mega', 'GEAK_TASK.md',
), 'utf8');

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# Mega task prompt is stable and run-independent');
ok(/Use \$\{SKILL_DIR\} to optimize the MegaMoE V2 megakernel/.test(task) &&
   /EP8, 8× MI355X/.test(task) && /mode=mega/.test(task),
  'the fixed prompt identifies workflow, operator, hardware and mode');
ok(/one persistent kernel per rank/.test(task) &&
   /launches=2/.test(task) &&
   /GEMM1, GEMM2[\s\S]*combine with real overlap/.test(task),
  'the fixed prompt states the complete fusion goal');
ok(/--bs-list 128,512,8192/.test(task) &&
   /--accuracy-max-bs 8192/.test(task) &&
   /--rtol 0\.10/.test(task) &&
   /path=MEGA/.test(task) &&
   /SCATTERED fallback does not count/.test(task),
  'the fixed prompt carries the correctness command and activation rule');
ok(/--tokens 8192/.test(task) &&
   /--route uniform/.test(task) &&
   /--iters 10/.test(task) &&
   /--mega-only/.test(task) &&
   /rank-max/.test(task),
  'the fixed prompt carries the authoritative performance command');
ok(/Output `exp_root`: `\$\{EXP_ROOT\}`/.test(task),
  'only bootstrap resolves the output root');
ok(!/resume|STATE\.json|next_blocker|contract|75\/75|26\/75|0515cb97|wf_[a-z0-9]/i.test(task),
  'run state, contract feedback and attempt ids never enter the fixed task');
ok(task.length < 2000,
  'the canonical task remains compact enough to pass verbatim');

console.log(failures === 0
  ? '\nPASS: Mega uses one stable task prompt; changing state does not change the goal.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
