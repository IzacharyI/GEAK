#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');
const src = fs.readFileSync(path.resolve(__dirname, '..', 'kernel_workflow.js'), 'utf8');
const launch = JSON.parse(fs.readFileSync(
  path.resolve(__dirname, '..', 'tasks', 'megamoe_v2_ep8_mega', 'launch_args.json'), 'utf8',
));

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# force-emitted control claims are pending, never zero-valued failures');
ok(/const pcComplete = MODE !== 'mega' \|\| pc\.claim_complete === true/.test(src),
  'Mega requires an atomic completed positive-control claim');
ok(/Positive control PENDING: claim_complete is not true/.test(src),
  'the incomplete state is surfaced explicitly');
ok(/candidate authoring may continue, but mega performance ranking/i.test(src),
  'an incomplete control blocks ranking rather than candidate authoring');

console.log('\n# the null-only bypass is gone');
ok(!Object.prototype.hasOwnProperty.call(launch.positive_control, 'measure_null_only'),
  'Mega config does not pretend a null arm proves sensitivity');
ok(!/measure_null_only/.test(src),
  'the orchestrator no longer has a null-only positive-control bypass');
ok(launch.positive_control.abort_on_fail === false,
  'a control problem preserves authoring progress instead of aborting the whole portfolio');

console.log('\n# scores remain disabled until calibration passes');
ok(/megaMeasurementCalibration\.ready\s*\?\s*selectMegaCandidate/.test(src),
  'candidate ranking is conditional on completed calibration');
ok(/if \(!batch\.length \|\| !megaMeasurementCalibration\.ready\) return null/.test(src),
  'finalist selection cannot run on an uncalibrated harness');

console.log('\n# verification is tiered');
const verifyRole = fs.readFileSync(path.resolve(__dirname, '..', 'roles', 'verify_engineer.md'), 'utf8');
ok(/`score`:[\s\S]*8192_uniform[\s\S]*short liveness/.test(verifyRole),
  'score tier is the cheap target-only admission measurement');
ok(/`finalist`:[\s\S]*all target\/regression guards[\s\S]*overlap\/attribution/.test(verifyRole),
  'expensive full evidence is reserved for finalists');

console.log(failures === 0
  ? '\nPASS: incomplete measurements preserve work but cannot rank or ship a candidate.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
