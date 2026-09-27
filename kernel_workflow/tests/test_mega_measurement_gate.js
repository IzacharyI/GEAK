#!/usr/bin/env node
// Generic-mega measurement inputs must be empty for Expert-Skill runs and every non-mega mode:
// a skill-on mega wave has to replay with byte-identical prompts after this line was added.
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const def = src.match(/const GMI = ([\s\S]*?);\n/);
ok(!!def, 'GMI is defined');
const gmi = (MODE, USE_EXPERT_SKILLS) =>
  // eslint-disable-next-line no-new-func
  new Function('MODE', 'USE_EXPERT_SKILLS', 'WORKFLOW_DIR', `return ${def[1]};`)(
    MODE, USE_EXPERT_SKILLS, '/wf');
ok(Object.keys(gmi('mega', true)).length === 0, 'skill-on mega gets no measurement inputs');
ok(Object.keys(gmi('optimize', false)).length === 0, 'non-mega gets no measurement inputs');
ok(gmi('mega', false).MEGA_MEASUREMENT_GUIDE === '/wf/knowledge/mega_measurement.md',
  'generic mega gets the guide');

// The generic graph-contract runner is the default only for generic mega; a pinned skill keeps its
// runtime, and an explicit graph_contract_tool arg still wins.
const gct = src.match(/const GRAPH_CONTRACT_TOOL = ([\s\S]*?);\n/);
ok(!!gct, 'GRAPH_CONTRACT_TOOL is defined');
const tool = (MODE, USE_EXPERT_SKILLS, PINNED_MEGA_SKILL, A = {}) =>
  // eslint-disable-next-line no-new-func
  new Function('MODE', 'USE_EXPERT_SKILLS', 'PINNED_MEGA_SKILL', 'A', 'WORKFLOW_DIR',
    `return ${gct[1]};`)(MODE, USE_EXPERT_SKILLS, PINNED_MEGA_SKILL, A, '/wf');
ok(tool('mega', true, true) === '/wf/tools/expert_skill_runtime.py', 'pinned skill keeps its runtime');
ok(tool('mega', true, false) === '', 'unpinned skill-on mega keeps an empty tool');
ok(tool('optimize', false, false) === '', 'non-mega keeps an empty tool');
ok(tool('mega', false, false) === '/wf/tools/generic_graph_contract.py', 'generic mega gets the runner');
ok(tool('mega', false, false, { graph_contract_tool: '/x.py' }) === '/x.py', 'explicit arg still wins');

// Every use is either a spread (adds nothing when {}) or the guard of the generic-only Profile call.
const uses = src.split('\n').filter((l) => /\bGMI\b/.test(l) && !/const GMI =/.test(l));
// profile guard + profile, analysis, benchmark setup, engineer, verify, finalist spreads.
ok(uses.length === 7, `GMI used at 7 sites (got ${uses.length})`);
ok(uses.every((l) => /\.\.\.GMI\b/.test(l) || /if \(GMI\.MEGA_MEASUREMENT_GUIDE && /.test(l)),
  'every GMI use is a spread or the generic-only guard');
ok(/profileSummary = \{ \.\.\.profileSummary, \.\.\.mp \}/.test(src) &&
  !/mp\.__agent_timed_out/.test(src), 'a failed baseline measurement keeps the stub, never ends the wave');

for (const f of ['knowledge/mega_measurement.md',
  'scripts/multi_rank_analysis/reduce_stage_meter.py', 'tools/generic_graph_contract.py']) {
  ok(fs.existsSync(path.join(WF, f)), `${f} exists`);
}
const guide = fs.readFileSync(path.join(WF, 'knowledge/mega_measurement.md'), 'utf8');
ok(/GEAK_MEGA_STAGE_METER=1/.test(guide) && /never produces a scored latency/.test(guide),
  'guide fixes the meter switch and keeps the meter out of scoring');
ok(/graph_contract_adapter\.py/.test(guide) && /replaces\*\* the role's `GRAPH_CONTRACT_TOOL` command/.test(guide),
  'guide defines the adapter and the generic claim command');

if (failures) { console.error(`FAIL: ${failures} assertion(s) failed.`); process.exit(1); }
console.log('PASS: generic-mega inputs and graph-contract runner are gated off for skill-on and non-mega runs.');
