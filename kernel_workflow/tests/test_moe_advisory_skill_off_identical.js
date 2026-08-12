#!/usr/bin/env node
// Static/behavioral regression guard for the optional analysis_skill prompt injection.
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const FILE = path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js');
const src = fs.readFileSync(FILE, 'utf8');
let failures = 0;
const ok = (condition, message) => {
  if (condition) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log(`\n# ${path.relative(ROOT, FILE)}`);

const gate = src.match(
  /const ANALYSIS_SKILL = [\s\S]*?if \(ANALYSIS_SKILL_ON\) log\([^\n]+\);/,
);
ok(!!gate, 'analysis-skill gating block found');
if (gate) {
  const make = new Function(
    'A',
    'WORKFLOW_DIR',
    'log',
    `${gate[0]}\nreturn { ANALYSIS_SKILL, ANALYSIS_SKILL_ON, ANALYSIS_SKILL_INPUTS };`,
  );
  const probe = (args) => make(args, '/wf', () => {});
  for (const args of [
    {},
    { analysis_skill: 'none' },
    { analysis_skill: 'false' },
    { analysis_skill: '' },
    { analysis_skill: '   ' },
  ]) {
    const result = probe(args);
    ok(result.ANALYSIS_SKILL_ON === false, `OFF for ${JSON.stringify(args)}`);
    ok(
      Object.keys(result.ANALYSIS_SKILL_INPUTS).length === 0,
      `OFF injects zero input keys for ${JSON.stringify(args)}`,
    );
  }
  const on = probe({ analysis_skill: 'moe_bottleneck' });
  ok(on.ANALYSIS_SKILL_ON === true, 'explicit safe skill enables analysis');
  ok(
    on.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR ===
      '/wf/knowledge/analysis_skills/moe_bottleneck',
    'ON resolves the expected skill directory',
  );
  for (const unsafe of ['../roles', 'a/b', '/tmp/x', 'UPPER']) {
    let threw = false;
    try { probe({ analysis_skill: unsafe }); } catch (_) { threw = true; }
    ok(threw, `unsafe skill slug rejected: ${unsafe}`);
  }
}

const blockSource = src.match(
  /function analysisSkillBlock[\s\S]*?\n}\n\nfunction roleAgent/,
);
ok(!!blockSource, 'dynamic analysisSkillBlock found');
if (blockSource) {
  const functionText = blockSource[0].replace(/\n\nfunction roleAgent$/, '');
  const makeBlock = new Function(
    'ANALYSIS_SKILL_ON',
    'ANALYSIS_SKILL_INPUTS',
    `${functionText}\nreturn analysisSkillBlock;`,
  );
  const off = makeBlock(false, {});
  ok(off('profile_engineer', 'baseline') === '', 'OFF adds no profile role text');
  ok(off('tech_lead', 'plan_round') === '', 'OFF adds no TechLead role text');
  const on = makeBlock(true, {
    ANALYSIS_SKILL: 'moe_bottleneck',
    ANALYSIS_SKILL_DIR: '/wf/knowledge/analysis_skills/moe_bottleneck',
  });
  ok(on('profile_engineer', 'baseline').includes('deterministic'), 'ON injects deterministic profile instructions');
  ok(on('tech_lead', 'plan_round').includes('Step-2 evidence'), 'ON injects analysis-to-planning boundary');
  ok(on('engineer', 'optimize') === '', 'unrelated roles receive no analysis text');
}

const sites = (src.match(/\.\.\.ANALYSIS_SKILL_INPUTS/g) || []).length;
ok(sites === 2, `inputs spread only at baseline/reprofile profile calls (${sites})`);
ok(
  /base \+ expertSkillsBlock\(role\) \+ analysisSkillBlock\(role, phase\)/.test(src),
  'roleAgent appends analysis text dynamically',
);
ok(
  /moe_analysis_json:\s*\{\s*type:\s*'string'\s*\}/.test(src),
  'PROFILE_SCHEMA explicitly allows moe_analysis_json',
);

for (const roleFile of ['profile_engineer.md', 'tech_lead.md']) {
  const text = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', roleFile), 'utf8');
  ok(
    !text.includes('ANALYSIS_SKILL') && !text.includes('moe_analysis_json'),
    `${roleFile} stays free of always-on analysis-skill prompt text`,
  );
}

console.log(
  failures === 0
    ? '\nPASS: OFF injects no inputs or role text; ON is safe-slugged and analysis-only.'
    : `\nFAILED: ${failures} assertion(s).`,
);
process.exit(failures === 0 ? 0 : 1);
