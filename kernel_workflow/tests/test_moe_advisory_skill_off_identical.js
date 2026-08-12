#!/usr/bin/env node
// Regression guard for the `analysis_skill` toggle in kernel_workflow.js (no GPU, no model needed).
// Mirrors e2e_workflow/scripts/test_analysis_skill_off_identical.js for the transplanted mechanism.
//
// Invariant under test: with the skill OFF (the default here, unlike e2e's default-ON 'roofline'),
// the profile-analysis feature injects NOTHING into any role prompt, so the run behaves exactly as it
// did before the feature existed. We prove this behaviorally by extracting the ACTUAL
// ANALYSIS_SKILL_* block from the workflow script and asserting that
//   (a) OFF (the default, and every off-spelling) yields empty strings for every input, and
//   (b) the object SHAPE is identical on and off (same keys), so a spread can never add or drop a key,
//   (c) the inputs are spread at every profile_engineer call site and collide with no other key.
//
// Run:  node kernel_workflow/tests/test_moe_advisory_skill_off_identical.js
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');            // .../GEAK
const FILE = path.join(ROOT, 'kernel_workflow', 'kernel_workflow.js');

let failures = 0;
const ok = (cond, msg) => { if (!cond) { console.error('  FAIL:', msg); failures++; } else console.log('  ok:', msg); };

console.log(`\n# ${path.relative(ROOT, FILE)}`);
const src = fs.readFileSync(FILE, 'utf8');

// 1) Extract the real gating block and probe it with controlled module-scope deps.
const m = src.match(/const ANALYSIS_SKILL = [\s\S]*?ANALYSIS_SKILL_DIR: '' \};/);
ok(!!m, 'ANALYSIS_SKILL_* gating block found');
if (m) {
  const make = new Function('A', 'WORKFLOW_DIR', 'log',
    m[0] + '\nreturn { ANALYSIS_SKILL, ANALYSIS_SKILL_ON, ANALYSIS_SKILL_INPUTS };');
  const probe = (a) => make(a, '/wf', () => {});

  // Default (no arg passed at all) must be OFF -- kernel_workflow serves many non-MoE kernels.
  const def = probe({});
  ok(def.ANALYSIS_SKILL === '' && def.ANALYSIS_SKILL_ON === false,
    'defaults to OFF (analysis_skill=none) -- unlike e2e_workflow\'s default-ON roofline');

  // OFF, by every spelling a caller might use
  for (const a of [{ analysis_skill: 'none' }, { analysis_skill: 'false' },
                   { analysis_skill: '' }, { analysis_skill: '   ' }]) {
    const r = probe(a);
    ok(r.ANALYSIS_SKILL_ON === false, `OFF for ${JSON.stringify(a)}`);
    ok(r.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL === '' && r.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '',
      `OFF -> every input is '' for ${JSON.stringify(a)}`);
  }

  // ON: explicit opt-in resolves under kernel_workflow's own knowledge/analysis_skills dir
  const on = probe({ analysis_skill: 'moe_bottleneck' });
  ok(on.ANALYSIS_SKILL === 'moe_bottleneck' && on.ANALYSIS_SKILL_ON === true, 'explicit opt-in turns it ON');
  ok(on.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '/wf/knowledge/analysis_skills/moe_bottleneck',
    'ON -> skill dir resolves under knowledge/analysis_skills/<skill>');
  const alt = probe({ analysis_skill: 'some-other-skill' });
  ok(alt.ANALYSIS_SKILL_INPUTS.ANALYSIS_SKILL_DIR === '/wf/knowledge/analysis_skills/some-other-skill',
    'ON -> an arbitrary skill name is pluggable (dir swap only)');

  // Shape stability: same keys on and off, so a spread never adds/removes a key.
  const keysOn = Object.keys(on.ANALYSIS_SKILL_INPUTS).sort().join(',');
  const keysOff = Object.keys(def.ANALYSIS_SKILL_INPUTS).sort().join(',');
  ok(keysOn === keysOff, `input object shape identical ON vs OFF (${keysOn})`);
}

// 2) The spread must be additive at every call site: the ANALYSIS_SKILL_* keys are set nowhere else,
//    so spreading them can never shadow an existing input.
const sites = (src.match(/\.\.\.ANALYSIS_SKILL_INPUTS/g) || []).length;
ok(sites === 2, `spread into both profile_engineer call sites (found ${sites}, expected 2: baseline + reprofile)`);
if (m) {
  const outside = src.replace(m[0], '');
  const stray = (outside.match(/ANALYSIS_SKILL(_DIR)?\s*:/g) || []).length;
  ok(stray === 0,
    `ANALYSIS_SKILL/_DIR used as an object key ONLY inside the gating block (found ${stray} elsewhere) ` +
    `-> the spread can never shadow another input`);
}

// 3) The gating block itself must not collide with the pre-existing, differently-shaped
//    expert_skills mechanism (USE_EXPERT_SKILLS / EXPERT_SKILLS_DIR / expertSkillsBlock).
ok(/USE_EXPERT_SKILLS/.test(src) && /ANALYSIS_SKILL_ON/.test(src),
  'both the pre-existing expert_skills toggle and the new analysis_skill toggle coexist');
ok(!/EXPERT_SKILL_ROLES.*ANALYSIS_SKILL|ANALYSIS_SKILL.*EXPERT_SKILL_ROLES/.test(src),
  'the two advisory mechanisms remain independent (no shared gating variable)');

// 4) Consumers must treat the prior as optional and advisory.
ok(/ADVISORY|advisory/.test(src), 'gating block documents the prior as advisory');
const profileEngineerSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', 'profile_engineer.md'), 'utf8');
ok(profileEngineerSrc.includes('ANALYSIS_SKILL_DIR'), 'roles/profile_engineer.md consumes ANALYSIS_SKILL_DIR');
ok(/non-empty|EXISTS|else skip|otherwise skip/i.test(profileEngineerSrc),
  'roles/profile_engineer.md guards on the prior being present');
const techLeadSrc = fs.readFileSync(path.join(ROOT, 'kernel_workflow', 'roles', 'tech_lead.md'), 'utf8');
ok(techLeadSrc.includes('moe_advisory_json'), 'roles/tech_lead.md consults moe_advisory_json (advisory, additive)');

console.log(failures === 0
  ? '\nPASS: analysis_skill OFF injects nothing (feature is purely additive); default is OFF.'
  : `\nFAILED: ${failures} assertion(s).`);
process.exit(failures === 0 ? 0 : 1);
