#!/usr/bin/env node
// mega_staged_authoring: a default-OFF authorization that fixes the validated_skill lane's
// scaffold-and-bail (2-min turns that emit a structured result far under the lease with the body never
// authored). When ON, ONLY the mega_engineer (validated_skill) lane gets a "STAGED-AUTHORING AUTHORIZED"
// block appended to its prompt: forbid scaffold-and-bail, author the next single stage, build+run on-card
// before emitting, keep intermediates as authoring fault-localizers (never terminal).
//
// Properties pinned, because each is a way the feature could silently go wrong:
//   1. DEFAULT-OFF, BYTE-IDENTICAL. When off the injected block is '' and the prompt is unchanged, so no
//      existing wave (or its resume cache key, keyed on the prompt) shifts under a default flag.
//   2. SCOPED TO THE SKILL LANE. The block is gated on source==='validated_skill'; the search lane
//      (role=engineer) never sees it — mirrors the STABILITY-EDIT precedent.
//   3. THE DISCIPLINE IS THE ONE THAT MADE THE SMOKE SUCCEED. The block must forbid scaffold-and-bail,
//      require an on-card build+run before emit, and keep staged intermediates non-terminal.
'use strict';

const fs = require('fs');
const path = require('path');
const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_workflow.js'), 'utf8');
const role = fs.readFileSync(path.join(WF, 'roles', 'mega_engineer.md'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

console.log('\n# 1. the flag is default-off (byte-identical when absent)');
ok(/const MEGA_STAGED_AUTHORING = String\(A\.mega_staged_authoring != null \? A\.mega_staged_authoring : 'false'\) === 'true';/.test(src),
   'MEGA_STAGED_AUTHORING defaults to false');

console.log('\n# 2. the injected block is gated on the flag AND the validated_skill lane, else empty string');
ok(/const stagedAuthoringBlock = \(MEGA_STAGED_AUTHORING && source === 'validated_skill'\)\s*\n\s*\? `STAGED-AUTHORING AUTHORIZED:/.test(src),
   'stagedAuthoringBlock is non-empty only when the flag is on and the lane is validated_skill');
ok(/`Only a real on-card \[RESULT\] counts; never narrate an unmeasured launches=2\. `\s*\n\s*: '';/.test(src),
   'the block collapses to the empty string in every other case (search lanes + flag off)');

console.log('\n# 3. the block is actually appended to the mega_engineer prompt');
ok(/stabilityEditBlock \+\s*\n\s*stagedAuthoringBlock \+/.test(src),
   'stagedAuthoringBlock is concatenated into the Optimize prompt right after stabilityEditBlock');

console.log('\n# 4. the injected discipline is the smoke-winning one (forbid bail, on-card build+run, non-terminal stages)');
ok(/do NOT scaffold-and-bail/.test(src) && /NOT emit your structured result with only scaffolding/.test(src),
   'the block forbids scaffold-and-bail (emitting with only scaffolding as progress)');
ok(/BUILD and RUN the current stage on-card/.test(src) && /never narrate an unmeasured launches=2/.test(src),
   'the block requires an on-card build+run before emit and forbids narrating an unmeasured launches=2');
ok(/dispatch -> \+GEMM1 -> \+flat GEMM2/.test(src) && /combine as the 3rd ticketed queue/.test(src) &&
   /ret=None launch drop \(launches==2\)/.test(src),
   'the block names the fixed staged order ending in the ret=None launch drop');
ok(/intermediates stay candidate_status:"authoring"[\s\S]{0,120}NEVER[\s\S]{0,120}terminal topology/.test(src),
   'staged intermediates are kept non-terminal (the no-intermediate-terminal rule is preserved)');

console.log('\n# 5. the role .md documents the gated discipline (literal-phrase trigger, mirrors STABILITY-EDIT)');
ok(/## Staged authoring \(only when the prompt carries `STAGED-AUTHORING AUTHORIZED`\)/.test(role),
   'mega_engineer.md has a Staged-authoring section gated on the same literal phrase the workflow injects');
ok(/No scaffold-and-bail/.test(role) && /Build \+ run on-card before you emit/.test(role) &&
   /Staging is a fault-localizer path, not a deliverable/.test(role),
   'the role section states: no bail, on-card build+run before emit, staging is non-terminal');

console.log('\n# 6. scoped to mega_engineer only — the search lane role file is untouched by this feature');
ok(!/STAGED-AUTHORING AUTHORIZED/.test(fs.readFileSync(path.join(WF, 'roles', 'engineer.md'), 'utf8')),
   'engineer.md (the search lane role) contains no staged-authoring authorization');

console.log(failures
  ? `\nFAILED: ${failures} assertion(s).`
  : '\nPASS: mega_staged_authoring is default-off byte-identical, scoped to the validated_skill lane, and encodes the smoke-winning discipline.');
process.exit(failures ? 1 : 0);
