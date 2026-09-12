#!/usr/bin/env node
// Guards the mega-lane ACTIVATION-SWITCH channel — the fix for "authored concurrency never speeds up".
//
// What happened (cont17 r3:a50): the m25_skill lane correctly authored SITE-3 fine-ready
// (atomic_add_agent / int32_wait_until_greater) behind a DEFAULT-OFF env flag AITER_MEGAMOE_FINE_READY,
// launches=2 + relL2=0.0298 correct — yet the paired A/B cand arm measured 10.5ms (the serial floor)
// because the flag was never exported in the measured build. Root cause: the mega engineer returns
// MEGA_CANDIDATE_SCHEMA (topology_sig only), the inherited ENG_SCHEMA.activation single-lever A/B switch
// channel was never wired for mega, and the mega verify call never threaded ACTIVATION. So a concurrency
// lever gated behind a default-off flag was always measured OFF → serial floor, forever.
//
// The fix reuses the proven activation machinery (verify_engineer.md step 4d exports switch_name=
// switch_value for the CAND arm only): MEGA_CANDIDATE_SCHEMA gains an optional `activation`, the mega
// engineer prompt gains a gated activationSwitchBlock instructing the lane to declare it, and the mega
// verify call threads ACTIVATION — all gated by MEGA_TOPOLOGY_LEVERS so default-off is byte-identical.

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const read = (p) => fs.readFileSync(path.join(ROOT, p), 'utf8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

const wf = read('kernel_workflow.js');
const megaEng = read('roles/mega_engineer.md');
const verify = read('roles/verify_engineer.md');

// --- 1. MEGA_CANDIDATE_SCHEMA carries an optional activation switch ----------
console.log('\n# schema: mega candidate can declare an activation switch');
const megaSchema = wf.slice(wf.indexOf('const MEGA_CANDIDATE_SCHEMA = obj('),
                            wf.indexOf('const MEGA_TOPOLOGY_SCHEMA'));
ok(/activation: obj\(\{/.test(megaSchema),
   'MEGA_CANDIDATE_SCHEMA carries an activation object');
ok(/switch_name: \{ type: 'string' \}, switch_value/.test(megaSchema),
   'the mega activation declaration has switch_name + switch_value');
ok(/\}, \[\]\)/.test(megaSchema.slice(megaSchema.indexOf('activation: obj('))),
   'the activation object is OPTIONAL (empty required list) — absence is byte-identical');
ok(/switches: \{\s*\n?\s*type: 'array'/.test(megaSchema) &&
   /items: obj\(\{ switch_name.*switch_value/.test(megaSchema),
   'the activation object carries a COUPLED switches[] array (multi-lever, one deep lease)');

// --- 2. the engineer prompt block is gated + only for the skill lane --------
console.log('\n# prompt: activationSwitchBlock gated by MEGA_TOPOLOGY_LEVERS, skill-lane only');
ok(/const activationSwitchBlock = \(MEGA_TOPOLOGY_LEVERS && source === 'validated_skill'\)/.test(wf),
   'activationSwitchBlock is gated by MEGA_TOPOLOGY_LEVERS AND the validated_skill source');
ok(/const activationSwitchBlock[\s\S]{0,3600}?: '';/.test(wf),
   'activationSwitchBlock falls back to empty string when the gate is off');
ok(/stagedAuthoringBlock \+\s*\n\s*activationSwitchBlock \+/.test(wf),
   'activationSwitchBlock is concatenated into the mega engineer prompt');
ok(/AITER_MEGAMOE_FINE_READY/.test(wf) && /mode:"switch"/.test(wf),
   'the block names a concrete default-off env flag and the switch mode');
// COUPLED contract: the block must instruct authoring the sites TOGETHER and declaring the whole set.
const blockSlice = wf.slice(wf.indexOf('const activationSwitchBlock'),
                            wf.indexOf('const existing = megaCandidateById'));
ok(/COUPLED/.test(blockSlice) && /switches:\[/.test(blockSlice),
   'activationSwitchBlock instructs the COUPLED form (switches[]), not one switch per turn');
ok(/AITER_MEGAMOE_ROLE_PARTITION/.test(blockSlice) &&
   /AITER_MEGAMOE_FINE_READY/.test(blockSlice) &&
   /SITE-2|PIPELINE_DEPTH|pipeline/.test(blockSlice),
   'the block names all three coupled sites (SITE-1 dynamic / SITE-3 fine / SITE-2 pipeline)');
ok(/PHASE METER|phase-meter|phase meter/i.test(blockSlice),
   'the block rules the build with the in-kernel phase meter (not per-stage roofline)');

// --- 3. the mega verify call threads ACTIVATION, gated the same way ---------
console.log('\n# threaded: mega verify receives the activation declaration when levers are on');
ok(/MEGA_TOPOLOGY_LEVERS\s*\n?\s*\?\s*\{ ACTIVATION: \(eng && eng\.activation\)/.test(wf),
   'the mega verify call threads eng.activation as ACTIVATION, gated by MEGA_TOPOLOGY_LEVERS');
ok(/: 'UNDECLARED'/.test(wf.slice(wf.indexOf('mega:verify') - 2000, wf.indexOf('mega:verify') + 200)) ||
   /ACTIVATION: \(eng && eng\.activation\) \? JSON\.stringify\(eng\.activation\) : 'UNDECLARED'/.test(wf),
   'absent activation threads as UNDECLARED (verify establishes it itself, same as inherited path)');

// --- 4. byte-identical when MEGA_TOPOLOGY_LEVERS is off ----------------------
console.log('\n# byte-identical guard: default (levers off) adds nothing to prompt/inputs');
// The block is `(GATE) ? '...' : ''` and the input is `...(GATE ? {..} : {})` — both evaluate to a
// no-op (empty string / empty spread) when the gate is false, so the prompt string and the verify
// input object are unchanged from before the fix. Assert the two no-op shapes are present.
ok(/activationSwitchBlock = \(MEGA_TOPOLOGY_LEVERS[\s\S]*?\n\s*: '';/.test(wf),
   'levers off → activationSwitchBlock === "" (empty concat, no prompt change)');
ok(/\.\.\.\(MEGA_TOPOLOGY_LEVERS\s*\n?\s*\? \{ ACTIVATION[\s\S]*?\n?\s*: \{\}\)/.test(wf),
   'levers off → ACTIVATION spread is {} (no verify input change)');

// --- 5. roles document the contract ----------------------------------------
console.log('\n# roles: engineer declares it, verify honors it');
ok(/activation.*mode.*switch.*switches.*AITER_MEGAMOE_FINE_READY/s.test(megaEng) &&
   /DECLARE ALL ITS SWITCHES/.test(megaEng),
   'mega_engineer.md instructs authoring the COUPLED grid and declaring ALL its switches');
ok(/COUPLED/.test(megaEng) && /AITER_MEGAMOE_ROLE_PARTITION/.test(megaEng) &&
   /PIPELINE_DEPTH/.test(megaEng) && /PHASE METER|phase meter/i.test(megaEng),
   'mega_engineer.md names the coupled sites + rules the build with the phase meter');
ok(/serial floor/i.test(megaEng),
   'mega_engineer.md explains an undeclared switch measures as the serial floor');
ok(/CANDIDATE arm ONLY/.test(verify) && /leave the base arm serial/.test(verify),
   'verify_engineer.md honors mode:switch — cand arm only, base arm serial');
ok(/switches: \[\{switch_name,switch_value\}/.test(verify) &&
   /export \*\*every\*\*|export the WHOLE set|exporting a subset/i.test(verify),
   'verify_engineer.md 4d exports the WHOLE coupled switches[] set for the cand arm');

console.log(`\n${failures ? 'FAILED ' + failures : 'PASSED'} — test_mega_activation_switch`);
process.exit(failures ? 1 : 0);
