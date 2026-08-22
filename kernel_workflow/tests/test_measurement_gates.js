#!/usr/bin/env node
// Guards for the three checks added after a run reported 1.000x on a tree that was missing an
// already-measured +4.71% win, with nothing in the workflow able to tell "found nothing" apart from
// "cannot see anything":
//
//   1. POSITIVE CONTROL  — a known-effect change run during Benchmark, before any budget is spent.
//                          Calibrates the DETECTION floor (the null arm only calibrates noise).
//   2. PRIOR ART         — directions already implemented in/beside the tree, surfaced at Analyze.
//   3. PROVENANCE        — reps + null_arm_pct on every claimed win, marking thin ones PROVISIONAL.
//
// These are prose-plus-schema changes across four files, which is exactly the kind of thing that
// gets reflowed away. The assertions below are the ratchet.

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
const bench = read('roles/benchmark_engineer.md');
const lead = read('roles/tech_lead.md');
const verify = read('roles/verify_engineer.md');

// --- 1. positive control ---------------------------------------------------
console.log('\n# positive control');
ok(/const POSITIVE_CONTROL = \(A\.positive_control/.test(wf),
   'kernel_workflow.js reads args.positive_control');
ok(/positive_control: \{ type: 'object'/.test(wf),
   'BENCH_SCHEMA carries a positive_control result object');
// The gate must sit between Benchmark and the optimize loop. If it ran later, the budget it exists
// to protect would already be spent.
const iBench = wf.indexOf("phase('Benchmark')");
const iGate = wf.indexOf('Positive control FAILED');
const iLoop = wf.indexOf('while (dispatched < BUDGET');
ok(iBench > 0 && iGate > iBench && iLoop > iGate,
   'the gate fires after Benchmark and before the optimization loop');
ok(/if \(PC_ABORT\) throw new Error\(why\)/.test(wf),
   'a failed control aborts by default rather than warning');
ok(/abort_on_fail !== false/.test(wf),
   'abort is opt-out, not opt-in');
ok(/unfalsifiable/.test(wf),
   'the code states WHY a missing control makes 1.000x unfalsifiable');
ok(/\+4\.71%/.test(wf), 'the originating incident keeps its number in the source');

// The benchmark engineer must be told to interleave the control like a real candidate, with a null
// arm — a control measured more loosely than the thing it certifies proves nothing.
ok(/POSITIVE CONTROL/.test(bench), 'benchmark_engineer has a positive-control step');
ok(/≥5 pairs/.test(bench), 'the control requires >=5 interleaved pairs');
ok(/null arm/i.test(bench), 'the control requires a null arm alongside it');
ok(/do not quietly retry until it passes/i.test(bench),
   'the engineer is barred from retrying the control until it passes');
ok(/positive = faster/.test(bench), 'the sign convention for measured_pct is pinned down');

// --- 2. prior art ----------------------------------------------------------
console.log('\n# prior art');
ok(/prior_art: \{ type: 'array'/.test(wf), 'ANALYZE_SCHEMA carries prior_art');
ok(/PRIOR ART NOT IN BASELINE/.test(wf),
   'absent-from-baseline prior art gets its own distinct, louder log line');
ok(/pointed at the wrong target/.test(wf),
   'the log names the real failure mode: optimizing the wrong tree');
ok(/Measure it, do not re-derive it/.test(wf),
   'present-but-off prior art is routed to measurement, not engineering');
ok(/Search for PRIOR ART before you plan anything/i.test(lead),
   'tech_lead is told to search for prior art at analyze time');
// NB: this file wraps at ~100 cols; match across a hard wrap. A plain-space regex here has broken
// silently before when a paragraph was re-flowed.
ok(/~1000 lines of already-written, already-measured/.test(lead),
   'the tech_lead rule carries the incident that motivated it');
// Wording tightened after the key was omitted outright and the distinction had to be enforced in
// code as well as prose — see tests/test_prior_art_provenance.js.
ok(/`\[\]` is a real answer that is not the same as omitting it/.test(lead),
   'the [] vs omitted distinction is spelled out so "no prior art" is a claim, not a silence');

// --- 3. result provenance --------------------------------------------------
console.log('\n# result provenance');
ok(/reps: \{ type: 'number' \}, null_arm_pct: \{ type: 'number' \}/.test(wf),
   'VERIFY_SCHEMA carries reps and null_arm_pct');
ok(/r\.provisional = reasons\.join/.test(wf), 'thin results are marked provisional');
ok(/PROVISIONAL \$\{r\.d\.id\}/.test(wf), 'provisional results are logged per direction');
// The point of PROVISIONAL is that it does NOT delete the result. Dropping under-repped wins would
// re-create the exact failure this was built to catch (a real win discarded on weak evidence).
ok(/does NOT reject the result/.test(wf),
   'the code states that provisional marking is not rejection');
ok(/claim \$\{claimPct\.toFixed\(2\)\}% <= null arm/.test(wf),
   'a claim at or below the null arm is flagged');
ok(/Under-repped wins are \*\*not discarded\*\*/.test(verify),
   'verify_engineer is told thin results survive as provisional');
ok(/reps` is the count of interleaved\s+A,B pairs behind the median/.test(verify),
   'reps is defined as pairs, not process launches');

// --- 4. it has to reach the report ----------------------------------------
console.log('\n# the report must carry it');
ok(/Measurement confidence/.test(lead), 'the report has a measurement-confidence section');
ok(/1\.000x with no positive control is not a\s+finding/.test(lead),
   'the report is required to disown an uncalibrated zero');
ok(/required even \(especially\) when the final speedup is\s+1\.000x/.test(lead),
   'the section is mandatory precisely in the case that produced this fix');

// --- 5. the gate's ARITHMETIC, exercised on the real source ----------------
// Everything above only proves the text is present. This lifts the actual in-band expression out of
// kernel_workflow.js and runs it, so a subtly wrong comparison (>= vs >, NaN slipping through as
// "pass") fails here instead of in production. Extracting rather than re-typing is deliberate: a
// hand-copied duplicate of the logic drifts and then tests itself.
console.log('\n# gate arithmetic (extracted from source, not re-implemented)');
const gateSrc = wf.match(
  /const ABSURD = ([\s\S]*?)\n  const ok = ran && !tooSmall && !absurd && \(!overshoot \|\| nullQuiet\);/,
);
ok(!!gateSrc, 'the decision expression can be located in kernel_workflow.js');
if (gateSrc) {
  const decide = new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
    const ABSURD = ${gateSrc[1]}
    const ok = ran && !tooSmall && !absurd && (!overshoot || nullQuiet);
    return { ok, overshoot };
  `);
  const band = { expected_pct_lo: 3.55, expected_pct_hi: 4.93 };  // the real +4.71% guard
  const quiet = 0.38;   // the null arm actually observed alongside the +5.13% reading
  const cases = [
    [{ ran: true, measured_pct: 4.71, null_arm_pct: quiet }, true, 'the known +4.71% passes'],
    [{ ran: true, measured_pct: 3.55, null_arm_pct: quiet }, true, 'the lower bound is inclusive'],
    [{ ran: true, measured_pct: 4.93, null_arm_pct: quiet }, true, 'the upper bound is inclusive'],
    // The incident: a real reproduction 0.20pp over the ceiling with a quiet null arm. The whole
    // point of the asymmetry is that this must NOT kill the run.
    [{ ran: true, measured_pct: 5.13, null_arm_pct: quiet }, true,
     'a 0.20pp overshoot with a quiet null arm PASSES (the incident that motivated the split)'],
    [{ ran: true, measured_pct: 5.13, null_arm_pct: 2.9 }, false,
     'the same overshoot with a LOUD null arm fails — that excess is drift, not sensitivity'],
    [{ ran: true, measured_pct: 5.13 }, false,
     'an overshoot with an UNREPORTED null arm fails: nothing rules out drift'],
    [{ ran: true, measured_pct: 0.4, null_arm_pct: quiet }, false,
     'a harness that sees almost nothing still fails — the lower bound is untouched'],
    [{ ran: true, measured_pct: -4.71, null_arm_pct: quiet }, false,
     'the right magnitude with the WRONG SIGN fails'],
    [{ ran: true, measured_pct: 40.0, null_arm_pct: quiet }, false,
     'an absurd delta still fails: past 2x the ceiling it is not this change being measured'],
    [{ ran: true, measured_pct: 9.86, null_arm_pct: quiet }, true,
     'exactly at the plausibility ceiling is still a pass (absurd is strictly above)'],
    [{ ran: false }, false, 'a control that did not run fails'],
    [{ ran: true }, false, 'a missing measurement is not silently a pass'],
    [{ ran: true, measured_pct: 'n/a' }, false, 'an unparseable measurement is not a pass'],
    [{ ran: true, measured_pct: NaN }, false, 'NaN is not a pass'],
    [{}, false, 'an empty result object is not a pass'],
  ];
  for (const [pc, want, msg] of cases) ok(decide(pc, band).ok === want, msg);
  // An explicit ceiling overrides the 2*hi default.
  ok(decide({ ran: true, measured_pct: 6.0, null_arm_pct: quiet },
            { ...band, implausible_pct: 5.5 }).ok === false,
     'an explicit implausible_pct overrides the 2x default');
  // A pass that overshot must be DISTINGUISHABLE from a clean pass, or the caveat cannot travel.
  ok(decide({ ran: true, measured_pct: 5.13, null_arm_pct: quiet }, band).overshoot === true,
     'an overshooting pass is flagged as such, so the report can carry the scale caveat');
  ok(decide({ ran: true, measured_pct: 4.71, null_arm_pct: quiet }, band).overshoot === false,
     'a clean pass is not flagged as an overshoot');
}

// A per-case noise floor that only lives in an insight is decoration. The aggregate is where an
// unreadable number turns into a claim, so the rule has to bind at aggregation time.
console.log('\n# unresolved cases must not be aggregated at face value');
ok(/CONTRIBUTES 1\.000 TO THE HEADLINE/.test(lead),
   'a case inside its own same-arm spread folds in as 1.000, not as its point estimate');
ok(/UNRESOLVED \(delta, n, wins\/n, spread\)/.test(lead),
   'the table prints the evidence that made the case unresolved, not just the verdict');
ok(/same-arm spread/.test(lead) && /sign test/.test(lead),
   'both readability tests are named: magnitude vs spread AND the paired sign test');
// Without the incident the rule reads as pedantry and gets skipped under budget pressure.
ok(/1\.01769x/.test(lead) && /-4\.14%/.test(lead) && /1\.0021x/.test(lead),
   'the 1.56pp inflation incident keeps its three numbers in the role file');

// A measurement that exists on disk but not in the return value is still a measurement. The workflow
// cannot read files, so it has to ASK — and the recovery must be recovery, not a silent re-run.
console.log('\n# an unreturned baseline is recovered from disk, not thrown away');
ok(/label: 'benchmark_recover'/.test(wf), 'a recovery agent runs before the phase aborts');
ok(/no baseline recorded, and none recoverable from EVAL_DIR/.test(wf),
   'the abort still exists — recovery narrows the failure, it does not remove it');
ok(/Do NOT run any GPU command and do/.test(wf) && /a fresh measurement here is a/.test(wf),
   'the recovery agent is forbidden from re-measuring, which would hide the cost of the failure');
ok(/claim_complete:true IS the positive/.test(wf),
   'a completed positive control on disk is reused rather than re-run — it is the phase\'s costliest step');
ok(/70 min of an 8-card lease/.test(wf),
   'the incident keeps its cost in the source, which is the argument for the extra agent');
// The role side: persist first, then do the expensive interruptible things.
ok(/PERSIST THE BASELINE THE MOMENT IT EXISTS/.test(bench),
   'the engineer writes the baseline before the control and correctness, not after');
ok(/READ THEM AND REUSE THEM/.test(bench),
   'an interrupted attempt\'s artifacts are inputs to the retry, not litter');


// Same failure class as the unreturned baseline, one level down: a direction that measured and then
// died before writing its result is recorded as a zero. Both happened in the same wave.
console.log('\n# a killed engineer must leave a partial result, not a silence');
const eng = read('roles/engineer.md');
ok(/REWRITE IT AFTER EVERY UNIT OF EVIDENCE/.test(eng),
   'the result file is refreshed per rung/arm, not written once at the end');
ok(/before your first long-running\s*\n?command/.test(eng),
   'the first write happens before the first thing that can kill the agent');
ok(/15-rung hardware bisection/.test(eng) && /scored \*\*zero\*\*/.test(eng),
   'the incident keeps its cost: a completed bisection recorded as no result');
ok(/is a \*\*zero\*\*, and the difference is entirely in when you wrote the file/.test(eng),
   'the rule states the consequence in the ledger, not just the practice');

// The caveat has to reach the report, or "passed with a known upward bias" degrades to "passed".
console.log('\n# an overshooting pass carries its caveat forward');
ok(/let PC_OVERSHOOT = ''/.test(wf), 'the overshoot caveat is captured in a variable');
ok(/POSITIVE CONTROL OVERSHOOT/.test(wf), 'it is logged distinctly from a clean pass');
ok(/POSITIVE_CONTROL_OVERSHOOT: PC_OVERSHOOT/.test(wf),
   'it is threaded into the report phase');
ok(/reads \\nHIGH by roughly|reads `? ?HIGH by roughly|HIGH by roughly/.test(wf),
   'the caveat quantifies the bias rather than just noting it');


// A marker proves the host path ran. Under a JIT with a disk cache, two arms can print two markers
// and execute one binary — which reads 1.000 with every existing gate satisfied. This closed an
// entire optimization axis on a void experiment before it was caught by a retroactive audit.
console.log('\n# arms must be proved to be DIFFERENT CODE, not just different markers');
ok(/IT DOES NOT PROVE THE ARMS COMPILED TO DIFFERENT CODE/.test(verify),
   'the marker/binary distinction is stated where activation is judged');
ok(/artifact-distinctness proof/.test(verify),
   'a positive obligation is named, not just a warning');
ok(/cache keys, IR\/ISA hashes, or binary paths/.test(verify),
   'the acceptable evidence forms are concrete enough to run');
ok(/name-normalised/.test(verify),
   'a symbol rename cannot fake a distinct artifact');
ok(/Same hash across arms ⇒ `status:"inactive"`/.test(verify),
   'identical artifacts are a VOID experiment, never a null result');
ok(/machine-global/.test(verify),
   'the separate-checkout escape hatch is closed: a global cache root defeats it');
ok(/2026-08-22 a retroactive audit/.test(verify) && /had to be reopened/.test(verify),
   'the incident keeps its consequence: a closed axis was reopened');

// The rule above was prose in a role file for one wave. Prose is advisory — the pool-availability
// rule sat in exactly this state and a whole wave walked past it. The script now compares the hashes
// itself, so a verifier that buries them in `notes` cannot close an axis on one binary.
console.log('\n# the script compares the artifacts itself, so the rule is not merely advisory');
ok(/artifact_distinct: \{ type: 'string' \}/.test(wf),
   'VERIFY_SCHEMA carries a dedicated artifact_distinct field');
ok(/artifact_hash_base: \{ type: 'string' \}/.test(wf) &&
   /artifact_hash_candidate: \{ type: 'string' \}/.test(wf),
   'the two hashes are structured fields the script can compare, not prose in notes');
ok(/ARTIFACT DISTINCTNESS/.test(wf) && /SAME BINARY/.test(wf),
   'an unproven comparison and a proven-identical one get distinct log lines');
// The gate must run after activation, so a patch that never executed is reported as inactive rather
// than mislabelled as a same-binary void — the two have different remedies.
const iAct = wf.indexOf('--- ACTIVATION: did the candidate');
const iArt = wf.indexOf('--- ARTIFACT DISTINCTNESS');
const iVerifiedFilter = wf.indexOf("const verified = clean.filter(");
ok(iAct > 0 && iArt > iAct && iVerifiedFilter > iArt,
   'the artifact gate runs after activation and before the winner filter');
ok(/is VOID, not a null/.test(wf),
   'an identical-artifact result is voided rather than filed as a negative finding');
ok(/`n\/a` \(not a JIT candidate\) and `unknown`/.test(wf),
   'n/a and unknown stay non-fatal, so the verifier is not pushed into a false "yes"');
ok(/Report this in the three dedicated fields, not only in prose/.test(verify),
   'the verifier is told the script reads the fields, not the notes');

// Exercise the actual comparison rather than trusting the text around it. Lifted from source so a
// flipped condition fails here instead of in a wave.
console.log('\n# artifact comparison arithmetic (extracted from source)');
const artSrc = wf.match(
  /(const ad = String\(r\.ver\.artifact_distinct[\s\S]*?const sameHash = [^\n]*\n)\s*if \(ad !== 'no' && !sameHash\) \{/,
);
ok(!!artSrc, 'the artifact decision expression can be located in kernel_workflow.js');
if (artSrc) {
  const decideArt = new Function('ver', `
    const r = { ver };
    ${artSrc[1]}
    return { fatal: !(ad !== 'no' && !sameHash), sameHash };
  `);
  const cases = [
    [{ artifact_distinct: 'yes', artifact_hash_base: 'aaa', artifact_hash_candidate: 'bbb' }, false,
     'differing hashes with a yes verdict survive'],
    [{ artifact_distinct: 'yes', artifact_hash_base: 'aaa', artifact_hash_candidate: 'aaa' }, true,
     'IDENTICAL hashes void the result even when the verifier claimed yes — the hashes outrank the verdict'],
    [{ artifact_distinct: 'no', artifact_hash_base: 'aaa', artifact_hash_candidate: 'bbb' }, true,
     'an explicit no voids it even without matching hashes'],
    [{ artifact_distinct: 'n/a' }, false, 'a non-JIT candidate is not penalised'],
    [{ artifact_distinct: 'unknown' }, false, 'an unrunnable proof warns but does not void'],
    [{}, false, 'a silent verifier does not void the result — activation already covers silence'],
    [{ artifact_distinct: 'NO' }, true, 'the verdict comparison is case-insensitive'],
    [{ artifact_hash_base: '  aaa ', artifact_hash_candidate: 'aaa' }, true,
     'hashes are trimmed before comparison, so whitespace cannot fake a difference'],
    [{ artifact_hash_base: '', artifact_hash_candidate: '' }, false,
     'two EMPTY hashes are not "identical artifacts" — that is a verifier that reported nothing'],
  ];
  for (const [ver, wantFatal, msg] of cases) ok(decideArt(ver).fatal === wantFatal, msg);
}

console.log(
  failures === 0
    ? '\nPASS: detection floor, prior art, and result provenance are all gated and reach the report.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
