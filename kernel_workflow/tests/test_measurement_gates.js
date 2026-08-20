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
ok(/an empty array means you looked, an omitted field\s+means you did not/.test(lead),
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
  /const inBand = ([\s\S]*?);\n\s*const ok = pc\.ran !== false && inBand;/,
);
ok(!!gateSrc, 'the in-band expression can be located in kernel_workflow.js');
if (gateSrc) {
  const decide = new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
    const inBand = ${gateSrc[1]};
    return pc.ran !== false && inBand;
  `);
  const band = { expected_pct_lo: 3.55, expected_pct_hi: 4.93 };  // the real +4.71% guard
  const cases = [
    [{ ran: true, measured_pct: 4.71 }, true, 'the known +4.71% passes'],
    [{ ran: true, measured_pct: 3.55 }, true, 'the lower bound is inclusive'],
    [{ ran: true, measured_pct: 4.93 }, true, 'the upper bound is inclusive'],
    [{ ran: true, measured_pct: 0.4 }, false, 'a harness that sees almost nothing fails'],
    [{ ran: true, measured_pct: -4.71 }, false, 'the right magnitude with the WRONG SIGN fails'],
    [{ ran: true, measured_pct: 40.0 }, false, 'an implausibly large delta fails too (band, not floor)'],
    [{ ran: false }, false, 'a control that did not run fails'],
    [{ ran: true }, false, 'a missing measurement is not silently a pass'],
    [{ ran: true, measured_pct: 'n/a' }, false, 'an unparseable measurement is not a pass'],
    [{ ran: true, measured_pct: NaN }, false, 'NaN is not a pass'],
    [{}, false, 'an empty result object is not a pass'],
  ];
  for (const [pc, want, msg] of cases) ok(decide(pc, band) === want, msg);
}

console.log(
  failures === 0
    ? '\nPASS: detection floor, prior art, and result provenance are all gated and reach the report.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
