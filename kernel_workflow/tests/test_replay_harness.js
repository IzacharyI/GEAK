#!/usr/bin/env node
// The replay harness is the first thing in this suite that tests BEHAVIOUR rather than prose, so it
// had better be tested that way itself.
//
// Every other suite here asserts that a sentence is present in a role file or that a regex matches
// the workflow source. That catches a deletion and nothing else: a change to the gate arithmetic
// that silently re-decides finished runs passes all of them. `scripts/replay_runs.js` exists to
// close that hole by re-deciding recorded runs on no GPU. This file proves it can:
//
//   1. it computes the paired effect the way the discipline requires (within-rep, rank-max, VOID
//      arms dropped whole);
//   2. it pairs the two arms of a control by WORKSPACE, not by whichever arm is declared base —
//      the mistake that made a real +4.5% control read +2.5% and abort a run;
//   3. a change to the gate in kernel_workflow.js CHANGES ITS OUTPUT. Without this assertion the
//      harness could be a no-op that always prints the same thing, which would be worse than having
//      no harness, because the clean --check would be read as evidence.
//
// The corpus is synthesised here rather than shipped, so the test runs in any checkout and carries
// no measurement from any particular machine.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const REPLAY = path.join(ROOT, 'scripts', 'replay_runs.js');
const WF = path.join(ROOT, 'kernel_workflow.js');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}
function run(args) {
  try { return { code: 0, out: execFileSync('node', [REPLAY, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }) }; }
  catch (e) { return { code: e.status == null ? -1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` }; }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_replay_test_'));
process.on('exit', () => { try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ } });

// ---------------------------------------------------------------- synthetic corpus
// Four arms in the shape a real control takes: the run's own baseline, a byte-identical null copy of
// it, and a matched off/on pair living together in a separate frozen workspace. The frozen pair is
// 4% apart; the frozen tree as a whole is 2% off the run's baseline, which is exactly the confounder
// that makes cross-workspace pairing wrong.
const corpus = path.join(tmp, 'runs', 'team_x', 'kern');
fs.mkdirSync(corpus, { recursive: true });
const REPS = 6;
const runs = [];
const mk = (arm, guard, rep, max) => ({ arm, guard, rep, rc: 0, ok: true, marker_ok: true, mean: max * 0.99, max });
for (let rep = 1; rep <= REPS; rep++) {
  const drift = 1 + 0.004 * Math.sin(rep);       // shared, so pairing cancels it
  const base = 4.60 * drift;
  runs.push(mk('base', 'g1', rep, base));
  runs.push(mk('null', 'g1', rep, base * 1.0005));  // byte-identical: quiet
  runs.push(mk('off', 'g1', rep, base * 1.02));     // frozen tree, switch off — 2% SLOWER than base
  runs.push(mk('on', 'g1', rep, base * 1.02 * 0.96)); // switch on — 4% faster than its own off arm
}
// One dead rep on the candidate: a crashed arm must remove the whole pair, not half of it. And one
// rep that completed and reported a latency but never printed the path marker — the more dangerous
// case, because it carries a plausible number that silently belongs to the fallback path.
runs.push({ arm: 'on', guard: 'g1', rep: 99, rc: 1, ok: false, max: null, marker_ok: false });
runs.push(mk('off', 'g1', 99, 4.60));
runs.push(Object.assign(mk('on', 'g1', 98, 4.10), { marker_ok: false }));
runs.push(mk('off', 'g1', 98, 4.60));
fs.writeFileSync(path.join(corpus, 'setup_ab_control.json'), JSON.stringify({
  arms: [
    { name: 'base', ws: '/w/run', env: {} },
    { name: 'null', ws: '/w/run_copy', env: {} },
    { name: 'off', ws: '/w/frozen', env: {} },
    { name: 'on', ws: '/w/frozen', env: { OPT_SWITCH: '1' } },
  ],
  guards: ['g1'], base: 'base', runs,
}, null, 1));

const RUNS_DIR = path.join(tmp, 'runs');

console.log('\n# the harness reads a corpus and reports what it dropped');
let r = run(['--runs', RUNS_DIR]);
ok(r.code === 0, `replay exits clean (${r.code})`);
ok(/setup_ab_control\.json/.test(r.out), 'the A/B artifact is discovered');
ok(/void=2/.test(r.out), 'the two unusable runs are counted VOID; their healthy partners are not');
ok(/rc=1/.test(r.out) && /path marker absent/.test(r.out),
   'each VOID run says WHY — a nonzero exit and a missing path marker are different failures');
ok(/dropped:[^\n]*rep98[^\n]*on\(path marker absent\)/.test(r.out) && /rep99/.test(r.out),
   'a rep with a VOID arm is dropped WHOLE and named — half a pair is not a pair');

console.log('\n# the two arms of a control are paired by workspace, not by declared base');
ok(/on vs off \(same-workspace\)/.test(r.out),
   'the frozen pair is matched to each other');
ok(!/on vs base/.test(r.out),
   'the candidate is NOT compared against the run\'s own baseline across workspaces — the pairing error that aborted a passing control');
const m = r.out.match(/measured=([-\d.]+)%\s+\((\d+) pairs/);
ok(!!m, 'a paired effect is reported with its pair count');
if (m) {
  const pct = Number(m[1]);
  ok(Math.abs(pct - 4.0) < 0.15, `the paired effect recovers the injected 4% (got ${pct}%)`);
  ok(Number(m[2]) === REPS, `the crashed rep is excluded from the pair count (${m[2]} of ${REPS})`);
}
const nul = r.out.match(/null=([-\d.]+)%/);
ok(nul && Math.abs(Number(nul[1])) < 0.2,
   'the null arm is paired against the baseline it copies, so it reads quiet rather than picking up the cross-workspace offset');

console.log('\n# both signs are decided on the same data');
ok(/band=speedup_3\.55\.\.4\.93[\s\S]*?-> PASS/.test(r.out),
   'a +4% effect passes a speedup band');
ok(/band=slowdown_-5\.0\.\.-2\.5[^\n]*\n?[^\n]*-> FAIL_WRONG_SIGN/.test(r.out),
   'the same +4% effect is WRONG SIGN against a slowdown band — the gate is sign-aware on real records, not just on hand-written cases');

console.log('\n# snapshot / --check is a working regression gate');
const snap = path.join(tmp, 'snap.txt');
ok(run(['--runs', RUNS_DIR, '--snapshot', snap]).code === 0 && fs.existsSync(snap), 'a snapshot can be recorded');
ok(run(['--runs', RUNS_DIR, '--check', snap]).code === 0, 'an unchanged workflow re-checks clean');

// ---------------------------------------------------------------- falsification
// The assertion that gives every other one its meaning: perturb the live gate and the snapshot must
// break. A harness whose --check cannot fail is a harness that certifies nothing.
console.log('\n# a change to the live gate is detected');
const orig = fs.readFileSync(WF, 'utf8');
let restored = false;
const restore = () => { if (!restored) { fs.writeFileSync(WF, orig); restored = true; } };
process.on('exit', restore);
try {
  // Swap the floor of the sensitivity band for its ceiling, inside the replayed region: an in-band
  // effect then reads as insensitivity. Chosen because it is the kind of change that is invisible in
  // a source diff review and catastrophic in a run.
  const patched = orig.replace(
    'const mLo = Math.min(Math.abs(lo), Math.abs(hi));',
    'const mLo = Math.max(Math.abs(lo), Math.abs(hi));');
  ok(patched !== orig, 'the perturbation target exists in the replayed region');
  fs.writeFileSync(WF, patched);
  const after = run(['--runs', RUNS_DIR, '--check', snap]);
  ok(after.code === 4, `--check reports a difference (exit ${after.code}, expected 4)`);
  ok(/re-decided a finished run/.test(after.out), 'the failure says what it means: a finished run would now be decided differently');
  ok(/^\s*[-+] /m.test(after.out), 'the diff shows which lines changed, so the blast radius is visible');
} finally { restore(); }
ok(fs.readFileSync(WF, 'utf8') === orig, 'the workflow source is restored after the falsification');
ok(run(['--runs', RUNS_DIR, '--check', snap]).code === 0, 'and the snapshot checks clean again');

console.log('\n# the lift refuses to run against a source it cannot locate');
const stripped = orig.replace('// <<REPLAY:pc_gate>>', '// (markers removed)');
try {
  fs.writeFileSync(WF, stripped);
  const noMark = run(['--runs', RUNS_DIR]);
  ok(noMark.code === 1 && /markers are missing/.test(noMark.out),
     'without the markers the harness stops instead of replaying a stale copy of the arithmetic');
} finally { fs.writeFileSync(WF, orig); }

console.log(
  failures === 0
    ? '\nPASS: finished runs can be re-decided, and a gate change is caught without a GPU.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
