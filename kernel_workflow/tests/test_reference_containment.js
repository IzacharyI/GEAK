#!/usr/bin/env node
// Guard for the reference-containment assertion, added after the SECOND imported-answer incident.
//
// tests/test_capability_eval.js already covers the doctrine half: engineers are never handed the
// reference location, the roadmap is scrubbed, and verify rejects byte-identical files. All of that
// was in place and correctly worded during wave 2 — and wave 2 still filed a candidate whose 11 of
// 12 files were byte-identical to the sibling reference checkout, differing only in the default of
// one `os.environ.get` call. Wave 1 had done the same at 8 of 11.
//
// The reason is structural, not behavioural: the reference tree lived one directory above the
// workspace. Not telling an agent where something is does not stop `ls ..`. So containment is now
// asserted at startup rather than requested in prose:
//
//   1. IT ABORTS      — a reference inside the run's own tree stops the run before any budget.
//   2. ANCESTOR       — "inside" means the common ancestor of task + workflow + artifacts, not cwd.
//   3. IT PASSES OUT  — a relocated reference runs normally (the fix must be usable).
//   4. HONEST SCOPE   — the pass log admits a path check cannot cover git-reachable copies.

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

// --- 1. the assertion exists, and it aborts -------------------------------
console.log('\n# the check exists and is fatal');
ok(/REFERENCE NOT CONTAINED/.test(wf), 'the containment failure has a distinct, greppable name');
ok(/throw new Error\(\s*\n?\s*`REFERENCE NOT CONTAINED/.test(wf),
   'it THROWS — a warning would leave the run free to measure a copy');
ok(/capability_eval is on/.test(wf), 'it only applies to capability_eval runs');
// The incident numbers stay in the source: they are the argument for why prose was insufficient.
ok(/8-of-11/.test(wf) && /11 of 12/.test(wf),
   'both incidents keep their file counts in the source');
ok(/one `ls \.\.`|walking up one directory/.test(wf),
   'the source names the actual reach mechanism, not a vague "leak"');

// --- 2. the check fires BEFORE any budget is spent ------------------------
console.log('\n# it fires before the run costs anything');
const iCheck = wf.indexOf('REFERENCE NOT CONTAINED');
const iSetup = wf.indexOf("phase('Setup')");
const iLoop = wf.indexOf('while (dispatched < BUDGET');
ok(iCheck > 0 && iSetup > iCheck, 'containment is asserted before Setup');
ok(iCheck > 0 && iLoop > iCheck, 'containment is asserted before the optimize loop');

// --- 3. the ancestor arithmetic, extracted and exercised ------------------
// Re-typing the rule here would let the copy drift from the original; lift the real expression.
console.log('\n# ancestor arithmetic (extracted from source)');
const src = wf.match(/  const abs = \(p\) => \{[\s\S]*?const inside = KNOWN_REFERENCE_PATHS\.filter\(\s*\n?\s*\(p\) => ancestor && \(abs\(p\) === ancestor \|\| abs\(p\)\.startsWith\(ancestor \+ '\/'\)\)\);/);
ok(!!src, 'the containment expression can be located in kernel_workflow.js');
if (src) {
  const decide = new Function('KERNEL_PATH_ORIG', 'EXP_ROOT', 'WORKFLOW_DIR', 'KNOWN_REFERENCE_PATHS',
    `${src[0]}\n return { inside, ancestor };`);
  const K = '/proj/app/tasks/megamoe';
  const E = '/proj/app/artifacts/runs';
  const W = '/proj/app/GEAK/kernel_workflow';
  const run = (refs) => decide(K, E, W, refs);

  ok(run([]).ancestor === '/proj/app',
     'the ancestor is the common prefix of task + artifacts + workflow');
  // The real incident: a sibling of the task dir, one level up from the workspace.
  ok(run(['/proj/app/AITER-candidate']).inside.length === 1,
     'a sibling reference checkout inside the run tree is caught (the actual incident)');
  ok(run(['/root/geak_reference/AITER-candidate']).inside.length === 0,
     'a relocated reference passes — the fix has to be usable, not just diagnostic');
  ok(run(['/proj/app/tasks/megamoe/ref']).inside.length === 1,
     'a reference nested inside the task dir itself is caught');
  ok(run(['/proj/app']).inside.length === 1,
     'the ancestor directory itself counts as inside');
  // Prefix-vs-path-boundary: /proj/application must NOT be swallowed by the /proj/app ancestor.
  ok(run(['/proj/application/ref']).inside.length === 0,
     'a sibling whose name merely starts with the ancestor string is NOT flagged');
  ok(run(['/proj/app/../app/AITER-candidate']).inside.length === 1,
     'a path that only escapes via .. is resolved first, then caught');
  ok(run(['/root/ref-a', '/proj/app/ref-b']).inside.length === 1,
     'a mixed list reports only the offending tree');
}

// --- 4. the pass path must not overclaim ----------------------------------
// A path check cannot see a reference that is also a branch in a repo the run can read. Saying so is
// the difference between a guarantee and a filter; only one of those is true here.
console.log('\n# the pass log states what it does NOT cover');
ok(/REFERENCE CONTAINMENT ok/.test(wf), 'a passing check is logged, not silent');
ok(/checks paths only/.test(wf), 'the log admits the check is path-scoped');
ok(/branch\/commit in a repository the run can read/.test(wf),
   'the git-reachable hole is named explicitly rather than left implied');
ok(/only verify's byte-identity check stands between that and a counted win/.test(wf),
   'the log says which gate is load-bearing once containment passes');

console.log(
  failures === 0
    ? '\nPASS: a reference inside the run tree stops the run; a relocated one does not.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
