#!/usr/bin/env node
// The portability guard: "GEAK + a baseline AITER checkout + handoff.md" has to be enough to stand
// this task up in an environment that has never seen it.
//
// It was not enough. The task text lived beside the run tree with this machine's paths baked into
// it, and the invocation recipe lived in a per-wave args file next to a reference patch — so a
// fresh environment could clone the workflow, clone the baseline, and still have no way to start.
// The gap does not announce itself: everything looks present, and the first sign of trouble is an
// import error an hour into a lease.
//
// Unlike the sibling suites, this one does not read the script and assert on its prose. It RUNS the
// bootstrap against a synthetic baseline in a temp dir and asserts on what lands on disk, because
// the failure being guarded is a substitution that silently did not happen.

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const BOOT = path.join(ROOT, 'scripts', 'bootstrap_task.sh');
const TASK_DIR = path.join(ROOT, 'tasks', 'megamoe_v2_ep8');

let failures = 0;
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

// Run the bootstrap; never throw, so a non-zero exit is an assertable value rather than a crash.
function run(args, opts = {}) {
  try {
    const out = execFileSync('bash', [BOOT, ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], ...opts });
    return { code: 0, out };
  } catch (e) {
    return { code: e.status == null ? -1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` };
  }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_boot_test_'));
process.on('exit', () => { try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ } });

// A synthetic "baseline": enough structure to satisfy the sanity check, no AITER required.
const base = path.join(tmp, 'base');
fs.mkdirSync(path.join(base, 'aiter', 'ops', 'flydsl'), { recursive: true });
fs.writeFileSync(path.join(base, 'aiter', 'ops', 'flydsl', 'marker.py'), 'BASELINE_MARKER = 1\n');

console.log('\n# the task template is self-describing rather than machine-specific');
const tpl = fs.readFileSync(path.join(TASK_DIR, 'GEAK_TASK.md'), 'utf8');
ok(/\$\{MORI_ROOT\}/.test(tpl) && /\$\{AITER_JIT_DIR\}/.test(tpl),
   'the two machine-local paths are placeholders in the template, not this box\'s paths');
ok(!/\/sgl-workspace\//.test(tpl),
   'no path from the environment it was authored in survives into the template');
ok(/## Environment prerequisites/.test(tpl),
   'the template states what an environment must provide, so a fresh one can tell if it qualifies');
ok(/8-rank intra-node collective|EP8 is baked into the guards/.test(tpl),
   'the prerequisites explain WHY 8 GPUs, so nobody tries to scale the task down to fit a box');
const argTpl = fs.readFileSync(path.join(TASK_DIR, 'launch_args.json'), 'utf8');
ok(!/\/root\/geak_reference|\/sgl-workspace\//.test(argTpl),
   'the launch recipe points at no reference implementation and no prior run tree');
// The synthetic control is the whole reason this task can be launched somewhere the answer does not
// exist: a control that applies a finished patch is unavailable in exactly the environment that
// needs it most.
const argJson = JSON.parse(argTpl.replace(/\$\{[A-Z_]+\}/g, ''));
const pc = argJson.positive_control;
ok(pc && Number(pc.expected_pct_lo) < 0 && Number(pc.expected_pct_hi) < 0,
   'the positive control is a synthetic SLOWDOWN — it needs no working optimization to exist');
ok(/jit_arm_isolation\.md/.test(pc.how),
   'the control tells the engineer to anchor the JIT cache key before authoring its gate');
ok(/noise floor|1\.29-1\.46/.test(pc.how),
   'the injected cost is sized against the guard\'s measured noise floor, not guessed');

console.log('\n# it refuses what it cannot assemble, with the remedy named');
let r = run(['--no-probe', '--baseline', path.join(tmp, 'nope'), '--out', path.join(tmp, 'x1')]);
ok(r.code === 1 && /not a directory/.test(r.out), 'a missing baseline is rejected');
r = run(['--no-probe', '--baseline', tmp, '--out', path.join(tmp, 'x2')]);
ok(r.code === 1 && /does not look like an AITER checkout/.test(r.out),
   'a directory that is not an AITER checkout is rejected before anything is copied');
r = run(['--no-probe', '--baseline', base, '--out', path.join(tmp, 'x3'), '--task', 'no_such_task']);
ok(r.code === 1 && /no task template at/.test(r.out) && /have:/.test(r.out),
   'an unknown task names the templates that do exist');

console.log('\n# a real assembly resolves every placeholder');
const ws = path.join(tmp, 'ws');
r = run(['--no-probe', '--baseline', base, '--out', ws, '--mori-root', '/mori/here', '--jit-dir', '/jit/here']);
ok(r.code === 0, `assembly succeeds (exit ${r.code})`);
ok(/environment probe: SKIPPED/.test(r.out),
   '--no-probe says so out loud rather than quietly producing an uncertified workspace');

const got = fs.existsSync(path.join(ws, 'GEAK_TASK.md')) ? fs.readFileSync(path.join(ws, 'GEAK_TASK.md'), 'utf8') : '';
const gotArgsPath = path.join(tmp, 'launch_args.json');
const gotArgs = fs.existsSync(gotArgsPath) ? fs.readFileSync(gotArgsPath, 'utf8') : '';
// This is the assertion the whole file exists for. An unresolved ${...} is not a cosmetic defect:
// the run starts, the path does not resolve, and it surfaces as an import error inside a lease.
ok(got && !/\$\{[A-Z_]+\}/.test(got.replace(/unsubstituted `\$\{\.\.\.\}`/g, '')),
   'no placeholder survives into the materialised task text');
ok(gotArgs && !/\$\{[A-Z_]+\}/.test(gotArgs),
   'no placeholder survives into the materialised launch args');
ok(/PYTHONPATH="\$PWD:\/mori\/here:\/mori\/here\/python"/.test(got),
   'the MORI path is substituted everywhere it appears, including inside the run command');
ok(/AITER_JIT_DIR=\/jit\/here/.test(got), 'the JIT cache path is substituted');

const ja = JSON.parse(gotArgs);
ok(ja.kernel_path === ws, 'kernel_path points at the workspace that was just built');
ok(ja.workflow_dir === ROOT, 'workflow_dir points at the workflow that built it');
ok(ja.gpus_per_job === 8 && ja.gpu_ids === '0,1,2,3,4,5,6,7',
   'the collective takes all 8 cards as one lease');
ok(String(ja.capability_eval) === 'true',
   'capability_eval stays on by default — the task exists to evaluate the loop, not just to emit a kernel');
ok(fs.existsSync(ja.exp_root) && fs.existsSync(ja.state_dir),
   'exp_root and state_dir are created, so round 1 does not fail on a missing directory');

// kernel_workflow.js does `const TASK = A.task || ''` and no role file names GEAK_TASK.md, so the
// materialised task text reaches the wave through this field or not at all. Asserting the field is
// non-empty is not enough -- an empty-ish string would pass that and still starve the run -- so
// compare it against the file it is supposed to carry.
ok(typeof ja.task === 'string' && ja.task.trim().length > 0,
   'launch args carry a non-empty task; without it the wave runs with TASK=\'\'');
ok(ja.task === got,
   'the inlined task is the materialised GEAK_TASK.md verbatim, placeholders already resolved');
ok(!/\$\{[A-Z_]+\}/.test(String(ja.task).replace(/unsubstituted `\$\{\.\.\.\}`/g, '')),
   'the inlined task text is placeholder-free too, not just the paths around it');

console.log('\n# the workspace is a real copy, not a view of the baseline');
const copied = path.join(ws, 'aiter', 'ops', 'flydsl', 'marker.py');
ok(fs.existsSync(copied) && !fs.lstatSync(copied).isSymbolicLink(),
   'baseline files are copied, not symlinked');
fs.writeFileSync(copied, 'BASELINE_MARKER = 2\n');
ok(fs.readFileSync(path.join(base, 'aiter', 'ops', 'flydsl', 'marker.py'), 'utf8') === 'BASELINE_MARKER = 1\n',
   'editing the workspace cannot mutate the baseline — the denominator stays frozen');

console.log('\n# a non-empty --out is not silently clobbered');
r = run(['--no-probe', '--baseline', base, '--out', ws]);
ok(r.code === 1 && /--force/.test(r.out), 'a populated output dir is refused unless --force is given');
r = run(['--no-probe', '--baseline', base, '--out', ws, '--force']);
ok(r.code === 0, '--force allows the overwrite');

console.log(
  failures === 0
    ? '\nPASS: workflow + a baseline checkout is enough to stand the task up.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
