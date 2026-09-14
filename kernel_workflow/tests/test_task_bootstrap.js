#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const CANONICAL_TASK = 'megamoe_v2_ep8_mega';
const TASK_DIR = path.join(ROOT, 'tasks', CANONICAL_TASK);

let failures = 0;
const ok = (value, message) => {
  if (value) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_boot_test_'));
process.on('exit', () => {
  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ }
});

// Copy only the bootstrap package surface into a temporary workflow. This lets the
// test pin a synthetic baseline commit without weakening the real task's immutable
// public-AITER pin.
const fixtureRoot = path.join(tmp, 'workflow');
fs.mkdirSync(path.join(fixtureRoot, 'scripts'), { recursive: true });
fs.mkdirSync(path.join(fixtureRoot, 'tasks', CANONICAL_TASK), { recursive: true });
fs.copyFileSync(
  path.join(ROOT, 'scripts', 'bootstrap_task.sh'),
  path.join(fixtureRoot, 'scripts', 'bootstrap_task.sh'),
);
for (const name of ['GEAK_TASK.md', 'launch_args.json']) {
  fs.copyFileSync(path.join(TASK_DIR, name),
    path.join(fixtureRoot, 'tasks', CANONICAL_TASK, name));
}

const base = path.join(tmp, 'base');
fs.mkdirSync(path.join(base, 'aiter', 'ops', 'flydsl'), { recursive: true });
fs.writeFileSync(path.join(base, 'aiter', 'ops', 'flydsl', 'marker.py'),
  'BASELINE_MARKER = 1\n');
execFileSync('git', ['init', '-q'], { cwd: base });
execFileSync('git', ['-c', 'user.name=test', '-c', 'user.email=test@example.com',
  'add', '-A'], { cwd: base });
execFileSync('git', ['-c', 'user.name=test', '-c', 'user.email=test@example.com',
  'commit', '-q', '-m', 'baseline'], { cwd: base });
const baseHead = execFileSync('git', ['rev-parse', 'HEAD'], {
  cwd: base, encoding: 'utf8',
}).trim();
const fixtureArgsPath = path.join(
  fixtureRoot, 'tasks', CANONICAL_TASK, 'launch_args.json',
);
const fixtureArgs = JSON.parse(fs.readFileSync(fixtureArgsPath, 'utf8'));
fixtureArgs.expert_skill_baseline_commit = baseHead;
fs.writeFileSync(fixtureArgsPath, `${JSON.stringify(fixtureArgs, null, 2)}\n`);

const boot = path.join(fixtureRoot, 'scripts', 'bootstrap_task.sh');
const run = (args) => {
  try {
    const out = execFileSync('bash', [boot, ...args], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { code: 0, out };
  } catch (error) {
    return {
      code: error.status == null ? -1 : error.status,
      out: `${error.stdout || ''}${error.stderr || ''}`,
    };
  }
};

console.log('\n# only the canonical MegaMoE task is active');
ok(fs.existsSync(TASK_DIR), 'canonical task exists');
ok(!fs.existsSync(path.join(ROOT, 'tasks', 'megamoe_v2_ep8')),
  'legacy strict task is absent');
ok(!fs.existsSync(path.join(ROOT, 'tasks', 'megamoe_v2_ep8_optimize')),
  'legacy optimize task is absent');
const bootstrapSource = fs.readFileSync(
  path.join(ROOT, 'scripts', 'bootstrap_task.sh'), 'utf8',
);
ok(/TASK=megamoe_v2_ep8_mega/.test(bootstrapSource),
  'bootstrap defaults to the canonical task');

console.log('\n# invalid inputs fail before assembly');
let result = run(['--no-probe', '--baseline', path.join(tmp, 'missing'),
  '--out', path.join(tmp, 'missing-out')]);
ok(result.code === 1 && /not a directory/.test(result.out),
  'missing baseline is rejected');
result = run(['--no-probe', '--baseline', tmp,
  '--out', path.join(tmp, 'invalid-out')]);
ok(result.code === 1 && /does not look like an AITER checkout/.test(result.out),
  'non-AITER baseline is rejected');
result = run(['--no-probe', '--baseline', base,
  '--out', path.join(tmp, 'unknown-out'), '--task', 'unknown']);
ok(result.code === 1 && /no task template/.test(result.out) &&
   result.out.includes(CANONICAL_TASK),
  'unknown task reports the one canonical choice');

console.log('\n# canonical assembly is pinned, portable and exact');
const out = path.join(tmp, 'workspace');
const expRoot = path.join(tmp, 'runs');
const stateDir = path.join(tmp, 'state');
const argsOut = path.join(tmp, 'launch', 'launch_args.json');
result = run([
  '--no-probe',
  '--baseline', base,
  '--out', out,
  '--exp-root', expRoot,
  '--state-dir', stateDir,
  '--mori-root', '/mori/here',
  '--jit-dir', '/jit/here',
  '--args-out', argsOut,
]);
ok(result.code === 0, `assembly succeeds${result.code ? `: ${result.out}` : ''}`);
const task = fs.readFileSync(path.join(out, 'GEAK_TASK.md'), 'utf8');
const args = JSON.parse(fs.readFileSync(argsOut, 'utf8'));
ok(args.kernel_path === out && args.workflow_dir === fixtureRoot,
  'resolved paths identify the assembled workspace and workflow');
ok(args.exp_root === expRoot && args.state_dir === stateDir,
  'output and resumable state roots are explicit');
ok(args.expert_skill_baseline_commit === baseHead,
  'assembly was bound to the exact baseline commit');
ok(args.task === task && task.includes(fixtureRoot) && task.includes(expRoot),
  'the fixed task is inlined verbatim after path substitution');
ok(!/\$\{[A-Z_]+\}/.test(task) &&
   !/\$\{[A-Z_]+\}/.test(JSON.stringify(args)),
  'no unresolved placeholder survives');
ok(task.length < 2000 && !/resume|STATE\.json|next_blocker|contract|wf_/i.test(task),
  'the canonical task remains compact and run-independent');
ok(args.use_expert_skills === 'true' &&
   args.expert_skill_id === 'megamoe_ep_mega_fusion',
  'Skill selection remains structured launch data');
ok(args.gpus_per_job === 8 && args.launch_target === 2,
  'EP8 and launch target remain structured launch data');

console.log('\n# workspace is a copy, never a view of the denominator');
const copied = path.join(out, 'aiter', 'ops', 'flydsl', 'marker.py');
ok(fs.existsSync(copied) && !fs.lstatSync(copied).isSymbolicLink(),
  'baseline source is copied into the workspace');
fs.writeFileSync(copied, 'BASELINE_MARKER = 2\n');
ok(fs.readFileSync(
  path.join(base, 'aiter', 'ops', 'flydsl', 'marker.py'), 'utf8',
) === 'BASELINE_MARKER = 1\n',
'editing the workspace cannot mutate the frozen baseline');

console.log(failures === 0
  ? '\nPASS: bootstrap exposes one canonical, stable MegaMoE task.'
  : `\nFAIL: ${failures} assertion(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
