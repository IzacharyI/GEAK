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
let markerFile = '';
function ok(cond, msg) {
  if (cond) { console.log(`  ok: ${msg}`); }
  else { console.error(`  FAIL: ${msg}`); failures++; }
}

// Run the bootstrap; never throw, so a non-zero exit is an assertable value rather than a crash.
function run(args, opts = {}) {
  try {
    const out = execFileSync('bash', [BOOT, ...args], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
      ...opts,
      env: { ...process.env, ...(markerFile ? { MARKER_FILE: markerFile } : {}), ...(opts.env || {}) },
    });
    return { code: 0, out };
  } catch (e) {
    return { code: e.status == null ? -1 : e.status, out: `${e.stdout || ''}${e.stderr || ''}` };
  }
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_boot_test_'));
process.on('exit', () => { try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ } });
const markerRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_boot_markers_'));
process.on('exit', () => { try { fs.rmSync(markerRoot, { recursive: true, force: true }); } catch { /* best effort */ } });
markerFile = path.join(markerRoot, 'markers.txt');
fs.writeFileSync(markerFile,
  Array.from({ length: 60 }, (_, i) => `SYNTHETIC_BOOT_MARKER_${String(i).padStart(3, '0')}`).join('\n') + '\n');

// A synthetic "baseline": enough structure to satisfy the sanity check, no AITER required.
const base = path.join(tmp, 'base');
fs.mkdirSync(path.join(base, 'aiter', 'ops', 'flydsl'), { recursive: true });
fs.writeFileSync(path.join(base, 'aiter', 'ops', 'flydsl', 'marker.py'), 'BASELINE_MARKER = 1\n');

console.log('\n# the task template is self-describing rather than machine-specific');
const tpl = fs.readFileSync(path.join(TASK_DIR, 'GEAK_TASK.md'), 'utf8');
ok(!fs.existsSync(path.join(TASK_DIR, 'HANDVER_SKILL.md')),
   'the strict task tree contains no operator-specific hand-version recipe');
ok(/\$\{MORI_ROOT\}/.test(tpl) && /\$\{AITER_JIT_DIR\}/.test(tpl),
   'the two machine-local paths are placeholders in the template, not this box\'s paths');
ok(!/\/sgl-workspace\//.test(tpl),
   'no path from the environment it was authored in survives into the template');
ok(/## Environment prerequisites/.test(tpl),
   'the template states what an environment must provide, so a fresh one can tell if it qualifies');
ok(/8-rank intra-node collective|EP8 is baked into the guards/.test(tpl),
   'the prerequisites explain WHY 8 GPUs, so nobody tries to scale the task down to fit a box');
ok(/--bs-list 128,512,8192/.test(tpl) && /--accuracy-max-bs 8192/.test(tpl),
   'correctness covers the large configuration where the target fused path runs');
const argTpl = fs.readFileSync(path.join(TASK_DIR, 'launch_args.json'), 'utf8');
ok(!/\/root\/|\/sgl-workspace\//.test(argTpl),
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
// The dose must be sized against the statistic the GATE uses. It used to say "3x the guard's noise
// floor", which reads as careful and is off by the ratio between a floor and the worst of a handful
// of pairs -- wave 10 built exactly what it was told to, measured -2.41% with 6/6 sign agreement,
// and was aborted at 1.3x a 1.881pp worst null pair. Sizing and judging must name the same number.
ok(/worst pair|WORST PAIR/.test(pc.how) && /null arm FIRST|null FIRST|measure the null first/i.test(pc.how),
   'the injected cost is sized against the worst NULL PAIR, and the null is measured first');
ok(!/roughly 3x that guard's [\d.]+-[\d.]+% noise floor/.test(pc.how),
   'the old floor-based sizing rule is gone -- it guaranteed a correctly-built control would fail');
ok(Math.abs(Number(pc.expected_pct_lo)) > Math.abs(Number(pc.expected_pct_hi)) &&
   Math.abs(Number(pc.expected_pct_hi)) >= 3.0,
   'the band is wide enough that 4x a realistic worst null pair lands inside it');

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
ok(ja.analysis_skill === 'none',
   'blind proof does not inject the operator-specific MegaMoE post-mortem analysis skill');
ok(String(ja.strict_autonomy) === 'true' && ja.objective === 'working_kernel',
   'the portable task preserves slower staged artifacts but keeps strict final acceptance');
ok(ja.promotion_metric === 'operator_e2e' &&
   ja.target_guards.join(',') === '8192_uniform' &&
   ja.regression_guards.length === 3,
   'only the explicit uniform target creates credit; the other guards are vetoes');
ok(ja.launch_target === 2 && String(ja.require_overlap) === 'true' &&
   String(ja.require_attribution) === 'true' &&
   String(ja.require_artifact_distinct) === 'true' && ja.required_replays === 1000,
   'the two-launch, overlap, attribution and replay requirements are machine-readable');
ok(ja.required_pairs_by_guard['8192_uniform'] === 5 &&
   ja.required_pairs_by_guard['512_uniform'] === 16,
   'pair depth is explicit per guard instead of silently under-sampling the bimodal 512 cases');
ok(ja.accuracy_metric === 'relL2' && ja.accuracy_threshold === 0.10,
   'accuracy uses the fixed task metric and threshold rather than verifier-selected labels');
ok(ja.containment_preflight && ja.containment_preflight.clean === true &&
   ja.containment_preflight.marker_manifest_sha256 &&
   ja.containment_preflight.content_scan === 'clean' &&
   ja.containment_preflight.skill_address_scan === 'clean' &&
   ja.containment_preflight.workflow_revision &&
   ja.containment_preflight.roots.includes(ws) &&
   ja.containment_preflight.roots.includes(ROOT),
   'trusted bootstrap binds clean content/address scans to workflow revision and readable roots');
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

console.log('\n# the reference paths are not written into the tree an engineer walks');
// launch_args.json holds `known_reference_paths` — the addresses the provenance check exists to
// refuse. The default destination was the workspace's own parent, so assembling a capability-eval
// wave published the answer's location inside the walk path, silently, in the one file whose whole
// job is to hold what nobody may read. Containment here is by DISTANCE (under uid 0 a chmod
// quarantine is inert), so only the location can fix it.
const REF = path.join(markerRoot, 'hidden-reference');
fs.mkdirSync(path.join(REF, 'aiter', 'ops', 'flydsl'), { recursive: true });
fs.writeFileSync(path.join(REF, 'aiter', 'ops', 'flydsl', 'marker.py'),
  'BASELINE_MARKER = 1\nREFERENCE_ONLY_CHANGE = 1\n');
const wsr = path.join(tmp, 'wsref');
// Genuinely outside: `tmp` IS the workspace's parent, which is precisely the swept tree, so a
// subdirectory of it would not be outside anything.
const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'gk_boot_args_'));
const argsOut = path.join(outside, 'launch_args.json');
r = run(['--no-probe', '--baseline', base, '--out', wsr, '--known-reference', REF,
         '--args-out', argsOut]);
ok(r.code === 0 && fs.existsSync(argsOut), '--args-out places the launch args where it is told');
const refArgs = JSON.parse(fs.readFileSync(argsOut, 'utf8'));
ok(refArgs.known_reference_paths === '' && refArgs.known_reference_hashes.length === 1,
   'strict args replace the reference path with one opaque changed-file hash row');
ok(refArgs.known_reference_hashes[0].path_sha256 &&
   refArgs.known_reference_hashes[0].sha256 &&
   refArgs.known_reference_hashes[0].normalized_sha256,
   'the manifest carries path/raw/normalized digests without a filename');
ok(JSON.stringify(refArgs).indexOf(REF) < 0 && !/marker\.py/.test(JSON.stringify(refArgs.known_reference_hashes)),
   'neither the source address nor reference filename survives in agent-readable args');
ok(!fs.existsSync(path.join(tmp, 'launch_args.json')) ||
   !fs.readFileSync(path.join(tmp, 'launch_args.json'), 'utf8').includes(REF),
   'nothing naming the reference is left beside the workspace');
r = run(['--no-probe', '--baseline', base, '--out', path.join(tmp, 'wsref2'),
         '--known-reference', REF, '--args-out', path.join(tmp, 'launch_args.json')]);
ok(r.code === 1 && /refusing to write launch args/.test(r.out),
   'an in-tree --args-out is REFUSED, not warned about — a warning scrolls past and the file is already written');

console.log('\n# strict autonomy refuses inherited code/state before launch');
const dirtyState = path.join(tmp, 'dirty_state');
fs.mkdirSync(path.join(dirtyState, 'best'), { recursive: true });
fs.writeFileSync(path.join(dirtyState, 'best', 'candidate.py'), 'ANSWER = 1\n');
r = run(['--no-probe', '--baseline', base, '--out', path.join(tmp, 'wsdirty'),
         '--state-dir', dirtyState]);
ok(r.code === 1 && /strict_autonomy task refuses inherited state/.test(r.out) &&
   /NEW empty directory/.test(r.out),
   'a proof workspace cannot be assembled from STATE_DIR/best');

const leakyExp = path.join(outside, 'leaky_exp');
fs.mkdirSync(leakyExp, { recursive: true });
fs.writeFileSync(path.join(leakyExp, 'copied_answer.py'), 'SYNTHETIC_BOOT_MARKER_000 = 1\n');
r = run(['--no-probe', '--baseline', base, '--out', path.join(tmp, 'wsleakyexp'),
         '--exp-root', leakyExp, '--state-dir', path.join(tmp, 'fresh_state')]);
ok(r.code === 1 && /containment preflight failed/.test(r.out) && /leaky_exp/.test(r.out),
   'a custom exp root containing reference markers is scanned and refused');
// The no-reference case must keep the old, convenient default: this rule is about the reference,
// not about making every assembly harder.
r = run(['--no-probe', '--baseline', base, '--out', path.join(tmp, 'wsplain')]);
ok(r.code === 0 && fs.existsSync(path.join(tmp, 'launch_args.json')),
   'with no reference declared the args still land next to the workspace, as before');

console.log('\n# a non-empty --out is not silently clobbered');
r = run(['--no-probe', '--baseline', base, '--out', ws]);
ok(r.code === 1 && /NEW empty workspace/.test(r.out),
   'a populated strict-proof workspace is refused');
r = run(['--no-probe', '--baseline', base, '--out', ws, '--force']);
ok(r.code === 1 && /ignores --force/.test(r.out),
   '--force cannot preserve stale files inside a strict-proof workspace');

console.log(
  failures === 0
    ? '\nPASS: workflow + a baseline checkout is enough to stand the task up.'
    : `\nFAIL: ${failures} assertion(s) failed.`,
);
process.exit(failures === 0 ? 0 : 1);
