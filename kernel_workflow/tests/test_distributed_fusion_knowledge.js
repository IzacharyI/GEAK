#!/usr/bin/env node
// Wiring guard for the `distributed` specialty and its knowledge file.
//
// The knowledge is only a capability if it is REACHABLE: the orchestrator must accept the
// specialty, the roles that consume knowledge must route to the file, and the file must carry the
// invariants that make a fused multi-rank kernel correct rather than merely fast. A plain
// documentation drop passes none of these.
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..', '..');
const WF = path.join(ROOT, 'kernel_workflow');
const read = (...p) => fs.readFileSync(path.join(WF, ...p), 'utf8');

let failures = 0;
const ok = (condition, message) => {
  if (condition) console.log('  ok:', message);
  else { console.error('  FAIL:', message); failures++; }
};

console.log('\n# distributed specialty wiring');

// 1. Orchestrator accepts it. Without this the TechLead cannot emit the direction at all:
//    PLAN_SCHEMA validation rejects the round and the knowledge is unreachable.
const src = read('kernel_workflow.js');
const enumLine = src.match(/specialty:\s*\{\s*type:\s*'string',\s*enum:\s*\[([^\]]+)\]/);
ok(!!enumLine, 'PLAN_SCHEMA specialty enum found');
if (enumLine) {
  const values = enumLine[1].split(',').map((s) => s.trim().replace(/'/g, ''));
  ok(values.includes('distributed'), 'enum accepts `distributed`');
  for (const pre of ['algorithm', 'memory', 'compute', 'host_runtime', 'deep_explore']) {
    ok(values.includes(pre), `pre-existing specialty preserved: ${pre}`);
  }
}

// 2. Dispatch stays generic. Only `deep_explore` may be special-cased; if `distributed` needed
//    bespoke orchestration the enum change alone would silently do nothing.
ok(
  !/specialty === 'distributed'/.test(src),
  'dispatch treats `distributed` generically (no special-casing needed)',
);

// 3. The file exists and the consuming roles route to it.
const KNOWLEDGE = 'distributed_fusion.md';
ok(fs.existsSync(path.join(WF, 'knowledge', KNOWLEDGE)), `knowledge/${KNOWLEDGE} exists`);

const engineer = read('roles', 'engineer.md');
ok(/`SPECIALTY`[^\n]*distributed/.test(engineer), 'engineer.md declares the specialty');
ok(
  new RegExp(`- distributed\\s+→[\\s\\S]{0,200}${KNOWLEDGE}`).test(engineer),
  'engineer.md routes `distributed` to the knowledge file',
);

const deep = read('roles', 'deep_engineer.md');
ok(deep.includes(KNOWLEDGE), 'deep_engineer.md lists the knowledge file');
ok(
  /distributed_fusion\.md[\s\S]{0,240}(ONLY when|Skip entirely)/.test(deep),
  'deep_engineer.md gates the read so single-GPU kernels do not load it',
);

const lead = read('roles', 'tech_lead.md');
ok(/candidate_directions[\s\S]{0,400}distributed/.test(lead), 'tech_lead.md may emit the specialty');
ok(lead.includes(KNOWLEDGE), 'tech_lead.md points at the knowledge file');
ok(
  /no-payload control/.test(lead),
  'tech_lead.md requires the no-payload control before spending a round',
);

ok(read('README.md').includes('distributed_fusion'), 'README knowledge list mentions it');

// 3b. The two gates the doctrine assumes. Knowledge that says "you must clear a liveness gate" is
//     inert unless verify actually applies one, and an interleaving rule is inert unless the
//     COMMANDMENT is built to interleave.
ok(
  /\.\.\.\(d\.specialty \? \{ SPECIALTY: d\.specialty \} : \{\}\)/.test(src),
  'verify_engineer receives the direction specialty',
);
ok(/liveness: \{ type: 'string' \}/.test(src), 'VERIFY_SCHEMA carries a liveness verdict');

const verify = read('roles', 'verify_engineer.md');
ok(/`SPECIALTY` is `distributed`/.test(verify), 'verify gates the liveness stress on the specialty');
ok(/≥1000 times/.test(verify), 'liveness stress is quantified');
ok(/two different problem sizes/.test(verify), 'liveness varies problem size (residency changes with it)');
ok(/timeout here is a FAILURE, never a skip/.test(verify), 'a hang cannot be silently skipped');
ok(/liveness.*only when SPECIALTY=distributed/.test(verify), 'return contract documents the field');

const bench = read('roles', 'benchmark_engineer.md');
ok(/RESOLUTION FLOOR/.test(bench), 'baseline drift is recorded as a resolution floor');
ok(/alternately inside ONE process invocation/.test(bench), 'COMMANDMENT interleaves when drift is large');
ok(/paired\*\* delta/.test(bench), 'the reported delta is paired, not two independent medians');
ok(/RANK-MAX, not rank-mean/.test(bench), 'multi-rank metric is rank-max');

// The gates must stay OFF by default: a normal single-GPU run has no specialty match and no drift,
// so neither addition may change its behavior.
ok(/ONLY if `SPECIALTY` is `distributed`/.test(verify), 'liveness gate is opt-in, not always-on');
ok(/If `drift` exceeds/.test(bench), 'interleaving is conditional on measured drift');

// 3c. Author mode (0→1). The rank count must actually reach the two roles that branch on it, and
//     the author must be told the seed is the SCATTERED form — a from-scratch fused megakernel is
//     the single most likely way to burn the whole budget on a hang.
const plumbed = src.match(/GPUS_PER_JOB: String\(GPU_RESOURCE\.gpusPerJob\)/g) || [];
ok(plumbed.length >= 2, 'GPUS_PER_JOB reaches both author_engineer and tech_lead analyze');

const author = read('roles', 'author_engineer.md');
ok(/`GPUS_PER_JOB`/.test(author), 'author_engineer documents GPUS_PER_JOB as an input');
ok(/GPUS_PER_JOB` > 1/.test(author), 'the distributed section is gated on rank count > 1');
ok(
  /NAIVE, SCATTERED, MULTI-LAUNCH form — never a fused megakernel/.test(author),
  'author seeds the scattered form and leaves fusion to the optimize specialty',
);
ok(/shape the seed to be\s*\*fusable\*/.test(author), 'the seed is required to stay legible to the optimizer');
ok(/timeout 600 bash/.test(author), 'every distributed GPU command carries a wall-clock timeout');
ok(/return `authored:false`/.test(author), 'a hang is a clean failure, not a stall');
ok(/launch all N ranks/.test(author), 'the correctness loop launches every rank');
ok(/distributed_fusion\.md` Levers 7–9/.test(author), 'author cross-references the deadlock levers');
ok(
  /`GPUS_PER_JOB == "1"` disqualifies the specialty outright/.test(lead),
  'tech_lead keys distributed eligibility off the resolved rank count',
);

// 4. Doctrine content. These are the load-bearing claims: each one, if absent, turns a fusion
//    that merely underperforms into one that hangs or silently reads stale data.
console.log(`\n# knowledge/${KNOWLEDGE} content`);
const doc = read('knowledge', KNOWLEDGE);
const required = [
  [/removes a \*wait\*, not[\s\S]{0,40}\*launch\*/, 'Lever 3: fusion pays for waits, not launches'],
  [/acquire fence/i, 'every wait pairs with an acquire fence'],
  [/system-scope atomics/, 'publication uses system scope on multi-die parts'],
  [/write-through/i, 'write-through stores as the cheap alternative to a release fence'],
  [/every participant index must land inside/i, 'residency invariant is stated as an invariant'],
  [/coordinator → workers → coordinator/, 'acyclicity rule names the cycle'],
  [/Never clear an arrival counter/i, 'reset-free counter rule'],
  [/1000 CUDA-Graph replays/, 'liveness gate is quantified'],
  [/[Ii]nterleave the arms/, 'interleaved A/B is mandated'],
  [/spread/i, 'per-rep spread is required alongside the median'],
  [/rank-max/i, 'rank-max is named as the metric that matters'],
  [/no measured overlap change is suspicious/, 'unexplained latency wins are rejected'],
];
for (const [re, what] of required) ok(re.test(doc), what);

// Anti-patterns must stay measured, not folkloric: each carries a number so a future round can
// tell "we tried it and it cost X" apart from "someone thought this was a bad idea".
const antipatterns = [
  [/~89 ns/, 'work-stealing claim queue cost'],
  [/4\.4535[\s\S]{0,40}4\.4476/, 'grid-stride unroll null result'],
  [/\+2\.33 ms/, 'LDS-reduction dead end'],
  [/\+0\.411 ms/, 'release-fence cost'],
  [/876\.6 µs/, 'skew peer-wait measurement'],
];
for (const [re, what] of antipatterns) ok(re.test(doc), `anti-pattern keeps its measurement: ${what}`);

ok(
  /Priority/.test(doc) && /no-payload control/.test(doc),
  'file ends with an actionable priority ordering',
);

console.log(
  failures === 0
    ? '\nPASS: `distributed` is reachable end-to-end and its doctrine carries its evidence.'
    : `\nFAILED: ${failures} assertion(s).`,
);
process.exit(failures === 0 ? 0 : 1);
