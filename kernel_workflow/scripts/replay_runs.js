#!/usr/bin/env node
/*
 * replay_runs.js — re-decide finished runs with today's workflow, on no GPU at all.
 *
 * The problem this exists for: every change to the decision logic in `kernel_workflow.js` was, until
 * now, validated by a test that greps the source for the sentence describing the change. That proves
 * the prose is present. It does not prove the loop behaves differently, and it is the same
 * "prose is advisory" failure the workflow warns its own agents about, one level up. The honest
 * alternative — launch a wave — costs an 8-card lease and half a day, so in practice the check was
 * skipped and changes landed unfalsified.
 *
 * But a finished run tree already contains the inputs to almost every decision the workflow makes.
 * `setup_ab*.json` records each individual benchmark invocation: arm, guard, rep, rank-mean,
 * rank-max, exit code, and the path markers the process printed. From those raw runs this script
 * recomputes the paired effect, then feeds it through the CURRENT gate arithmetic — lifted verbatim
 * out of `kernel_workflow.js` between its `<<REPLAY:pc_gate>>` markers, so the replay cannot drift
 * away from the thing it is replaying.
 *
 * The output is a deterministic snapshot. Commit it; change the workflow; run this again; diff. A
 * change that flips a historical verdict shows up as a line in that diff, in seconds. A change that
 * flips nothing is either inert or aimed at something the corpus does not cover — and being made to
 * say which is the point.
 *
 *   node scripts/replay_runs.js --runs <dir> [--runs <dir> ...] [--json] \
 *        [--snapshot <file>] [--check <file>]
 *
 * Exit codes: 0 ok / snapshot matches, 1 usage or corpus error, 4 --check found a difference.
 *
 * What it does NOT replay: anything that depends on an agent's judgement. Agent outputs are recorded
 * verdicts, not reproducible functions, so a replay of them would only confirm the recording. This
 * covers the deterministic layer — aggregation and gating — which is exactly the layer where a
 * silent arithmetic change does the most damage.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const WF = path.join(ROOT, 'kernel_workflow.js');

// ---------------------------------------------------------------- args
const argv = process.argv.slice(2);
const roots = [];
let asJson = false, snapshotOut = null, checkAgainst = null;
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--runs') roots.push(argv[++i]);
  else if (a === '--json') asJson = true;
  else if (a === '--snapshot') snapshotOut = argv[++i];
  else if (a === '--check') checkAgainst = argv[++i];
  else if (a === '-h' || a === '--help') { console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0]); process.exit(0); }
  else { console.error(`replay_runs: unknown argument: ${a}`); process.exit(1); }
}
if (!roots.length) { console.error('replay_runs: at least one --runs <dir> is required'); process.exit(1); }

// ---------------------------------------------------------------- lift the live gate
// Verbatim, between markers. If someone deletes the markers the replay stops rather than silently
// falling back to a stale copy of the arithmetic — a replay that has quietly forked from the
// workflow is worse than no replay, because it reports confidence in the wrong code.
function liftGate() {
  const src = fs.readFileSync(WF, 'utf8');
  const m = src.match(/\/\/ <<REPLAY:pc_gate>>[\s\S]*?\n([\s\S]*?)\n\s*\/\/ <<\/REPLAY:pc_gate>>/);
  if (!m) {
    console.error('replay_runs: the <<REPLAY:pc_gate>> markers are missing from kernel_workflow.js.');
    console.error('  The gate arithmetic is lifted from between them; without them this script would');
    console.error('  be replaying a copy that can drift. Restore the markers around the block that');
    console.error('  computes mLo/mHi/wantSign/tooSmall/absurd/overshoot/ok.');
    process.exit(1);
  }
  return new Function('pc', 'POSITIVE_CONTROL', `
    const lo = Number(POSITIVE_CONTROL.expected_pct_lo);
    const hi = Number(POSITIVE_CONTROL.expected_pct_hi);
    const got = Number(pc.measured_pct);
${m[1]}
    return { ok, ran, tooSmall, absurd, overshoot, undershoot, nullQuiet, resolved, signUnanimous, mLo, mHi, wantSign };
  `);
}
const gate = liftGate();

// ---------------------------------------------------------------- corpus
function walk(dir, out = []) {
  let ents;
  try { ents = fs.readdirSync(dir, { withFileTypes: true }); } catch { return out; }
  for (const e of ents) {
    const p = path.join(dir, e.name);
    // The workspaces under a run tree are full AITER checkouts with thousands of tuned-config JSONs.
    // Descending into them turns a 2-second replay into a minute of stat() calls for nothing.
    if (e.isDirectory()) {
      if (/^(workspace|baseline|baseline_copy|optimized|validation_workspace|inherited_assets|\.git|__pycache__)$/.test(e.name)) continue;
      walk(p, out);
    } else if (/^setup_ab.*\.json$/.test(e.name)) out.push(p);
  }
  return out;
}

// ---------------------------------------------------------------- schema normalisation
//
// THE REPLAY MUST READ WHAT THE ENGINEERS ACTUALLY WRITE, AND SAY SO WHEN IT CANNOT.
//
// This script was written against one artifact shape: `{arms:[{name,env,ws}], base, guards, runs:[
// {arm,guard,rep,max,rc}]}`. Wave 7's benchmark engineer wrote `{records:[{arm,guard,env,tree,
// e2e_max_ms,idx}]}` instead — same information, different names, and nothing in the workflow ever
// told them which to use. The replay read it, found no `runs` key, and printed `runs=0 void=0` with
// an empty gate table. That is a FAIL-OPEN: the one instrument built so that gate changes are
// validated by behaviour rather than by grepping the source silently reported "nothing to re-decide"
// for a corpus that contained a complete 6-pair control. Every gate change validated against it
// would have been validated against nothing.
//
// So: accept both shapes, and treat an artifact that parses but yields no usable pair as a LOUD
// error, never as a quiet zero. `--check` fails on an empty corpus for the same reason.
const NUMS = ['max', 'e2e_max_ms', 'rank_max_ms', 'latency_max_ms'];
const MEANS = ['mean', 'e2e_mean_ms', 'rank_mean_ms'];

// An arm is the BASE of its pair when its switch is off. "Off" is not only an absent env: wave 7's
// base arm sets `MEGAMOE_CTRL_SPIN=0`, which is the same switch at zero. Treat an env whose every
// value is falsy as inert, because that is what the engineer meant by it.
function envInert(env) {
  const e = env || {};
  const keys = Object.keys(e);
  if (!keys.length) return true;
  return keys.every((k) => e[k] === 0 || e[k] === '0' || e[k] === '' || e[k] === false || e[k] === null);
}

function normalise(doc) {
  const raw = Array.isArray(doc.runs) ? doc.runs
    : Array.isArray(doc.records) ? doc.records : [];
  // `rep` may be absent (wave 7 records a global interleave index `idx` instead). In an interleaved
  // A/B the k-th occurrence of each arm IS pair k — that is what interleaving means — so derive the
  // rep from per-arm occurrence order rather than dropping every pair for want of a field.
  const seenPerArm = new Map();
  const runs = raw.map((r) => {
    const arm = r.arm;
    const n = (seenPerArm.get(arm) || 0); seenPerArm.set(arm, n + 1);
    const max = NUMS.map((k) => r[k]).find((v) => Number.isFinite(Number(v)));
    const mean = MEANS.map((k) => r[k]).find((v) => Number.isFinite(Number(v)));
    return {
      ...r, arm,
      guard: r.guard || (r.tokens != null && r.route ? `${r.tokens}_${r.route}` : undefined),
      rep: r.rep != null ? r.rep : n,
      max: Number.isFinite(Number(max)) ? Number(max) : Number(mean),
    };
  });
  // Arms come from the declaration when there is one, and are otherwise inferred from the records —
  // an inferred arm list is not a degraded one, it is the same list the engineer wrote down twice.
  let armMeta = new Map((Array.isArray(doc.arms) ? doc.arms : [])
    .map((a) => [a.name, { env: a.env || {}, ws: String(a.ws || a.tree || '') }]));
  if (!armMeta.size) {
    armMeta = new Map();
    for (const r of runs) {
      if (!r.arm || armMeta.has(r.arm)) continue;
      armMeta.set(r.arm, { env: r.env || {}, ws: String(r.ws || r.tree || '') });
    }
  }
  const arms = [...armMeta.keys()];
  const guards = Array.isArray(doc.guards) && doc.guards.length
    ? doc.guards : [...new Set(runs.map((r) => r.guard).filter(Boolean))];
  const base = doc.base && arms.includes(doc.base)
    ? doc.base
    : (arms.find((a) => envInert(armMeta.get(a).env)) || arms[0]);
  return { runs, arms, armMeta, guards, base };
}

// A run is USABLE only if it completed, reported a latency, AND printed the marker proving the path
// under test actually executed. Anything else is VOID. This is not bookkeeping pedantry: an arm that
// died or fell back silently still carries a plausible-looking number, and averaging it in is how a
// harness reports 1.000x for an experiment that never happened.
function voidReason(r) {
  if (r.rc !== 0) return `rc=${r.rc}`;
  if (r.ok === false) return 'ok=false';
  if (!Number.isFinite(Number(r.max))) return 'no rank-max';
  if (r.marker_ok === false) return 'path marker absent';
  return null;
}

// Paired, never pooled. Two arms timed minutes apart disagree by more than the effect, in either
// direction, so the only defensible statistic is the within-rep difference. A rep where either arm
// is VOID drops whole — half a pair is not a pair.
function pairedPct(runs, guard, armA, armB) {
  const byRep = new Map();
  for (const r of runs) {
    if (r.guard !== guard) continue;
    if (r.arm !== armA && r.arm !== armB) continue;
    const k = `${r.rep}`;
    if (!byRep.has(k)) byRep.set(k, {});
    // Several reps can share an index across a re-run; last write wins, which matches how the
    // benchmark engineer reports them.
    byRep.get(k)[r.arm] = r;
  }
  const deltas = []; const dropped = [];
  for (const [rep, pair] of [...byRep.entries()].sort()) {
    const a = pair[armA], b = pair[armB];
    if (!a || !b) { dropped.push(`rep${rep}:missing-arm`); continue; }
    const va = voidReason(a), vb = voidReason(b);
    if (va || vb) { dropped.push(`rep${rep}:${va ? armA + '(' + va + ')' : ''}${va && vb ? '+' : ''}${vb ? armB + '(' + vb + ')' : ''}`); continue; }
    // Positive = armB is FASTER than armA, on rank-max. The metric is rank-max because a collective
    // is gated by its slowest rank; rank-mean can improve while the operator gets slower.
    deltas.push((Number(a.max) - Number(b.max)) / Number(a.max) * 100);
  }
  if (!deltas.length) return { pct: NaN, n: 0, dropped, deltas, spread: NaN };
  const sorted = [...deltas].sort((x, y) => x - y);
  const med = sorted.length % 2 ? sorted[(sorted.length - 1) / 2]
    : (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2;
  return { pct: med, n: deltas.length, dropped, deltas, spread: sorted[sorted.length - 1] - sorted[0] };
}

// Two bands, deliberately. The first is the historical speedup control; the second is the synthetic
// slowdown the packaged task ships. Running both over the same corpus is what proves the gate is
// sign-agnostic on real data rather than only on the hand-written cases in the test suite.
const BANDS = {
  'speedup_3.55..4.93': { expected_pct_lo: 3.55, expected_pct_hi: 4.93 },
  'slowdown_-5.0..-2.5': { expected_pct_lo: -5.0, expected_pct_hi: -2.5, implausible_pct: 15.0 },
  // The same slowdown band declared as a TARGET rather than a measurement. Both are replayed over
  // every corpus on purpose: the pair is what shows, on real recorded data, that `magnitude` changes
  // the verdict only where it is supposed to — an under-read — and nowhere else.
  'slowdown_-5.0..-2.5_constructed': {
    expected_pct_lo: -5.0, expected_pct_hi: -2.5, implausible_pct: 15.0, magnitude: 'constructed' },
};

const records = [];
for (const root of roots) {
  const files = walk(path.resolve(root)).sort();
  if (!files.length) records.push({ kind: 'corpus', root, note: 'no setup_ab*.json found' });
  for (const f of files) {
    let doc;
    try { doc = JSON.parse(fs.readFileSync(f, 'utf8')); }
    catch (e) { records.push({ kind: 'unreadable', file: f, note: String(e.message) }); continue; }
    const { runs, arms, armMeta, guards, base: declaredBase } = normalise(doc);
    // Path is recorded relative to the corpus root so the snapshot is diffable across machines.
    const rel = path.relative(path.resolve(root), f);
    // Loud, not quiet. A file named setup_ab*.json that yields no arm, no guard or no timing is an
    // artifact this script cannot re-decide, and the only wrong thing to do with it is print a zero.
    //
    // ONE ARM IS NOT A BROKEN ARTIFACT. A seeded-baseline timing file legitimately holds a single arm
    // and nothing to pair it against; reporting it as an error every run is how a reader learns to
    // skim past the `!!` lines, which is how the real one gets missed. Separate the two cases.
    if (runs.length && arms.length === 1 && guards.length && runs.some((r) => Number.isFinite(r.max))) {
      records.push({
        kind: 'single_arm', root: path.basename(path.resolve(root)), file: rel,
        note: `${runs.length} run(s) of one arm (${arms[0]}) over ${guards.length} guard(s) — a baseline, not an A/B; no pair to re-decide.`,
      });
      continue;
    }
    if (!runs.length || arms.length < 2 || !guards.length || !runs.some((r) => Number.isFinite(r.max))) {
      records.push({
        kind: 'unusable', root: path.basename(path.resolve(root)), file: rel,
        note: `parsed but not replayable: ${runs.length} record(s), ${arms.length} arm(s), ` +
              `${guards.length} guard(s), ${runs.filter((r) => Number.isFinite(r.max)).length} with a rank-max. ` +
              `Expected per record: arm, guard (or tokens+route), and one of ${NUMS.join('/')}.`,
      });
      continue;
    }

    const voids = runs.map((r) => ({ arm: r.arm, guard: r.guard, rep: r.rep, why: voidReason(r) }))
      .filter((v) => v.why);
    records.push({
      kind: 'ab', root: path.basename(path.resolve(root)), file: rel,
      arms, guards, n_runs: runs.length, n_void: voids.length,
      voids: voids.slice(0, 8),
    });

    // WHICH TWO ARMS. This is the part a replay must not guess, because guessing it wrong is exactly
    // the mistake the original run made: a control whose two arms live in a separate frozen workspace
    // was reported against the run's own seeded baseline instead of against its OWN off-arm. That
    // reads ~+2.5% for an effect that is really ~+4.5%, and it trips the abort gate on a control that
    // passed. The two arms of a control differ ONLY by the gated switch, which on disk means: same
    // workspace, different env. So pair within a workspace whenever a workspace holds two arms, and
    // fall back to the declared base only when it does not.
    const wsOf = (n) => String((armMeta.get(n) || {}).ws || '');
    const nullArm = arms.find((a) => /null/i.test(a));
    const pairs = [];
    const byWs = new Map();
    for (const a of arms) { if (a === nullArm) continue; const k = wsOf(a); if (!byWs.has(k)) byWs.set(k, []); byWs.get(k).push(a); }
    for (const [, group] of byWs) {
      if (group.length < 2) continue;
      // Within a workspace the base is the arm with no env override — the switch is what defines the
      // candidate, so the arm that sets nothing is the one being compared against.
      const off = group.find((a) => envInert((armMeta.get(a) || {}).env)) || group[0];
      for (const on of group) if (on !== off) pairs.push({ cand: on, base: off, how: 'same-workspace' });
    }
    if (!pairs.length) {
      for (const a of arms) if (a !== declaredBase && a !== nullArm) pairs.push({ cand: a, base: declaredBase, how: 'declared-base' });
    }
    for (const guard of guards) {
      for (const { cand, base, how } of pairs) {
        const eff = pairedPct(runs, guard, base, cand);
        // The null arm is a byte-identical COPY of the run's own baseline, so it is always paired
        // against that baseline — never against whichever arm happens to be this pair's base. Pairing
        // it across workspaces measures the gap between two trees, which is not what "is the
        // interleave quiet?" asks.
        let nul = nullArm ? pairedPct(runs, guard, declaredBase, nullArm) : { pct: NaN, n: 0 };
        // Some engineers run the null arm in a separate invocation and report only its summary. A
        // recorded null is weaker evidence than a recomputed one — it cannot be audited pair by pair —
        // but it is much stronger than NaN, which silently turns `nullQuiet` false and converts a
        // benign overshoot into a hard failure. Use it, and label it as recorded in the snapshot.
        let nullSrc = nullArm ? 'recomputed' : 'absent';
        if (!nul.n && Number.isFinite(Number(doc.null_arm_pct))) {
          nul = { pct: Number(doc.null_arm_pct), n: Array.isArray(doc.null_pairs_pct) ? doc.null_pairs_pct.length : 0 };
          nullSrc = 'recorded';
        }
        // The gate reads the individual pairs, not only their medians: sign agreement and the worst
        // null pair are what let a small effect be told apart from drift. Supply both, or a replay
        // decides a case the live workflow would decide differently — which is the one thing a replay
        // must never do.
        const pc = {
          measured_pct: eff.pct, null_arm_pct: nul.pct, ran: eff.n > 0,
          control_pairs_pct: eff.deltas, null_pairs_pct: nul.deltas || doc.null_pairs_pct,
        };
        // Replayed against the band the PACKAGED task ships, not the band the historical run used:
        // the question a snapshot diff answers is "what would today's workflow decide", and today's
        // workflow is the one being changed.
        for (const [bandName, band] of Object.entries(BANDS)) {
          const v = gate(pc, band);
          v.wrongSign = Number.isFinite(eff.pct) && Math.sign(eff.pct) !== v.wantSign;
          records.push({
            kind: 'gate', root: path.basename(path.resolve(root)), file: rel, guard,
            pair: `${cand} vs ${base} (${how})`, band: bandName,
            measured_pct: round2(eff.pct), pairs: eff.n, spread_pp: round2(eff.spread),
            null_pct: round2(nul.pct), null_pairs: nul.n, null_src: nullSrc,
            dropped: eff.dropped.length ? eff.dropped.slice(0, 4) : undefined,
            verdict: verdictOf(v),
          });
        }
      }
    }
  }
}

function round2(x) { return Number.isFinite(x) ? Number(x.toFixed(2)) : null; }
function verdictOf(v) {
  if (!v.ran) return 'NOT_RUN';
  if (v.absurd) return 'FAIL_ABSURD';
  if (v.tooSmall) return v.wrongSign ? 'FAIL_WRONG_SIGN' : 'FAIL_INSENSITIVE';
  if (v.overshoot) return v.nullQuiet ? 'PASS_OVERSHOOT' : 'FAIL_OVERSHOOT_LOUD_NULL';
  if (v.undershoot) return 'PASS_UNDERSHOOT';
  return 'PASS';
}

// ---------------------------------------------------------------- report
const snap = { corpus: roots.map((r) => path.basename(path.resolve(r))).sort(), records };
const text = asJson ? JSON.stringify(snap, null, 1) : render(snap);

function render(s) {
  const L = [];
  L.push(`replay corpus: ${s.corpus.join(', ')}`);
  const ab = s.records.filter((r) => r.kind === 'ab');
  const gates = s.records.filter((r) => r.kind === 'gate');
  const bad = s.records.filter((r) => r.kind === 'unreadable' || r.kind === 'corpus' || r.kind === 'unusable');
  for (const r of bad) L.push(`  !! ${r.kind}: ${r.file || r.root} — ${r.note}`);
  for (const r of s.records.filter((x) => x.kind === 'single_arm')) L.push(`  -- ${r.file} — ${r.note}`);
  L.push('');
  L.push('A/B artifacts');
  for (const r of ab) {
    L.push(`  ${r.root}/${r.file}`);
    L.push(`    arms=[${r.arms.join(', ')}] guards=[${r.guards.join(', ')}] runs=${r.n_runs} void=${r.n_void}`);
    for (const v of r.voids) L.push(`      VOID ${v.arm}/${v.guard}/rep${v.rep}: ${v.why}`);
  }
  L.push('');
  L.push('gate verdicts (today\'s arithmetic, recorded data)');
  for (const g of gates) {
    L.push(`  ${g.root}/${g.file} [${g.guard}] ${g.pair}`);
    L.push(`    band=${g.band} measured=${g.measured_pct}% (${g.pairs} pairs, spread ${g.spread_pp}pp) ` +
           `null=${g.null_pct}% (${g.null_pairs}, ${g.null_src}) -> ${g.verdict}` +
           (g.dropped ? `  dropped: ${g.dropped.join(' ')}` : ''));
  }
  return L.join('\n') + '\n';
}

// AN EMPTY CORPUS IS NOT A PASS. `--check` exists to answer "did this change re-decide a finished
// run?", and a corpus with no gate verdicts in it cannot answer that question either way. Matching a
// snapshot of nothing against nothing is the fail-open this script was built to replace.
const nGates = records.filter((r) => r.kind === 'gate').length;
const nUnusable = records.filter((r) => r.kind === 'unusable' || r.kind === 'unreadable').length;
if (!nGates) {
  console.error(`replay_runs: the corpus produced NO gate verdicts` +
    (nUnusable ? ` (${nUnusable} artifact(s) parsed but were not replayable — see above)` : '') + '.');
  console.error('  Nothing was re-decided, so this run cannot validate a workflow change. Point --runs');
  console.error('  at a tree containing a completed A/B, or fix the artifact schema it reports above.');
  if (checkAgainst || snapshotOut) { process.stdout.write(text); process.exit(1); }
}

if (snapshotOut) { fs.writeFileSync(snapshotOut, text); console.log(`snapshot written: ${snapshotOut}`); }
if (checkAgainst) {
  const want = fs.existsSync(checkAgainst) ? fs.readFileSync(checkAgainst, 'utf8') : null;
  if (want === null) { console.error(`replay_runs: no snapshot at ${checkAgainst} to check against`); process.exit(1); }
  if (want === text) { console.log(`replay matches ${path.basename(checkAgainst)} — no historical verdict changed.`); process.exit(0); }
  const a = want.split('\n'), b = text.split('\n');
  console.error(`replay DIFFERS from ${path.basename(checkAgainst)} — a change to the workflow re-decided a finished run:`);
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    if (a[i] !== b[i]) { console.error(`  - ${a[i] === undefined ? '(eof)' : a[i]}`); console.error(`  + ${b[i] === undefined ? '(eof)' : b[i]}`); }
  }
  console.error('\nIf that is what the change was for, re-record with --snapshot. If it is not, the change');
  console.error('has a side effect on decisions you did not intend, and the diff above is where.');
  process.exit(4);
}
if (!snapshotOut) process.stdout.write(text);
