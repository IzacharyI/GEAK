#!/usr/bin/env node
// A resumed wave whose analyze fast path finds nothing must not proceed with an empty ladder.
//
// roles/tech_lead.md lets analyze skip the cold re-read when INCREMENTAL_RESUME is set, on the
// condition that it does the full analysis if no prior roadmap exists. kernel_workflow.js cannot
// check that condition -- it has no filesystem -- so the only observable is the returned schema.
//
// Wave 15: bootstrap_task.sh assembles a fresh EVAL_DIR, the prior roadmap was not in it, the fast
// path returned a valid schema with no candidate_directions, and three rounds ran with no ladder.
// The rung ids were carried forward by hand from wave 14 by individual engineers, and D2 went
// unspent for three waves. The LADDER MISSING caveat fired every round and changed nothing.
//
// Each assertion below is a way this fallback could be wrong in a way nobody would notice:
//   - it fires on a NON-resumed run, turning every genuinely-ladderless first wave into a double
//     analyze call (cost, and it would mask the real finding)
//   - it fails to fire on the exact wave-15 shape
//   - it fires when a ladder IS present, i.e. it is reading the wrong field
//   - it is defined but never called, or called after the ladder gate has already read `analysis`
'use strict';

const fs = require('fs');
const path = require('path');

const WF = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(WF, 'kernel_workflow.js'), 'utf8');

let failures = 0;
const ok = (cond, msg) => {
  if (cond) console.log('  ok:', msg);
  else { console.error('  FAIL:', msg); failures++; }
};

const m = src.match(/\/\/ <<REPLAY:analyze_resume_fallback>>([\s\S]*?)\/\/ <<\/REPLAY:analyze_resume_fallback>>/);
if (!m) { console.error('  FAIL: no <<REPLAY:analyze_resume_fallback>> region — nothing to test'); process.exit(1); }
const { analyzeResumeDegenerate } =
  // eslint-disable-next-line no-new-func
  new Function(`${m[1]}\nreturn { analyzeResumeDegenerate };`)();

const LADDER = { candidate_directions: [
  { id: 'D0', title: 'D0 Instrumentation' },
  { id: 'D2', title: 'D2 Per-token readiness' },
] };

console.log('\n# the wave-15 shape');
{
  const d = analyzeResumeDegenerate(true, { kernel_type: 'flydsl', candidate_directions: [] });
  ok(d.retry === true, 'a resumed analyze that returns an empty ladder triggers the full re-run');
  ok(/INCREMENTAL_RESUME/.test(d.reason) && /roadmap/.test(d.reason),
     'and the reason names the flag and the artifact, so the log says what happened rather than that something happened');
}
{
  ok(analyzeResumeDegenerate(true, {}).retry === true,
     'a missing candidate_directions key is the same failure as an empty one');
  ok(analyzeResumeDegenerate(true, null).retry === true,
     'and so is no analysis object at all');
  ok(analyzeResumeDegenerate(true, { candidate_directions: [{}, { title: '' }] }).retry === true,
     'rungs with neither id nor title are not a ladder — this is the same filter the ladder gate uses');
}

console.log('\n# it must not fire anywhere else');
{
  ok(analyzeResumeDegenerate(false, { candidate_directions: [] }).retry === false,
     'a FIRST wave with no ladder is a real finding and must not be retried into silence');
  ok(analyzeResumeDegenerate(true, LADDER).retry === false,
     'a resumed wave that did inherit its ladder proceeds untouched — the fast path is the point');
  ok(!analyzeResumeDegenerate(true, LADDER).reason,
     'and says nothing, so the log is not noisy on the normal path');
}

console.log('\n# it is wired in, once, and before anything reads the ladder');
{
  ok(/const d = analyzeResumeDegenerate\(INCREMENTAL, analysis\)/.test(src),
     'the check is called with the resume flag and the analyze result');
  ok(/let analysis = await agentT\(/.test(src),
     '`analysis` is a let, so the recovered full analysis can replace the degenerate one');
  ok(/tech_lead:analyze:full/.test(src),
     'the re-run gets its own label — two analyze calls that look identical in the progress tree are unreadable');
  const reRun = src.indexOf("'tech_lead:analyze:full'");
  const ladderRead = src.indexOf('const LADDER = (analysis');
  ok(reRun > 0 && ladderRead > reRun,
     'the re-run happens BEFORE LADDER is computed — recovering a ladder nothing reads is not a fix');
  ok((src.match(/tech_lead:analyze:full/g) || []).length === 1,
     'and it happens once: a retry loop against a phase this expensive is how a wave spends its budget on nothing');
}

console.log(failures ? `\n${failures} FAILED` : '\nall passed');
process.exit(failures ? 1 : 0);
