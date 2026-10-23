#!/usr/bin/env python3
"""KernelFusion front end to the kernel-extract skill's observed_workload.py.

Same CLI and output as
knowledge/kernel_extraction_skills/kernel-extract/scripts/observed_workload.py; the
skill's script runs unchanged. KernelFusion only registers one more scheduler-log
dialect before calling it, because hybrid sliding-window models (DeepSeek-V4, ...)
log the full and the SWA KV pool separately and the skill's dialect matches neither:

  prefill  `full token usage: .., swa token usage: ..` instead of `token usage: ..`
  decode   `#full token: N` instead of `#token: N`, and under speculation extra
           cap / block-accept / drafter-estimate fields before `cuda graph:`

`#full token` is the full-attention KV in use, i.e. the resident context the skill
takes its per-request median from. DP-attention prefixes (`DP0 TP0 EP0`) and an
optional `[step]` tag after `Prefill/Decode batch` are accepted. The extra fields are
skipped, never parsed.

It also reads the measured window from the CURRENT aiperf client log, whose phase lines are
`HH:MM:SS.fff NOTICE Phase profiling (profiling) started|complete` -- no date and no quoted
phase name, so the skill's `--bench-log` reader finds nothing and stops. When the skill's own
reader would not match, `--bench-log` is turned into `--window-start/--window-end` (the date is
the serving log's), and the leg stays scoped to its measured window.

Stdlib only.
"""
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPT = os.path.join(
    os.path.dirname(HERE), "knowledge", "kernel_extraction_skills", "kernel-extract",
    "scripts", "observed_workload.py")

HYBRID_PREFILL_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?) (?:TP|DP)(?P<rank>\d+)[^\]]*\] Prefill batch"
    r"(?: \[\d+\])?[,.] "
    r"#new-seq: (?P<new_seq>\d+), #new-token: (?P<new_token>\d+), "
    r"#cached-token: (?P<cached_token>\d+), "
    r"full token usage: (?P<usage>[\d.]+), (?:swa token usage: [\d.]+, )?(?:mamba usage: [\d.]+, )?"
    r"#running-req: (?P<running>\d+), #queue-req: (?P<queue>\d+)"
    r"(?:, #pending-token: (?P<pending>\d+))?"
    r"(?:, #replay-token: \d+)?"
    r"(?:, cuda graph: (?P<graph>\w+))?"
)

HYBRID_DECODE_RE = re.compile(
    r"^\[(?P<ts>[\d\-]+ [\d:]+(?:\.\d+)?) (?:TP|DP)(?P<rank>\d+)[^\]]*\] Decode batch"
    r"(?: \[\d+\])?[,.] "
    r"#running-req: (?P<running>\d+), #full token: (?P<token>\d+), "
    r"full token usage: (?P<usage>[\d.]+), "
    r"(?:#swa token: \d+, swa token usage: [\d.]+, )?"
    r"(?:accept len: (?P<accept_len>[\d.]+), )?"
    r"(?:accept rate: (?P<accept_rate>[\d.]+), )?"
    r"(?:[^,]*, )*?"
    r"cuda graph: (?P<graph>\w+), "
    r"gen throughput \(token/s\): (?P<gen_tps>[\d.]+), #queue-req: (?P<queue>\d+)"
)

DIALECT = "sglang-scheduler-hybrid-swa"


def load_skill_module(path=SKILL_SCRIPT):
    if not os.path.isfile(path):
        raise SystemExit("fusion_observed_workload: kernel-extract script missing: %s" % path)
    spec = importlib.util.spec_from_file_location("kernel_extract_observed_workload", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    register_hybrid_dialect(mod)
    return mod


def register_hybrid_dialect(mod):
    """Append the hybrid-SWA dialect to the skill module's LOG_DIALECTS (idempotent)."""
    if any(name == DIALECT for name, _ in mod.LOG_DIALECTS):
        return

    def prefill(line):
        m = HYBRID_PREFILL_RE.match(line)
        if not m:
            return None
        return dict(kind="prefill", rank=int(m.group("rank")), ts=mod._ts(m.group("ts")),
                    new_seq=int(m.group("new_seq")), new_token=int(m.group("new_token")),
                    cached_token=int(m.group("cached_token")),
                    running=int(m.group("running")), queue=int(m.group("queue")),
                    cuda_graph=mod._b(m.group("graph")))

    def decode(line):
        m = HYBRID_DECODE_RE.match(line)
        if not m:
            return None
        acc = m.group("accept_len")
        return dict(kind="decode", rank=int(m.group("rank")), ts=mod._ts(m.group("ts")),
                    running=int(m.group("running")), token=int(m.group("token")),
                    accept_len=float(acc) if acc else 1.0,
                    gen_tps=float(m.group("gen_tps")), queue=int(m.group("queue")),
                    cuda_graph=mod._b(m.group("graph")))

    mod.LOG_DIALECTS.append((DIALECT, [prefill, decode]))


AIPERF_PHASE_RE = re.compile(
    r"^(?P<t>\d\d:\d\d:\d\d(?:\.\d+)?)\s+\w+\s+Phase (?P<ph>\w+) \(\w+\) "
    r"(?P<ev>started|complete)\b")
SKILL_PHASE_RE = re.compile(r"phase ['\"]?\w+['\"]? (?:started|complete)")
LOG_DATE_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d) \d\d:\d\d:\d\d")


def _first_log_date(paths):
    for path in paths:
        op = __import__("gzip").open if path.endswith(".gz") else open
        with op(path, "rt", errors="replace") as fh:
            for line in fh:
                m = LOG_DATE_RE.match(line)
                if m:
                    return m.group(1)
    return None


def aiperf_window(bench_log, phase, log_paths):
    """(start, end) 'YYYY-MM-DD HH:MM:SS.f' strings from aiperf's phase lines, else None."""
    start = end = None
    with open(bench_log, "r", errors="replace") as fh:
        for raw in fh:
            for line in raw.split("\r"):
                m = AIPERF_PHASE_RE.match(line.strip())
                if not m or m.group("ph") != phase:
                    continue
                if m.group("ev") == "started":
                    start = start or m.group("t")
                elif "sending complete" not in line:
                    end = m.group("t")
    day = _first_log_date(log_paths)
    if not (start and end and day):
        return None
    import datetime as dt
    d0 = dt.date.fromisoformat(day)
    # The serving log starts before the client: a window that wraps midnight ends next day.
    d1 = d0 + dt.timedelta(days=1) if end < start else d0
    return "%s %s" % (d0.isoformat(), start), "%s %s" % (d1.isoformat(), end)


def translate_bench_log(argv):
    """Replace --bench-log with an explicit window when only aiperf's phase dialect is in it."""
    if "--bench-log" not in argv or "--window-start" in argv:
        return argv
    i = argv.index("--bench-log")
    bench_log = argv[i + 1]
    phase = argv[argv.index("--bench-phase") + 1] if "--bench-phase" in argv else "profiling"
    with open(bench_log, "r", errors="replace") as fh:
        if SKILL_PHASE_RE.search(fh.read()):
            return argv
    logs = []
    if "--log" in argv:
        for tok in argv[argv.index("--log") + 1:]:
            if tok.startswith("--"):
                break
            logs.append(tok)
    window = aiperf_window(bench_log, phase, logs)
    if window is None:
        return argv  # let the skill report it
    print("fusion_observed_workload: measured window from aiperf phase lines: %s -> %s"
          % window, file=sys.stderr)
    return argv[:i] + argv[i + 2:] + ["--window-start", window[0], "--window-end", window[1]]


def main():
    mod = load_skill_module()
    sys.argv = [sys.argv[0]] + translate_bench_log(sys.argv[1:])
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
