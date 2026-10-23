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


def main():
    mod = load_skill_module()
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
