#!/usr/bin/env python3
"""Long-context requests sent alongside the engagement preflight's short gsm8k prompts.

Under DP attention every rank runs its own batch; with only short prompts all ranks see
near-identical small batches, so a fused path whose engagement depends on the local token
count takes the same branch everywhere. DeepSeek-V4 run 14's stack passed that preflight and
then aborted on an NCCL collective timeout (ranks at different collective sequence numbers)
eight minutes into the AgentX leg, where long prefills and small decodes mix. These requests
put long prefills on some ranks while others decode short ones.

    fusion_preflight_load.py --base-url URL --model NAME [--n 16] [--tokens 32768]

Prints one JSON line {sent, ok, failed}; exits 0 either way (the session watchdog and the
banner check judge the server, not this client).
"""
import argparse
import concurrent.futures
import json
import urllib.request


def one(base_url, model, words, max_tokens, timeout, seed):
    body = json.dumps({
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": ("%d " % seed) + ("lorem " * words)
                      + "\nSummarise the text above in one sentence."}],
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - a failure is counted, not raised
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=32768, help="approximate prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args(argv)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.n)) as pool:
        results = list(pool.map(lambda i: one(a.base_url, a.model, a.tokens, a.max_tokens,
                                              a.timeout, i), range(a.n)))
    print(json.dumps({"sent": a.n, "ok": sum(results), "failed": a.n - sum(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
