"""fusion_combined_ab.sh: each side is one owned server; bench (REUSE_SERVER=1) and gsm8k
both measure that same server, then the decision compares the sides. No GPU."""
import json
import os
import subprocess
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AB = os.path.join(SCRIPTS, "fusion_combined_ab.sh")

FAKE_ADAPTER = r'''
adapter_launch() {
  ${SERVER_LAUNCH_PREFIX:-} env python3 -c '
import http.server, json, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "served/name"}]}).encode()
        self.send_response(200); self.end_headers()
        if self.path == "/v1/models": self.wfile.write(body)
    def log_message(self, *a): pass
ov = os.environ.get("OVERLAY_PYTHONPATH", "")
if ov: print("[overlay-fuse1] ENGAGED", flush=True)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$PORT" > "$LOG" 2>&1 &
  SERVER_PID=$!
}
adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }
adapter_bench() { :; }
'''

# Stands in for bench_e2e.sh: proves it was asked to reuse the live server, then writes a
# summary whose throughput depends on the side's overlay.
FAKE_BENCH = r'''
[ "$REUSE_SERVER" = 1 ] || { echo "not reusing"; exit 9; }
curl -sf "$BASE_URL/health" >/dev/null || exit 8
tp=1000; [ -n "$OVERLAY_PYTHONPATH" ] && tp=1100
echo "{\"throughput_tok_s_median\": $tp, \"all_throughput\": [$tp]}" > "$OUT_DIR/bench_summary.json"
echo "E2E_SUMMARY output_tok_s=$tp mode=$GEAK_REPEAT_MODE repeats=$REPEATS isolated=${GEAK_ISOLATED_REPLICA:-0}"
'''

FAKE_GSM8K = r'''
import argparse, json
ap = argparse.ArgumentParser()
for a in ("--base-url", "--model", "--limit", "--max-tokens", "--seed", "--concurrency", "--out"):
    ap.add_argument(a)
ap.add_argument("--no-thinking", action="store_true")
a = ap.parse_args()
json.dump({"exact_match": 0.9, "n": 200, "model": a.model, "no_thinking": a.no_thinking},
          open(a.out, "w"))
'''


class CombinedAbSessionTest(unittest.TestCase):
    def test_both_sides_measure_their_own_server_and_a_decision_is_written(self):
        self._run_ab("warm_server", "mode=warm_server repeats=3 isolated=0")

    def test_replay_once_runs_one_leg_without_an_outer_warmup(self):
        self._run_ab("replay_once", "mode=legacy repeats=1 isolated=1")

    def _run_ab(self, mode, expect_line):
        with tempfile.TemporaryDirectory(prefix="fusion ab ") as tmp:
            paths = {}
            for name, body in (("fake.sh", FAKE_ADAPTER), ("bench.sh", FAKE_BENCH),
                               ("gsm8k.py", FAKE_GSM8K)):
                paths[name] = os.path.join(tmp, name)
                with open(paths[name], "w") as fh:
                    fh.write(body)
            out = os.path.join(tmp, "ab")
            env = dict(os.environ, OUT_DIR=out, CAND_OVERLAY=os.path.join(tmp, "ovl"),
                       MODEL="/models/fake", ADAPTER=paths["fake.sh"],
                       BENCH_SCRIPT=paths["bench.sh"], GSM8K_EVAL_SCRIPT=paths["gsm8k.py"],
                       SERVING_GPU_LOCK=os.path.join(tmp, "gpu.lock"), AB_REPEATS="3",
                       EXPECT_BANNERS="fuse1", NOISE_BAND_PCT="0.5", STALL_WINDOW_SEC="60",
                       AB_REPEAT_MODE=mode)
            proc = subprocess.run(["bash", AB], env=env, capture_output=True, text=True,
                                  timeout=300)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            for side, tp in (("base", 1000), ("cand", 1100)):
                with open(os.path.join(out, side, "bench_summary.json")) as fh:
                    self.assertEqual(json.load(fh)["throughput_tok_s_median"], tp)
                with open(os.path.join(out, side, "bench.log")) as fh:
                    self.assertIn(expect_line, fh.read())
                with open(os.path.join(out, side, "gsm8k.json")) as fh:
                    gsm = json.load(fh)
                self.assertEqual(gsm["model"], "served/name")
                self.assertTrue(gsm["no_thinking"])
            with open(os.path.join(out, "combined_decision.json")) as fh:
                decision = json.load(fh)
            self.assertEqual(decision["decision"], "accept", decision)


if __name__ == "__main__":
    unittest.main()
