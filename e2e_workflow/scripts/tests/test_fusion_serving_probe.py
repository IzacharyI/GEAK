"""fusion_serving_probe.py against a fake SGLang server (no GPU).

The fake answers /generate, /metrics, /start_profile, /stop_profile and /get_model_info the
way the probe uses them: generate requests count as running while they sleep, and a
start_profile writes a per-rank trace whose step spans reflect what is resident."""
import gzip
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_serving_probe as probe  # noqa: E402


class FakeSglang:
    def __init__(self, dp=2, vocab=5000, step_name="TARGET_VERIFY"):
        self.dp, self.vocab, self.step_name = dp, vocab, step_name
        self.step_fmt = None
        self.running = 0
        self.per_rank = [0] * dp
        self.routed = []
        self.lock = threading.Lock()
        self.prompts = []
        self.profiled = None
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/metrics":
                    lines = []
                    for r in range(fake.dp):
                        per = fake.per_rank[r]
                        lines.append('sglang:num_running_reqs{dp_rank="%d"} %d' % (r, per))
                        lines.append('sglang:gen_throughput{dp_rank="%d"} %d' % (r, 10 if per else 0))
                    body = ("\n".join(lines) + "\n").encode()
                    self.send_response(200); self.end_headers(); self.wfile.write(body)
                elif self.path == "/get_model_info":
                    self._json(200, {"model_path": fake.model_dir})
                else:
                    self._json(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                req = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/generate":
                    ids = req["input_ids"]
                    fake.prompts.append(ids)
                    rank = req.get("routed_dp_rank")
                    fake.routed.append(rank)
                    # Without pinning, the fake balancer piles everything onto rank 0.
                    rank = 0 if rank is None else rank
                    assert max(ids) < fake.vocab, "token id past the vocabulary"
                    mnt = req["sampling_params"]["max_new_tokens"]
                    if mnt > 1:
                        with fake.lock:
                            fake.running += 1
                            fake.per_rank[rank] += 1
                        time.sleep(3.0)
                        with fake.lock:
                            fake.running -= 1
                            fake.per_rank[rank] -= 1
                    self._json(200, {"text": "", "meta_info": {
                        "prompt_tokens": len(ids), "completion_tokens": mnt,
                        "finish_reason": {"type": "length"}}})
                elif self.path == "/start_profile":
                    fake.profiled = req
                    out = req["output_dir"]
                    os.makedirs(out, exist_ok=True)
                    bs = fake.running // fake.dp
                    for r in range(fake.dp):
                        ev = []
                        for i in range(4):
                            name = fake.step_fmt or ("step[%s bs=%d]" % (fake.step_name, bs))
                            ev.append({"cat": "gpu_user_annotation", "name": name,
                                       "ts": i * 1000, "dur": 800})
                            ev.append({"cat": "kernel", "name": "k", "ts": i * 1000 + 10, "dur": 150})
                        with gzip.open(os.path.join(out, "t-TP-%d.trace.json.gz" % r), "wt") as fh:
                            json.dump({"traceEvents": ev}, fh)
                    self._json(200, {"ok": True})
                elif self.path == "/stop_profile":
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {})

        self.tmp = tempfile.mkdtemp(prefix="fake sglang ")
        self.model_dir = os.path.join(self.tmp, "model")
        os.makedirs(self.model_dir)
        with open(os.path.join(self.model_dir, "config.json"), "w") as fh:
            json.dump({"vocab_size": vocab}, fh)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


class ServingProbeTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeSglang()
        self.addCleanup(self.fake.close)
        self.tmp = tempfile.TemporaryDirectory(prefix="probe ")
        self.addCleanup(self.tmp.cleanup)
        self.points = os.path.join(self.tmp.name, "points.json")
        with open(self.points, "w") as fh:
            json.dump({"points": [
                {"point_id": "de_3", "phase": "DECODE", "batch_size": 3, "context_tokens": 3000},
                {"point_id": "pr_700", "phase": "PREFILL", "new_tokens": 700, "running": 12,
                 "cached_tokens": 2000}]}, fh)

    def run_probe(self, *extra, point="de_3"):
        out = os.path.join(self.tmp.name, point + ".journal.json")
        proc = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "fusion_serving_probe.py"), "--points", self.points,
             "--point-id", point, "--base-url", self.fake.url,
             "--trace-dir", os.path.join(self.tmp.name, point), "--dp-size", "2",
             "--record-shapes", "--with-stack", "--steady-timeout", "60", "--decode-tokens", "64",
             "--num-steps", "4", "--out", out, *extra],
            capture_output=True, text=True, timeout=180)
        with open(out) as fh:
            return proc, json.load(fh)

    def test_decode_point_uses_a_cached_shared_prefix_and_measures_step_cost(self):
        proc, doc = self.run_probe()
        self.assertEqual(doc["status"], "ok", proc.stdout + proc.stderr)
        events = [e["event"] for e in doc["journal"]]
        self.assertIn("prefix_cached", events)
        self.assertLess(events.index("prefix_cached"), events.index("steady_state_confirmed"))
        # 2 prefix warm-ups (one per DP rank), then 3 x 2 decode requests of prefix + suffix.
        warm = [p for p in self.fake.prompts if len(p) == 3000 - 512]
        batch = [p for p in self.fake.prompts if len(p) == 3000]
        self.assertEqual(len(warm), 2)
        self.assertEqual(len(batch), 6)
        self.assertTrue(all(p[:2488] == warm[0] for p in batch))
        self.assertEqual(len({tuple(p[2488:]) for p in batch}), 6)
        # Every request is pinned: one warm-up per rank, then the batch split evenly.
        self.assertEqual(self.fake.routed[:2], [0, 1] if self.fake.routed[0] == 0 else [1, 0])
        self.assertEqual(sorted(self.fake.routed[2:]), [0, 0, 0, 1, 1, 1])
        # TARGET_VERIFY (speculative decoding) is the target's decode step.
        self.assertEqual(doc["step_cost"]["us_per_step"], 150.0)
        self.assertEqual(doc["step_cost"]["basis"], "kernel_device_time")
        self.assertEqual(doc["step_cost"]["span_us_per_step"], 800.0)
        self.assertTrue(self.fake.profiled["record_shapes"])
        self.assertTrue(self.fake.profiled["with_stack"])
        done = [e for e in doc["journal"] if e["event"] == "request_done" and e["label"].startswith("decode")]
        self.assertTrue(all(e.get("completion_tokens") == 64 for e in done))

    def test_prefill_point_is_one_sequence_per_rank_over_its_cached_prefix(self):
        self.fake.step_fmt = "step[EXTEND bs=1 toks=700]"
        proc, doc = self.run_probe(point="pr_700")
        self.assertEqual(doc["status"], "ok", proc.stdout + proc.stderr)
        events = [e["event"] for e in doc["journal"]]
        # The cached context is in place BEFORE the window opens, so the window holds the extend.
        self.assertLess(events.index("prefix_cached"), events.index("armed"))
        warm = [p for p in self.fake.prompts if len(p) == 2000]
        seqs = [p for p in self.fake.prompts if len(p) == 2700]
        self.assertEqual(len(warm), 2)
        # `running` (12) is the decode population beside the batch, not a sequence count.
        self.assertEqual(len(seqs), 2)
        self.assertTrue(all(p[:2000] == warm[0] for p in seqs))

    def test_fully_distinct_prompts_on_request(self):
        proc, doc = self.run_probe("--shared-prefix", "0")
        self.assertEqual(doc["status"], "ok", proc.stdout + proc.stderr)
        self.assertNotIn("prefix_cached", [e["event"] for e in doc["journal"]])
        self.assertEqual(len(self.fake.prompts), 6)

    def test_random_ids_stay_inside_the_served_vocabulary(self):
        ids = probe.rand_ids(20000, 7, 5000)
        self.assertLess(max(ids), 5000)


class ClassifyTest(unittest.TestCase):
    def test_speculative_and_plain_step_names(self):
        self.assertEqual(probe._classify("step[TARGET_VERIFY bs=16]"), ("DECODE", 16))
        self.assertEqual(probe._classify("step[DECODE bs=8]"), ("DECODE", 8))
        self.assertEqual(probe._classify("step[EXTEND bs=2 toks=4096]"), ("PREFILL", 4096))
        self.assertIsNone(probe._classify("step[DRAFT_EXTEND_V2 bs=16]"))


if __name__ == "__main__":
    unittest.main()
