"""fusion_serving_session.sh: one server, one command, identity-verified teardown.

Runs against a fake adapter whose 'server' is a tiny HTTP process, so no GPU is needed."""
import json
import os
import subprocess
import tempfile
import time
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = os.path.join(SCRIPTS, "fusion_serving_session.sh")

FAKE_ADAPTER = r'''
adapter_launch() {
  ${SERVER_LAUNCH_PREFIX:-} env python3 -c '
import http.server, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404); self.end_headers()
    def log_message(self, *a): pass
print("fake server overlay=%s" % os.environ.get("OVERLAY_PYTHONPATH", ""), flush=True)
http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
' "$PORT" > "$LOG" 2>&1 &
  SERVER_PID=$!
}
adapter_health() { curl -sf "${BASE_URL}/health" >/dev/null 2>&1; }
adapter_bench() { :; }
'''


class FusionServingSessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fusion session ")
        self.addCleanup(self.tmp.cleanup)
        self.adapter = os.path.join(self.tmp.name, "fake.sh")
        with open(self.adapter, "w") as fh:
            fh.write(FAKE_ADAPTER)

    def run_session(self, cmd, **env):
        out = os.path.join(self.tmp.name, "out")
        full = dict(os.environ, OUT_DIR=out, SESSION_CMD=cmd, MODEL="/models/fake",
                    ADAPTER=self.adapter, SERVING_GPU_LOCK=os.path.join(self.tmp.name, "gpu.lock"),
                    OVERLAY_PYTHONPATH="/ovl", STALL_WINDOW_SEC="60",
                    SERVER_STARTUP_TIMEOUT_SEC="60", **env)
        proc = subprocess.run(["bash", SESSION], env=full, capture_output=True, text=True,
                              timeout=120)
        return proc, out

    def test_command_sees_the_live_server_and_the_server_is_stopped_after(self):
        proc, out = self.run_session(
            'curl -sf "$BASE_URL/health" && echo "url=$BASE_URL reuse=$REUSE_SERVER '
            'pid=[${SERVER_PID:-}] session=$SESSION_SERVER_PID" > "$OUT_DIR/cmd.txt"')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        with open(os.path.join(out, "cmd.txt")) as fh:
            line = fh.read()
        self.assertIn("reuse=1", line)
        # A bench_e2e.sh inside the command must not inherit the pid its teardown would kill.
        self.assertIn("pid=[]", line)
        session_pid = int(line.split("session=")[1].split()[0])
        time.sleep(0.5)
        self.assertFalse(os.path.exists("/proc/%d" % session_pid) and
                         open("/proc/%d/stat" % session_pid).read().split()[2] != "Z",
                         "server outlived its session")
        with open(os.path.join(out, "server_start.json")) as fh:
            self.assertEqual(json.load(fh)["status"], "ok")
        with open(os.path.join(out, "server.log")) as fh:
            self.assertIn("overlay=/ovl", fh.read())
        with open(os.path.join(out, "session_cmd.rc")) as fh:
            self.assertEqual(fh.read().strip(), "0")

    def test_command_failure_is_the_session_exit_code(self):
        proc, out = self.run_session("exit 7")
        self.assertEqual(proc.returncode, 7)

    def test_a_server_that_never_comes_up_skips_the_command(self):
        with open(self.adapter, "a") as fh:
            fh.write('\nadapter_launch() { ${SERVER_LAUNCH_PREFIX:-} sh -c "exit 1" > "$LOG" 2>&1 & SERVER_PID=$!; }\n')
        proc, out = self.run_session('touch "$OUT_DIR/ran"')
        self.assertEqual(proc.returncode, 2)
        self.assertFalse(os.path.exists(os.path.join(out, "ran")))
        with open(os.path.join(out, "server_start.json")) as fh:
            self.assertEqual(json.load(fh)["reason"], "died_early")


if __name__ == "__main__":
    unittest.main()
