"""fusion_reuse_verdicts.py: the best earlier PASSING verdict per candidate seeds this round."""
import json
import os
import sys
import tempfile
import unittest

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS)
import fusion_reuse_verdicts as frv  # noqa: E402


def _verdict(root, run, name, **doc):
    path = os.path.join(root, run, "verdict", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh)


class ReuseVerdictsTest(unittest.TestCase):
    def test_best_passing_verdict_per_candidate_is_copied(self):
        with tempfile.TemporaryDirectory() as tmp:
            _verdict(tmp, "r5", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=1.50)
            _verdict(tmp, "r8", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=0.97)
            _verdict(tmp, "r8", "c23__self.json", candidate_id="c23", parity="pass", isolated_speedup=1.05)
            _verdict(tmp, "r8", "c14__self.json", candidate_id="c14", parity="fail", isolated_speedup=0.0)
            _verdict(tmp, "r6", "e01.rundefined__self.json", candidate_id="e01.rundefined", parity="pass", isolated_speedup=2.0)
            # A verdict an aggregate moved into a subdirectory still counts.
            _verdict(tmp, "r7", "attempted/c21__self.json", candidate_id="c21", parity="pass", isolated_speedup=1.04)
            out = os.path.join(tmp, "now", "verdict")
            argv = ["--to", out] + sum((["--from", os.path.join(tmp, r)] for r in ("r5", "r6", "r7", "r8")), [])
            frv.main(argv)
            self.assertEqual(sorted(os.listdir(out)), ["c21__self.json", "c23__self.json", "e01.r4__self.json"])
            with open(os.path.join(out, "e01.r4__self.json")) as fh:
                doc = json.load(fh)
            self.assertEqual(doc["isolated_speedup"], 1.50)
            self.assertTrue(doc["reused_from"].endswith("r5/verdict/e01.r4__self.json"))

    def test_this_rounds_own_verdict_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            _verdict(tmp, "old", "c23__self.json", candidate_id="c23", parity="pass", isolated_speedup=1.2)
            _verdict(tmp, "now", "c23__self.json", candidate_id="c23", parity="pass", isolated_speedup=1.01)
            frv.main(["--to", os.path.join(tmp, "now", "verdict"), "--from", os.path.join(tmp, "old")])
            with open(os.path.join(tmp, "now", "verdict", "c23__self.json")) as fh:
                self.assertEqual(json.load(fh)["isolated_speedup"], 1.01)

    def test_prefer_best_keeps_the_fastest_attempt_of_this_round(self):
        with tempfile.TemporaryDirectory() as tmp:
            _verdict(tmp, "now", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=0.97)
            _verdict(tmp, "now", "c23__self.json", candidate_id="c23", parity="pass", isolated_speedup=1.30)
            _verdict(tmp, "now", "e04.r3__self.json", candidate_id="e04.r3", parity="not_engaged", isolated_speedup=0)
            _verdict(tmp, "a2", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=1.22)
            _verdict(tmp, "a3", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=1.48)
            _verdict(tmp, "a2", "c23__self.json", candidate_id="c23", parity="pass", isolated_speedup=1.10)
            _verdict(tmp, "a3", "e04.r3__self.json", candidate_id="e04.r3", parity="pass", isolated_speedup=1.21)
            out = os.path.join(tmp, "now", "verdict")
            frv.main(["--prefer-best", "--to", out, "--from", os.path.join(tmp, "a2"),
                      "--from", os.path.join(tmp, "a3")])
            speed = {}
            for name in ("e01.r4", "c23", "e04.r3"):
                with open(os.path.join(out, name + "__self.json")) as fh:
                    speed[name] = json.load(fh)["isolated_speedup"]
            self.assertEqual(speed, {"e01.r4": 1.48, "c23": 1.30, "e04.r3": 1.21})
            # the replaced first attempt is kept, but not as a verdict the harness would read
            self.assertTrue(os.path.isfile(os.path.join(out, "e01.r4__self.json.prev")))
            self.assertFalse(os.path.exists(os.path.join(out, "c23__self.json.prev")))

    def test_a_renumbered_board_gets_verdicts_by_window(self):
        def board(root, run, doc):
            path = os.path.join(root, run, "fusion_capture", "fusion", "fusion_topk.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(doc, fh)
            return path
        with tempfile.TemporaryDirectory() as tmp:
            board(tmp, "r5", {"execution_list": [{"exec_id": "e01", "candidate_ids": ["c01"],
                                                  "row_ids": ["a", "b", "c"],
                                                  "alt_ladder": [{"rung": 0}, {"rung": 4, "row_ids": ["a", "b"]}]}]})
            _verdict(tmp, "r5", "e01.r4__self.json", candidate_id="e01.r4", parity="pass", isolated_speedup=1.43)
            _verdict(tmp, "r5", "c01__self.json", candidate_id="c01", parity="pass", isolated_speedup=1.2)
            new = board(tmp, "now", {"execution_list": [
                {"exec_id": "e01", "candidate_ids": ["c09"], "row_ids": ["x", "y"],
                 "alt_ladder": [{"rung": 0}, {"rung": 1, "row_ids": ["a", "b"]}]},
                {"exec_id": "e02", "candidate_ids": ["c01"], "row_ids": ["p", "q"]}]})
            out = os.path.join(tmp, "now", "verdict")
            frv.main(["--to", out, "--to-topk", new, "--from", os.path.join(tmp, "r5")])
            # e01.r4's window is e01.r1 on the new board; old c01 (a,b,c) is not on it, and the
            # new c01 is a different window, so nothing lands there.
            self.assertEqual(os.listdir(out), ["e01.r1__self.json"])
            with open(os.path.join(out, "e01.r1__self.json")) as fh:
                doc = json.load(fh)
            self.assertEqual((doc["candidate_id"], doc["reused_candidate_id"]), ("e01.r1", "e01.r4"))


if __name__ == "__main__":
    unittest.main()
