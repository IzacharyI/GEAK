"""Tests for tools/ab_retry.py -- the retrying interleaved A/B runner.

Two things are tested and they are tested for different reasons.

The RETRY behaviour is what the tool was written for: wave 15 round 2 lost six verify
legs to a transient out-of-memory between torchrun launches, the frozen driver recorded
rc!=0 and advanced, the losses landed on whichever guards the schedule happened to
reach, and a reproduced +3.6% scored zero. A leg must therefore count only if an
attempt succeeded, and a leg that never succeeds must appear in `dropped_legs` rather
than be silently absent -- an absent leg is indistinguishable from a leg that was never
scheduled, which is how a partial claim reads as a complete one.

The kernel_time block is what makes the tool feed the attribution gate. This is the
kernel workflow, so the number that decides a win is the changed kernel's own time
against the kernels it replaced. Wave 15 round 3 promoted a candidate on +4.24%
end-to-end whose fused kernel was 2.18% SLOWER than the two it replaced; the entire
claim sat in the gaps between launches. test_e2e_win_on_slower_kernels_is_visible
reconstructs exactly that shape and asserts the driver reports both numbers, so the
contradiction is on the page instead of having to be noticed.

Stdlib + pytest only: no torch, no GPU, no bench. The bench is replaced by --fake-cmd,
which the tool supports for this purpose.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pytest

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "tools", "ab_retry.py")


def _load():
    spec = importlib.util.spec_from_file_location("ab_retry", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(arm, guard, e2e, s1, s2):
    return {"arm": arm, "guard": guard, "e2e_max_ms": e2e, "stage1_max_ms": s1,
            "stage2_combine_max_ms": s2, "residual_max_ms": round(e2e - s1 - s2, 4)}


# --------------------------------------------------------------------------- kernel_time

def test_e2e_win_on_slower_kernels_is_visible():
    """The wave-15 shape: end-to-end improves while the kernels get worse."""
    m = _load()
    g = [_rec("base", "8192_uniform", 4.5, 3.0, 1.0),
         _rec("cand", "8192_uniform", 4.3, 3.1, 1.0),
         _rec("base", "8192_uniform", 4.5, 3.0, 1.0),
         _rec("cand", "8192_uniform", 4.3, 3.1, 1.0)]
    kt = m.kernel_time_block(g, "base", "cand")

    assert m.pct_pairs(g, "base", "cand")[0] > 0, "e2e says the candidate won"
    assert kt["kernel_sum_pct"] < 0, "and the kernel sum says it did not -- both must be reported"
    assert kt["base"]["kernel_sum_ms"] == 4.0
    assert kt["cand"]["kernel_sum_ms"] == 4.1
    # the win is in the gaps, and the residual is where that becomes checkable
    assert kt["base"]["residual_ms"] == 0.5
    assert kt["cand"]["residual_ms"] == 0.2


def test_sign_convention_is_not_inverted():
    """Positive kernel_sum_pct must mean the candidate's kernels are FASTER."""
    m = _load()
    g = [_rec("base", "g", 4.5, 3.0, 1.0), _rec("cand", "g", 4.5, 2.5, 1.0)]
    assert m.kernel_time_block(g, "base", "cand")["kernel_sum_pct"] > 0


def test_half_filled_record_set_does_not_crash():
    """The doc is rewritten after EVERY leg, so this runs before the candidate has one.

    Caught by the self-test on the merge: the first version divided into a None and
    took the whole run down after leg 1, destroying the incremental-write guarantee
    that is the reason the aggregate is rewritten each time.
    """
    m = _load()
    kt = m.kernel_time_block([_rec("base", "g", 4.5, 3.0, 1.0)], "base", "cand")
    assert kt["kernel_sum_pct"] is None
    assert kt["cand"]["n"] == 0


# --------------------------------------------------------------------------- end to end

FAKE = r"""#!/bin/bash
if [[ "$FAKE_TAG" == *_cand ]]; then S1=3.1; S2=1.0; E=4.3; else S1=3.0; S2=1.0; E=4.5; fi
if [[ -n "$FAKE_FAIL_ARM" && "$FAKE_TAG" == *_$FAKE_FAIL_ARM ]]; then
  echo "hip failed with out of memory" >&2; exit 255
fi
for r in 0 1; do echo "[megamoe] path=MEGA rank=$r"; done
echo "[RESULT] tokens=8192 mega_e2e=${E}/${E}ms stage1=${S1}/${S1}ms stage2_combine=${S2}/${S2}ms"
"""

PLAN = {"guards": ["8192_uniform"], "blocks": 2, "iters": 3, "base_arm": "base",
        "arms": [{"name": "base", "tree": "/tmp"}, {"name": "cand", "tree": "/tmp"}]}


def _run(tmp, env_extra=None, attempts=1):
    fake = os.path.join(tmp, "fake.sh")
    open(fake, "w").write(FAKE)
    plan = os.path.join(tmp, "plan.json")
    json.dump(PLAN, open(plan, "w"))
    out = os.path.join(tmp, "out.json")
    env = dict(os.environ)
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "2",
                        "--expect-path", "MEGA",
                        "--attempts", str(attempts), "--retry-sleep", "0"],
                       capture_output=True, text=True, env=env, timeout=180)
    return p, json.load(open(out))


def test_full_run_completes_and_reports_both_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp)
        assert p.returncode == 0, p.stdout + p.stderr
        assert doc["claim_complete"] is True
        assert doc["n_dropped"] == 0
        cmp_ = doc["by_guard"]["8192_uniform"]["base_vs_cand"]
        assert cmp_["n"] == 2
        assert min(cmp_["e2e_pairs"]) > 0
        assert cmp_["kernel_time"]["kernel_sum_pct"] < 0


def test_a_leg_that_never_succeeds_is_dropped_loudly_not_omitted():
    with tempfile.TemporaryDirectory() as tmp:
        p, doc = _run(tmp, {"FAKE_FAIL_ARM": "cand"}, attempts=2)
        assert doc["n_dropped"] == 2, "both candidate legs failed and both must be recorded"
        assert all(d["arm"] == "cand" for d in doc["dropped_legs"])
        assert doc["dropped_legs"][0]["reason"].startswith("rc=")
        assert len(doc["dropped_legs"][0]["attempts"]) == 2, "it was retried before being dropped"
        assert "DROPPED" in p.stdout, "and the drop is printed, not only filed"
        # the surviving arm must not silently become a one-sided claim
        assert doc["by_guard"]["8192_uniform"]["base_vs_cand"]["e2e_pairs"] == []


def test_marker_count_violation_voids_the_leg():
    """An incomplete marker set is not a measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fake.sh")
        open(fake, "w").write(FAKE)
        plan = os.path.join(tmp, "plan.json")
        json.dump(PLAN, open(plan, "w"))
        out = os.path.join(tmp, "out.json")
        subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "8",
                        "--attempts", "1", "--retry-sleep", "0"],
                       capture_output=True, text=True, timeout=180)
        doc = json.load(open(out))
        assert doc["n_dropped"] == 4
        assert "marker count 2 != 8" in doc["dropped_legs"][0]["reason"]


def test_marker_value_violation_voids_the_leg():
    """Eight plausible markers are still void when the intended arm fell back."""
    with tempfile.TemporaryDirectory() as tmp:
        fake = os.path.join(tmp, "fake.sh")
        open(fake, "w").write(FAKE)
        plan_data = json.loads(json.dumps(PLAN))
        for arm in plan_data["arms"]:
            arm["expect_paths"] = ["SCATTERED"]
        plan = os.path.join(tmp, "plan.json")
        json.dump(plan_data, open(plan, "w"))
        out = os.path.join(tmp, "out.json")
        subprocess.run([sys.executable, TOOL, "--plan", plan, "--out", out,
                        "--fake-cmd", f"bash {fake}", "--expect-markers", "2",
                        "--attempts", "1", "--retry-sleep", "0"],
                       capture_output=True, text=True, timeout=180)
        doc = json.load(open(out))
        assert doc["n_dropped"] == 4
        assert "path markers ['MEGA'] != expected ['SCATTERED']" in \
            doc["dropped_legs"][0]["reason"]


def test_rotation_changes_arm_position_between_blocks():
    """A fixed arm-within-block position manufactured ~1pp of separation on a null."""
    m = _load()
    seq = m.rotate_sequence(["g"], ["base", "cand"], 2)
    assert [a for _, a in seq] == ["base", "cand", "cand", "base"]


def test_jit_dir_is_overridable_and_defaults_to_a_cache_not_a_checkout():
    """Pointing AITER_JIT_DIR at an AITER checkout is a write path into the baseline."""
    src = open(TOOL).read()
    assert 'os.environ.get(\n        "AB_RETRY_JIT_DIR"' in src or \
           'os.environ.get("AB_RETRY_JIT_DIR"' in src
    assert "aiter_jit_cache" in src


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
