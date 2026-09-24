"""Single-file Expert Skill runtime-loader tests."""

import ast
import contextlib
import functools
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "expert_skill_runtime.py"
MEGAMOE_SKILL = (
    Path(__file__).resolve().parents[2]
    / "perf_knowledge"
    / "expert_skills"
    / "skills"
    / "megamoe_ep_mega_fusion"
    / "runtime_validation.py"
)
SPEC = importlib.util.spec_from_file_location("expert_skill_runtime", TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_extracts_exactly_one_runtime_block(tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text(
        "# Skill\n\n```expert-skill-runtime-python\n"
        "def main():\n    return 0\n"
        "```\n"
    )
    assert "def main" in MODULE.extract_runtime(skill)


def test_rejects_missing_or_duplicate_runtime_blocks(tmp_path):
    skill = tmp_path / "skill.md"
    skill.write_text("# no runtime\n")
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.extract_runtime(skill)
    block = "```expert-skill-runtime-python\ndef main(): return 0\n```\n"
    skill.write_text(block + block)
    with pytest.raises(ValueError, match="exactly one"):
        MODULE.extract_runtime(skill)


def test_executes_embedded_main_with_forwarded_arguments(tmp_path, monkeypatch):
    output = tmp_path / "result.txt"
    skill = tmp_path / "skill.md"
    skill.write_text(
        "```expert-skill-runtime-python\n"
        "import argparse\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--output', required=True)\n"
        "    parser.add_argument('--value', required=True)\n"
        "    args = parser.parse_args()\n"
        "    open(args.output, 'w').write(args.value)\n"
        "    return 0\n"
        "```\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--skill-file",
            str(skill),
            "--output",
            str(output),
            "--value",
            "ok",
        ],
    )
    assert MODULE.main() == 0
    assert output.read_text() == "ok"


def test_megamoe_runtime_uses_candidate_numeric_reference_and_graph():
    source = MODULE.load_runtime(MEGAMOE_SKILL)
    assert "--candidate-tree" in source
    assert "test_mega_moe_v2.py" in source
    assert "/sgl-workspace" not in source
    assert "with torch.cuda.graph" in source
    assert "helper._reference(" in source
    assert "direct graph-captured candidate vs numeric reference" in source


def test_megamoe_runtime_mutates_routes_and_writes_atomic_progress():
    source = MODULE.load_runtime(MEGAMOE_SKILL)
    assert "route_weights.copy_(new_weights)" in source
    assert "ids.copy_(new_ids)" in source
    assert '"arrival_jitter": True' in source
    assert '"routing_changes": args.replays' in source
    assert "os.replace(tmp, output)" in source
    assert "write_result(True)" in source


PASSING_ACTIVATION = {"status": "pass"}
PASSING_RESOURCES = {"status": "pass"}
PASSING_SPEEDUP = {"speedup_gate": "pass"}


def _runtime_namespace():
    namespace = {
        "__name__": "__megamoe_runtime_under_test__",
        "__file__": str(MEGAMOE_SKILL),
    }
    source = MODULE.load_runtime(MEGAMOE_SKILL)
    exec(compile(source, str(MEGAMOE_SKILL), "exec"), namespace)
    return namespace


def _runtime_tree():
    return ast.parse(MODULE.load_runtime(MEGAMOE_SKILL))


def _runtime_function(name):
    return next(
        node for node in _runtime_tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _main_try_body():
    return next(
        node for node in _runtime_function("main").body if isinstance(node, ast.Try)
    ).body


def _resource_evidence(tmp_path):
    common = {"kernarg": 440, "sgpr": 106, "resident_workgroups": 1}
    rows = {
        "fixed_128": {
            **common, "threads": 256, "group_segment": 32144, "vgpr": 168,
            "vgpr_spills": 0, "sgpr_spills": 106, "private": 0,
        },
        "large_8192": {
            **common, "threads": 512, "group_segment": 160400, "vgpr": 256,
            "vgpr_spills": 80, "sgpr_spills": 116, "private": 316,
        },
    }
    path = tmp_path / "resources.json"
    path.write_text(json.dumps({"cases": rows}))
    return path


def test_megamoe_runtime_cli_keeps_old_arguments_and_adds_optional_new_ones():
    options = {}
    for node in ast.walk(_runtime_function("main")):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            options[node.args[0].value] = {
                keyword.arg: (
                    ast.unparse(keyword.value) if keyword.arg == "type"
                    else ast.literal_eval(keyword.value)
                )
                for keyword in node.keywords
            }
    assert {
        "--candidate-tree", "--profile-adapter", "--profile", "--accuracy-cases",
        "--liveness-cases", "--routes", "--replays", "--numeric-checkpoint-interval",
        "--rtol", "--perf-iters", "--pairs", "--min-speedup", "--resource-evidence",
        "--json-output",
    } <= set(options)
    assert options["--frozen-baseline-ms"] == {"type": "float", "default": None}
    assert options["--denominator-tolerance"] == {"type": "float", "default": 0.05}
    assert options["--path-marker"] == {"default": "path=MEGA"}
    assert {
        name for name, spec in options.items() if spec.get("required")
    } == {"--candidate-tree", "--json-output"}


def test_megamoe_min_speedup_gates_frozen_absolute_ratio_not_same_tree_ratio():
    runtime = _runtime_namespace()
    summary = runtime["_speedup_summary"]
    regressed = summary(4.70, [6.0, 6.1, 5.9], [11.0, 11.2, 10.8], 1.03, 0.05)
    assert regressed["incremental_switch_speedup"] == pytest.approx(11.0 / 6.0)
    assert regressed["switch_off_ms_median"] == pytest.approx(11.0)
    assert regressed["absolute_speedup"] == pytest.approx(4.70 / 6.0)
    assert regressed["paired_rank_max_speedup"] == regressed["absolute_speedup"]
    assert regressed["speedup_gate"] == "fail"
    mismatch = regressed["denominator_mismatch"]
    assert mismatch["switch_off_ms"] == pytest.approx(11.0)
    assert mismatch["frozen_ms"] == pytest.approx(4.70)
    assert mismatch["deviation_pct"] == pytest.approx(100 * (11.0 - 4.70) / 4.70)
    assert "regressed shared code" in mismatch["note"]

    improved = summary(5.0, [4.5, 4.4, 4.6], [4.0, 4.1, 3.9], 1.03, 0.05)
    assert improved["incremental_switch_speedup"] < 1.0
    assert improved["absolute_speedup"] == pytest.approx(5.0 / 4.5)
    assert improved["speedup_gate"] == "pass"
    assert "changed shared code" in improved["denominator_mismatch"]["note"]

    matched = summary(4.70, [4.0] * 3, [4.8] * 3, 1.03, 0.05)
    assert "denominator_mismatch" not in matched
    assert matched["denominator_deviation_pct"] == pytest.approx(100 * 0.1 / 4.70)


@pytest.mark.parametrize("frozen", [None, 0.0, -4.7, float("nan"), float("inf")])
def test_megamoe_missing_frozen_baseline_is_not_evaluated_and_blocks_claim(frozen):
    runtime = _runtime_namespace()
    summary = runtime["_speedup_summary"](frozen, [4.0] * 3, [11.0] * 3, 1.03, 0.05)
    assert summary["absolute_speedup"] is None
    assert summary["paired_rank_max_speedup"] is None
    assert summary["speedup_gate"] == "not_evaluated: missing --frozen-baseline-ms"
    assert summary["incremental_switch_speedup"] == pytest.approx(2.75)
    assert "denominator_mismatch" not in summary
    assert runtime["_claim_blockers"](
        PASSING_ACTIVATION, PASSING_RESOURCES, summary
    ) == ["frozen_baseline_missing"]


def test_megamoe_min_speedup_is_compared_only_with_absolute_speedup():
    gates = []
    for node in ast.walk(_runtime_tree()):
        if isinstance(node, ast.Compare):
            operands = {ast.unparse(item) for item in (node.left, *node.comparators)}
            if any(operand.endswith("min_speedup") for operand in operands):
                gates.append(operands)
    assert gates == [{"absolute", "min_speedup"}]

    main = _runtime_function("main")
    parents = {
        child: parent
        for parent in ast.walk(main)
        for child in ast.iter_child_nodes(parent)
    }

    def ancestors(node):
        while node in parents:
            node = parents[node]
            yield node

    uses = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Attribute) and ast.unparse(node) == "args.min_speedup"
    ]
    assert uses
    for use in uses:
        assert any(
            isinstance(node, ast.JoinedStr)
            or (isinstance(node, ast.Call) and ast.unparse(node.func) == "_speedup_summary")
            for node in ancestors(use)
        )
    summary_call = next(
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "_speedup_summary"
    )
    assert [ast.unparse(arg) for arg in summary_call.args] == [
        "args.frozen_baseline_ms",
        "cand_readings",
        "switch_off_readings",
        "args.min_speedup",
        "args.denominator_tolerance",
    ]


def test_megamoe_restore_route_copies_uniform_snapshot_in_place():
    runtime = _runtime_namespace()

    class Tensor:
        def __init__(self, value):
            self.value = value

        def clone(self):
            return Tensor(self.value)

        def copy_(self, other):
            self.value = other.value
            return self

    weights, ids = Tensor("uniform-weights"), Tensor("uniform-ids")
    snapshot = (weights.clone(), ids.clone())
    weights.copy_(Tensor("skew-weights"))
    ids.copy_(Tensor("skew-ids"))
    runtime["_restore_route"]((weights, ids), snapshot)
    assert (weights.value, ids.value) == ("uniform-weights", "uniform-ids")
    with pytest.raises(ValueError):
        runtime["_restore_route"]((weights,), snapshot)


def test_megamoe_uniform_route_is_restored_before_launch_control_and_timing():
    body = _main_try_body()
    texts = [ast.unparse(statement) for statement in body]

    def index(predicate):
        return next(position for position, text in enumerate(texts) if predicate(text))

    capture_loop = index(lambda text: text.startswith("for tokens in cases:"))
    restore = index(
        lambda text: text == "_restore_route((route_weights, ids), uniform_route)"
    )
    launch = index(lambda text: "_profile_launch_count(" in text)
    control = index(
        lambda text: text.startswith("control_graph = _capture(torch, helper, control_body)")
    )
    timing = index(lambda text: text.startswith("for pair in range(args.pairs):"))
    assert capture_loop < restore < launch < control < timing
    loop = texts[capture_loop]
    assert "uniform_route = (route_weights.clone(), ids.clone())" in loop
    assert loop.index("uniform_route = (route_weights.clone()") < loop.index(
        "route_weights.copy_(new_weights)"
    )
    after_restore = "\n".join(texts[restore + 1:timing + 1])
    assert "route_weights.copy_" not in after_restore
    assert "ids.copy_" not in after_restore


def test_megamoe_activation_requires_path_marker_on_every_rank():
    runtime = _runtime_namespace()
    summary = runtime["_activation_summary"]
    observed = summary("path=MEGA", 8, 8, 1, [1] * 8)
    assert observed["status"] == "pass"
    assert observed["observed_ranks"] == 8
    assert observed["all_ranks_observed"] is True
    missing = summary("path=MEGA", 8, 7, 0, [1] * 7 + [0])
    assert missing["status"] == "fail"
    assert missing["observed_ranks"] == 7
    assert [row["rank"] for row in missing["per_rank"] if not row["observed"]] == [7]
    assert summary("path=MEGA", 8, 8, 0, [1] * 8)["status"] == "fail"
    assert summary("path=MEGA", 8, 8, 1, [1] * 7)["status"] == "fail"
    assert runtime["_claim_blockers"](
        missing, PASSING_RESOURCES, PASSING_SPEEDUP
    ) == ["activation_marker_missing"]


def test_megamoe_activation_is_observed_across_ranks_not_asserted():
    main = _runtime_function("main")
    statuses = []
    for node in ast.walk(main):
        if (
            isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "result['activation']"
            and isinstance(node.value, ast.Dict)
        ):
            statuses.extend(
                ast.literal_eval(value)
                for key, value in zip(node.value.keys, node.value.values)
                if key is not None and ast.literal_eval(key) == "status"
            )
    assert statuses == ["pending"]
    text = ast.unparse(main)
    assert "_capture(torch, helper, body, warmup)" in text
    assert "_captured_output_fds(output_path.parent, capture_prefix)" in text
    assert (
        "_reduce_marker_observation(torch, dist, device, marker_counts[0], world)"
        in text
    )
    assert (
        "_activation_summary(args.path_marker, world, observed_sum, observed_min, per_rank)"
        in text
    )
    reducer = ast.unparse(_runtime_function("_reduce_marker_observation"))
    assert "dist.ReduceOp.SUM" in reducer
    assert "dist.ReduceOp.MIN" in reducer
    capture = ast.unparse(_runtime_function("_captured_output_fds"))
    assert "os.dup(fd)" in capture
    assert "os.dup2(capture_fd, fd)" in capture
    assert "os.dup2(saved_fd, fd)" in capture


def test_megamoe_marker_capture_is_fd_level_restored_and_re_emitted(tmp_path):
    script = textwrap.dedent(
        r"""
        import os, sys
        namespace = {"__name__": "probe"}
        exec(compile(open(sys.argv[1]).read(), sys.argv[1], "exec"), namespace)
        capture = namespace["_captured_output_fds"]
        before = (os.fstat(1).st_ino, os.fstat(2).st_ino)
        with capture(sys.argv[2], "probe") as captured:
            print("[megamoe] path=MEGA stdout")
            sys.stderr.write("[megamoe] path=MEGA stderr")
            os.write(2, b"\n[megamoe] path=MEGA fd2\n")
        try:
            with capture(sys.argv[2], "probe"):
                raise RuntimeError("inside window")
        except RuntimeError:
            pass
        sys.stderr.write("\nafter-restore\n")
        restored = before == (os.fstat(1).st_ino, os.fstat(2).st_ino)
        leftover = len(os.listdir(sys.argv[2]))
        count = captured["text"].count("path=MEGA")
        print(f"count={count} restored={restored} leftover={leftover}")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(MEGAMOE_SKILL), str(tmp_path / "captures")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "count=3 restored=True leftover=0" in completed.stdout
    assert "[megamoe] path=MEGA stdout" in completed.stdout
    assert "[megamoe] path=MEGA stderr" in completed.stderr
    assert "[megamoe] path=MEGA fd2" in completed.stderr
    assert completed.stderr.rstrip().endswith("after-restore")


def test_megamoe_claim_complete_requires_resource_status_pass(tmp_path):
    runtime = _runtime_namespace()
    checks = runtime["_resource_checks"]
    assert checks("")["status"] == "delegated"
    assert checks(str(tmp_path / "absent.json"))["status"] == "missing"
    assert checks(str(_resource_evidence(tmp_path)))["status"] == "pass"
    blockers = runtime["_claim_blockers"]
    for resources in ({"status": "delegated"}, {"status": "missing"}, {}, None):
        assert blockers(PASSING_ACTIVATION, resources, PASSING_SPEEDUP) == [
            "resource_evidence_missing"
        ]
    assert blockers(PASSING_ACTIVATION, PASSING_RESOURCES, PASSING_SPEEDUP) == []


def test_megamoe_claim_complete_is_written_only_without_blockers():
    assert MODULE.load_runtime(MEGAMOE_SKILL).count("write_result(True)") == 1
    body = _main_try_body()
    texts = [ast.unparse(statement) for statement in body]
    claim = texts.index("write_result(True)")
    guard = body[claim - 1]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "claim_blockers"
    assert isinstance(guard.body[-1], ast.Return)
    assert texts[claim - 2] == "result['claim_blockers'] = claim_blockers"
    assert texts[claim - 3] == (
        "claim_blockers = _claim_blockers(result.get('activation'), "
        "result.get('resources'), speedup)"
    )


class _FakeTensor:
    """Content-tagged stand-in: ops derive tags, copy_ moves content in place."""

    def __init__(self, tag=None, value=0, shape=()):
        self.tag = tag
        self.value = value
        self.shape = tuple(shape)

    def _derive(self, name, other=None):
        return _FakeTensor(
            (name, self.tag, getattr(other, "tag", other)), shape=self.shape
        )

    def __add__(self, other):
        return self._derive("add", other)

    def __mul__(self, other):
        return self._derive("mul", other)

    def __lt__(self, other):
        return self._derive("lt", other)

    def __sub__(self, other):
        return self._derive("sub", other)

    def __truediv__(self, other):
        return _FakeTensor(value=self.value / other.value)

    def __getitem__(self, _):
        return self

    def softmax(self, dim):
        return self._derive("softmax", dim)

    def contiguous(self):
        return self

    def float(self):
        return self

    def to(self, _dtype):
        return self

    def item(self):
        return self.value

    def clone(self):
        return _FakeTensor(self.tag, self.value, self.shape)

    def copy_(self, other):
        self.tag, self.value = other.tag, other.value
        return self


class _FakeGenerator:
    def __init__(self, device=None):
        self.seed = None
        self.draws = 0

    def manual_seed(self, seed):
        self.seed = seed
        return self

    def draw(self, name, shape):
        self.draws += 1
        return _FakeTensor((name, self.seed, self.draws), shape=shape)


class _FakeRank:
    """One simulated rank of an 8-rank job whose peers behave identically."""

    world = 8

    def __init__(self, marker, fused_ms, unfused_ms):
        self.marker = marker
        self.marker_printed = False
        self.fused_ms = fused_ms
        self.unfused_ms = unfused_ms
        self.clock = 0.0
        self.timing = False
        self.capturing = None
        self.profiler = None
        self.runs = []

    def torch_modules(self):
        fake = self

        class Graph:
            def __init__(self):
                self.kernels = []

            def replay(self):
                for kernel in self.kernels:
                    kernel()

        @contextlib.contextmanager
        def graph(target, stream=None):
            fake.capturing = target
            try:
                yield
            finally:
                fake.capturing = None

        class Event:
            def __init__(self, enable_timing=False):
                self.time = None

            def record(self):
                self.time = fake.clock
                fake.timing = not fake.timing

            def elapsed_time(self, end):
                return end.time - self.time

        class Profile:
            def __init__(self, activities=None):
                self.names = []

            def __enter__(self):
                fake.profiler = self
                return self

            def __exit__(self, *exc):
                fake.profiler = None

            def key_averages(self):
                return [
                    types.SimpleNamespace(
                        key=name, count=self.names.count(name), device_time_total=1.0
                    )
                    for name in sorted(set(self.names))
                ]

        def all_reduce(tensor, op=None):
            if op == "sum":
                tensor.value *= fake.world

        def all_gather(outputs, tensor):
            for output in outputs:
                output.value = tensor.value

        def vector_norm(tensor):
            tag = tensor.tag
            if isinstance(tag, tuple) and tag[0] == "sub":
                return _FakeTensor(value=0.0 if tag[1] == tag[2] else 1.0)
            return _FakeTensor(value=1.0)

        def draw(name):
            def op(*args, generator=None, **_):
                shape = args[-1]
                label = f"{name}{args[0]}" if name == "randint" else name
                return generator.draw(label, shape)
            return op

        dist = types.ModuleType("torch.distributed")
        dist.ReduceOp = types.SimpleNamespace(SUM="sum", MIN="min", MAX="max")
        dist.all_reduce = all_reduce
        dist.all_gather = all_gather
        profiler = types.ModuleType("torch.profiler")
        profiler.ProfilerActivity = types.SimpleNamespace(CUDA="cuda")
        profiler.profile = Profile
        torch = types.ModuleType("torch")
        torch.distributed = dist
        torch.profiler = profiler
        torch.bfloat16, torch.float32, torch.int32 = "bf16", "f32", "i32"
        torch.Generator = _FakeGenerator
        torch.randn = draw("randn")
        torch.rand = draw("rand")
        torch.randint = draw("randint")
        torch.topk = lambda tensor, k, dim=-1: types.SimpleNamespace(
            indices=_FakeTensor(("topk", tensor.tag, k), shape=(*tensor.shape[:-1], k))
        )
        torch.where = lambda cond, a, b: _FakeTensor(
            ("where", cond.tag, a.tag, b.tag), shape=a.shape
        )
        torch.zeros_like = lambda tensor: _FakeTensor(("zeros",), 0, tensor.shape)
        torch.tensor = lambda value, dtype=None, device=None: _FakeTensor(
            value=value[0] if isinstance(value, list) else value
        )
        torch.linalg = types.SimpleNamespace(vector_norm=vector_norm)
        torch.cuda = types.SimpleNamespace(
            CUDAGraph=Graph,
            Stream=object,
            graph=graph,
            Event=Event,
            synchronize=lambda: None,
        )
        return {"torch": torch, "torch.distributed": dist, "torch.profiler": profiler}

    def adapter_module(self):
        fake = self

        class FakeMegaMoE:
            def __init__(
                self, *, rank, world_size, model_dim, inter_dim, experts, topk,
                quant, w1, w1_scale, w2, w2_scale, max_tok_per_rank,
                swiglu_limit=0.0,
            ):
                self.output = _FakeTensor(("empty",))

            def __call__(self, x, route_weights, ids):
                fused = os.environ.get("AITER_MEGAMOE_FUSE_ALL") == "1"
                if fused and fake.marker and not fake.marker_printed:
                    fake.marker_printed = True
                    os.write(2, fake.marker.encode())
                kernel = functools.partial(self._run, fused, x, route_weights, ids)
                if fake.capturing is None:
                    kernel()
                else:
                    fake.capturing.kernels.append(kernel)
                return self.output

            def _run(self, fused, x, route_weights, ids):
                fake.clock += fake.fused_ms if fused else fake.unfused_ms
                if fake.profiler is not None:
                    fake.profiler.names.extend(
                        ["quant", "mega"] if fused
                        else ["quant", "dispatch", "gemm1", "gemm2", "combine"]
                    )
                self.output.tag = ("moe", x.tag, route_weights.tag, ids.tag)
                fake.runs.append({
                    "fused": fused,
                    "tokens": x.shape[0],
                    "route": (route_weights.tag, ids.tag),
                    "timed": fake.timing,
                })

        adapter = types.ModuleType("_megamoe_runtime_fake")
        adapter.PROFILES = {
            "v4_pro": {"model_dim": 7168, "inter_dim": 3072, "experts": 384, "topk": 6}
        }
        adapter.FakeMegaMoE = FakeMegaMoE
        adapter._setup_dist = lambda: (0, fake.world, "cpu-fake")
        adapter._barrier = lambda: None
        adapter._cleanup = lambda: None
        adapter._reduce_float = lambda value, device, op: value
        adapter._quantize_weights = lambda *args: tuple(
            _FakeTensor(("weight", index)) for index in range(8)
        )
        adapter._reference = lambda x, route_weights, ids, *args: _FakeTensor(
            ("moe", x.tag, route_weights.tag, ids.tag)
        )
        return adapter


def _fake_megamoe_main(
    tmp_path, monkeypatch, *, marker="[megamoe] path=MEGA\n", fused_ms=4.0,
    unfused_ms=11.0, extra=(),
):
    fake = _FakeRank(marker, fused_ms, unfused_ms)
    for name, module in {
        **fake.torch_modules(), "_megamoe_runtime_fake": fake.adapter_module()
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in (
        "AITER_MEGAMOE_FUSE_ALL", "AITER_MEGAMOE_FUSE_COMBINE", "AITER_MEGAMOE_FUSE_QUANT"
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(sys, "path", list(sys.path))
    adapter = tmp_path / "candidate" / "op_tests" / "multigpu_tests" / "test_mega_moe_v2.py"
    adapter.parent.mkdir(parents=True)
    adapter.write_text(
        "from _megamoe_runtime_fake import (  # noqa: F401\n"
        "    PROFILES, FakeMegaMoE, _barrier, _cleanup, _quantize_weights,\n"
        "    _reduce_float, _reference, _setup_dist,\n"
        ")\n"
    )
    output = tmp_path / "evidence" / "graph_contract.json"
    monkeypatch.setattr(sys, "argv", [
        str(MEGAMOE_SKILL),
        "--candidate-tree", str(tmp_path / "candidate"),
        "--replays", "3",
        "--numeric-checkpoint-interval", "2",
        "--pairs", "3",
        "--perf-iters", "2",
        "--json-output", str(output),
        *extra,
    ])
    return fake, output, _runtime_namespace()["main"]


def test_megamoe_runtime_main_claims_with_frozen_gate_marker_and_uniform_timing(
    tmp_path, monkeypatch, capfd
):
    fake, output, main = _fake_megamoe_main(
        tmp_path,
        monkeypatch,
        extra=(
            "--frozen-baseline-ms", "4.70",
            "--resource-evidence", str(_resource_evidence(tmp_path)),
        ),
    )
    assert main() == 0
    data = json.loads(output.read_text())
    assert data["claim_complete"] is True
    assert data["claim_blockers"] == []
    assert data["speedup_gate"] == "pass"
    assert data["absolute_speedup"] == pytest.approx(4.70 / 4.0)
    assert data["paired_rank_max_speedup"] == data["absolute_speedup"]
    assert data["incremental_switch_speedup"] == pytest.approx(11.0 / 4.0)
    assert data["switch_off_ms_median"] == pytest.approx(11.0)
    assert data["denominator_mismatch"]["frozen_ms"] == pytest.approx(4.70)
    assert {row["base_arm"] for row in data["paired_readings"]} == {
        "same_tree_switch_off"
    }
    assert data["resources"]["status"] == "pass"
    activation = data["activation"]
    assert activation["status"] == "pass"
    assert activation["observed_ranks"] == 8
    assert [row["marker_count"] for row in activation["per_rank"]] == [1] * 8
    assert activation["window"] == "first eager candidate forward (tokens=128)"

    uniform = next(run["route"] for run in fake.runs if run["tokens"] == 8192)
    timed = [run for run in fake.runs if run["timed"]]
    assert {run["fused"] for run in timed} == {True, False}
    assert all(run["tokens"] == 8192 and run["route"] == uniform for run in timed)
    assert any(run["tokens"] == 8192 and run["route"] != uniform for run in fake.runs)
    assert "[megamoe] path=MEGA" in capfd.readouterr().err
    assert sorted(path.name for path in output.parent.iterdir()) == [
        "graph_contract.json"
    ]


def test_megamoe_runtime_main_old_arguments_never_gate_on_same_tree_ratio(
    tmp_path, monkeypatch
):
    _, output, main = _fake_megamoe_main(tmp_path, monkeypatch, unfused_ms=3.0)
    assert main() == 1
    data = json.loads(output.read_text())
    assert data["claim_complete"] is False
    assert "error" not in data
    assert data["claim_blockers"] == [
        "resource_evidence_missing", "frozen_baseline_missing"
    ]
    assert data["speedup_gate"] == "not_evaluated: missing --frozen-baseline-ms"
    assert data["absolute_speedup"] is None
    assert data["paired_rank_max_speedup"] is None
    assert data["incremental_switch_speedup"] == pytest.approx(0.75)
    assert data["resources"]["status"] == "delegated"
    assert data["activation"]["status"] == "pass"
    assert data["correctness"] == "pass"


def test_megamoe_runtime_main_without_marker_fails_activation(tmp_path, monkeypatch):
    _, output, main = _fake_megamoe_main(
        tmp_path,
        monkeypatch,
        marker=None,
        extra=(
            "--frozen-baseline-ms", "4.70",
            "--resource-evidence", str(_resource_evidence(tmp_path)),
        ),
    )
    assert main() == 1
    data = json.loads(output.read_text())
    assert data["claim_complete"] is False
    assert data["claim_blockers"] == ["activation_marker_missing"]
    assert data["activation"]["status"] == "fail"
    assert data["activation"]["observed_ranks"] == 0
    assert data["speedup_gate"] == "pass"
    assert len(data["paired_readings"]) == 3


def test_megamoe_runtime_main_fails_absolute_gate_despite_same_tree_win(
    tmp_path, monkeypatch
):
    _, output, main = _fake_megamoe_main(
        tmp_path,
        monkeypatch,
        extra=(
            "--frozen-baseline-ms", "3.9",
            "--resource-evidence", str(_resource_evidence(tmp_path)),
        ),
    )
    with pytest.raises(AssertionError, match="absolute speedup"):
        main()
    data = json.loads(output.read_text())
    assert data["claim_complete"] is False
    assert data["speedup_gate"] == "fail"
    assert data["incremental_switch_speedup"] == pytest.approx(11.0 / 4.0)
    assert data["absolute_speedup"] == pytest.approx(3.9 / 4.0)
    assert data["error"].startswith("AssertionError: absolute speedup")
