"""Single-file Expert Skill runtime-loader tests."""

import importlib.util
import sys
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
