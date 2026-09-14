"""Static contract tests for the GPU-only MegaMoE graph harness."""

from pathlib import Path


TOOL = Path(__file__).resolve().parent / "graph_validation.py"
SRC = TOOL.read_text()


def test_uses_candidate_task_reference_without_machine_paths():
    assert "--candidate-tree" in SRC
    assert "test_mega_moe_v2.py" in SRC
    assert "/sgl-workspace" not in SRC
    assert "/root/" not in SRC


def test_compares_graph_output_directly_to_numeric_reference():
    assert "with torch.cuda.graph" in SRC
    assert "helper._reference(" in SRC
    assert "direct graph-captured candidate vs numeric reference" in SRC


def test_changes_routing_between_replays_and_records_jitter():
    assert "route_weights.copy_(new_weights)" in SRC
    assert "ids.copy_(new_ids)" in SRC
    assert '"arrival_jitter": True' in SRC
    assert '"routing_changes": args.replays' in SRC
    assert "--numeric-checkpoint-interval" in SRC
    assert "replay_rel_l2 = relative_l2(state[\"output\"], reference)" in SRC
    assert '"numeric_checkpoints": len(checkpoint_rel_l2)' in SRC
    assert '"max_checkpoint_relL2": max(checkpoint_rel_l2)' in SRC


def test_writes_incremental_atomic_claim_before_cleanup():
    assert 'tmp.write_text(json.dumps(result, indent=2)' in SRC
    assert "os.replace(tmp, output)" in SRC
    assert "write_result(True)" in SRC
    assert "helper._cleanup()" in SRC
