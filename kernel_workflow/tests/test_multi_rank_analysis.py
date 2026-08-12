"""Unit tests for `multi_rank_analysis` (stdlib only; no pytest, no torch, no GPU needed).

These lock two things:

1. **Shape fidelity.** Fed synthetic per-rank records shaped like the one existing multi-rank
   analysis artifact this framework generalizes (`artifacts/analysis/mega_moe_v2_analysis.json`,
   schema `geak-megamoe-analysis-v1`), `merge_rank_records`/`bucket_trace_events`/`build_report`
   must reproduce that artifact's field names and nesting — proving the generic library can stand in
   for the one-off generator without containing any operator-specific code.
2. **Never raises on partial/malformed input.** A missing metric on some ranks, an unreadable trace
   file, or a bad category-map pattern must degrade (excluded ranks / `error` field / `__errors__`
   key), never throw, so one bad rank cannot block the rest of an analysis.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from multi_rank_analysis import (  # noqa: E402
    SCHEMA_VERSION,
    bucket_trace_events,
    build_report,
    classify_case_speedup,
    load_category_map,
    merge_rank_records,
)

WORLD_SIZE = 8


def _synthetic_rank_records(base_ms=0.49, skew_rank=None, skew_factor=3.0):
    """8 per-rank timing_ms.stage1 records; optionally skew one rank (models route imbalance)."""
    records = []
    for r in range(WORLD_SIZE):
        ms = base_ms * skew_factor if r == skew_rank else base_ms
        records.append({"rank": r, "timing_ms": {"stage1": ms, "e2e": ms * 1.5}})
    return records


class TestMergeRankRecords(unittest.TestCase):
    def test_uniform_records_shape_matches_artifact(self):
        records = _synthetic_rank_records()
        merged = merge_rank_records(records, ["timing_ms.stage1", "timing_ms.e2e"])
        for path in ("timing_ms.stage1", "timing_ms.e2e"):
            entry = merged[path]
            for key in ("rank_mean", "rank_max", "rank_min", "rank_tail_spread_pct", "missing_ranks"):
                self.assertIn(key, entry, f"{path} missing key {key}")
        self.assertAlmostEqual(merged["timing_ms.stage1"]["rank_mean"], 0.49, places=6)
        self.assertAlmostEqual(merged["timing_ms.stage1"]["rank_max"], 0.49, places=6)
        self.assertEqual(merged["timing_ms.stage1"]["rank_tail_spread_pct"], 0.0)
        self.assertEqual(merged["timing_ms.stage1"]["missing_ranks"], [])

    def test_skewed_rank_raises_tail_spread(self):
        records = _synthetic_rank_records(skew_rank=3, skew_factor=48.0)  # ~expert_max_to_mean seen in evidence
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertGreater(entry["rank_max"], entry["rank_min"])
        self.assertGreater(entry["rank_tail_spread_pct"], 1000.0)  # (48x-1x)/1x*100 ≈ 4700%

    def test_missing_metric_on_some_ranks_excludes_not_raises(self):
        records = _synthetic_rank_records()
        del records[5]["timing_ms"]["stage1"]  # rank 5 never produced this metric
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertEqual(entry["missing_ranks"], [5])
        self.assertIsNotNone(entry["rank_mean"])  # still computed from the remaining 7 ranks

    def test_all_missing_degrades_to_none_not_raise(self):
        records = [{"timing_ms": {}} for _ in range(WORLD_SIZE)]
        merged = merge_rank_records(records, ["timing_ms.stage1"])
        entry = merged["timing_ms.stage1"]
        self.assertEqual(len(entry["missing_ranks"]), WORLD_SIZE)
        self.assertIsNone(entry["rank_mean"])
        self.assertIsNone(entry["rank_tail_spread_pct"])

    def test_repetitions_produce_runs_and_span_pct(self):
        reps = [_synthetic_rank_records(base_ms=m) for m in (0.4958, 0.4977, 0.4947)]
        merged = merge_rank_records(reps[0], ["timing_ms.stage1"], repetitions=reps)
        entry = merged["timing_ms.stage1"]
        self.assertEqual(len(entry["rank_mean_runs"]), 3)
        self.assertEqual(len(entry["rank_max_runs"]), 3)
        self.assertGreaterEqual(entry["rank_mean_span_pct"], 0.0)
        self.assertLess(entry["rank_mean_span_pct"], 5.0)  # matches the noise band seen in evidence


class TestClassifyCaseSpeedup(unittest.TestCase):
    def test_pass_above_floor(self):
        r = classify_case_speedup(candidate_rank_max=0.30, baseline_rank_max=0.84, min_pct=110.0)
        self.assertEqual(r["status"], "pass")
        self.assertGreater(r["speedup_pct"], 110.0)

    def test_fail_below_floor(self):
        r = classify_case_speedup(candidate_rank_max=0.80, baseline_rank_max=0.84, min_pct=110.0)
        self.assertEqual(r["status"], "fail")

    def test_unguarded_when_no_floor_given(self):
        r = classify_case_speedup(candidate_rank_max=0.30, baseline_rank_max=0.84)
        self.assertEqual(r["status"], "unguarded")

    def test_raises_on_nonpositive_baseline(self):
        with self.assertRaises(ValueError):
            classify_case_speedup(candidate_rank_max=0.3, baseline_rank_max=0.0)


class TestTraceCategories(unittest.TestCase):
    def _write_trace(self, kernel_specs):
        """kernel_specs: list of (name, dur_us)."""
        events = [
            {"ph": "X", "cat": "kernel", "name": name, "pid": 2, "tid": 3, "ts": 0.0, "dur": dur}
            for name, dur in kernel_specs
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"traceEvents": events}, f)
        f.close()
        return f.name

    def test_bucketing_matches_evidence_categories(self):
        cat_map = load_category_map({
            "stage1": "megamoe_stage1",
            "stage2": "megamoe_stage2",
            "combine": "ep_combine",
            "quantize": "per_1x32_mx_quant",
        })
        path = self._write_trace([
            ("megamoe_stage1_compact_t128x512x256_w8", 2784.7),
            ("megamoe_stage2_compact_t64x256x256_sbm128", 2124.1),
            ("ep_combine_intranode_0", 312.5),
            ("per_1x32_mx_quant_fp8_n7168", 32.5),
            ("hipDeviceSynchronize", 15208.2),  # cuda_runtime, not "kernel" cat below -> excluded anyway
        ])
        try:
            result = bucket_trace_events(path, cat_map)
        finally:
            os.unlink(path)
        self.assertIn("stage1", result["per_category_ms"])
        self.assertIn("stage2", result["per_category_ms"])
        self.assertIn("combine", result["per_category_ms"])
        self.assertIn("quantize", result["per_category_ms"])
        self.assertAlmostEqual(result["per_category_ms"]["combine"], 0.3125, places=4)

    def test_unmatched_kernel_goes_to_unclassified_not_dropped(self):
        cat_map = load_category_map({"stage1": "megamoe_stage1"})
        path = self._write_trace([("some_other_kernel_xyz", 100.0)])
        try:
            result = bucket_trace_events(path, cat_map)
        finally:
            os.unlink(path)
        self.assertIn("unclassified", result["per_category_ms"])
        self.assertNotIn("stage1", result["per_category_ms"])

    def test_unreadable_trace_degrades_not_raises(self):
        result = bucket_trace_events("/nonexistent/path/does_not_exist.json", {})
        self.assertIn("error", result)
        self.assertEqual(result["per_category_ms"], {})

    def test_bad_pattern_recorded_in_errors_not_raised(self):
        cat_map = load_category_map({"good": "megamoe_stage1", "bad": "("})
        self.assertIn("good", cat_map)
        self.assertIn("__errors__", cat_map)
        self.assertEqual(cat_map["__errors__"][0]["category"], "bad")


class TestBuildReport(unittest.TestCase):
    def test_envelope_shape(self):
        report = build_report(
            primary_metric="candidate E2E rank-max latency",
            cases=[{"case_id": "t512_uniform"}, {"case_id": "t512_skew"}],
            route_comparisons=[{"tokens_per_rank": 512}],
            secondary_comparator_role="secondary comparison only; never the speedup denominator",
        )
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["cases"]), 2)
        self.assertIn("route_comparisons", report)
        self.assertIn("secondary_comparator_role", report)

    def test_optional_fields_omitted_when_not_given(self):
        report = build_report(primary_metric="x", cases=[])
        self.assertNotIn("route_comparisons", report)
        self.assertNotIn("secondary_comparator_role", report)

    def test_no_operator_names_in_schema_version(self):
        # Regression guard: the generic schema must never hardcode an operator name.
        for banned in ("mega", "moe", "aiter", "mori", "flydsl"):
            self.assertNotIn(banned, SCHEMA_VERSION.lower())


if __name__ == "__main__":
    unittest.main()
