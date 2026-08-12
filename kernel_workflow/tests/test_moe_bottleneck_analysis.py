"""Tests for the deterministic measurement-first MoE analysis runner."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYZER = os.path.join(
    ROOT,
    "knowledge",
    "analysis_skills",
    "moe_bottleneck",
    "analyze.py",
)
SPEC = importlib.util.spec_from_file_location("moe_bottleneck_analyze", ANALYZER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _report(with_tracks=False):
    report = {
        "schema_version": "geak-megamoe-analysis-v1",
        "cases": [
            {"case_id": "t8192_uniform"},
            {"case_id": "t8192_rank_mixed_skew"},
        ],
        "route_comparisons": [
            {
                "tokens_per_rank": 8192,
                "e2e_rank_max_delta_ms": 0.8208,
                "e2e_rank_max_delta_pct": 17.30,
                "profile_category_delta": {
                    "combine": {
                        "rank_max_delta_ms": 0.343084,
                        "rank_max_delta_pct": 97.6772,
                    },
                    "stage1": {
                        "rank_max_delta_ms": 0.2785767,
                        "rank_max_delta_pct": 9.6523,
                    },
                    "stage2": {
                        "rank_max_delta_ms": 0.3557637,
                        "rank_max_delta_pct": 17.1293,
                    },
                },
            }
        ],
    }
    if with_tracks:
        report["measurement_tracks"] = {
            track: {
                "status": "complete",
                "evidence": {
                    "artifact_refs": [f"synthetic:{track}"],
                    "metrics": [f"synthetic_metric:{track}"],
                    "provenance_refs": [f"synthetic_provenance:{track}"],
                },
            }
            for track in MODULE.REQUIRED_TRACKS
        }
        report["measurement_tracks"]["fusion_dag"]["evidence"]["bounds"] = [
            {
                "name": "synthetic",
                "ceiling_speedup": 1.1,
            }
        ]
        hardware_context = {
            "schema_version": "geak-hardware-context-v1",
            "vendor": "AMD",
            "model": "MI355X",
            "arch": "gfx950",
            "device_count": 8,
            "execution_units_per_device": 256,
            "thread_group_width": 64,
            "local_memory_bytes_per_execution_unit": 163840,
            "device_memory_bytes": 309220868096,
            "interconnect": {"type": "XGMI", "topology": "fully_connected"},
            "runtime": {"rocm": "7.2"},
            "measured": {
                "pairwise_interconnect_gbps": 1.0,
                "all_to_all_interconnect_gbps": 1.0,
                "device_memory_gbps": 1.0,
                "launch_overhead_us": 1.0,
            },
        }
        hardware_context["provenance"] = {
            field: {
                "collector": "test",
                "timestamp": "2026-01-01T00:00:00Z",
                "confidence": "high",
                "raw_artifact": "synthetic:test",
            }
            for field in (
                "vendor",
                "model",
                "arch",
                "device_count",
                "execution_units_per_device",
                "thread_group_width",
                "local_memory_bytes_per_execution_unit",
                "device_memory_bytes",
                "interconnect",
                "runtime",
                "measured.pairwise_interconnect_gbps",
                "measured.all_to_all_interconnect_gbps",
                "measured.device_memory_gbps",
                "measured.launch_overhead_us",
            )
        }
        report["hardware_context"] = hardware_context
    return report


class TestMoEBottleneckAnalysis(unittest.TestCase):
    def test_corrects_relative_growth_vs_absolute_share(self):
        analysis = MODULE.build_analysis(_report())
        comparison = analysis["route_comparisons"][0]
        categories = comparison["categories"]
        self.assertAlmostEqual(
            categories["combine"]["relative_growth_pct"],
            97.6772,
        )
        self.assertAlmostEqual(
            categories["combine"]["positive_absolute_delta_share_pct"],
            35.1008,
            places=3,
        )
        self.assertGreater(
            categories["stage2"]["positive_absolute_delta_share_pct"],
            categories["combine"]["positive_absolute_delta_share_pct"],
        )
        hypotheses = analysis["hypotheses"]
        self.assertIn("stage2", hypotheses[0]["statement"])
        self.assertEqual(
            hypotheses[0]["supports"]["relative_growth_leader"],
            "combine",
        )
        self.assertNotIn("directions", analysis)
        serialized = json.dumps(analysis)
        self.assertNotIn('"directions"', serialized)
        self.assertNotIn("specialty", serialized)
        self.assertFalse(
            analysis["analysis_boundary"]["decision_fields_emitted"]
        )
        self.assertEqual(
            analysis["analysis_boundary"]["decision_owner"],
            "Step-3 TechLead",
        )

    def test_missing_tracks_force_draft_measurement_first(self):
        analysis = MODULE.build_analysis(_report())
        self.assertEqual(analysis["framework_status"], "ready")
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")
        self.assertFalse(analysis["claims"]["root_cause_proven"])
        self.assertEqual(len(analysis["unknowns"]), len(MODULE.REQUIRED_TRACKS) + 1)
        self.assertEqual(analysis["hardware_guidance"]["status"], "missing")

    def test_complete_tracks_remove_degradation(self):
        analysis = MODULE.build_analysis(_report(with_tracks=True))
        self.assertEqual(analysis["analysis_status"], "evidence_complete")
        self.assertEqual(analysis["degraded"], [])
        self.assertEqual(analysis["hardware_guidance"]["status"], "ready")

    def test_complete_label_without_evidence_is_invalid(self):
        report = _report(with_tracks=True)
        report["measurement_tracks"]["communication_bytes"] = {"status": "complete"}
        analysis = MODULE.build_analysis(report)
        self.assertEqual(analysis["analysis_status"], "awaiting_measurement")
        self.assertEqual(
            analysis["measurement_coverage"]["communication_bytes"]["status"],
            "invalid",
        )

    def test_cli_writes_versioned_json(self):
        with tempfile.TemporaryDirectory() as directory:
            report_path = os.path.join(directory, "report.json")
            output_path = os.path.join(directory, "advisory.json")
            with open(report_path, "w") as f:
                json.dump(_report(), f)
            process = subprocess.run(
                [
                    sys.executable,
                    ANALYZER,
                    "--report",
                    report_path,
                    "--output",
                    output_path,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            with open(output_path) as f:
                analysis = json.load(f)
            self.assertEqual(analysis["schema_version"], MODULE.SCHEMA_VERSION)
            self.assertEqual(
                len(analysis["analysis_inputs"]["report"]["sha256"]),
                64,
            )


if __name__ == "__main__":
    unittest.main()
