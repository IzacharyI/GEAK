"""Controlled-experiment manifests and non-additive variant comparisons."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

EXPERIMENT_SCHEMA_VERSION = "geak-controlled-experiment-v1"

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "compare_controlled_variants",
    "validate_experiment_manifest",
]


def validate_experiment_manifest(manifest: Mapping[str, Any]) -> dict:
    """Validate and normalize a portable controlled-experiment manifest."""
    if not isinstance(manifest, Mapping):
        raise TypeError("experiment manifest must be a mapping")
    if manifest.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported experiment schema: {manifest.get('schema_version')!r}"
        )
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")
    workload = manifest.get("workload")
    if not isinstance(workload, Mapping) or not workload:
        raise ValueError("workload must be a non-empty mapping")
    variants = manifest.get("variants")
    if not isinstance(variants, Sequence) or isinstance(variants, (str, bytes)):
        raise ValueError("variants must be a sequence")
    normalized_variants = []
    names = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, Mapping):
            raise TypeError(f"variant {index} must be a mapping")
        name = variant.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"variant {index} must have a non-empty name")
        if name in names:
            raise ValueError(f"duplicate variant name: {name!r}")
        names.add(name)
        command = variant.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"variant {name!r} must have a command")
        normalized_variants.append(dict(variant))
    if "full" not in names:
        raise ValueError("variants must include the full implementation")
    return {
        **dict(manifest),
        "experiment_id": experiment_id.strip(),
        "workload": dict(workload),
        "variants": normalized_variants,
    }


def _finite_metrics(name: str, metrics: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(metrics, Mapping):
        raise TypeError(f"metrics for {name!r} must be a mapping")
    normalized = {}
    for metric, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}.{metric} must be finite, got {value!r}")
        normalized[str(metric)] = number
    return normalized


def compare_controlled_variants(
    baseline_metrics: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    metric_directions: Mapping[str, str],
) -> dict:
    """Compare controlled variants without pretending deltas are additive.

    ``metric_directions`` maps each metric to ``"lower"`` or ``"higher"``.
    Every variant must contain the same metrics as the baseline.
    """
    baseline = _finite_metrics("baseline", baseline_metrics)
    if not baseline:
        raise ValueError("baseline_metrics must not be empty")
    if set(metric_directions) != set(baseline):
        raise ValueError("metric_directions must cover exactly the baseline metrics")
    output = {}
    for name, raw_metrics in variants.items():
        metrics = _finite_metrics(str(name), raw_metrics)
        if set(metrics) != set(baseline):
            raise ValueError(f"variant {name!r} metrics do not match baseline")
        comparisons = {}
        for metric, baseline_value in baseline.items():
            direction = metric_directions[metric]
            if direction not in ("lower", "higher"):
                raise ValueError(
                    f"metric {metric!r} direction must be 'lower' or 'higher'"
                )
            value = metrics[metric]
            delta = value - baseline_value
            if baseline_value == 0.0:
                delta_pct = None
            else:
                delta_pct = delta / abs(baseline_value) * 100.0
            improvement_pct = (
                -delta_pct
                if direction == "lower" and delta_pct is not None
                else delta_pct
            )
            comparisons[metric] = {
                "baseline": baseline_value,
                "value": value,
                "delta": delta,
                "delta_pct": delta_pct,
                "improvement_pct": improvement_pct,
                "direction": direction,
            }
        output[str(name)] = comparisons
    return {
        "baseline": baseline,
        "variants": output,
        "note": (
            "Variant deltas are controlled bounds and are not additive when "
            "communication, computation, and waiting overlap."
        ),
    }
