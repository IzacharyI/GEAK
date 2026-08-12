"""Neutral top-level report schema for multi-rank analysis output.

Generalizes the shape of the one existing multi-rank analysis artifact this framework was built
from (schema ``geak-megamoe-analysis-v1``) into an operator-agnostic version: ``primary_metric`` and
every ``case_id`` are caller-supplied strings rather than hardcoded to one operator's name. See this
package's README for the full field-by-field mapping to that artifact.
"""

from __future__ import annotations

from typing import Any, Sequence

SCHEMA_VERSION = "geak-multirank-analysis-v1"

__all__ = ["SCHEMA_VERSION", "build_report"]


def build_report(
    primary_metric: str,
    cases: Sequence[dict],
    route_comparisons: Sequence[dict] | None = None,
    status: str = "pass",
    secondary_comparator_role: str | None = None,
) -> dict:
    """Assemble the top-level report object.

    ``primary_metric``: human-readable description of what latency number is the actual optimization
        target (e.g. "candidate E2E rank-max latency"). Never hardcoded by this module.
    ``cases``: list of case dicts, each expected to at least carry a ``case_id`` — but this module
        does not enforce a case's internal shape (that is the caller's/Skill's concern; this
        function only assembles the envelope).
    ``route_comparisons``: optional list of cross-case delta dicts (e.g. uniform vs. skewed route).
    ``secondary_comparator_role``: optional free-text note (e.g. "secondary comparison only; never
        the speedup denominator") — generalizes the artifact's ``mori_role`` field without naming
        MORI; omitted entirely when the caller has no secondary comparator.
    """
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "primary_metric": primary_metric,
        "cases": list(cases),
    }
    if secondary_comparator_role:
        report["secondary_comparator_role"] = secondary_comparator_role
    if route_comparisons:
        report["route_comparisons"] = list(route_comparisons)
    return report
