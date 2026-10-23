#!/usr/bin/env bash
# The benchmark the semantic Shape replay (run_semantic_shape_capture.py) drives: bench_e2e.sh with
# the run's measurement identity removed. The replay only has to make the server execute forwards
# while graph construction records shapes; it is not a measurement. Under an AgentX run the
# inherited E2E_METRIC names an AgentX axis that a native synthetic round cannot produce, and
# bench_summarize.py refusing it would fail the replay (it runs under check=True). The replay picks
# its own free port, which bench_e2e.sh would otherwise move into its port window.
unset E2E_METRIC GEAK_METRIC_BASIS GEAK_ISL_OSL_INACTIVE GEAK_WORKLOAD_KIND
export PORT_ENFORCE_RANGE=0
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bench_e2e.sh" "$@"
