---
key: whether collapsing graph nodes is worth funding on a decode MoE already captured as one CUDA graph
type: lever
confidence: ★★
effect: Folding the sort prologue into stage-1's grid cut the captured graph from 3 nodes to 1 and bought +1.8% marginal weighted (1.2913 -> 1.3279) for 656 lines; the deletion control is the real result -- removing a prologue node recovered ZERO measurable wall time on every decode case, while the surviving stage-1 kernel grew by over half the deleted node's own device time.
confirms_cited: 0
confirms_blind: 0
losses: 0
attempts: 1
toolchain: flydsl 0.3.4.1 / aiter@4faf8713 / ROCm 7.2
last_seen: 2026-10-12
name: the-cuda-graph-launch-floor-is-per-replay-not-per-node-moe-grouped-gemm-gfx950-decode
description: Anti-pattern: deleting a 4.49us graph node recovered zero wall time; the surviving kernel absorbed +2.50us. Fold nodes for locality, never for node count.
keywords: ['anti-pattern', 'closed-axis', 'moe-grouped-gemm', 'cuda-graph', 'graph-replay', 'launch-overhead', 'dispatch-collapse', 'kernel-fusion', 'decode', 'flydsl', 'gfx950']
kernels: ['gemm1_a4w4_port']
platforms: ['gfx950']
kernel_class: moe_grouped_gemm
regime: decode
language: flydsl
decision_refs: []
layer: learned
lifecycle: active
origin_kernels: ['glm52_mxfp4_fused_moe']
cost: L2
verified_on: 2026-10-12
levers: ['host.dispatch-collapse']
---
# The CUDA-graph launch floor is per-replay, not per-node
- lever: Before funding a node-count reduction on an already-captured graph, delete one node and measure: under graph replay the launch floor is paid once per replay, so node count is not a cost term.
- apply: Keep the deletion control as a separate arm from the fusion arm, so 'the node is free' and 'the fusion helps' are measured apart rather than inferred from one number.
- stack: What does pay is locality and the work the fusion lets you skip; budget the round against that, not against the node count.
- verify: A node-count change with no measured host residue is a null by construction -- price the exposed host fraction first, and if a replay-floor measurement shows no floor, the axis is closed before any code is written.
- pitfall: The same reasoning is correct and valuable WITHOUT graph capture, where each launch is a real dispatch -> carrying it into a captured-graph decode path funded a 656-line rewrite for +1.8% -> check which regime the benchmark actually times.
- caution: A fused prologue silently widens the kernel's correctness surface; the win here was marginal enough that the added parity risk is the dominant cost, not the lines.
- source: run wwd7t6625 / team_volck_A_20261012_082749 round 3, 2026-10-12, gfx950/MI355X, decode-only CUDA-graph capture/replay harness; director weighted 1.3205 correctness pass
