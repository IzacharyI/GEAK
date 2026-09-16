# MegaMoE v7 hardware handoff

Date: 2026-09-16

## 1. Decision first

An 8-GPU MI300/MI300X machine is useful for validating:

- GEAK's fixed eight-GPU lease;
- `torchrun --nproc_per_node=8`;
- RCCL all-reduce and process cleanup;
- the ROCm/PyTorch/FlyDSL/MORI installation.

It cannot run or validate the current MegaMoE v7 candidate. MI300 is CDNA3
`gfx942`, while both the pinned public AITER baseline and the candidate reject
anything except CDNA4 `gfx95x`:

```text
MegaMoE v2 stage1 requires CDNA4 (gfx95x), got gfx942
MegaMoE v2 stage2 requires CDNA4 (gfx95x), got gfx942
```

This is not merely a conservative model-name gate. The exact two-launch plan
uses 160400 bytes of local memory per workgroup; this is within gfx950's
163840-byte limit but exceeds the 65536-byte gfx942 limit. The kernel also uses
gfx950-specific MFMA/conversion, FP8-format, cache-policy, and tuning choices.
Removing the architecture checks would not make the result valid.

Therefore:

- use MI300 for the infrastructure smoke in section 5;
- use an 8x MI355X/MI350-class `gfx950` node for sections 6-8;
- do not interpret an MI300 architecture rejection as evidence that the v7
  combine repair passed or failed.

## 2. Immutable identities

GEAK:

```text
remote: https://github.com/IzacharyI/GEAK.git
branch: mega
minimum workflow commit: d8a0fca8fa00a2202fe76b88d7399b4df4991b34
```

Public AITER baseline:

```text
remote: https://github.com/ROCm/aiter.git
commit: 8775229e003af030abea7c50d2ed9e956311843d
subject: swap gfx950 kernel to hold 64bit memory addr (#4578)
```

Do not use the moving AITER `main` tip. Clone `main`, then check out the exact
commit above.

The original workflow candidate was:

```text
lane HEAD: 6b41e1dc5e0dc324236c690940e94689fe075dbb
source-tree digest: e37000741042bf4b4d84ae164ba2ed7d6ada9f6607b97c52302b4bc182a57714
topology: 2 launches (quant + persistent dispatch/GEMM1/GEMM2/P2P/combine)
```

`6b41e1d` belongs to a synthetic GEAK candidate lineage and is not a commit in
ROCm/AITER. Reproduce its source on public AITER with the portable patch shipped
beside this file:

```text
file: aiter_8775229_to_6b41e1d.patch
sha256: 167169c99e82afbb10c8e42f61b8f46761d4aae250615d25d6df769a916d9235
```

The patch applies cleanly to `8775229e` and reproduces all nine changed FlyDSL
source files byte-for-byte.

Do not substitute the workflow-internal
`cumulative_lane_r1_6b41e1d.patch` or a run-local `final_patch.diff`. Those
patches are rooted at synthetic commit `cc77c22`, omit public-baseline deltas,
and fail against public AITER in `mega_moe_stage2.py` and `mega_moe_v2.py`.

Expert Skill v7:

```text
id: megamoe_ep_mega_fusion
revision: mega-ep-fusion-v7
bundle: c772ae7f6f1021da5a4b91b7d5cc3154a78daef80b0672350e5f4cebffec5c71
planner extension: 8ac289067ea4caf8cf6efba6352881b1f91512366bfe7e2f68d5e7a2d6c617c4
contract: 05834114e29de6c6bec3a59211307177e75f913f7d34341efa87ea37ac5f3635
last GPU-free result: compatible, 172/172, plan_consistent=true,
                      independent_structure_pass=true, contract_failures=0
```

## 3. Software and machine prerequisites

The last validated source/structural environment was:

```text
Python: 3.10.12
PyTorch: 2.9.1+rocm7.2.0.git7e1940d4
ROCm/HIP: 7.2.0 / HIP 7.2.26015
FlyDSL: 0.3.0
MORI: 96ffa169710f214e76e07abe5008d686fe54522b
Node.js: 22.22.1
```

Use a ROCm container or environment with a ROCm-matched PyTorch build. Do not
let `pip` replace it with a CUDA or CPU wheel.

For the exact operator, the machine additionally needs:

- one node with eight mutually P2P/XGMI-connected GPUs;
- eight `gfx950` GPUs for the real candidate test;
- at least 150 GiB free VRAM on every card before the EP8 job;
- MORI-SHMEM and its Python package;
- a writable AITER JIT directory and a separate writable FlyDSL cache;
- `libpci-dev`; RCCL available through the ROCm PyTorch build;
- no other EP8 job using the selected cards.

The existing ROCm 7.2 setup required a small MORI JIT compatibility patch:

```text
file: mori_96ffa_rocm7_unbundle.patch
sha256: 6383725f6e48dff4d70830a06634275a4c48cee0d5a9d3328ac9a6bc35bfeb15
```

It unbundles `hipcc --genco` output before `hipModuleLoad`. Apply it if the
unpatched MORI reports `hipErrorInvalidKernelFile` or when reproducing the
known ROCm 7.2 environment exactly.

## 4. Prepare the checkouts

Choose a root with enough space:

```bash
export ROOT="${ROOT:-$HOME/mega-v7-handoff}"
mkdir -p "$ROOT"/{cache,logs}
cd "$ROOT"
```

Clone GEAK:

```bash
git clone --branch mega https://github.com/IzacharyI/GEAK.git GEAK
cd "$ROOT/GEAK"
git merge-base --is-ancestor \
  d8a0fca8fa00a2202fe76b88d7399b4df4991b34 HEAD
sha256sum aiter_8775229_to_6b41e1d.patch \
          mori_96ffa_rocm7_unbundle.patch
```

If the tagged handoff has not been pushed to GitHub, transfer it without any
remote dependency:

```bash
# Source machine
git -C /sgl-workspace/mega_test/GEAK bundle create \
  /sgl-workspace/mega_test/GEAK-mega-v7-handoff.bundle \
  refs/heads/mega refs/tags/mega-v7-handoff-20260916

# Target machine
git clone --branch mega-v7-handoff-20260916 \
  GEAK-mega-v7-handoff.bundle "$ROOT/GEAK"
git -C "$ROOT/GEAK" status --short
```

Clone and patch AITER:

```bash
cd "$ROOT"
git clone --recursive https://github.com/ROCm/aiter.git aiter-v7
cd "$ROOT/aiter-v7"
git checkout --detach 8775229e003af030abea7c50d2ed9e956311843d
git submodule sync
git submodule update --init --recursive
git switch -c mega-v7-handoff

git apply --check "$ROOT/GEAK/aiter_8775229_to_6b41e1d.patch"
git apply "$ROOT/GEAK/aiter_8775229_to_6b41e1d.patch"
git diff --check
git status --short
```

Exactly these nine files should be modified:

```text
aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py
aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py
aiter/ops/flydsl/kernels/mega_moe/gemm1.py
aiter/ops/flydsl/kernels/mega_moe/gemm2.py
aiter/ops/flydsl/kernels/mega_moe/gemm_util.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_config.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage2.py
aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
```

Commit locally so every log has an immutable AITER identity:

```bash
git add aiter/ops/flydsl
git -c user.name=zhengyao -c user.email=zhengyao@localhost \
  commit -m "test: materialize MegaMoE v7 hardware candidate"
git status --short
```

Clone and install MORI:

```bash
cd "$ROOT"
git clone --recursive https://github.com/ROCm/mori.git mori
cd "$ROOT/mori"
git checkout 96ffa169710f214e76e07abe5008d686fe54522b
git apply --check "$ROOT/GEAK/mori_96ffa_rocm7_unbundle.patch"
git apply "$ROOT/GEAK/mori_96ffa_rocm7_unbundle.patch"
python3 -m pip install -e . --no-build-isolation
```

Install AITER/FlyDSL in the existing ROCm Python environment:

```bash
python3 -m pip install --pre "flydsl==0.3.0"
cd "$ROOT/aiter-v7"
python3 setup.py develop
```

Set paths. The selected AITER checkout must come first in `PYTHONPATH`;
otherwise `torchrun` can silently import another checkout.

```bash
export GEAK="$ROOT/GEAK"
export AITER="$ROOT/aiter-v7"
export MORI="$ROOT/mori"
export PYTHONPATH="$AITER:$MORI:$MORI/python"
export AITER_JIT_DIR="$ROOT/cache/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$ROOT/cache/flydsl"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export MORI_SOCKET_IFNAME=lo
export MORI_SHMEM_HEAP_SIZE=40G
mkdir -p "$AITER_JIT_DIR" "$FLYDSL_RUNTIME_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
```

CPU-side import and syntax preflight:

```bash
cd "$AITER"
python3 -m py_compile \
  aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_kernel.py \
  aiter/ops/flydsl/kernels/flydsl_dispatch_combine_intranode_op.py \
  aiter/ops/flydsl/kernels/mega_moe/{gemm1,gemm2,gemm_util,mega_moe_config,mega_moe_stage1,mega_moe_stage2,mega_moe_v2}.py

python3 - <<'PY'
import torch, flydsl, mori
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
print("torch", torch.__version__)
print("flydsl", flydsl.__version__)
print("mori", mori.__file__)
print("MegaMoEV2", MegaMoEV2)
PY

python3 "$GEAK/kernel_workflow/tools/expert_skill_contract.py" \
  --contract "$GEAK/perf_knowledge/expert_skills/skills/megamoe_ep_mega_fusion/skill.md" \
  --candidate "$AITER" --digest-only
```

The final command must print:

```text
e37000741042bf4b4d84ae164ba2ed7d6ada9f6607b97c52302b4bc182a57714
```

## 5. MI300-only infrastructure smoke

First confirm that all devices are `gfx942` and that the intended group is
visible. These commands intentionally inspect the target test machine:

```bash
rocminfo | grep -m8 -E 'Name:.*gfx'
amd-smi list
amd-smi topology

python3 - <<'PY'
import torch
assert torch.cuda.device_count() >= 8
for i in range(8):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, getattr(p, "gcnArchName", "unknown"),
          p.total_memory // 2**30, "GiB")
PY
```

Run GEAK's fixed-group/RCCL smoke. This validates lease acquisition,
visibility remapping, eight ranks, RCCL all-reduce, timeout handling, and lock
release:

```bash
cd "$AITER"
GEAK_GPU_REQUIRE_IDLE=1 \
GEAK_GPU_MAX_BUSY_PCT=5 \
GEAK_GPU_MAX_VRAM_MB=1024 \
GEAK_GPU_WAIT_TIMEOUT=1800 \
GEAK_GPU_RUN_TIMEOUT=300 \
bash "$GEAK/kernel_workflow/scripts/gpu_lock.sh" \
  pool:8:0,1,2,3,4,5,6,7 -- \
  timeout 4m torchrun --standalone --nproc_per_node=8 \
    "$GEAK/kernel_workflow/tests/gpu_group_smoke.py" \
  2>&1 | tee "$ROOT/logs/mi300_gpu_group_smoke.log"
```

Required output:

```text
GPU_GROUP_SMOKE_PASS world_size=8
```

Re-run it once to prove the prior lease and process group were cleaned up.

Stop here for this MegaMoE candidate on MI300. Do not run the v7 hardware
acceptance command, do not remove the `gfx95x` checks, and do not quote an
MI300 performance number as an M2.5/MI355X result.

If a full GEAK GPU lifecycle must be exercised on MI300, use a separate
gfx942-compatible operator and task. Passing such a task validates GEAK's
generic lifecycle, not this MegaMoE implementation.

## 6. Exact candidate smoke on eight gfx950 GPUs

Only run this section on an eight-GPU `gfx950` node. Confirm every card has at
least 150 GiB free before starting.

There is no hardware result yet for candidate `6b41e1d`; its current authority
is GPU-free structural verification only. The first `bs=128` run below is new
evidence, not a confirmation of an already-tested candidate.

Use explicit switches; they are read at import time:

```bash
export AITER_MEGAMOE_FUSE_ALL=1
export AITER_MEGAMOE_FUSE_COMBINE=1
export AITER_MEGAMOE_FUSE_QUANT=0
export FLYDSL_RUNTIME_CACHE_DIR="$ROOT/cache/flydsl-fused"
mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR"
```

Run the smallest correctness/JIT smoke first:

```bash
cd "$AITER"
GEAK_GPU_REQUIRE_IDLE=1 \
GEAK_GPU_MAX_BUSY_PCT=5 \
GEAK_GPU_MAX_VRAM_MB=1024 \
GEAK_GPU_WAIT_TIMEOUT=1800 \
GEAK_GPU_RUN_TIMEOUT=3600 \
bash "$GEAK/kernel_workflow/scripts/gpu_lock.sh" \
  pool:8:0,1,2,3,4,5,6,7 -- \
  env PYTHONPATH="$PYTHONPATH" \
      AITER_JIT_DIR="$AITER_JIT_DIR" \
      FLYDSL_RUNTIME_CACHE_DIR="$FLYDSL_RUNTIME_CACHE_DIR" \
      MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G \
      AITER_MEGAMOE_FUSE_ALL=1 \
      AITER_MEGAMOE_FUSE_COMBINE=1 \
      AITER_MEGAMOE_FUSE_QUANT=0 \
  timeout 50m torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_mega_moe_v2.py \
    --network v4_pro --bs-list 128 --iters 1 \
    --accuracy-max-bs 128 --rtol 0.10 \
  2>&1 | tee "$ROOT/logs/v7_bs128_smoke.log"
```

Pass conditions:

- eight occurrences of `[megamoe] path=MEGA`;
- no JIT/trace exception;
- no `Memory access fault` and no `(nil)` address;
- `relL2 < 0.10`;
- command completes without a timeout.

The previous hardware failure was a deterministic all-rank nil-address fault
at `bs=128` in the combine reducer. This first run is the decisive check for
whether the direct-item JIT-frame repair changed that outcome.

## 7. Correctness and liveness on gfx950

Run the full shape set only after section 6 passes:

```bash
cd "$AITER"
GEAK_GPU_REQUIRE_IDLE=1 \
GEAK_GPU_MAX_BUSY_PCT=5 \
GEAK_GPU_MAX_VRAM_MB=1024 \
GEAK_GPU_WAIT_TIMEOUT=1800 \
GEAK_GPU_RUN_TIMEOUT=3600 \
bash "$GEAK/kernel_workflow/scripts/gpu_lock.sh" \
  pool:8:0,1,2,3,4,5,6,7 -- \
  env PYTHONPATH="$PYTHONPATH" \
      AITER_JIT_DIR="$AITER_JIT_DIR" \
      FLYDSL_RUNTIME_CACHE_DIR="$FLYDSL_RUNTIME_CACHE_DIR" \
      MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G \
      AITER_MEGAMOE_FUSE_ALL=1 \
      AITER_MEGAMOE_FUSE_COMBINE=1 \
      AITER_MEGAMOE_FUSE_QUANT=0 \
  timeout 50m torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_mega_moe_v2.py \
    --network v4_pro --bs-list 128,512,8192 --iters 10 \
    --accuracy-max-bs 8192 --rtol 0.10 \
  2>&1 | tee "$ROOT/logs/v7_correctness.log"
```

Then run the existing repeated-call liveness path:

```bash
# Use the same lease and env wrapper as above; replace its torchrun command with:
torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_mega_moe_v2.py \
  --network v4_pro --bs-list 128 --burst-depth 256
```

Required output is `[BURST] completed=256/256`. Deepen this to 1000 for the
terminal repeated-call check.

Important limitation: the current `--burst-depth` implementation repeatedly
calls the operator with fixed inputs. It does not mutate routes and does not
itself prove CUDA-graph replay safety. The normal timing path captures a graph
but only replays it for `--iters`. Preserve the repeated-call result as
liveness evidence, then collect separate graph-capture/replay and route-mutation
evidence before final promotion.

Runtime launch count must be established from a profiler/trace: one quant
launch plus one persistent megakernel per rank. The static `launches=2` field
and the `path=MEGA` marker are necessary but are not runtime launch-count
proof.

## 8. Performance on gfx950 only

Do not benchmark until correctness and liveness pass. Compare fresh processes,
alternating:

```text
A: AITER_MEGAMOE_FUSE_ALL=0
B: AITER_MEGAMOE_FUSE_ALL=1, AITER_MEGAMOE_FUSE_COMBINE=1
sequence: A,B,A,B,...
```

Each arm must have its own `torchrun` process and FlyDSL cache. The authoritative
command for one case is:

```bash
torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/bench_mega_moe_v2.py \
  --tokens 8192 --route uniform --iters 30 --mega-only
```

Always run it through the same eight-GPU lease and environment wrapper used in
sections 6-7.

Primary metric:

```text
mega_e2e rank-max = the second number in mega_e2e=<mean>/<max>ms
```

Required pairs:

- `8192_uniform`: 5;
- `8192_rank-mixed-skew`: 5;
- `512_uniform`: 8;
- `512_rank-mixed-skew`: 8.

The old MI355X frozen denominator was approximately 4.6744 ms for
`8192_uniform`, but promotion must use a fresh same-machine A/B denominator.
Never compare an MI300 number with that MI355X denominator.

## 9. What to send back

Return one archive containing:

```text
logs/
  mi300_gpu_group_smoke.log
  v7_bs128_smoke.log          # gfx950 only
  v7_correctness.log          # gfx950 only
  liveness.log                # gfx950 only
  benchmark_*.log             # gfx950 only
machine.txt
versions.txt
git.txt
```

Record:

```bash
{
  date -Is
  uname -a
  amd-smi list
  amd-smi topology
  rocminfo | grep -m8 -E 'Name:.*gfx'
} > "$ROOT/logs/machine.txt" 2>&1

{
  python3 --version
  python3 -c 'import torch,flydsl,mori; print(torch.__version__, flydsl.__version__, mori.__file__)'
  hipcc --version
} > "$ROOT/logs/versions.txt" 2>&1

{
  git -C "$GEAK" rev-parse HEAD
  git -C "$AITER" rev-parse HEAD
  git -C "$AITER" status --short
  git -C "$MORI" rev-parse HEAD
  git -C "$MORI" status --short
  sha256sum "$GEAK/aiter_8775229_to_6b41e1d.patch"
} > "$ROOT/logs/git.txt" 2>&1
```

For the first failure, include:

- exact command and exit code;
- GPU model and `gfx` architecture;
- first failing batch size;
- count of `path=MEGA` markers;
- whether failure occurred during import, JIT trace/emit, device execution,
  correctness, or liveness;
- complete rank traceback/device-fault text.

Do not summarize away the first error and do not rerun after editing source.
Send the evidence first; any production repair must pass the v7 structural
contract again before another hardware run.
