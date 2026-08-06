#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const workflow = path.join(__dirname, "..", "kernel_workflow.js");
const source = fs.readFileSync(workflow, "utf8");
const e2eSource = fs.readFileSync(
  path.join(__dirname, "..", "..", "e2e_workflow", "e2e_workflow.js"),
  "utf8"
);
const match = source.match(
  /function resolveGpuRequest\(A\) \{[\s\S]*?\n\}\nconst GPU_RESOURCE = resolveGpuRequest\(A\);/
);
if (!match) {
  throw new Error("resolveGpuRequest block not found in kernel_workflow.js");
}
const resolveGpuRequest = new Function(
  `${match[0].replace(/\nconst GPU_RESOURCE[\s\S]*$/, "")}; return resolveGpuRequest;`
)();

function equal(actual, expected, label) {
  const left = JSON.stringify(actual);
  const right = JSON.stringify(expected);
  if (left !== right) {
    throw new Error(`${label}: expected ${right}, got ${left}`);
  }
}

equal(resolveGpuRequest({}), {
  gpuList: ["0"],
  gpusPerJob: 1,
  fixedIds: [],
  sharedSpec: "",
}, "default remains one pinned GPU");

const legacy = resolveGpuRequest({ gpu_ids: "0,1" });
equal(legacy.specForIndex(0), "0", "legacy lane 0");
equal(legacy.specForIndex(1), "1", "legacy lane 1");
equal(legacy.specForIndex(2), "0", "legacy round robin");

const fixed = resolveGpuRequest({
  gpu_ids: "0,1,2,3",
  job_gpu_ids: "3,1",
});
equal(fixed.gpusPerJob, 2, "fixed group infers count");
equal(fixed.sharedSpec, "group:3,1", "fixed group spec");
equal(fixed.specForIndex(4), "group:3,1", "all directions share fixed group");

const dynamic = resolveGpuRequest({
  gpu_ids: "0,1,2,3",
  gpus_per_job: 2,
});
equal(dynamic.sharedSpec, "pool:2:0,1,2,3", "dynamic group spec");

const taskDeclared = resolveGpuRequest({
  gpu_ids: "0,1,2,3",
  job_gpu_ids: "",
  op_spec: {
    resource: {
      gpus_per_job: 2,
      job_gpu_ids: "2,3",
    },
  },
});
equal(taskDeclared.sharedSpec, "group:2,3", "task resource fixed group spec");

for (const invalid of [
  { gpu_ids: "" },
  { gpu_ids: "0,0" },
  { gpu_ids: "0,00" },
  { gpu_ids: "0,1", gpus_per_job: 3 },
  { gpu_ids: "0,1", gpus_per_job: "2x" },
  { gpu_ids: "0,1", gpus_per_job: 1.5 },
  { gpu_ids: "0,1", job_gpu_ids: "0,2" },
  { gpu_ids: "0,1", job_gpu_ids: "0,1", gpus_per_job: 1 },
]) {
  let threw = false;
  try {
    resolveGpuRequest(invalid);
  } catch (_error) {
    threw = true;
  }
  if (!threw) {
    throw new Error(`invalid request accepted: ${JSON.stringify(invalid)}`);
  }
}

if (/GPU_LIST\[0\]|GPU_LIST\[i % GPU_LIST\.length\]/.test(source)) {
  throw new Error("legacy direct GPU_LIST assignment remains in kernel workflow");
}
const propagated = (source.match(/GPU_RESOURCE\.specForIndex\(/g) || []).length;
if (propagated < 7) {
  throw new Error(`GPU resource is not propagated to every phase (found ${propagated})`);
}
const passthrough = e2eSource.match(
  /if \(!MODEL_PATH && KERNEL_PATH\) \{[\s\S]*?\n\}/
);
if (!passthrough ||
    !/gpus_per_job: A\.gpus_per_job/.test(passthrough[0]) ||
    !/job_gpu_ids: A\.job_gpu_ids/.test(passthrough[0])) {
  throw new Error("e2e single-kernel pass-through drops GPU resource arguments");
}

console.log("PASS: kernel workflow GPU request resolution");
