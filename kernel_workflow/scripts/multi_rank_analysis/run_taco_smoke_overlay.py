#!/usr/bin/env python3
"""Run Taco MegaMoE smoke with its source vector module and no external baseline."""

from __future__ import annotations

import os
import importlib.abc
import importlib.machinery
import runpy
import re
import sys
import types

import flydsl
import flydsl.expr


source_root = os.environ.get("GEAK_TACO_FLYDSL_SOURCE", "/sgl-workspace/FlyDSL/python/flydsl")
flydsl.__path__.insert(0, source_root)
flydsl.expr.__path__.insert(0, os.path.join(source_root, "expr"))
import flydsl.expr.vector as vector  # noqa: E402

flydsl.expr.vector = vector
if not hasattr(type(flydsl.expr.T), "f8"):
    type(flydsl.expr.T).f8 = property(
        lambda _self: flydsl.expr.Float8E4M3FN.ir_type
    )

# Old Taco sources call MLIR type objects (`T.i32()`) while current FlyDSL
# exposes properties (`T.i32`). Make the underlying MLIR types idempotently
# callable so modules imported before the source transformer remain compatible.
from flydsl._mlir import ir  # noqa: E402

with ir.Context():
    for _name in ("i32", "i64", "f16", "f32", "f64", "bf16"):
        _value = getattr(flydsl.expr.T, _name)
        _type = type(_value)
        if "__call__" not in _type.__dict__:
            _type.__call__ = lambda self: self


class _CompatLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        source = re.sub(
            r"\b(T|_epkT)\.(i32|i64|f32|f64|f16|bf16)\(\)",
            r"\1.\2",
            source,
        )
        return self.source_to_code(source, self.path)


class _CompatFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("aiter.ops.flydsl"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec is not None
            and isinstance(spec.origin, str)
            and spec.origin.endswith(".py")
        ):
            spec.loader = _CompatLoader(fullname, spec.origin)
        return spec


sys.meta_path.insert(0, _CompatFinder())

from aiter.ops.flydsl.mega_moe import MegaMoE  # noqa: E402

# The original smoke compares AITER against an external source checkout. For
# timing the selected Taco tree itself, point that optional source import back
# at the same implementation; its separately printed `fused_ms` remains valid.
kernels = types.ModuleType("kernels")
moe = types.ModuleType("kernels.moe")
mega_moe = types.ModuleType("kernels.moe.mega_moe")
mega_moe.MegaMoE = MegaMoE
kernels.moe = moe
moe.mega_moe = mega_moe
sys.modules["kernels"] = kernels
sys.modules["kernels.moe"] = moe
sys.modules["kernels.moe.mega_moe"] = mega_moe

script = sys.argv.pop(1)
runpy.run_path(script, run_name="__main__")
