#!/usr/bin/env python3
"""Statically check a candidate's FlyDSL imports and referenced module attributes.

This is an early compatibility gate, not a compiler substitute. It catches recipes copied across
FlyDSL versions when a module or public symbol is absent before a GPU compile is funded. Dynamic
values and call semantics still require the immutable correctness test.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path


def python_files(paths):
    seen = set()
    for raw in paths:
        path = Path(raw)
        candidates = path.rglob("*.py") if path.is_dir() else [path]
        for candidate in candidates:
            resolved = candidate.resolve()
            if candidate.suffix == ".py" and resolved not in seen:
                seen.add(resolved)
                yield candidate


def is_flydsl_module(name):
    # Same scope as detect_language's FlyDSL markers: FlyDSL itself, and kernels written in it that
    # a package ships under a `flydsl` subpackage (aiter.ops.flydsl.*), which the author role reuses.
    return "flydsl" in name.split(".")


def collect_references(paths):
    imports = []
    attributes = []
    parse_errors = []
    for path in python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if is_flydsl_module(alias.name):
                        local = alias.asname or alias.name.split(".")[0]
                        # `import flydsl.expr` binds the top-level name `flydsl`; an explicit alias
                        # (`as fx`) binds the dotted module itself.
                        aliases[local] = (
                            alias.name if alias.asname else alias.name.split(".")[0],
                            None,
                        )
                        imports.append({
                            "file": str(path), "line": node.lineno, "module": alias.name,
                            "name": None,
                        })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if is_flydsl_module(module):
                    for alias in node.names:
                        imports.append({
                            "file": str(path), "line": node.lineno, "module": module,
                            "name": alias.name,
                        })
                        aliases[alias.asname or alias.name] = (module, alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            parts = []
            root = node
            while isinstance(root, ast.Attribute):
                parts.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name) or root.id not in aliases:
                continue
            module, imported_name = aliases[root.id]
            path_parts = ([imported_name] if imported_name else []) + list(reversed(parts))
            if path_parts:
                attributes.append({
                    "file": str(path), "line": node.lineno, "module": module,
                    "name": ".".join(path_parts), "path": path_parts,
                })
    return imports, attributes, parse_errors


@contextlib.contextmanager
def safe_import_path():
    """Exclude task roots whose `unittest.py` shadows Python's stdlib package."""
    old = list(sys.path)
    cwd = Path.cwd()
    filtered = []
    for raw in old:
        candidate = Path(raw).resolve() if raw else cwd.resolve()
        if (candidate / "unittest.py").is_file():
            continue
        filtered.append(raw)
    sys.path[:] = filtered
    try:
        yield
    finally:
        sys.path[:] = old


def package_info():
    try:
        with safe_import_path():
            module = importlib.import_module("flydsl")
    except Exception as exc:
        return {
            "available": False, "version": "not_captured", "module": "not_captured",
            "provider": "not_captured", "error": repr(exc),
        }
    try:
        version = importlib.metadata.version("flydsl")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "not_captured"))
    module_path = str(getattr(module, "__file__", "not_captured"))
    provider = "aiter_vendored_flydsl" if "/aiter/" in module_path.replace("\\", "/").lower() \
        else "standalone_flydsl"
    return {
        "available": True, "version": version, "module": module_path, "provider": provider,
    }


def yields_runtime_value(value):
    return (not inspect.ismodule(value) and not inspect.isclass(value)
            and hasattr(type(value), "__get__"))


def resolve_reference(reference):
    module_name = reference["module"]
    path = reference.get("path")
    if path is None:
        name = reference.get("name")
        path = [] if not name or name == "*" else [name]
    try:
        with safe_import_path():
            module = importlib.import_module(module_name)
    except Exception as exc:
        return {**reference, "reason": "module_import_failed", "detail": repr(exc)}
    if not path:
        return None
    current = module
    module_prefix = module_name
    for index, name in enumerate(path):
        try:
            # Avoid executing descriptors such as FlyDSL's T.f32, which requires an active MLIR
            # Context merely to evaluate. This gate checks surface presence, not call semantics.
            current = inspect.getattr_static(current, name)
            module_prefix = f"{module_prefix}.{name}"
            if index + 1 < len(path) and yields_runtime_value(current):
                # A property such as MLIR's `ir.Context.current` produces its object only when
                # evaluated; the remaining components name attributes of that object, which a
                # static lookup cannot see. Present as far as it can be checked.
                return None
            continue
        except AttributeError:
            pass
        try:
            inspect.getattr_static(type(current), "__getattr__")
            # Dynamic DSL proxy objects (thread_idx.x, block_idx.y, etc.) intentionally do not
            # publish every legal child as a static Python attribute.
            return None
        except AttributeError:
            pass
        # `from pkg import child_module` and lazy package attributes may require importing the
        # progressively longer module path.
        candidate_module = f"{module_name}." + ".".join(path[:index + 1])
        try:
            with safe_import_path():
                current = importlib.import_module(candidate_module)
            module_prefix = candidate_module
            continue
        except Exception:
            return {
                **reference, "reason": "name_absent", "missing_component": name,
                "resolved_prefix": module_prefix,
            }
    return None


def check(paths):
    package = package_info()
    imports, attributes, parse_errors = collect_references(paths)
    missing = []
    if package["available"]:
        for reference in [*imports, *attributes]:
            failure = resolve_reference(reference)
            if failure and failure not in missing:
                missing.append(failure)
    elif imports or attributes:
        missing.append({"reason": "flydsl_unavailable", "detail": package.get("error")})
    compatible = bool(imports) and package["available"] and not parse_errors and not missing
    return {
        "schema_version": 1,
        "compatible": compatible,
        "flydsl": package,
        "files_scanned": len(list(python_files(paths))),
        "imports_checked": len(imports),
        "attributes_checked": len(attributes),
        "missing": missing,
        "parse_errors": parse_errors,
        "reason": "compatible" if compatible else (
            "parse_error" if parse_errors else
            "no_flydsl_import" if not imports else
            "flydsl_unavailable" if not package["available"] else
            "missing_api"
        ),
    }


def check_package_only():
    package = package_info()
    return {
        "schema_version": 1,
        "compatible": package["available"],
        "flydsl": package,
        "files_scanned": 0,
        "imports_checked": 0,
        "attributes_checked": 0,
        "missing": [] if package["available"] else [{
            "reason": "flydsl_unavailable", "detail": package.get("error"),
        }],
        "parse_errors": [],
        "reason": "package_available" if package["available"] else "flydsl_unavailable",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.package_only and not args.paths:
        parser.error("paths are required unless --package-only is set")
    result = check_package_only() if args.package_only else check(args.paths)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{'PASS' if result['compatible'] else 'FAIL'} "
            f"flydsl={result['flydsl']['version']} reason={result['reason']}"
        )
        for missing in result["missing"]:
            print(f"  {missing}")
    # Structured workflow callers branch on `compatible`; keep JSON available even for a failed
    # gate instead of turning the transport itself into an agent/tool error.
    return 0 if args.json or result["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
