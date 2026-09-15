#!/usr/bin/env python3
"""Execute the single embedded runtime validator from an exact Skill file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


TAG = "expert-skill-runtime-python"


def extract_runtime(skill_file: Path) -> str:
    matches = re.findall(
        rf"```{TAG}[^\n]*\n(.*?)\n```",
        skill_file.read_text(),
        flags=re.S,
    )
    if len(matches) != 1:
        raise ValueError(
            f"{skill_file} must contain exactly one fenced {TAG} block"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--skill-file", required=True)
    known, remaining = parser.parse_known_args()
    skill_file = Path(known.skill_file).resolve()
    source = extract_runtime(skill_file)
    namespace = {
        "__name__": "__embedded_expert_skill_runtime__",
        "__file__": f"{skill_file}#{TAG}",
    }
    old_argv = sys.argv
    try:
        sys.argv = [str(skill_file), *remaining]
        exec(compile(source, namespace["__file__"], "exec"), namespace)
        embedded_main = namespace.get("main")
        if not callable(embedded_main):
            raise RuntimeError("embedded runtime block must define main()")
        result = embedded_main()
        return int(result or 0)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
