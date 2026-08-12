"""Parse rocprofv3 ATT per-instruction statistics into evidence categories."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

ATT_COLUMNS = ("Instruction", "Hitcount", "Latency", "Stall", "Idle")

__all__ = [
    "ATT_COLUMNS",
    "load_instruction_category_map",
    "parse_att_stats_csv",
    "read_att_occupancy",
]


def load_instruction_category_map(path_or_dict) -> dict:
    if isinstance(path_or_dict, Mapping):
        raw = dict(path_or_dict)
    else:
        raw = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    compiled = {}
    for category, pattern in raw.items():
        compiled[str(category)] = re.compile(str(pattern))
    return compiled


def _instruction_category(instruction: str, category_map) -> tuple[str, str]:
    opcode = instruction.split()[0].lower() if instruction.strip() else "?"
    for category, pattern in category_map.items():
        if pattern.search(opcode):
            return category, opcode
    return "unclassified", opcode


def _integer(row: Mapping[str, str], field: str, line: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ATT CSV line {line} has invalid {field}") from error
    if value < 0:
        raise ValueError(f"ATT CSV line {line} has negative {field}")
    return value


def parse_att_stats_csv(path: str | Path, category_map) -> dict:
    """Aggregate ATT Hitcount/Latency/Stall/Idle by instruction category."""
    source = Path(path)
    categories = defaultdict(
        lambda: {
            "static_instruction_count": 0,
            "hitcount": 0,
            "latency_cycles": 0,
            "stall_cycles": 0,
            "idle_cycles": 0,
            "opcodes": set(),
        }
    )
    instructions = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in ATT_COLUMNS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"ATT CSV is missing columns: {missing}")
        for line, row in enumerate(reader, start=2):
            instruction = (row.get("Instruction") or "").strip()
            if not instruction or instruction.startswith(";"):
                continue
            category, opcode = _instruction_category(instruction, category_map)
            values = {
                "hitcount": _integer(row, "Hitcount", line),
                "latency_cycles": _integer(row, "Latency", line),
                "stall_cycles": _integer(row, "Stall", line),
                "idle_cycles": _integer(row, "Idle", line),
            }
            entry = categories[category]
            entry["static_instruction_count"] += 1
            entry["opcodes"].add(opcode)
            for key, value in values.items():
                entry[key] += value
            instructions.append(
                {
                    "instruction": instruction,
                    "opcode": opcode,
                    "category": category,
                    **values,
                    "stall_idle_cycles": values["stall_cycles"] + values["idle_cycles"],
                    "source": row.get("Source") or "",
                }
            )
    totals = {
        key: sum(entry[key] for entry in categories.values())
        for key in ("hitcount", "latency_cycles", "stall_cycles", "idle_cycles")
    }
    totals["busy_cycles"] = (
        totals["latency_cycles"] + totals["stall_cycles"] + totals["idle_cycles"]
    )
    output_categories = {}
    for category, entry in categories.items():
        category_cycles = (
            entry["latency_cycles"] + entry["stall_cycles"] + entry["idle_cycles"]
        )
        output_categories[category] = {
            **{key: value for key, value in entry.items() if key != "opcodes"},
            "opcodes": sorted(entry["opcodes"]),
            "accounted_cycle_share_pct": (
                category_cycles / totals["busy_cycles"] * 100.0
                if totals["busy_cycles"]
                else 0.0
            ),
            "stall_share_pct": (
                entry["stall_cycles"] / category_cycles * 100.0
                if category_cycles
                else 0.0
            ),
            "idle_share_pct": (
                entry["idle_cycles"] / category_cycles * 100.0
                if category_cycles
                else 0.0
            ),
        }
    top = sorted(
        instructions,
        key=lambda item: item["stall_idle_cycles"],
        reverse=True,
    )[:20]
    for value in totals.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("ATT aggregate produced a non-finite value")
    return {
        "source": str(source),
        "categories": output_categories,
        "totals": totals,
        "top_stall_idle_instructions": top,
        "semantics": {
            "hitcount": "instruction issues observed on traced SIMD/waves",
            "latency_cycles": "cycles attributed to instruction execution",
            "stall_cycles": "wave stalled on dependency or resource at instruction",
            "idle_cycles": "wave had no instruction to issue at instruction/barrier",
        },
        "scope_warning": (
            "ATT is sampled thread/wave evidence from selected GPU/CU/SIMDs; "
            "it is not whole-device time or cross-GPU traffic."
        ),
    }


def read_att_occupancy(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ATT occupancy JSON must be an object")
    return dict(payload)
