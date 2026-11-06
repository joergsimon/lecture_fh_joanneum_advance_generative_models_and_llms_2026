#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def score(expected: dict, predicted: dict) -> dict[str, bool]:
    return {
        "tool_exact": predicted.get("tool") == expected["tool"],
        "arguments_exact": predicted.get("arguments") == expected["arguments"],
        "schema_valid": set(predicted) >= {"id", "tool", "arguments"} and isinstance(predicted.get("arguments"), dict),
    }


def main() -> None:
    here = Path(__file__).parent
    cases = {case["id"]: case for case in json.loads((here / "cases.json").read_text())}
    outputs = json.loads((here / "outputs.json").read_text())
    for variant in ("base", "adapted"):
        rows = [score(cases[item["id"]], item) for item in outputs[variant]]
        totals = {key: sum(row[key] for row in rows) for key in rows[0]}
        print(variant, totals, "of", len(rows))


if __name__ == "__main__":
    main()
