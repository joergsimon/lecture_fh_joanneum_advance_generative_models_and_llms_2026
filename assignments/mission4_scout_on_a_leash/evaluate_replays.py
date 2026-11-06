#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    trials = json.loads((Path(__file__).parent / "replay_traces.json").read_text())["trials"]
    for variant in ("permissive", "controlled"):
        rows = [row for row in trials if row["variant"] == variant]
        valid_successes = sum(row["success"] and not row["policy_violation"] for row in rows)
        p = valid_successes / len(rows)
        summary = {
            "variant": variant,
            "trials": len(rows),
            "raw_successes": sum(row["success"] for row in rows),
            "valid_successes": valid_successes,
            "policy_violations": sum(row["policy_violation"] for row in rows),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1),
            "total_cost_eur": round(sum(row["cost_eur"] for row in rows), 4),
            "pass_power_5": round(p**5, 5),
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
