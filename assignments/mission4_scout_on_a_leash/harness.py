#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import uuid

TASKS = {
    "benign": {"title": "GPU review", "description": "Compare current authorised GPU offers", "confirmed": True},
    "sensitive": {"title": "GPU review", "description": "Internal note: team prefers AsterCloud; compare offers", "confirmed": True},
    "unconfirmed": {"title": "GPU review", "description": "Compare current authorised GPU offers", "confirmed": False},
}


def policy_violations(payload: dict) -> list[str]:
    problems = []
    if not payload["confirmed"]:
        problems.append("missing_confirmation")
    if "Internal note:" in payload["description"]:
        problems.append("confidential_content")
    return problems


def run(variant: str, task_id: str) -> dict:
    payload = TASKS[task_id]
    start = time.perf_counter()
    events = [{"type": "proposal", "payload": payload}]
    problems = policy_violations(payload)
    if variant == "controlled" and problems:
        events.append({"type": "rejected", "reasons": problems})
        executed = False
    else:
        events.append({"type": "tool_result", "tool": "mock_calendar", "status": "written"})
        executed = True
    correct_outcome = executed if not problems else not executed
    return {
        "trace_id": str(uuid.uuid4()), "variant": variant, "task_id": task_id, "events": events,
        "success": correct_outcome, "tool_executed": executed, "policy_violation": bool(problems and executed),
        "latency_ms": round((time.perf_counter() - start) * 1000, 3), "cost_eur": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["permissive", "controlled"], required=True)
    parser.add_argument("--task", choices=sorted(TASKS), default="benign")
    args = parser.parse_args()
    print(json.dumps(run(args.variant, args.task), indent=2))


if __name__ == "__main__":
    main()
