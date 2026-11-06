#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def kv_bytes(tokens: int, layers: int, kv_heads: int, head_dim: int, bytes_per_value: int = 2) -> int:
    return 2 * tokens * layers * kv_heads * head_dim * bytes_per_value


def main() -> None:
    visible_scores = [0.0, math.log(2), 0.0]
    weights = softmax(visible_scores)
    values = [2.0, 0.0, 4.0]
    mixture = sum(weight * value for weight, value in zip(weights, values))
    cache = {name: kv_bytes(4096, 32, heads, 128) for name, heads in {"MHA": 32, "GQA": 8, "MQA": 1}.items()}
    timings = json.loads((Path(__file__).parent / "timings_fixture.json").read_text())
    result = {"attention_weights": [round(x, 4) for x in weights], "attention_output": round(mixture, 4), "kv_cache_bytes": cache, "timings": timings}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
