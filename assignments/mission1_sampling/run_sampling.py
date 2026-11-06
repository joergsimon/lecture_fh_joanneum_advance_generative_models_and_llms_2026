#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

CONFIGS = [
    {"name": "greedy", "temperature": 0.0, "top_p": 1.0},
    {"name": "balanced", "temperature": 0.7, "top_p": 0.9},
    {"name": "exploratory", "temperature": 1.2, "top_p": 0.95},
]


def ollama_generate(model: str, prompt: str, config: dict[str, float]) -> tuple[str, float]:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": {"temperature": config["temperature"], "top_p": config["top_p"]}}).encode()
    start = time.perf_counter()
    request = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        text = json.loads(response.read())["response"]
    return text, round((time.perf_counter() - start) * 1000, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--ollama-model")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    here = Path(__file__).parent
    if args.replay or not args.ollama_model:
        data = json.loads((here / "replay_outputs.json").read_text())
        for config in data["configurations"]:
            config["latency_ms"] = [0.0] * len(config["samples"])
    else:
        prompt = (here / "prompt.txt").read_text()
        data = {"model": args.ollama_model, "configurations": []}
        for config in CONFIGS:
            samples, latencies = [], []
            for _ in range(5):
                sample, latency = ollama_generate(args.ollama_model, prompt, config)
                samples.append(sample)
                latencies.append(latency)
            data["configurations"].append({**config, "samples": samples, "latency_ms": latencies})
    args.out.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
