#!/usr/bin/env python3
"""Run lightweight checks that do not require the lecture environments."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATTERNS = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"C:\\Users\\[^\\\s]+\\"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
)


def check_notebooks() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*.ipynb"):
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: invalid notebook JSON: {exc}")
            continue
        if notebook.get("nbformat") != 4:
            errors.append(f"{path.relative_to(ROOT)}: expected notebook format 4")
    return errors


def check_python() -> list[str]:
    errors: list[str] = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
    return errors


def check_local_paths_and_tokens() -> list[str]:
    errors: list[str] = []
    suffixes = {".md", ".py", ".json", ".yml", ".yaml", ".sh", ".ipynb"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(
                    f"{path.relative_to(ROOT)}: contains forbidden pattern {pattern.pattern!r}"
                )
    return errors


def main() -> int:
    errors = check_notebooks() + check_python() + check_local_paths_and_tokens()
    if errors:
        print("Repository checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Repository checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
