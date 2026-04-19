#!/usr/bin/env -S uv run
"""Emit LongCoT questions as JSON lines (stdout) for the TS RLM harness."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import longcot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default="logic",
        help="logic | cs | chemistry | chess | math (default: logic)",
    )
    parser.add_argument(
        "--difficulty",
        default="easy",
        help="easy | medium | hard | longcot-mini alias not supported here; use easy (default: easy)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=5,
        help="Cap number of questions after filtering (default: 5)",
    )
    args = parser.parse_args()

    questions = longcot.load_questions(domain=args.domain, difficulty=args.difficulty)
    for q in questions[: max(args.max, 0)]:
        line = json.dumps(asdict(q), ensure_ascii=False)
        sys.stdout.write(line + "\n")


if __name__ == "__main__":
    main()
