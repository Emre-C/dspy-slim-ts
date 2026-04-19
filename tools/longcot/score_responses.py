#!/usr/bin/env -S uv run
"""Score a JSONL of RLM outputs using LongCoT verify()."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import longcot
from longcot import (
    ChemistryVerifyOptions,
    MathVerifyOptions,
    Question,
    VerifyOptions,
)


def _question_from_obj(obj: dict[str, Any]) -> Question:
    return Question(
        question_id=str(obj["question_id"]),
        domain=str(obj["domain"]),
        difficulty=str(obj["difficulty"]),
        prompt=str(obj["prompt"]),
        problem=obj.get("problem"),
        answer=obj.get("answer"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "responses_jsonl",
        help="Path to JSONL: each line has 'question' (object) and 'response_text' (string)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable Gemini fallback judges (stricter local verification only)",
    )
    args = parser.parse_args()

    options: VerifyOptions | None = None
    if args.no_fallback:
        options = VerifyOptions(
            math=MathVerifyOptions(enable_fallback=False),
            chemistry=ChemistryVerifyOptions(enable_fallback=False),
        )

    total = 0
    correct = 0
    incorrect = 0
    failed = 0

    with open(args.responses_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            q = _question_from_obj(row["question"])
            text = str(row.get("response_text") or "")
            err = row.get("error")
            if err:
                failed += 1
                continue
            try:
                ok = longcot.verify(q, text, options=options)
            except Exception:
                failed += 1
                continue
            if ok:
                correct += 1
            else:
                incorrect += 1

    denom = correct + incorrect
    accuracy = (correct / denom) if denom else 0.0
    overall = (correct / total) if total else 0.0

    summary = {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "failed": failed,
        "accuracy": accuracy,
        "overall_accuracy": overall,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
