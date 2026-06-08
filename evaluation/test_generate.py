"""Run the shared evaluation question bank through `generate.answer()` and
write a timestamped report into `evaluation/`.

Per question, the report records:
  - the question and expected answer (lifted from questions.json)
  - a refusal flag (yes/no)
  - the full LLM answer, including the appended References block
  - the retrieved chunks (rank, distance, source, parent_title, preview)

Output:
  evaluation/test_generate_output_YYYYMMDDHHMMSS.txt

Usage:
    python evaluation/test_generate.py                     # all questions
    python evaluation/test_generate.py 1 3 5               # specific IDs
    python evaluation/test_generate.py -k 6 -t 0.3         # tune retrieval/sampling
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the repo root to sys.path so `from generate import ...` works regardless
# of the user's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from embed import DEFAULT_TOP_K  # noqa: E402
from generate import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    REFUSAL,
    answer,
)

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
PREVIEW_CHARS = 400


def _load_questions() -> list[dict]:
    with QUESTIONS_PATH.open() as f:
        return json.load(f)


def _is_refusal(result: dict) -> bool:
    # The refusal path in generate.answer() returns sources == [] and the
    # body equal to the REFUSAL sentence (no References block appended).
    return not result["sources"] and result["answer"].strip() == REFUSAL


def _format_chunks(hits: list[dict], out: list[str]) -> None:
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        title = f' — "{m["parent_title"]}"' if m.get("parent_title") else ""
        out.append(
            f"\n  #{i}  distance={h['distance']:.3f}  "
            f"{m['source']}#{m['chunk_index']}{title}"
        )
        text = h["text"]
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
        for line in preview.splitlines():
            out.append(f"     {line}")


def _run_one(
    item: dict,
    k: int,
    temperature: float,
    max_tokens: int,
    out: list[str],
) -> bool:
    out.append(f"\n{'=' * 90}")
    out.append(f"Q{item['id']}: {item['query']}")
    out.append(f"Expected: {item['expected']}")
    out.append(f"{'-' * 90}")

    result = answer(
        item["query"],
        k=k,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    refused = _is_refusal(result)
    out.append(f"\n  [refusal: {'yes' if refused else 'no'}]")

    out.append("\n  --- System answer ---")
    for line in result["answer"].splitlines():
        out.append(f"  {line}")

    out.append(f"\n  --- Retrieved chunks (k={k}) ---")
    if not result["hits"]:
        out.append("  (no hits — collection empty?)")
    else:
        _format_chunks(result["hits"], out)

    return refused


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ids",
        nargs="*",
        type=int,
        help="optional list of question IDs to run; default = all",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"top-k chunks per query (default {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Groq sampling temperature (default {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"cap on LLM response length (default {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    all_questions = _load_questions()
    valid_ids = {q["id"] for q in all_questions}

    if args.ids:
        wanted = set(args.ids)
        missing = wanted - valid_ids
        if missing:
            print(
                f"unknown question id(s): {sorted(missing)} "
                f"(valid: {sorted(valid_ids)})",
                file=sys.stderr,
            )
            return 2
        questions = [q for q in all_questions if q["id"] in wanted]
    else:
        questions = all_questions

    lines: list[str] = []
    lines.append(
        f"Running {len(questions)} eval question(s) through generate.answer() "
        f"at top-k={args.k}, temperature={args.temperature}, max_tokens={args.max_tokens}"
    )

    refusals = 0
    for q in questions:
        if _run_one(q, args.k, args.temperature, args.max_tokens, lines):
            refusals += 1

    answered = len(questions) - refusals
    lines.append(f"\n{'=' * 90}")
    lines.append(f"summary: {answered} answered / {refusals} refusals")

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = Path(__file__).parent / f"test_generate_output_{stamp}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
