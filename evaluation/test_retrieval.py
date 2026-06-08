"""Run the shared evaluation question bank through ChromaDB retrieval and write
a timestamped report into `evaluation/`.

Distance verdicts follow the thresholds:
  best distance < 0.5     → [good]
  0.5 ≤ best distance < 0.6 → [weak]
  best distance ≥ 0.6     → [bad]

Output:
  evaluation/test_retrieval_output_YYYYMMDDHHMMSS.txt

Usage:
    python evaluation/test_retrieval.py            # run every question in questions.json
    python evaluation/test_retrieval.py 1 3 5      # run specific question IDs
    python evaluation/test_retrieval.py -k 6       # override top-k (default = embed.DEFAULT_TOP_K)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add the repo root to sys.path so `from embed import ...` works regardless of
# the user's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from embed import DEFAULT_TOP_K, retrieve  # noqa: E402  (import after sys.path edit)

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
PREVIEW_CHARS = 400


def _load_questions() -> list[dict]:
    with QUESTIONS_PATH.open() as f:
        return json.load(f)


def verdict_for(best_distance: float) -> str:
    # Thresholds chosen for cosine distance returned by Chroma/HNSW.
    # Lower distance = higher similarity. Adjust if the embedding model
    # or distance metric changes.
    if best_distance < 0.5:
        return "[good]"
    if best_distance < 0.6:
        return "[weak]"
    return "[bad]"


def _run_one(item: dict, k: int, out: list[str]) -> str:
    out.append(f"\n{'=' * 90}")
    out.append(f"Q{item['id']}: {item['query']}")
    out.append(f"Expected: {item['expected']}")
    out.append(f"{'-' * 90}")

    hits = retrieve(item["query"], k=k)
    if not hits:
        out.append("  (no hits — collection empty?)")
        return "[bad]"

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

    best = hits[0]["distance"]
    v = verdict_for(best)
    out.append(f"\n  verdict: {v}  best distance = {best:.3f}")
    return v


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
    lines.append(f"Running {len(questions)} eval query/queries at top-k={args.k}")

    tally = {"[good]": 0, "[weak]": 0, "[bad]": 0}
    for q in questions:
        tally[_run_one(q, args.k, lines)] += 1

    lines.append(f"\n{'=' * 90}")
    lines.append(
        f"summary: {tally['[good]']} good / {tally['[weak]']} weak / {tally['[bad]']} bad"
    )

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = Path(__file__).parent / f"test_retrieval_output_{stamp}.txt"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
