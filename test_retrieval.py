"""Run the 5 evaluation queries from planning.md through the ChromaDB retriever 
and print the top-k chunks with distance scores + a verdict per query.

Distance verdicts follow the thresholds:
  best distance < 0.5     → [good]
  0.5 ≤ best distance < 0.6 → [weak]
  best distance ≥ 0.6     → [bad]

Usage:
    python test_retrieval.py            # run all 5 queries
    python test_retrieval.py 1 3 5      # run specific query IDs
    python test_retrieval.py -k 6       # override top-k (default = embed.DEFAULT_TOP_K = 4)
"""

from __future__ import annotations

import argparse
import sys

from embed import DEFAULT_TOP_K, retrieve

# Note: `retrieve(query, k)` is implemented in `embed.py` and returns a list of
# dicts: {"text": str, "metadata": dict, "distance": float}. The
# `distance` value is a cosine-distance (lower is more similar).

EVAL_QUERIES: list[dict] = [
    {
        "id": 1,
        "query": "How are the reliability and wait times of the shuttles at The Residences at Lorenzo?",
        "expected": "The shuttles are rarely on time and are often crowded to maximum capacity.",
    },
    {
        "id": 2,
        "query": "Which specific streets or patrol boundaries are considered the safest for walking around the USC campus at night?",
        "expected": "30th St, 29th St, Ellendale Place, Orchard Ave, and USC Village.",
    },
    {
        "id": 3,
        "query": "When comparing Tuscany and Icon, which apartment complex is more expensive?",
        "expected": "Icon Plaza is generally more expensive because it offers true private single-bedroom units.",
    },
    {
        "id": 4,
        "query": "Which off-campus housing companies or apartment buildings offer furnished rooms or lenient guarantor requirements for international students?",
        "expected": "International students frequently choose housing companies such as Tripalink, Stuho, and Orion Housing, as well as large apartment complexes like The Lorenzo and University Gateway.",
    },
    {
        "id": 5,
        "query": "What are the most common daily annoyances regarding the elevators and street noise at University Gateway?",
        "expected": "Tenants frequently complain that the eight available elevators take too long to arrive during peak morning hours and other busy times. Units facing the main roads also experience high levels of city and traffic noise.",
    },
    {
        "id": 6,
        "query": "Why do students complain about using their cell phones inside the University Gateway apartments?",
        "expected": "The physical design of the building blocks cell data, resulting in no signal or no bars, which forces residents to leave the building to do anything requiring the internet."
    },
    {
        "id": 7,
        "query": "Which four streets form the perimeter of the USC bubble, an area noted for being highly patrolled by private security?",
        "expected": "The area is bounded by Expo, Fig (Figueroa), Adams, and Vermont."
    }
]

PREVIEW_CHARS = 400


def verdict_for(best_distance: float) -> str:
    # Thresholds chosen for cosine distance returned by Chroma/HNSW.
    # Lower distance = higher similarity. Adjust thresholds if you change
    # the embedding model or distance metric.
    if best_distance < 0.5:
        return "[good]"
    if best_distance < 0.6:
        return "[weak]"
    return "[bad]"


def run_one(item: dict, k: int) -> str:
    print(f"\n{'=' * 90}")
    print(f"Q{item['id']}: {item['query']}")
    print(f"Expected: {item['expected']}")
    print(f"{'-' * 90}")

    # Call into the Chroma-backed retriever. Each hit is a dict with
    # - "text": the chunk text
    # - "metadata": fields like source, parent_title, chunk_index
    # - "distance": float (cosine distance; lower is better)
    hits = retrieve(item["query"], k=k)
    if not hits:
        print("  (no hits — collection empty?)")
        return "[bad]"

    for i, h in enumerate(hits, 1):
        # Metadata is the original chunk metadata written at index time.
        m = h["metadata"]
        title = f' — "{m["parent_title"]}"' if m.get("parent_title") else ""
        print(
            f"\n  #{i}  distance={h['distance']:.3f}  "
            f"{m['source']}#{m['chunk_index']}{title}"
        )
        # Show a preview of the chunk text; the full chunk is stored in the
        # Chroma collection and available via `h["text"]`.
        text = h["text"]
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"
        # Indent the preview so it's visually grouped under the hit header
        for line in preview.splitlines():
            print(f"     {line}")

    best = hits[0]["distance"]
    v = verdict_for(best)
    print(f"\n  verdict: {v}  best distance = {best:.3f}")
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ids",
        nargs="*",
        type=int,
        help="optional list of query IDs (1-5) to run; default = all",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"top-k chunks per query (default {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    if args.ids:
        wanted = set(args.ids)
        queries = [q for q in EVAL_QUERIES if q["id"] in wanted]
        missing = wanted - {q["id"] for q in queries}
        if missing:
            print(f"unknown query id(s): {sorted(missing)} (valid: 1-5)", file=sys.stderr)
            return 2
    else:
        queries = EVAL_QUERIES

    print(f"Running {len(queries)} eval query/queries at top-k={args.k}")

    tally = {"[good]": 0, "[weak]": 0, "[bad]": 0}
    for q in queries:
        tally[run_one(q, args.k)] += 1

    print(f"\n{'=' * 90}")
    print(
        f"summary: {tally['[good]']} good / {tally['[weak]']} weak / {tally['[bad]']} bad"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
