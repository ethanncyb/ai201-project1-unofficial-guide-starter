"""Grounded generation using the Groq LLM with paper-style numbered citations.

Pulls the top-k chunks from `embed.retrieve()`, assigns a citation number per
unique source (first-seen-first-numbered), builds a strict grounding prompt
that tells the LLM to cite with `[1]` / `[2]`, calls Groq's
`llama-3.3-70b-versatile`, and programmatically appends a `References:`
block to the answer so attribution survives even if the model forgets to
cite inline.

Importable:
    from generate import answer
    result = answer("which streets are safest at night?")
    print(result["answer"])
    print(result["sources"])   # [{"number": 1, "source": "source_10_reddit_clean.txt"}, ...]

CLI:
    python generate.py -q "..."                              # one-off generation
    python generate.py -q "..." -k 6 -t 0.4 --max-tokens 800
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from embed import DEFAULT_TOP_K, retrieve

REPO_ROOT = Path(__file__).parent

# Load .env once at module load so importers (app.py) also pick up the key.
load_dotenv(REPO_ROOT / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512

# Strict grounding contract. {context} is replaced with the retrieved chunks,
# each prefixed by a `[N] source_filename — parent_title` header so the LLM
# can cite with `[N]`. Rule 3's refusal string is verbatim so the README
# evaluation table and the test harness can grep for it.
SYSTEM_PROMPT = """You are "The Unofficial Guide," a careful assistant that answers questions about off-campus housing AND day-to-day living near USC / Downtown LA — including building reviews, shuttle reliability, neighborhood safety, supermarkets and nearby amenities, transit, commuting, and the everyday realities of living in the area. Use only the student-generated documents provided below.

Strict rules — follow all of them:
1. Answer ONLY using the information in the documents below. Do not use outside knowledge, prior training, or general assumptions.
2. Cite sources inline using bracketed numbers like [1] or [1][2]. The numbers correspond to the [N] source_filename headers in the Documents block below. Do not invent new citation numbers and do not write filenames inline — use only the numbers.
3. If the documents do not contain enough information to answer the question, reply with exactly this sentence and nothing else: I don't have enough information on that.
4. Do not invent facts, building names, prices, quotes, or details. If something is not in the documents, leave it out.
5. Keep the answer concise — a short paragraph or a few bullets is usually enough. Do not write your own "References:" section at the end; one is appended programmatically.

Documents:
{context}"""

REFUSAL = "I don't have enough information on that."

# Matches inline citations like [1], [12], etc. Used to figure out which
# citation numbers the LLM actually wrote in its answer, so we can render a
# References block that only lists the cited sources (not every retrieved
# one). Anchored to digits-only to avoid matching things like [URL] or [...].
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _cited_numbers(body: str) -> set[int]:
    """Return the set of [N] citation numbers actually used in `body`."""
    return {int(m.group(1)) for m in CITATION_PATTERN.finditer(body)}


def _assign_citation_numbers(hits: list[dict]) -> list[dict]:
    """Assign [1], [2], … to each retrieved chunk in retrieval order.

    Returns a list of dicts {"number": int, "source": str, "chunk_index": int}
    in citation order (index 0 is `[1]`, index 1 is `[2]`, etc.). Each chunk
    gets its own number, even if multiple chunks share a source filename.
    """
    refs: list[dict] = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        refs.append({
            "number": i,
            "source": m["source"],
            "chunk_index": m["chunk_index"],
        })
    return refs


def _build_context(hits: list[dict], refs: list[dict]) -> str:
    # Each chunk renders under a `[N] source#chunk_index — parent_title` header
    # so the LLM cites with the chunk-specific number. Each chunk gets its own
    # number, in retrieval order.
    blocks = []
    for h, r in zip(hits, refs):
        m = h["metadata"]
        title = m.get("parent_title") or ""
        header = f"[{r['number']}] {m['source']}#{m['chunk_index']}"
        if title:
            header += f" — {title}"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def _format_references(refs: list[dict]) -> str:
    return "\n".join(f"[{r['number']}] {r['source']}#{r['chunk_index']}" for r in refs)


def answer(
    query: str,
    k: int = DEFAULT_TOP_K,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Run retrieval + grounded generation.

    Returns:
      - "answer":  LLM prose with inline [N] citations, followed by a
                   programmatic `References:` block (omitted when the model
                   returns the exact refusal sentence).
      - "sources": [{"number": int, "source": str, "chunk_index": int}, ...]
                   in citation order (one entry per retrieved chunk).
      - "hits":    raw retrieval list (for the UI debug panel).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set — copy .env.example to .env and add your key"
        )

    hits = retrieve(query, k=k)
    if not hits:
        return {"answer": REFUSAL, "sources": [], "hits": [], "cited_numbers": set()}

    refs = _assign_citation_numbers(hits)

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=_build_context(hits, refs)),
            },
            {"role": "user", "content": query},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    body = (completion.choices[0].message.content or "").strip()

    # Don't append references when the model legitimately refused — the brief
    # wants the refusal sentence to be the entire response.
    if body == REFUSAL:
        return {"answer": body, "sources": [], "hits": hits, "cited_numbers": set()}

    # Filter the References block to only the citations the model actually used.
    # If the model wrote an answer but cited nothing, fall back to listing every
    # retrieved source so the viewer still sees what the answer was drawn from.
    cited = _cited_numbers(body)
    shown_refs = [r for r in refs if r["number"] in cited] if cited else refs
    cited_numbers = {r["number"] for r in shown_refs}

    answer_text = f"{body}\n\nReferences:\n{_format_references(shown_refs)}"
    return {"answer": answer_text, "sources": refs, "hits": hits, "cited_numbers": cited_numbers}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-off grounded-generation CLI.")
    parser.add_argument("--query", "-q", required=True, help="question to ask")
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"top-k chunks (default {DEFAULT_TOP_K})",
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

    result = answer(
        args.query,
        k=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("\n=== Answer ===\n")
    print(result["answer"])
    if result["sources"]:
        print("\n=== Sources (cite map) ===")
        for r in result["sources"]:
            print(f"  [{r['number']}] {r['source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
