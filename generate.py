"""Grounded generation using the Groq LLM.

Pulls the top-k chunks from `embed.retrieve()`, builds a strict grounding
prompt, calls Groq's `llama-3.3-70b-versatile`, and programmatically returns
the unique source filenames alongside the answer so attribution does not
depend on the model remembering to cite (the system prompt also asks it to
cite inline, but we don't trust that alone).

Importable:
    from generate import answer
    result = answer("which streets are safest at night?")
    print(result["answer"])
    print(result["sources"])

CLI:
    python generate.py --query "..."         # one-off generation
    python generate.py -q "..." -k 6         # override top-k
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from embed import DEFAULT_TOP_K, retrieve

REPO_ROOT = Path(__file__).parent

# Load .env once at module load so importers (app.py) also pick up the key.
load_dotenv(REPO_ROOT / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"

# Strict grounding contract. The retrieved chunks are injected into {context};
# the user's question is sent as the user message. Rule 3's refusal string is
# exact-match so the README evaluation can grep for it.
SYSTEM_PROMPT = """You are "The Unofficial Guide," a careful assistant that answers questions about off-campus housing near USC / Downtown LA using only the student-generated documents provided below.

Strict rules — follow all of them:
1. Answer ONLY using the information in the documents below. Do not use outside knowledge, prior training, or general assumptions.
2. Cite the source filename in square brackets immediately after any claim drawn from it — e.g. "Shuttles are often late [source_8_yelp_clean.txt]." If a sentence combines multiple sources, cite each one.
3. If the documents do not contain enough information to answer the question, reply with exactly this sentence and nothing else: I don't have enough information on that.
4. Do not invent facts, building names, prices, quotes, or details. If something is not in the documents, leave it out.
5. Keep the answer concise — a short paragraph or a few bullets is usually enough.

Documents:
{context}"""


def _build_context(hits: list[dict]) -> str:
    # Each chunk gets a header with its source filename (so the LLM can cite
    # by name) and its parent_title when present (so "this building" / "it"
    # references in the chunk are still disambiguated — this is the mitigation
    # for the "Lost Pronoun Context Across Chunks" risk in planning.md).
    blocks = []
    for h in hits:
        m = h["metadata"]
        title = m.get("parent_title") or ""
        header = f"[source: {m['source']}]"
        if title:
            header += f" — {title}"
        blocks.append(f"{header}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def _unique_sources(hits: list[dict]) -> list[str]:
    # Dedupe while preserving retrieval order so the most-relevant source
    # appears first in the citation list.
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        s = h["metadata"]["source"]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def answer(query: str, k: int = DEFAULT_TOP_K) -> dict:
    """Run retrieval + grounded generation. Returns answer, sources, raw hits."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set — copy .env.example to .env and add your key"
        )

    hits = retrieve(query, k=k)
    if not hits:
        return {
            "answer": "I don't have enough information on that.",
            "sources": [],
            "hits": [],
        }

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=_build_context(hits))},
            {"role": "user", "content": query},
        ],
        # Low temperature: grounded QA wants faithful, deterministic answers.
        temperature=0.2,
    )
    text = (completion.choices[0].message.content or "").strip()
    return {"answer": text, "sources": _unique_sources(hits), "hits": hits}


def main() -> int:
    parser = argparse.ArgumentParser(description="One-off grounded-generation CLI.")
    parser.add_argument("--query", "-q", required=True, help="question to ask")
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"top-k chunks (default {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    result = answer(args.query, k=args.k)
    print("\n=== Answer ===\n")
    print(result["answer"])
    print("\n=== Sources retrieved ===")
    for s in result["sources"]:
        print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
