"""Chunk the cleaned text files into retrieval-ready blocks.

Splits each documents/source_*_clean.txt with LangChain's RecursiveCharacterTextSplitter
sized to the MiniLM tokenizer (the same one used downstream for embeddings) at the
~200 token / ~40 token overlap target from planning.md. Writes documents/chunks.jsonl.

Usage:
    python chunk.py                  # chunk everything in documents/
    python chunk.py --inspect 8      # also print 8 sample chunks
    python chunk.py --seed 7         # reproducible sample selection
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNKS_PATH = DOCUMENTS_DIR / "chunks.jsonl"

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = 200
CHUNK_OVERLAP_TOKENS = 40
MIN_CHUNK_CHARS = 40  # drop trailing splinters too small to carry signal

CLEAN_RE = re.compile(r"^source_(\d+)_(reddit|wiki|ratemydorm|yelp)_clean\.txt$")

# Pull the parent context (thread title / dorm name / business / wiki title) so each chunk
# carries its source's headline in metadata even after the chunker drops it.
PARENT_TITLE_PATTERNS = {
    "reddit": re.compile(r"^Thread Title:\s*(.+)$", re.MULTILINE),
    "wiki": re.compile(r"^# Guide:\s*(.+)$", re.MULTILINE),
    "ratemydorm": re.compile(r"^Dorm:\s*(.+)$", re.MULTILINE),
    "yelp": re.compile(r"^Business:\s*(.+)$", re.MULTILINE),
}


def parent_title(source_type: str, text: str) -> str:
    pat = PARENT_TITLE_PATTERNS.get(source_type)
    if not pat:
        return ""
    m = pat.search(text)
    return m.group(1).strip() if m else ""


def build_splitter() -> RecursiveCharacterTextSplitter:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )


def chunk_all() -> list[dict]:
    splitter = build_splitter()
    out: list[dict] = []
    sources = sorted(DOCUMENTS_DIR.glob("source_*_clean.txt"))
    if not sources:
        print(f"no clean text files found in {DOCUMENTS_DIR} — run preprocess.py first")
        return out

    for path in sources:
        m = CLEAN_RE.match(path.name)
        if not m:
            print(f"skip {path.name}: filename does not match expected pattern")
            continue
        source_id, source_type = int(m.group(1)), m.group(2)
        text = path.read_text(encoding="utf-8")
        title = parent_title(source_type, text)

        raw_chunks = splitter.split_text(text)
        kept = 0
        for i, chunk in enumerate(raw_chunks):
            stripped = chunk.strip()
            if len(stripped) < MIN_CHUNK_CHARS:
                continue
            out.append(
                {
                    "text": stripped,
                    "source": path.name,
                    "source_id": source_id,
                    "source_type": source_type,
                    "parent_title": title,
                    "chunk_index": kept,
                    "char_count": len(stripped),
                }
            )
            kept += 1
        print(f"  {path.name}: {kept} chunks ({len(raw_chunks) - kept} dropped as too small)")

    return out


def write_jsonl(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def print_samples(chunks: list[dict], n: int, seed: int | None) -> None:
    if not chunks:
        return
    rng = random.Random(seed) if seed is not None else random.Random()
    picks = rng.sample(chunks, min(n, len(chunks)))
    print(f"\n=== {len(picks)} sample chunks ===")
    for i, c in enumerate(picks):
        print(f"\n=== Sample chunks {i+1} ===")
        title_part = f" — {c['parent_title']}" if c["parent_title"] else ""
        print(
            f"\n[{c['source']} #{c['chunk_index']} | {c['char_count']} chars{title_part}]"
        )
        text = c["text"]
        preview = text if len(text) <= 600 else text[:600] + "…"
        print(preview)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=int, default=5, help="print N sample chunks (default 5; 0 to skip)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for sample selection")
    args = parser.parse_args()

    chunks = chunk_all()
    if not chunks:
        return 1

    write_jsonl(chunks, CHUNKS_PATH)

    char_counts = [c["char_count"] for c in chunks]
    print(
        f"\nWrote {len(chunks)} chunks to {CHUNKS_PATH}"
        f" (min={min(char_counts)}, avg={sum(char_counts) // len(chunks)}, max={max(char_counts)} chars)"
    )

    if args.inspect > 0:
        print_samples(chunks, args.inspect, args.seed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
