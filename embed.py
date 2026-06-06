"""Embed cleaned chunks into a local ChromaDB collection and provide a retrieval helper.

Indexing (one-time / re-runnable):
    python embed.py                # upsert every chunk from documents/chunks.jsonl
    python embed.py --rebuild      # drop the collection first, then re-index from scratch

Retrieval demo:
    python embed.py --query "is bike theft common around USC"          # default top-4
    python embed.py --query "safest streets at night" -k 6

Importable from downstream code (e.g. the Milestone 5 generation script):
    from embed import retrieve
    hits = retrieve("which complex has the worst elevators", k=4)
    for h in hits:
        print(h["metadata"]["source"], h["distance"], h["text"])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

REPO_ROOT = Path(__file__).parent
CHUNKS_PATH = REPO_ROOT / "documents" / "chunks.jsonl"
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "unofficial_guide"
EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4

# Built once at module load: the same SentenceTransformer instance is reused for indexing
# AND for query-time embedding (ChromaDB stores the embedding_function on the collection
# and calls it for both .upsert(documents=...) and .query(query_texts=...)).
_EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
# `_EMBED_FN` is a SentenceTransformer wrapper used by ChromaDB both at index
# time and at query time so embeddings are consistent. We rely on the
# `all-MiniLM-L6-v2` model (a semantic embedding model) for similarity search.


def _client() -> chromadb.PersistentClient:
    # PersistentClient writes vectors + an HNSW index to CHROMA_DIR so the DB survives
    # between Python invocations. (chromadb.Client() is in-memory and would force a
    # re-embed every run.) The directory is already gitignored.
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(client: chromadb.PersistentClient):
    # get_or_create_collection is idempotent — creates on first call, re-attaches after.
    # The metadata={"hnsw:space": "cosine"} flag is important: ChromaDB defaults to L2,
    # but all-MiniLM-L6-v2 was trained with cosine similarity, so cosine distance matches
    # the geometry the model actually learned. (Only takes effect on creation; for an
    # existing collection with the wrong metric, you need --rebuild.)
    # The `metadata={"hnsw:space": "cosine"}` flag sets the ANN index metric
    # to cosine distance (1 - cosine_similarity). This matches the geometry
    # expected by most sentence-transformer models. If you change the metric
    # (e.g. to L2) you should rebuild the collection with `--rebuild`.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"chunks file not found: {path} — run `python chunk.py` first"
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_index(rebuild: bool = False) -> int:
    client = _client()

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # collection didn't exist — fine

    collection = _collection(client)
    chunks = load_chunks()

    # Stable IDs: re-running the script overwrites in place instead of duplicating vectors.
    # Without stable IDs, re-indexing would grow the collection on each run.
    ids = [f"{c['source']}#{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    # ChromaDB metadata values must be str/int/float/bool — none of ours are nested or None,
    # so we can pass the chunk's metadata fields straight through.
    metadatas = [
        {
            "source": c["source"],
            "source_id": c["source_id"],
            "source_type": c["source_type"],
            "parent_title": c["parent_title"],
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]

    # upsert() inserts new ids and overwrites existing ones. add() would raise on the
    # second run for any duplicate id, which is exactly what we want to avoid here.
    # 500 is a comfortable batch — ChromaDB handles much larger, but smaller batches
    # give clearer progress output and keep peak memory bounded.
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i : i + BATCH],
            documents=documents[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )
        print(f"  upserted {min(i + BATCH, len(ids))}/{len(ids)} chunks")

    print(
        f"\ncollection '{COLLECTION_NAME}' now holds {collection.count()} vectors "
        f"(persisted to {CHROMA_DIR})"
    )
    return len(ids)


def retrieve(query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top-k chunks most relevant to `query`.

    Each hit is a dict with:
      - "text":      the chunk's document text
      - "metadata":  dict of source, source_type, parent_title, chunk_index, ...
      - "distance":  cosine distance (0 = identical, ~1 = orthogonal). Lower is better.
    """
    collection = _collection(_client())
    if collection.count() == 0:
        raise RuntimeError(
            f"collection '{COLLECTION_NAME}' is empty — run `python embed.py` first"
        )

    # query() embeds the query with the same model the collection was built with,
    # then runs an HNSW nearest-neighbor search. The return shape is parallel arrays
    # keyed by query_texts (a list), so for our single query we always index [0].
    # `collection.query(...)` first embeds the query using the same embedding
    # function used for indexing, then performs an HNSW nearest-neighbor
    # search over the stored vectors. The result contains parallel arrays for
    # documents, metadatas and distances keyed by query index; for a single
    # query we read index 0.
    #
    # Distances are cosine distances (0 = identical vectors, ~1 = orthogonal).
    res = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def _print_hits(query: str, hits: list[dict]) -> None:
    print(f"\nTop {len(hits)} chunks for: {query!r}\n")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        title = f' — "{m["parent_title"]}"' if m.get("parent_title") else ""
        print(
            f"--- #{i}  distance={h['distance']:.3f}  "
            f"{m['source']}#{m['chunk_index']}{title} ---"
        )
        text = h["text"]
        print(text if len(text) <= 500 else text[:500] + "…")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB index or run a retrieval test."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="run a retrieval test against the existing index instead of building",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="drop the collection before re-indexing (use after model/distance changes)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"top-k for --query (default {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    if args.query:
        hits = retrieve(args.query, k=args.k)
        _print_hits(args.query, hits)
        return 0

    build_index(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(main())
