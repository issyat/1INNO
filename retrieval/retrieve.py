"""
Hybrid retrieval: BM25 (sparse) + ChromaDB dense + cross-encoder re-ranking.

Why this stack:
  - BM25         catches exact clinical terms the intern uses ("hasn't eaten", "crying")
  - Dense        catches semantic neighbours even when wording differs
  - Re-ranker    scores every candidate pair (query, chunk) with a cross-encoder
                 and keeps only the genuinely relevant ones — this is the real
                 quality gate, not a cosine-similarity threshold.

All three components are CPU-friendly and load once at startup.
"""

import re
from pathlib import Path

import chromadb

PROJECT_ROOT    = Path(__file__).parent.parent
CHROMA_PATH     = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documents"

# ── singleton caches ────────────────────────────────────────────────────────
_collection = None   # ChromaDB collection
_bm25       = None   # BM25Okapi index
_bm25_docs  = None   # parallel list of chunk dicts (same order as BM25 index)
_reranker   = None   # CrossEncoder model


# ── bibliography filter ──────────────────────────────────────────────────────
_BIBLIO_SIGNALS = [
    re.compile(r"\b\d{4};\d+[:\(]\d+"),
    re.compile(r"doi:\s*10\.\d+", re.IGNORECASE),
    re.compile(r"\bet al\b\.", re.IGNORECASE),
    re.compile(r"©\s*Author", re.IGNORECASE),
    re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+[A-Z]{2,3}\b"),
    re.compile(r"To cite:", re.IGNORECASE),
    re.compile(r"[A-Z][a-z]+,\d+\s+[A-Z][a-z]"),
    re.compile(r"\b\d{4}\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\s*[;,]\s*\d+\s*[\(:]"),
    re.compile(r"\b(ISSN|ISBN|PMID|PMC\d{4,})\b", re.IGNORECASE),
    re.compile(r"Abbreviations?\s*\n", re.IGNORECASE),
]


def _is_bibliography(text: str) -> bool:
    return sum(1 for p in _BIBLIO_SIGNALS if p.search(text)) >= 2


# ── conversational query cleaning ────────────────────────────────────────────
_CONV_PREFIX = re.compile(
    r"^(my patient (just )?(told me|said|mentioned|reported|admitted|disclosed|explained)|"
    r"i (have|had|just saw|just had|am seeing|am working with) a patient (who|that)|"
    r"a patient (just |of mine )?(told me|said|came in|presented)|"
    r"during (the |my |a )?(session|consultation|appointment),?\s*|"
    r"(the |my )?(client|patient) (just |recently )?(said|told me|reported|mentioned|disclosed),?\s*)",
    re.IGNORECASE,
)


def _clinical_query(query: str) -> str:
    cleaned = _CONV_PREFIX.sub("", query).strip()
    cleaned = re.sub(r"^[,\s]+", "", cleaned).strip()
    cleaned = re.sub(r"^they\b", "patient", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^he\b",   "patient", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^she\b",  "patient", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btheir\b", "patient's", cleaned, flags=re.IGNORECASE)
    if cleaned and not re.match(r"^(patient|client|individual|person)", cleaned, re.IGNORECASE):
        cleaned = "patient " + cleaned
    return cleaned.strip() if cleaned else query


# ── component loaders ────────────────────────────────────────────────────────
def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_bm25():
    """Build BM25 index from all non-bibliography chunks in ChromaDB."""
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25, _bm25_docs

    from rank_bm25 import BM25Okapi

    collection = _get_collection()
    all_data = collection.get(include=["documents", "metadatas"])

    _bm25_docs = []
    tokenised  = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        if _is_bibliography(doc):
            continue
        _bm25_docs.append({
            "text":    doc,
            "source":  meta.get("document_title", "Unknown"),
            "section": meta.get("section", ""),
            "page":    meta.get("page", 0),
        })
        tokenised.append(doc.lower().split())

    _bm25 = BM25Okapi(tokenised)
    return _bm25, _bm25_docs


def _get_reranker():
    """Load cross-encoder once and cache it."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        # Small, fast, strong MS-MARCO cross-encoder (~24 MB, CPU-only)
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    return _reranker


# ── reciprocal rank fusion ───────────────────────────────────────────────────
def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Combine multiple ranked lists into one using Reciprocal Rank Fusion.
    Returns item IDs sorted by fused score (best first).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── main retrieve function ───────────────────────────────────────────────────
def retrieve(query: str, k: int = 4) -> list[dict]:
    """
    Hybrid retrieval pipeline:
      1. Clean the query (strip conversational framing)
      2. BM25 sparse search  (exact terms)
      3. ChromaDB dense search (semantic similarity)
      4. Merge with Reciprocal Rank Fusion
      5. Cross-encoder re-ranking → final top-k

    Args:
        query: Raw user question or clinical scenario.
        k:     Number of chunks to return.

    Returns:
        List of dicts: text, source, section, page, relevance_score.
    """
    import numpy as np

    cleaned  = _clinical_query(query)
    POOL     = max(k * 5, 20)   # over-fetch for better RRF + reranker input

    # ── 1. Dense retrieval ───────────────────────────────────────────────────
    collection = _get_collection()
    dense_results = collection.query(
        query_texts=[cleaned],
        n_results=POOL,
        include=["documents", "metadatas", "distances"],
    )

    dense_pool: dict[str, dict] = {}
    dense_ranking: list[str]    = []

    for doc, meta, dist in zip(
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
    ):
        if _is_bibliography(doc):
            continue
        cid = meta.get("chunk_id") or f"{meta.get('document_title','')}_{meta.get('page',0)}"
        dense_pool[cid] = {
            "text":            doc,
            "source":          meta.get("document_title", "Unknown"),
            "section":         meta.get("section", ""),
            "page":            meta.get("page", 0),
            "relevance_score": round(1 - dist, 3),
        }
        dense_ranking.append(cid)

    # ── 2. BM25 sparse retrieval ─────────────────────────────────────────────
    bm25, bm25_docs = _get_bm25()
    tokens     = cleaned.lower().split()
    raw_scores = bm25.get_scores(tokens)
    top_idx    = np.argsort(raw_scores)[::-1][:POOL]

    bm25_pool: dict[str, dict] = {}
    bm25_ranking: list[str]    = []

    for rank, idx in enumerate(top_idx):
        if raw_scores[idx] <= 0:
            break
        info = bm25_docs[idx]
        cid  = f"bm25_{idx}"
        bm25_pool[cid] = {**info, "relevance_score": round(float(raw_scores[idx]), 3)}
        bm25_ranking.append(cid)

    # ── 3. Reciprocal Rank Fusion ────────────────────────────────────────────
    fused_ids = _rrf([dense_ranking, bm25_ranking])

    # Build unified pool (dense takes priority for metadata; BM25 fills gaps)
    unified: dict[str, dict] = {}
    for cid in fused_ids:
        if cid in dense_pool:
            unified[cid] = dense_pool[cid]
        elif cid in bm25_pool:
            unified[cid] = bm25_pool[cid]

    candidates = [unified[cid] for cid in fused_ids if cid in unified][:POOL]

    if not candidates:
        return []

    # ── 4. Cross-encoder re-ranking ──────────────────────────────────────────
    reranker = _get_reranker()
    pairs    = [(query, c["text"]) for c in candidates]
    scores   = reranker.predict(pairs, show_progress_bar=False)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in ranked[:k]:
        chunk = dict(chunk)
        chunk["relevance_score"] = round(float(score), 3)
        results.append(chunk)

    return results


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    query  = sys.argv[1] if len(sys.argv) > 1 else "What are the symptoms of major depression?"
    chunks = retrieve(query, k=4)
    print(f"Query: {query}\n")
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] score={c['relevance_score']}  {c['source']} p.{c['page']}")
        print(f"     {c['text'][:120]}...\n")
