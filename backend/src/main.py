"""
FastAPI backend for the clinical RAG assistant.

Wraps the hybrid retrieval pipeline (BM25 + ChromaDB dense + cross-encoder reranker
+ Gemma 4 E2B) exposed in retrieval/pipeline.py.

Run with:
    cd backend/src
    uvicorn main:app --host 127.0.0.1 --port 8080 --reload

Endpoints:
    POST /api/v1/query      — ask a clinical question
    GET  /api/v1/health     — pipeline and collection status
    GET  /api/v1/documents  — list indexed documents with chunk counts
"""

import sys
from pathlib import Path

# Make retrieval/ importable regardless of where uvicorn is launched from
_RETRIEVAL_DIR = Path(__file__).parent.parent.parent / "retrieval"
if str(_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_RETRIEVAL_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import run_rag_pipeline

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical RAG Assistant API",
    description=(
        "Hybrid BM25 + semantic retrieval with cross-encoder reranking, "
        "grounded generation via Gemma 4 E2B."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    scenario: str = Field(..., min_length=3, description="Clinical question or scenario")
    user_type: str = Field("trainee", description="User type (unused, reserved for future)")
    k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    debug: bool = Field(False, description="Include raw LLM output in response")


class SourceChunk(BaseModel):
    document: str
    section: str
    page: int
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    found_in_documents: bool
    confidence: str
    sources_used: list[str]
    sources: list[SourceChunk]
    processing_time_ms: int
    raw_llm_output: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/api/v1/query", response_model=QueryResponse)
def query(body: QueryRequest):
    """
    Run the full RAG pipeline for a clinical scenario.

    Steps:
      1. Clean conversational framing ("my patient said…")
      2. BM25 sparse + ChromaDB dense retrieval merged via Reciprocal Rank Fusion
      3. Cross-encoder reranking
      4. Gemma 4 E2B grounded generation
      5. Returns structured JSON — answer sourced from documents only
    """
    try:
        result = run_rag_pipeline(body.scenario, k=body.k, debug=body.debug)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [
        SourceChunk(
            document=s.get("source", "Unknown"),
            section=s.get("section", ""),
            page=s.get("page", 0),
            relevance_score=s.get("relevance_score", 0.0),
        )
        for s in result.get("sources", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        found_in_documents=result["found_in_documents"],
        confidence=result["confidence"],
        sources_used=result.get("sources_used", []),
        sources=sources,
        processing_time_ms=result["processing_time_ms"],
        raw_llm_output=result.get("raw_llm_output"),
    )


@app.get("/api/v1/health")
def health():
    """
    Check that ChromaDB collection is reachable and report document count.
    The LLM and reranker are loaded lazily on first /query call.
    """
    try:
        import chromadb
        from config import config

        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = client.get_collection(config.COLLECTION_NAME)
        doc_count = collection.count()
        return {
            "status": "ok",
            "version": app.version,
            "collection": config.COLLECTION_NAME,
            "document_chunks": doc_count,
        }
    except Exception as exc:
        return {"status": "degraded", "reason": str(exc)}


@app.get("/api/v1/documents")
def documents():
    """
    List all unique documents indexed in ChromaDB with their section and chunk counts.
    """
    try:
        import chromadb
        from config import config

        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = client.get_collection(config.COLLECTION_NAME)
        all_meta = collection.get(include=["metadatas"])["metadatas"]

        # Aggregate by document title
        doc_map: dict[str, dict] = {}
        for meta in all_meta:
            title = meta.get("document_title") or meta.get("source", "Unknown")
            if title not in doc_map:
                doc_map[title] = {"name": title, "sections": set(), "chunks": 0}
            section = meta.get("section", "")
            if section:
                doc_map[title]["sections"].add(section)
            doc_map[title]["chunks"] += 1

        docs = [
            {
                "name": v["name"],
                "sections": len(v["sections"]),
                "chunks": v["chunks"],
            }
            for v in sorted(doc_map.values(), key=lambda x: x["name"])
        ]
        return {"documents": docs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/")
def root():
    return {
        "service": "Clinical RAG Assistant",
        "docs": "/docs",
        "health": "/api/v1/health",
        "query": "POST /api/v1/query",
        "documents": "/api/v1/documents",
    }
