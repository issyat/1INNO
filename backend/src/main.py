"""
FastAPI backend for the clinical RAG assistant.

Wraps the hybrid retrieval pipeline (BM25 + ChromaDB dense + cross-encoder reranker
+ Gemma 4 E2B) exposed in retrieval/pipeline.py.

Run with:
    cd backend/src
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /query   — ask a clinical question
    GET  /health  — pipeline and collection status
"""

import sys
import time
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
    question: str = Field(..., min_length=3, description="Clinical question or scenario")
    k: int = Field(4, ge=1, le=20, description="Number of chunks to retrieve")
    debug: bool = Field(False, description="Include raw LLM output in response")


class SourceChunk(BaseModel):
    text: str
    source: str
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
@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    """
    Run the full RAG pipeline for a clinical question.

    The pipeline:
      1. Cleans conversational framing ("my patient said…")
      2. BM25 sparse + ChromaDB dense retrieval, merged via Reciprocal Rank Fusion
      3. Cross-encoder reranking
      4. Gemma 4 E2B grounded generation
      5. Returns structured JSON — answer is sourced from documents only
    """
    try:
        result = run_rag_pipeline(body.question, k=body.k, debug=body.debug)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result["answer"],
        found_in_documents=result["found_in_documents"],
        confidence=result["confidence"],
        sources_used=result.get("sources_used", []),
        sources=[SourceChunk(**s) for s in result.get("sources", [])],
        processing_time_ms=result["processing_time_ms"],
        raw_llm_output=result.get("raw_llm_output"),
    )


@app.get("/health")
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
            "collection": config.COLLECTION_NAME,
            "document_chunks": doc_count,
            "chroma_path": config.CHROMA_PATH,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.get("/")
def root():
    return {
        "service": "Clinical RAG Assistant",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query",
    }
