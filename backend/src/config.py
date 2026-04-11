"""
Configuration for the backend API.
Paths are resolved relative to the project root (two levels above backend/src/).
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class BackendConfig:
    # ── ChromaDB (must match retrieval/retrieve.py) ──────────────────────────
    CHROMA_PATH     = str(_PROJECT_ROOT / "chroma_db")
    COLLECTION_NAME = "documents"

    # ── Embedding model (must match the model used when chunks were stored) ──
    EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

    # ── Chunking (kept for legacy load_data / vector_store modules) ──────────
    CHUNK_SIZE    = 500
    CHUNK_OVERLAP = 50

    # ── Search defaults ───────────────────────────────────────────────────────
    DEFAULT_K_RESULTS = 4

    # ── Data path (legacy) ───────────────────────────────────────────────────
    DATA_PATH  = str(_PROJECT_ROOT / "data")
    BATCH_SIZE = 100


config = BackendConfig()
