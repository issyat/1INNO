"""
RAG Document Chunking Pipeline

A production-ready document ingestion and chunking module for RAG systems.
Combines structure-aware chunking, spaCy-based sentence segmentation,
recursive chunking with overlap, and optional semantic fallback.
"""

from .pipeline import chunk_document, ChunkingPipeline, PipelineConfig
from .chunker import ChunkingConfig
from .models import Chunk, Section, ChunkMetadata
from .parser import PDFParser
from .chunker import RecursiveChunker
from .spacy_processor import SpacyProcessor
from .semantic import SemanticGrouper

__version__ = "1.0.0"

__all__ = [
    "chunk_document",
    "ChunkingPipeline",
    "PipelineConfig",
    "ChunkingConfig",
    "Chunk",
    "Section",
    "ChunkMetadata",
    "PDFParser",
    "RecursiveChunker",
    "SpacyProcessor",
    "SemanticGrouper",
]
