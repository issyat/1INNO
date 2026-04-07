# RAG Document Chunking Pipeline - Implementation Plan

## Overview
Build a production-ready document ingestion and chunking pipeline for RAG systems with structure-aware chunking, spaCy integration, recursive splitting, and optional semantic fallback.

## Project Structure
```
rag_chunker/
├── __init__.py
├── parser.py          # PDF parsing and structure extraction
├── chunker.py         # Core chunking logic with recursive splitting
├── spacy_processor.py # spaCy integration for sentence segmentation
├── semantic.py        # Optional semantic fallback module
├── models.py          # Data models/types
├── utils.py           # Utility functions (token counting, etc.)
└── pipeline.py        # Main pipeline orchestrator
tests/
├── __init__.py
├── test_parser.py
├── test_chunker.py
├── test_pipeline.py
└── fixtures/          # Test PDF files
requirements.txt
example_run.py
```

## Implementation Approach
1. Use PyMuPDF (fitz) for PDF parsing - fast and reliable
2. spaCy with en_core_web_sm for sentence segmentation
3. tiktoken for accurate token counting (GPT-compatible)
4. sentence-transformers for optional semantic fallback

## Modules to Implement

### 1. models.py - Data structures
- Section, Chunk, ChunkMetadata dataclasses
- Type definitions

### 2. utils.py - Utilities
- Token counting with tiktoken
- UUID generation for chunk IDs

### 3. parser.py - PDF Parser
- Extract text with structure (headings, paragraphs, pages)
- Handle tables as separate blocks
- Fallback for unstructured PDFs

### 4. spacy_processor.py - NLP Processing
- Sentence segmentation
- Entity extraction
- Lazy model loading

### 5. chunker.py - Core Chunking
- Structure-aware chunking
- Recursive splitting algorithm
- Overlap implementation
- Token-level fallback for long sentences

### 6. semantic.py - Semantic Fallback
- Sentence embedding computation
- Coherence detection
- Semantic clustering/grouping

### 7. pipeline.py - Orchestrator
- Main `chunk_document()` function
- Configuration options
- Pipeline coordination

## Edge Cases
- No headings → paragraph-based grouping
- Long sentences → token-level split
- Tables → separate blocks
- Empty/malformed blocks → skip or handle gracefully
