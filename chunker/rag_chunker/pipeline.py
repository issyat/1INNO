"""
Main pipeline orchestrator for the RAG chunking system.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from .models import Chunk, Section, DocumentInfo
from .parser import PDFParser
from .chunker import RecursiveChunker, ChunkingConfig
from .spacy_processor import SpacyProcessor
from .semantic import SemanticGrouper, SemanticConfig


@dataclass
class PipelineConfig:
    """
    Complete configuration for the chunking pipeline.
    """
    # Chunking settings
    max_tokens: int = 512
    overlap_tokens: int = 75
    overlap_sentences: int = 2
    prefer_sentence_boundaries: bool = True
    extract_entities: bool = True
    min_chunk_tokens: int = 50
    
    # Parser settings
    detect_headings: bool = True
    
    # Semantic settings
    enable_semantic_fallback: bool = False
    semantic_coherence_threshold: float = 0.6
    semantic_model: str = "all-MiniLM-L6-v2"
    
    # spaCy settings
    spacy_model: str = "en_core_web_sm"


class ChunkingPipeline:
    """
    Orchestrates the complete document chunking pipeline.
    
    Pipeline stages:
    1. PDF Parsing - Extract structured content
    2. Section Processing - Process each section
    3. Sentence Segmentation - spaCy-based segmentation
    4. Chunking - Recursive chunking with overlap
    5. (Optional) Semantic Fallback - For low-coherence sections
    6. Output - Return chunks ready for embedding
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration.
        """
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.parser = PDFParser(detect_headings=self.config.detect_headings)
        
        self.spacy = SpacyProcessor(model_name=self.config.spacy_model)
        
        chunking_config = ChunkingConfig(
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
            overlap_sentences=self.config.overlap_sentences,
            prefer_sentence_boundaries=self.config.prefer_sentence_boundaries,
            extract_entities=self.config.extract_entities,
            min_chunk_tokens=self.config.min_chunk_tokens
        )
        self.chunker = RecursiveChunker(
            config=chunking_config,
            spacy_processor=self.spacy
        )
        
        # Semantic grouper (lazy initialized)
        self._semantic_grouper = None
    
    @property
    def semantic_grouper(self) -> SemanticGrouper:
        """Lazy initialization of semantic grouper."""
        if self._semantic_grouper is None:
            semantic_config = SemanticConfig(
                model_name=self.config.semantic_model,
                coherence_threshold=self.config.semantic_coherence_threshold,
                max_tokens=self.config.max_tokens
            )
            self._semantic_grouper = SemanticGrouper(config=semantic_config)
            self._semantic_grouper.enabled = self.config.enable_semantic_fallback
        return self._semantic_grouper
    
    def process(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF document and return chunks.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of chunk dictionaries ready for vector DB.
        """
        chunks = self.process_to_chunks(pdf_path)
        return [chunk.to_dict() for chunk in chunks]
    
    def process_to_chunks(self, pdf_path: str) -> List[Chunk]:
        """
        Process a PDF document and return Chunk objects.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of Chunk objects.
        """
        # Stage 1: Parse PDF
        sections, doc_info = self.parser.parse(pdf_path)
        
        if not sections:
            return []
        
        # Stage 2-5: Process sections
        all_chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self._process_section(
                section, doc_info, chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return all_chunks
    
    def _process_section(self, section: Section,
                        doc_info: DocumentInfo,
                        start_index: int) -> List[Chunk]:
        """
        Process a single section through the pipeline.
        
        Decides whether to use regular chunking or semantic fallback.
        """
        text = section.full_text
        if not text.strip():
            return []
        
        # Check if semantic fallback should be used
        if self.config.enable_semantic_fallback:
            sentences = self.spacy.segment_sentences(
                text, extract_entities=self.config.extract_entities
            )
            
            if self.semantic_grouper.should_use_semantic(sentences):
                # Use semantic grouping
                return self.semantic_grouper.group_semantically(
                    sentences, section, doc_info, start_index
                )
        
        # Regular chunking
        return self.chunker.chunk_section(section, doc_info, start_index)
    
    def get_document_info(self, pdf_path: str) -> DocumentInfo:
        """
        Get document information without full processing.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            DocumentInfo object.
        """
        _, doc_info = self.parser.parse(pdf_path)
        return doc_info
    
    def get_sections(self, pdf_path: str) -> List[Section]:
        """
        Get document sections without chunking.
        
        Useful for inspection/debugging.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of Section objects.
        """
        sections, _ = self.parser.parse(pdf_path)
        return sections


def chunk_document(pdf_path: str, 
                  config: Optional[PipelineConfig] = None,
                  **kwargs) -> List[Dict[str, Any]]:
    """
    Main entry point: Chunk a PDF document.
    
    This is the primary function interface for the module.
    
    Args:
        pdf_path: Path to PDF file.
        config: Pipeline configuration. If not provided, defaults are used.
        **kwargs: Override specific config options.
        
    Returns:
        List of chunk dictionaries with structure:
        [
            {
                "text": str,
                "metadata": {
                    "chunk_id": str,
                    "document_id": str,
                    "section": str,
                    "page": int,
                    "chunk_index": int,
                    "entities": optional list,
                    ...
                }
            },
            ...
        ]
    
    Example:
        >>> chunks = chunk_document("report.pdf")
        >>> for chunk in chunks:
        ...     print(f"Section: {chunk['metadata']['section']}")
        ...     print(f"Text: {chunk['text'][:100]}...")
        
        >>> # With custom config
        >>> chunks = chunk_document(
        ...     "report.pdf",
        ...     max_tokens=256,
        ...     enable_semantic_fallback=True
        ... )
    """
    # Build config
    if config is None:
        config = PipelineConfig(**kwargs)
    elif kwargs:
        # Apply overrides to existing config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    pipeline = ChunkingPipeline(config=config)
    return pipeline.process(pdf_path)


# Convenience re-export
__all__ = [
    "chunk_document",
    "ChunkingPipeline", 
    "PipelineConfig",
    "ChunkingConfig"
]
