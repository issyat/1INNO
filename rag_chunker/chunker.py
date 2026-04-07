"""
Recursive chunking module with overlap support.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import uuid

from .models import Section, Chunk, ChunkMetadata, Sentence, DocumentInfo
from .spacy_processor import SpacyProcessor
from .utils import count_tokens, split_by_tokens, generate_chunk_id


@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    max_tokens: int = 512
    overlap_tokens: int = 75  # Target 50-100 tokens overlap
    overlap_sentences: int = 2  # Preferred: overlap by sentences
    prefer_sentence_boundaries: bool = True
    extract_entities: bool = True
    min_chunk_tokens: int = 50  # Avoid very small chunks
    
    def __post_init__(self):
        # Ensure overlap doesn't exceed chunk size
        if self.overlap_tokens >= self.max_tokens // 2:
            self.overlap_tokens = self.max_tokens // 4


class RecursiveChunker:
    """
    Implements recursive chunking with overlap.
    
    Algorithm:
    1. Start with section text
    2. If section <= max_tokens → keep as single chunk
    3. If too large:
       - Split into sentences (via spaCy)
       - Aggregate sentences into chunks up to token limit
    4. If single sentence exceeds limit:
       - Fallback to token-based splitting
    5. Apply overlap between consecutive chunks
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None,
                 spacy_processor: Optional[SpacyProcessor] = None):
        """
        Initialize chunker.
        
        Args:
            config: Chunking configuration.
            spacy_processor: spaCy processor for sentence segmentation.
        """
        self.config = config or ChunkingConfig()
        self.spacy = spacy_processor or SpacyProcessor()
    
    def chunk_section(self, section: Section, 
                     document_info: DocumentInfo,
                     start_index: int = 0) -> List[Chunk]:
        """
        Chunk a single section into retrieval-ready chunks.
        
        Args:
            section: Section to chunk.
            document_info: Document metadata.
            start_index: Starting chunk index for this section.
            
        Returns:
            List of Chunk objects.
        """
        text = section.full_text
        if not text.strip():
            return []
        
        total_tokens = count_tokens(text)
        
        # Case 1: Section fits in one chunk
        if total_tokens <= self.config.max_tokens:
            return [self._create_chunk(
                text=text,
                section=section,
                document_info=document_info,
                chunk_index=start_index,
                entities=self._extract_entities(text) if self.config.extract_entities else None
            )]
        
        # Case 2: Need to split - use sentence-based chunking
        sentences = self.spacy.segment_sentences(
            text, 
            extract_entities=self.config.extract_entities
        )
        
        if not sentences:
            # Fallback: token-based splitting
            return self._chunk_by_tokens(text, section, document_info, start_index)
        
        # Aggregate sentences into chunks with overlap
        return self._aggregate_sentences(
            sentences, section, document_info, start_index
        )
    
    def _aggregate_sentences(self, sentences: List[Sentence],
                            section: Section,
                            document_info: DocumentInfo,
                            start_index: int) -> List[Chunk]:
        """
        Aggregate sentences into chunks respecting token limits.
        Implements overlap using sentence units.
        """
        if not sentences:
            return []
        
        chunks = []
        current_sentences: List[Sentence] = []
        current_tokens = 0
        chunk_index = start_index
        
        # Track sentences for overlap
        previous_overlap_sentences: List[Sentence] = []
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # Handle extremely long sentences
            if sentence.token_count > self.config.max_tokens:
                # Flush current chunk first
                if current_sentences:
                    chunk = self._create_chunk_from_sentences(
                        current_sentences, section, document_info, 
                        chunk_index, previous_overlap_sentences
                    )
                    chunks.append(chunk)
                    previous_overlap_sentences = current_sentences[-self.config.overlap_sentences:]
                    chunk_index += 1
                    current_sentences = []
                    current_tokens = 0
                
                # Split long sentence by tokens
                sub_chunks = self._split_long_sentence(
                    sentence, section, document_info, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                
                # No overlap from split sentence (would be incoherent)
                previous_overlap_sentences = []
                i += 1
                continue
            
            # Check if adding this sentence exceeds limit
            potential_tokens = current_tokens + sentence.token_count
            
            if potential_tokens <= self.config.max_tokens:
                # Add sentence to current chunk
                current_sentences.append(sentence)
                current_tokens = potential_tokens
                i += 1
            else:
                # Current chunk is full, create it
                if current_sentences:
                    chunk = self._create_chunk_from_sentences(
                        current_sentences, section, document_info,
                        chunk_index, previous_overlap_sentences
                    )
                    chunks.append(chunk)
                    
                    # Prepare overlap for next chunk
                    previous_overlap_sentences = current_sentences[-self.config.overlap_sentences:]
                    chunk_index += 1
                
                # Start new chunk (will include overlap)
                current_sentences = []
                current_tokens = 0
                # Don't increment i - process same sentence in next iteration
        
        # Don't forget the last chunk
        if current_sentences:
            chunk = self._create_chunk_from_sentences(
                current_sentences, section, document_info,
                chunk_index, previous_overlap_sentences
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_from_sentences(self, sentences: List[Sentence],
                                    section: Section,
                                    document_info: DocumentInfo,
                                    chunk_index: int,
                                    overlap_sentences: List[Sentence]) -> Chunk:
        """Create a chunk from a list of sentences, including overlap."""
        # Combine overlap sentences with current sentences
        all_sentences = overlap_sentences + sentences
        
        # Build text
        text = " ".join(s.text for s in all_sentences)
        
        # Merge entities from all sentences
        entities = []
        if self.config.extract_entities:
            seen_entities = set()
            for sent in all_sentences:
                for ent in sent.entities:
                    key = (ent["text"], ent["label"])
                    if key not in seen_entities:
                        seen_entities.add(key)
                        entities.append({"text": ent["text"], "label": ent["label"]})
        
        return self._create_chunk(
            text=text,
            section=section,
            document_info=document_info,
            chunk_index=chunk_index,
            entities=entities if entities else None
        )
    
    def _split_long_sentence(self, sentence: Sentence,
                            section: Section,
                            document_info: DocumentInfo,
                            start_index: int) -> List[Chunk]:
        """Split a sentence that exceeds max tokens using token-based splitting."""
        text_chunks = split_by_tokens(sentence.text, self.config.max_tokens)
        
        chunks = []
        for i, text in enumerate(text_chunks):
            chunk = self._create_chunk(
                text=text,
                section=section,
                document_info=document_info,
                chunk_index=start_index + i,
                entities=None  # Entities may be split, so exclude
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_tokens(self, text: str,
                        section: Section,
                        document_info: DocumentInfo,
                        start_index: int) -> List[Chunk]:
        """Fallback: chunk text purely by token count."""
        # Use overlapping token windows
        text_chunks = split_by_tokens(text, self.config.max_tokens)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_chunk(
                text=chunk_text,
                section=section,
                document_info=document_info,
                chunk_index=start_index + i,
                entities=self._extract_entities(chunk_text) if self.config.extract_entities else None
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, text: str,
                     section: Section,
                     document_info: DocumentInfo,
                     chunk_index: int,
                     entities: Optional[List[dict]] = None) -> Chunk:
        """Create a Chunk object with full metadata."""
        chunk_id = generate_chunk_id(
            document_info.document_id,
            section.title,
            chunk_index
        )
        
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            document_id=document_info.document_id,
            section=section.title,
            page=section.page,
            chunk_index=chunk_index,
            entities=entities,
            parent_section_id=section.parent_section_id,
            document_title=document_info.title,
            token_count=count_tokens(text)
        )
        
        return Chunk(text=text, metadata=metadata)
    
    def _extract_entities(self, text: str) -> List[dict]:
        """Extract entities using spaCy."""
        return self.spacy.extract_entities(text)
    
    def chunk_sections(self, sections: List[Section],
                      document_info: DocumentInfo) -> List[Chunk]:
        """
        Chunk multiple sections.
        
        Args:
            sections: List of sections to chunk.
            document_info: Document metadata.
            
        Returns:
            List of all chunks from all sections.
        """
        all_chunks = []
        chunk_index = 0
        
        for section in sections:
            section_chunks = self.chunk_section(
                section, document_info, start_index=chunk_index
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return all_chunks
