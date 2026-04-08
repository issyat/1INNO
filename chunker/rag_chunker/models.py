"""
Data models for the RAG chunking pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class BlockType(Enum):
    """Types of content blocks extracted from PDF."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    UNKNOWN = "unknown"
    NOISE = "noise"  # Running headers, watermarks, page identifiers — skip entirely


@dataclass
class TextBlock:
    """A single block of text extracted from PDF."""
    text: str
    block_type: BlockType
    page: int
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1)
    font_size: Optional[float] = None
    is_bold: bool = False
    
    def __post_init__(self):
        self.text = self.text.strip()


@dataclass
class Section:
    """
    A structural section of the document.
    Groups content under a heading or page boundary.
    """
    title: str
    content: List[str]  # List of paragraphs
    page: int
    section_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_section_id: Optional[str] = None
    level: int = 1  # Heading level (1 = top-level)
    
    @property
    def full_text(self) -> str:
        """Combine all content into single text."""
        return "\n\n".join(self.content)
    
    def __repr__(self):
        content_preview = self.full_text[:50] + "..." if len(self.full_text) > 50 else self.full_text
        return f"Section(title='{self.title}', page={self.page}, content='{content_preview}')"


@dataclass
class ChunkMetadata:
    """Metadata attached to each chunk."""
    chunk_id: str
    document_id: str
    section: str
    page: int
    chunk_index: int
    entities: Optional[List[Dict[str, str]]] = None
    parent_section_id: Optional[str] = None
    document_title: Optional[str] = None
    token_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }
        if self.entities is not None:
            result["entities"] = self.entities
        if self.parent_section_id is not None:
            result["parent_section_id"] = self.parent_section_id
        if self.document_title is not None:
            result["document_title"] = self.document_title
        if self.token_count is not None:
            result["token_count"] = self.token_count
        return result


@dataclass
class Chunk:
    """
    A chunk of text ready for embedding and vector storage.
    """
    text: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for vector DB insertion."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict()
        }
    
    def __repr__(self):
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk(text='{text_preview}', section='{self.metadata.section}', page={self.metadata.page})"


@dataclass
class Sentence:
    """A sentence with its metadata."""
    text: str
    start_char: int
    end_char: int
    entities: List[Dict[str, str]] = field(default_factory=list)
    token_count: int = 0
    embedding: Optional[List[float]] = None
    
    def __repr__(self):
        text_preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        return f"Sentence('{text_preview}')"


@dataclass 
class DocumentInfo:
    """Information about the source document."""
    document_id: str
    title: Optional[str] = None
    path: Optional[str] = None
    page_count: int = 0
    
    def __repr__(self):
        return f"DocumentInfo(id='{self.document_id}', title='{self.title}', pages={self.page_count})"
