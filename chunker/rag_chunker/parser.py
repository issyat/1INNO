"""
PDF Parser module for extracting structured content from PDF documents.
Uses PyMuPDF (fitz) for extraction.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

from .models import Section, TextBlock, BlockType, DocumentInfo
from .utils import (
    clean_text, is_heading, estimate_heading_level,
    merge_hyphenated_words, is_table_content, is_noise_block
)


class PDFParser:
    """
    Extracts structured content from PDF documents.
    
    Preserves document structure including:
    - Headings and titles
    - Paragraphs grouped by sections
    - Page numbers
    - Table detection
    """
    
    def __init__(self, detect_headings: bool = True):
        """
        Initialize PDF parser.
        
        Args:
            detect_headings: Whether to detect headings heuristically.
        """
        self.detect_headings = detect_headings
        self._fitz = None
    
    def _get_fitz(self):
        """Lazy load fitz (PyMuPDF)."""
        if self._fitz is None:
            import fitz
            self._fitz = fitz
        return self._fitz
    
    def parse(self, pdf_path: str) -> Tuple[List[Section], DocumentInfo]:
        """
        Parse PDF and extract structured sections.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            Tuple of (list of sections, document info).
        """
        fitz = self._get_fitz()
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        try:
            # Get document info
            doc_info = self._extract_doc_info(doc, pdf_path)
            
            # Extract all text blocks with metadata
            all_blocks = self._extract_blocks(doc)
            
            # Calculate document-wide font statistics
            font_stats = self._compute_font_stats(all_blocks)
            
            # Classify blocks as headings or paragraphs
            if self.detect_headings:
                all_blocks = self._classify_blocks(all_blocks, font_stats)
            
            # Group blocks into sections
            sections = self._group_into_sections(all_blocks, doc_info)
            
            return sections, doc_info
            
        finally:
            doc.close()
    
    def _extract_doc_info(self, doc, pdf_path: str) -> DocumentInfo:
        """Extract document metadata."""
        from .utils import generate_document_id
        
        metadata = doc.metadata
        title = metadata.get('title') if metadata else None
        
        # Fallback to filename if no title
        if not title:
            title = Path(pdf_path).stem
        
        return DocumentInfo(
            document_id=generate_document_id(pdf_path),
            title=title,
            path=pdf_path,
            page_count=len(doc)
        )
    
    def _extract_blocks(self, doc) -> List[TextBlock]:
        """Extract all text blocks from document."""
        blocks = []
        
        for page_num, page in enumerate(doc):
            # Get text blocks with positioning
            page_dict = page.get_text("dict", flags=11)  # Include font info
            
            for block in page_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    text_block = self._process_text_block(block, page_num + 1)
                    if text_block and text_block.text:
                        blocks.append(text_block)
                elif block.get("type") == 1:  # Image block
                    # Skip images but could extract captions
                    pass
        
        return blocks
    
    def _process_text_block(self, block: dict, page: int) -> Optional[TextBlock]:
        """Process a single text block from PyMuPDF."""
        lines = block.get("lines", [])
        if not lines:
            return None
        
        text_parts = []
        font_sizes = []
        is_bold_count = 0
        total_spans = 0
        
        for line in lines:
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                if span_text:
                    line_text += span_text
                    font_sizes.append(span.get("size", 12))
                    total_spans += 1
                    
                    # Check if bold
                    flags = span.get("flags", 0)
                    if flags & 16:  # Bold flag
                        is_bold_count += 1
            
            if line_text.strip():
                text_parts.append(line_text)
        
        if not text_parts:
            return None
        
        # Join lines, handling hyphenation
        text = "\n".join(text_parts)
        text = merge_hyphenated_words(text)
        text = clean_text(text)
        
        if not text:
            return None
        
        # Calculate average font size
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
        is_bold = is_bold_count > total_spans * 0.5 if total_spans > 0 else False
        
        # Get bounding box
        bbox = (block.get("bbox", (0, 0, 0, 0)))
        
        return TextBlock(
            text=text,
            block_type=BlockType.UNKNOWN,
            page=page,
            bbox=bbox,
            font_size=avg_font_size,
            is_bold=is_bold
        )
    
    def _compute_font_stats(self, blocks: List[TextBlock]) -> dict:
        """Compute font size statistics across document."""
        if not blocks:
            return {"avg": 12, "max": 12, "min": 12}
        
        font_sizes = [b.font_size for b in blocks if b.font_size]
        
        if not font_sizes:
            return {"avg": 12, "max": 12, "min": 12}
        
        return {
            "avg": sum(font_sizes) / len(font_sizes),
            "max": max(font_sizes),
            "min": min(font_sizes)
        }
    
    def _classify_blocks(self, blocks: List[TextBlock],
                        font_stats: dict) -> List[TextBlock]:
        """Classify blocks as headings, paragraphs, tables, or noise."""
        for block in blocks:
            # Noise first — running headers, watermarks, page identifiers
            if is_noise_block(block.text):
                block.block_type = BlockType.NOISE
                continue

            # Check for table content
            if is_table_content(block.text):
                block.block_type = BlockType.TABLE
                continue

            # Check for heading
            if is_heading(
                block.text,
                font_size=block.font_size,
                avg_font_size=font_stats["avg"],
                is_bold=block.is_bold
            ):
                block.block_type = BlockType.HEADING
            else:
                block.block_type = BlockType.PARAGRAPH

        return blocks
    
    def _group_into_sections(self, blocks: List[TextBlock], 
                            doc_info: DocumentInfo) -> List[Section]:
        """Group text blocks into logical sections."""
        if not blocks:
            return []
        
        sections = []
        current_section: Optional[Section] = None
        section_stack = []  # For nested sections
        
        # Check if document has any headings
        has_headings = any(b.block_type == BlockType.HEADING for b in blocks)
        
        if not has_headings:
            # Fallback: group by pages
            return self._group_by_pages(blocks, doc_info)
        
        for block in blocks:
            # Skip noise blocks entirely — don't let them create sections or add content
            if block.block_type == BlockType.NOISE:
                continue

            if block.block_type == BlockType.HEADING:
                # Save current section if exists
                if current_section and current_section.content:
                    sections.append(current_section)
                
                # Start new section
                level = estimate_heading_level(
                    block.text,
                    font_size=block.font_size,
                    max_font_size=max(b.font_size for b in blocks if b.font_size)
                )
                
                # Determine parent section
                parent_id = None
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                if section_stack:
                    parent_id = section_stack[-1][1]
                
                current_section = Section(
                    title=block.text,
                    content=[],
                    page=block.page,
                    level=level,
                    parent_section_id=parent_id
                )
                
                section_stack.append((level, current_section.section_id))
                
            else:
                # Add content to current section
                if current_section is None:
                    # Content before first heading - create intro section
                    current_section = Section(
                        title="Introduction",
                        content=[],
                        page=block.page,
                        level=1
                    )
                
                if block.block_type == BlockType.TABLE:
                    # Mark tables specially
                    current_section.content.append(f"[TABLE]\n{block.text}\n[/TABLE]")
                else:
                    current_section.content.append(block.text)
        
        # Don't forget the last section
        if current_section and current_section.content:
            sections.append(current_section)
        
        return sections
    
    def _group_by_pages(self, blocks: List[TextBlock], 
                       doc_info: DocumentInfo) -> List[Section]:
        """Fallback: group content by pages when no headings detected."""
        pages = defaultdict(list)
        
        for block in blocks:
            if block.block_type == BlockType.NOISE:
                continue
            if block.block_type == BlockType.TABLE:
                pages[block.page].append(f"[TABLE]\n{block.text}\n[/TABLE]")
            else:
                pages[block.page].append(block.text)
        
        sections = []
        for page_num in sorted(pages.keys()):
            if pages[page_num]:
                sections.append(Section(
                    title=f"Page {page_num}",
                    content=pages[page_num],
                    page=page_num,
                    level=1
                ))
        
        return sections
    
    def extract_text_simple(self, pdf_path: str) -> str:
        """
        Simple text extraction without structure.
        Useful for quick previews or debugging.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            Full text content of PDF.
        """
        fitz = self._get_fitz()
        doc = fitz.open(pdf_path)
        
        try:
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            return "\n\n".join(text_parts)
        finally:
            doc.close()
