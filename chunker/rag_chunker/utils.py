"""
Utility functions for the RAG chunking pipeline.
"""

import hashlib
import re
from typing import List, Optional
import uuid


# Lazy-loaded tiktoken encoder
_encoder = None


def get_token_encoder():
    """Get or create tiktoken encoder (lazy loading)."""
    global _encoder
    if _encoder is None:
        import tiktoken
        # Use cl100k_base (GPT-4/GPT-3.5-turbo compatible)
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken (GPT-compatible).
    
    Args:
        text: Input text to count tokens for.
        
    Returns:
        Number of tokens in the text.
    """
    if not text:
        return 0
    encoder = get_token_encoder()
    return len(encoder.encode(text))


def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Split text into chunks of at most max_tokens.
    Used as fallback when sentence splitting isn't possible.
    
    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        
    Returns:
        List of text chunks.
    """
    if not text:
        return []
    
    encoder = get_token_encoder()
    tokens = encoder.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def generate_chunk_id(document_id: str, section: str, index: int) -> str:
    """
    Generate a deterministic chunk ID.
    Same input always produces same output for reproducibility.
    
    Args:
        document_id: Document identifier.
        section: Section name/title.
        index: Chunk index within section.
        
    Returns:
        Deterministic chunk ID string.
    """
    content = f"{document_id}:{section}:{index}"
    hash_digest = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"chunk_{hash_digest}"


def generate_document_id(pdf_path: str) -> str:
    """
    Generate a document ID from file path.
    
    Args:
        pdf_path: Path to PDF file.
        
    Returns:
        Document ID string.
    """
    # Use path hash for determinism
    hash_digest = hashlib.sha256(pdf_path.encode()).hexdigest()[:12]
    return f"doc_{hash_digest}"


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text to clean.
        
    Returns:
        Cleaned text.
    """
    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters except newlines
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def is_heading(text: str, font_size: Optional[float] = None, 
               avg_font_size: Optional[float] = None, is_bold: bool = False) -> bool:
    """
    Heuristically determine if text is a heading.
    
    Args:
        text: Text to check.
        font_size: Font size of the text.
        avg_font_size: Average font size in document.
        is_bold: Whether text is bold.
        
    Returns:
        True if likely a heading.
    """
    if not text:
        return False
    
    text = text.strip()
    
    # Short text that ends without punctuation
    if len(text) < 100 and not text.endswith(('.', ',', ';', ':')):
        # Check for heading patterns
        heading_patterns = [
            r'^(?:Chapter|Section|Part)\s+\d+',  # Chapter 1, Section 2
            r'^\d+\.?\s+[A-Z]',  # 1. Introduction, 1 Overview
            r'^[IVXLCDM]+\.\s+',  # Roman numerals
            r'^[A-Z][^.]*$',  # Single line, starts with capital, no period
        ]
        for pattern in heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
    
    # Font-based detection
    if font_size and avg_font_size:
        if font_size > avg_font_size * 1.2:  # 20% larger than average
            return True
    
    # Bold short text is often a heading
    if is_bold and len(text) < 100:
        return True
    
    return False


def estimate_heading_level(text: str, font_size: Optional[float] = None,
                          max_font_size: Optional[float] = None) -> int:
    """
    Estimate heading level (1-6, like HTML).
    
    Args:
        text: Heading text.
        font_size: Font size of heading.
        max_font_size: Maximum font size in document.
        
    Returns:
        Heading level (1 = most important).
    """
    # Check for explicit numbering
    if re.match(r'^(?:Chapter|Part)\s+', text, re.IGNORECASE):
        return 1
    if re.match(r'^\d+\.\s+', text):
        return 2
    if re.match(r'^\d+\.\d+\s+', text):
        return 3
    if re.match(r'^\d+\.\d+\.\d+\s+', text):
        return 4
    
    # Font-size based
    if font_size and max_font_size:
        ratio = font_size / max_font_size
        if ratio > 0.9:
            return 1
        elif ratio > 0.75:
            return 2
        elif ratio > 0.6:
            return 3
        else:
            return 4
    
    return 2  # Default to level 2


def merge_hyphenated_words(text: str) -> str:
    """
    Merge words that were hyphenated at line breaks.
    
    Args:
        text: Text with potential hyphenation.
        
    Returns:
        Text with hyphenation fixed.
    """
    # Pattern: word- at end of line followed by continuation
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)


def is_noise_block(text: str) -> bool:
    """
    Detect running headers, watermarks, and metadata fragments from academic PDFs.

    These blocks are not meaningful content — they pollute heading detection and
    produce duplicate / tiny chunks.  Patterns detected:

    - NIH / HHS manuscript watermarks  ("NIH-PA Author Manuscript" repeated)
    - Journal page headers             ("Psychiatry 2006[SEPTEMBER] 62")
    - Page-of-N identifiers            ("Healthcare2025,13,1247 2of26")
    - DOI lines                        ("DOI:10.3399/...")
    - CC-BY licence lines              ("Attribution(CCBY)license")
    - Published-date metadata          ("Published:8September2025")
    - Short article identifiers        ("bs15091220", "healthcare13202601")
    - Author bylines starting with by  ("byALEXANDER L. CHAPMAN,PhD")
    """
    if not text:
        return False

    t = text.strip()

    # ── Repeating watermark phrases ──────────────────────────────────────────
    if re.search(r'(?:NIH-PA\s+Author\s+Manuscript\s*){2,}', t, re.IGNORECASE):
        return True
    if re.search(r'(?:Author\s+Manuscript\s*){3,}', t, re.IGNORECASE):
        return True
    if re.match(r'^(?:NIH|HHS)\s+(?:Public\s+Access|PA)\b', t, re.IGNORECASE):
        return True

    # ── Page-of-N (e.g. "2of26", "3 of 25") ────────────────────────────────
    if re.search(r'\b\d+\s*of\s*\d+\b', t) and len(t) < 60:
        return True

    # ── DOI lines ───────────────────────────────────────────────────────────
    if re.match(r'^DOI\s*:\s*10\.\d{4,}', t, re.IGNORECASE):
        return True

    # ── Licence / attribution ────────────────────────────────────────────────
    if re.search(r'Attribution\s*\(?CC\s*[-–]?\s*BY\)?', t, re.IGNORECASE):
        return True

    # ── Published-date metadata ──────────────────────────────────────────────
    if re.match(r'^Published\s*:\s*\d', t, re.IGNORECASE):
        return True

    # ── Journal + year + [MONTH] + page  ("Psychiatry 2006[SEPTEMBER] 62")
    if re.match(r'^[\[\(]?[A-Z][A-Za-z\s]+[\]\)]?\s*\d{4}\s*[\[\(][A-Z]+[\]\)]\s+\d+$', t):
        return True
    if re.match(r'^[A-Za-z\s]+\d{4}\s*[\[\(][A-Z]+[\]\)]\s+\d+$', t):
        return True

    # ── Short article / journal identifiers (no spaces, all lowercase+digits)
    if re.match(r'^[a-z]{2,15}\d{5,}$', t):
        return True

    # ── "byAUTHOR NAME" bylines ─────────────────────────────────────────────
    if re.match(r'^by[A-Z]', t):
        return True

    return False


def is_table_content(text: str) -> bool:
    """
    Detect if text appears to be table content.
    
    Args:
        text: Text to check.
        
    Returns:
        True if likely table content.
    """
    if not text:
        return False
    
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    # Check for consistent column separators
    tab_lines = sum(1 for line in lines if '\t' in line)
    pipe_lines = sum(1 for line in lines if '|' in line)
    
    if tab_lines > len(lines) * 0.5:
        return True
    if pipe_lines > len(lines) * 0.5:
        return True
    
    # Check for grid-like structure (many short segments)
    segments = [len(line.split()) for line in lines]
    if segments and max(segments) > 5 and min(segments) > 2:
        variance = sum((s - sum(segments)/len(segments))**2 for s in segments) / len(segments)
        if variance < 4:  # Low variance = consistent columns
            return True
    
    return False
