"""
Quick test script for your own PDF files.

Usage:
    python test_my_pdf.py                      # Process all PDFs in input_pdfs/
    python test_my_pdf.py input_pdfs\\file.pdf  # Process specific file
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_chunker import chunk_document, ChunkingPipeline, PipelineConfig

# Default input directory
INPUT_DIR = Path(__file__).parent / "input_pdfs"


def test_pdf(pdf_path: str):
    """Test chunking on your PDF file."""
    
    print(f"\nProcessing: {pdf_path}\n")
    print("=" * 60)
    
    # Process the PDF
    chunks = chunk_document(pdf_path)
    
    print(f"Total chunks created: {len(chunks)}")
    
    # Token statistics
    tokens = [c['metadata']['token_count'] for c in chunks]
    if tokens:
        print(f"Token range: {min(tokens)} - {max(tokens)}")
        print(f"Average tokens/chunk: {sum(tokens)/len(tokens):.1f}")
    
    # Sections found
    sections = set(c['metadata']['section'] for c in chunks)
    print(f"\nSections found ({len(sections)}):")
    for section in sections:
        count = sum(1 for c in chunks if c['metadata']['section'] == section)
        print(f"  - {section} ({count} chunks)")
    
    # Show all chunks
    print("\n" + "=" * 60)
    print("ALL CHUNKS:")
    print("=" * 60)
    
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1}/{len(chunks)} ---")
        print(f"ID: {chunk['metadata']['chunk_id']}")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Tokens: {chunk['metadata']['token_count']}")
        
        entities = chunk['metadata'].get('entities', [])
        if entities:
            entity_texts = [e['text'] for e in entities[:5]]
            print(f"Entities: {entity_texts}")
        
        # Show text (truncated for readability)
        text = chunk['text']
        if len(text) > 300:
            print(f"Text:\n{text[:300]}...\n[{len(text)} chars total]")
        else:
            print(f"Text:\n{text}")
    
    # Save to JSON
    output_path = Path(pdf_path).stem + "_chunks.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved all chunks to: {output_path}")


def process_all_in_folder():
    """Process all PDFs in the input_pdfs folder."""
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in: {INPUT_DIR}")
        print("\nPlace your PDF files in the 'input_pdfs' folder and run again.")
        print("Or specify a file: python test_my_pdf.py path\\to\\file.pdf")
        return
    
    print(f"Found {len(pdf_files)} PDF(s) in input_pdfs/\n")
    
    for pdf_path in pdf_files:
        test_pdf(str(pdf_path))
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # No argument - process all PDFs in input_pdfs/
        process_all_in_folder()
    else:
        # Specific file provided
        pdf_path = sys.argv[1]
        
        if not Path(pdf_path).exists():
            print(f"Error: File not found: {pdf_path}")
            sys.exit(1)
        
        test_pdf(pdf_path)
