"""
Example script demonstrating the RAG chunking pipeline.

This script shows how to use the chunking module with:
1. Default configuration
2. Custom configuration
3. With semantic fallback enabled

Run this script after installing dependencies:
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
"""

import json
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_chunker import (
    chunk_document,
    ChunkingPipeline,
    PipelineConfig,
    PDFParser,
)


def create_sample_pdf():
    """
    Create a sample PDF for demonstration.
    Returns path to the created PDF.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not installed. Please run: pip install PyMuPDF")
        return None
    
    pdf_path = project_root / "sample_document.pdf"
    
    # Create a new PDF document
    doc = fitz.open()
    
    # Page 1: Title and Introduction
    page1 = doc.new_page()
    
    # Title
    page1.insert_text(
        (72, 72),
        "Artificial Intelligence in Healthcare",
        fontsize=24,
        fontname="helv"
    )
    
    # Section: Executive Summary
    page1.insert_text((72, 120), "Executive Summary", fontsize=16, fontname="helv")
    
    summary_text = """
Artificial Intelligence (AI) is transforming healthcare delivery across the globe. 
This document explores the current applications, challenges, and future prospects 
of AI in medical diagnosis, treatment planning, and patient care management.

Key findings indicate that AI can improve diagnostic accuracy by up to 30% in 
certain conditions while reducing the time required for analysis. However, 
implementation challenges remain, including data privacy concerns, regulatory 
compliance, and the need for extensive validation studies.
"""
    
    # Insert wrapped text
    rect = fitz.Rect(72, 145, 540, 400)
    page1.insert_textbox(rect, summary_text.strip(), fontsize=11, fontname="helv")
    
    # Page 2: Introduction and Background
    page2 = doc.new_page()
    
    page2.insert_text((72, 72), "1. Introduction", fontsize=16, fontname="helv")
    
    intro_text = """
The integration of Artificial Intelligence into healthcare systems represents one 
of the most significant technological shifts in modern medicine. Machine learning 
algorithms, natural language processing, and computer vision are enabling new 
approaches to disease detection, drug discovery, and personalized treatment.

Healthcare organizations worldwide are investing heavily in AI technologies. 
According to recent market analysis, the global AI in healthcare market is 
expected to reach $45 billion by 2030, growing at a compound annual rate of 
over 40 percent.
"""
    
    rect = fitz.Rect(72, 95, 540, 280)
    page2.insert_textbox(rect, intro_text.strip(), fontsize=11, fontname="helv")
    
    page2.insert_text((72, 300), "1.1 Background", fontsize=14, fontname="helv")
    
    background_text = """
The concept of using computers to assist medical diagnosis dates back to the 
1970s with early expert systems like MYCIN. However, the current wave of AI 
in healthcare is powered by deep learning and the availability of large 
medical datasets. Electronic health records, medical imaging databases, and 
genomic data have created opportunities for training sophisticated AI models.

Notable milestones include Google's DeepMind achieving human-level performance 
in detecting diabetic retinopathy from retinal scans, and IBM Watson's 
application in oncology treatment recommendations.
"""
    
    rect = fitz.Rect(72, 320, 540, 520)
    page2.insert_textbox(rect, background_text.strip(), fontsize=11, fontname="helv")
    
    # Page 3: Applications
    page3 = doc.new_page()
    
    page3.insert_text((72, 72), "2. Applications of AI in Healthcare", fontsize=16, fontname="helv")
    
    page3.insert_text((72, 100), "2.1 Medical Imaging", fontsize=14, fontname="helv")
    
    imaging_text = """
AI algorithms have shown remarkable success in analyzing medical images. 
Convolutional neural networks can detect abnormalities in X-rays, CT scans, 
MRIs, and pathology slides with accuracy comparable to or exceeding human 
experts. Applications include tumor detection, fracture identification, 
and screening for eye diseases.

Studies have demonstrated that AI-assisted radiologists can reduce reading 
times by 30-50% while maintaining or improving diagnostic accuracy.
"""
    
    rect = fitz.Rect(72, 120, 540, 280)
    page3.insert_textbox(rect, imaging_text.strip(), fontsize=11, fontname="helv")
    
    page3.insert_text((72, 300), "2.2 Drug Discovery", fontsize=14, fontname="helv")
    
    drug_text = """
Pharmaceutical companies are leveraging AI to accelerate drug discovery 
pipelines. Machine learning models can predict molecular properties, 
identify potential drug candidates, and optimize clinical trial designs. 
This approach has the potential to reduce drug development timelines from 
10-15 years to 3-5 years.

Recent successes include the use of AI to identify existing drugs that 
could be repurposed for new therapeutic applications, particularly 
during the COVID-19 pandemic response.
"""
    
    rect = fitz.Rect(72, 320, 540, 480)
    page3.insert_textbox(rect, drug_text.strip(), fontsize=11, fontname="helv")
    
    # Page 4: Challenges and Conclusion
    page4 = doc.new_page()
    
    page4.insert_text((72, 72), "3. Challenges and Considerations", fontsize=16, fontname="helv")
    
    challenges_text = """
Despite promising advances, several challenges must be addressed for 
widespread AI adoption in healthcare:

Data Privacy: Medical data is highly sensitive, requiring robust 
security measures and compliance with regulations like HIPAA and GDPR.

Algorithmic Bias: AI models can perpetuate or amplify existing biases 
in healthcare if training data is not representative of diverse populations.

Regulatory Approval: Medical AI systems require extensive validation 
and regulatory approval before clinical deployment.

Integration: Implementing AI tools within existing clinical workflows 
requires careful planning and change management.
"""
    
    rect = fitz.Rect(72, 95, 540, 320)
    page4.insert_textbox(rect, challenges_text.strip(), fontsize=11, fontname="helv")
    
    page4.insert_text((72, 340), "4. Conclusion", fontsize=16, fontname="helv")
    
    conclusion_text = """
Artificial Intelligence holds tremendous potential to improve healthcare 
outcomes, reduce costs, and enhance the patient experience. While challenges 
remain, continued research, regulatory development, and stakeholder 
collaboration will be essential to realizing the full benefits of AI in 
medicine. The future of healthcare will likely be characterized by 
human-AI collaboration, combining the strengths of both to deliver 
better care for patients worldwide.
"""
    
    rect = fitz.Rect(72, 365, 540, 520)
    page4.insert_textbox(rect, conclusion_text.strip(), fontsize=11, fontname="helv")
    
    # Save the PDF
    doc.save(str(pdf_path))
    doc.close()
    
    print(f"Created sample PDF: {pdf_path}")
    return str(pdf_path)


def demonstrate_basic_usage(pdf_path: str):
    """Demonstrate basic usage of the chunking pipeline."""
    print("\n" + "="*60)
    print("BASIC USAGE - Default Configuration")
    print("="*60)
    
    # Simple one-line usage
    chunks = chunk_document(pdf_path)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print("\nFirst 3 chunks:")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Token count: {chunk['metadata'].get('token_count', 'N/A')}")
        print(f"Entities: {chunk['metadata'].get('entities', [])[:3]}")  # First 3 entities
        print(f"Text preview: {chunk['text'][:150]}...")


def demonstrate_custom_config(pdf_path: str):
    """Demonstrate usage with custom configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION")
    print("="*60)
    
    # Custom configuration for smaller chunks
    config = PipelineConfig(
        max_tokens=256,  # Smaller chunks
        overlap_sentences=1,  # Less overlap
        extract_entities=True,
        detect_headings=True,
    )
    
    pipeline = ChunkingPipeline(config=config)
    chunks = pipeline.process(pdf_path)
    
    print(f"\nWith max_tokens=256:")
    print(f"Total chunks: {len(chunks)}")
    
    # Show token distribution
    token_counts = [c['metadata'].get('token_count', 0) for c in chunks]
    if token_counts:
        print(f"Token range: {min(token_counts)} - {max(token_counts)}")
        print(f"Average tokens per chunk: {sum(token_counts) / len(token_counts):.1f}")


def demonstrate_section_inspection(pdf_path: str):
    """Demonstrate inspecting document sections before chunking."""
    print("\n" + "="*60)
    print("SECTION INSPECTION")
    print("="*60)
    
    # Get sections without chunking
    pipeline = ChunkingPipeline()
    sections = pipeline.get_sections(pdf_path)
    
    print(f"\nDocument has {len(sections)} sections:")
    for section in sections:
        print(f"\n  [{section.page}] {section.title}")
        print(f"      Content length: {len(section.full_text)} chars")
        print(f"      Preview: {section.full_text[:80]}...")


def demonstrate_output_format(pdf_path: str):
    """Show the full output format for vector DB insertion."""
    print("\n" + "="*60)
    print("OUTPUT FORMAT (Vector DB Ready)")
    print("="*60)
    
    chunks = chunk_document(pdf_path, max_tokens=200)
    
    if chunks:
        print("\nSample chunk (JSON format):")
        print(json.dumps(chunks[0], indent=2))


def main():
    """Main demonstration function."""
    print("="*60)
    print("RAG Document Chunking Pipeline - Example Run")
    print("="*60)
    
    # Create or locate sample PDF
    pdf_path = create_sample_pdf()
    
    if pdf_path is None:
        print("\nCannot create sample PDF. Please provide a PDF file path.")
        if len(sys.argv) > 1:
            pdf_path = sys.argv[1]
        else:
            print("Usage: python example_run.py [path_to_pdf]")
            return
    
    try:
        # Run demonstrations
        demonstrate_basic_usage(pdf_path)
        demonstrate_custom_config(pdf_path)
        demonstrate_section_inspection(pdf_path)
        demonstrate_output_format(pdf_path)
        
        print("\n" + "="*60)
        print("Example run completed successfully!")
        print("="*60)
        
    except ImportError as e:
        print(f"\nMissing dependency: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        print("And download spaCy model: python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
