"""
Run the RAG chunker on data/input_pdfs/ and save results to data/chunks/.

Usage:
    python test_my_pdf.py                        # process all PDFs
    python test_my_pdf.py path/to/file.pdf       # process one file
"""

import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_chunker import chunk_document

# Paths relative to the project root (one level above this script)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR    = PROJECT_ROOT / "data" / "input_pdfs"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "chunks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ISSUES = []


def sp(*args):
    """Print with safe encoding on Windows."""
    text = " ".join(str(a) for a in args)
    enc  = sys.stdout.encoding or "utf-8"
    print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def process_pdf(pdf_path: Path):
    sp(f"\n{'='*60}")
    sp(f"Processing: {pdf_path.name}")
    sp(f"{'='*60}")

    try:
        chunks = chunk_document(str(pdf_path))
    except Exception as e:
        ISSUES.append((pdf_path.name, f"EXCEPTION: {e}"))
        sp(f"  [ERROR] {e}")
        traceback.print_exc()
        return None

    if not chunks:
        ISSUES.append((pdf_path.name, "No chunks produced"))
        sp("  [WARN] No chunks produced")
        return None

    # Stats
    tokens = [c["metadata"]["token_count"] for c in chunks]
    over   = sum(1 for t in tokens if t > 512)
    under  = sum(1 for t in tokens if t < 50)

    sp(f"  Chunks : {len(chunks)}")
    sp(f"  Tokens : min={min(tokens)}  max={max(tokens)}  avg={sum(tokens)/len(tokens):.1f}")
    if over:
        sp(f"  [WARN] Over-512-token chunks : {over}")
    if under:
        sp(f"  [WARN] Under-50-token chunks : {under}")

    # Sections
    sections: dict = {}
    for c in chunks:
        sec = c["metadata"].get("section", "UNKNOWN")
        sections[sec] = sections.get(sec, 0) + 1
    sp(f"  Sections : {len(sections)}")
    for sec, cnt in list(sections.items())[:6]:
        sp(f"    * {sec[:55]:<55} {cnt} chunk(s)")
    if len(sections) > 6:
        sp(f"    ... and {len(sections)-6} more")

    # Save
    out = OUTPUT_DIR / (pdf_path.stem + "_chunks.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    sp(f"  Saved -> {out.relative_to(PROJECT_ROOT)}")

    return {"total": len(chunks), "min": min(tokens), "max": max(tokens),
            "avg": round(sum(tokens)/len(tokens), 1), "over": over, "under": under}


def main():
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            sp(f"File not found: {pdf_path}")
            sys.exit(1)
        process_pdf(pdf_path)
        return

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        sp(f"No PDF files found in {INPUT_DIR}")
        return

    sp(f"Found {len(pdf_files)} PDF(s) in data/input_pdfs/")

    all_stats = {}
    for pdf in pdf_files:
        stats = process_pdf(pdf)
        if stats:
            all_stats[pdf.name] = stats

    # Summary
    sp(f"\n{'='*60}")
    sp("SUMMARY")
    sp(f"{'='*60}")
    sp(f"Processed : {len(all_stats)} / {len(pdf_files)}")
    sp(f"Total chunks    : {sum(s['total'] for s in all_stats.values())}")
    avgs = [s['avg'] for s in all_stats.values()]
    sp(f"Avg tokens/chunk: {sum(avgs)/len(avgs):.1f}")
    sp(f"Over-512 chunks : {sum(s['over'] for s in all_stats.values())}")
    sp(f"Under-50 chunks : {sum(s['under'] for s in all_stats.values())}")

    if ISSUES:
        sp(f"\nIssues ({len(ISSUES)}):")
        for name, msg in ISSUES:
            sp(f"  [{name}] {msg}")
    else:
        sp("\nAll PDFs chunked successfully.")


if __name__ == "__main__":
    main()
