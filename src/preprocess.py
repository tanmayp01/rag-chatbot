"""
preprocess.py
-------------
Cleans and chunks source documents into 100–300 word segments
using sentence-aware splitting. Saves chunks to /chunks as JSON.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Optional heavy imports – guarded so the module can be imported without them
# ---------------------------------------------------------------------------
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF using PyMuPDF."""
    if fitz is None:
        raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")
    doc = fitz.open(file_path)
    pages = [page.get_text("text") for page in doc]
    return "\n".join(pages)


def extract_text_from_docx(file_path: str) -> str:
    """Extract raw text from a .docx file."""
    if DocxDocument is None:
        raise ImportError("python-docx not installed. Run: pip install python-docx")
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_from_txt(file_path: str) -> str:
    """Read a plain-text file."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Dispatch extraction based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove common artefacts found in legal/policy documents:
    - Multiple blank lines  → single blank line
    - Page headers/footers  → stripped (lines shorter than 6 chars)
    - HTML tags             → removed
    - Non-breaking spaces   → regular space
    - Windows line endings  → Unix
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Non-breaking space
    text = text.replace("\xa0", " ")
    # Remove lines that look like page numbers or stray headers (< 6 chars)
    lines = text.split("\n")
    lines = [ln for ln in lines if len(ln.strip()) >= 6 or ln.strip() == ""]
    text = "\n".join(lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter (sentence-aware).

    chunk_size    – target characters per chunk  (~200-300 words at 2.5 chars/word)
    chunk_overlap – overlap to preserve context across boundaries
    """
    if RecursiveCharacterTextSplitter is None:
        raise ImportError("langchain not installed. Run: pip install langchain")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_document(
    file_path: str,
    output_dir: str = "chunks",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Dict]:
    """
    Full preprocessing pipeline:
    1. Extract text
    2. Clean text
    3. Chunk text
    4. Save chunks as JSON

    Returns a list of chunk dicts:
    {
        "chunk_id": "doc_name_0",
        "source":   "filename.pdf",
        "text":     "...",
        "word_count": 213
    }
    """
    print(f"[preprocess] Loading: {file_path}")
    raw_text = extract_text(file_path)

    print("[preprocess] Cleaning text …")
    clean = clean_text(raw_text)

    print(f"[preprocess] Chunking (size={chunk_size}, overlap={chunk_overlap}) …")
    raw_chunks = chunk_text(clean, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    source_name = Path(file_path).name
    base_name = Path(file_path).stem

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append(
            {
                "chunk_id": f"{base_name}_{i}",
                "source": source_name,
                "text": text.strip(),
                "word_count": len(text.split()),
            }
        )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[preprocess] Saved {len(chunks)} chunks → {out_path}")
    return chunks


def load_chunks(chunks_dir: str = "chunks") -> List[Dict]:
    """Load all chunk JSON files from a directory."""
    all_chunks = []
    for p in Path(chunks_dir).glob("*_chunks.json"):
        with open(p, "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))
    print(f"[preprocess] Loaded {len(all_chunks)} total chunks from {chunks_dir}/")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess documents for RAG.")
    parser.add_argument("file", help="Path to input document (PDF, DOCX, TXT)")
    parser.add_argument("--output-dir", default="chunks", help="Output directory for chunk JSON")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=80)
    args = parser.parse_args()

    process_document(
        file_path=args.file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
