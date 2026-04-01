"""
setup.py
--------
Run this ONCE before starting the app.
It will:
  1. Install all Python dependencies
  2. Check that Ollama is installed
  3. Pull the Mistral model if not already available
  4. Ingest any document placed in /data

Usage:
    python setup.py
    python setup.py --doc data/my_document.pdf --model mistral
"""

import os
import sys
import subprocess
import argparse


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=None, help="Path to document to ingest")
    parser.add_argument("--model", default="mistral", help="Ollama model to pull")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip Ollama check/pull")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG Chatbot — Setup")
    print("=" * 60)

    # 1. Install dependencies
    print("\n[1/4] Installing Python dependencies …")
    run(f"{sys.executable} -m pip install -r requirements.txt --quiet")

    # 2. Check Ollama
    if not args.skip_ollama:
        print("\n[2/4] Checking Ollama …")
        result = run("ollama --version", check=False)
        if result.returncode != 0:
            print(
                "\n⚠️  Ollama not found!\n"
                "  Install it from: https://ollama.ai\n"
                "  Then re-run setup.py\n"
                "  Or pass --skip-ollama to continue without it."
            )
            sys.exit(1)
        print(f"  Pulling model: {args.model} (this may take a few minutes) …")
        run(f"ollama pull {args.model}", check=False)
    else:
        print("\n[2/4] Skipping Ollama check (--skip-ollama)")

    # 3. Ingest document
    doc = args.doc
    if doc is None:
        # Auto-detect first file in /data
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        candidates = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]
        doc = candidates[0] if candidates else None

    if doc:
        print(f"\n[3/4] Ingesting document: {doc}")
        sys.path.insert(0, "src")
        from src.preprocess import process_document
        from src.embedder import Embedder

        chunks = process_document(doc, output_dir="chunks")
        emb = Embedder(db_path="vectordb")
        emb.index_chunks(chunks)
        print(f"  ✓ {emb.stats()['total_chunks']} chunks indexed")
    else:
        print(
            "\n[3/4] No document found in /data — skipping ingestion.\n"
            "  Put a PDF/DOCX/TXT in the /data folder and re-run, or\n"
            "  upload directly through the Streamlit sidebar."
        )

    print("\n[4/4] All done!")
    print("\n" + "=" * 60)
    print("  Start the chatbot:")
    print("  streamlit run app.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
