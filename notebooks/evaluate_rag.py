"""
notebooks/evaluate_rag.py
--------------------------
Offline evaluation of the RAG pipeline.
Run this to test example queries and inspect quality.

Usage:
    cd rag-chatbot
    python notebooks/evaluate_rag.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import RAGPipeline

# ── Sample queries ────────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "What is the main purpose of this document?",
    "What are the user's obligations or responsibilities?",
    "Are there any limitations of liability described?",
    "What happens if the terms are violated?",
    "How is user data handled or protected?",
    # Failure case — likely not in document
    "What is the capital of France?",
]

def evaluate():
    print("Loading RAG pipeline …\n")
    pipeline = RAGPipeline(top_k=5)
    stats = pipeline.stats()
    print(f"Stats: {stats}\n")
    print("=" * 70)

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n[Q{i}] {query}")
        print("-" * 60)
        answer, sources = pipeline.query(query)
        print(f"Answer:\n{answer.strip()}\n")
        print(f"Sources used ({len(sources)}):")
        for s in sources:
            preview = s['text'][:120].replace('\n', ' ')
            print(f"  [{s['source']}] score={s['score']} | {preview}…")
        print("=" * 70)

if __name__ == "__main__":
    evaluate()
