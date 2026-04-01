"""
retriever.py
------------
High-level retriever interface used by the RAG pipeline.
Wraps the Embedder's query method and applies optional re-ranking / filtering.
"""

from typing import List, Dict, Optional
from embedder import Embedder, get_embedder


class Retriever:
    """
    Retrieves the top-k most relevant chunks for a user query.

    Usage:
        retriever = Retriever(db_path="vectordb")
        chunks = retriever.retrieve("What is the refund policy?", top_k=5)
    """

    def __init__(self, db_path: str = "vectordb", top_k: int = 5):
        self.embedder: Embedder = get_embedder(db_path=db_path)
        self.default_top_k = top_k

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: float = 1.0,   # cosine distance; lower = better
    ) -> List[Dict]:
        """
        Semantic search. Returns a list of relevant chunks sorted by
        relevance (best first).

        Parameters
        ----------
        query           : natural language question from the user
        top_k           : how many chunks to retrieve (default: self.default_top_k)
        score_threshold : discard chunks with cosine distance > threshold
                          (1.0 means keep everything)

        Returns
        -------
        List of dicts, each with keys:
            chunk_id, text, source, score
        """
        k = top_k if top_k is not None else self.default_top_k
        hits = self.embedder.query(query, top_k=k)

        # Optional filtering by score
        filtered = [h for h in hits if h["score"] <= score_threshold]
        return filtered

    # ------------------------------------------------------------------
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Build a single context string from retrieved chunks,
        formatted for injection into the LLM prompt.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i} – {chunk['source']}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        return self.embedder.stats()


# ---------------------------------------------------------------------------
# CLI entry point – sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the privacy policy?"
    r = Retriever()
    chunks = r.retrieve(query, top_k=3)
    print(f"\nTop {len(chunks)} results for: '{query}'\n{'='*60}")
    for c in chunks:
        print(f"\n[{c['chunk_id']}] score={c['score']}\n{c['text'][:300]} …")
