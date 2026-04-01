"""
pipeline.py
-----------
End-to-end RAG pipeline that ties together:
    Retriever  →  context assembly  →  Generator (streaming)

This is the single entry point used by app.py.
"""

from typing import Generator, Dict, List, Tuple, Optional
from retriever import Retriever
from generator import Generator as LLMGenerator


class RAGPipeline:
    """
    Orchestrates the full Retrieve-then-Generate loop.

    Usage:
        pipeline = RAGPipeline()

        # Streaming (for Streamlit)
        for token in pipeline.stream("What is the refund policy?"):
            print(token, end="")

        # Non-streaming (for testing)
        answer, sources = pipeline.query("What is the refund policy?")
    """

    def __init__(
        self,
        db_path: str = "vectordb",
        model: str = "mistral",
        top_k: int = 5,
        temperature: float = 0.1,
        score_threshold: float = 1.0,
    ):
        print("[pipeline] Initialising RAG pipeline …")
        self.retriever = Retriever(db_path=db_path, top_k=top_k)
        self.llm = LLMGenerator(model=model, temperature=temperature)
        self.top_k = top_k
        self.score_threshold = score_threshold
        print("[pipeline] Ready ✓")

    # ------------------------------------------------------------------
    def retrieve(self, question: str) -> List[Dict]:
        """Return top-k relevant chunks for a question."""
        return self.retriever.retrieve(
            question,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

    # ------------------------------------------------------------------
    def stream(
        self,
        question: str,
        chat_history: Optional[List[Dict]] = None,
    ) -> Generator[str, None, None]:
        """
        Full streaming RAG response.

        Yields:
        - First, a special marker token "__SOURCES__:<json>" with retrieved sources
        - Then, the LLM response tokens one by one

        app.py splits on "__SOURCES__" to separate sources from the answer stream.
        """
        # 1. Retrieve relevant chunks
        chunks = self.retrieve(question)

        if not chunks:
            yield "⚠️ No relevant documents found in the knowledge base."
            return

        # 2. Emit sources metadata as first yield (JSON-encoded, caught by app.py)
        import json
        yield f"__SOURCES__{json.dumps(chunks)}\n"

        # 3. Build context string
        context = self.retriever.format_context(chunks)

        # 4. Stream LLM response
        for token in self.llm.stream(context, question):
            yield token

    # ------------------------------------------------------------------
    def query(self, question: str) -> Tuple[str, List[Dict]]:
        """
        Non-streaming version – returns (answer_text, source_chunks).
        Useful for testing / evaluation.
        """
        chunks = self.retrieve(question)
        if not chunks:
            return "No relevant documents found.", []
        context = self.retriever.format_context(chunks)
        answer = self.llm.generate(context, question)
        return answer, chunks

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        return {
            **self.retriever.stats(),
            "llm_model": self.llm.model,
            "top_k": self.top_k,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    pipeline = RAGPipeline()
    print(f"\nQ: {question}\n{'='*60}")

    sources_printed = False
    for token in pipeline.stream(question):
        if token.startswith("__SOURCES__"):
            import json
            sources = json.loads(token.replace("__SOURCES__", "").strip())
            print(f"\n[Sources: {[s['source'] for s in sources]}]\n")
            sources_printed = True
        else:
            print(token, end="", flush=True)
    print()
