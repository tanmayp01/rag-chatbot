"""
generator.py
------------
Handles prompt construction and streaming LLM generation via Ollama.

Model  : mistral  (pulled via `ollama pull mistral`)
Stream : token-by-token using Ollama's streaming API
"""

import ollama
from typing import List, Dict, Generator, Optional


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise, helpful AI assistant.
Your ONLY knowledge source is the context passages provided below.
Rules you must follow:
1. Answer ONLY from the provided context. Do NOT use outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have enough information in the provided documents to answer that."
3. Be concise and factual. Cite the source number (e.g., [Source 1]) when you use it.
4. Never make up facts, dates, names, or numbers.
5. If the question is ambiguous, state your interpretation before answering."""

PROMPT_TEMPLATE = """{system}

=== CONTEXT ===
{context}

=== QUESTION ===
{question}

=== ANSWER ==="""


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class Generator:
    """
    Wraps Ollama to provide:
    - build_prompt()  : assembles system + context + question
    - stream()        : yields tokens one by one
    - generate()      : returns full response (non-streaming)
    """

    def __init__(self, model: str = "mistral", temperature: float = 0.1):
        """
        Parameters
        ----------
        model       : Ollama model name.  Run `ollama pull mistral` first.
        temperature : Lower = more factual / deterministic (0.1 recommended for RAG)
        """
        self.model = model
        self.temperature = temperature
        self._verify_model()

    # ------------------------------------------------------------------
    def _verify_model(self) -> None:
        """Check that the model is available in Ollama."""
        try:
            available = [m["name"] for m in ollama.list()["models"]]
            if not any(self.model in name for name in available):
                print(
                    f"[generator] WARNING: model '{self.model}' not found in Ollama.\n"
                    f"  Run:  ollama pull {self.model}\n"
                    f"  Available: {available}"
                )
            else:
                print(f"[generator] Model '{self.model}' ready ✓")
        except Exception as e:
            print(f"[generator] Could not connect to Ollama: {e}\n"
                  "  Make sure Ollama is running: https://ollama.ai")

    # ------------------------------------------------------------------
    def build_prompt(self, context: str, question: str) -> str:
        """Assemble the full prompt string."""
        return PROMPT_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            context=context,
            question=question,
        )

    # ------------------------------------------------------------------
    def stream(
        self,
        context: str,
        question: str,
    ) -> Generator[str, None, None]:
        """
        Generator that yields response tokens one by one.
        Use this for real-time streaming in Streamlit.

        Usage:
            for token in generator.stream(context, question):
                print(token, end="", flush=True)
        """
        prompt = self.build_prompt(context, question)
        try:
            stream = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=True,
                options={"temperature": self.temperature, "num_predict": 1024},
            )
            for chunk in stream:
                token = chunk.get("response", "")
                if token:
                    yield token
        except Exception as e:
            yield f"\n\n⚠️ Error connecting to Ollama: {e}\nMake sure Ollama is running and the model is pulled."

    # ------------------------------------------------------------------
    def generate(self, context: str, question: str) -> str:
        """Non-streaming generation — returns full answer as a string."""
        return "".join(self.stream(context, question))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    dummy_context = "[Source 1 – sample.txt]\nThis document outlines the terms and conditions of service."

    gen = Generator()
    print(f"\nQ: {question}\n{'='*60}")
    for tok in gen.stream(dummy_context, question):
        print(tok, end="", flush=True)
    print()
