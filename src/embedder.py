"""
embedder.py
-----------
Generates sentence embeddings for all chunks and stores them in
a persistent ChromaDB vector database.

Embedding model : all-MiniLM-L6-v2  (fast, lightweight, great quality)
Vector DB       : ChromaDB           (local, no server required)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_documents"
BATCH_SIZE = 64          # embed this many chunks per forward pass


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------

class Embedder:
    """
    Loads the embedding model once and provides methods to:
    - embed a list of texts
    - build / update the ChromaDB collection
    - query the collection for similar chunks
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        db_path: str = "vectordb",
        collection_name: str = COLLECTION_NAME,
    ):
        print(f"[embedder] Loading embedding model: {model_name} …")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.collection_name = collection_name

        os.makedirs(db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        print(f"[embedder] ChromaDB ready at '{db_path}/' — "
              f"collection '{collection_name}' has {self.collection.count()} docs.")

    # ------------------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors for the given texts."""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    # ------------------------------------------------------------------
    def index_chunks(self, chunks: List[Dict], force_reindex: bool = False) -> None:
        """
        Add chunks to ChromaDB.  Skips chunks that are already indexed
        unless force_reindex=True (which clears the collection first).
        """
        if force_reindex:
            print("[embedder] force_reindex=True → clearing existing collection …")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Find which chunk_ids are already in the DB
        existing_ids: set = set()
        if self.collection.count() > 0:
            existing = self.collection.get(include=[])
            existing_ids = set(existing["ids"])

        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]
        if not new_chunks:
            print("[embedder] All chunks already indexed — nothing to do.")
            return

        print(f"[embedder] Indexing {len(new_chunks)} new chunks …")

        for i in tqdm(range(0, len(new_chunks), BATCH_SIZE), desc="Embedding"):
            batch = new_chunks[i : i + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            ids = [c["chunk_id"] for c in batch]
            metadatas = [
                {"source": c["source"], "word_count": c["word_count"]}
                for c in batch
            ]
            embeddings = self.embed_texts(texts)

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        print(f"[embedder] Collection now has {self.collection.count()} documents.")

    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Semantic search: returns top_k most similar chunks.

        Returns list of dicts:
        {
            "chunk_id": ...,
            "text":     ...,
            "source":   ...,
            "score":    float  (cosine distance – lower is more similar)
        }
        """
        query_embedding = self.embed_texts([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "chunk_id": results["ids"][0][len(hits)],
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "score": round(dist, 4),
                }
            )
        return hits

    # ------------------------------------------------------------------
    def stats(self) -> Dict:
        return {
            "embedding_model": self.model_name,
            "collection": self.collection_name,
            "total_chunks": self.collection.count(),
        }


# ---------------------------------------------------------------------------
# Convenience functions used by other modules
# ---------------------------------------------------------------------------

_embedder_singleton: Optional[Embedder] = None


def get_embedder(db_path: str = "vectordb") -> Embedder:
    """Return a cached Embedder instance (lazy singleton)."""
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = Embedder(db_path=db_path)
    return _embedder_singleton


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from preprocess import load_chunks

    parser = argparse.ArgumentParser(description="Embed chunks into ChromaDB.")
    parser.add_argument("--chunks-dir", default="chunks")
    parser.add_argument("--db-path", default="vectordb")
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks_dir)
    emb = Embedder(db_path=args.db_path)
    emb.index_chunks(chunks, force_reindex=args.force_reindex)
    print("[embedder] Done!", emb.stats())
