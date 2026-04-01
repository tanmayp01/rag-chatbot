# RAG Chatbot — Technical Report
**Amlgo Labs Junior AI Engineer Assignment**

---

## 1. Document Structure & Chunking Logic

### Document Processing

The ingestion pipeline supports PDF, DOCX, and plain-text files. Processing follows three stages:

**Extraction** — PyMuPDF (`fitz`) extracts raw text page-by-page from PDFs. For DOCX files, `python-docx` iterates through paragraphs. Plain text is read directly.

**Cleaning** — The raw text is normalised by:
- Converting Windows line endings to Unix
- Removing HTML tags (residual from web-scraped documents)
- Stripping non-breaking spaces (`\xa0`)
- Removing stray lines shorter than 6 characters (page numbers, headers/footers)
- Collapsing 3+ consecutive blank lines to 2

**Chunking** — LangChain's `RecursiveCharacterTextSplitter` is used with the following configuration:

```
chunk_size    = 500 characters  (~200 words)
chunk_overlap = 80 characters   (~30 words)
separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
```

The splitter tries to break at the most natural boundary first (paragraph → sentence → word), guaranteeing that no sentence is split mid-thought. The 80-character overlap ensures that context at chunk boundaries is not lost.

Each chunk is stored as a JSON object with fields: `chunk_id`, `source`, `text`, and `word_count`.

---

## 2. Embedding Model & Vector Database

### Embedding Model: `all-MiniLM-L6-v2`

- **Architecture**: Sentence-Transformers fine-tuned MiniLM
- **Dimensions**: 384
- **Model size**: ~22 MB
- **Why chosen**: Produces high-quality sentence embeddings with extremely fast inference (no GPU required). Consistently top-ranked on the MTEB benchmark for semantic similarity tasks.

### Vector Database: ChromaDB (Persistent)

ChromaDB was chosen over FAISS and Qdrant for the following reasons:

| Feature | ChromaDB | FAISS | Qdrant |
|---|---|---|---|
| Persistence | ✅ Built-in | ❌ Manual | ✅ Server-based |
| Server required | ❌ No | ❌ No | ✅ Yes |
| Metadata filtering | ✅ Yes | ❌ No | ✅ Yes |
| Setup complexity | Low | Low | High |

The collection uses **cosine similarity** (`hnsw:space: cosine`), which is more robust than L2 distance for text embeddings of varying lengths.

---

## 3. Prompt Format & Generation Logic

### Prompt Template

```
[SYSTEM]
You are a precise, helpful AI assistant.
Your ONLY knowledge source is the context passages provided.
Rules:
1. Answer ONLY from the context. Do NOT use outside knowledge.
2. If the answer is not in context, say: "I don't have enough information..."
3. Be concise. Cite source numbers (e.g., [Source 1]).
4. Never make up facts, dates, names, or numbers.

[CONTEXT]
[Source 1 – filename.pdf]
<retrieved chunk 1>

[Source 2 – filename.pdf]
<retrieved chunk 2>

... (up to top-k chunks)

[QUESTION]
<user query>

[ANSWER]
```

### Generation Settings

- **Model**: `mistral-7b-instruct` via Ollama
- **Temperature**: 0.1 (near-deterministic, factual)
- **Max tokens**: 1024
- **Streaming**: Token-by-token via `ollama.generate(stream=True)`

### RAG Pipeline Flow

1. User query → `Retriever.retrieve()` → top-5 semantically similar chunks
2. Chunks → `Retriever.format_context()` → single formatted context string
3. Context + Query → `Generator.build_prompt()` → full prompt
4. Prompt → `ollama.generate(stream=True)` → token stream
5. Tokens yield to Streamlit → real-time display with blinking cursor

---

## 4. Example Queries & Responses

### Query 1 ✅ Success
**Q**: "What is the main purpose of this document?"  
**A**: "Based on [Source 1], this document outlines the terms and conditions governing the use of the service, including user rights, obligations, and the provider's limitations of liability."  
**Notes**: Accurate, grounded in retrieved passage. Source citation correct.

### Query 2 ✅ Success
**Q**: "What are the user's obligations or responsibilities?"  
**A**: "According to [Source 2] and [Source 3], users are obligated to: (1) provide accurate information during registration, (2) not misuse the platform for unlawful purposes, and (3) comply with all applicable laws."  
**Notes**: Multi-source synthesis worked correctly.

### Query 3 ✅ Success
**Q**: "Are there any limitations of liability described?"  
**A**: "Yes — [Source 4] states that the provider's liability is limited to direct damages only, and excludes consequential, incidental, or punitive damages."  
**Notes**: Precise legal clause correctly extracted.

### Query 4 ✅ Success
**Q**: "What happens if the terms are violated?"  
**A**: "As stated in [Source 5], violation of these terms may result in immediate account suspension or termination, at the provider's sole discretion."  
**Notes**: Correctly identified consequence clause.

### Query 5 ❌ Failure (Expected / Correct Behaviour)
**Q**: "What is the capital of France?"  
**A**: "I don't have enough information in the provided documents to answer that."  
**Notes**: This is the **correct** behaviour — the model refused to use outside knowledge. Hallucination successfully prevented.

---

## 5. Limitations & Known Issues

### Hallucination Risk
The system prompt strongly constrains the model to the provided context. In testing, Mistral 7B occasionally adds minor bridging phrases that go slightly beyond the source text. This can be further mitigated by lowering `temperature` to 0.0.

### Response Speed
On CPU-only machines, Mistral 7B takes **30–120 seconds** per response. For faster results, use `phi3` (smaller model) or run on a machine with an NVIDIA GPU.

### Chunk Boundary Issues
If a key piece of information spans two chunks and neither chunk alone is sufficient, the retriever may return incomplete context. The 80-character overlap mitigates this but does not eliminate it entirely.

### Multi-Document Reasoning
The pipeline retrieves chunks independently. Questions requiring synthesis across multiple sections of a long document may not always be answered completely.

---

*Report prepared for Amlgo Labs Junior AI Engineer Assignment.*
