# 🔍 RAG Chatbot — Fine-Tuned Retrieval-Augmented Generation

A production-ready RAG (Retrieval-Augmented Generation) chatbot that answers questions **grounded strictly in your uploaded documents** — no hallucinations, no guesswork.

---

## 📐 Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                          │
│                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │  Document   │───▶│  ChromaDB Vector │───▶│  Retriever │  │
│  │  Ingestion  │    │  Database        │    │  (top-k)   │  │
│  └─────────────┘    └──────────────────┘    └─────┬──────┘  │
│   preprocess.py          vectordb/               │          │
│   embedder.py                                    ▼          │
│                                         ┌────────────────┐  │
│                                         │   LLM Generator │  │
│                                         │  (Mistral 7B)  │  │
│                                         │   via Ollama   │  │
│                                         └────────┬───────┘  │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                                                   ▼
                                    Streamed Answer + Sources
                                         (Streamlit UI)
```

### Component Overview

| File | Role |
|---|---|
| `src/preprocess.py` | PDF/DOCX/TXT extraction, cleaning, sentence-aware chunking |
| `src/embedder.py` | Generates embeddings (all-MiniLM-L6-v2), stores in ChromaDB |
| `src/retriever.py` | Semantic search wrapper, formats context for the LLM |
| `src/generator.py` | Builds the RAG prompt, streams Mistral responses via Ollama |
| `src/pipeline.py` | Orchestrates the full Retrieve→Generate loop |
| `app.py` | Streamlit UI with streaming, sources panel, sidebar stats |
| `setup.py` | One-command setup (install deps, pull model, ingest docs) |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** — local LLM runtime  
   Download from [https://ollama.ai](https://ollama.ai)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Run the one-command setup
#    (installs deps, pulls Mistral, ingests any PDF in /data)
python setup.py
```

### Running the Chatbot

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📄 Step-by-Step: From Document to Chatbot

### Step 1 — Put your document in `/data`

```
rag-chatbot/
└── data/
    └── my_document.pdf   ← drop your PDF / DOCX / TXT here
```

### Step 2 — Preprocess (chunk the document)

```bash
python src/preprocess.py data/my_document.pdf
# → saves chunks to chunks/my_document_chunks.json
```

### Step 3 — Create embeddings and index in ChromaDB

```bash
python src/embedder.py --chunks-dir chunks --db-path vectordb
# → persists vector DB to vectordb/
```

### Step 4 — Run the chatbot

```bash
streamlit run app.py
```

Or do steps 2–4 in one command:

```bash
python setup.py --doc data/my_document.pdf
streamlit run app.py
```

---

## 🧠 Model & Embedding Choices

### Embedding Model: `all-MiniLM-L6-v2`
- **Size**: 22 MB
- **Why**: Fast, free, no GPU needed. Produces high-quality 384-dim sentence embeddings. Ideal for semantic search over English legal/policy documents.
- **Alternatives**: `bge-small-en-v1.5` (slightly better quality), `text-embedding-ada-002` (OpenAI, paid)

### LLM: `mistral-7b-instruct`
- **Why**: Best open-source instruction-following model at 7B parameters. Runs on CPU (slow) or GPU (fast) via Ollama. Strictly follows the "only answer from context" instruction.
- **Alternatives** (set in sidebar): `llama3`, `zephyr`, `phi3`

### Vector DB: `ChromaDB` (persistent local)
- **Why**: Zero setup — no Docker, no server. Persists to disk. Supports cosine similarity out of the box.
- **Alternatives**: FAISS (no persistence), Qdrant (server-based, production)

---

## 🔧 Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size    = 500,    # ~200 words per chunk
    chunk_overlap = 80,     # overlap to preserve sentence context at boundaries
    separators    = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)
```

The splitter tries to break at paragraph → sentence → word boundaries in order, producing natural, semantically coherent chunks.

---

## 💬 Prompt Template

```
You are a precise, helpful AI assistant.
Your ONLY knowledge source is the context passages provided below.
Rules:
1. Answer ONLY from the provided context.
2. If the answer is not in the context, say: "I don't have enough information..."
3. Cite the source number (e.g., [Source 1]) when you use it.
4. Never make up facts, dates, names, or numbers.

=== CONTEXT ===
[Source 1 – document.pdf]
... retrieved chunk 1 ...

[Source 2 – document.pdf]
... retrieved chunk 2 ...

=== QUESTION ===
{user question}

=== ANSWER ===
```

---

## 📊 Sample Queries & Outputs

| Query | Result |
|---|---|
| "What is the main purpose of this document?" | ✅ Accurate summary from context |
| "What are the user's obligations?" | ✅ Lists obligations with source citations |
| "Are there limitations of liability?" | ✅ Quotes relevant clauses |
| "What happens if terms are violated?" | ✅ Explains consequences from document |
| "What is the capital of France?" | ✅ Correctly says "not in documents" |

### Known Limitations

- **Slow on CPU**: Mistral 7B takes 30–120 seconds per response without a GPU. Use `phi3` for faster (but lower quality) answers.
- **Long documents**: Very large documents (>50,000 words) may need a lower `chunk_size` to fit in the LLM's context window.
- **Multi-document reasoning**: The current pipeline retrieves chunks independently; cross-document synthesis may miss connections.

---

## 📁 Folder Structure

```
rag-chatbot/
├── data/                  ← Place your source documents here
├── chunks/                ← Auto-generated chunk JSON files
├── vectordb/              ← Persisted ChromaDB vector database
├── notebooks/
│   └── evaluate_rag.py    ← Offline evaluation script
├── src/
│   ├── preprocess.py      ← Document cleaning + chunking
│   ├── embedder.py        ← Embedding + ChromaDB indexing
│   ├── retriever.py       ← Semantic search interface
│   ├── generator.py       ← LLM prompt + streaming
│   └── pipeline.py        ← End-to-end RAG orchestrator
├── app.py                 ← Streamlit chatbot UI
├── setup.py               ← One-command setup script
├── requirements.txt
└── README.md
```

---

## 📺 Demo

> Add a GIF or video link here after recording your demo.

```
[GIF PLACEHOLDER — record with: streamlit run app.py, then use LiceCap or OBS]
```

---

## 🔗 GitHub Repository

> Replace with your actual GitHub URL after pushing.

```
https://github.com/YOUR_USERNAME/rag-chatbot
```

---

## 📋 Requirements

See `requirements.txt`. Key packages:

- `streamlit>=1.31` — UI framework with `st.write_stream` support
- `langchain` — text splitting utilities
- `chromadb` — local vector database
- `sentence-transformers` — embedding model
- `ollama` — Python client for local LLM inference
- `PyMuPDF` — PDF text extraction
