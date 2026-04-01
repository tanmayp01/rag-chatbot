"""
app.py
------
Streamlit chatbot interface with:
  - Real-time streaming responses
  - Source passage display
  - Sidebar: model info, chunk count, document upload
  - Clear chat / reset functionality
"""

import os
import sys
import json
import time
import streamlit as st

# Make src/ importable when app.py lives at project root
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background: #0f1117; color: #e8eaf0; }

    /* Header */
    .rag-header {
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 1px solid #2a2d3a;
        margin-bottom: 1.5rem;
    }
    .rag-header h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        color: #7dd3fc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .rag-header p { color: #94a3b8; font-size: 0.85rem; margin: 0.25rem 0 0 0; }

    /* Chat messages */
    .user-msg {
        background: #1e2130;
        border-left: 3px solid #7dd3fc;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
    }
    .bot-msg {
        background: #161922;
        border-left: 3px solid #34d399;
        border-radius: 4px;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.65;
    }

    /* Source pills */
    .source-container {
        margin-top: 0.75rem;
        padding-top: 0.6rem;
        border-top: 1px dashed #2a2d3a;
    }
    .source-header {
        font-size: 0.72rem;
        color: #64748b;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .source-pill {
        display: inline-block;
        background: #1e2a38;
        color: #7dd3fc;
        border: 1px solid #1d4ed8;
        border-radius: 3px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        margin: 2px 4px 2px 0;
    }
    .source-text {
        background: #12151e;
        border-radius: 4px;
        padding: 0.6rem 0.85rem;
        font-size: 0.8rem;
        color: #94a3b8;
        font-family: 'IBM Plex Mono', monospace;
        margin-top: 0.3rem;
        white-space: pre-wrap;
        line-height: 1.55;
        border: 1px solid #1e2130;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #090c13;
        border-right: 1px solid #1e2130;
    }
    .stat-card {
        background: #141720;
        border: 1px solid #1e2130;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
    }
    .stat-label { color: #64748b; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; }
    .stat-value { color: #7dd3fc; font-size: 1.05rem; font-weight: 600; margin-top: 0.15rem; }

    /* Input area */
    .stTextInput > div > div > input {
        background: #141720 !important;
        border: 1px solid #2a2d3a !important;
        color: #e8eaf0 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #7dd3fc !important;
        box-shadow: 0 0 0 2px rgba(125,211,252,0.15) !important;
    }

    /* Buttons */
    .stButton > button {
        background: #1e2130;
        border: 1px solid #2a2d3a;
        color: #e8eaf0;
        border-radius: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        padding: 0.4rem 1rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #252a40;
        border-color: #7dd3fc;
        color: #7dd3fc;
    }

    /* Spinner */
    .thinking-badge {
        display: inline-block;
        background: #0f2940;
        color: #7dd3fc;
        border: 1px solid #1d4ed8;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 3px 10px;
        margin-bottom: 0.5rem;
        animation: pulse 1.2s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    div[data-testid="stExpander"] { border-color: #1e2130 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Pipeline loader (cached so it loads only once) ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline(model: str, db_path: str):
    from pipeline import RAGPipeline
    return RAGPipeline(db_path=db_path, model=model)


# ── Ingest helper ────────────────────────────────────────────────────────────
def ingest_file(uploaded_file, db_path: str) -> int:
    """Save upload, chunk it, embed it, return chunk count."""
    import tempfile
    from preprocess import process_document
    from embedder import Embedder

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1],
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunks = process_document(tmp_path, output_dir="chunks")
    emb = Embedder(db_path=db_path)
    emb.index_chunks(chunks)
    os.unlink(tmp_path)
    return emb.stats()["total_chunks"]


# ── Session state defaults ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, sources}]
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model_name = st.selectbox(
        "LLM Model",
        ["mistral", "llama3", "zephyr", "phi3"],
        index=0,
        help="Make sure the model is pulled via: ollama pull <model>",
    )

    db_path = st.text_input("Vector DB path", value="vectordb")

    top_k = st.slider("Chunks retrieved (top-k)", 2, 10, 5)

    st.divider()
    st.markdown("## 📄 Upload Document")
    uploaded = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )
    if uploaded and uploaded.name not in st.session_state.ingested_files:
        with st.spinner(f"Ingesting {uploaded.name} …"):
            try:
                count = ingest_file(uploaded, db_path)
                st.session_state.total_chunks = count
                st.session_state.ingested_files.append(uploaded.name)
                st.success(f"✓ Indexed {count} chunks")
                # Reset cached pipeline so it picks up new chunks
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.markdown("## 📊 Stats")

    # Try to get real stats from DB
    try:
        from embedder import get_embedder
        emb = get_embedder(db_path=db_path)
        s = emb.stats()
        total_chunks = s["total_chunks"]
        embed_model = s["embedding_model"]
    except Exception:
        total_chunks = st.session_state.total_chunks
        embed_model = "all-MiniLM-L6-v2"

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">LLM Model</div>
            <div class="stat-value">{model_name}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Embedding Model</div>
            <div class="stat-value" style="font-size:0.75rem">{embed_model}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Indexed Chunks</div>
            <div class="stat-value">{total_chunks}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Top-K Retrieval</div>
            <div class="stat-value">{top_k}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="rag-header">
        <h1>🔍 RAG Chatbot</h1>
        <p>Ask anything about your uploaded documents. Answers are grounded in retrieved passages.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Render chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-msg">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        sources_html = ""
        if msg.get("sources"):
            pills = "".join(
                f'<span class="source-pill">[{i+1}] {s["source"]}</span>'
                for i, s in enumerate(msg["sources"])
            )
            sources_html = f"""
            <div class="source-container">
                <div class="source-header">📎 Retrieved sources</div>
                {pills}
            </div>"""

        st.markdown(
            f'<div class="bot-msg">🤖 {msg["content"]}{sources_html}</div>',
            unsafe_allow_html=True,
        )

        # Expandable source text
        if msg.get("sources"):
            with st.expander("View source passages"):
                for i, s in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-text">[{i}] {s["source"]} (score: {s["score"]})\n\n{s["text"]}</div>',
                        unsafe_allow_html=True,
                    )


# ── Input area ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([8, 1])
with col_input:
    user_input = st.text_input(
        "Ask a question",
        placeholder="e.g. What are the key terms and conditions?",
        label_visibility="collapsed",
        key="user_input",
    )
with col_btn:
    send = st.button("Send", use_container_width=True)

# Sample questions
st.markdown("<br>", unsafe_allow_html=True)
sample_qs = [
    "What is the main purpose of this document?",
    "What are the user's rights and responsibilities?",
    "Are there any limitations of liability mentioned?",
]
cols = st.columns(len(sample_qs))
for col, q in zip(cols, sample_qs):
    if col.button(q, key=f"sample_{q[:20]}"):
        user_input = q
        send = True


# ── Handle send ──────────────────────────────────────────────────────────────
if send and user_input.strip():
    question = user_input.strip()

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    st.markdown(
        f'<div class="user-msg">🧑 {question}</div>',
        unsafe_allow_html=True,
    )

    # Load pipeline
    try:
        pipeline = load_pipeline(model=model_name, db_path=db_path)
        pipeline.top_k = top_k
    except Exception as e:
        st.error(f"Failed to load pipeline: {e}")
        st.stop()

    # Stream response
    st.markdown('<span class="thinking-badge">⚡ Retrieving & generating …</span>', unsafe_allow_html=True)

    answer_placeholder = st.empty()
    full_answer = ""
    sources = []
    first_token = True

    for token in pipeline.stream(question):
        if token.startswith("__SOURCES__"):
            raw = token.replace("__SOURCES__", "").strip()
            try:
                sources = json.loads(raw)
            except Exception:
                sources = []
            continue

        full_answer += token
        if first_token:
            first_token = False

        answer_placeholder.markdown(
            f'<div class="bot-msg">🤖 {full_answer}▌</div>',
            unsafe_allow_html=True,
        )

    # Final render without cursor
    sources_html = ""
    if sources:
        pills = "".join(
            f'<span class="source-pill">[{i+1}] {s["source"]}</span>'
            for i, s in enumerate(sources)
        )
        sources_html = f"""
        <div class="source-container">
            <div class="source-header">📎 Retrieved sources</div>
            {pills}
        </div>"""

    answer_placeholder.markdown(
        f'<div class="bot-msg">🤖 {full_answer}{sources_html}</div>',
        unsafe_allow_html=True,
    )

    # Source passages expander
    if sources:
        with st.expander("View source passages"):
            for i, s in enumerate(sources, 1):
                st.markdown(
                    f'<div class="source-text">[{i}] {s["source"]} (score: {s["score"]})\n\n{s["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # Save to history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources}
    )
