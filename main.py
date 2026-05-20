"""News Research Tool - a modern RAG app for Q&A over news/finance articles.

Run locally:   streamlit run main.py
Deploy:        Streamlit Community Cloud (see README.md)
"""
from __future__ import annotations

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

import rag

load_dotenv()

st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")


# ---------------------------------------------------------------------------
# Secrets / API keys (prefer st.secrets, fall back to environment)
# ---------------------------------------------------------------------------
def _get_secret(name: str):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # st.secrets raises if no secrets file exists locally
        pass
    return os.environ.get(name)


for _key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    _val = _get_secret(_key)
    if _val:
        os.environ[_key] = _val


# ---------------------------------------------------------------------------
# Cached ingestion (re-runs only when inputs change)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_and_split(urls_tuple, pdf_blobs, chunk_size, chunk_overlap):
    """Load + split. Cached on the URL list, PDF bytes, and chunk params so
    re-running the app doesn't re-fetch unchanged sources."""
    pdf_paths = []
    tmp_files = []
    for name, data in pdf_blobs:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.write(data)
        tf.flush()
        tf.close()
        pdf_paths.append(tf.name)
        tmp_files.append(tf.name)
    try:
        docs, errors = rag.load_documents(list(urls_tuple), pdf_paths)
        chunks = rag.split_documents(docs, chunk_size, chunk_overlap)
    finally:
        for p in tmp_files:
            try:
                os.unlink(p)
            except OSError:
                pass
    return chunks, errors


# ---------------------------------------------------------------------------
# Sidebar - configuration
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

provider = st.sidebar.selectbox("Provider", list(rag.PROVIDER_MODELS.keys()))
model = st.sidebar.selectbox("Model", rag.PROVIDER_MODELS[provider])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

with st.sidebar.expander("Retrieval settings", expanded=False):
    top_k = st.slider("Top-k chunks", 2, 10, 4)
    chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 50)
    use_rerank = st.checkbox(
        "Rerank results (Flashrank)",
        value=False,
        help="Cross-encoder reranking for better precision. Needs the optional "
        "`flashrank` package; downloads a small model on first use.",
    )

st.sidebar.divider()
st.sidebar.subheader("📥 Sources")
url_text = st.sidebar.text_area(
    "Article URLs (one per line)",
    height=140,
    placeholder="https://example.com/article-1\nhttps://example.com/article-2",
)
pdf_uploads = st.sidebar.file_uploader(
    "or upload PDFs", type=["pdf"], accept_multiple_files=True
)

col_a, col_b = st.sidebar.columns(2)
process_clicked = col_a.button("Process", type="primary", use_container_width=True)
reset_clicked = col_b.button("Reset", use_container_width=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if reset_clicked:
    st.session_state.messages = []
    st.session_state.retriever = None
    cached_load_and_split.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Process sources
# ---------------------------------------------------------------------------
def _have_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


if process_clicked:
    if not _have_openai_key():
        st.sidebar.error("OPENAI_API_KEY missing (embeddings require it). Add it to secrets/.env.")
    else:
        urls = [u for u in url_text.splitlines() if u.strip()]
        pdf_blobs = [(f.name, f.getvalue()) for f in (pdf_uploads or [])]
        if not urls and not pdf_blobs:
            st.sidebar.warning("Add at least one URL or PDF.")
        else:
            with st.spinner("Loading and indexing sources..."):
                chunks, errors = cached_load_and_split(
                    tuple(urls), tuple(pdf_blobs), chunk_size, chunk_overlap
                )
                if errors:
                    for src, msg in errors:
                        st.sidebar.warning(f"⚠️ {src}: {msg}")
                if not chunks:
                    st.sidebar.error("No content could be indexed from the given sources.")
                else:
                    embeddings = rag.get_embeddings()
                    retriever, _ = rag.build_retriever(
                        chunks, embeddings, k=top_k, use_rerank=use_rerank
                    )
                    st.session_state.retriever = retriever
                    st.session_state.messages = []
                    st.sidebar.success(f"Indexed {len(chunks)} chunks. Ask away!")


# ---------------------------------------------------------------------------
# Main panel - chat
# ---------------------------------------------------------------------------
st.title("📈 News Research Tool")
st.caption(
    "Hybrid RAG (BM25 + FAISS) over news & finance articles with conversational "
    "follow-ups and cited sources."
)

if st.session_state.retriever is None:
    st.info(
        "👈 Add article URLs or PDFs in the sidebar and click **Process** to start.",
        icon="💡",
    )

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question about the articles...",
    disabled=st.session_state.retriever is None,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            llm = rag.get_llm(provider, model, temperature=temperature, streaming=True)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

        chain = rag.build_rag_chain(llm, st.session_state.retriever)
        history = rag.to_lc_history(st.session_state.messages[:-1])
        ctx_holder: dict = {}

        try:
            from langchain_community.callbacks import get_openai_callback

            with get_openai_callback() as cb:
                answer = st.write_stream(
                    rag.stream_answer(chain, question, history, ctx_holder)
                )
            if cb.total_tokens:
                st.caption(
                    f"🔢 {cb.total_tokens} tokens • 💵 est. ${cb.total_cost:.4f}"
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Error while answering: {exc}")
            st.stop()

        # Sources (deduped by source URL/file)
        context_docs = ctx_holder.get("context", [])
        if context_docs:
            seen = set()
            with st.expander(f"📚 Sources ({len(context_docs)} chunks retrieved)"):
                for doc in context_docs:
                    src = doc.metadata.get("source", "unknown")
                    if src not in seen:
                        seen.add(src)
                        if str(src).startswith("http"):
                            st.markdown(f"- [{src}]({src})")
                        else:
                            st.markdown(f"- `{src}`")
                    snippet = doc.page_content.strip().replace("\n", " ")[:240]
                    st.caption(f"…{snippet}…")

    st.session_state.messages.append({"role": "assistant", "content": answer})
