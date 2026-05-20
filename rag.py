"""Core RAG logic for the News Research Tool.

Kept separate from the Streamlit UI (main.py) so it can be unit-tested and
reused by the evaluation script (eval/eval.py).

Stack (2026):
  - Generation: ChatOpenAI / ChatAnthropic / ChatOllama (provider switch)
  - Embeddings: OpenAI text-embedding-3-small
  - Retrieval: hybrid (BM25 + FAISS) via EnsembleRetriever, optional Flashrank rerank
  - Orchestration: LangChain LCEL (create_retrieval_chain + history-aware retriever)
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

# LangChain 1.x moved these legacy constructors into the `langchain_classic`
# package. (In 0.3.x they lived under `langchain.*`.)
from langchain_classic.retrievers import (
    EnsembleRetriever,
    ContextualCompressionRetriever,
)
from langchain_classic.chains import (
    create_retrieval_chain,
    create_history_aware_retriever,
)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# WebBaseLoader emits a warning (and some sites 403) without a User-Agent.
os.environ.setdefault(
    "USER_AGENT",
    "news-research-tool/2.0 (+https://github.com/Rohitn96/PortfolioProjects)",
)

EMBEDDING_MODEL = "text-embedding-3-small"

# Models exposed in the UI dropdown, grouped by provider.
PROVIDER_MODELS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
    "Anthropic": ["claude-3-5-haiku-latest", "claude-3-5-sonnet-latest"],
    "Ollama (local)": ["llama3.1", "mistral", "qwen2.5"],
}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def get_embeddings():
    """OpenAI embeddings are used for the vector store regardless of the chat
    provider (Anthropic has no embeddings API)."""
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm(provider: str, model: str, temperature: float = 0.2, streaming: bool = True):
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            streaming=streaming,
            stream_usage=True,  # surfaces token usage during streaming
        )
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, streaming=streaming)
    if provider.startswith("Ollama"):
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError(
                "Ollama support needs `pip install langchain-ollama` and a running "
                "Ollama server (https://ollama.com). It does not work on hosted "
                "Streamlit Cloud."
            ) from exc
        return ChatOllama(model=model, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def load_documents(
    urls: List[str], pdf_paths: Optional[List[str]] = None
) -> Tuple[List[Document], List[Tuple[str, str]]]:
    """Load URLs and PDFs into Documents.

    Returns (documents, errors) where errors is a list of (source, message) so
    the UI can report exactly which inputs failed instead of silently dropping
    them (a real bug in the original tool).
    """
    docs: List[Document] = []
    errors: List[Tuple[str, str]] = []

    for raw in urls:
        url = raw.strip()
        if not url:
            continue
        try:
            loaded = WebBaseLoader(url).load()
            if not loaded or not any(d.page_content.strip() for d in loaded):
                errors.append((url, "No readable text found (paywall or JS-only page?)"))
                continue
            docs.extend(loaded)
        except Exception as exc:  # noqa: BLE001 - report any loader failure
            errors.append((url, str(exc)))

    for path in pdf_paths or []:
        try:
            loaded = PyPDFLoader(path).load()
            if not loaded:
                errors.append((path, "No text extracted from PDF"))
                continue
            docs.extend(loaded)
        except Exception as exc:  # noqa: BLE001
            errors.append((path, str(exc)))

    return docs, errors


def split_documents(
    docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _get_reranker(top_n: int = 4):
    """Lazily build a Flashrank cross-encoder reranker. Returns None if the
    optional `flashrank` package is unavailable so the app still runs."""
    try:
        try:
            from langchain_community.document_compressors import FlashrankRerank
        except ImportError:
            from langchain.retrievers.document_compressors import FlashrankRerank
        return FlashrankRerank(top_n=top_n)
    except Exception:  # noqa: BLE001 - degrade gracefully
        return None


def build_retriever(
    chunks: List[Document], embeddings, k: int = 4, use_rerank: bool = False
):
    """Hybrid retriever: BM25 (keyword) + FAISS (semantic), optionally reranked.

    Returns (retriever, vectorstore). The vectorstore is returned so the caller
    can persist it with FAISS.save_local.
    """
    vectorstore = FAISS.from_documents(chunks, embeddings)
    dense = vectorstore.as_retriever(search_kwargs={"k": k})

    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = k

    retriever = EnsembleRetriever(retrievers=[sparse, dense], weights=[0.4, 0.6])

    if use_rerank:
        compressor = _get_reranker(top_n=k)
        if compressor is not None:
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
    return retriever, vectorstore


def save_index(vectorstore, path: str = "faiss_index") -> None:
    vectorstore.save_local(path)


def load_index(embeddings, path: str = "faiss_index"):
    # allow_dangerous_deserialization is required for FAISS pickle; safe here
    # because we only ever load an index this app created.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


# ---------------------------------------------------------------------------
# Chain (history-aware RAG)
# ---------------------------------------------------------------------------

CONTEXTUALIZE_PROMPT = (
    "Given the chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer it; just reformulate it "
    "if needed, otherwise return it as is."
)

ANSWER_PROMPT = (
    "You are a research assistant for news and financial articles. "
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know rather than guessing. "
    "Be concise and reference specific facts from the sources.\n\n"
    "Context:\n{context}"
)


def build_rag_chain(llm, retriever):
    """create_retrieval_chain wrapping a history-aware retriever.

    Output of .invoke/.stream is a dict with keys: input, chat_history,
    context (retrieved Documents), and answer.
    """
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware, qa_chain)


def to_lc_history(messages: List[dict]):
    """Convert [{'role','content'}, ...] UI history into LangChain messages."""
    from langchain_core.messages import AIMessage, HumanMessage

    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))
    return history


def stream_answer(chain, question: str, history, ctx_holder: dict):
    """Generator that yields answer tokens for st.write_stream and captures the
    retrieved context into ctx_holder['context'] as a side effect."""
    for chunk in chain.stream({"input": question, "chat_history": history}):
        if "context" in chunk:
            ctx_holder["context"] = chunk["context"]
        if "answer" in chunk:
            yield chunk["answer"]
