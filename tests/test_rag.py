"""Offline unit tests (no network / no API key needed).

Run:  pytest -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document  # noqa: E402

import rag  # noqa: E402


def test_split_documents_produces_multiple_chunks():
    doc = Document(page_content="sentence. " * 500, metadata={"source": "t"})
    chunks = rag.split_documents([doc], chunk_size=400, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(c.metadata["source"] == "t" for c in chunks)


def test_split_overlap_respected():
    doc = Document(page_content="word " * 1000, metadata={"source": "t"})
    chunks = rag.split_documents([doc], chunk_size=500, chunk_overlap=100)
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_to_lc_history_roundtrip():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "more"},
    ]
    history = rag.to_lc_history(msgs)
    assert len(history) == 3
    assert history[0].content == "hi"
    assert history[1].content == "hello"


def test_provider_models_shape():
    assert "OpenAI" in rag.PROVIDER_MODELS
    assert all(isinstance(v, list) and v for v in rag.PROVIDER_MODELS.values())


def test_load_documents_handles_bad_url_gracefully():
    docs, errors = rag.load_documents(["not-a-real-url-zzz.invalid"])
    assert docs == []
    assert len(errors) == 1
