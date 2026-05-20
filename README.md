---
title: News Research Tool
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: main.py
python_version: "3.12"
pinned: false
---

# News Research Tool 📈

A modern **retrieval-augmented generation (RAG)** app for asking questions about
news and financial articles. Paste article URLs (or upload PDFs), and chat with
the content — with conversational follow-ups, cited sources, and live streaming
answers.

> Rebuilt in 2026 from a 2023-era LangChain prototype: migrated to the current
> LCEL API, hybrid retrieval, multi-provider models, evaluation, and tests.

![News Research Tool Screenshot](NewsResearchTool.png)

## Features

- **Hybrid retrieval** — combines BM25 (keyword) and FAISS (semantic) via an
  `EnsembleRetriever`, with optional **Flashrank cross-encoder reranking**.
- **Conversational** — history-aware retriever rewrites follow-up questions, so
  "what about its revenue?" works after an earlier question.
- **Cited sources** — every answer shows the exact chunks and source links used.
- **Streaming answers** — tokens render live (`st.write_stream`).
- **Multi-provider** — switch between **OpenAI**, **Anthropic**, and local
  **Ollama** models from the sidebar.
- **Multiple inputs** — many URLs (one per line) and/or PDF uploads.
- **Cost visibility** — shows token usage and estimated cost per answer (OpenAI).
- **Evaluated** — RAGAS metrics (faithfulness, answer relevancy, context
  precision/recall) in `eval/`.

## Architecture

```
URLs / PDFs ─► load (WebBaseLoader / PyPDFLoader)
            ─► split (RecursiveCharacterTextSplitter)
            ─► embed (text-embedding-3-small) ─► FAISS  ┐
            ─► BM25 index                                ├─► EnsembleRetriever ─►(optional Flashrank rerank)
                                                         ┘
question + chat history ─► history-aware retriever ─► stuff-documents chain ─► LLM ─► streamed answer + sources
```

![SystemDiagram](System.png)

## Project layout

| File | Purpose |
|------|---------|
| `main.py` | Streamlit UI (chat, sidebar config, sources) |
| `rag.py` | Core RAG logic (models, ingestion, retrieval, chain) — unit-testable |
| `requirements.txt` | App dependencies |
| `Dockerfile` | Container build |
| `eval/` | RAGAS evaluation script + dataset |
| `tests/` | Offline unit tests (`pytest`) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# provide your key (either a .env file or .streamlit/secrets.toml)
cp .env.example .env            # then edit it
streamlit run main.py
```

Then in the browser: paste article URLs in the sidebar → **Process** → ask
questions.

## Deploy (Streamlit Community Cloud)

1. Push this folder to GitHub.
2. Go to <https://share.streamlit.io> → **Create app** → pick the repo.
3. **Main file path:** `main.py`
4. **Advanced settings → Python version:** 3.12
5. **Advanced settings → Secrets:**
   ```toml
   OPENAI_API_KEY = "sk-..."
   # ANTHROPIC_API_KEY = "sk-ant-..."   # optional
   ```
6. **Deploy.**

## Run the tests

```bash
pip install pytest
pytest -q          # offline, no API key required
```

## Run the evaluation

```bash
pip install -r eval/requirements-eval.txt
export OPENAI_API_KEY=sk-...
python eval/eval.py     # prints RAGAS scores, writes eval/eval_results.csv
```

## Docker

```bash
docker build -t news-research-tool .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... news-research-tool
```

## Notes & limits

- Embeddings always use OpenAI (`text-embedding-3-small`), so an
  `OPENAI_API_KEY` is required even when chatting via Anthropic.
- The Ollama provider only works locally (needs a running Ollama server) and is
  not available on hosted Streamlit Cloud.
- Streamlit Cloud storage is ephemeral — the FAISS index lives for the session
  and is rebuilt after a restart.
- Some sites block scraping or are JS-only; those URLs are reported as warnings
  rather than failing silently.
