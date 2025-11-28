# Contextual Chatbot — LangChain + Ollama + Streamlit (Local, runnable)

This repository contains a fully runnable local RAG chatbot demo. It runs **entirely locally**:

- Embeddings: `sentence-transformers` (all-MiniLM-L6-v2)
- Vector DB: `chromadb` (local, persistent)
- Orchestration: `langchain`
- UI: `streamlit`
- LLM: **Preferably** Ollama (local runner) via CLI if you installed it. If Ollama is NOT available, the code automatically falls back to a local `transformers` text-generation model.

## What I built

A privacy-first contextual chatbot that ingests documents, creates embeddings, stores them in Chroma, and answers user queries by retrieving relevant chunks and generating grounded answers.

## Requirements & Setup (complete, exact)

### 1) System prerequisites

- Python 3.10+
- git
- (Optional but recommended) GPU + CUDA for local transformer model acceleration

### 2) Install Ollama (optional, recommended for production-like behavior)

Ollama is a local MLC/LLM runner. It is **optional** — if present, the system uses it via the `ollama` CLI for low-latency local generation.
Follow instructions: https://ollama.com (install the Ollama app / CLI and pull a model, for example `ollama pull mistral`).

> If you don't install Ollama, the code will automatically use a local `transformers` model as fallback.

### 3) Python env and packages

```bash
git clone <your-repo-url>
cd contextual-chatbot-real
python -m venv .venv
source .venv/bin/activate    # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

### 4) Ingest sample docs (creates embeddings & Chroma DB)

```bash
python ingest.py
```

### 5) Run the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Files added in this repo (runnable, no placeholders)

- `streamlit_app.py` — Streamlit UI for chat
- `rag_pipeline.py` — Full pipeline: embedding, Chroma, retriever, LLM wrapper with Ollama CLI + fallback to Transformers
- `ingest.py` — Ingest sample docs into Chroma DB with recursive chunking
- `requirements.txt` — exact Python packages
- `prompt_templates/default_prompt.md`
- `data/sample_docs/sample.md` — small demo document
- `docs/images/architecture.png` & `docs/images/techstack.png` — simple generated PNGs
- `.gitignore`, `LICENSE`, `notebooks/demo.ipynb`

## Notes about Ollama integration

This repo uses a simple, robust strategy:

- It first tries to call the `ollama` CLI (e.g., `ollama run <model> --prompt "<text>"`) using `subprocess` and will use its stdout as the response.
- If the `ollama` CLI is not available, it falls back to a local `transformers` text-generation pipeline (Hugging Face) with `gpt2`-style models or any other local model you have downloaded.
  This ensures the repo is immediately runnable without any external paid APIs.

##

---
