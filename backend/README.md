# RAG Codebase Assistant — Backend

A retrieval-augmented generation (RAG) system for understanding codebases. Combines hybrid retrieval (dense embeddings + BM25), call-graph expansion, cross-encoder reranking, and grounded LLM generation with strict citation validation.

## Architecture

| Module | Purpose |
|---|---|
| `chunker.py` | AST-based Python code chunking with decorator & nested-class support |
| `indexer.py` | Dense (FAISS) + Sparse (BM25) indexing, call-graph construction |
| `retriever.py` | Reciprocal Rank Fusion, cross-encoder reranking, multi-hop expansion |
| `generator.py` | LLM abstraction: Ollama (local LLaMA), Groq, Gemini, Claude |
| `pipeline.py` | Grounded prompt engineering, citation parsing & validation |
| `main.py` | FastAPI server with REST endpoints |

## Quick Start

```bash
# 1. Create venv and install
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 2. Install Ollama (recommended for CPU)
# Download from https://ollama.com, then:
ollama pull llama3.2:3b

# 3. Configure
cp .env.example .env
# Edit .env — default uses Ollama (no API key needed)

# 4. Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/index` | Clone & index a GitHub repository |
| GET | `/index/status/{job_id}` | Poll indexing job status |
| POST | `/ask` | Ask a question about the indexed codebase |
| GET | `/file` | View source file with line range |
| GET | `/stats` | Index statistics |
| GET | `/models` | List available LLM providers |
| GET | `/health` | Health check |
