import os
import shutil
import uuid
import zipfile
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from git import Repo

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from app.schemas import IndexRequest, AskRequest, AskResponse, StatsResponse
from app.generator import get_generator
from app.pipeline import answer_question_proposed, answer_question_baseline
from app.chunker import chunk_repository
from app.indexer import build_index_and_graph
from app.retriever import Retriever

REPO_DIR = "repo_data"
INDEX_DIR = "index_data"

# Globals holding our loaded modules
retriever: Retriever = None
generator = None
indexing_jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager to replace @app.on_event('startup')."""
    global retriever, generator

    logger.info("Initializing RAG Codebase Assistant...")
    os.makedirs(REPO_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load configured LLM generator
    generator = get_generator()

    # Load index and retriever if available
    retriever = Retriever(INDEX_DIR)
    if retriever.embeddings is not None:
        logger.info(f"Loaded existing index with {len(retriever.chunks)} chunks.")
    else:
        logger.info("No existing index found. Please index a repository.")

    yield
    # Shutdown logic goes here if needed


app = FastAPI(title="RAG Codebase Assistant", lifespan=lifespan)

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve static files from app/static
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return f.read()

import stat

def remove_readonly(func, path, _):
    """Clear the readonly bit and reattempt the removal (fixes Windows git folder deletion)"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def background_index_repo(repo_url: str, job_id: str):
    """Background task to clone, chunk, and index a GitHub repository."""
    global retriever
    logger.info(f"Job {job_id}: Starting indexing for {repo_url}")
    indexing_jobs[job_id] = {"status": "cloning", "progress": 0, "error": None}

    try:
        if os.path.exists(REPO_DIR):
            shutil.rmtree(REPO_DIR, onerror=remove_readonly)
        os.makedirs(REPO_DIR)

        # Clone using git, fallback to ZIP download
        try:
            logger.info(f"Job {job_id}: Attempting git clone...")
            Repo.clone_from(repo_url, REPO_DIR)
        except Exception as e:
            logger.warning(f"Job {job_id}: Git clone failed ({e}), falling back to ZIP...")
            import requests
            api_url = repo_url.replace("github.com", "api.github.com/repos") + "/zipball"
            r = requests.get(api_url, stream=True)
            r.raise_for_status()
            zip_path = os.path.join(REPO_DIR, "repo.zip")
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(REPO_DIR)
            os.remove(zip_path)
            
            # The inner folder is usually named like user-repo-hash
            inner_dirs = [os.path.join(REPO_DIR, d) for d in os.listdir(REPO_DIR) if os.path.isdir(os.path.join(REPO_DIR, d))]
            if len(inner_dirs) == 1:
                inner_dir = inner_dirs[0]
                for item in os.listdir(inner_dir):
                    shutil.move(os.path.join(inner_dir, item), REPO_DIR)
                shutil.rmtree(inner_dir)

        indexing_jobs[job_id] = {"status": "chunking", "progress": 30, "error": None}
        logger.info(f"Job {job_id}: Cloning complete. Starting chunking...")

        chunks = chunk_repository(REPO_DIR)

        indexing_jobs[job_id] = {"status": "indexing", "progress": 60, "error": None}
        logger.info(f"Job {job_id}: Chunking complete. Starting indexing...")

        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR)

        build_index_and_graph(chunks, REPO_DIR, INDEX_DIR)

        # Reload retriever with new data
        retriever = Retriever(INDEX_DIR)

        indexing_jobs[job_id] = {"status": "completed", "progress": 100, "error": None}
        logger.info(f"Job {job_id}: Indexing completed successfully.")

    except Exception as e:
        logger.exception(f"Job {job_id}: Indexing failed")
        indexing_jobs[job_id] = {"status": "failed", "progress": 0, "error": str(e)}


@app.post("/index")
async def index_repo(req: IndexRequest, bg_tasks: BackgroundTasks):
    """Start an indexing job for a GitHub repository."""
    job_id = str(uuid.uuid4())
    bg_tasks.add_task(background_index_repo, req.repo_url, job_id)
    return {"job_id": job_id, "message": "Indexing started"}


@app.get("/index/status/{job_id}")
async def get_index_status(job_id: str):
    """Check the status of an indexing job."""
    if job_id not in indexing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return indexing_jobs[job_id]


@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Ask a question about the indexed codebase."""
    if not retriever or retriever.embeddings is None:
        raise HTTPException(
            status_code=400,
            detail="No repository is indexed. Please index a repository first.",
        )

    logger.info(f"Answering question in {req.mode} mode: {req.question}")

    if req.mode == "proposed":
        res = answer_question_proposed(req.question, retriever, generator, k=req.k)
    else:
        res = answer_question_baseline(req.question, retriever, generator, k=req.k)

    return res


@app.get("/file")
async def get_file(filepath: str, start_line: int = Query(None), end_line: int = Query(None)):
    """Fetch lines of code from a file in the indexed repository."""
    full_path = os.path.join(REPO_DIR, filepath)
    if not os.path.abspath(full_path).startswith(os.path.abspath(REPO_DIR)):
        raise HTTPException(status_code=403, detail="Path traversal attempt")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")

    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    start = max(1, start_line) if start_line else 1
    end = min(len(lines), end_line) if end_line else len(lines)

    selected = lines[start - 1:end]
    return {"content": "".join(selected), "total_lines": len(lines)}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the indexed repository and backend state."""
    total_chunks = 0
    total_files = 0
    index_loaded = False

    if retriever and retriever.embeddings is not None:
        index_loaded = True
        total_chunks = len(retriever.chunks)
        total_files = len(set(c["file"] for c in retriever.chunks))

    return StatsResponse(
        total_chunks=total_chunks,
        total_files=total_files,
        index_loaded=index_loaded,
        generator_type=os.getenv("GENERATOR", "ollama"),
    )


@app.get("/models")
async def get_models():
    """List available LLM generator options."""
    return {"models": ["ollama", "groq", "gemini", "claude", "mock"]}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
