# backend/services/rag_service.py
# --------------------------------------------------------------------
# Modern, warning-free RAG implementation using:
# - langchain_chroma (new official package)
# - sentence-transformers/all-MiniLM-L6-v2 embeddings
# - Automatic empty-DB handling
# - Cached searches & clean architecture
# --------------------------------------------------------------------

import os
from typing import List, Dict, Optional
import requests

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma                     # 🆕 new package
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------------------
# PATHS
# ----------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

_vectordb = None
_search_cache: Dict[str, List[str]] = {}

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_URL = os.getenv("SERPER_URL", "https://google.serper.dev/search")


# --------------------------------------------------------------------
# INIT VECTORSTORE
# --------------------------------------------------------------------
def init_vectorstore():
    """Initialize Chroma vectorstore. Safe for first-run & empty folders."""
    global _vectordb

    if _vectordb is not None:
        return _vectordb

    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # CPU-friendly embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # ----------------------------------------
    # 1. If DB already exists → load it
    # ----------------------------------------
    if os.listdir(DB_DIR):
        _vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
        return _vectordb

    # ----------------------------------------
    # 2. Build vector DB (first run)
    # ----------------------------------------
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    docs = loader.load()

    # No PDFs? → create empty DB
    if len(docs) == 0:
        _vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
        _vectordb.persist()
        return _vectordb

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(docs)

    # Still no chunks? → create empty DB
    if len(chunks) == 0:
        _vectordb = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
        _vectordb.persist()
        return _vectordb

    # Build the DB
    _vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    _vectordb.persist()

    return _vectordb


# --------------------------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------------------------
def search_context(query: str, k: int = 5) -> List[str]:
    """Perform Chroma search (safe, cached, no-crash)."""
    if query in _search_cache:
        return _search_cache[query]

    vectordb = init_vectorstore()

    try:
        docs = vectordb.as_retriever(search_kwargs={"k": k}).get_relevant_documents(query)
        results = [d.page_content for d in docs]
    except Exception:
        results = []

    _search_cache[query] = results
    return results


# --------------------------------------------------------------------
# SERPER WEB SEARCH
# --------------------------------------------------------------------
def web_search(query: str, num_results: int = 5) -> Optional[str]:
    """Web search using Serper API. Clean fallback on error."""
    if not SERPER_API_KEY:
        return None

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": num_results}

    try:
        resp = requests.post(SERPER_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        parts: List[str] = []

        # Organic results
        for o in data.get("organic", [])[:num_results]:
            parts.append(f"{o.get('title', '')}: {o.get('snippet', '')}")

        # Knowledge graph
        kg = data.get("knowledgeGraph") or data.get("answer")
        if isinstance(kg, dict):
            text = kg.get("description") or kg.get("snippet") or ""
            if text:
                parts.insert(0, text)

        return "\n\n".join(parts).strip()

    except Exception as e:
        return f"(serper_error: {e})"
