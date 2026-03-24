"""
Embeddings & FAISS Index Management — load, create, save with atomic writes and backups.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_DIR,
    MAX_INDEX_BACKUPS,
)


# ── Singleton Embeddings ─────────────────────────────────────────

_embeddings_instance = None


def get_embeddings():
    """Return a cached HuggingFace BGE embedding model instance."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


# ── FAISS Index Operations ───────────────────────────────────────

def get_index_path(name: str) -> Path:
    """Return the directory path for a named FAISS index."""
    return FAISS_DIR / name


def load_index(name: str):
    """
    Load a FAISS index from disk. Returns None if index doesn't exist.

    Args:
        name: one of 'problems', 'concepts', 'sessions'

    Returns:
        FAISS vectorstore or None
    """
    from langchain_community.vectorstores import FAISS

    index_path = get_index_path(name)
    index_file = index_path / "index.faiss"

    if not index_file.exists():
        return None

    try:
        store = FAISS.load_local(
            str(index_path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return store
    except Exception as e:
        print(f"⚠ Failed to load index '{name}': {e}")
        return None


def create_index(documents: List[Document], name: str):
    """
    Create a new FAISS index from documents and save to disk.

    Args:
        documents: list of LangChain Document objects
        name: index name (problems, concepts, sessions)

    Returns:
        FAISS vectorstore
    """
    from langchain_community.vectorstores import FAISS

    if not documents:
        # Create an empty index with a dummy doc, then remove it
        dummy = Document(page_content="__init__", metadata={"_dummy": True})
        store = FAISS.from_documents([dummy], get_embeddings())
        # We keep it — it'll be overwritten on first real add
        save_index(store, name)
        return store

    store = FAISS.from_documents(documents, get_embeddings())
    save_index(store, name)
    return store


def load_or_create_index(name: str, documents: List[Document] = None):
    """
    Load an existing index or create a new one.

    Args:
        name: index name
        documents: initial documents if creating new

    Returns:
        FAISS vectorstore
    """
    store = load_index(name)
    if store is not None:
        return store
    return create_index(documents or [], name)


def save_index(store, name: str):
    """
    Atomically save a FAISS index with rolling backups.

    Strategy:
      1. Write to a temp directory
      2. Rotate existing backups (keep last N)
      3. Move current index to backup
      4. Move temp to final location
    """
    index_path = get_index_path(name)
    index_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Save to temporary location
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"faiss_{name}_"))
    try:
        store.save_local(str(tmp_dir))

        # Step 2: Rotate backups
        _rotate_backups(name)

        # Step 3: If current index exists, move to backup
        if (index_path / "index.faiss").exists():
            backup_path = FAISS_DIR / f"{name}_backup_0"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(str(index_path), str(backup_path))

        # Step 4: Copy new index files to final location
        for item in tmp_dir.iterdir():
            dest = index_path / item.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(str(item), str(dest))

    finally:
        # Clean up temp directory
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def add_documents_to_index(name: str, documents: List[Document]):
    """
    Add documents to an existing index (or create if needed).

    Args:
        name: index name
        documents: documents to add

    Returns:
        Updated FAISS vectorstore
    """
    store = load_or_create_index(name)

    if documents:
        store.add_documents(documents)
        save_index(store, name)

    return store


def _rotate_backups(name: str):
    """Rotate backup directories, keeping only MAX_INDEX_BACKUPS."""
    for i in range(MAX_INDEX_BACKUPS - 1, 0, -1):
        old = FAISS_DIR / f"{name}_backup_{i - 1}"
        new = FAISS_DIR / f"{name}_backup_{i}"
        if old.exists():
            if new.exists():
                shutil.rmtree(new)
            old.rename(new)

    # Remove oldest if beyond limit
    oldest = FAISS_DIR / f"{name}_backup_{MAX_INDEX_BACKUPS}"
    if oldest.exists():
        shutil.rmtree(oldest)
