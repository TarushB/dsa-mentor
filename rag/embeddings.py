"""
Embeddings & FAISS Index Management — load, create, save with atomic writes aur backups.
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


# ── Singleton Embeddings — ek baar load karo, baar baar use karo ──

_embeddings_instance = None


def get_embeddings():
    """Cached HuggingFace BGE embedding model instance return karo. Baar baar load nahi karna padega."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance


# ── FAISS Index Operations — index banao, load karo, save karo ──

def get_index_path(name: str) -> Path:
    """Named FAISS index ke liye directory path return karo."""
    return FAISS_DIR / name


def load_index(name: str):
    """
    Disk se FAISS index load karo. Agar index exist nahi karta toh None dega.

    Args:
        name: 'problems', 'concepts', ya 'sessions' mein se ek

    Returns:
        FAISS vectorstore ya None
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
    Documents se naya FAISS index banao aur disk pe save karo.

    Args:
        documents: LangChain Document objects ki list
        name: index ka naam (problems, concepts, sessions)

    Returns:
        FAISS vectorstore
    """
    from langchain_community.vectorstores import FAISS

    if not documents:
        # Dummy doc se empty index banao, warna FAISS crash karega
        dummy = Document(page_content="__init__", metadata={"_dummy": True})
        store = FAISS.from_documents([dummy], get_embeddings())
        # Rehne do ise — pehli real doc add karne pe overwrite ho jaayega
        save_index(store, name)
        return store

    store = FAISS.from_documents(documents, get_embeddings())
    save_index(store, name)
    return store


def load_or_create_index(name: str, documents: List[Document] = None):
    """
    Existing index load karo ya naya bana do.

    Args:
        name: index ka naam
        documents: naya banate waqt initial documents

    Returns:
        FAISS vectorstore
    """
    store = load_index(name)
    if store is not None:
        return store
    return create_index(documents or [], name)


def save_index(store, name: str):
    """
    FAISS index ko safely save karo rolling backups ke saath.

    Strategy:
      1. Pehle temp directory mein likho
      2. Purane backups ko rotate karo (last N rakho)
      3. Current index ko backup mein move karo
      4. Temp se final location pe le aao
    """
    index_path = get_index_path(name)
    index_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Temporary location mein save karo pehle
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"faiss_{name}_"))
    try:
        store.save_local(str(tmp_dir))

        # Step 2: Backups ko rotate karo
        _rotate_backups(name)

        # Step 3: Agar current index hai toh backup mein daal do
        if (index_path / "index.faiss").exists():
            backup_path = FAISS_DIR / f"{name}_backup_0"
            if backup_path.exists():
                shutil.rmtree(backup_path)
            shutil.copytree(str(index_path), str(backup_path))

        # Step 4: Naye index files ko final jagah pe copy karo
        for item in tmp_dir.iterdir():
            dest = index_path / item.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(str(item), str(dest))

    finally:
        # Temp directory clean karo, chahe kuch bhi ho jaaye
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def add_documents_to_index(name: str, documents: List[Document]):
    """
    Existing index mein documents add karo (ya naya banao agar nahi hai).

    Args:
        name: index ka naam
        documents: add karne wale documents

    Returns:
        Updated FAISS vectorstore
    """
    store = load_or_create_index(name)

    if documents:
        store.add_documents(documents)
        save_index(store, name)

    return store


def _rotate_backups(name: str):
    """Backup directories ko rotate karo, sirf MAX_INDEX_BACKUPS rakho."""
    for i in range(MAX_INDEX_BACKUPS - 1, 0, -1):
        old = FAISS_DIR / f"{name}_backup_{i - 1}"
        new = FAISS_DIR / f"{name}_backup_{i}"
        if old.exists():
            if new.exists():
                shutil.rmtree(new)
            old.rename(new)

    # Sabse purana backup delete karo agar limit se bahar hai
    oldest = FAISS_DIR / f"{name}_backup_{MAX_INDEX_BACKUPS}"
    if oldest.exists():
        shutil.rmtree(oldest)
