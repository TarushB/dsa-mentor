"""
Embeddings & FAISS Index Management — load, create, save with atomic writes.
Single-user installation: flat index paths (FAISS_DIR/problems/, FAISS_DIR/sessions/).
No backups, no per-user subdirectories, no concept index.
"""
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDING_MODEL_NAME, FAISS_DIR


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
    """Return the directory path for a named FAISS index (flat — no user subdir)."""
    return FAISS_DIR / name


def load_index(name: str):
    """
    Load a FAISS index from disk. Returns None if index doesn't exist.

    Args:
        name: 'problems' or 'sessions'

    Returns:
        FAISS vectorstore or None
    """
    from langchain_community.vectorstores import FAISS

    index_path = get_index_path(name)
    if not (index_path / "index.faiss").exists():
        return None

    try:
        return FAISS.load_local(
            str(index_path),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Warning: Failed to load index '{name}': {e}")
        return None


def create_index(documents: List[Document], name: str):
    """
    Create a new FAISS index from documents and save to disk.

    Args:
        documents: list of LangChain Document objects
        name:      index name ('problems' or 'sessions')

    Returns:
        FAISS vectorstore
    """
    from langchain_community.vectorstores import FAISS

    if not documents:
        print(f"[FAISS] Creating empty '{name}' index (no documents).")
        dummy = Document(page_content="__init__", metadata={"_dummy": True})
        store = FAISS.from_documents([dummy], get_embeddings())
    else:
        print(f"[FAISS] Building '{name}' index — embedding {len(documents)} documents...", flush=True)
        store = FAISS.from_documents(documents, get_embeddings())
        print(f"[FAISS] Embedding complete.")

    save_index(store, name)
    return store


def load_or_create_index(name: str, documents: List[Document] = None):
    """Load an existing index or create a new one from documents."""
    store = load_index(name)
    if store is not None:
        return store
    return create_index(documents or [], name)


def save_index(store, name: str):
    """
    Atomically save a FAISS index (write to temp → copy to final location).
    No backups are kept.
    """
    index_path = get_index_path(name)
    index_path.mkdir(parents=True, exist_ok=True)

    print(f"[FAISS] Saving '{name}' index to {index_path} ...", flush=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"faiss_{name}_"))
    try:
        store.save_local(str(tmp_dir))
        for item in tmp_dir.iterdir():
            dest = index_path / item.name
            if dest.exists():
                dest.unlink()
            shutil.copy2(str(item), str(dest))
        print(f"[FAISS] '{name}' index saved OK.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def add_documents_to_index(name: str, documents: List[Document]):
    """
    Append documents to an existing index (or create if it doesn't exist yet).

    Args:
        name:      index name
        documents: documents to add

    Returns:
        Updated FAISS vectorstore
    """
    store = load_or_create_index(name)
    if documents:
        print(f"[FAISS] Appending {len(documents)} document(s) to '{name}' index...", flush=True)
        store.add_documents(documents)
        save_index(store, name)
    return store
