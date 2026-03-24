"""
Hybrid Retrievers — metadata-filtered FAISS + BM25 ensemble for concept docs.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RETRIEVAL_K_PROBLEMS, RETRIEVAL_K_SESSIONS, RETRIEVAL_K_CONCEPTS


# ── Problem Retriever ────────────────────────────────────────────

class ProblemRetriever:
    """
    Retrieves similar past problems from the FAISS problem_index.
    Supports metadata pre-filtering by topic tags / patterns.
    """

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("problems")

    def retrieve(
        self,
        query: str,
        filter_tags: List[str] = None,
        filter_patterns: List[str] = None,
        k: int = RETRIEVAL_K_PROBLEMS,
    ) -> List[Document]:
        """
        Retrieve similar problems with optional metadata filtering.

        Args:
            query: natural language query or problem description
            filter_tags: only return docs whose tags overlap with these
            filter_patterns: only return docs whose patterns overlap with these
            k: number of results

        Returns:
            List of matching Document objects
        """
        if self.store is None:
            return []

        # Fetch more than k to allow post-filter trimming
        fetch_k = k * 4
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        # Post-filter by metadata
        if filter_tags or filter_patterns:
            results = self._filter_results(results, filter_tags, filter_patterns)

        return results[:k]

    @staticmethod
    def _filter_results(
        docs: List[Document],
        tags: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """Keep only docs that share at least one tag or pattern."""
        tag_set: Set[str] = set(t.lower() for t in (tags or []))
        pattern_set: Set[str] = set(patterns or [])
        filtered = []

        for doc in docs:
            meta = doc.metadata
            doc_tags = set(t.lower() for t in meta.get("topic_tags", []))
            doc_patterns = set(meta.get("patterns", []))

            if tag_set and doc_tags & tag_set:
                filtered.append(doc)
            elif pattern_set and doc_patterns & pattern_set:
                filtered.append(doc)
            elif not tag_set and not pattern_set:
                filtered.append(doc)

        return filtered


# ── Concept Retriever (Hybrid: BM25 + FAISS) ────────────────────

class ConceptRetriever:
    """
    Hybrid retriever combining BM25 (keyword) + FAISS (semantic) for concept docs.
    BM25 catches exact pattern-name matches; FAISS catches semantic similarity.
    """

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("concepts")
        self._bm25_retriever = None
        self._ensemble = None

    def _build_ensemble(self):
        """Lazily build the BM25 + FAISS ensemble retriever."""
        if self._ensemble is not None or self.store is None:
            return

        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            # Extract all docs from FAISS for BM25 index
            # FAISS doesn't have a direct "get all docs" — we access the docstore
            all_docs = []
            if hasattr(self.store, "docstore") and hasattr(self.store, "index_to_docstore_id"):
                for doc_id in self.store.index_to_docstore_id.values():
                    doc = self.store.docstore.search(doc_id)
                    if doc and hasattr(doc, "page_content"):
                        all_docs.append(doc)

            if all_docs:
                self._bm25_retriever = BM25Retriever.from_documents(
                    all_docs, k=RETRIEVAL_K_CONCEPTS
                )
                faiss_retriever = self.store.as_retriever(
                    search_kwargs={"k": RETRIEVAL_K_CONCEPTS}
                )
                self._ensemble = EnsembleRetriever(
                    retrievers=[self._bm25_retriever, faiss_retriever],
                    weights=[0.4, 0.6],  # slightly favor semantic
                )
        except ImportError:
            # rank_bm25 not installed — fall back to pure FAISS
            pass

    def retrieve(self, query: str, k: int = RETRIEVAL_K_CONCEPTS) -> List[Document]:
        """
        Retrieve relevant concept documents using hybrid search.

        Args:
            query: concept name or description
            k: number of results

        Returns:
            List of concept Document objects
        """
        if self.store is None:
            return []

        self._build_ensemble()

        try:
            if self._ensemble:
                results = self._ensemble.invoke(query)
            else:
                results = self.store.similarity_search(query, k=k)
            return results[:k]
        except Exception:
            return []


# ── Session Retriever ────────────────────────────────────────────

class SessionRetriever:
    """Retrieves past Q&A session entries from the FAISS session_index."""

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("sessions")

    def retrieve(
        self,
        query: str,
        filter_patterns: List[str] = None,
        k: int = RETRIEVAL_K_SESSIONS,
    ) -> List[Document]:
        """
        Retrieve relevant past sessions.

        Args:
            query: description of current struggle or problem context
            filter_patterns: optional pattern filter
            k: number of results

        Returns:
            List of session Document objects
        """
        if self.store is None:
            return []

        fetch_k = k * 3
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        # Post-filter by pattern
        if filter_patterns:
            pattern_set = set(filter_patterns)
            filtered = [
                doc for doc in results
                if set(doc.metadata.get("patterns", [])) & pattern_set
            ]
            results = filtered if filtered else results

        return results[:k]


# ── Unified Multi-Index Retrieval ────────────────────────────────

def retrieve_all(
    query: str,
    problem_tags: List[str] = None,
    problem_patterns: List[str] = None,
) -> Dict[str, List[Document]]:
    """
    Run retrieval across all three indexes and return organized results.

    Args:
        query: the user's question or problem description
        problem_tags: topic tags from the current problem
        problem_patterns: detected patterns of the current problem

    Returns:
        Dict with keys 'problems', 'concepts', 'sessions'
    """
    results = {}

    # Problem retrieval (with metadata filter)
    try:
        pr = ProblemRetriever()
        results["problems"] = pr.retrieve(query, filter_tags=problem_tags, filter_patterns=problem_patterns)
    except Exception:
        results["problems"] = []

    # Concept retrieval (hybrid BM25 + FAISS)
    try:
        cr = ConceptRetriever()
        results["concepts"] = cr.retrieve(query)
    except Exception:
        results["concepts"] = []

    # Session retrieval (semantic search with pattern filter)
    try:
        sr = SessionRetriever()
        results["sessions"] = sr.retrieve(query, filter_patterns=problem_patterns)
    except Exception:
        results["sessions"] = []

    return results
