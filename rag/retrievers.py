"""
Hybrid Retrievers — metadata-filtered FAISS + BM25 ensemble, concept docs ke liye.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RETRIEVAL_K_PROBLEMS, RETRIEVAL_K_SESSIONS, RETRIEVAL_K_CONCEPTS


# ── Problem Retriever — purane problems dhundhne ke liye ────────────

class ProblemRetriever:
    """
    FAISS problem_index se similar purane problems dhundhta hai.
    Topic tags ya patterns se metadata filtering bhi support karta hai.
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
        Similar problems dhundho, metadata filtering ke saath.

        Args:
            query: natural language query ya problem description
            filter_tags: sirf wahi docs laao jinke tags match karte hain
            filter_patterns: sirf wahi docs jinke patterns match hote hain
            k: kitne results chahiye

        Returns:
            Matching Document objects ki list
        """
        if self.store is None:
            return []

        # k se zyada fetch karo taaki filter ke baad bhi enough results hon
        fetch_k = k * 4
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        # Ab metadata ke basis pe filter karo
        if filter_tags or filter_patterns:
            results = self._filter_results(results, filter_tags, filter_patterns)

        return results[:k]

    @staticmethod
    def _filter_results(
        docs: List[Document],
        tags: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """Sirf wahi docs rakho jinke kam se kam ek tag ya pattern match karta ho."""
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


# ── Concept Retriever (Hybrid: BM25 + FAISS) — dono milke kaam karte hain ──

class ConceptRetriever:
    """
    Hybrid retriever — BM25 (keyword) + FAISS (semantic) dono use karta hai concepts ke liye.
    BM25 exact pattern-name match pakadta hai; FAISS meaning samajhke match karta hai.
    """

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("concepts")
        self._bm25_retriever = None
        self._ensemble = None

    def _build_ensemble(self):
        """Lazily BM25 + FAISS ensemble retriever banao — jab zaroorat ho tabhi."""
        if self._ensemble is not None or self.store is None:
            return

        try:
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever

            # FAISS se saare docs nikalo BM25 index banane ke liye
            # FAISS mein seedha "get all docs" nahi hota — docstore se nikalte hain
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
                    weights=[0.4, 0.6],  # semantic ko thoda zyada weight diya hai
                )
        except ImportError:
            # rank_bm25 installed nahi hai — pure FAISS pe fall back karo
            pass

    def retrieve(self, query: str, k: int = RETRIEVAL_K_CONCEPTS) -> List[Document]:
        """
        Hybrid search se relevant concept documents dhundho.

        Args:
            query: concept name ya description
            k: kitne results chahiye

        Returns:
            Concept Document objects ki list
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


# ── Session Retriever — purane Q&A sessions dhundhne ke liye ────────

class SessionRetriever:
    """FAISS session_index se purani Q&A session entries dhundhta hai."""

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
        Relevant purane sessions dhundho.

        Args:
            query: current struggle ya problem ka context
            filter_patterns: optional pattern filter
            k: kitne results chahiye

        Returns:
            Session Document objects ki list
        """
        if self.store is None:
            return []

        fetch_k = k * 3
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        # Pattern ke basis pe filter karo
        if filter_patterns:
            pattern_set = set(filter_patterns)
            filtered = [
                doc for doc in results
                if set(doc.metadata.get("patterns", [])) & pattern_set
            ]
            results = filtered if filtered else results

        return results[:k]


# ── Unified Multi-Index Retrieval — teeno indexes ek saath query karo ──

def retrieve_all(
    query: str,
    chat_history: list = None,
    problem_tags: List[str] = None,
    problem_patterns: List[str] = None,
) -> Dict[str, List[Document]]:
    """
    Teeno indexes pe retrieval chalao aur organized results do.

    Agar chat_history diya hai, toh pehle vague follow-up queries ko
    standalone queries mein rewrite karta hai using query rewriter LLM,
    phir FAISS search karta hai.

    Args:
        query: user ka question ya problem description
        chat_history: current session ke {"role": ..., "content": ...} messages
        problem_tags: current problem ke topic tags
        problem_patterns: current problem ke detected patterns

    Returns:
        Dict with keys 'problems', 'concepts', 'sessions', aur 'rewritten_query'
    """
    # Vague follow-ups ko standalone queries mein badlo
    if chat_history:
        from rag.query_rewriter import rewrite_query
        query = rewrite_query(query, chat_history)

    results = {"rewritten_query": query}

    # Problem retrieval (metadata filter ke saath)
    try:
        pr = ProblemRetriever()
        results["problems"] = pr.retrieve(query, filter_tags=problem_tags, filter_patterns=problem_patterns)
    except Exception:
        results["problems"] = []

    # Concept retrieval (hybrid BM25 + FAISS dono milke)
    try:
        cr = ConceptRetriever()
        results["concepts"] = cr.retrieve(query)
    except Exception:
        results["concepts"] = []

    # Session retrieval (semantic search + pattern filter)
    try:
        sr = SessionRetriever()
        results["sessions"] = sr.retrieve(query, filter_patterns=problem_patterns)
    except Exception:
        results["sessions"] = []

    return results

