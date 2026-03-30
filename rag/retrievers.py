"""
Retrievers — FAISS-based retrieval for problems and sessions.
Single-user, two indices only: 'problems' and 'sessions'.
Concept index has been removed.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RETRIEVAL_K_PROBLEMS, RETRIEVAL_K_SESSIONS


#  Problem Retriever 

class ProblemRetriever:
    """
    Retrieves similar past problems from the FAISS problems index.
    Supports metadata pre-filtering by topic tags / patterns.

    Fallback: auto-builds from user_data.json if the index is missing.
    """

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("problems")

        # Fallback: auto-build from user_data.json if index is missing
        if self.store is None:
            self.store = self._try_build_from_json()

    @staticmethod
    def _try_build_from_json():
        """Build a FAISS problems index from user_data.json if data exists."""
        try:
            from ingestion.profile_builder import load_problems
            from ingestion.problem_parser import ProblemRecord
            from rag.embeddings import create_index

            raw_problems = load_problems()
            if not raw_problems:
                return None

            documents = []
            for p in raw_problems:
                if isinstance(p, dict):
                    try:
                        record = ProblemRecord(**p)
                    except Exception:
                        continue
                else:
                    record = p

                doc = Document(
                    page_content=record.to_document_text(),
                    metadata={
                        "problem_id": record.problem_id,
                        "title":      record.title,
                        "difficulty": record.difficulty,
                        "topic_tags": record.topic_tags,
                        "patterns":   record.patterns,
                    },
                )
                documents.append(doc)

            if documents:
                print(f"  [RAG] Auto-building problems index ({len(documents)} docs)...")
                return create_index(documents, "problems")
        except Exception as e:
            print(f"  [RAG] Failed to auto-build problems index: {e}")
        return None

    def retrieve(
        self,
        query: str,
        filter_tags: List[str] = None,
        filter_patterns: List[str] = None,
        k: int = RETRIEVAL_K_PROBLEMS,
    ) -> List[Document]:
        if self.store is None:
            return []

        fetch_k = k * 4
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        if filter_tags or filter_patterns:
            results = self._filter_results(results, filter_tags, filter_patterns)

        return results[:k]

    @staticmethod
    def _filter_results(
        docs: List[Document],
        tags: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        tag_set:     Set[str] = set(t.lower() for t in (tags or []))
        pattern_set: Set[str] = set(patterns or [])
        filtered = []

        for doc in docs:
            meta        = doc.metadata
            doc_tags    = set(t.lower() for t in meta.get("topic_tags", []))
            doc_patterns = set(meta.get("patterns", []))

            if tag_set and doc_tags & tag_set:
                filtered.append(doc)
            elif pattern_set and doc_patterns & pattern_set:
                filtered.append(doc)
            elif not tag_set and not pattern_set:
                filtered.append(doc)

        return filtered


#  Session Retriever 

class SessionRetriever:
    """Retrieves past Q&A session entries from the FAISS sessions index."""

    def __init__(self):
        from rag.embeddings import load_index
        self.store = load_index("sessions")

    def retrieve(
        self,
        query: str,
        filter_patterns: List[str] = None,
        k: int = RETRIEVAL_K_SESSIONS,
    ) -> List[Document]:
        if self.store is None:
            return []

        fetch_k = k * 3
        try:
            results = self.store.similarity_search(query, k=fetch_k)
        except Exception:
            return []

        if filter_patterns:
            pattern_set = set(filter_patterns)
            filtered = [
                doc for doc in results
                if set(doc.metadata.get("patterns", [])) & pattern_set
            ]
            results = filtered if filtered else results

        return results[:k]


#  Unified Retrieval 

def retrieve_all(
    query: str,
    chat_history: list = None,
    problem_tags: List[str] = None,
    problem_patterns: List[str] = None,
    skip_rewrite: bool = False,
) -> Dict[str, List[Document]]:
    """
    Run retrieval across problems and sessions indexes.

    Args:
        query:            the user's question or problem description
        chat_history:     list of {"role": ..., "content": ...} from current session
        problem_tags:     topic tags from the current problem (for filtering)
        problem_patterns: detected patterns of the current problem (for filtering)
        skip_rewrite:     True if query is already final (skip query rewriting)

    Returns:
        Dict with keys 'problems', 'sessions', and 'rewritten_query'
    """
    if chat_history and not skip_rewrite:
        from rag.query_rewriter import rewrite_query
        query = rewrite_query(query, chat_history)

    results = {"rewritten_query": query}

    try:
        pr = ProblemRetriever()
        results["problems"] = pr.retrieve(
            query,
            filter_tags=problem_tags,
            filter_patterns=problem_patterns,
        )
    except Exception as e:
        print(f"  [RAG] Problem retrieval failed: {e}")
        results["problems"] = []

    try:
        sr = SessionRetriever()
        results["sessions"] = sr.retrieve(query, filter_patterns=problem_patterns)
    except Exception as e:
        print(f"  [RAG] Session retrieval failed: {e}")
        results["sessions"] = []

    return results
