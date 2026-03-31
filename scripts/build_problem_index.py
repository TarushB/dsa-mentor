"""
Build Problem Index — solved_problems.json ko FAISS problem index mein convert karta hai.

Usage:
  python scripts/build_problem_index.py
"""
import json
import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from rag.embeddings import create_index


def build_problem_index() -> int:
    """
    solved_problems.json padho aur FAISS problem index banao.

    Returns:
        Kitne documents index hue
    """
    records_path = Path(DATA_DIR) / "solved_problems.json"
    if not records_path.exists():
        print(" No solved_problems.json found. Run ingest_leetcode.py first.")
        return 0

    with open(records_path, "r", encoding="utf-8") as f:
        problems = json.load(f)

    print(f" Found {len(problems)} problems to index.\n")

    documents = []
    for p in problems:
        # Document ka text banao (ProblemRecord.to_document_text wala format)
        lines = [
            f"Problem: {p['title']} [{p.get('difficulty', 'MEDIUM').upper()}]",
            f"Patterns: {', '.join(p.get('patterns', []))}",
            f"Tags: {', '.join(p.get('topic_tags', []))}",
        ]
        if p.get("solved_at"):
            lines.append(f"Solved: {p['solved_at']} | Attempts: {p.get('num_attempts', 1)}")
        if p.get("personal_notes"):
            lines.append(f"Notes: {p['personal_notes']}")

        doc = Document(
            page_content="\n".join(lines),
            metadata={
                "problem_id": p.get("problem_id", ""),
                "title": p.get("title", ""),
                "difficulty": p.get("difficulty", "MEDIUM"),
                "topic_tags": p.get("topic_tags", []),
                "patterns": p.get("patterns", []),
            },
        )
        documents.append(doc)
        print(f"  + {p['title']}")

    print(f"\n Building FAISS problem index with {len(documents)} documents...")
    create_index(documents, "problems")
    print(" Problem index saved.")

    return len(documents)


if __name__ == "__main__":
    print("\n Building Problem FAISS Index\n")
    count = build_problem_index()
    print(f"\n Done! Indexed {count} problems.")
