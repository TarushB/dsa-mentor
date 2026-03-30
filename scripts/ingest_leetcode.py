"""
Ingest solved LeetCode problems from a CSV file — CLI entry point.

This is the sole data ingestion method. Pass a CSV export of your solved problems;
the script tags algorithmic patterns, builds the user profile, and indexes everything
into FAISS for RAG retrieval.

Usage:
  python scripts/ingest_leetcode.py --csv-path tests/sample_data.csv
  python scripts/ingest_leetcode.py --csv-path my_problems.csv --no-llm
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.leetcode_client import LeetCodeClient
from ingestion.problem_parser import PatternTagger, raw_to_problem_record
from ingestion.profile_builder import build_profile, save_data


def ingest_csv(csv_path: str, use_llm_tagger: bool = True):
    """Parse CSV, tag patterns, build profile, save to user_data.json + FAISS index."""
    print(f"\n=== Importing from CSV: {csv_path} ===\n")

    raw_problems = LeetCodeClient.import_from_csv(csv_path)
    if not raw_problems:
        print("  No problems found in CSV file.")
        return

    print(f"  Found {len(raw_problems)} problems in CSV.\n")

    # Pattern tagging
    tagger = PatternTagger() if use_llm_tagger else None
    if tagger and not tagger._available:
        print("  (!) LLM not available — using rule-based pattern tagging.")
        tagger = None

    problems = []
    total    = len(raw_problems)
    for i, raw in enumerate(raw_problems, 1):
        record = raw_to_problem_record(raw, tagger, index=i, total=total)
        problems.append(record)
        if i % 10 == 0 or i == total or i <= 5:
            print(f"  [{i}/{total}] {record.title} -> {record.patterns}")

    # Build and persist profile
    profile = build_profile(problems)
    save_data(profile, problems)
    print(f"\n  Saved {len(problems)} problems to user_data.json")
    print(f"  Easy: {profile.by_difficulty['EASY']}, "
          f"Medium: {profile.by_difficulty['MEDIUM']}, "
          f"Hard: {profile.by_difficulty['HARD']}")
    if profile.strong_patterns:
        print(f"  Strong patterns: {profile.strong_patterns}")
    if profile.weak_patterns:
        print(f"  Weak patterns:   {profile.weak_patterns}")

    # Build FAISS problem index
    print(f"\n  Building FAISS problem index ({len(problems)} docs)...")
    try:
        from langchain_core.documents import Document
        from rag.embeddings import create_index

        docs = [
            Document(
                page_content=p.to_document_text(),
                metadata={
                    "problem_id": p.problem_id,
                    "title":      p.title,
                    "difficulty": p.difficulty,
                    "topic_tags": p.topic_tags,
                    "patterns":   p.patterns,
                },
            )
            for p in problems
        ]
        create_index(docs, "problems")
        print(f"  FAISS index saved.")
    except Exception as e:
        print(f"  Warning: FAISS indexing failed (non-critical): {e}")

    return problems, profile


def main():
    parser = argparse.ArgumentParser(description="Ingest solved problems from CSV into DSA Mentor")
    parser.add_argument(
        "--csv-path", required=True,
        help="Path to CSV file (exported from LeetCode or manually created)"
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM pattern tagging, use rule-based fallback instead"
    )
    args = parser.parse_args()

    ingest_csv(args.csv_path, use_llm_tagger=not args.no_llm)


if __name__ == "__main__":
    main()
