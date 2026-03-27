"""
Ingest LeetCode data — CLI entry point.

Usage:
  python scripts/ingest_leetcode.py --mode username --username <tumhara_username>    # public, no auth
  python scripts/ingest_leetcode.py --mode graphql  --username <tumhara_username>    # cookies chahiye
  python scripts/ingest_leetcode.py --mode csv      --csv-path data/problems.csv  # manual import
"""
import argparse
import json
import sys
from pathlib import Path

# Project root ko path mein add karo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from ingestion.leetcode_client import LeetCodeClient
from ingestion.problem_parser import PatternTagger, raw_to_problem_record, ProblemRecord
from ingestion.profile_builder import build_profile, save_profile


def _process_and_save(raw_problems, username, use_llm_tagger):
    """Shared logic: patterns tag karo, profile banao, sab kuch save karo."""
    # Pattern tagger initialize karo
    tagger = PatternTagger() if use_llm_tagger else None
    if tagger and not tagger._available:
        print("  (!) Ollama not available -- using rule-based pattern tagging.")
        tagger = None

    # Convert aur tag karo
    problems = []
    for i, raw in enumerate(raw_problems, 1):
        title = raw.get("title", "?")
        if i % 10 == 0 or i == len(raw_problems) or i <= 5:
            print(f"  [{i}/{len(raw_problems)}] {title}...", end=" ")
            record = raw_to_problem_record(raw, tagger)
            problems.append(record)
            print(f"-> {record.patterns}")
        else:
            record = raw_to_problem_record(raw, tagger)
            problems.append(record)

    # Profile banao aur save karo
    profile = build_profile(problems, username=username)
    save_path = save_profile(profile)
    print(f"\n  Profile saved to {save_path}")
    print(f"  Total: {profile.total_solved} problems")
    print(f"  Easy: {profile.by_difficulty['EASY']}, "
          f"Medium: {profile.by_difficulty['MEDIUM']}, "
          f"Hard: {profile.by_difficulty['HARD']}")
    if profile.strong_patterns:
        print(f"  Strong patterns: {profile.strong_patterns}")
    if profile.weak_patterns:
        print(f"  Weak patterns: {profile.weak_patterns}")

    # Raw problem records bhi save karo
    records_path = DATA_DIR / "solved_problems.json"
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump([p.model_dump(mode="json") for p in problems], f, indent=2, default=str)
    print(f"  Problems saved to {records_path}")

    return problems, profile


def ingest_by_username(username: str, use_llm_tagger: bool = True):
    """Sirf username se problems fetch karo (PUBLIC, cookies nahi chahiye)."""
    print(f"\n=== Fetching LeetCode data for '{username}' (public mode) ===\n")
    client = LeetCodeClient(use_auth=False)

    # Pehle user stats dikhao
    stats = client.get_user_stats(username)
    if stats and stats.get("submitStatsGlobal"):
        ac_stats = stats["submitStatsGlobal"].get("acSubmissionNum", [])
        for s in ac_stats:
            if s["difficulty"] == "All":
                print(f"  Profile found: {stats.get('username', username)} ({s['count']} problems solved)")
                break

    # Saare solved problems fetch karo
    raw_problems = client.fetch_solved_problems_public(username)
    print(f"\n  Retrieved {len(raw_problems)} solved problems with metadata.\n")

    if not raw_problems:
        print("  ERROR: No problems found. Check if the username is correct and the profile is public.")
        return

    return _process_and_save(raw_problems, username, use_llm_tagger)


def ingest_graphql(username: str, use_llm_tagger: bool = True):
    """Auth cookies se problems fetch karo (GraphQL with session)."""
    print(f"\n=== Fetching LeetCode data for '{username}' (authenticated mode) ===\n")
    client = LeetCodeClient(use_auth=True)

    if not client.use_auth:
        print("  WARNING: No LEETCODE_SESSION cookie found in .env")
        print("  Falling back to public mode...\n")
        return ingest_by_username(username, use_llm_tagger)

    raw_problems = client.get_user_solved_list_auth(username)
    print(f"\n  Retrieved {len(raw_problems)} solved problems.\n")

    if not raw_problems:
        print("  No problems found. Trying public mode instead...\n")
        return ingest_by_username(username, use_llm_tagger)

    return _process_and_save(raw_problems, username, use_llm_tagger)


def ingest_csv(csv_path: str, username: str = "csv_user", use_llm_tagger: bool = True):
    """CSV file se problems import karo."""
    print(f"\n=== Importing from CSV: {csv_path} ===\n")

    raw_problems = LeetCodeClient.import_from_csv(csv_path)
    print(f"  Found {len(raw_problems)} problems in CSV.\n")

    if not raw_problems:
        print("  No problems found in CSV file.")
        return

    return _process_and_save(raw_problems, username, use_llm_tagger)


def main():
    parser = argparse.ArgumentParser(description="Ingest LeetCode data into DSA Mentor")
    parser.add_argument("--mode", choices=["username", "graphql", "csv"], required=True,
                        help="'username' (public, no auth), 'graphql' (needs cookies), 'csv' (file)")
    parser.add_argument("--username", default="", help="LeetCode username")
    parser.add_argument("--csv-path", default="", help="Path to CSV file (for csv mode)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip Ollama LLM tagging, use rule-based fallback only")

    args = parser.parse_args()

    if args.mode == "username":
        if not args.username:
            print("ERROR: --username is required for username mode")
            sys.exit(1)
        ingest_by_username(args.username, use_llm_tagger=not args.no_llm)

    elif args.mode == "graphql":
        if not args.username:
            print("ERROR: --username is required for graphql mode")
            sys.exit(1)
        ingest_graphql(args.username, use_llm_tagger=not args.no_llm)

    elif args.mode == "csv":
        if not args.csv_path:
            print("ERROR: --csv-path is required for csv mode")
            sys.exit(1)
        ingest_csv(args.csv_path, username=args.username or "csv_user",
                   use_llm_tagger=not args.no_llm)


if __name__ == "__main__":
    main()
