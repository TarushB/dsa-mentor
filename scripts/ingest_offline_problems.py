"""
Ingest offline LeetCode problems from output_leetcode_questions.txt
into data/offline_problems.json (for fast, API-free lookups).

Format of the source file:
  Leetcode Offline

  ------------------------------
  Problem Title
  ------------------------------
  Description text...

  ------------------------------
  ------------------------------
  Next Problem Title
  ------------------------------
  ...

Usage:
  python scripts/ingest_offline_problems.py [--path /custom/path.txt]
  python scripts/ingest_offline_problems.py --index   # also rebuild FAISS descriptions index
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, OFFLINE_PROBLEMS_PATH, LEETCODE_QUESTIONS_PATH, DESCRIPTIONS_DIR

# Default source is now inside the data/ folder (universal path, no hardcoded user dir)
DEFAULT_SOURCE = LEETCODE_QUESTIONS_PATH

SEPARATOR = "-" * 30


def _slugify(text: str) -> str:
    """Convert title to URL slug."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def parse_offline_file(file_path: Path) -> list[dict]:
    """
    Parse the offline LeetCode problems text file.

    Format:
        Leetcode Offline
        ------------------------------
        Title
        ------------------------------
        Description...
        ------------------------------
        ------------------------------
        Next Title
        ------------------------------
        Description...

    Strategy: use regex to find all SEP → title → SEP → description blocks.
    The title is always a SHORT chunk (≤ 3 lines, ≤ 60 chars/line) sandwiched
    between two SEP lines. Descriptions are the text that follows.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    sep = re.escape(SEPARATOR)

    # Find all title blocks: SEP\n<title text>\nSEP
    # Captures: (title, description) where description is everything up to next SEP
    pattern = re.compile(
        rf"{sep}\s*\n"               # opening separator
        rf"([ \t]*\S[^\n]*)\s*\n"   # title line (non-empty, not just whitespace)
        rf"{sep}\s*\n"               # closing separator
        rf"(.*?)"                    # description (lazy)
        rf"(?={sep}|$)",             # stop before next separator or end of file
        re.DOTALL
    )

    problems = []
    for m in pattern.finditer(raw):
        title_raw = m.group(1).strip()
        description = m.group(2).strip()

        # Skip header line and very short/invalid titles
        if not title_raw or title_raw.lower() == "leetcode offline":
            continue
        # Title must be ≤ 60 chars (problem names are short)
        if len(title_raw) > 70:
            continue

        problems.append({
            "title": title_raw,
            "slug": _slugify(title_raw),
            "description": description,
            "difficulty": _infer_difficulty(description),
            "tags": _infer_tags(title_raw, description),
            "source": "offline",
        })

    return problems


def _infer_difficulty(description: str) -> str:
    """Rough difficulty inference from description keywords."""
    desc_lower = description.lower()
    hard_hints = ["o(log n)", "o(1) space", "nlogn", "topological", "segment tree",
                  "suffix", "manacher", "kmp", "floyd", "dijkstra", "palindrome",
                  "regular expression", "wildcard", "edit distance"]
    easy_hints = ["return the", "find if", "reverse a string", "fizz", "count",
                  "sum of", "maximum depth", "single number", "missing number"]
    for h in hard_hints:
        if h in desc_lower:
            return "Hard"
    for e in easy_hints:
        if e in desc_lower:
            return "Easy"
    return "Medium"


def _infer_tags(title: str, description: str) -> list[str]:
    """Infer topic tags from title and description text."""
    text = (title + " " + description).lower()
    tag_map = [
        ("Array",              ["array", "subarray", "element", "index"]),
        ("Hash Table",         ["hash", "map", "dictionary", "hashmap"]),
        ("Dynamic Programming",["dp", "dynamic programming", "memo", "bottom-up", "top-down",
                                 "coin", "knapsack", "palindrome", "subsequence"]),
        ("String",             ["string", "substring", "character", "palindrome", "anagram"]),
        ("Linked List",        ["linked list", "node", "next pointer", "-> "]),
        ("Binary Search",      ["binary search", "sorted array", "log n", "mid"]),
        ("Tree",               ["tree", "node", "root", "leaf", "height", "depth", "bst",
                                 "binary tree"]),
        ("Graph",              ["graph", "vertex", "edge", "neighbor", "bfs", "dfs"]),
        ("Two Pointers",       ["two pointer", "left", "right", "slow", "fast"]),
        ("Sliding Window",     ["sliding window", "window", "subarray"]),
        ("Sorting",            ["sort", "sorted", "merge sort", "quick sort"]),
        ("Backtracking",       ["backtrack", "permutation", "combination", "subset"]),
        ("Greedy",             ["greedy", "minimum", "maximum", "optimal"]),
        ("Math",               ["math", "number", "digit", "prime", "palindrome", "mod"]),
        ("Bit Manipulation",   ["bit", "xor", "bitwise", "binary"]),
        ("Stack",              ["stack", "push", "pop", "monotonic"]),
        ("Queue",              ["queue", "dequeue", "enqueue", "circular"]),
        ("Heap",               ["heap", "priority queue", "kth", "k largest"]),
    ]
    found = []
    for tag, keywords in tag_map:
        if any(kw in text for kw in keywords):
            found.append(tag)
    return found[:5]  # limit to 5 most relevant


def _extract_number_from_title(title: str):
    """
    Split titles like '3874. Valid Subarrays With Exactly One Peak'
    into (number_str, clean_title).  Returns (None, title) if no prefix.
    """
    m = re.match(r'^(\d+)\.\s*(.*)', title.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return None, title


def parse_description_dir(descriptions_dir: Path) -> list[dict]:
    """
    Parse all questions_batch_*.txt files inside *descriptions_dir*.

    Each file uses the same SEP / title / SEP / description format as the
    single-file parser; titles typically carry a leading number
    (e.g. '69. Sqrt(x)') which is extracted and stored as 'number'.
    """
    batch_files = sorted(descriptions_dir.glob("questions_batch_*.txt"))
    all_problems: list[dict] = []

    for batch_file in batch_files:
        try:
            problems = parse_offline_file(batch_file)
        except Exception as exc:
            print(f"  Warning: could not parse {batch_file.name}: {exc}")
            continue

        for p in problems:
            num, clean_title = _extract_number_from_title(p["title"])
            if num:
                p["number"] = num
                p["title"]  = clean_title
                p["slug"]   = _slugify(clean_title)

        all_problems.extend(problems)

    return all_problems


def save_offline_problems(problems: list[dict]):
    """Save parsed problems to OFFLINE_PROBLEMS_PATH as JSON keyed by slug."""
    keyed = {}
    for p in problems:
        slug = p["slug"]
        if slug:
            keyed[slug] = p

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OFFLINE_PROBLEMS_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(keyed, f, indent=2)
    tmp.replace(OFFLINE_PROBLEMS_PATH)
    return keyed


def build_faiss_index(keyed: dict):
    """Build a shared FAISS 'descriptions' index from offline problems."""
    from langchain_core.documents import Document
    from rag.embeddings import create_index

    docs = []
    for slug, p in keyed.items():
        content = f"{p['title']}\n\n{p['description'][:1000]}"
        docs.append(Document(
            page_content=content,
            metadata={
                "title": p["title"],
                "slug": slug,
                "difficulty": p["difficulty"],
                "tags": ", ".join(p.get("tags", [])),
                "source": "offline",
            }
        ))
    if docs:
        store = create_index(docs, "descriptions")
        print(f"  Built FAISS 'descriptions' index with {len(docs)} problems.")
        return store
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_SOURCE),
                        help="Path to the single .txt file (default: data/output_leetcode_questions.txt)")
    parser.add_argument("--descriptions-dir", default=str(DESCRIPTIONS_DIR),
                        help="Directory with questions_batch_*.txt files (default: data/descriptions/)")
    parser.add_argument("--index", action="store_true",
                        help="Also build FAISS descriptions index")
    args = parser.parse_args()

    all_problems: list[dict] = []

    # 1. Single legacy file (optional)
    source = Path(args.path)
    if source.exists():
        print(f"Parsing single file: {source}")
        problems = parse_offline_file(source)
        print(f"  Found {len(problems)} problems.")
        all_problems.extend(problems)
    else:
        print(f"Single file not found, skipping: {source}")

    # 2. Batch description files
    desc_dir = Path(args.descriptions_dir)
    if desc_dir.exists():
        print(f"\nParsing descriptions directory: {desc_dir}")
        desc_problems = parse_description_dir(desc_dir)
        print(f"  Found {len(desc_problems)} problems from batch files.")
        all_problems.extend(desc_problems)
    else:
        print(f"Descriptions directory not found, skipping: {desc_dir}")

    if not all_problems:
        print("\nERROR: No problems found from any source.")
        sys.exit(1)

    # Deduplicate by slug; later entries (batch files) win over the legacy file
    keyed: dict[str, dict] = {}
    for p in all_problems:
        slug = p.get("slug", "")
        if slug:
            keyed[slug] = p

    print(f"\nTotal unique problems: {len(keyed)}")
    keyed = save_offline_problems(list(keyed.values()))
    print(f"Saved -> {OFFLINE_PROBLEMS_PATH}  ({len(keyed)} entries)")

    # Print sample
    sample = list(keyed.values())[:3]
    for p in sample:
        print(f"  • {p['title']} ({p['difficulty']}) [{', '.join(p['tags'][:2])}]")

    if args.index:
        print("\nBuilding FAISS descriptions index...")
        build_faiss_index(keyed)

    print("\nDone! Offline problem cache is ready.")
    if not args.index:
        print("Tip: add --index to also build a FAISS vector index for RAG retrieval.")


if __name__ == "__main__":
    main()
