"""
Profile Builder — constructs and persists a UserProfile from solved ProblemRecords.
Single-user installation: all data lives in USER_DATA_PATH (data/user_data.json).

Schema of user_data.json:
{
    "profile":      { ...UserProfile fields... },
    "problems":     [ ...ProblemRecord dicts... ],
    "session_logs": [ ...session log dicts... ]
}
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX, USER_DATA_PATH


# ── Models ───────────────────────────────────────────────────────

class PatternMastery(BaseModel):
    """Tracks mastery of a single algorithmic pattern."""
    solved: int = 0
    confidence: str = "LOW"  # LOW | MEDIUM | HIGH


class UserProfile(BaseModel):
    """Aggregate user skill profile derived from solved problems."""
    total_solved: int = 0
    by_difficulty: Dict[str, int] = Field(
        default_factory=lambda: {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    )
    pattern_mastery: Dict[str, PatternMastery] = Field(default_factory=dict)
    weak_patterns:   List[str] = Field(default_factory=list)
    strong_patterns: List[str] = Field(default_factory=list)
    recent_activity: List[dict] = Field(default_factory=list)
    last_updated: Optional[str] = None

    def summary_text(self) -> str:
        """Render a concise text summary for LLM system prompts."""
        lines = [
            f"Total solved: {self.total_solved}",
            f"Breakdown: Easy={self.by_difficulty.get('EASY', 0)}, "
            f"Medium={self.by_difficulty.get('MEDIUM', 0)}, "
            f"Hard={self.by_difficulty.get('HARD', 0)}",
        ]
        if self.strong_patterns:
            lines.append(f"Strong patterns: {', '.join(self.strong_patterns)}")
        if self.weak_patterns:
            lines.append(f"Weak patterns: {', '.join(self.weak_patterns)}")
        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────

def _compute_confidence(count: int) -> str:
    if count < CONFIDENCE_LOW_MAX:
        return "LOW"
    elif count <= CONFIDENCE_MED_MAX:
        return "MEDIUM"
    return "HIGH"


# ── Builder ──────────────────────────────────────────────────────

def build_profile(problems: list) -> UserProfile:
    """Build a UserProfile from a list of ProblemRecord objects."""
    profile = UserProfile()
    profile.total_solved = len(problems)

    diff_counts: Dict[str, int] = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    for p in problems:
        key = p.difficulty.upper()
        if key in diff_counts:
            diff_counts[key] += 1
    profile.by_difficulty = diff_counts

    pattern_counts: Dict[str, int] = {}
    for p in problems:
        for pat in p.patterns:
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    mastery: Dict[str, PatternMastery] = {}
    weak:   List[str] = []
    strong: List[str] = []
    for pat, count in sorted(pattern_counts.items()):
        conf = _compute_confidence(count)
        mastery[pat] = PatternMastery(solved=count, confidence=conf)
        if conf == "LOW":
            weak.append(pat)
        elif conf == "HIGH":
            strong.append(pat)

    profile.pattern_mastery = mastery
    profile.weak_patterns   = weak
    profile.strong_patterns = strong

    cutoff = datetime.now() - timedelta(days=30)
    recent = []
    for p in problems:
        if p.solved_at and p.solved_at >= cutoff:
            recent.append({
                "problem_id": p.problem_id,
                "title":      p.title,
                "difficulty": p.difficulty,
                "patterns":   p.patterns,
                "solved_at":  p.solved_at.isoformat(),
            })
    profile.recent_activity = sorted(recent, key=lambda x: x["solved_at"], reverse=True)
    profile.last_updated = datetime.now().isoformat()

    return profile


# ── Single-user JSON storage ──────────────────────────────────────

def _load_data() -> dict:
    """Load the full user data store. Returns empty dict if not found."""
    if not USER_DATA_PATH.exists():
        return {}
    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_data(data: dict):
    """Atomically save the full user data store."""
    USER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USER_DATA_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(USER_DATA_PATH)


def save_data(profile: UserProfile, problems: list):
    """Save profile and problems into user_data.json, preserving existing session_logs."""
    data = _load_data()
    data["profile"]  = profile.model_dump(mode="json")
    data["problems"] = [p.model_dump(mode="json") for p in problems]
    # Preserve existing session_logs if present
    if "session_logs" not in data:
        data["session_logs"] = []
    _save_data(data)


def load_profile() -> Optional[UserProfile]:
    """Load the UserProfile from user_data.json."""
    data = _load_data()
    if "profile" not in data:
        return None
    try:
        return UserProfile(**data["profile"])
    except Exception:
        return None


def load_problems() -> list:
    """Load the list of raw problem dicts from user_data.json."""
    return _load_data().get("problems", [])


def get_problems_by_pattern(pattern_key: str) -> list[dict]:
    """
    Return all solved problems that include *pattern_key* in their patterns list.
    Reads directly from user_data.json — no FAISS needed, always complete.
    """
    problems = _load_data().get("problems", [])
    matched = []
    for p in problems:
        pats = p.get("patterns", [])
        if pattern_key in pats:
            matched.append({
                "title":      p.get("title", p.get("problem_id", "?")),
                "difficulty": p.get("difficulty", "Medium"),
                "patterns":   pats,
            })
    return matched


def get_problems_by_patterns(pattern_keys: list[str]) -> list[dict]:
    """
    Return all solved problems that match ANY of the given pattern keys.
    Deduplicates by title. Used when multiple patterns map to one query (e.g. 'dp').
    """
    seen  = set()
    result = []
    for key in pattern_keys:
        for p in get_problems_by_pattern(key):
            if p["title"] not in seen:
                seen.add(p["title"])
                result.append(p)
    return result


# Map of natural-language keywords → one OR more taxonomy pattern keys
# Value is a list so "dp" can map to both dp_1d and dp_2d at once.
PATTERN_KEYWORD_MAP: dict[str, list[str]] = {
    # DP — generic catches both 1d and 2d
    "dynamic programming":  ["dp_1d", "dp_2d"],
    " dp ":                 ["dp_1d", "dp_2d"],
    "dp question":          ["dp_1d", "dp_2d"],
    "dp problems":          ["dp_1d", "dp_2d"],
    "dp problem":           ["dp_1d", "dp_2d"],
    "done dp":              ["dp_1d", "dp_2d"],
    "which dp":             ["dp_1d", "dp_2d"],
    "what dp":              ["dp_1d", "dp_2d"],
    # DP — specific
    "2d dp":                ["dp_2d"],
    "dp 2d":                ["dp_2d"],
    "dp_2d":                ["dp_2d"],
    "two dimensional dp":   ["dp_2d"],
    "2d dynamic":           ["dp_2d"],
    "1d dp":                ["dp_1d"],
    "dp_1d":                ["dp_1d"],
    "1d dynamic":           ["dp_1d"],
    # Other patterns
    "sliding window":       ["sliding_window"],
    "two pointer":          ["two_pointers"],
    "two pointers":         ["two_pointers"],
    "binary search":        ["binary_search"],
    "bfs":                  ["bfs"],
    "dfs":                  ["dfs"],
    "backtracking":         ["backtracking"],
    "greedy":               ["greedy"],
    "graph":                ["bfs", "dfs", "graph_topological", "graph_shortest_path"],
    "trie":                 ["trie"],
    "heap":                 ["heap_kth_element"],
    "priority queue":       ["heap_kth_element"],
    "union find":           ["union_find"],
    "disjoint set":         ["union_find"],
    "prefix sum":           ["prefix_sum"],
    "monotonic stack":      ["monotonic_stack"],
    "bit manipulation":     ["bit_manipulation"],
    "linked list":          ["linked_list"],
    "interval":             ["intervals"],
    "segment tree":         ["segment_tree"],
    "topological":          ["graph_topological"],
    "dijkstra":             ["graph_shortest_path"],
    "shortest path":        ["graph_shortest_path"],
    "string matching":      ["string_matching"],
    "hashing":              ["hashing"],
    "hash map":             ["hashing"],
    "hash table":           ["hashing"],
    "divide and conquer":   ["divide_and_conquer"],
    "recursion":            ["recursion"],
    "simulation":           ["simulation"],
    "design":               ["design"],
    "math":                 ["math"],
}


def detect_patterns_in_query(query: str) -> list[str]:
    """
    Scan a user query for pattern keywords.
    Returns a deduplicated list of matching taxonomy keys (may be multiple).
    Returns [] if no known pattern is detected.
    Longer keywords are checked first to avoid short substrings stealing matches.
    """
    q    = f" {query.lower()} "   # pad so " dp " doesn't match "sdp"
    seen = set()
    found = []
    for keyword, keys in sorted(PATTERN_KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if keyword in q:
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    found.append(k)
    return found
