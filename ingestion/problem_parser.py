"""
Problem Parser — ProblemRecord schema + LLM-powered pattern tagging via Ollama.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TAXONOMY, OLLAMA_BASE_URL, OLLAMA_MODEL


# ── Pydantic Models ──────────────────────────────────────────────

class ProblemRecord(BaseModel):
    """Schema for a single solved LeetCode problem."""
    problem_id: str                          # slug, e.g. "two-sum"
    title: str
    difficulty: str = "Medium"               # EASY | MEDIUM | HARD
    topic_tags: List[str] = Field(default_factory=list)
    solved_at: Optional[datetime] = None
    num_attempts: int = 1
    time_to_solve: Optional[float] = None    # minutes
    patterns: List[str] = Field(default_factory=list)
    personal_notes: str = ""

    def to_document_text(self) -> str:
        """Render as a text block for FAISS indexing."""
        lines = [
            f"Problem: {self.title} [{self.difficulty.upper()}]",
            f"Patterns: {', '.join(self.patterns) if self.patterns else 'unknown'}",
            f"Tags: {', '.join(self.topic_tags) if self.topic_tags else 'none'}",
        ]
        if self.solved_at:
            lines.append(f"Solved: {self.solved_at.strftime('%Y-%m-%d')} | Attempts: {self.num_attempts}")
        if self.personal_notes:
            lines.append(f"Notes: {self.personal_notes}")
        return "\n".join(lines)


# ── Pattern Tagger ───────────────────────────────────────────────

PATTERN_TAG_PROMPT = """You are a DSA problem classifier. Given a LeetCode problem's title, difficulty, and topic tags, classify it into one or more algorithmic patterns from this fixed taxonomy:

{taxonomy}

Problem title: {title}
Difficulty: {difficulty}
Topic tags: {tags}

Rules:
1. Return ONLY a JSON array of matching pattern strings from the taxonomy above.
2. Usually 1-3 patterns per problem.
3. If unsure, pick the most likely pattern.
4. Do NOT invent patterns outside the taxonomy.

Example output: ["two_pointers", "binary_search"]

Your answer (JSON array only):"""


class PatternTagger:
    """Tags problems with algorithmic patterns using Ollama (local LLM)."""

    def __init__(self):
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.1,
                format="json",
            )
            self._available = True
        except Exception:
            self._available = False

    def tag_problem(self, title: str, difficulty: str, tags: List[str]) -> List[str]:
        """Classify a single problem into taxonomy patterns."""
        if not self._available:
            return self._rule_based_fallback(tags)

        prompt = PATTERN_TAG_PROMPT.format(
            taxonomy=json.dumps(TAXONOMY, indent=2),
            title=title,
            difficulty=difficulty,
            tags=", ".join(tags) if tags else "none",
        )

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            patterns = self._parse_patterns(content)
            return patterns if patterns else self._rule_based_fallback(tags)
        except Exception as e:
            print(f"  ⚠ LLM tagging failed for '{title}': {e}")
            return self._rule_based_fallback(tags)

    def _parse_patterns(self, text: str) -> List[str]:
        """Extract valid taxonomy patterns from LLM output."""
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [p for p in data if p in TAXONOMY]
            if isinstance(data, dict):
                # Handle {"patterns": [...]} or similar
                for v in data.values():
                    if isinstance(v, list):
                        return [p for p in v if p in TAXONOMY]
        except json.JSONDecodeError:
            pass

        # Fallback: regex extraction
        found = []
        for pattern in TAXONOMY:
            if pattern in text:
                found.append(pattern)
        return found

    @staticmethod
    def _rule_based_fallback(tags: List[str]) -> List[str]:
        """Map LeetCode topic tags to taxonomy patterns when LLM is unavailable."""
        tag_to_pattern = {
            "two pointers": "two_pointers",
            "sliding window": "sliding_window",
            "binary search": "binary_search",
            "prefix sum": "prefix_sum",
            "stack": "monotonic_stack",
            "monotonic stack": "monotonic_stack",
            "breadth-first search": "bfs",
            "depth-first search": "dfs",
            "backtracking": "backtracking",
            "dynamic programming": "dp_1d",
            "greedy": "greedy",
            "union find": "union_find",
            "trie": "trie",
            "heap (priority queue)": "heap_kth_element",
            "priority queue": "heap_kth_element",
            "topological sort": "graph_topological",
            "graph": "dfs",
            "bit manipulation": "bit_manipulation",
            "math": "math",
            "array": "two_pointers",
            "hash table": "two_pointers",
            "string": "sliding_window",
            "tree": "dfs",
            "binary tree": "dfs",
            "binary search tree": "binary_search",
            "linked list": "two_pointers",
            "matrix": "dp_2d",
            "sorting": "two_pointers",
        }

        patterns = set()
        for tag in tags:
            normalized = tag.lower().strip()
            if normalized in tag_to_pattern:
                patterns.add(tag_to_pattern[normalized])

        return list(patterns) if patterns else ["math"]


# ── Problem Conversion ──────────────────────────────────────────

def raw_to_problem_record(raw: Dict[str, Any], tagger: PatternTagger = None) -> ProblemRecord:
    """Convert a raw API/CSV dict into a ProblemRecord with pattern tags."""
    # Extract fields
    title = raw.get("title", "Unknown")
    slug = raw.get("titleSlug", raw.get("title_slug", _slugify(title)))
    difficulty = raw.get("difficulty", "Medium").upper()
    if difficulty not in ("EASY", "MEDIUM", "HARD"):
        difficulty = "MEDIUM"

    # Extract topic tags
    topic_tags_raw = raw.get("topicTags", [])
    if isinstance(topic_tags_raw, list) and topic_tags_raw and isinstance(topic_tags_raw[0], dict):
        topic_tags = [t["name"] for t in topic_tags_raw]
    elif isinstance(topic_tags_raw, list):
        topic_tags = [str(t) for t in topic_tags_raw]
    else:
        topic_tags = []

    # Parse timestamp if available
    solved_at = None
    ts = raw.get("timestamp")
    if ts:
        try:
            solved_at = datetime.fromtimestamp(int(ts))
        except (ValueError, TypeError, OSError):
            pass

    # Tag patterns
    if tagger:
        patterns = tagger.tag_problem(title, difficulty, topic_tags)
    else:
        patterns = PatternTagger._rule_based_fallback(topic_tags)

    return ProblemRecord(
        problem_id=slug,
        title=title,
        difficulty=difficulty,
        topic_tags=topic_tags,
        solved_at=solved_at,
        patterns=patterns,
    )


def _slugify(text: str) -> str:
    return text.lower().strip().replace(" ", "-").replace("'", "").replace(",", "")
