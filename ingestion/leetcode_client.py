"""
LeetCode Client — problem description lookup + sync + CSV import.

Three responsibilities:
  1. Problem *lookup* (mentor UI) — offline cache + GraphQL, read-only.
  2. Recent-solved sync — public GraphQL API (no auth), last N accepted
     submissions for a username, used by the incremental sync feature.
  3. CSV bulk import — parse a CSV export for initial / full ingestion.
"""
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LEETCODE_GRAPHQL_URL, LEETCODE_REQUEST_DELAY, OFFLINE_PROBLEMS_PATH


# ── Offline problem cache ─────────────────────────────────────────

_offline_cache: Optional[dict] = None


def _load_offline_cache() -> dict:
    """Load offline problems JSON (keyed by slug). Cached in memory."""
    global _offline_cache
    if _offline_cache is None:
        if OFFLINE_PROBLEMS_PATH.exists():
            try:
                with open(OFFLINE_PROBLEMS_PATH, "r", encoding="utf-8") as f:
                    _offline_cache = json.load(f)
            except Exception:
                _offline_cache = {}
        else:
            _offline_cache = {}
    return _offline_cache


def _lookup_offline(slug: str) -> Optional[Dict[str, Any]]:
    """Find a problem by slug in the offline cache."""
    cache = _load_offline_cache()
    entry = cache.get(slug)
    if not entry:
        return None
    return {
        "number":      entry.get("number", "?"),
        "title":       entry.get("title", slug),
        "slug":        slug,
        "difficulty":  entry.get("difficulty", "Medium"),
        "tags":        entry.get("tags", []),
        "description": entry.get("description", ""),
        "examples":    entry.get("examples", ""),
        "hints":       entry.get("hints", []),
        "source":      "offline",
    }


def _slugify(text: str) -> str:
    """Convert a problem title to a URL slug."""
    return text.lower().strip().replace(" ", "-").replace("'", "").replace(",", "")


# ── LeetCode GraphQL client (read-only problem lookup) ───────────

class LeetCodeClient:
    """
    Handles read-only problem lookups (offline cache + GraphQL) and CSV import.
    No authentication, no user-account queries.
    """

    def __init__(self):
        self._last_request_time = 0.0

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Referer":      "https://leetcode.com",
            "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < LEETCODE_REQUEST_DELAY:
            time.sleep(LEETCODE_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _graphql_request(self, query: str, variables: dict = None) -> dict:
        """Send a rate-limited synchronous GraphQL request."""
        self._rate_limit()
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                LEETCODE_GRAPHQL_URL,
                json=payload,
                headers=self._headers(),
            )
            resp.raise_for_status()
            return resp.json()

    # ── Problem description lookup ────────────────────────────────

    def get_problem_description(self, slug: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a full problem description.
        Checks offline cache first, then falls back to LeetCode GraphQL.

        Returns dict with: number, title, slug, difficulty, tags, description,
                           examples, hints, source
        """
        # 1. Offline cache (instant, no API call)
        offline = _lookup_offline(slug)
        if offline:
            return offline

        # 2. LeetCode GraphQL
        query = """
        query questionDetail($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionFrontendId
                title
                titleSlug
                difficulty
                content
                exampleTestcases
                hints
                topicTags { name }
                stats
            }
        }
        """
        try:
            import re
            result = self._graphql_request(query, {"titleSlug": slug})
            q = result.get("data", {}).get("question")
            if not q:
                return None

            # Strip HTML from content
            content_html = q.get("content", "") or ""
            text = re.sub(r"<br\s*/?>", "\n", content_html)
            text = re.sub(r"<p>",       "\n", text)
            text = re.sub(r"</p>",      "\n", text)
            text = re.sub(r"<li>",      "  • ", text)
            text = re.sub(r"</li>",     "\n", text)
            text = re.sub(r"<ul>|</ul>|<ol>|</ol>", "\n", text)
            text = re.sub(r"<pre>|</pre>",           "\n", text)
            text = re.sub(r"<strong>|</strong>|<b>|</b>", "**", text)
            text = re.sub(r"<em>|</em>",             "_", text)
            text = re.sub(r"<code>(.*?)</code>",     r"`\1`", text, flags=re.DOTALL)
            text = re.sub(r"<sup>(.*?)</sup>",       r"^\1", text)
            text = re.sub(r"<sub>(.*?)</sub>",       r"_\1", text)
            text = re.sub(r"<[^>]+>", "", text)
            text = (text
                    .replace("&lt;",  "<").replace("&gt;",  ">")
                    .replace("&amp;", "&").replace("&nbsp;", " ")
                    .replace("&quot;", '"').replace("&#39;", "'"))
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            return {
                "number":      q.get("questionFrontendId", "?"),
                "title":       q.get("title", "Unknown"),
                "slug":        q.get("titleSlug", slug),
                "difficulty":  q.get("difficulty", "Medium"),
                "tags":        [t["name"] for t in q.get("topicTags", [])],
                "description": text,
                "examples":    q.get("exampleTestcases", ""),
                "hints":       q.get("hints", []),
            }
        except Exception as e:
            print(f"  Warning: Could not fetch description for '{slug}': {e}")
            return None

    def get_problem_by_number(self, number: int) -> Optional[Dict[str, Any]]:
        """Look up a problem by its frontend question number (e.g. 1 → two-sum)."""
        query = """
        query problemsetQuestionList($skip: Int, $limit: Int, $filters: QuestionListFilterInput) {
            problemsetQuestionList: questionList(
                categorySlug: ""
                limit: $limit
                skip: $skip
                filters: $filters
            ) {
                questions: data {
                    frontendQuestionId: questionFrontendId
                    titleSlug
                    title
                    difficulty
                }
            }
        }
        """
        skip = max(0, number - 3)
        try:
            result    = self._graphql_request(query, {"skip": skip, "limit": 10, "filters": {}})
            questions = result.get("data", {}).get("problemsetQuestionList", {}).get("questions", [])
            for q in questions:
                if str(q.get("frontendQuestionId", "")) == str(number):
                    desc = self.get_problem_description(q["titleSlug"])
                    if desc and desc.get("number") in ("?", None, ""):
                        desc = dict(desc)
                        desc["number"] = str(number)
                    return desc
        except Exception:
            pass
        return None

    def search_problem_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a problem by name/title.
        Strategy:
          1. Exact offline slug lookup
          2. Fuzzy offline title match
          3. LeetCode API via direct slug conversion
        """
        slug = _slugify(name)

        offline = _lookup_offline(slug)
        if offline:
            return offline

        cache      = _load_offline_cache()
        name_lower = name.lower().strip()
        for entry_slug, entry in cache.items():
            entry_title = entry.get("title", "").lower()
            if name_lower in entry_title or entry_title in name_lower:
                return _lookup_offline(entry_slug)

        return self.get_problem_description(slug)

    # ── All-solved sync (public GraphQL, no auth) ────────────────

    def get_all_solved(self, username: str, limit: int = 3000) -> List[Dict[str, Any]]:
        """
        Fetch ALL accepted submissions for a public LeetCode username via the
        public GraphQL API (no cookies / auth required).

        `limit` caps the submission list fetched (LeetCode's API returns up to
        ~3000 recent AC submissions; raise it only if needed).

        Returns a list of deduplicated, normalised problem dicts compatible with
        raw_to_problem_record(), each having:
            title, titleSlug, difficulty, topicTags, timestamp
        """
        print(f"[LeetCode] Fetching accepted submissions for '{username}' (limit={limit})...")
        submissions = self._get_recent_ac_submissions(username, limit=limit)
        if not submissions:
            print(f"[LeetCode] No submissions returned for '{username}'.")
            return []

        print(f"[LeetCode] Got {len(submissions)} raw submissions. Deduplicating...")

        # Deduplicate by slug (a problem may be submitted multiple times)
        seen: Dict[str, dict] = {}
        for sub in submissions:
            slug = sub.get("titleSlug", "")
            if slug and slug not in seen:
                seen[slug] = sub

        total = len(seen)
        print(f"[LeetCode] {total} unique problems found. Fetching metadata...")

        problems: List[Dict[str, Any]] = []
        for i, (slug, sub) in enumerate(seen.items(), 1):
            print(f"[LeetCode] Metadata {i}/{total}: {slug}", flush=True)
            try:
                meta = self._get_problem_metadata(slug)
            except Exception as e:
                print(f"[LeetCode]   ⚠  metadata failed for '{slug}': {e}")
                meta = None

            if meta:
                meta["timestamp"] = sub.get("timestamp")
                problems.append(meta)
            else:
                # Fallback: use whatever the submission gave us
                problems.append({
                    "title":      sub.get("title", slug),
                    "titleSlug":  slug,
                    "difficulty": "Medium",
                    "topicTags":  [],
                    "timestamp":  sub.get("timestamp"),
                })

        print(f"[LeetCode] Done. {len(problems)}/{total} problems fetched successfully.")
        return problems

    def _get_recent_ac_submissions(self, username: str, limit: int = 20) -> List[Dict]:
        """Fetch recent accepted submissions for a public LeetCode username."""
        query = """
        query recentAcSubmissions($username: String!, $limit: Int!) {
            recentAcSubmissionList(username: $username, limit: $limit) {
                id
                title
                titleSlug
                timestamp
            }
        }
        """
        try:
            result = self._graphql_request(query, {"username": username, "limit": limit})
            return result.get("data", {}).get("recentAcSubmissionList", [])
        except Exception as e:
            print(f"  [Sync] Error fetching submissions for '{username}': {e}")
            return []

    def _get_problem_metadata(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch lightweight problem metadata (title, difficulty, tags) by slug."""
        query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                questionFrontendId
                title
                titleSlug
                difficulty
                topicTags { name slug }
            }
        }
        """
        result = self._graphql_request(query, {"titleSlug": slug})
        return result.get("data", {}).get("question")

    # ── CSV import (bulk ingestion) ───────────────────────────────

    @staticmethod
    def import_from_csv(csv_path: str) -> List[Dict[str, Any]]:
        """
        Import solved problems from a CSV export.

        Expected columns (flexible naming):
            id / question_id / frontend_question_id
            title / question_title
            title_slug / titleslug / slug
            difficulty / level
            tags / topic_tags / related_topics   (comma or semicolon separated)

        Returns a list of normalised problem dicts ready for raw_to_problem_record().
        """
        problems = []
        path     = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {
                    k.strip().lower().replace(" ", "_"): v.strip()
                    for k, v in row.items()
                }

                problem = {
                    "frontendQuestionId": (
                        normalized.get("id")
                        or normalized.get("question_id")
                        or normalized.get("frontend_question_id")
                        or ""
                    ),
                    "title": (
                        normalized.get("title")
                        or normalized.get("question_title")
                        or ""
                    ),
                    "titleSlug": (
                        normalized.get("title_slug")
                        or normalized.get("titleslug")
                        or normalized.get("slug")
                        or _slugify(normalized.get("title", ""))
                    ),
                    "difficulty": (
                        normalized.get("difficulty")
                        or normalized.get("level")
                        or "Medium"
                    ),
                    "topicTags": [],
                    "status":    "ac",
                }

                tags_str = (
                    normalized.get("tags")
                    or normalized.get("topic_tags")
                    or normalized.get("related_topics")
                    or ""
                )
                if tags_str:
                    tags = [t.strip() for t in tags_str.replace(";", ",").split(",") if t.strip()]
                    problem["topicTags"] = [
                        {"name": t, "slug": t.lower().replace(" ", "-")} for t in tags
                    ]

                problems.append(problem)


        return problems
