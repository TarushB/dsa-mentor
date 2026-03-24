"""
LeetCode GraphQL Client — fetches solved problems, submissions, and problem metadata.

Three modes:
  1. PUBLIC  — uses only username, no auth needed (default)
  2. AUTH    — uses LEETCODE_SESSION + csrftoken cookies for richer data
  3. CSV     — manual CSV import fallback
"""
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LEETCODE_GRAPHQL_URL, LEETCODE_REQUEST_DELAY


class LeetCodeClient:
    """Fetches LeetCode data via GraphQL (public or authenticated) or CSV import."""

    def __init__(self, use_auth: bool = False):
        self.session_cookie = os.getenv("LEETCODE_SESSION", "")
        self.csrf_token = os.getenv("CSRF_TOKEN", "")
        self.use_auth = use_auth and bool(self.session_cookie)
        self._last_request_time = 0.0

    # ── HTTP helpers ─────────────────────────────────────────────

    def _headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Referer": "https://leetcode.com",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if self.use_auth:
            headers["Cookie"] = f"LEETCODE_SESSION={self.session_cookie}; csrftoken={self.csrf_token}"
            headers["x-csrftoken"] = self.csrf_token
        return headers

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < LEETCODE_REQUEST_DELAY:
            time.sleep(LEETCODE_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _graphql_request(self, query: str, variables: dict = None) -> dict:
        """Synchronous GraphQL request with rate limiting."""
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

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC QUERIES — work with just a username, no cookies needed
    # ══════════════════════════════════════════════════════════════

    def fetch_solved_problems_public(self, username: str) -> List[Dict[str, Any]]:
        """
        Fetch ALL solved problems for a user using public queries.
        No authentication required — just the LeetCode username.

        Strategy:
          1. Get recent AC submissions (gives titleSlug + timestamp)
          2. Deduplicate by slug
          3. Fetch full problem metadata for each (title, difficulty, tags)

        Returns list of problem dicts ready for parsing.
        """
        print(f"  [1/3] Fetching recent accepted submissions for '{username}'...")
        submissions = self._get_recent_ac_submissions(username)

        if not submissions:
            print(f"  ERROR: No submissions found. Is '{username}' a valid public profile?")
            return []

        # Deduplicate by slug (user may have submitted same problem multiple times)
        seen = {}
        for sub in submissions:
            slug = sub.get("titleSlug", "")
            if slug and slug not in seen:
                seen[slug] = sub

        unique_slugs = list(seen.keys())
        print(f"  [2/3] Found {len(unique_slugs)} unique solved problems.")

        # Fetch full metadata for each problem
        print(f"  [3/3] Fetching problem details (this may take a moment)...")
        problems = []
        for i, slug in enumerate(unique_slugs, 1):
            if i % 10 == 0 or i == len(unique_slugs):
                print(f"         ...{i}/{len(unique_slugs)}")
            try:
                problem_data = self._get_problem_metadata(slug)
                if problem_data:
                    # Merge with submission data (timestamp)
                    problem_data["timestamp"] = seen[slug].get("timestamp")
                    problems.append(problem_data)
            except Exception as e:
                print(f"    Warning: Failed to fetch '{slug}': {e}")
                # Still add basic info from submission
                problems.append({
                    "title": seen[slug].get("title", slug),
                    "titleSlug": slug,
                    "difficulty": "Medium",
                    "topicTags": [],
                    "timestamp": seen[slug].get("timestamp"),
                })

        return problems

    def _get_recent_ac_submissions(self, username: str, limit: int = 3000) -> List[Dict]:
        """Fetch recent accepted submissions (PUBLIC — no auth needed)."""
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
            print(f"    Error fetching submissions: {e}")
            return []

    def _get_problem_metadata(self, slug: str) -> Optional[Dict[str, Any]]:
        """Fetch problem details by slug (PUBLIC — no auth needed)."""
        query = """
        query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId
                questionFrontendId
                title
                titleSlug
                difficulty
                topicTags {
                    name
                    slug
                }
            }
        }
        """
        result = self._graphql_request(query, {"titleSlug": slug})
        return result.get("data", {}).get("question")

    def get_user_stats(self, username: str) -> Dict[str, Any]:
        """Fetch user profile stats (PUBLIC)."""
        query = """
        query userProblemsSolved($username: String!) {
            matchedUser(username: $username) {
                username
                submitStatsGlobal {
                    acSubmissionNum {
                        difficulty
                        count
                    }
                }
            }
        }
        """
        try:
            result = self._graphql_request(query, {"username": username})
            return result.get("data", {}).get("matchedUser", {})
        except Exception:
            return {}

    # ══════════════════════════════════════════════════════════════
    #  AUTH QUERIES — require LEETCODE_SESSION cookie
    # ══════════════════════════════════════════════════════════════

    def get_user_solved_list_auth(self, username: str) -> List[Dict[str, Any]]:
        """Fetch solved problems using auth cookie (if available)."""
        all_problems = []
        skip = 0
        limit = 50

        while True:
            query = """
            query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
                problemsetQuestionList: questionList(
                    categorySlug: $categorySlug
                    limit: $limit
                    skip: $skip
                    filters: $filters
                ) {
                    total: totalNum
                    questions: data {
                        frontendQuestionId: questionFrontendId
                        title
                        titleSlug
                        difficulty
                        topicTags {
                            name
                            slug
                        }
                        status
                    }
                }
            }
            """
            variables = {
                "categorySlug": "",
                "skip": skip,
                "limit": limit,
                "filters": {"status": "AC"},
            }
            result = self._graphql_request(query, variables)
            data = result.get("data", {}).get("problemsetQuestionList", {})
            questions = data.get("questions", [])

            if not questions:
                break

            all_problems.extend(questions)
            total = data.get("total", 0)
            skip += limit
            print(f"    ...fetched {len(all_problems)}/{total}")

            if skip >= total:
                break

        return all_problems

    # ══════════════════════════════════════════════════════════════
    #  CSV FALLBACK
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def import_from_csv(csv_path: str) -> List[Dict[str, Any]]:
        """
        Import solved problems from a LeetCode progress CSV export.
        Expected columns: id, title, title_slug, difficulty, tags
        """
        problems = []
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                normalized = {k.strip().lower().replace(" ", "_"): v.strip() for k, v in row.items()}

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
                    "status": "ac",
                }

                tags_str = (
                    normalized.get("tags")
                    or normalized.get("topic_tags")
                    or normalized.get("related_topics")
                    or ""
                )
                if tags_str:
                    tags = [t.strip() for t in tags_str.replace(";", ",").split(",") if t.strip()]
                    problem["topicTags"] = [{"name": t, "slug": t.lower().replace(" ", "-")} for t in tags]

                problems.append(problem)

        return problems


def _slugify(text: str) -> str:
    """Convert a title to a URL slug."""
    return text.lower().strip().replace(" ", "-").replace("'", "").replace(",", "")
