"""
LeetCode GraphQL Client — solved problems, submissions, aur problem metadata fetch karta hai.

Teen modes hain:
  1. PUBLIC  — sirf username chahiye, auth ki zaroorat nahi (default)
  2. AUTH    — LEETCODE_SESSION + csrftoken cookies lagenge, zyada data milega
  3. CSV     — manual CSV import fallback, agar kuch aur kaam na aaye
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
    """LeetCode se data GraphQL (public ya authenticated) ya CSV import se fetch karta hai."""

    def __init__(self, use_auth: bool = False):
        self.session_cookie = os.getenv("LEETCODE_SESSION", "")
        self.csrf_token = os.getenv("CSRF_TOKEN", "")
        self.use_auth = use_auth and bool(self.session_cookie)
        self._last_request_time = 0.0

    # ── HTTP helpers — request bhejne ke liye setup ─────────────────────

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
        """Rate limiting ke saath synchronous GraphQL request bhejo."""
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
    #  PUBLIC QUERIES — sirf username se kaam ho jaata hai, cookies nahi chahiye
    # ══════════════════════════════════════════════════════════════

    def fetch_solved_problems_public(self, username: str) -> List[Dict[str, Any]]:
        """
        Public queries se user ke SAARE solved problems fetch karo.
        Authentication ki zaroorat nahi — bas LeetCode username do.

        Strategy:
          1. Recent AC submissions lo (titleSlug + timestamp milega)
          2. Slug se deduplicate karo
          3. Har ek ke liye full problem metadata fetch karo (title, difficulty, tags)

        Parsing ke liye ready problem dicts ki list return karta hai.
        """
        print(f"  [1/3] Fetching recent accepted submissions for '{username}'...")
        submissions = self._get_recent_ac_submissions(username)

        if not submissions:
            print(f"  ERROR: No submissions found. Is '{username}' a valid public profile?")
            return []

        # Slug se deduplicate karo (same problem multiple baar submit ho sakta hai)
        seen = {}
        for sub in submissions:
            slug = sub.get("titleSlug", "")
            if slug and slug not in seen:
                seen[slug] = sub

        unique_slugs = list(seen.keys())
        print(f"  [2/3] Found {len(unique_slugs)} unique solved problems.")

        # Har problem ka full metadata fetch karo
        print(f"  [3/3] Fetching problem details (this may take a moment)...")
        problems = []
        for i, slug in enumerate(unique_slugs, 1):
            if i % 10 == 0 or i == len(unique_slugs):
                print(f"         ...{i}/{len(unique_slugs)}")
            try:
                problem_data = self._get_problem_metadata(slug)
                if problem_data:
                    # Submission data se timestamp merge karo
                    problem_data["timestamp"] = seen[slug].get("timestamp")
                    problems.append(problem_data)
            except Exception as e:
                print(f"    Warning: Failed to fetch '{slug}': {e}")
                # Phir bhi basic info toh add karo submission se
                problems.append({
                    "title": seen[slug].get("title", slug),
                    "titleSlug": slug,
                    "difficulty": "Medium",
                    "topicTags": [],
                    "timestamp": seen[slug].get("timestamp"),
                })

        return problems

    def _get_recent_ac_submissions(self, username: str, limit: int = 3000) -> List[Dict]:
        """Recent accepted submissions fetch karo (PUBLIC — auth nahi chahiye)."""
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
        """Slug se problem details fetch karo (PUBLIC — auth nahi chahiye)."""
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
        """User ki profile stats fetch karo (PUBLIC)."""
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
    #  AUTH QUERIES — LEETCODE_SESSION cookie lagegi
    # ══════════════════════════════════════════════════════════════

    def get_user_solved_list_auth(self, username: str) -> List[Dict[str, Any]]:
        """Auth cookie use karke solved problems fetch karo (agar available hai)."""
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
    #  CSV FALLBACK — jab kuch aur kaam na aaye
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def import_from_csv(csv_path: str) -> List[Dict[str, Any]]:
        """
        LeetCode progress CSV export se solved problems import karo.
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
    """Title ko URL slug mein convert karo."""
    return text.lower().strip().replace(" ", "-").replace("'", "").replace(",", "")
