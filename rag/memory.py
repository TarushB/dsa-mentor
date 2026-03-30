"""
Memory Module — session write-back and user profile updates after each interaction.
Single-user: reads/writes USER_DATA_PATH directly, no username keying.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX, USER_DATA_PATH


def format_session(
    problem_id: str,
    problem_title: str,
    patterns: List[str],
    conversation_turns: List[Dict[str, str]],
    outcome: str,
    key_learning: str = "",
) -> str:
    """Format a session into a text block for FAISS indexing."""
    lines = [
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Problem: {problem_title}",
        f"Patterns: {', '.join(patterns)}",
        f"Outcome: {outcome}",
    ]

    struggles, hints = [], []
    for turn in conversation_turns:
        role    = turn.get("role", "")
        content = turn.get("content", "")[:200]
        if role == "user":
            struggles.append(content)
        elif role == "assistant":
            hints.append(content)

    if struggles:
        lines.append(f"User struggles: {'; '.join(struggles[:3])}")
    if hints:
        lines.append(f"Hints given: {'; '.join(hints[:3])}")
    if key_learning:
        lines.append(f"Key learning: {key_learning}")

    return "\n".join(lines)


def save_session(
    problem_id: str,
    problem_title: str,
    patterns: List[str],
    conversation_turns: List[Dict[str, str]],
    outcome: str,
    key_learning: str = "",
    hints_used: int = 0,
    hint_tier_reached: int = 0,
    difficulty: str = "MEDIUM",
) -> Document:
    """
    Save a session to the FAISS sessions index and append a structured log to user_data.json.

    Returns:
        The Document added to the sessions index.
    """
    from rag.embeddings import add_documents_to_index

    content = format_session(
        problem_id, problem_title, patterns,
        conversation_turns, outcome, key_learning
    )

    doc = Document(
        page_content=content,
        metadata={
            "problem_id": problem_id,
            "date":       datetime.now().isoformat(),
            "patterns":   patterns,
            "outcome":    outcome,
        },
    )

    add_documents_to_index("sessions", [doc])
    _append_session_log(
        problem_id=problem_id,
        problem_title=problem_title,
        difficulty=difficulty,
        patterns=patterns,
        outcome=outcome,
        hints_used=hints_used,
        hint_tier_reached=hint_tier_reached,
    )

    return doc


def _append_session_log(
    problem_id: str,
    problem_title: str,
    difficulty: str,
    patterns: List[str],
    outcome: str,
    hints_used: int,
    hint_tier_reached: int,
):
    """Append a structured session log entry to user_data.json."""
    if not USER_DATA_PATH.exists():
        return

    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    log_entry = {
        "date":               datetime.now().strftime("%Y-%m-%d"),
        "problem_id":         problem_id,
        "problem_title":      problem_title,
        "difficulty":         difficulty.upper(),
        "patterns":           patterns,
        "outcome":            outcome,
        "hints_used":         hints_used,
        "hint_tier_reached":  hint_tier_reached,
    }

    session_logs = data.get("session_logs", [])
    session_logs.append(log_entry)
    data["session_logs"] = session_logs

    tmp = USER_DATA_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(USER_DATA_PATH)


def update_user_profile(
    problem_id: str,
    patterns: List[str],
    outcome: str,
    difficulty: str = "MEDIUM",
    title: str = "",
):
    """
    Incrementally update the user profile after a session outcome.
    Reads and writes USER_DATA_PATH directly.
    """
    if not USER_DATA_PATH.exists():
        return

    with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = data.get("profile", {})

    if outcome == "SOLVED":
        existing_ids = {a["problem_id"] for a in profile.get("recent_activity", [])}
        if problem_id not in existing_ids:
            profile["total_solved"] = profile.get("total_solved", 0) + 1
            diff_key = difficulty.upper()
            by_diff  = profile.get("by_difficulty", {"EASY": 0, "MEDIUM": 0, "HARD": 0})
            if diff_key in by_diff:
                by_diff[diff_key] += 1
            profile["by_difficulty"] = by_diff

        mastery = profile.get("pattern_mastery", {})
        for pat in patterns:
            if pat not in mastery:
                mastery[pat] = {"solved": 0, "confidence": "LOW"}
            mastery[pat]["solved"] += 1
            count = mastery[pat]["solved"]
            if count < CONFIDENCE_LOW_MAX:
                mastery[pat]["confidence"] = "LOW"
            elif count <= CONFIDENCE_MED_MAX:
                mastery[pat]["confidence"] = "MEDIUM"
            else:
                mastery[pat]["confidence"] = "HIGH"
        profile["pattern_mastery"] = mastery

        weak   = [p for p, m in mastery.items() if m["confidence"] == "LOW"]
        strong = [p for p, m in mastery.items() if m["confidence"] == "HIGH"]
        profile["weak_patterns"]   = weak
        profile["strong_patterns"] = strong

    activity = profile.get("recent_activity", [])
    activity.insert(0, {
        "problem_id": problem_id,
        "title":      title or problem_id,
        "difficulty": difficulty,
        "patterns":   patterns,
        "outcome":    outcome,
        "solved_at":  datetime.now().isoformat(),
    })
    profile["recent_activity"] = activity[:100]
    profile["last_updated"]    = datetime.now().isoformat()

    data["profile"] = profile

    tmp = USER_DATA_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(USER_DATA_PATH)
