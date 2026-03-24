"""
Memory Module — session write-back + user profile updates after each interaction.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PROFILE_PATH, CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX


def format_session(
    problem_id: str,
    problem_title: str,
    patterns: List[str],
    conversation_turns: List[Dict[str, str]],
    outcome: str,
    key_learning: str = "",
) -> str:
    """
    Format a session into a text block for FAISS indexing.

    Args:
        problem_id: problem slug
        problem_title: human-readable title
        patterns: algorithmic patterns used
        conversation_turns: list of {"role": ..., "content": ...}
        outcome: "SOLVED" | "GAVE_UP" | "IN_PROGRESS"
        key_learning: summary of what the user learned

    Returns:
        Formatted text string for the session document
    """
    lines = [
        f"Date: {datetime.now().strftime('%Y-%m-%d')}",
        f"Problem: {problem_title}",
        f"Patterns: {', '.join(patterns)}",
        f"Outcome: {outcome}",
    ]

    # Summarize conversation (extract user struggles and hints given)
    struggles = []
    hints = []
    for turn in conversation_turns:
        role = turn.get("role", "")
        content = turn.get("content", "")[:200]  # truncate for index
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
) -> Document:
    """
    Save a session to the FAISS session_index.

    Returns:
        The Document that was added
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
            "date": datetime.now().isoformat(),
            "patterns": patterns,
            "outcome": outcome,
        },
    )

    add_documents_to_index("sessions", [doc])
    return doc


def update_user_profile(
    problem_id: str,
    patterns: List[str],
    outcome: str,
    difficulty: str = "MEDIUM",
):
    """
    Incrementally update the user profile after a session.

    Updates:
      - total_solved (if outcome == SOLVED and problem not already counted)
      - pattern_mastery counts
      - weak/strong pattern lists
      - recent_activity
    """
    if not PROFILE_PATH.exists():
        return

    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Only update stats if solved
    if outcome == "SOLVED":
        # Check if already counted (avoid double-counting)
        existing_ids = {a["problem_id"] for a in profile.get("recent_activity", [])}
        if problem_id not in existing_ids:
            profile["total_solved"] = profile.get("total_solved", 0) + 1

            # Update difficulty count
            diff_key = difficulty.upper()
            if diff_key in profile.get("by_difficulty", {}):
                profile["by_difficulty"][diff_key] += 1

        # Update pattern mastery
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

        # Recompute weak/strong lists
        weak = [p for p, m in mastery.items() if m["confidence"] == "LOW"]
        strong = [p for p, m in mastery.items() if m["confidence"] == "HIGH"]
        profile["weak_patterns"] = weak
        profile["strong_patterns"] = strong

    # Append to recent activity
    activity = profile.get("recent_activity", [])
    activity.insert(0, {
        "problem_id": problem_id,
        "difficulty": difficulty,
        "patterns": patterns,
        "outcome": outcome,
        "solved_at": datetime.now().isoformat(),
    })
    # Keep only last 30 days of activity
    profile["recent_activity"] = activity[:100]  # cap at 100 entries

    profile["last_updated"] = datetime.now().isoformat()

    # Save atomically
    tmp_path = PROFILE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, default=str)
    tmp_path.replace(PROFILE_PATH)
