"""
Profile Builder — solved ProblemRecords se UserProfile banao aur save karo.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIDENCE_LOW_MAX, CONFIDENCE_MED_MAX, PROFILE_PATH


# ── Models — profile ka structure ──────────────────────────────────

class PatternMastery(BaseModel):
    """Ek single algorithmic pattern ki mastery track karta hai."""
    solved: int = 0
    confidence: str = "LOW"  # LOW | MEDIUM | HIGH


class UserProfile(BaseModel):
    """Solved problems se bani aggregate user skill profile."""
    username: str = ""
    total_solved: int = 0
    by_difficulty: Dict[str, int] = Field(default_factory=lambda: {"EASY": 0, "MEDIUM": 0, "HARD": 0})
    pattern_mastery: Dict[str, PatternMastery] = Field(default_factory=dict)
    weak_patterns: List[str] = Field(default_factory=list)
    strong_patterns: List[str] = Field(default_factory=list)
    recent_activity: List[dict] = Field(default_factory=list)  # last 30 din ki activity
    last_updated: Optional[str] = None

    def summary_text(self) -> str:
        """LLM system prompts ke liye concise text summary render karo."""
        lines = [
            f"Student: {self.username} | Total solved: {self.total_solved}",
            f"Breakdown: Easy={self.by_difficulty.get('EASY', 0)}, "
            f"Medium={self.by_difficulty.get('MEDIUM', 0)}, "
            f"Hard={self.by_difficulty.get('HARD', 0)}",
        ]
        if self.strong_patterns:
            lines.append(f"Strong patterns: {', '.join(self.strong_patterns)}")
        if self.weak_patterns:
            lines.append(f"Weak patterns: {', '.join(self.weak_patterns)}")
        return "\n".join(lines)


# ── Builder — profile banana yahan hota hai ───────────────────────

def _compute_confidence(count: int) -> str:
    if count < CONFIDENCE_LOW_MAX:
        return "LOW"
    elif count <= CONFIDENCE_MED_MAX:
        return "MEDIUM"
    return "HIGH"


def build_profile(problems: list, username: str = "") -> UserProfile:
    """
    ProblemRecord objects ki list se UserProfile banao.

    Args:
        problems: ProblemRecord ki list (problem_parser se aati hai)
        username: LeetCode username

    Returns:
        Fully computed UserProfile
    """
    profile = UserProfile(username=username)
    profile.total_solved = len(problems)

    # Difficulty wise count karo
    diff_counts = {"EASY": 0, "MEDIUM": 0, "HARD": 0}
    for p in problems:
        key = p.difficulty.upper()
        if key in diff_counts:
            diff_counts[key] += 1
    profile.by_difficulty = diff_counts

    # Pattern counts aggregate karo
    pattern_counts: Dict[str, int] = {}
    for p in problems:
        for pat in p.patterns:
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    # Mastery map banao
    mastery = {}
    weak = []
    strong = []
    for pat, count in sorted(pattern_counts.items()):
        conf = _compute_confidence(count)
        mastery[pat] = PatternMastery(solved=count, confidence=conf)
        if conf == "LOW":
            weak.append(pat)
        elif conf == "HIGH":
            strong.append(pat)

    profile.pattern_mastery = mastery
    profile.weak_patterns = weak
    profile.strong_patterns = strong

    # Recent activity (last 30 din ki)
    cutoff = datetime.now() - timedelta(days=30)
    recent = []
    for p in problems:
        if p.solved_at and p.solved_at >= cutoff:
            recent.append({
                "problem_id": p.problem_id,
                "title": p.title,
                "difficulty": p.difficulty,
                "patterns": p.patterns,
                "solved_at": p.solved_at.isoformat(),
            })
    profile.recent_activity = sorted(recent, key=lambda x: x["solved_at"], reverse=True)
    profile.last_updated = datetime.now().isoformat()

    return profile


def save_profile(profile: UserProfile, path: Path = None) -> Path:
    """UserProfile ko JSON mein save karo."""
    save_path = path or PROFILE_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(profile.model_dump(), f, indent=2, default=str)
    return save_path


def load_profile(path: Path = None) -> Optional[UserProfile]:
    """Pehle se saved UserProfile load karo."""
    load_path = path or PROFILE_PATH
    if not load_path.exists():
        return None
    with open(load_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return UserProfile(**data)
