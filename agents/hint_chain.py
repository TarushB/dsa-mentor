"""
Hint Chain — Progressive hint system with anti-gaming protection.
Tracks hint tier per problem and escalates through 5 levels.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MIN_HINTS_BEFORE_SOLUTION, MAX_HINT_TIERS
from prompts.hints import get_hint_prompt, HINT_TIERS


class HintChain:
    """
    Manages progressive hints for a single problem session.

    Tiers:
      1 = Pattern Nudge (just name the pattern)
      2 = Approach Direction (high-level strategy)
      3 = Detailed Strategy (pseudocode)
      4 = Implementation Guide (partial code)
      5 = Full Solution Walkthrough

    Anti-gaming: must go through MIN_HINTS_BEFORE_SOLUTION hints
    before tier 5 is unlocked.
    """

    def __init__(self):
        self.current_tier: int = 0
        self.hints_given: int = 0
        self.hint_history: List[Dict] = []

    def get_next_hint_prompt(self, problem: str, user_message: str) -> str:
        """
        Get the prompt for the next hint tier.

        Auto-escalates: each call moves to the next tier.
        """
        self.current_tier = min(self.current_tier + 1, MAX_HINT_TIERS)
        self.hints_given += 1

        prompt = get_hint_prompt(self.current_tier, problem, user_message)
        self.hint_history.append({
            "tier": self.current_tier,
            "user_message": user_message[:200],
        })

        return prompt

    def get_specific_tier_prompt(self, tier: int, problem: str, user_message: str) -> str:
        """Get prompt for a specific tier (with anti-gaming check)."""
        tier = max(1, min(MAX_HINT_TIERS, tier))

        # Anti-gaming: can't jump to solution without enough hints
        if tier >= MAX_HINT_TIERS and self.hints_given < MIN_HINTS_BEFORE_SOLUTION:
            tier = min(self.current_tier + 1, MAX_HINT_TIERS - 1)

        self.current_tier = tier
        self.hints_given += 1

        prompt = get_hint_prompt(tier, problem, user_message)
        self.hint_history.append({
            "tier": tier,
            "user_message": user_message[:200],
        })

        return prompt

    def can_show_solution(self) -> bool:
        """Check if enough hints have been given to unlock the full solution."""
        return self.hints_given >= MIN_HINTS_BEFORE_SOLUTION

    def get_tier_info(self) -> Dict:
        """Get info about current hint state."""
        tier_data = HINT_TIERS.get(self.current_tier, HINT_TIERS[1])
        return {
            "current_tier": self.current_tier,
            "tier_name": tier_data["name"],
            "tier_description": tier_data["description"],
            "hints_given": self.hints_given,
            "can_show_solution": self.can_show_solution(),
            "max_tiers": MAX_HINT_TIERS,
        }

    def reset(self):
        """Reset for a new problem."""
        self.current_tier = 0
        self.hints_given = 0
        self.hint_history = []
