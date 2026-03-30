"""
Router — LLM-based intent classification for user messages.

Design rationale:
  LLM classification is more robust than regex for DSA mentoring: it handles
  colloquial phrasing, mixed intents, and novel formulations that fixed patterns
  miss. The trade-off is one small extra LLM call per message (~0.3 s on Gemini
  flash), which is acceptable given the mentor already calls the LLM for the
  actual response.

  Two unambiguous signals are handled before the LLM call to avoid unnecessary
  latency:
    • Code block present (``` markers) → CODE_FIX immediately
    • Bare integer input               → QUESTION_LOOKUP immediately
"""
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Intent(Enum):
    """Possible user intents."""
    QUESTION_LOOKUP  = "question_lookup"   # "What is Two Sum?" / "LeetCode 121"
    CODE_FIX         = "code_fix"          # "Fix my code" / code block present
    HINT_REQUEST     = "hint_request"      # "Give me a hint" / "I'm stuck"
    CONCEPT_QUESTION = "concept_question"  # "Explain sliding window" / "When to use DP?"
    GENERAL_CHAT     = "general_chat"      # Greetings, follow-ups, anything else
    SOLUTION_REQUEST = "solution_request"  # "Show me the solution" / "I give up"


# ── LLM Classification Prompt ────────────────────────────────────

_ROUTER_PROMPT = """\
You are classifying a message from a DSA (Data Structures & Algorithms) student.
Classify the intent into exactly ONE of these categories:

QUESTION_LOOKUP   — student wants to look up a specific LeetCode problem by name or number
                    Examples: "What is Two Sum?", "LeetCode 121", "Problem 42", "Tell me about Merge Intervals"

CODE_FIX          — student wants help debugging or fixing code
                    Examples: "Fix my code", "What's wrong with this?", "Wrong output", "debug this"

HINT_REQUEST      — student is stuck and wants a hint, nudge, or approach guidance
                    Examples: "Give me a hint", "I'm stuck", "How do I approach this?", "Where do I start?"

SOLUTION_REQUEST  — student wants the full solution or answer
                    Examples: "Show me the solution", "I give up", "Just give me the answer", "Full solution"

CONCEPT_QUESTION  — student wants to understand a DSA concept or algorithm
                    Examples: "Explain sliding window", "When should I use BFS?", "What is dynamic programming?"

GENERAL_CHAT      — greetings, thank-yous, general follow-ups, anything not in the above
                    Examples: "Thanks!", "Can you say more?", "Why does this work?", "Hello"

Student message: "{message}"

Reply with ONLY the category name (e.g. HINT_REQUEST). No explanation, no punctuation.\
"""

_VALID_INTENTS = {i.name for i in Intent}


def detect_intent(message: str, has_code_block: bool = False) -> Intent:
    """
    Classify user intent using LLM, with fast-path for unambiguous signals.

    Fast-path (no LLM call):
      1. Code block markers present → CODE_FIX
      2. Bare integer (pure number)  → QUESTION_LOOKUP

    Everything else → LLM classification with GENERAL_CHAT fallback.
    """
    # ── Fast-path 1: code block ───────────────────────────────────
    if has_code_block or "```" in message:
        return Intent.CODE_FIX

    # ── Fast-path 2: pure number ──────────────────────────────────
    if re.match(r"^\d+$", message.strip()):
        return Intent.QUESTION_LOOKUP

    # ── LLM classification ────────────────────────────────────────
    try:
        from utils.llm import invoke_llm
        print(f"[Router] Classifying: \"{message[:80]}{'...' if len(message) > 80 else ''}\"", flush=True)
        prompt   = _ROUTER_PROMPT.format(message=message.strip())
        response = invoke_llm(prompt, temperature=0.0).strip().upper()

        # Strip punctuation and take first word in case model adds extras
        response = re.sub(r"[^A-Z_]", "", response.split()[0]) if response.split() else ""

        if response in _VALID_INTENTS:
            print(f"[Router] Intent → {response}", flush=True)
            return Intent[response]

        print(f"[Router] Unrecognised response '{response}', falling back to GENERAL_CHAT")
    except Exception as e:
        print(f"[Router] LLM classification failed ({e}), falling back to GENERAL_CHAT")

    return Intent.GENERAL_CHAT


# ── Helper: extract problem identifier ───────────────────────────

def extract_problem_identifier(message: str) -> Optional[str]:
    """
    Extract a LeetCode problem name or number from a user message.
    Returns the identifier string or None.
    """
    msg = message.strip()

    # "LeetCode 121" / "LC #1" / "Problem 42"
    match = re.search(r"(?:leetcode|lc|problem)\s*#?\s*(\d+)", msg, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pure number
    if re.match(r"^\d+$", msg):
        return msg

    # "1. Two Sum" format
    match = re.match(r"^#?(\d+)[\.\s]", msg)
    if match:
        return msg

    # Problem name in quotes
    match = re.search(r"['\"](.+?)['\"]", msg)
    if match:
        return match.group(1)

    # "What is Two Sum" / "Tell me about Merge Intervals"
    for prefix in [
        "what is ", "describe ", "explain ", "tell me about ",
        "look up ", "find ", "fetch ",
    ]:
        if msg.lower().startswith(prefix):
            return msg[len(prefix):].strip().strip("?").strip()

    return msg if len(msg.split()) <= 6 else None


# ── Helper: extract code block ────────────────────────────────────

def extract_code_block(message: str) -> Optional[str]:
    """Extract code from a markdown code block in the message."""
    match = re.search(r"```(?:\w+)?\s*\n?(.*?)```", message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None
