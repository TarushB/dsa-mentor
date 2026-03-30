"""
Query Rewriter — vague follow-up queries ko standalone queries mein rewrite karta hai
in-session chat history use karke taaki RAG retrieval se sahi results aayein.

Flow:  User follow-up  →  LLM rewrite kare  →  standalone query  →  FAISS search
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ── Chat History Buffer — session ke dauran history yahan rakhte hain ────

class ChatHistory:
    """
    In-session chat history buffer.

    Last `max_turns` exchanges memory mein rakhta hai conversation ke dauran.
    Yeh save_session() se alag hai — woh session khatam hone pe FAISS mein dalta hai,
    yeh *session ke beech mein* query rewriting ke liye chahiye.
    """

    def __init__(self, max_turns: int = 10):
        self.turns: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add_user_message(self, content: str):
        self.turns.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.turns.append({"role": "assistant", "content": content})
        self._trim()

    def get_history(self) -> List[Dict[str, str]]:
        """Poori chat history return karo."""
        return list(self.turns)

    def get_formatted_history(self) -> str:
        """History ko readable string mein format karo rewriter prompt ke liye."""
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            # Lambe messages ko chota karo taaki prompt zyada bada na ho
            content = turn["content"][:300]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        self.turns = []

    def _trim(self):
        """Sirf last max_turns * 2 messages rakho (user+assistant ke pairs)."""
        max_messages = self.max_turns * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]


# ── Query Rewriter — vague queries ko samajhdar banao ─────────────────

REWRITE_PROMPT = """Given the following conversation history and a follow-up question, rewrite the follow-up question to be a fully standalone question that captures the full context from the conversation.

Rules:
- If the follow-up question is already a clear, standalone question, return it unchanged.
- Resolve all pronouns ("it", "this", "that") using the conversation context.
- Keep the rewritten question concise and focused.
- Return ONLY the rewritten question, nothing else.

Chat History:
{history}

Follow-up Question: {question}

Standalone Question:"""


def rewrite_query(
    query: str,
    chat_history: List[Dict[str, str]],
) -> str:
    """
    Vague follow-up query ko chat history use karke standalone query mein badlo.

    Args:
        query: user ki current (shayad vague) query
        chat_history: {"role": ..., "content": ...} dicts ki list

    Returns:
        Standalone, context-rich query jo FAISS similarity search ke liye sahi ho.
        Agar rewriting fail ho ya history empty ho toh original query return hogi.
    """
    # History nahi hai toh kuch resolve karne ki zaroorat nahi, as-is return karo
    if not chat_history:
        return query

    # History ko format karo LLM ke liye
    lines = []
    for turn in chat_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"][:300]
        lines.append(f"{role}: {content}")
    formatted_history = "\n".join(lines)

    # LLM se rewrite karwao
    try:
        from utils.llm import invoke_llm

        prompt = REWRITE_PROMPT.format(
            history=formatted_history,
            question=query,
        )
        rewritten = invoke_llm(prompt, temperature=0.0)

        # Sanity check: agar LLM empty ya bahut lamba kuch de toh original pe fall back
        if rewritten and len(rewritten) < 500:
            print(f"  Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten
        else:
            return query

    except Exception as e:
        print(f"  WARNING: Query rewrite failed, using original: {e}")
        return query
