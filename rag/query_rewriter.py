"""
Query Rewriter — rewrites vague follow-up queries into standalone queries
using in-session chat history so RAG retrieval produces relevant results.

Flow:  User follow-up  →  LLM rewrites  →  standalone query  →  FAISS search
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_BASE_URL, OLLAMA_MODEL


# ── Chat History Buffer ──────────────────────────────────────────

class ChatHistory:
    """
    In-session chat history buffer.

    Keeps the last `max_turns` exchanges in memory during a conversation.
    This is separate from save_session() which persists to FAISS *after*
    a session ends — this is needed *during* the session for query rewriting.
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
        """Return the full chat history."""
        return list(self.turns)

    def get_formatted_history(self) -> str:
        """Format history as a readable string for the rewriter prompt."""
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            role = "User" if turn["role"] == "user" else "Assistant"
            # Truncate long messages to keep prompt short
            content = turn["content"][:300]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        self.turns = []

    def _trim(self):
        """Keep only the last max_turns * 2 messages (pairs of user+assistant)."""
        max_messages = self.max_turns * 2
        if len(self.turns) > max_messages:
            self.turns = self.turns[-max_messages:]


# ── Query Rewriter ───────────────────────────────────────────────

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
    Rewrite a vague follow-up query into a standalone query using chat history.

    Args:
        query: the user's current (possibly vague) query
        chat_history: list of {"role": ..., "content": ...} dicts

    Returns:
        A standalone, context-rich query suitable for FAISS similarity search.
        Returns the original query if rewriting fails or history is empty.
    """
    # No history = nothing to resolve, return as-is
    if not chat_history:
        return query

    # Format the history
    lines = []
    for turn in chat_history:
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"][:300]
        lines.append(f"{role}: {content}")
    formatted_history = "\n".join(lines)

    # Call LLM to rewrite
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,  # deterministic rewriting
        )
        prompt = REWRITE_PROMPT.format(
            history=formatted_history,
            question=query,
        )
        response = llm.invoke(prompt)
        rewritten = response.content.strip()

        # Sanity check: if LLM returns empty or something too long, fall back
        if rewritten and len(rewritten) < 500:
            print(f"  🔄 Query rewritten: '{query}' → '{rewritten}'")
            return rewritten
        else:
            return query

    except Exception as e:
        print(f"  ⚠ Query rewrite failed, using original: {e}")
        return query
