"""
Mentor Agent — Main orchestrator that routes user messages through the right pipeline.
Combines LLM-based intent detection, RAG retrieval, hint chain, code fixing, and
question lookup. Single-user: no username parameter needed.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.router import Intent, detect_intent, extract_problem_identifier, extract_code_block
from agents.hint_chain import HintChain
from prompts.mentor import build_mentor_prompt
from prompts.hints import get_code_fix_prompt, get_question_lookup_prompt


class MentorAgent:
    """
    Main DSA Mentor agent. Handles all user interactions with context-aware responses.

    Usage:
        agent = MentorAgent()
        response = agent.handle_message("How do I solve Two Sum?")
    """

    def __init__(self):
        self.hint_chain:      HintChain           = HintChain()
        self.current_problem: Optional[str]       = None
        self.chat_history:    List[Dict[str, str]] = []

    def handle_message(
        self, user_message: str,
        provider: str = None, model: str = None,
    ) -> str:
        """
        Process a user message and return the mentor's response.

        Pipeline:
          1. Detect intent (LLM-based with fast-path)
          2. Retrieve RAG context (problems + sessions)
          3. Build appropriate prompt
          4. Invoke LLM
          5. Update chat history
        """
        from rag.retrievers import retrieve_all

        print(f"\n[Mentor]  New message ", flush=True)
        print(f"[Mentor] User: \"{user_message[:120]}{'...' if len(user_message) > 120 else ''}\"", flush=True)

        # Detect intent
        has_code = "```" in user_message
        intent   = detect_intent(user_message, has_code_block=has_code)
        print(f"[Mentor] Intent: {intent.value}", flush=True)

        # Retrieve RAG context
        print(f"[Mentor] Running RAG retrieval...", flush=True)
        rag_results = retrieve_all(
            query=user_message,
            chat_history=self.chat_history,
        )
        n_problems = len(rag_results.get("problems", []))
        n_sessions = len(rag_results.get("sessions", []))
        print(f"[Mentor] RAG results — problems: {n_problems}, sessions: {n_sessions}", flush=True)

        # Format RAG results into context strings
        similar_problems = self._format_problems(rag_results.get("problems", []))
        session_context  = self._format_sessions(rag_results.get("sessions", []))

        # Pattern-specific lookup: if query mentions a known pattern, inject ALL
        # matching solved problems directly from user_data.json (not just FAISS top-k)
        pattern_history = self._get_pattern_history(user_message)
        # Get user profile summary
        profile_summary = self._get_profile_summary()

        # Build system prompt with RAG context
        system_prompt = build_mentor_prompt(
            user_profile_summary=profile_summary,
            similar_problems=similar_problems,
            session_context=session_context,
            current_problem=self.current_problem or "",
            pattern_history=pattern_history,
        )

        # Build the user prompt based on intent
        if intent == Intent.CODE_FIX:
            print(f"[Mentor] Handling: CODE_FIX", flush=True)
            user_prompt = self._handle_code_fix(user_message)
        elif intent == Intent.QUESTION_LOOKUP:
            print(f"[Mentor] Handling: QUESTION_LOOKUP", flush=True)
            user_prompt = self._handle_question_lookup(user_message)
        elif intent == Intent.HINT_REQUEST:
            print(f"[Mentor] Handling: HINT_REQUEST (tier {self.hint_chain.current_tier})", flush=True)
            user_prompt = self._handle_hint_request(user_message)
        elif intent == Intent.SOLUTION_REQUEST:
            print(f"[Mentor] Handling: SOLUTION_REQUEST (hints given: {self.hint_chain.hints_given})", flush=True)
            user_prompt = self._handle_solution_request(user_message)
        else:
            print(f"[Mentor] Handling: {intent.value}", flush=True)
            user_prompt = user_message

        # Invoke LLM (with automatic key-rotation + Ollama fallback)
        print(f"[Mentor] Sending to LLM (system={len(system_prompt)} chars, prompt={len(user_prompt)} chars)...", flush=True)
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from utils.llm import invoke_llm_messages

        lc_messages = [SystemMessage(content=system_prompt)]
        for turn in self.chat_history[-6:]:
            if turn["role"] == "user":
                lc_messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                lc_messages.append(AIMessage(content=turn["content"]))
        lc_messages.append(HumanMessage(content=user_prompt))

        try:
            answer = invoke_llm_messages(lc_messages, temperature=0.3,
                                         provider=provider, model=model)
            print(f"[Mentor] OK Response received ({len(answer)} chars).", flush=True)
        except Exception as e:
            print(f"[Mentor] ERR LLM error: {e}", flush=True)
            answer = f"Sorry, all LLM providers are currently unavailable. Please try again shortly.\n\nError: {e}"

        # Update chat history
        self.chat_history.append({"role": "user",      "content": user_message})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer

    #  Intent handlers 

    def _handle_code_fix(self, message: str) -> str:
        """Build prompt for code-fix intent."""
        import re
        code          = extract_code_block(message) or ""
        question_part = re.sub(r"```[\s\S]*?```", "", message).strip()
        if not question_part:
            question_part = "Please fix the errors in my code."
        return get_code_fix_prompt(
            code=code,
            user_message=question_part,
            problem_context=self.current_problem or "",
        )

    def _handle_question_lookup(self, message: str) -> str:
        """Build prompt for question-lookup intent."""
        identifier          = extract_problem_identifier(message) or message
        self.current_problem = identifier
        self.hint_chain.reset()
        return get_question_lookup_prompt(identifier)

    def _handle_hint_request(self, message: str) -> str:
        """Build prompt for progressive hint."""
        problem = self.current_problem or "the current problem"
        return self.hint_chain.get_next_hint_prompt(problem, message)

    def _handle_solution_request(self, message: str) -> str:
        """Build prompt for solution request — enforces anti-gaming check."""
        problem = self.current_problem or "the current problem"
        if self.hint_chain.can_show_solution():
            return self.hint_chain.get_specific_tier_prompt(5, problem, message)
        remaining = 3 - self.hint_chain.hints_given
        return (
            f"The student wants the full solution but hasn't tried enough hints yet. "
            f"They need {remaining} more hint(s) before the solution is unlocked. "
            f"Give them a Tier {self.hint_chain.current_tier + 1} hint instead and "
            f"encourage them to keep trying. Problem: {problem}. "
            f"Student's message: {message}"
        )

    #  Context helpers 

    @staticmethod
    def _get_pattern_history(query: str) -> str:
        """
        If the query mentions known patterns, return ALL solved problems for those
        patterns directly from user_data.json (bypasses FAISS k-limit).
        """
        try:
            from ingestion.profile_builder import detect_patterns_in_query, get_problems_by_patterns
            pattern_keys = detect_patterns_in_query(query)
            if not pattern_keys:
                return ""
            problems = get_problems_by_patterns(pattern_keys)
            label = " / ".join(k.replace("_", " ").upper() for k in pattern_keys)
            if not problems:
                return f"Pattern history for [{label}]: No solved problems found in your records."
            lines = [f"All [{label}] problems you have solved ({len(problems)} total):"]
            for p in problems:
                diff = p.get("difficulty", "?")
                pats = ", ".join(p.get("patterns", []))
                lines.append(f"  - {p['title']} [{diff}] (patterns: {pats})")
            return "\n".join(lines)
        except Exception:
            return ""

    @staticmethod
    def _get_profile_summary() -> str:
        """Get user profile summary for the system prompt."""
        try:
            from ingestion.profile_builder import load_profile
            profile = load_profile()
            if profile:
                return profile.summary_text()
        except Exception:
            pass
        return ""

    @staticmethod
    def _format_problems(docs) -> str:
        if not docs:
            return ""
        lines = []
        for doc in docs:
            meta = doc.metadata
            lines.append(
                f"- {meta.get('title', '?')} [{meta.get('difficulty', '?')}] "
                f"(patterns: {', '.join(meta.get('patterns', []))})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_sessions(docs) -> str:
        if not docs:
            return ""
        return "\n---\n".join(doc.page_content[:300] for doc in docs)

    #  Public control methods 

    def set_current_problem(self, problem: str):
        """Set the active problem context and reset the hint chain."""
        self.current_problem = problem
        self.hint_chain.reset()

    def get_hint_info(self) -> Dict:
        """Return current hint chain state."""
        return self.hint_chain.get_tier_info()

    def clear_history(self):
        """Clear chat history and hint state for a new session."""
        self.chat_history    = []
        self.current_problem = None
        self.hint_chain.reset()
