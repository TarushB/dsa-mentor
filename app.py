"""
DSA Mentor — Streamlit Application (single-user)
Features:
  - LeetCode problem lookup by name or number (offline cache + GraphQL API)
  - Personalized problem recommendations based on weak patterns
  - Practice by pattern selector
  - Progressive 5-tier hint system
  - Full solution reveal in C++ (after 3 hints)
  - Free chat with MentorAgent (RAG-powered)
  - Code debugger (paste code, get detailed fix)
  - Mark problem as Solved / Gave Up (persists session + updates profile)
  - CSV import with incremental sync (checks last 20 solved, appends new ones)
  - Progress dashboard
"""
import json
import random
import re
import streamlit as st
import pandas as pd
from pathlib import Path

from config import DATA_DIR, USER_DATA_PATH, FAISS_DIR, OFFLINE_PROBLEMS_PATH
from ingestion.profile_builder import load_profile, UserProfile


# ═══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DSA Mentor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.problem-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 10px;
}
.difficulty-easy   { color: #a6e3a1; font-weight: bold; }
.difficulty-medium { color: #f9e2af; font-weight: bold; }
.difficulty-hard   { color: #f38ba8; font-weight: bold; }
.tag-badge {
    background: #45475a;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    margin: 2px;
    display: inline-block;
}
.pattern-badge {
    background: #313244;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 13px;
    margin: 3px;
    display: inline-block;
    cursor: pointer;
}
.rec-card {
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "current_page":       "solve",   # solve | dashboard | sync
        "problem":            None,
        "hint_tier":          0,
        "hints_given":        0,
        "chat_messages":      [],        # hints tab history
        "free_chat_messages": [],        # free chat tab history
        "solution_unlocked":  False,
        "agent":              None,
        "session_saved":      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def _get_agent():
    if st.session_state.agent is None:
        from agents.mentor_agent import MentorAgent
        st.session_state.agent = MentorAgent()
    return st.session_state.agent


def _llm_provider() -> str:
    return st.session_state.get("llm_provider", "")


def _llm_model() -> str:
    return st.session_state.get("llm_model", "")


def _invoke(prompt: str, temperature: float = 0.3) -> str:
    from utils.llm import invoke_llm
    return invoke_llm(prompt, temperature=temperature,
                      provider=_llm_provider(), model=_llm_model())


def _stream(prompt: str, temperature: float = 0.3):
    """Generator: streams LLM response tokens for use with st.write_stream()."""
    from utils.llm import stream_llm
    yield from stream_llm(prompt, temperature=temperature,
                          provider=_llm_provider(), model=_llm_model())


def _stream_messages(messages: list, temperature: float = 0.3):
    """Generator: streams LLM response tokens from a chat message list."""
    from utils.llm import stream_llm_messages
    yield from stream_llm_messages(messages, temperature=temperature,
                                   provider=_llm_provider(), model=_llm_model())


def _difficulty_color(diff: str) -> str:
    d = diff.upper()
    if d == "EASY": return "🟢"
    if d == "HARD": return "🔴"
    return "🟡"


def _reset_problem():
    st.session_state.problem          = None
    st.session_state.hint_tier        = 0
    st.session_state.hints_given      = 0
    st.session_state.chat_messages    = []
    st.session_state.solution_unlocked = False
    st.session_state.session_saved    = False
    if st.session_state.agent:
        st.session_state.agent.clear_history()


def _load_offline_cache() -> dict:
    if not OFFLINE_PROBLEMS_PATH.exists():
        return {}
    with open(OFFLINE_PROBLEMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
#  MODEL SELECTOR (sidebar widget)
# ═══════════════════════════════════════════════════════════════

def _show_model_selector():
    from utils.llm import list_available_providers
    from config import OLLAMA_MODEL

    st.markdown("#### 🤖 AI Model")

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "ollama"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = OLLAMA_MODEL

    models = list_available_providers()[0]["models"]
    if st.session_state.llm_model not in models:
        st.session_state.llm_model = OLLAMA_MODEL

    prev_model = st.session_state.llm_model
    st.session_state.llm_model = st.selectbox(
        "Ollama Model", models,
        index=models.index(st.session_state.llm_model),
        key="llm_model_select",
    )
    st.caption(f"Local Ollama — make sure `ollama serve` is running.")

    if prev_model != st.session_state.llm_model:
        st.session_state.agent = None


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════

def show_sidebar():
    with st.sidebar:
        st.markdown("### 🧠 DSA Mentor")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["🎯 Solve Problems", "📊 Dashboard", "📥 Import CSV"],
            label_visibility="collapsed",
        )
        if "Solve" in page:     st.session_state.current_page = "solve"
        elif "Dashboard" in page: st.session_state.current_page = "dashboard"
        elif "Import" in page:  st.session_state.current_page = "sync"

        st.markdown("---")

        profile = load_profile()
        if profile and profile.total_solved > 0:
            st.metric("Problems Solved", profile.total_solved)
            c1, c2, c3 = st.columns(3)
            c1.metric("Easy",  profile.by_difficulty.get("EASY", 0))
            c2.metric("Med",   profile.by_difficulty.get("MEDIUM", 0))
            c3.metric("Hard",  profile.by_difficulty.get("HARD", 0))
            if profile.weak_patterns:
                st.warning("Weak: " + ", ".join(
                    p.replace("_", " ").title() for p in profile.weak_patterns[:3]
                ))
        else:
            st.info("No data yet.\nImport a CSV to get started!")

        if st.session_state.problem:
            st.markdown("---")
            p    = st.session_state.problem
            diff = p.get("difficulty", "Medium")
            st.markdown(f"**Current:** {_difficulty_color(diff)} {p['title']}")
            if st.session_state.hints_given > 0:
                tier = st.session_state.hint_tier
                st.progress(tier / 5, text=f"Hint tier {tier}/5")
            if st.button("🔄 New Problem", use_container_width=True):
                _reset_problem()
                st.rerun()

        st.markdown("---")
        _show_recent_history()

        st.markdown("---")
        _show_model_selector()


def _show_recent_history():
    """Show last 10 attempted problems in the sidebar with outcome icons."""
    if not USER_DATA_PATH.exists():
        return
    try:
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    logs = data.get("session_logs", [])
    if not logs:
        return

    recent = logs[-10:][::-1]   # last 10, newest first

    st.markdown("#### 📜 Recent History")
    for entry in recent:
        outcome = entry.get("outcome", "")
        icon    = "✅" if outcome == "SOLVED" else "🏳️"
        diff    = entry.get("difficulty", "MEDIUM").upper()
        diff_ic = _difficulty_color(diff)
        title   = entry.get("problem_title", entry.get("problem_id", "?"))
        date    = entry.get("date", "")

        # Clicking a history item reloads that problem
        label = f"{icon} {diff_ic} {title}"
        if st.button(label, key=f"hist_{entry.get('problem_id','')}_{date}", use_container_width=True):
            _load_problem_sidebar(entry.get("problem_id", title))

        st.caption(date)


def _load_problem_sidebar(query: str):
    """Load a problem from the sidebar history (triggers rerun on solve page)."""
    with st.spinner(f"Loading '{query}'..."):
        from ingestion.leetcode_client import LeetCodeClient
        client  = LeetCodeClient()
        problem = client.search_problem_by_name(query) or client.get_problem_description(query)
    if problem:
        _reset_problem()
        st.session_state.problem      = problem
        st.session_state.current_page = "solve"
        _get_agent().set_current_problem(
            f"#{problem.get('number','?')}: {problem.get('title','?')} [{problem.get('difficulty','?')}]"
        )
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  PROBLEM LOOKUP
# ═══════════════════════════════════════════════════════════════

def fetch_problem(query_str: str) -> dict | None:
    from ingestion.leetcode_client import LeetCodeClient
    client    = LeetCodeClient()
    query_str = query_str.strip()

    if re.match(r"^\d+$", query_str):
        result = client.get_problem_by_number(int(query_str))
        if result:
            return result

    result = client.search_problem_by_name(query_str)
    if result:
        return result

    return _llm_describe_problem(query_str)


def _llm_describe_problem(name: str) -> dict:
    prompt = f"""Describe the LeetCode problem "{name}" in full detail.
Include:
1. Problem number and title
2. Full problem statement
3. At least 2 examples with Input/Output/Explanation
4. Constraints
5. Difficulty (Easy/Medium/Hard)

Format it clearly. If you don't know this exact problem, say so."""
    description = _invoke(prompt, temperature=0.1)
    return {
        "number": "?", "title": name,
        "slug":   name.lower().replace(" ", "-"),
        "difficulty": "Medium", "tags": [],
        "description": description, "examples": "", "hints": [],
        "source": "llm",
    }


def show_problem_card(problem: dict):
    diff      = problem.get("difficulty", "Medium").upper()
    diff_icon = _difficulty_color(diff)
    num       = problem.get("number", "?")
    title     = problem.get("title", "Unknown")
    tags      = problem.get("tags", [])
    desc      = problem.get("description", "")

    st.markdown(f"## {num}. {title}  {diff_icon} {diff}")

    if tags:
        tags_html = " ".join(f'<span class="tag-badge">{t}</span>' for t in tags)
        st.markdown(tags_html, unsafe_allow_html=True)
        st.markdown("")

    with st.expander("📋 Problem Description", expanded=True):
        st.markdown(desc)

    if problem.get("source") == "llm":
        st.caption("⚠️ Description generated by AI (LeetCode API unavailable).")


# ═══════════════════════════════════════════════════════════════
#  SESSION SAVE
# ═══════════════════════════════════════════════════════════════

def _save_current_session(outcome: str):
    """Persist session to FAISS + user_data.json."""
    if st.session_state.session_saved:
        return
    problem = st.session_state.problem
    if not problem:
        return

    try:
        from rag.memory import save_session, update_user_profile

        # Map common tag names to taxonomy patterns
        tag_to_pattern = {
            "Dynamic Programming": "dp_1d",
            "Array":               "two_pointers",
            "Hash Table":          "hashing",
            "Binary Search":       "binary_search",
            "Tree":                "dfs",
            "Graph":               "bfs",
            "Stack":               "monotonic_stack",
            "Linked List":         "linked_list",
            "Sliding Window":      "sliding_window",
            "Two Pointers":        "two_pointers",
            "Backtracking":        "backtracking",
            "Greedy":              "greedy",
            "Math":                "math",
        }
        patterns = []
        for tag in problem.get("tags", []):
            pat = tag_to_pattern.get(tag)
            if pat and pat not in patterns:
                patterns.append(pat)

        save_session(
            problem_id        = problem.get("slug", problem.get("title", "unknown")),
            problem_title     = problem.get("title", "Unknown"),
            patterns          = patterns,
            conversation_turns = st.session_state.chat_messages,
            outcome           = outcome,
            key_learning      = "",
            hints_used        = st.session_state.hints_given,
            hint_tier_reached = st.session_state.hint_tier,
            difficulty        = problem.get("difficulty", "MEDIUM").upper(),
        )

        update_user_profile(
            problem_id = problem.get("slug", problem.get("title", "unknown")),
            title      = problem.get("title", ""),
            patterns   = patterns,
            outcome    = outcome,
            difficulty = problem.get("difficulty", "MEDIUM").upper(),
        )

        st.session_state.session_saved = True
    except Exception as e:
        st.warning(f"Session save warning: {e}")


# ═══════════════════════════════════════════════════════════════
#  HINT SYSTEM
# ═══════════════════════════════════════════════════════════════

HINT_TIER_LABELS = {
    1: "💡 Nudge (Pattern)",
    2: "🧭 Approach Direction",
    3: "📝 Detailed Strategy",
    4: "🔧 Implementation Guide",
    5: "✅ Full Solution",
}


def get_hint(tier: int, problem: dict, user_context: str = "") -> str:
    from prompts.hints import get_hint_prompt

    problem_text = (
        f"Problem #{problem.get('number','?')}: {problem.get('title','?')}\n"
        f"Difficulty: {problem.get('difficulty','?')}\n"
        f"Tags: {', '.join(problem.get('tags', []))}\n\n"
        f"{problem.get('description', '')[:1500]}"
    )
    prompt  = get_hint_prompt(tier, problem_text, user_context or "I need a hint.")
    profile = load_profile()
    if profile:
        prompt = f"Student profile: {profile.summary_text()}\n\n" + prompt
    return _invoke(prompt, temperature=0.3)


def get_full_solution(problem: dict) -> str:
    problem_text = (
        f"Problem #{problem.get('number','?')}: {problem.get('title','?')}\n"
        f"Difficulty: {problem.get('difficulty','?')}\n\n"
        f"{problem.get('description','')[:2000]}"
    )
    prompt = f"""Provide a complete solution for this LeetCode problem in C++.

{problem_text}

Include:
1. **Approach** — which pattern/technique and why
2. **Algorithm** — step-by-step walkthrough
3. **C++ Solution** — complete working code using STL with inline comments
4. **Complexity** — time and space analysis
5. **Walkthrough** — trace through Example 1
6. **Edge Cases** — what to watch out for

Use proper C++ style: correct types, STL containers, and necessary headers.
"""
    return _invoke(prompt, temperature=0.2)


def debug_code(code: str, problem: dict, user_question: str, language: str = "cpp") -> str:
    from prompts.hints import get_code_fix_prompt
    problem_ctx = ""
    if problem:
        problem_ctx = (
            f"Problem #{problem.get('number','?')}: {problem.get('title','?')} "
            f"[{problem.get('difficulty','?')}]\n"
            f"{problem.get('description','')[:800]}"
        )
    prompt = get_code_fix_prompt(
        code=code,
        user_message=user_question or "Please find and fix all errors in my code.",
        problem_context=problem_ctx,
        language=language,
    )
    return _invoke(prompt, temperature=0.2)


# ═══════════════════════════════════════════════════════════════
#  SOLVE PAGE — SEARCH
# ═══════════════════════════════════════════════════════════════

_PATTERN_DISPLAY = {
    "two_pointers":      "Two Pointers",
    "sliding_window":    "Sliding Window",
    "binary_search":     "Binary Search",
    "prefix_sum":        "Prefix Sum",
    "monotonic_stack":   "Monotonic Stack",
    "bfs":               "BFS",
    "dfs":               "DFS / Trees",
    "backtracking":      "Backtracking",
    "dp_1d":             "Dynamic Programming",
    "dp_2d":             "DP 2D",
    "greedy":            "Greedy",
    "union_find":        "Union Find",
    "trie":              "Trie",
    "heap_kth_element":  "Heap / Top-K",
    "graph_topological": "Topological Sort",
    "bit_manipulation":  "Bit Manipulation",
    "math":              "Math",
    "hashing":           "Hash Map",
    "linked_list":       "Linked List",
    "stack_queue":       "Stack / Queue",
    "intervals":         "Intervals",
    "string_matching":   "String Matching",
    "recursion":         "Recursion",
}

_TAG_TO_PATTERN_KEYWORDS = {
    "two_pointers":      ["Two Pointers", "Array"],
    "sliding_window":    ["Sliding Window"],
    "binary_search":     ["Binary Search"],
    "prefix_sum":        ["prefix sum", "prefix"],
    "monotonic_stack":   ["Stack", "monotonic"],
    "bfs":               ["Graph", "BFS"],
    "dfs":               ["Tree", "DFS", "Depth-First Search"],
    "backtracking":      ["Backtracking"],
    "dp_1d":             ["Dynamic Programming"],
    "dp_2d":             ["Dynamic Programming"],
    "greedy":            ["Greedy"],
    "union_find":        ["union find", "disjoint", "number of province", "number of island", "number of connected"],
    "trie":              ["trie", "prefix tree", "word search"],
    "heap_kth_element":  ["Heap", "kth", "top-k", "k largest", "k smallest", "top k"],
    "graph_topological": ["topolog", "course schedule", "alien", "prerequisite"],
    "bit_manipulation":  ["Bit Manipulation"],
    "math":              ["Math"],
    "hashing":           ["Hash Table", "Hash Map"],
    "linked_list":       ["Linked List"],
    "stack_queue":       ["Stack", "Queue"],
    "intervals":         ["interval", "meeting room"],
    "string_matching":   ["substring", "palindrome", "anagram"],
    "recursion":         ["Backtracking"],
}


def _get_recommendations(profile) -> list[dict]:
    """Return recommended problem dicts based on weak patterns (from offline cache)."""
    if not profile or not profile.weak_patterns:
        return []

    cache = _load_offline_cache()
    if not cache:
        return []

    recommendations = []
    seen_slugs      = set()

    for weak_pat in profile.weak_patterns[:4]:
        keywords = _TAG_TO_PATTERN_KEYWORDS.get(weak_pat, [])
        if not keywords:
            continue
        matches = [
            (slug, entry) for slug, entry in cache.items()
            if any(kw in entry.get("tags", []) for kw in keywords)
        ]
        if matches:
            sample = random.sample(matches, min(2, len(matches)))
            for slug, entry in sample:
                if slug not in seen_slugs:
                    seen_slugs.add(slug)
                    recommendations.append({
                        "slug":       slug,
                        "title":      entry.get("title", slug),
                        "difficulty": entry.get("difficulty", "Medium"),
                        "pattern":    _PATTERN_DISPLAY.get(weak_pat, weak_pat),
                        "tags":       entry.get("tags", []),
                    })

    return recommendations[:8]


def _get_pattern_problem(pattern_key: str) -> dict | None:
    """Return a random offline problem matching the given pattern key."""
    cache    = _load_offline_cache()
    if not cache:
        return None
    keywords = _TAG_TO_PATTERN_KEYWORDS.get(pattern_key, [_PATTERN_DISPLAY.get(pattern_key, pattern_key)])
    matches  = [
        (slug, entry) for slug, entry in cache.items()
        if any(kw.lower() in " ".join(entry.get("tags", [])).lower()
               or kw.lower() in entry.get("title", "").lower()
               for kw in keywords)
    ]
    if not matches:
        return None
    slug, entry = random.choice(matches)
    return {
        "number":      entry.get("number", "?"),
        "title":       entry.get("title", slug),
        "slug":        slug,
        "difficulty":  entry.get("difficulty", "Medium"),
        "tags":        entry.get("tags", []),
        "description": entry.get("description", ""),
        "examples":    entry.get("examples", ""),
        "hints":       entry.get("hints", []),
        "source":      "offline",
    }


def show_problem_search():
    st.markdown("## 🎯 Find a Problem")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Problem",
            placeholder="e.g.  1  or  Two Sum  or  Valid Parentheses",
            label_visibility="collapsed",
        )
    with col2:
        search = st.button("🔍 Search", type="primary", use_container_width=True)

    if search and query:
        _load_problem(query)
        return

    # Personalized recommendations
    profile = load_profile()
    recs    = _get_recommendations(profile)

    if recs:
        st.markdown("---")
        st.markdown("#### 🎯 Recommended for You  *(based on your weak areas)*")
        rec_cols = st.columns(4)
        for i, rec in enumerate(recs):
            col = rec_cols[i % 4]
            with col:
                diff_icon = _difficulty_color(rec["difficulty"])
                if col.button(f"{diff_icon} {rec['title']}", key=f"rec_{rec['slug']}_{i}", use_container_width=True):
                    _load_problem(rec["title"])
                    return
                col.caption(f"*{rec['pattern']}*")

    # Practice by pattern
    st.markdown("---")
    st.markdown("#### 🏋️ Practice by Pattern")

    patterns_to_show = list(_PATTERN_DISPLAY.items())[:16]
    p_rows = [patterns_to_show[i:i+4] for i in range(0, len(patterns_to_show), 4)]
    for row in p_rows:
        cols = st.columns(4)
        for col, (key, label) in zip(cols, row):
            if col.button(label, key=f"pat_{key}", use_container_width=True):
                st.session_state["browse_pattern"] = (key, label)
                st.session_state.pop("browse_pattern_choice", None)
                st.rerun()

    # Dropdown to pick a specific problem from the selected pattern
    if st.session_state.get("browse_pattern"):
        pat_key, pat_label = st.session_state["browse_pattern"]
        cache    = _load_offline_cache()
        keywords = _TAG_TO_PATTERN_KEYWORDS.get(pat_key, [_PATTERN_DISPLAY.get(pat_key, pat_key)])
        matches  = [
            entry.get("title", slug)
            for slug, entry in cache.items()
            if any(kw.lower() in " ".join(entry.get("tags", [])).lower()
                   or kw.lower() in entry.get("title", "").lower()
                   for kw in keywords)
        ]
        if not matches:
            st.warning(f"No problems found for {pat_label}.")
        else:
            st.markdown(f"**{pat_label}** — {len(matches)} problems available")
            choice = st.selectbox("Select a question:", matches,
                                  key="browse_pattern_choice",
                                  label_visibility="collapsed")
            if st.button("▶ Start Problem", type="primary", key="pat_go"):
                st.session_state.pop("browse_pattern", None)
                st.session_state.pop("browse_pattern_choice", None)
                _load_problem(choice)
                return



def _load_problem(query: str):
    with st.spinner(f"Fetching '{query}'..."):
        problem = fetch_problem(query)
    if problem:
        _reset_problem()
        st.session_state.problem = problem
        _get_agent().set_current_problem(
            f"#{problem.get('number','?')}: {problem.get('title','?')} [{problem.get('difficulty','?')}]"
        )
        st.rerun()
    else:
        st.error(f"Could not find '{query}'. Try the number (e.g. 1) or exact name.")


# ═══════════════════════════════════════════════════════════════
#  WORKSPACE
# ═══════════════════════════════════════════════════════════════

def _get_similar_solved(problem: dict, k: int = 4) -> list[dict]:
    """
    Return up to *k* solved problems similar to *problem* using the FAISS index.
    Results are cached in session_state per problem slug to avoid repeated lookups.
    """
    slug      = problem.get("slug", problem.get("title", ""))
    cache_key = f"_similar_{slug}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        from rag.retrievers import ProblemRetriever
        query = (
            f"{problem.get('title', '')} "
            f"{' '.join(problem.get('tags', []))} "
            f"{problem.get('description', '')[:300]}"
        )
        docs    = ProblemRetriever().retrieve(query, k=k + 1)   # fetch one extra to drop self
        results = []
        for doc in docs:
            meta = doc.metadata
            pid  = meta.get("problem_id", "")
            if pid == slug:          # skip the problem itself if it's in the index
                continue
            results.append({
                "slug":       pid,
                "title":      meta.get("title", pid),
                "difficulty": meta.get("difficulty", "Medium"),
                "patterns":   meta.get("patterns", []),
                "tags":       meta.get("topic_tags", []),
            })
            if len(results) == k:
                break
    except Exception:
        results = []

    st.session_state[cache_key] = results
    return results


def _show_similar_solved(similar: list[dict]):
    """Render the 'Similar Problems You've Solved' expander."""
    if not similar:
        return

    with st.expander("🔁 Similar Problems You've Already Solved", expanded=True):
        st.caption("These are from your history — review them for patterns and ideas before diving in.")
        cols = st.columns(len(similar))
        for col, p in zip(cols, similar):
            diff  = p.get("difficulty", "Medium")
            icon  = _difficulty_color(diff)
            pats  = [pat.replace("_", " ").title() for pat in p.get("patterns", [])[:2]]
            label = f"{icon} **{p['title']}**"
            col.markdown(label)
            col.caption(f"{diff}" + (f"  ·  {', '.join(pats)}" if pats else ""))


def show_problem_workspace():
    problem = st.session_state.problem

    show_problem_card(problem)

    similar = _get_similar_solved(problem)
    _show_similar_solved(similar)

    st.markdown("")
    c1, c2, c3 = st.columns([2, 2, 6])
    with c1:
        if st.button("✅ Mark as Solved", type="primary", use_container_width=True):
            _save_current_session("SOLVED")
            st.success("🎉 Saved! Problem marked as solved.")
            _reset_problem()
            st.rerun()
    with c2:
        if st.button("🏳️ Gave Up", use_container_width=True):
            _save_current_session("GAVE_UP")
            st.info("Session saved. Keep practising!")
            _reset_problem()
            st.rerun()

    st.markdown("---")

    tab_hints, tab_chat, tab_debug = st.tabs([
        "💡 Hints & Solution",
        "💬 Chat with Mentor",
        "🐛 Debug My Code",
    ])

    with tab_hints:
        show_hints_tab(problem)
    with tab_chat:
        show_free_chat_tab(problem)
    with tab_debug:
        show_debug_tab(problem)


def show_hints_tab(problem: dict):
    MIN_HINTS_FOR_SOLUTION = 3

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    hints_given = st.session_state.hints_given
    tier        = st.session_state.hint_tier

    st.markdown("---")
    col_h1, col_h2, col_h3 = st.columns([2, 2, 2])

    pending_action = None

    with col_h1:
        next_tier = min(tier + 1, 5)
        if tier < 5:
            if st.button(
                f"💡 Hint {hints_given + 1} — {HINT_TIER_LABELS.get(next_tier, '')}",
                type="primary", use_container_width=True
            ):
                pending_action = ("hint", next_tier)

    with col_h2:
        if st.button("❓ Ask a specific question", use_container_width=True):
            st.session_state["show_question_input"] = True

    with col_h3:
        locked = hints_given < MIN_HINTS_FOR_SOLUTION
        if locked:
            st.button(
                f"🔒 Solution (need {MIN_HINTS_FOR_SOLUTION - hints_given} more hints)",
                disabled=True, use_container_width=True,
            )
        else:
            if st.button("✅ Show Full Solution", type="secondary", use_container_width=True):
                pending_action = ("solution",)

    if st.session_state.get("show_question_input"):
        with st.form("custom_q_form", clear_on_submit=True):
            user_q = st.text_area(
                "Your question:",
                placeholder="e.g. Why do we need two pointers here?",
                height=80,
            )
            if st.form_submit_button("Send", type="primary"):
                if user_q:
                    st.session_state["show_question_input"] = False
                    pending_action = ("question", user_q)

    if hints_given > 0:
        st.markdown("")
        prog_cols = st.columns(5)
        for i in range(1, 6):
            label = HINT_TIER_LABELS[i].split(" ", 1)[1] if i in HINT_TIER_LABELS else f"Tier {i}"
            if i <= tier:
                prog_cols[i-1].markdown(f"✅ ~~{label}~~")
            else:
                prog_cols[i-1].markdown(f"⬜ {label}")

    # Execute outside column context so streaming fills full width
    if pending_action:
        if pending_action[0] == "hint":
            _give_hint(problem, pending_action[1])
        elif pending_action[0] == "solution":
            _give_solution(problem)
        elif pending_action[0] == "question":
            _ask_question(problem, pending_action[1])


def _give_hint(problem: dict, tier: int):
    from prompts.hints import get_hint_prompt

    problem_text = (
        f"Problem #{problem.get('number','?')}: {problem.get('title','?')}\n"
        f"Difficulty: {problem.get('difficulty','?')}\n"
        f"Tags: {', '.join(problem.get('tags', []))}\n\n"
        f"{problem.get('description', '')[:1500]}"
    )
    profile = load_profile()
    prompt  = get_hint_prompt(tier, problem_text, "I need a hint.")
    if profile:
        prompt = f"Student profile: {profile.summary_text()}\n\n" + prompt

    label = HINT_TIER_LABELS.get(tier, f"Tier {tier} Hint")
    st.session_state.chat_messages.append({"role": "user", "content": f"*(Requested: {label})*"})

    with st.chat_message("assistant"):
        hint_text = st.write_stream(_stream(prompt, temperature=0.3))

    response = f"**{label}**\n\n{hint_text}"
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.session_state.hint_tier   = tier
    st.session_state.hints_given += 1
    if tier >= 5:
        st.session_state.solution_unlocked = True

    agent = _get_agent()
    agent.chat_history.append({"role": "user",      "content": f"Give me hint tier {tier}"})
    agent.chat_history.append({"role": "assistant", "content": hint_text})
    st.rerun()


def _give_solution(problem: dict):
    problem_text = (
        f"Problem #{problem.get('number','?')}: {problem.get('title','?')}\n"
        f"Difficulty: {problem.get('difficulty','?')}\n\n"
        f"{problem.get('description','')[:2000]}"
    )
    prompt = f"""Provide a complete solution for this LeetCode problem in C++.

{problem_text}

Include:
1. **Approach** — which pattern/technique and why
2. **Algorithm** — step-by-step walkthrough
3. **C++ Solution** — complete working code using STL with inline comments
4. **Complexity** — time and space analysis
5. **Walkthrough** — trace through Example 1
6. **Edge Cases** — what to watch out for

Use proper C++ style: correct types, STL containers, and necessary headers.
"""
    st.session_state.chat_messages.append({"role": "user", "content": "*(Requested full solution)*"})

    with st.chat_message("assistant"):
        solution = st.write_stream(_stream(prompt, temperature=0.2))

    response = f"**Full Solution**\n\n{solution}"
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.session_state.solution_unlocked = True
    st.rerun()


def _ask_question(problem: dict, question: str):
    from prompts.mentor import build_mentor_prompt
    from rag.retrievers import retrieve_all

    problem_text = (
        f"Problem #{problem.get('number','?')}: {problem.get('title','?')}\n"
        f"{problem.get('description','')[:1000]}"
    )
    rag     = retrieve_all(question)
    profile = load_profile()
    system  = build_mentor_prompt(
        user_profile_summary=profile.summary_text() if profile else "",
        current_problem=problem_text,
    )
    history_text = "\n".join(
        f"{'User' if m['role']=='user' else 'Mentor'}: {m['content'][:300]}"
        for m in st.session_state.chat_messages[-6:]
    )
    full_prompt = (
        f"{system}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Student's question: {question}"
    )
    st.session_state.chat_messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        answer = st.write_stream(_stream(full_prompt, temperature=0.3))

    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
    st.rerun()


# ═══════════════════════════════════════════════════════════════
#  FREE CHAT TAB
# ═══════════════════════════════════════════════════════════════

def show_free_chat_tab(problem: dict | None):
    st.markdown("### 💬 Chat with Your Mentor")
    if problem:
        st.caption(
            "Ask anything about the problem, patterns, complexity, or DSA concepts. "
            "The mentor uses your profile and past sessions to give personalised answers."
        )
    else:
        st.caption(
            "Ask anything — DSA theory, patterns, complexity, which problems you've solved, "
            "or just get guidance. No problem selected needed."
        )

    for msg in st.session_state.free_chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask your mentor anything…")
    if user_input:
        st.session_state.free_chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        agent = _get_agent()
        if problem and not agent.current_problem:
            agent.set_current_problem(
                f"#{problem.get('number','?')}: {problem.get('title','?')} [{problem.get('difficulty','?')}]"
            )

        with st.chat_message("assistant"):
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            from prompts.mentor import build_mentor_prompt
            from rag.retrievers import retrieve_all

            # Build the same prompt the agent would use, but stream it
            rag_results     = retrieve_all(user_input, chat_history=agent.chat_history)
            similar_probs   = agent._format_problems(rag_results.get("problems", []))
            session_ctx     = agent._format_sessions(rag_results.get("sessions", []))
            pattern_history = agent._get_pattern_history(user_input)
            profile_summary = agent._get_profile_summary()

            system_prompt = build_mentor_prompt(
                user_profile_summary=profile_summary,
                similar_problems=similar_probs,
                session_context=session_ctx,
                current_problem=agent.current_problem or "",
                pattern_history=pattern_history,
            )

            lc_messages = [SystemMessage(content=system_prompt)]
            for turn in agent.chat_history[-6:]:
                if turn["role"] == "user":
                    lc_messages.append(HumanMessage(content=turn["content"]))
                elif turn["role"] == "assistant":
                    lc_messages.append(AIMessage(content=turn["content"]))
            lc_messages.append(HumanMessage(content=user_input))

            response = st.write_stream(_stream_messages(lc_messages, temperature=0.3))

        agent.chat_history.append({"role": "user",      "content": user_input})
        agent.chat_history.append({"role": "assistant", "content": response})
        st.session_state.free_chat_messages.append({"role": "assistant", "content": response})


# ═══════════════════════════════════════════════════════════════
#  DEBUG TAB
# ═══════════════════════════════════════════════════════════════

def show_debug_tab(problem: dict):
    st.markdown("### 🐛 Debug Your Code")
    st.markdown("Paste your code below. The mentor will find **all errors**, explain them, and show the corrected version.")

    lang = st.selectbox("Language", ["c++", "python", "java", "javascript"], index=0)

    cpp_placeholder = (
        "// Paste your C++ code here...\n"
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n\n"
        "class Solution {\n"
        "public:\n"
        "    // your attempt\n"
        "};"
    )
    placeholder = cpp_placeholder if lang == "c++" else f"// Paste your {lang} code here..."

    code  = st.text_area("Your Code", height=300, placeholder=placeholder, key="debug_code_input")
    user_q = st.text_input(
        "What's wrong? (optional)",
        placeholder="e.g. Wrong output on [1,2,3] or I get a segfault",
        key="debug_question",
    )

    if st.button("🔍 Analyze & Fix My Code", type="primary", use_container_width=True):
        if not code.strip():
            st.warning("Please paste your code first.")
        else:
            from prompts.hints import get_code_fix_prompt
            problem_ctx = ""
            if problem:
                problem_ctx = (
                    f"Problem #{problem.get('number','?')}: {problem.get('title','?')} "
                    f"[{problem.get('difficulty','?')}]\n"
                    f"{problem.get('description','')[:800]}"
                )
            prompt = get_code_fix_prompt(
                code=code,
                user_message=user_q or "Please find and fix all errors in my code.",
                problem_context=problem_ctx,
                language=lang,
            )
            st.markdown("---")
            st.markdown("### 🛠️ Analysis & Fix")
            result = st.write_stream(_stream(prompt, temperature=0.2))
            st.session_state.chat_messages.append({
                "role": "user",
                "content": f"Debug my code:\n```{lang}\n{code}\n```\n{user_q}",
            })
            st.session_state.chat_messages.append({"role": "assistant", "content": result})


# ═══════════════════════════════════════════════════════════════
#  SOLVE PAGE (top-level)
# ═══════════════════════════════════════════════════════════════

def show_solve_page():
    if st.session_state.problem is not None:
        show_problem_workspace()
        return

    tab_find, tab_chat = st.tabs(["🔍 Find a Problem", "💬 Chat with Mentor"])
    with tab_find:
        show_problem_search()
    with tab_chat:
        show_free_chat_tab(problem=None)


# ═══════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════

def show_dashboard_page():
    import plotly.express as px

    st.title("📊 Your Progress Dashboard")
    profile = load_profile()

    if not profile or profile.total_solved == 0:
        st.info("No data yet. Go to **Import CSV** to import your solved problems, or solve problems here.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Solved", profile.total_solved)
    c2.metric("Easy",   profile.by_difficulty.get("EASY", 0))
    c3.metric("Medium", profile.by_difficulty.get("MEDIUM", 0))
    c4.metric("Hard",   profile.by_difficulty.get("HARD", 0))
    st.markdown("---")

    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.subheader("Difficulty Distribution")
        fig = px.pie(
            values=[
                profile.by_difficulty.get("EASY", 0),
                profile.by_difficulty.get("MEDIUM", 0),
                profile.by_difficulty.get("HARD", 0),
            ],
            names=["Easy", "Medium", "Hard"],
            color_discrete_sequence=["#a6e3a1", "#f9e2af", "#f38ba8"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, width="stretch")

    with col_bar:
        st.subheader("Top 10 Patterns")
        mastery_data = []
        for pat, data in profile.pattern_mastery.items():
            solved = data.get("solved", 0) if isinstance(data, dict) else data.solved
            conf   = data.get("confidence", "LOW") if isinstance(data, dict) else data.confidence
            mastery_data.append({
                "Pattern": pat.replace("_", " ").title(),
                "Solved":  solved,
                "Level":   conf,
            })
        if mastery_data:
            df = pd.DataFrame(mastery_data).sort_values("Solved", ascending=False).head(10)
            color_map = {"LOW": "#f38ba8", "MEDIUM": "#f9e2af", "HIGH": "#a6e3a1"}
            fig2 = px.bar(df, x="Solved", y="Pattern", orientation="h",
                          color="Level", color_discrete_map=color_map)
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, width="stretch")

    st.markdown("---")
    sc, wc = st.columns(2)
    with sc:
        st.subheader("✅ Strong Patterns")
        for p in profile.strong_patterns:
            st.success(p.replace("_", " ").title())
        if not profile.strong_patterns:
            st.info("Keep solving to build strengths!")
    with wc:
        st.subheader("⚠️ Needs Practice")
        for p in profile.weak_patterns:
            if st.button(f"🏋️ {p.replace('_',' ').title()}", key=f"dash_weak_{p}", use_container_width=True):
                prob = _get_pattern_problem(p)
                if prob:
                    _reset_problem()
                    st.session_state.problem      = prob
                    st.session_state.current_page = "solve"
                    st.rerun()
        if not profile.weak_patterns:
            st.success("No weak areas identified!")

    if mastery_data:
        st.markdown("---")
        st.subheader("Pattern Mastery Table")
        df_full = pd.DataFrame(mastery_data).sort_values("Solved", ascending=False)
        st.dataframe(df_full, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🕒 Recent Activity")
    recent = profile.recent_activity
    if recent:
        recent_df = pd.DataFrame(recent[:20])
        cols_to_show = [c for c in ["solved_at", "title", "difficulty", "patterns", "outcome"] if c in recent_df.columns]
        st.dataframe(recent_df[cols_to_show], use_container_width=True, hide_index=True)
    else:
        st.info("No recent activity yet.")

    st.markdown("---")
    st.subheader("📝 Session Logs")
    if USER_DATA_PATH.exists():
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        logs = user_data.get("session_logs", [])
        if logs:
            st.dataframe(pd.DataFrame(logs[-20:][::-1]), use_container_width=True, hide_index=True)
        else:
            st.info("No sessions logged yet. Solve problems and mark them done to see logs here.")


# ═══════════════════════════════════════════════════════════════
#  IMPORT / SYNC PAGE
# ═══════════════════════════════════════════════════════════════

def show_sync_page():
    st.title("📥 Import / Sync Problems")

    profile = load_profile()
    if profile and profile.total_solved > 0:
        st.success(
            f"Currently stored: **{profile.total_solved}** problems  |  "
            f"Last updated: {profile.last_updated or 'Unknown'}"
        )

    tab_sync, tab_csv = st.tabs(["🔄 Quick Sync (LeetCode API)", "📄 Bulk Import (CSV)"])

    with tab_sync:
        _show_quick_sync_tab()

    with tab_csv:
        _show_csv_import_tab()


# ── Quick Sync tab ────────────────────────────────────────────────

def _show_quick_sync_tab():
    st.markdown("### 🔄 Sync via LeetCode API")
    st.markdown(
        "Enter your public LeetCode username. The app fetches **all your accepted "
        "submissions**, compares them against the local database, and appends only the "
        "problems that aren't stored yet — no duplicates, no data loss."
    )
    st.info("ℹ️ Your LeetCode profile must be **public** for this to work.", icon="ℹ️")

    with st.form("quick_sync_form"):
        lc_user = st.text_input(
            "LeetCode username",
            placeholder="e.g.  tarushb",
            help="Must match your exact LeetCode public username",
        )
        go = st.form_submit_button("🚀 Sync Now", type="primary")

    if go and lc_user.strip():
        _run_api_sync(lc_user.strip())


def _run_api_sync(lc_username: str):
    """
    API-based incremental sync:
      1. Fetch ALL accepted submissions for `lc_username` via public GraphQL.
      2. Load existing stored problems.
      3. Identify every problem not yet indexed (by slug).
      4. Tag patterns for new problems only.
      5. Merge with existing records, rebuild profile + FAISS index.
    """
    from ingestion.leetcode_client import LeetCodeClient
    from ingestion.problem_parser import PatternTagger, raw_to_problem_record, ProblemRecord
    from ingestion.profile_builder import load_problems, build_profile, save_data
    from rag.embeddings import create_index
    from langchain_core.documents import Document

    bar = st.progress(0, "Connecting to LeetCode API…")

    try:
        client = LeetCodeClient()

        # ── 1. Fetch ALL solved problems ─────────────────────────────────────────
        bar.progress(10, f"Fetching all solved problems for '{lc_username}'…")
        all_raw = client.get_all_solved(lc_username)

        if not all_raw:
            st.error(
                f"No accepted submissions found for **{lc_username}**. "
                "Check the username and make sure the profile is public."
            )
            return

        st.info(f"Found **{len(all_raw)}** unique solved problem(s) on LeetCode.")

        # ── 2. Load existing stored problems ────────────────────────────────────
        bar.progress(25, "Comparing against local database…")
        existing_raw = load_problems()

        def _slug(p: dict) -> str:
            return (
                p.get("titleSlug") or p.get("title_slug")
                or p.get("problem_id")
                or p.get("title", "").lower().replace(" ", "-")
            ).lower().strip()

        existing_slugs = {_slug(p) for p in existing_raw}

        # ── 3. Identify every problem not yet indexed ────────────────────────────
        new_raw = [r for r in all_raw if _slug(r).lower() not in existing_slugs]

        if not new_raw:
            st.success(
                f"✅ Already up to date! All {len(all_raw)} solved problems are already "
                f"in the database. ({len(existing_raw)} total stored)"
            )
            return

        st.info(f"Found **{len(new_raw)}** new problem(s) to add out of {len(all_raw)} total.")

        # ── 4. Tag patterns for new problems only ────────────────────────────────
        bar.progress(35, f"Tagging patterns for {len(new_raw)} new problem(s)…")
        tagger = PatternTagger()
        if not tagger._available:
            tagger = None
            st.caption("Using rule-based pattern tagging (LLM not available).")

        new_records = []
        total_new = len(new_raw)
        for i, raw in enumerate(new_raw, 1):
            record = raw_to_problem_record(raw, tagger, index=i, total=total_new)
            new_records.append(record)
            pct = 35 + int(35 * i / total_new)
            bar.progress(pct, f"Tagged {i}/{total_new}: {record.title}")

        # ── 5. Reconstruct existing records ─────────────────────────────────────
        existing_records = []
        for p in existing_raw:
            try:
                existing_records.append(ProblemRecord(**p))
            except Exception:
                pass

        # ── 6. Merge + rebuild profile + FAISS ───────────────────────────────────
        bar.progress(75, "Rebuilding profile…")
        all_records = existing_records + new_records
        profile     = build_profile(all_records)
        save_data(profile, all_records)

        bar.progress(90, "Updating search index…")
        docs = [
            Document(
                page_content=p.to_document_text(),
                metadata={
                    "problem_id": p.problem_id,
                    "title":      p.title,
                    "difficulty": p.difficulty,
                    "topic_tags": p.topic_tags,
                    "patterns":   p.patterns,
                },
            )
            for p in all_records
        ]
        try:
            create_index(docs, "problems")
        except Exception as e:
            st.warning(f"Index warning (non-critical): {e}")

        bar.progress(100, "Done!")
        st.success(
            f"✅ Sync complete! Added **{len(new_records)}** new problem(s).  "
            f"Total stored: **{len(all_records)}**  |  "
            f"Easy: {profile.by_difficulty['EASY']} | "
            f"Medium: {profile.by_difficulty['MEDIUM']} | "
            f"Hard: {profile.by_difficulty['HARD']}"
        )

        if new_records:
            with st.expander(f"📋 {len(new_records)} new problem(s) added"):
                for r in new_records:
                    st.markdown(f"- **{r.title}** [{r.difficulty}] — `{', '.join(r.patterns)}`")

        st.session_state.agent = None  # reset agent to pick up refreshed profile

    except Exception as e:
        st.error(f"Sync failed: {e}")


# ── Bulk CSV Import tab ───────────────────────────────────────────

def _show_csv_import_tab():
    st.markdown("### 📄 Bulk Import from CSV")
    st.markdown(
        "Upload a full CSV export of your solved problems. Useful for first-time "
        "setup or re-importing after a fresh install. The import checks **all rows** "
        "in the file and appends only problems not already stored."
    )

    with st.expander("📄 Expected CSV format"):
        st.markdown("""
Your CSV needs at least these columns (exact names are flexible):

| Column | Aliases accepted |
|--------|-----------------|
| `title` | `question_title` |
| `difficulty` | `level` |
| `title_slug` | `titleslug`, `slug` |
| `id` | `question_id`, `frontend_question_id` |
| `tags` | `topic_tags`, `related_topics` |

Tags should be comma-separated (e.g. `Array, Hash Table`).

```
id,title,difficulty,title_slug,tags
1,Two Sum,Easy,two-sum,"Array, Hash Table"
121,Best Time to Buy and Sell Stock,Easy,best-time-to-buy-and-sell-stock,Array
```
        """)

    uploaded = st.file_uploader("Upload solved problems CSV", type=["csv"])
    if uploaded:
        _run_csv_sync(uploaded)


def _run_csv_sync(uploaded_file):
    """
    Incremental CSV sync — parses the file, checks the last 20 rows for new
    problems not already in the database, tags and appends only those.
    """
    from ingestion.leetcode_client import LeetCodeClient
    from ingestion.problem_parser import PatternTagger, raw_to_problem_record, ProblemRecord
    from ingestion.profile_builder import load_problems, build_profile, save_data
    from rag.embeddings import create_index
    from langchain_core.documents import Document
    import tempfile, os

    bar = st.progress(0, "Reading CSV…")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_all = LeetCodeClient.import_from_csv(tmp_path)
        os.unlink(tmp_path)

        if not raw_all:
            st.error("No problems found in the CSV file. Check the format.")
            return

        st.info(f"CSV contains **{len(raw_all)}** problems.")
        bar.progress(20, "Checking for new problems…")

        existing_raw = load_problems()

        def _slug(p: dict) -> str:
            return (
                p.get("titleSlug") or p.get("title_slug")
                or p.get("problem_id")
                or p.get("title", "").lower().replace(" ", "-")
            ).lower().strip()

        existing_slugs = {_slug(p) for p in existing_raw}
        new_raw        = [r for r in raw_all if _slug(r).lower() not in existing_slugs]

        if not new_raw:
            st.success(
                f"✅ Already up to date! All {len(raw_all)} problem(s) from the CSV are "
                f"already in the database. ({len(existing_raw)} total stored)"
            )
            return

        st.info(f"Found **{len(new_raw)}** new problem(s) to add.")
        bar.progress(40, f"Tagging patterns…")

        tagger = PatternTagger()
        if not tagger._available:
            tagger = None
            st.caption("Using rule-based pattern tagging (LLM not available).")

        new_records = []
        total_new = len(new_raw)
        for i, raw in enumerate(new_raw, 1):
            record = raw_to_problem_record(raw, tagger, index=i, total=total_new)
            new_records.append(record)
            bar.progress(40 + int(30 * i / total_new), f"Tagged {i}/{total_new}: {record.title}")

        existing_records = []
        for p in existing_raw:
            try:
                existing_records.append(ProblemRecord(**p))
            except Exception:
                pass

        bar.progress(75, "Building profile…")
        all_records = existing_records + new_records
        profile     = build_profile(all_records)
        save_data(profile, all_records)

        bar.progress(90, "Updating search index…")
        docs = [
            Document(
                page_content=p.to_document_text(),
                metadata={
                    "problem_id": p.problem_id,
                    "title":      p.title,
                    "difficulty": p.difficulty,
                    "topic_tags": p.topic_tags,
                    "patterns":   p.patterns,
                },
            )
            for p in all_records
        ]
        try:
            create_index(docs, "problems")
        except Exception as e:
            st.warning(f"Index warning (non-critical): {e}")

        bar.progress(100, "Done!")
        st.success(
            f"✅ Import complete! Added **{len(new_records)}** new problem(s). "
            f"Total stored: **{len(all_records)}**  |  "
            f"Easy: {profile.by_difficulty['EASY']} | "
            f"Medium: {profile.by_difficulty['MEDIUM']} | "
            f"Hard: {profile.by_difficulty['HARD']}"
        )
        st.session_state.agent = None

    except Exception as e:
        st.error(f"Import failed: {e}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    show_sidebar()

    page = st.session_state.current_page
    if page == "solve":
        show_solve_page()
    elif page == "dashboard":
        show_dashboard_page()
    elif page == "sync":
        show_sync_page()


if __name__ == "__main__":
    main()
