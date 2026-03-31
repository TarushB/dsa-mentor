#!/usr/bin/env python3
"""
RAG Evaluation Suite -- DSA Mentor
======================================================================
Measures retrieval quality, faithfulness, hallucination rate, answer
relevance, context precision, and intent classification accuracy.

All judgement is done via the local Ollama LLM (no external APIs).

Usage:
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --section retrieval
    python scripts/evaluate_rag.py --section intent
    python scripts/evaluate_rag.py --section faithfulness
    python scripts/evaluate_rag.py --output results/eval_run.json
"""

import sys
import json
import time
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS_THRESHOLD = 0.70   # score ≥ 70 % -> pass


# ==================================================================
#  Data structures
# ==================================================================

@dataclass
class CaseResult:
    query: str
    score: float
    passed: bool
    details: dict = field(default_factory=dict)


@dataclass
class SectionResult:
    name: str
    cases: List[CaseResult] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    @property
    def score(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.score for c in self.cases) / len(self.cases)

    @property
    def pass_rate(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed) / len(self.cases)


# ==================================================================
#  Test datasets
# ==================================================================

INTENT_CASES = [
    {"msg": "Give me a hint",                                           "expected": "HINT_REQUEST"},
    {"msg": "I'm stuck, can you nudge me?",                            "expected": "HINT_REQUEST"},
    {"msg": "How do I start this problem?",                            "expected": "HINT_REQUEST"},
    {"msg": "Show me the full solution",                               "expected": "SOLUTION_REQUEST"},
    {"msg": "I give up, just tell me the answer",                      "expected": "SOLUTION_REQUEST"},
    {"msg": "Explain dynamic programming",                              "expected": "CONCEPT_QUESTION"},
    {"msg": "When should I use BFS vs DFS?",                           "expected": "CONCEPT_QUESTION"},
    {"msg": "What is the sliding window technique?",                    "expected": "CONCEPT_QUESTION"},
    {"msg": "What is Two Sum?",                                         "expected": "QUESTION_LOOKUP"},
    {"msg": "42",                                                       "expected": "QUESTION_LOOKUP"},
    {"msg": "LeetCode 121",                                             "expected": "QUESTION_LOOKUP"},
    {"msg": "Tell me about Merge Intervals",                            "expected": "QUESTION_LOOKUP"},
    {"msg": "Hello!",                                                   "expected": "GENERAL_CHAT"},
    {"msg": "Thanks for the explanation",                               "expected": "GENERAL_CHAT"},
    {"msg": "Can you say more about that?",                             "expected": "GENERAL_CHAT"},
    {"msg": "```python\ndef two_sum(nums): pass\n```\nFix this",        "expected": "CODE_FIX"},
    {"msg": "My code crashes:\n```\nx = arr[n]\n```\nwhat's wrong?",   "expected": "CODE_FIX"},
]

# Queries where retrieval results should match at least one of the expected patterns
RETRIEVAL_CASES = [
    {
        "query":    "dynamic programming 1D array memoization",
        "patterns": ["dp_1d", "dp_2d"],
        "desc":     "DP retrieval",
    },
    {
        "query":    "two pointers sorted array pair sum",
        "patterns": ["two_pointers"],
        "desc":     "Two-pointer retrieval",
    },
    {
        "query":    "sliding window maximum subarray length",
        "patterns": ["sliding_window"],
        "desc":     "Sliding window retrieval",
    },
    {
        "query":    "binary search rotated sorted array find target",
        "patterns": ["binary_search"],
        "desc":     "Binary search retrieval",
    },
    {
        "query":    "BFS graph shortest path level order",
        "patterns": ["bfs", "graph_shortest_path"],
        "desc":     "BFS retrieval",
    },
    {
        "query":    "backtracking combinations subsets permutations",
        "patterns": ["backtracking"],
        "desc":     "Backtracking retrieval",
    },
    {
        "query":    "linked list reversal cycle detection",
        "patterns": ["linked_list", "two_pointers"],
        "desc":     "Linked list retrieval",
    },
]

# Queries used for faithfulness + answer relevance tests
FAITHFULNESS_CASES = [
    {
        "query":    "Explain the sliding window technique for arrays",
        "topic":    "sliding_window concept",
    },
    {
        "query":    "How do I approach dynamic programming problems?",
        "topic":    "dp approach",
    },
    {
        "query":    "What is the two-pointer technique?",
        "topic":    "two_pointers concept",
    },
    {
        "query":    "When should I use a hash map?",
        "topic":    "hashing concept",
    },
    {
        "query":    "How does BFS differ from DFS?",
        "topic":    "graph traversal concept",
    },
]


# ==================================================================
#  LLM judge helpers
# ==================================================================

def _call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Call the local Ollama LLM; returns empty string on failure."""
    try:
        from utils.llm import invoke_llm
        return invoke_llm(prompt, temperature=temperature)
    except Exception as e:
        return f"__LLM_ERROR__: {e}"


def _parse_json_score(text: str, key: str = "score") -> Optional[float]:
    """Extract a float score from an LLM JSON response."""
    # Try strict JSON first
    try:
        data = json.loads(text)
        return float(data[key])
    except Exception:
        pass
    # Fallback: regex
    match = re.search(rf'"{key}"\s*:\s*([0-9.]+)', text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


_FAITHFULNESS_PROMPT = """\
You are a strict evaluator of AI faithfulness.

Retrieved Context (what the AI had access to):
\"\"\"
{context}
\"\"\"

User Question: {question}
AI Response: {response}

Instructions:
1. List the factual claims in the AI response.
2. For each claim, judge whether it is supported by the retrieved context.
3. General well-known DSA algorithm facts (e.g. "BFS uses a queue") are acceptable even if not explicitly in the context.
4. Hallucinated specific details (wrong complexity, wrong step, invented problem names) count against faithfulness.

Score faithfulness 0.0–1.0:
  1.0 = All claims supported or standard DSA facts
  0.7 = Most claims supported, 1-2 minor unsupported details
  0.5 = Some unsupported claims
  0.2 = Many unsupported or contradicting claims
  0.0 = Response fabricates or contradicts context entirely

Reply with ONLY valid JSON, no markdown fences:
{{"score": 0.0, "unsupported_claims": ["..."], "reasoning": "one sentence"}}
"""

_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating answer relevance.

User Question: {question}
AI Response: {response}

Score how well the response answers the question (0.0–1.0):
  1.0 = Complete, direct answer
  0.7 = Mostly answers, minor gaps
  0.5 = Partial answer
  0.2 = Tangentially related, doesn't answer
  0.0 = Does not answer at all

Reply with ONLY valid JSON, no markdown fences:
{{"score": 0.0, "reasoning": "one sentence"}}
"""

_CONTEXT_RELEVANCE_PROMPT = """\
You are evaluating document relevance.

User Question: {question}

Retrieved Document:
\"\"\"
{document}
\"\"\"

Score how relevant this document is to helping answer the question (0.0–1.0):
  1.0 = Directly useful
  0.7 = Mostly useful context
  0.5 = Somewhat useful
  0.2 = Marginally useful
  0.0 = Not relevant

Reply with ONLY valid JSON, no markdown fences:
{{"score": 0.0, "reasoning": "one sentence"}}
"""

_HALLUCINATION_DETECT_PROMPT = """\
You are a hallucination detector for a DSA tutoring AI.

Retrieved Context:
\"\"\"
{context}
\"\"\"

AI Response:
\"\"\"
{response}
\"\"\"

Identify whether the AI response contains hallucinations -- statements that are:
- Factually wrong about DSA algorithms or data structures
- Invented problem names, numbers, or statistics not in the context
- Incorrect complexity claims (e.g., saying O(n) when O(n log n) is correct)

Respond with ONLY valid JSON:
{{"hallucination_detected": true/false, "severity": "none|minor|major", "examples": ["..."], "reasoning": "one sentence"}}
"""


# ==================================================================
#  Section evaluators
# ==================================================================

def eval_intent(verbose: bool = False) -> SectionResult:
    """Test intent classification accuracy."""
    result = SectionResult(name="Intent Classification")

    try:
        from agents.router import detect_intent, Intent
    except Exception as e:
        result.skipped = True
        result.skip_reason = f"Could not import router: {e}"
        return result

    _log("\n[Intent] Running intent classification tests...")
    for case in INTENT_CASES:
        msg      = case["msg"]
        expected = case["expected"]
        try:
            has_code = "```" in msg
            intent   = detect_intent(msg, has_code_block=has_code)
            actual   = intent.name
            correct  = actual == expected
            score    = 1.0 if correct else 0.0
            result.cases.append(CaseResult(
                query=msg[:60],
                score=score,
                passed=correct,
                details={"expected": expected, "actual": actual},
            ))
            if verbose or not correct:
                status = "[OK]" if correct else "[X]"
                print(f"  {status} [{expected:20s}] got=[{actual}]  msg={msg[:55]!r}")
        except Exception as e:
            result.cases.append(CaseResult(
                query=msg[:60], score=0.0, passed=False,
                details={"error": str(e)},
            ))
            print(f"  [X] ERROR: {e}  msg={msg[:55]!r}")

    return result


def eval_retrieval(verbose: bool = False) -> SectionResult:
    """Test retrieval precision -- does the top-k contain expected patterns?"""
    result = SectionResult(name="Retrieval Precision@K")

    try:
        from rag.retrievers import ProblemRetriever
        pr = ProblemRetriever()
        if pr.store is None:
            result.skipped = True
            result.skip_reason = "Problems FAISS index not found. Run scripts/build_problem_index.py first."
            return result
    except Exception as e:
        result.skipped = True
        result.skip_reason = f"Could not load retriever: {e}"
        return result

    _log("\n[Retrieval] Running retrieval precision tests...")
    for case in RETRIEVAL_CASES:
        query    = case["query"]
        expected = set(case["patterns"])
        desc     = case["desc"]
        try:
            docs = pr.retrieve(query, k=5)
            if not docs:
                result.cases.append(CaseResult(
                    query=query, score=0.0, passed=False,
                    details={"retrieved": 0, "desc": desc},
                ))
                print(f"  [X] [{desc}] No documents retrieved")
                continue

            # Precision: fraction of retrieved docs that match expected patterns
            hits = 0
            for doc in docs:
                doc_patterns = set(doc.metadata.get("patterns", []))
                if doc_patterns & expected:
                    hits += 1

            precision = hits / len(docs)
            hit_rate  = 1.0 if hits > 0 else 0.0
            # Combined score: hit_rate weighted more (0.6) + precision (0.4)
            score = 0.6 * hit_rate + 0.4 * precision
            passed = hit_rate == 1.0  # at least one relevant doc must appear

            titles = [d.metadata.get("title", "?") for d in docs]
            patterns_found = [list(d.metadata.get("patterns", [])) for d in docs]

            result.cases.append(CaseResult(
                query=query, score=score, passed=passed,
                details={
                    "desc": desc, "hits": hits, "total": len(docs),
                    "precision": round(precision, 2), "hit_rate": hit_rate,
                    "titles": titles, "patterns_found": patterns_found,
                },
            ))
            status = "[OK]" if passed else "[X]"
            print(f"  {status} [{desc}] hits={hits}/{len(docs)} precision={precision:.0%}  {titles[:3]}")
        except Exception as e:
            result.cases.append(CaseResult(
                query=query, score=0.0, passed=False,
                details={"error": str(e), "desc": desc},
            ))
            print(f"  [X] [{desc}] ERROR: {e}")

    return result


def eval_context_relevance(verbose: bool = False) -> SectionResult:
    """LLM-judge: are retrieved documents relevant to the query?"""
    result = SectionResult(name="Context Relevance")

    try:
        from rag.retrievers import ProblemRetriever
        pr = ProblemRetriever()
        if pr.store is None:
            result.skipped = True
            result.skip_reason = "Problems FAISS index not available."
            return result
    except Exception as e:
        result.skipped = True
        result.skip_reason = str(e)
        return result

    _log("\n[Context Relevance] LLM-judging retrieved document relevance...")

    # Use a subset of retrieval cases as context relevance queries
    queries = [c["query"] for c in RETRIEVAL_CASES[:5]]

    for query in queries:
        try:
            docs = pr.retrieve(query, k=3)
            if not docs:
                result.cases.append(CaseResult(
                    query=query, score=0.0, passed=False,
                    details={"note": "no docs retrieved"},
                ))
                continue

            doc_scores = []
            for doc in docs:
                prompt   = _CONTEXT_RELEVANCE_PROMPT.format(
                    question=query, document=doc.page_content[:600]
                )
                response = _call_llm(prompt)
                if "__LLM_ERROR__" in response:
                    doc_scores.append(0.5)  # neutral fallback
                    continue
                s = _parse_json_score(response, "score")
                doc_scores.append(s if s is not None else 0.5)

            avg_score = sum(doc_scores) / len(doc_scores)
            passed    = avg_score >= PASS_THRESHOLD

            result.cases.append(CaseResult(
                query=query[:60], score=avg_score, passed=passed,
                details={"doc_scores": [round(s, 2) for s in doc_scores]},
            ))
            status = "[OK]" if passed else "[X]"
            print(f"  {status} avg={avg_score:.2f}  per_doc={[round(s,2) for s in doc_scores]}  q={query[:55]!r}")
        except Exception as e:
            result.cases.append(CaseResult(
                query=query[:60], score=0.0, passed=False,
                details={"error": str(e)},
            ))
            print(f"  [X] ERROR: {e}")

    return result


def eval_faithfulness_and_relevance(verbose: bool = False) -> Tuple[SectionResult, SectionResult]:
    """
    Run the full RAG pipeline on test queries, then LLM-judge both:
      - Faithfulness  (is the answer grounded in the retrieved context?)
      - Answer Relevance (does the answer address the question?)
    Returns two SectionResult objects.
    """
    faith_result = SectionResult(name="Faithfulness")
    relev_result = SectionResult(name="Answer Relevance")

    try:
        from rag.retrievers import retrieve_all
    except Exception as e:
        msg = f"Could not import retrievers: {e}"
        faith_result.skipped = True
        faith_result.skip_reason = msg
        relev_result.skipped = True
        relev_result.skip_reason = msg
        return faith_result, relev_result

    try:
        from utils.llm import invoke_llm_messages
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        msg = f"Could not import LLM utils: {e}"
        faith_result.skipped = True
        faith_result.skip_reason = msg
        relev_result.skipped = True
        relev_result.skip_reason = msg
        return faith_result, relev_result

    _log("\n[Faithfulness / Answer Relevance] Running full RAG pipeline...")

    for case in FAITHFULNESS_CASES:
        query = case["query"]
        print(f"\n  Query: {query!r}")
        try:
            # Step 1: Retrieve context
            rag = retrieve_all(query, skip_rewrite=True)
            prob_docs = rag.get("problems", [])
            sess_docs = rag.get("sessions", [])

            context_parts = []
            for d in prob_docs:
                context_parts.append(d.page_content[:400])
            for d in sess_docs:
                context_parts.append(d.page_content[:300])
            context = "\n---\n".join(context_parts) if context_parts else "(no context retrieved)"

            # Step 2: Get answer from the LLM
            system = (
                "You are a DSA mentor. Answer the student's question clearly and concisely. "
                "Use the following context if relevant:\n\n" + context
            )
            messages = [
                SystemMessage(content=system),
                HumanMessage(content=query),
            ]
            answer = invoke_llm_messages(messages, temperature=0.3)
            print(f"  Answer ({len(answer)} chars): {answer[:120]}...")

            # Step 3: Judge faithfulness
            faith_prompt = _FAITHFULNESS_PROMPT.format(
                context=context[:1200], question=query, response=answer[:800]
            )
            faith_raw = _call_llm(faith_prompt)
            faith_score = _parse_json_score(faith_raw, "score")
            if faith_score is None:
                faith_score = 0.5
            faith_data = {}
            try:
                faith_data = json.loads(faith_raw)
            except Exception:
                pass

            # Step 4: Judge answer relevance
            relev_prompt = _ANSWER_RELEVANCE_PROMPT.format(question=query, response=answer[:800])
            relev_raw   = _call_llm(relev_prompt)
            relev_score = _parse_json_score(relev_raw, "score")
            if relev_score is None:
                relev_score = 0.5
            relev_data = {}
            try:
                relev_data = json.loads(relev_raw)
            except Exception:
                pass

            faith_result.cases.append(CaseResult(
                query=query, score=faith_score, passed=faith_score >= PASS_THRESHOLD,
                details={
                    "context_docs": len(context_parts),
                    "answer_len": len(answer),
                    "unsupported_claims": faith_data.get("unsupported_claims", []),
                    "reasoning": faith_data.get("reasoning", ""),
                },
            ))
            relev_result.cases.append(CaseResult(
                query=query, score=relev_score, passed=relev_score >= PASS_THRESHOLD,
                details={"reasoning": relev_data.get("reasoning", "")},
            ))

            f_status = "[OK]" if faith_score >= PASS_THRESHOLD else "[X]"
            r_status = "[OK]" if relev_score >= PASS_THRESHOLD else "[X]"
            print(f"  Faithfulness {f_status} {faith_score:.2f}  |  Relevance {r_status} {relev_score:.2f}")
            if faith_data.get("unsupported_claims"):
                print(f"  Unsupported claims: {faith_data['unsupported_claims']}")

        except Exception as e:
            print(f"  [X] ERROR: {e}")
            faith_result.cases.append(CaseResult(
                query=query, score=0.0, passed=False, details={"error": str(e)}
            ))
            relev_result.cases.append(CaseResult(
                query=query, score=0.0, passed=False, details={"error": str(e)}
            ))

    return faith_result, relev_result


def eval_hallucination(verbose: bool = False) -> SectionResult:
    """
    Detect hallucinations: run 3 targeted queries and check if the LLM
    fabricates information not in the retrieved context.
    Severity scoring: none=1.0, minor=0.6, major=0.0
    """
    result = SectionResult(name="Hallucination Detection")

    try:
        from rag.retrievers import retrieve_all
        from utils.llm import invoke_llm_messages
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        result.skipped = True
        result.skip_reason = str(e)
        return result

    halluc_queries = [
        "What is the time complexity of merge sort?",
        "Explain how Dijkstra's algorithm works step by step",
        "How many problems have I solved using dynamic programming?",
        "What are the problems I solved yesterday?",
    ]

    _log("\n[Hallucination] Testing for fabricated information...")
    SEVERITY_SCORE = {"none": 1.0, "minor": 0.6, "major": 0.1}

    for query in halluc_queries:
        try:
            rag = retrieve_all(query, skip_rewrite=True)
            prob_docs = rag.get("problems", [])
            sess_docs = rag.get("sessions", [])

            context_parts = [d.page_content[:400] for d in prob_docs]
            context_parts += [d.page_content[:300] for d in sess_docs]
            context = "\n---\n".join(context_parts) if context_parts else "(no context)"

            system = (
                "You are a DSA mentor. Answer based on retrieved context where applicable. "
                "If context does not contain the answer, say so rather than guessing.\n\n"
                "Context:\n" + context
            )
            messages = [SystemMessage(content=system), HumanMessage(content=query)]
            answer = invoke_llm_messages(messages, temperature=0.3)

            prompt = _HALLUCINATION_DETECT_PROMPT.format(
                context=context[:1200], response=answer[:800]
            )
            raw = _call_llm(prompt)
            data = {}
            try:
                data = json.loads(raw)
            except Exception:
                pass

            detected  = data.get("hallucination_detected", False)
            severity  = data.get("severity", "none").lower()
            score     = SEVERITY_SCORE.get(severity, 0.5)
            examples  = data.get("examples", [])
            reasoning = data.get("reasoning", "")

            result.cases.append(CaseResult(
                query=query, score=score, passed=score >= PASS_THRESHOLD,
                details={
                    "hallucination_detected": detected,
                    "severity": severity,
                    "examples": examples,
                    "reasoning": reasoning,
                },
            ))
            status = "[OK]" if score >= PASS_THRESHOLD else "[X]"
            print(f"  {status} [{severity:5s}] score={score:.2f}  q={query!r}")
            if examples:
                print(f"       examples: {examples}")
            if reasoning:
                print(f"       {reasoning}")

        except Exception as e:
            result.cases.append(CaseResult(
                query=query, score=0.0, passed=False, details={"error": str(e)}
            ))
            print(f"  [X] ERROR: {e}")

    return result


def eval_query_rewrite(verbose: bool = False) -> SectionResult:
    """Test that query rewriting produces standalone, searchable queries."""
    result = SectionResult(name="Query Rewriting")

    try:
        from rag.query_rewriter import rewrite_query
    except Exception as e:
        result.skipped = True
        result.skip_reason = str(e)
        return result

    rewrite_cases = [
        {
            "history": [
                {"role": "user",      "content": "Tell me about Two Sum"},
                {"role": "assistant", "content": "Two Sum requires finding two indices that add to target..."},
            ],
            "followup": "Can you give me a hint?",
            "should_contain_one_of": ["two sum", "hint", "approach"],
        },
        {
            "history": [
                {"role": "user",      "content": "Explain dynamic programming"},
                {"role": "assistant", "content": "DP breaks problems into subproblems..."},
            ],
            "followup": "Can you show an example?",
            "should_contain_one_of": ["dynamic programming", "dp", "example"],
        },
        {
            "history": [
                {"role": "user",      "content": "I'm working on sliding window maximum"},
                {"role": "assistant", "content": "Use a monotonic deque..."},
            ],
            "followup": "What's the time complexity?",
            "should_contain_one_of": ["sliding window", "time complexity", "complexity"],
        },
    ]

    _log("\n[Query Rewriting] Testing contextual query rewriting...")
    for case in rewrite_cases:
        followup = case["followup"]
        history  = case["history"]
        expected_keywords = case["should_contain_one_of"]
        try:
            rewritten = rewrite_query(followup, history)
            lower     = rewritten.lower()
            hit       = any(kw in lower for kw in expected_keywords)
            # Also check it's not identical to the followup (rewriting happened)
            was_rewritten = rewritten.strip().lower() != followup.strip().lower()

            score  = 1.0 if hit else (0.5 if was_rewritten else 0.2)
            passed = hit

            result.cases.append(CaseResult(
                query=followup, score=score, passed=passed,
                details={
                    "rewritten": rewritten,
                    "was_rewritten": was_rewritten,
                    "keyword_hit": hit,
                    "expected_keywords": expected_keywords,
                },
            ))
            status = "[OK]" if passed else "[X]"
            rw_note = "rewritten" if was_rewritten else "unchanged"
            print(f"  {status} [{rw_note}] {followup!r} -> {rewritten!r}")
        except Exception as e:
            result.cases.append(CaseResult(
                query=followup, score=0.0, passed=False, details={"error": str(e)}
            ))
            print(f"  [X] ERROR: {e}")

    return result


# ==================================================================
#  Report formatting
# ==================================================================

def _log(msg: str):
    print(msg, flush=True)


def print_report(sections: Dict[str, SectionResult], total_time: float):
    bar = "=" * 68
    print(f"\n{bar}")
    print(f"  DSA Mentor -- RAG Evaluation Report")
    print(f"{bar}")
    print(f"  {'Section':<30} {'Score':>7}  {'Pass Rate':>9}  {'Status':>8}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*8}")

    overall_scores = []
    for name, sec in sections.items():
        if sec.skipped:
            print(f"  {sec.name:<30}  {'SKIPPED':>7}  {'':>9}  {'--':>8}")
            print(f"    -> {sec.skip_reason}")
            continue
        score_pct   = sec.score * 100
        pass_pct    = sec.pass_rate * 100
        status      = "PASS" if sec.score >= PASS_THRESHOLD else "FAIL"
        status_icon = "[OK]" if status == "PASS" else "[X]"
        print(f"  {sec.name:<30}  {score_pct:>6.1f}%  {pass_pct:>8.1f}%  {status_icon} {status}")
        overall_scores.append(sec.score)

    if overall_scores:
        overall = sum(overall_scores) / len(overall_scores) * 100
        print(f"  {'-'*30}  {'-'*7}  {'-'*9}  {'-'*8}")
        overall_status = "PASS" if overall >= PASS_THRESHOLD * 100 else "FAIL"
        print(f"  {'OVERALL':<30}  {overall:>6.1f}%  {'':>9}  {'[OK]' if overall_status == 'PASS' else '[X]'} {overall_status}")

    print(f"{bar}")
    print(f"  Evaluation completed in {total_time:.1f}s")
    print(f"{bar}\n")

    # Failure details
    failures_printed = False
    for name, sec in sections.items():
        if sec.skipped:
            continue
        failed = [c for c in sec.cases if not c.passed]
        if failed:
            if not failures_printed:
                print("  -- Failures / Low Scores ------------------------------------")
                failures_printed = True
            print(f"\n  [{sec.name}]")
            for c in failed:
                print(f"    [X] score={c.score:.2f}  query={c.query!r}")
                for k, v in c.details.items():
                    if v and k not in ("titles", "patterns_found"):
                        print(f"       {k}: {v}")

    if failures_printed:
        print()


def save_results(sections: Dict[str, SectionResult], path: str, total_time: float):
    out = {
        "total_time_seconds": round(total_time, 2),
        "overall_score": None,
        "sections": {},
    }
    scores = []
    for name, sec in sections.items():
        if sec.skipped:
            out["sections"][sec.name] = {"skipped": True, "reason": sec.skip_reason}
            continue
        scores.append(sec.score)
        out["sections"][sec.name] = {
            "score": round(sec.score, 4),
            "pass_rate": round(sec.pass_rate, 4),
            "cases": [
                {
                    "query": c.query,
                    "score": round(c.score, 4),
                    "passed": c.passed,
                    "details": c.details,
                }
                for c in sec.cases
            ],
        }
    if scores:
        out["overall_score"] = round(sum(scores) / len(scores), 4)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Results saved -> {path}")


# ==================================================================
#  Main
# ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate the DSA Mentor RAG system")
    parser.add_argument(
        "--section",
        choices=["intent", "retrieval", "context", "faithfulness", "hallucination", "rewrite", "all"],
        default="all",
        help="Which evaluation section to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results (e.g. results/eval.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    run = args.section
    v   = args.verbose

    print("\n" + "=" * 68)
    print("  DSA Mentor -- RAG Evaluation Suite")
    print("=" * 68)
    print(f"  Sections: {run}  |  Verbose: {v}")
    print(f"  Project root: {ROOT}")
    print("=" * 68)

    sections: Dict[str, SectionResult] = {}
    t0 = time.time()

    if run in ("intent", "all"):
        sections["intent"] = eval_intent(verbose=v)

    if run in ("retrieval", "all"):
        sections["retrieval"] = eval_retrieval(verbose=v)

    if run in ("context", "all"):
        sections["context"] = eval_context_relevance(verbose=v)

    if run in ("faithfulness", "all"):
        faith, relev = eval_faithfulness_and_relevance(verbose=v)
        sections["faithfulness"]     = faith
        sections["answer_relevance"] = relev

    if run in ("hallucination", "all"):
        sections["hallucination"] = eval_hallucination(verbose=v)

    if run in ("rewrite", "all"):
        sections["query_rewrite"] = eval_query_rewrite(verbose=v)

    total_time = time.time() - t0
    print_report(sections, total_time)

    if args.output:
        save_results(sections, args.output, total_time)
    else:
        # Auto-save to results/
        ts  = time.strftime("%Y%m%d_%H%M%S")
        out = ROOT / "results" / f"rag_eval_{ts}.json"
        save_results(sections, str(out), total_time)


if __name__ == "__main__":
    main()
