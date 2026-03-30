"""
Hint Templates — progressive hint system (5 tiers) + code fix + question lookup prompts.
"""

#  5-Tier Progressive Hint System 

HINT_TIERS = {
    1: {
        "name": "Pattern Nudge",
        "description": "Just name the algorithmic pattern, no details.",
        "prompt": """The student is working on this problem and needs a gentle nudge.

Problem: {problem}
Student's message: {user_message}

Give a TIER 1 hint — the most abstract level:
- Only mention the general algorithmic PATTERN or TECHNIQUE that applies (e.g., "This is a sliding window problem" or "Think about using BFS here").
- Do NOT explain how to apply it.
- Do NOT give any code or pseudocode.
- Keep it to 1-2 sentences.
- Be encouraging.""",
    },
    2: {
        "name": "Approach Direction",
        "description": "Explain the high-level approach without implementation details.",
        "prompt": """The student needs more guidance on this problem.

Problem: {problem}
Student's message: {user_message}

Give a TIER 2 hint — approach direction:
- Explain the HIGH-LEVEL approach (e.g., "Use two pointers starting from both ends").
- Mention what data structure to use and WHY.
- Do NOT give pseudocode or code.
- Ask a guiding question to help them think further.
- 3-5 sentences max.""",
    },
    3: {
        "name": "Detailed Strategy",
        "description": "Step-by-step strategy with pseudocode outline.",
        "prompt": """The student is struggling and needs a more detailed hint.

Problem: {problem}
Student's message: {user_message}

Give a TIER 3 hint — detailed strategy:
- Break the solution into clear STEPS (numbered).
- Provide PSEUDOCODE outline (not actual code).
- Explain the key insight or trick.
- Mention time/space complexity.
- Still leave the actual implementation to the student.""",
    },
    4: {
        "name": "Implementation Guide",
        "description": "Partial code with key parts left for the student.",
        "prompt": """The student needs significant help with implementation.

Problem: {problem}
Student's message: {user_message}

Give a TIER 4 hint — implementation guide:
- Show a C++ code TEMPLATE with the structure filled in.
- Leave KEY PARTS as comments for the student to fill (e.g., "// TODO: update the window here").
- Explain what each section does.
- Include the C++ function signature with correct types (e.g., vector<int>, unordered_map).
- Return type must match the problem.""",
    },
    5: {
        "name": "Full Solution Walkthrough",
        "description": "Complete solution with detailed explanation.",
        "prompt": """The student has tried multiple times and needs the full solution.

Problem: {problem}
Student's message: {user_message}

Give a TIER 5 hint — full solution walkthrough:
- Show the COMPLETE working solution in C++.
- Use proper C++ STL (vector, unordered_map, stack, etc.).
- Add comments explaining each important line.
- Walk through the solution with an example.
- Explain time and space complexity.
- Mention common edge cases.
- Suggest similar problems to practice.""",
    },
}


def get_hint_prompt(tier: int, problem: str, user_message: str) -> str:
    """Get the formatted hint prompt for a specific tier."""
    tier = max(1, min(5, tier))
    template = HINT_TIERS[tier]["prompt"]
    return template.format(problem=problem, user_message=user_message)


#  Code Fix Prompt 

CODE_FIX_PROMPT = """You are an expert code debugger and DSA mentor. The student has C++ code with errors and needs help fixing it.

## Student's Code
```{language}
{code}
```

## Problem Context
{problem_context}

## Student's Question
{user_message}

## Instructions
1. **Carefully analyze the code** line by line.
2. **List ALL errors found** — be thorough:
   - Syntax errors
   - Logical errors (wrong algorithm, wrong conditions)
   - Edge case failures (empty input, single element, overflow)
   - Off-by-one errors
   - Wrong data structure / STL usage
   - Missing return statements
   - Incorrect variable initialization
   - Integer overflow (use long long when needed)
3. **For each error**, explain:
   - What line has the error
   - What's wrong
   - Why it's wrong
   - What the fix is
4. **Show the CORRECTED C++ code** with `// FIXED: ...` comments on changed lines.
5. **Walk through the fix** with a small example to prove it works.
6. **Mention the time/space complexity** of the corrected solution.

Be thorough but encouraging. The goal is for the student to LEARN from their mistakes."""


def get_code_fix_prompt(
    code: str,
    user_message: str,
    problem_context: str = "",
    language: str = "python",
) -> str:
    """Build the code fix prompt."""
    return CODE_FIX_PROMPT.format(
        code=code,
        user_message=user_message,
        problem_context=problem_context or "Not specified",
        language=language,
    )


#  LeetCode Question Lookup Prompt 

QUESTION_LOOKUP_PROMPT = """The student is asking about a LeetCode problem. Provide the full problem description and initial guidance.

## Problem Identifier
{problem_identifier}

## Instructions
1. **Show the full problem description** including:
   - Problem title and number (if known)
   - Problem statement (the actual question)
   - Example inputs and outputs (at least 2 examples)
   - Constraints (input size limits, value ranges)
2. **Identify the pattern(s)** that apply to this problem (e.g., Two Pointers, BFS, DP).
3. **Rate the difficulty** and explain why.
4. **Ask the student** what approach they're thinking of BEFORE giving hints.
5. **Do NOT give the solution** — just present the problem and ask for their thoughts.

If you recognize the problem, provide accurate details. If the identifier is ambiguous, mention the most likely match."""


def get_question_lookup_prompt(problem_identifier: str) -> str:
    """Build the question lookup prompt for a LeetCode problem name/number."""
    return QUESTION_LOOKUP_PROMPT.format(problem_identifier=problem_identifier)
