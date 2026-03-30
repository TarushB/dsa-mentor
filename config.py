"""
DSA Mentor — Central Configuration
All constants, paths, model names, and taxonomy in one place.
Single-user installation: one data file, flat FAISS indexes, no backups.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

#  Paths 
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_indexes"

# Single-user data store (profile + problems + session logs in one JSON)
USER_DATA_PATH = DATA_DIR / "user_data.json"

# Offline problem description cache (parsed from output_leetcode_questions.txt)
OFFLINE_PROBLEMS_PATH     = DATA_DIR / "offline_problems.json"
LEETCODE_QUESTIONS_PATH   = DATA_DIR / "output_leetcode_questions.txt"
DESCRIPTIONS_DIR          = DATA_DIR / "descriptions"   # batch txt files

# Ensure base directories exist
DATA_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)


#  LeetCode 
LEETCODE_GRAPHQL_URL    = "https://leetcode.com/graphql"
LEETCODE_REQUEST_DELAY  = 1.5  # seconds between API calls

#  Models 
# Ollama LLM — requires `ollama serve`
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embedding model — free, local via HuggingFace / sentence-transformers
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION  = 384

#  Pattern Taxonomy 
TAXONOMY = [
    "two_pointers",
    "sliding_window",
    "binary_search",
    "prefix_sum",
    "monotonic_stack",
    "bfs",
    "dfs",
    "backtracking",
    "dp_1d",
    "dp_2d",
    "dp_interval",
    "greedy",
    "union_find",
    "trie",
    "heap_kth_element",
    "graph_topological",
    "bit_manipulation",
    "math",
    "divide_and_conquer",
    "segment_tree",
    "binary_indexed_tree",
    "graph_shortest_path",
    "hashing",
    "linked_list",
    "stack_queue",
    "intervals",
    "string_matching",
    "recursion",
    "simulation",
    "design",
]

#  Confidence Thresholds 
CONFIDENCE_LOW_MAX = 3      # < 3 solved -> LOW
CONFIDENCE_MED_MAX = 7      # 3-7 solved -> MEDIUM
                             # > 7 solved -> HIGH

#  RAG Settings 
RETRIEVAL_K_PROBLEMS = 3     # top-k similar problems to retrieve
RETRIEVAL_K_SESSIONS = 2     # top-k past sessions to retrieve

#  Hint System 
MIN_HINTS_BEFORE_SOLUTION = 3   # require 3 hints before showing solution
MAX_HINT_TIERS = 5              # total hint tiers (1=abstract ... 5=solution walkthrough)
