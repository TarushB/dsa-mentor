"""
DSA Mentor — Central Configuration
All constants, paths, model names, and taxonomy in one place.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_indexes"
PROFILE_PATH = DATA_DIR / "user_profile.json"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)
(FAISS_DIR / "problems").mkdir(exist_ok=True)
(FAISS_DIR / "concepts").mkdir(exist_ok=True)
(FAISS_DIR / "sessions").mkdir(exist_ok=True)

# ─── LeetCode ────────────────────────────────────────────────────
LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
LEETCODE_REQUEST_DELAY = 1.5  # seconds between API calls

# ─── Models (all free / local via Ollama + HuggingFace) ──────────
# Ollama LLM — used for pattern tagging & concept generation
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embedding model — free, local via HuggingFace / sentence-transformers
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

# ─── Pattern Taxonomy ────────────────────────────────────────────
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
]

# ─── Confidence Thresholds ───────────────────────────────────────
CONFIDENCE_LOW_MAX = 3      # < 3 solved → LOW
CONFIDENCE_MED_MAX = 7      # 3–7 solved → MEDIUM
                             # > 7 solved → HIGH

# ─── RAG Settings ────────────────────────────────────────────────
CHUNK_SIZE = 300             # tokens per concept chunk
CHUNK_OVERLAP = 50           # overlap between concept chunks
RETRIEVAL_K_PROBLEMS = 3     # top-k similar problems to retrieve
RETRIEVAL_K_SESSIONS = 2     # top-k past sessions to retrieve
RETRIEVAL_K_CONCEPTS = 3     # top-k concept chunks to retrieve

# ─── Memory / Backup ────────────────────────────────────────────
MAX_INDEX_BACKUPS = 3        # rolling backup count for FAISS indexes
