"""
DSA Mentor — Saari settings ek jagah
Saare constants, paths, model names, aur taxonomy yahi milenge.
"""
import os
from pathlib import Path

# ─── Paths — saare important folders yahan define hain ─────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = DATA_DIR / "faiss_indexes"
PROFILE_PATH = DATA_DIR / "user_profile.json"

# Pehle check karo ki folders exist karte hain, nahi toh bana do
DATA_DIR.mkdir(exist_ok=True)
FAISS_DIR.mkdir(exist_ok=True)
(FAISS_DIR / "problems").mkdir(exist_ok=True)
(FAISS_DIR / "concepts").mkdir(exist_ok=True)
(FAISS_DIR / "sessions").mkdir(exist_ok=True)

# ─── LeetCode API ki settings ───────────────────────────────────
LEETCODE_GRAPHQL_URL = "https://leetcode.com/graphql"
LEETCODE_REQUEST_DELAY = 1.5  # API calls ke beech itna wait karo (seconds)

# ─── Models — sab free hain, local chalte hain Ollama + HuggingFace se ──
# Ollama LLM — pattern tagging aur concept generation ke liye use hota hai
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Embedding model — free hai, locally HuggingFace se chalta hai
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384

# ─── Pattern Taxonomy — yeh saare patterns hain jo hum track karte hain ──
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

# ─── Confidence Thresholds — kitne solve kiye toh kitna confidence ──
CONFIDENCE_LOW_MAX = 3      # 3 se kam solve kiya toh LOW
CONFIDENCE_MED_MAX = 7      # 3 se 7 ke beech MEDIUM
                             # 7 se zyada toh HIGH boss

# ─── RAG Settings — retrieval ke saare numbers yahan hain ───────
CHUNK_SIZE = 300             # ek chunk mein kitne tokens rakhne hain
CHUNK_OVERLAP = 50           # chunks ke beech kitna overlap hoga
RETRIEVAL_K_PROBLEMS = 3     # kitne similar problems laane hain
RETRIEVAL_K_SESSIONS = 2     # kitne purane sessions laane hain
RETRIEVAL_K_CONCEPTS = 3     # kitne concept chunks laane hain

# ─── Memory / Backup — FAISS indexes ka backup system ───────────
MAX_INDEX_BACKUPS = 3        # kitne purane backups rakhne hain FAISS ke
