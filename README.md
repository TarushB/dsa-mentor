# 🧠 DSA Mentor — Personalized LeetCode Learning with RAG

An AI-powered DSA (Data Structures & Algorithms) mentor that uses **Retrieval Augmented Generation** to provide personalized, Socratic-style guidance based on your LeetCode solving history.

> **100% Free & Local** — Uses [Ollama](https://ollama.ai) for LLM inference and [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) for embeddings. No API keys or paid services required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (Phase 4)                    │
│  [Problem Input] [Chat Interface] [Progress Dashboard] [Notes]   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     ORCHESTRATION LAYER (Phase 3)                │
│              LangChain Agent / ReAct Pipeline                    │
│   [Router Agent] → [Hint Chain] / [Gap-Bridge Chain] / [RAG]    │
└──────┬──────────────────┬───────────────────────┬───────────────┘
       │                  │                       │
┌──────▼──────┐  ┌────────▼────────┐  ┌──────────▼──────────────┐
│  MEMORY     │  │   RAG ENGINE    │  │    KNOWLEDGE BASE        │
│  MODULE     │  │   ✅ DONE       │  │     ✅ DONE              │
│  ✅ DONE    │  │                 │  │                          │
│ - Session   │  │ FAISS Index:    │  │ - 18 DSA Concept Notes   │
│   write-    │  │ - Problems      │  │ - Prerequisite Patterns  │
│   back      │  │ - Concepts      │  │ - LeetCode Problem Meta  │
│ - Profile   │  │ - Sessions      │  │                          │
│   updates   │  │                 │  │                          │
│ - Atomic    │  │ Hybrid:         │  └──────────────────────────┘
│   saves     │  │ BM25 + FAISS    │
└─────────────┘  └─────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    DATA INGESTION LAYER ✅ DONE                  │
│  [LeetCode GraphQL Scraper] → [Pattern Tagger] → [Profile]      │
│  [CSV Fallback Import]                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✅ What's Done (Phase 1 & 2)

### Phase 1: Data Ingestion & LeetCode Integration

| Module | Description |
|--------|-------------|
| `ingestion/leetcode_client.py` | GraphQL scraper with cookie auth + CSV fallback import |
| `ingestion/problem_parser.py` | `ProblemRecord` schema + Ollama-powered pattern tagger (18-pattern taxonomy) with rule-based fallback |
| `ingestion/profile_builder.py` | `UserProfile` with pattern mastery computation, confidence levels, and JSON persistence |
| `scripts/ingest_leetcode.py` | CLI entry point: `--mode graphql\|csv` |

### Phase 2: RAG Pipeline & Memory

| Module | Description |
|--------|-------------|
| `rag/embeddings.py` | FAISS index management with atomic saves, rolling backups (3 snapshots), and singleton BGE embeddings |
| `rag/retrievers.py` | 3 retrievers: **Problem** (metadata-filtered FAISS), **Concept** (BM25+FAISS ensemble), **Session** (semantic + pattern filter) |
| `rag/memory.py` | Session write-back to FAISS + incremental user profile updates |
| `scripts/build_concept_index.py` | Pre-written notes for all 18 patterns + optional Ollama enhancement |
| `scripts/build_problem_index.py` | Converts solved problems into FAISS problem index |

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running ([install guide](https://ollama.ai))
3. Pull a model:
   ```bash
   ollama pull llama3.2
   ```

### Setup

```bash
cd dsa-mentor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac
# Edit .env with your LeetCode cookies (optional — CSV import works without them)
```

### Ingesting Data

**Option A — From LeetCode (requires auth cookies in `.env`):**
```bash
python scripts/ingest_leetcode.py --mode graphql --username YOUR_LEETCODE_USERNAME
```

**Option B — From CSV (no auth needed):**
```bash
# Use the included sample data to test
python scripts/ingest_leetcode.py --mode csv --csv-path tests/sample_data.csv

# Or use your own CSV with columns: id, title, title_slug, difficulty, tags
```

**Option C — Without Ollama (rule-based fallback):**
```bash
python scripts/ingest_leetcode.py --mode csv --csv-path tests/sample_data.csv --no-llm
```

### Building FAISS Indexes

```bash
# Build concept knowledge base (18 DSA patterns)
python scripts/build_concept_index.py           # with Ollama
python scripts/build_concept_index.py --no-llm  # without Ollama

# Build problem index from ingested data
python scripts/build_problem_index.py
```

### Output

After running, you'll find:
```
data/
├── user_profile.json          # Your skill profile
├── solved_problems.json       # Parsed problem records
└── faiss_indexes/
    ├── problems/              # FAISS index of solved problems
    ├── concepts/              # FAISS index of DSA concept notes
    └── sessions/              # FAISS index of chat sessions (built over time)
```

---

## 📋 What's Left (Future Phases)

### Phase 3: Agent Logic & Prompt Engineering (Week 3)
- [ ] **Router Agent** — classifies new problems as READY / GAP_BRIDGE / REVIEW
- [ ] **Progressive Hint Chain** — 5-tier Socratic hint system (never gives away solutions)
- [ ] **Knowledge Gap Bridge** — generates "Pit Stop" learning modules for missing prerequisites
- [ ] **LangChain ReAct Agent** — wires retrievers as tools for the reasoning loop

### Phase 4: Streamlit UI (Week 4)
- [ ] **Sidebar** — user profile card, progress heatmap, session history
- [ ] **Problem Solver tab** — chat interface with hint tier indicator
- [ ] **Knowledge Base tab** — browse concept notes
- [ ] **Analytics tab** — pattern mastery radar chart
- [ ] **Code editor** — `streamlit-ace` for sharing attempts
- [ ] **"I solved it!" button** — triggers outcome logging and reinforcement

### Phase 5: Polish & Production (Week 5)
- [ ] Cold start assessment (5-question skill test for new users)
- [ ] Anti-gaming logic (minimum 3 hints before solution unlock)
- [ ] Hallucination guard (verify all referenced problems exist in solved DB)
- [ ] Session analytics and learning velocity tracking
- [ ] Monthly FAISS re-indexing for index hygiene

---

## 🏗️ Project Structure

```
dsa-mentor/
├── .env.example              # API keys & LeetCode session template
├── config.py                 # Taxonomy, models, paths, thresholds
├── requirements.txt          # All-free dependencies
├── data/                     # Generated data (gitignored)
│   ├── user_profile.json
│   ├── solved_problems.json
│   └── faiss_indexes/
├── ingestion/                # Phase 1: Data acquisition
│   ├── leetcode_client.py
│   ├── problem_parser.py
│   └── profile_builder.py
├── rag/                      # Phase 2: RAG pipeline
│   ├── embeddings.py
│   ├── retrievers.py
│   └── memory.py
├── agents/                   # Phase 3: Agent logic (TODO)
├── prompts/                  # Phase 3: Prompt templates (TODO)
├── scripts/                  # CLI tools
│   ├── ingest_leetcode.py
│   ├── build_concept_index.py
│   └── build_problem_index.py
├── ui/                       # Phase 4: Streamlit app (TODO)
└── tests/
    └── sample_data.csv       # 25 sample LeetCode problems
```

---

## 🔧 Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| LLM | Ollama (llama3.2) | Free / Local |
| Embeddings | BAAI/bge-small-en-v1.5 | Free / Local |
| Vector DB | FAISS (faiss-cpu) | Free |
| Framework | LangChain | Free |
| Hybrid Search | BM25 + FAISS ensemble | Free |
| Data Source | LeetCode GraphQL + CSV | Free |
