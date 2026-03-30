# 🧠 DSA Mentor — Personalized LeetCode Learning with RAG

An AI-powered DSA (Data Structures & Algorithms) mentor that uses **Retrieval Augmented Generation (RAG)** to provide personalized, Socratic-style guidance based on your LeetCode solving history.

> **Privacy-First & Multi-Provider Options** — Runs fully local via [Ollama](https://ollama.ai) (LLaMA 3, Qwen, DeepSeek) for complete privacy, or switch to Google Gemini via API for high-speed cloud inference. Always uses local AI for embedding search via `BAAI/bge-small-en-v1.5`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (Phase 4 - DONE)             │
│  [Dark Mode/Rich Aesthetics] [Chat Interface] [User Dashboard]   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     ORCHESTRATION LAYER (Phase 3 - DONE)         │
│               [Router Agent] — Classifies Intent                 │
│               [Hint Chain]   — Progresses through 5 Tiers        │
│               [Mentor Agent] — Manages context & history         │
└──────┬──────────────────┬───────────────────────┬───────────────┘
       │                  │                       │
┌──────▼──────┐  ┌────────▼────────┐  ┌──────────▼──────────────┐
│  MEMORY     │  │   RAG ENGINE    │  │    KNOWLEDGE BASE       │
│  MODULE     │  │   (FAISS)       │  │    (Local JSON Storage) │
│ - Session   │  │ - Problem Index │  │ - 18 DSA Concepts       │
│   History   │  │ - Concept Index │  │ - 50k+ Offline Problems │
│ - User      │  │ - Hybrid BM25   │  │ - Pattern Mastery Logs  │
│   Profile   │  │                 │  │                         │
└─────────────┘  └─────────────────┘  └─────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    DATA INGESTION LAYER (Phases 1-2 - DONE)      │
│  [LeetCode Auto-Pull] → [LLM Pattern Tagger] → [Profile Builder] │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running ([install guide](https://ollama.ai)) — *Optional if using Gemini*
3. Pull a local model:
   ```bash
   ollama pull llama3.2
   ```

### Setup Directory

```bash
git clone https://github.com/TarushB/dsa-mentor.git
cd dsa-mentor

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env to add your Gemini API key (optional) or LeetCode cookies
```

### Running the App

Start the main Streamlit application directly:
```bash
streamlit run app.py
```

### Optional: Ingesting Data

The app is capable of reading data directly from LeetCode or rebuilding local FAISS indexes manually if you update the raw JSON files.

**Option A — Pull from LeetCode:**
```bash
python scripts/ingest_leetcode.py --mode graphql --username YOUR_USERNAME
```

**Option B — Update Offline Vector DB:**
```bash
# Build concept knowledge base
python scripts/build_concept_index.py 
# Build problem index
python scripts/build_problem_index.py
```

---

## ✅ Completed Project Phases

### Phase 1: Data Ingestion & LeetCode Integration
- `ingestion/leetcode_client.py`: GraphQL scraper with cookie auth + CSV fallback import.
- `ingestion/problem_parser.py`: Uses an LLM to accurately assign DSA patterns (18-pattern taxonomy) to random problems.
- `ingestion/profile_builder.py`: Generates the continuous `user_data.json` profile tracker computing your confidence levels.

### Phase 2: RAG Pipeline & Memory
- `rag/embeddings.py`: FAISS index management with local sentence-transformers locking data to disk.
- `rag/retrievers.py`: Assembles results based on prompt vectors vs semantic concept spaces.
- `scripts/build_concept_index.py`: Tools that convert raw text strings into numeric embeddings.

### Phase 3: Agent Logic & Orchestration
- `agents/router.py`: Categorizes incoming messages rapidly into specific pipeline flows (Code Fix, Hint, Explain).
- `agents/hint_chain.py`: **Anti-gaming logic.** Keeps students from cheating by enforcing a 5-tier hint wall. Require 3 hints before a solution is revealed.
- `agents/mentor_agent.py`: Collects your stats, retrieves relevant DB docs, determines the step, prompts the LLM, handles edge errors.

### Phase 4: Modern App Interface
- `app.py`: Deploys a **Premium Dark UI** using Streamlit with sidebar player cards, radar charts, session persistence, and markdown-rich streaming code blocks.

---

## 📋 What's Left (Future Enhancements)
- [ ] Add an Ace Editor (`streamlit-ace`) inside the chat structure for live code running.
- [ ] Add hallucination guards on RAG references to prevent made-up line numbers.
- [ ] Support monthly auto-cleaning of older chat session vectors.

---

## 🔧 Tech Stack

| Component | Technology | Cost |
|-----------|-----------|------|
| Base UI | Streamlit / Custom CSS | Free |
| LLMs | Ollama (Local) / Gemini API | Free / Freemium |
| Embeddings | BAAI/bge-small-en-v1.5 | Free / Local |
| Vector DB | FAISS (faiss-cpu) | Free |
| Framework | LangChain | Free |
