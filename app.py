import streamlit as st
import json
import plotly.express as px
from pathlib import Path

from config import DATA_DIR
from ingestion.profile_builder import load_profile
from rag.retrievers import retrieve_all

# Streamlit page ka setup karte hain yahan
st.set_page_config(page_title="DSA Mentor Testing", page_icon="🧠", layout="wide")

st.title("🧠 DSA Mentor - Backend Test Interface")
st.markdown("This is a quick testing interface to explore the Phase 1 & 2 components you've built.")

# Sidebar mein user ki profile dikhaenge
with st.sidebar:
    st.header("👤 User Profile")
    try:
        profile = load_profile()
        st.subheader(f"Username: {profile.username}")
        st.metric("Total Solved", profile.total_solved)
        
        # Difficulty wise kitne solve kiye, woh dikhao
        st.write("### Difficulty Breakdown")
        cols = st.columns(3)
        cols[0].metric("Easy", profile.by_difficulty.get("EASY", 0))
        cols[1].metric("Medium", profile.by_difficulty.get("MEDIUM", 0))
        cols[2].metric("Hard", profile.by_difficulty.get("HARD", 0))

        # Pattern mastery ka data yahan render hoga
        st.write("### Pattern Mastery")
        mastery_data = []
        for pattern, data in profile.pattern_mastery.items():
            mastery_data.append({"Pattern": pattern, "Solved": data["solved"], "Confidence": data["confidence"]})
        
        if mastery_data:
            import pandas as pd
            df = pd.DataFrame(mastery_data).sort_values(by="Solved", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        st.write("### Strong Patterns")
        if profile.strong_patterns:
            st.success(", ".join(profile.strong_patterns))
        else:
            st.info("None yet")
            
        st.write("### Weak Patterns")
        if profile.weak_patterns:
            st.warning(", ".join(profile.weak_patterns))
        else:
            st.info("None yet")

    except Exception as e:
        st.error(f"Failed to load profile: {e}")

# Main body — yahan RAG retrieval test karenge
st.header("🔍 Multi-Index RAG Retrieval")
st.markdown("Test the hybrid FAISS + BM25 retrieval system across all three indexes simultaneously.")

# Chat history ko session state mein init karo agar pehle se nahi hai
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter a query, problem description, or concept:", "How to reverse a linked list")
col1, col2 = st.columns(2)
with col1:
    filter_tags = st.text_input("Optional: Filter by Problem Tags (comma-separated)", "")
with col2:
    filter_patterns = st.text_input("Optional: Filter by Patterns (comma-separated)", "")

col_search, col_clear = st.columns([3, 1])
with col_search:
    search_clicked = st.button("Search All Indexes", type="primary")
with col_clear:
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Dikhao kitne messages hain abhi chat history mein
if st.session_state.chat_history:
    st.caption(f"💬 Chat history: {len(st.session_state.chat_history)} messages (query rewriter active)")

if search_clicked:
    with st.spinner("Querying FAISS Problem, Concept, and Session indexes..."):
        # Filters ko process karo — comma se split karke list banao
        tags_list = [t.strip() for t in filter_tags.split(",")] if filter_tags else None
        pats_list = [p.strip() for p in filter_patterns.split(",")] if filter_patterns else None

        # Chat history ke saath retrieval chalao, query rewriting bhi hogi
        results = retrieve_all(
            query,
            chat_history=st.session_state.chat_history,
            problem_tags=tags_list,
            problem_patterns=pats_list,
        )

        # Agar query rewrite hui hai toh user ko dikhao kya badla
        rewritten = results.get("rewritten_query", query)
        if rewritten != query:
            st.info(f"🔄 **Query rewritten:** \"{query}\" → \"{rewritten}\"")

        # Is exchange ko chat history mein add karo
        st.session_state.chat_history.append({"role": "user", "content": query})
        # Jo retrieve hua uska summary banao assistant ke context ke liye
        concept_names = [doc.metadata.get("pattern", "") for doc in results.get("concepts", [])]
        problem_names = [doc.metadata.get("title", "") for doc in results.get("problems", [])]
        assistant_summary = f"Retrieved concepts: {', '.join(concept_names)}. Problems: {', '.join(problem_names)}."
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_summary})

        tab1, tab2, tab3 = st.tabs(["📚 Problems", "💡 Concepts", "💬 Sessions"])

        with tab1:
            st.subheader(f"Problem Matches ({len(results.get('problems', []))})")
            for i, doc in enumerate(results.get("problems", [])):
                with st.expander(f"{i+1}. {doc.metadata.get('title', 'Unknown Problem')} [{doc.metadata.get('difficulty', 'Unknown')}]"):
                    st.write("**Tags:**", ", ".join(doc.metadata.get("topic_tags", [])))
                    st.write("**Patterns:**", ", ".join(doc.metadata.get("patterns", [])))
                    st.text_area("Snippet", doc.page_content[:500] + "...", height=150, key=f"prob_{i}")

        with tab2:
            st.subheader(f"Concept Matches ({len(results.get('concepts', []))})")
            for i, doc in enumerate(results.get("concepts", [])):
                with st.expander(f"{i+1}. {doc.metadata.get('pattern', 'Unknown Concept')}"):
                    st.text_area("Content", doc.page_content, height=250, key=f"conc_{i}")

        with tab3:
            st.subheader(f"Session Matches ({len(results.get('sessions', []))})")
            if not results.get("sessions"):
                st.info("No past sessions found. (Memory index is currently empty)")
            for i, doc in enumerate(results.get("sessions", [])):
                with st.expander(f"Session {doc.metadata.get('timestamp', 'Unknown')}"):
                    st.text_area("Log", doc.page_content, height=150, key=f"sess_{i}")

