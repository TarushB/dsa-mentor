import streamlit as st
import json
import plotly.express as px
from pathlib import Path

from config import DATA_DIR
from ingestion.profile_builder import load_profile
from rag.retrievers import retrieve_all

# Configure the Streamlit page
st.set_page_config(page_title="DSA Mentor Testing", page_icon="🧠", layout="wide")

st.title("🧠 DSA Mentor - Backend Test Interface")
st.markdown("This is a quick testing interface to explore the Phase 1 & 2 components you've built.")

# Sidebar - User Profile
with st.sidebar:
    st.header("👤 User Profile")
    try:
        profile = load_profile()
        st.subheader(f"Username: {profile.username}")
        st.metric("Total Solved", profile.total_solved)
        
        # Difficulty Breakdown
        st.write("### Difficulty Breakdown")
        cols = st.columns(3)
        cols[0].metric("Easy", profile.by_difficulty.get("EASY", 0))
        cols[1].metric("Medium", profile.by_difficulty.get("MEDIUM", 0))
        cols[2].metric("Hard", profile.by_difficulty.get("HARD", 0))

        # Mastery Data
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

# Main Body - RAG Retrieval Testing
st.header("🔍 Multi-Index RAG Retrieval")
st.markdown("Test the hybrid FAISS + BM25 retrieval system across all three indexes simultaneously.")

query = st.text_input("Enter a query, problem description, or concept:", "How to reverse a linked list")
col1, col2 = st.columns(2)
with col1:
    filter_tags = st.text_input("Optional: Filter by Problem Tags (comma-separated)", "")
with col2:
    filter_patterns = st.text_input("Optional: Filter by Patterns (comma-separated)", "")

if st.button("Search All Indexes", type="primary"):
    with st.spinner("Querying FAISS Problem, Concept, and Session indexes..."):
        # Process filters
        tags_list = [t.strip() for t in filter_tags.split(",")] if filter_tags else None
        pats_list = [p.strip() for p in filter_patterns.split(",")] if filter_patterns else None
        
        # Run retrieval
        results = retrieve_all(query, problem_tags=tags_list, problem_patterns=pats_list)
        
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
