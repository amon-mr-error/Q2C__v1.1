"""
Q2C PDF Ingestion Streamlit Frontend
A user-friendly interface for PDF text extraction and chunking.
"""
#Web UI for PDF Ingestion & Chunking
import streamlit as st
import pandas as pd
import json
import os
from io import BytesIO, StringIO
from typing import List, Dict, Union, Optional
from dotenv import load_dotenv

# Import the ingest module
from ingest import ingest_pdf, extract_text_from_pdf

# Load environment variables
load_dotenv()

# -----------------------------------------------------------------------------
# Page Configuration & Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Q2C - Intelligent RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    with open("ux.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def count_words(text: str) -> int:
    return len(text.split())

def convert_df_to_csv(df: pd.DataFrame) -> str:
    return df.to_csv(index=True)

def create_json_export(chunks: List[Dict]) -> bytes:
    return json.dumps(chunks, indent=2).encode('utf-8')

# -----------------------------------------------------------------------------
# Sidebar & Initial Setup
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=60)
    st.markdown("### Q2C")
    st.caption("v2.0 • Mistral RAG Engine")
    
    st.divider()
    
    st.markdown("#### ⚙️ Configuration")
    with st.expander("Processing Settings", expanded=False):
        chunk_size = st.slider("Chunk Size", 100, 5000, 1000, 100)
        overlap = st.slider("Overlap", 0, 1000, 200, 50)
        top_k = st.slider("Top-K Results", 4, 10, 6, 1,
                          help="Number of passages retrieved per query")
        search_type = st.radio("Search Mode", ["mmr", "similarity"], index=0,
                               help="MMR diversifies results; Similarity is pure cosine")
        password = st.text_input("PDF Password", type="password")
        if overlap >= chunk_size:
            st.error("Overlap must be < Chunk Size")

    st.markdown("#### 🔐 API Keys")
    env_api_key = os.getenv("MISTRAL_API_KEY")
    default_key = env_api_key if env_api_key else ""
    mistral_api_key = st.text_input("Mistral API Key", value=default_key, type="password", help="Leave empty for local mode")
    
    st.divider()
    
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()

    st.caption("Powered by LangGraph & Mistral")

# -----------------------------------------------------------------------------
# Main Application Logic
# -----------------------------------------------------------------------------

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# Layout
st.title("Query2Clause")
st.markdown("#### Intelligent Document Analysis & Reasoning")

# Tab Navigation
tab_main, tab_data = st.tabs(["💬 Assistant", "📂 Documents & Data"])

# -----------------------------------------------------------------------------
# TAB 1: Chat Assistant
# -----------------------------------------------------------------------------
with tab_main:
    # Check if data is ready
    if not st.session_state.processed:
        st.info("👋 Welcome! Please upload documents in the **Documents** tab to start.")
        
        # Quick access to upload if empty
        with st.expander("🚀 Quick Upload", expanded=True):
            uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
            if uploaded_files:
                if st.button("Process Docs", type="primary"):
                    with st.status("Processing documents...", expanded=True) as status:
                        all_chunks = []
                        for file in uploaded_files:
                            status.write(f"Reading {file.name}...")
                            bytes_data = file.getvalue()
                            chunks = ingest_pdf(bytes_data, chunk_size, overlap, password if password else None)
                            for c in chunks:
                                c['meta']['filename'] = file.name
                            all_chunks.extend(chunks)
                        
                        st.session_state.chunks = all_chunks
                        st.session_state.processed = True
                        
                        # Build Graph
                        if mistral_api_key:
                            status.write("Building Knowledge Graph...")
                            try:
                                import graph_rag
                                rag_graph = graph_rag.RAGGraph(
                                    all_chunks,
                                    api_key=mistral_api_key,
                                    k=top_k,
                                    search_type=search_type,
                                )
                                st.session_state.rag_graph = rag_graph
                                status.update(label="✅ Ready to Chat!", state="complete", expanded=False)
                                st.rerun()
                            except Exception as e:
                                status.update(label="⚠️ Graph Build Failed", state="error")
                                st.error(f"Error: {e}")
                        else:
                            status.update(label="✅ Ready (Local Mode)", state="complete")
                            st.rerun()

    else:
        # Chat Interface
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg:
                    with st.expander("📚 Sources"):
                        for idx, doc in enumerate(msg["sources"]):
                            st.markdown(f"**Source {idx+1}** (Page {doc.metadata.get('page', '?')+1})")
                            st.markdown(f"```text\n{doc.page_content}\n```")

        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if 'rag_graph' not in st.session_state:
                    st.error("Knowledge Graph not initialized. Please re-process documents.")
                else:
                    try:
                        # ── Phase 1: Rewrite + Retrieve (show spinner) ──────────
                        with st.status("⚡ Retrieving relevant passages…", expanded=False) as status:
                            stream = st.session_state.rag_graph.stream_run(prompt)

                            # First event is always the metadata dict
                            meta = next(iter(stream))
                            docs       = meta.get("documents", [])
                            rewritten  = meta.get("rewritten", False)
                            used_query = meta.get("question", prompt)

                            # Update status now that retrieval is done
                            count = len(docs)
                            status.update(
                                label=f"✅ Retrieved {count} relevant passage{'s' if count != 1 else ''}",
                                state="complete",
                                expanded=False,
                            )

                        # Show rewritten query hint immediately
                        if rewritten and used_query != prompt:
                            st.info(f"🔍 **Query optimised to:** *{used_query}*")

                        # Show sources panel above the answer (populated before generation)
                        if docs:
                            with st.expander(f"📚 Sources Referenced ({count} passages)"):
                                for idx, doc in enumerate(docs):
                                    page_no = doc.metadata.get("page", -1) + 1
                                    fname   = doc.metadata.get("filename", "document")
                                    st.caption(f"**Source {idx+1}** • {fname} • Page {page_no}")
                                    st.code(doc.page_content, language="text")
                        else:
                            st.warning("⚠️ No sufficiently relevant passages found.")

                        # ── Phase 2: Streaming generation ──────────────────────
                        answer_parts = []
                        answer_placeholder = st.empty()

                        for event in stream:     # consume remaining token / done events
                            if event.get("type") == "token":
                                answer_parts.append(event["content"])
                                # Update display with every token chunk
                                answer_placeholder.markdown("".join(answer_parts) + "▌")
                            elif event.get("type") == "done":
                                break

                        full_answer = "".join(answer_parts)
                        answer_placeholder.markdown(full_answer)   # final render (no cursor)

                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_answer,
                            "sources": docs,
                            "rewritten_query": used_query if rewritten else None,
                        })

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 2: Data Exploration
# -----------------------------------------------------------------------------

#tab_data 
with tab_data:
    st.markdown("### 📂 Document Management")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Upload new documents here to update the Knowledge Base.")
        uploaded_files = st.file_uploader("Add Documents", type=["pdf"], accept_multiple_files=True, key="data_upload")
        if uploaded_files and st.button("Process & Update", type="primary"):
             with st.status("Updating Knowledge Base...") as status:
                all_chunks = []
                for file in uploaded_files:
                    status.write(f"Ingesting {file.name}...")
                    chunks = ingest_pdf(file.getvalue(), chunk_size, overlap, password)
                    for c in chunks: c['meta']['filename'] = file.name
                    all_chunks.extend(chunks)
                
                st.session_state.chunks = all_chunks
                st.session_state.processed = True
                
                # Rebuild Graph
                import graph_rag
                status.write("Rebuilding Graph Index...")
                rg = graph_rag.RAGGraph(
                    all_chunks,
                    api_key=mistral_api_key,
                    k=top_k,
                    search_type=search_type,
                )
                st.session_state.rag_graph = rg
                status.update(label="✅ Updated Successfully!", state="complete")
                st.rerun()

    with col2:
        if st.session_state.processed:
            chunks = st.session_state.chunks
            st.metric("Total Documents Processed", len(set(c['meta'].get('filename') for c in chunks)))
            st.metric("Total Text Chunks", len(chunks))
            
            with st.expander("🔍 Inspect Chunks"):
                df = pd.DataFrame([
                    {"File": c['meta'].get('filename'), "Page": c['meta']['page']+1, "Text": c['text']}
                    for c in chunks
                ])
                st.dataframe(df, use_container_width=True)
                
            st.download_button("📥 Export JSON", create_json_export(chunks), "knowledge_base.json", "application/json")
