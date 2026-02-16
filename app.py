"""
Q2C PDF Ingestion Streamlit Frontend
A user-friendly interface for PDF text extraction and chunking.
"""

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
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        
        /* Main Container */
        .main {
            background-color: #ffffff;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Chat Messages */
        .stChatMessage {
            background-color: #f8fafc;
            border-radius: 12px;
            border: 1px solid #edf2f7;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        [data-testid="chatAvatarIcon-user"] {
            background-color: #3b82f6;
        }
        [data-testid="chatAvatarIcon-assistant"] {
            background-color: #10b981;
        }

        /* Cards and Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            color: #1e293b;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #0f172a;
            font-weight: 700;
        }
        
        /* Custom Source Box */
        .source-box {
            background-color: #f1f5f9;
            border-left: 4px solid #3b82f6;
            padding: 0.75rem;
            margin-top: 0.5rem;
            border-radius: 0 4px 4px 0;
            font-size: 0.9rem;
            color: #334155;
        }
    </style>
    """, unsafe_allow_html=True)

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
                                rag_graph = graph_rag.RAGGraph(all_chunks, api_key=mistral_api_key)
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
                    # Fallback if graph missing
                    st.error("Knowledge Graph not initialized. Please re-process documents.")
                else:
                    message_placeholder = st.empty()
                    with st.status("Thinking...", expanded=False) as status:
                        try:
                            status.write("Retrieving context...")
                            response = st.session_state.rag_graph.run(prompt)
                            
                            answer = response.get("generation", "No answer generated.")
                            docs = response.get("documents", [])
                            
                            status.update(label="Generated Answer", state="complete")
                            
                            message_placeholder.markdown(answer)
                            
                            # Show sources
                            if docs:
                                with st.expander("📚 Sources Referenced"):
                                    for idx, doc in enumerate(docs):
                                        st.caption(f"**Source {idx+1}** • Page {doc.metadata.get('page', 'Unknown')+1} • {doc.metadata.get('filename', 'doc')}")
                                        st.code(doc.page_content, language="text")
                            
                            # Save to history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer,
                                "sources": docs
                            })
                            
                        except Exception as e:
                            status.update(label="Error", state="error")
                            st.error(f"An error occurred: {str(e)}")


# -----------------------------------------------------------------------------
# TAB 2: Data Exploration
# -----------------------------------------------------------------------------
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
                rg = graph_rag.RAGGraph(all_chunks, api_key=mistral_api_key)
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
