"""
Q2C PDF Ingestion Streamlit Frontend
A user-friendly interface for PDF text extraction and chunking.
"""

import streamlit as st
import pandas as pd
import json
from io import BytesIO, StringIO
from typing import List, Dict, Union, Optional

# Import the ingest module from the same directory
from ingest import ingest_pdf, extract_text_from_pdf


# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Q2C PDF Ingestion",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())

def count_chars(text: str) -> int:
    """Count characters in a text string."""
    return len(text)

def convert_df_to_csv(df: pd.DataFrame) -> str:
    """Convert DataFrame to CSV format for download."""
    return df.to_csv(index=True)

def create_json_export(chunks: List[Dict]) -> bytes:
    """Create JSON export of chunks."""
    return json.dumps(chunks, indent=2).encode('utf-8')

def create_text_export(chunks: List[Dict]) -> str:
    """Create plain text export of all chunks."""
    text_parts = []
    for i, chunk in enumerate(chunks):
        text_parts.append(f"--- Chunk {i+1} (Page {chunk['meta']['page'] + 1}) ---\n")
        text_parts.append(chunk['text'])
        text_parts.append("\n\n")
    return ''.join(text_parts)

# -----------------------------------------------------------------------------
# Sidebar - Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

st.sidebar.subheader("Chunking Parameters")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=100,
    max_value=5000,
    value=1000,
    step=100,
    help="Number of characters per chunk"
)

overlap = st.sidebar.slider(
    "Overlap Size",
    min_value=0,
    max_value=1000,
    value=200,
    step=50,
    help="Number of characters to overlap between chunks"
)

# Validation
if overlap >= chunk_size:
    st.sidebar.error(f"⚠️ Overlap ({overlap}) must be smaller than chunk size ({chunk_size})")
else:
    st.sidebar.success(f"✓ Valid configuration")

st.sidebar.subheader("PDF Password (Optional)")
password = st.sidebar.text_input("Password for encrypted PDFs", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Q2C PDF Ingestion**\n\n"
    "Upload PDF files, extract text, and create chunks for embedding models.\n\n"
    "Built with ❤️ using Streamlit"
)


# -----------------------------------------------------------------------------
# Main Content Area
# -----------------------------------------------------------------------------
st.title("📄 Q2C PDF Ingestion")
st.markdown("Extract text from PDFs and create overlapping text chunks for downstream processing.")


# -----------------------------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------------------------
st.subheader("📤 Upload PDFs")

uploaded_files = st.file_uploader(
    "Drag and drop PDF files here",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more PDF files to process"
)

if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded")
    
    # Display uploaded files
    with st.expander("View uploaded files"):
        for i, uploaded_file in enumerate(uploaded_files):
            st.text(f"{i+1}. {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")


# -----------------------------------------------------------------------------
# Processing Section
# -----------------------------------------------------------------------------
st.subheader("🚀 Process PDFs")

if uploaded_files:
    process_button = st.button(
        "📚 Process PDF(s)",
        type="primary",
        help="Extract text and create chunks from uploaded PDFs"
    )
    
    if process_button:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_chunks = []
        total_pages = 0
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing: {uploaded_file.name}...")
                progress_bar.progress((i + 0.5) / len(uploaded_files))
                
                # Read file bytes
                file_bytes = uploaded_file.getvalue()
                
                # Process PDF
                chunks = ingest_pdf(
                    source=file_bytes,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    password=password if password else None
                )
                
                # Add filename to meta
                for chunk in chunks:
                    chunk['meta']['filename'] = uploaded_file.name
                
                all_chunks.extend(chunks)
                
                # Count pages
                try:
                    from ingest import extract_text_from_pdf
                    pages = extract_text_from_pdf(file_bytes, password=password if password else None)
                    total_pages += len(pages)
                except:
                    pass
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text(f"✓ Processing complete! Generated {len(all_chunks)} chunks from {total_pages} pages.")
            progress_bar.progress(100)
            
            # Store results in session state
            st.session_state['chunks'] = all_chunks
            st.session_state['total_pages'] = total_pages
            st.session_state['processed'] = True
            
        except Exception as e:
            st.error(f"❌ Error processing PDFs: {str(e)}")
            st.session_state['processed'] = False
            
else:
    st.info("👆 Upload PDF files to get started")


# -----------------------------------------------------------------------------
# Results Section
# -----------------------------------------------------------------------------
if 'processed' in st.session_state and st.session_state['processed']:
    st.markdown("---")
    st.subheader("📊 Results")
    
    chunks = st.session_state['chunks']
    total_pages = st.session_state['total_pages']
    
    # Summary Statistics
    st.markdown("### Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", len(uploaded_files))
    with col2:
        st.metric("Total Pages", total_pages)
    with col3:
        st.metric("Total Chunks", len(chunks))
    with col4:
        avg_chunk_size = sum(len(chunk['text']) for chunk in chunks) / len(chunks) if chunks else 0
        st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Chunks Table", "📄 Page View", "🔍 Search", "💾 Export"])
    
    # Tab 1: Chunks Table
    with tab1:
        st.markdown("### All Chunks")
        
        if chunks:
            # Create DataFrame
            df_data = []
            for i, chunk in enumerate(chunks):
                df_data.append({
                    'Chunk #': i + 1,
                    'Page': chunk['meta']['page'] + 1,
                    'Filename': chunk['meta'].get('filename', 'Unknown'),
                    'Text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                    'Word Count': count_words(chunk['text']),
                    'Char Count': count_chars(chunk['text'])
                })
            
            df = pd.DataFrame(df_data)
            
            # Display with pagination
            page_size = 10
            total_pages_df = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
            page_num = st.number_input("Page", min_value=1, max_value=total_pages_df, value=1)
            
            start_idx = (page_num - 1) * page_size
            end_idx = min(start_idx + page_size, len(df))
            
            st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
            
            with st.expander("View Full Chunk Details"):
                selected_chunk_idx = st.selectbox("Select Chunk", range(len(chunks)), format_func=lambda x: f"Chunk {x+1} (Page {chunks[x]['meta']['page'] + 1})")
                if selected_chunk_idx is not None:
                    chunk = chunks[selected_chunk_idx]
                    st.markdown(f"**Chunk {selected_chunk_idx + 1}** - Page {chunk['meta']['page'] + 1}")
                    st.text_area("Full Text", chunk['text'], height=300)
    
    # Tab 2: Page View
    with tab2:
        st.markdown("### Page-by-Page View")
        
        # Group chunks by page
        page_data = {}
        for chunk in chunks:
            page = chunk['meta']['page'] + 1
            filename = chunk['meta'].get('filename', 'Unknown')
            key = f"Page {page} - {filename}"
            if key not in page_data:
                page_data[key] = []
            page_data[key].append(chunk)
        
        # Display each page
        for page_key, page_chunks in page_data.items():
            with st.expander(page_key, expanded=False):
                st.markdown(f"**{page_key}** - {len(page_chunks)} chunks")
                
                for i, chunk in enumerate(page_chunks):
                    chunk_num = chunk['meta']['chunk_index'] + 1
                    st.markdown(f"**Chunk {chunk_num}** ({count_words(chunk['text'])} words, {count_chars(chunk['text'])} chars)")
                    st.text(chunk['text'])
                    if i < len(page_chunks) - 1:
                        st.divider()
    
    # Tab 3: Search
    with tab3:
        st.markdown("### 🔍 Search Chunks")
        
        search_query = st.text_input("Search in all chunks", placeholder="Enter search term...")
        
        if search_query:
            search_results = []
            for i, chunk in enumerate(chunks):
                if search_query.lower() in chunk['text'].lower():
                    search_results.append({
                        'Chunk #': i + 1,
                        'Page': chunk['meta']['page'] + 1,
                        'Text': chunk['text'],
                        'Match': chunk['text'].lower().find(search_query.lower())
                    })
            
            if search_results:
                st.success(f"Found {len(search_results)} matches")
                
                for result in search_results[:10]:  # Limit to 10 results
                    with st.expander(f"Match in Chunk {result['Chunk #']} (Page {result['Page']})"):
                        # Highlight the match
                        text = result['Text']
                        highlighted = text.replace(
                            search_query, 
                            f"**{search_query.upper()}**",
                            1
                        )
                        st.markdown(highlighted)
                        
                        if len(search_results) > 10:
                            st.text(f"... and {len(search_results) - 10} more results")
            else:
                st.warning("No matches found")
    
    # Tab 4: Export
    with tab4:
        st.markdown("### 💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        # JSON Export
        with col1:
            json_data = create_json_export(chunks)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="pdf_chunks.json",
                mime="application/json",
                help="Download all chunks in JSON format"
            )
        
        # CSV Export
        with col2:
            df_export = pd.DataFrame([{
                'chunk_index': i,
                'page': chunk['meta']['page'] + 1,
                'filename': chunk['meta'].get('filename', ''),
                'text': chunk['text']
            } for i, chunk in enumerate(chunks)])
            
            csv_data = convert_df_to_csv(df_export)
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name="pdf_chunks.csv",
                mime="text/csv",
                help="Download all chunks in CSV format"
            )
        
        # TXT Export
        with col3:
            txt_data = create_text_export(chunks)
            st.download_button(
                label="📄 Download as TXT",
                data=txt_data,
                file_name="pdf_full_text.txt",
                mime="text/plain",
                help="Download full text in plain format"
            )
        
        # Show preview
        with st.expander("Preview Export Data"):
            st.json(chunks[0] if chunks else {}, expanded=False)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "*Q2C PDF Ingestion - Built with Streamlit*",
    help="Version 1.0.0"
)

