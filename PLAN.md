# Streamlit Frontend Plan for Q2C PDF Ingestion

## 📋 Project Analysis

### Current State:
- **Repository**: Q2C__v1.1/ with single file `ingest.py`
- **Core Functionality**: PDF text extraction and chunking
- **Main Functions**:
  - `extract_text_from_pdf()` - Extracts text from PDF pages
  - `ingest_pdf()` - High-level API that extracts and chunks text
  - `_chunk_text()` - Splits text into overlapping chunks
  - `_normalize_text()` - Basic text normalization

### Output Format:
Each chunk is a dict: `{"text": str, "meta": {"page": int, "chunk_index": int}}`

---

## 🎯 Streamlit Frontend Requirements

### Features to Implement:

1. **File Upload Section**
   - Drag & drop PDF file upload
   - Support multiple PDF uploads
   - Display uploaded files list

2. **Configuration Panel (Sidebar)**
   - Chunk size slider (default: 1000, range: 100-5000)
   - Overlap size slider (default: 200, range: 0-1000)
   - Password input for encrypted PDFs

3. **Processing Actions**
   - "Process PDF" button
   - Progress indicator during processing

4. **Results Display**
   - **Summary Statistics**:
     - Total pages processed
     - Total chunks generated
     - Average chunks per page
   - **Page-by-Page View**:
     - Expandable sections for each page
     - Show page text with word/character counts
   - **Chunks View**:
     - Dataframe/table display of all chunks
     - Search/filter functionality
     - Expandable chunk details

5. **Export Options**
   - Download chunks as JSON
   - Download chunks as CSV
   - Download full text as TXT

---

## 📁 File Structure

```
Q2C__v1.1/
├── ingest.py          # Existing PDF ingestion module
├── app.py             # Streamlit main application
├── requirements.txt   # Python dependencies
└── .streamlit/
    └── config.toml    # Streamlit configuration (optional)
```

---

## 🔧 Implementation Plan

### Step 1: Create `requirements.txt`
```txt
streamlit>=1.28.0
PyPDF2>=3.0.0
pandas>=2.0.0
```

### Step 2: Create `app.py`
- Import necessary libraries (streamlit, pandas, ingest module)
- Set up page configuration
- Implement sidebar with configuration options
- Create main content area with file upload
- Implement processing logic
- Display results with tabs for different views
- Add export functionality

### Step 3: Optional - Streamlit Config
- Set page title and icon
- Configure layout width
- Set theme options

---

## 🎨 UI/UX Considerations

- Clean, intuitive layout
- Clear section separation
- Expandable details for large content
- Responsive design
- Error handling with user-friendly messages
- Loading states during processing

---

## ✅ Success Criteria

1. ✅ Users can upload PDF files
2. ✅ Users can configure chunking parameters
3. ✅ PDF text is extracted and displayed
4. ✅ Chunks are generated and viewable
5. ✅ Results can be exported
6. ✅ Handles encrypted PDFs with password
7. ✅ Provides summary statistics
8. ✅ User-friendly error messages

---

## 🚀 Next Steps (Optional Enhancements)

- Preview chunk content with highlighting
- Compare different chunking configurations
- Save/load processing history
- Integration with embedding models
- Batch processing multiple files
- Real-time preview as parameters change

