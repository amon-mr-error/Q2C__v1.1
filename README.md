# Query2Clause (Q2C)

> **Intelligent document analysis powered by Retrieval-Augmented Generation.**

🔗 **Live Demo:** [query2clausev1.streamlit.app](https://query2clausev1.streamlit.app/)

---

## What is Q2C?

Q2C is an AI-powered document Q&A system. Upload any PDF — a legal contract, research paper, or technical report — and ask questions about it in plain English. The system retrieves the most relevant passages and generates a grounded, source-cited answer using a Mistral LLM backend.

---

## How it Works

```
PDF Upload → Text Extraction & Chunking → Vector Knowledge Graph → LangGraph RAG Pipeline → Streamed Answer + Sources
```

1. **Ingest** — PDFs are parsed and split into overlapping text chunks.
2. **Index** — Chunks are embedded and stored in a FAISS vector store.
3. **Retrieve** — On each query, MMR (Maximal Marginal Relevance) retrieval fetches the most relevant, diverse passages.
4. **Reason** — A 4-node LangGraph pipeline rewrites the query, grades relevance, and streams the final answer.
5. **Cite** — Every answer links back to the exact source passages and page numbers.

---

## Key Features

| Feature | Detail |
|---|---|
| 📄 Multi-PDF Upload | Process multiple documents simultaneously |
| 🔍 Adaptive Query Rewriting | Automatically refines vague queries for better retrieval |
| 📚 Source Attribution | Every answer cites the exact passages used |
| ⚙️ Tunable Retrieval | Adjust chunk size, overlap, top-K, and search mode (MMR / Similarity) |
| 🔐 Encrypted PDF Support | Password-protected PDFs handled natively |
| 📥 Export | Download the full knowledge base as JSON |

---

## Tech Stack

- **Frontend** — Streamlit + Custom CSS
- **RAG Pipeline** — LangGraph (4-node adaptive graph)
- **LLM** — Mistral (`mistral-large-2411`) via Mistral AI API
- **Embeddings** — `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector Store** — FAISS (in-memory)

---

## Local Setup

```bash
git clone <repo-url>
cd pbl

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY_1=your_primary_key_here
MISTRAL_API_KEY_2=your_secondary_key_here
MISTRAL_MODEL=mistral-large-2411
```

The project also accepts a single key using `MISTRAL_API_KEY` if you only have one key.

Run the app:

```bash
streamlit run app.py
```

---

## Project Structure

```
pbl/
├── app.py              # Streamlit UI
├── graph_rag.py        # LangGraph RAG pipeline
├── ingest.py           # PDF parsing & chunking
├── evaluate.py         # RAGAS evaluation script
├── requirements.txt
└── .env                # API keys (not committed)
```
