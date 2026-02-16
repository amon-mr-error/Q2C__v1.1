"""Small CLI demo for RAG using your ingestion layer and hf_rag helpers.

Usage (after installing optional deps in the venv):
    .venv/bin/python3 run_rag.py /path/to/file.pdf "What is the cancellation policy?"

This script is intentionally minimal and meant for local testing.
"""
import sys
from pathlib import Path

from ingest import ingest_pdf


def main():
    if len(sys.argv) < 3:
        print("Usage: run_rag.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    question = sys.argv[2]

    if not pdf_path.exists():
        print("PDF file not found:", pdf_path)
        sys.exit(1)

    # Ingest
    print("Ingesting PDF...")
    chunks = ingest_pdf(source=str(pdf_path), chunk_size=1000, overlap=200)
    print(f"Generated {len(chunks)} chunks")

    # Lazy import of hf_rag to avoid requiring optional deps unless used
    try:
        import hf_rag
    except ImportError as e:
        print("hf_rag or its dependencies not installed:", e)
        sys.exit(1)

    print("Building FAISS index (this may take a moment)...")
    idx, texts, metas, dim = hf_rag.build_faiss_index(chunks, embed_model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Index built. Dim=", dim)

    print("Retrieving top passages...")
    results = hf_rag.retrieve(idx, texts, metas, question, k=4)
    for text, meta, score in results:
        print(f"-- Page {meta.get('page', '?')+1} (score={score:.4f}) --")
        print(text[:500])
        print()

    prompt = "Use the retrieved passages to answer the question.\n\n" + "\n\n---\n\n".join([r[0] for r in results]) + f"\n\nQuestion: {question}\nAnswer concisely:"

    print("Generating answer...")
    answer = hf_rag.generate_answer("google/flan-t5-small", prompt, device=-1)
    print("\nAnswer:\n", answer)


if __name__ == '__main__':
    main()
