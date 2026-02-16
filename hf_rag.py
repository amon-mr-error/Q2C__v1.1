"""Small, lazy Hugging Face RAG helper for local use.

This module provides a safe, optional dependency wrapper so the main app
can import it even if heavy ML libraries are missing. It exposes three
primary helpers:

- build_faiss_index(chunks, embed_model_name) -> (index, texts, metadatas, embed_dim)
- retrieve(index, texts, metadatas, query, embedder, k=4) -> list of (text, meta, score)
- generate_answer(model_name, prompt, device=-1, max_length=256) -> str

Notes:
- This uses sentence-transformers for embeddings and faiss-cpu for vector
  search. Both are optional; the module raises informative ImportError
  if used without installing them.
- The functions prefer lazily-instantiated models so import-time is cheap.
"""
from typing import List, Tuple, Dict, Any, Optional


def _ensure_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "sentence-transformers is required for embeddings. Install with: pip install sentence-transformers"
        ) from e
    return SentenceTransformer


def _ensure_faiss():
    try:
        import faiss
    except Exception as e:
        raise ImportError(
            "faiss-cpu is required for local vector search. Install with: pip install faiss-cpu"
        ) from e
    return faiss


def _ensure_transformers():
    try:
        from transformers import pipeline
    except Exception as e:
        raise ImportError(
            "transformers is required for generation. Install with: pip install transformers"
        ) from e
    return pipeline


def build_faiss_index(
    chunks: List[Dict[str, Any]],
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = True,
) -> Tuple[Any, List[str], List[Dict[str, Any]], int]:
    """Build a FAISS index from chunk dicts returned by `ingest_pdf()`.

    Returns: (index, texts, metadatas, embed_dim)
    """
    SentenceTransformer = _ensure_embeddings()
    faiss = _ensure_faiss()

    texts = [c["text"] for c in chunks]
    metadatas = [c.get("meta", {}) for c in chunks]

    if len(texts) == 0:
        raise ValueError("No chunks provided to build index")

    embedder = SentenceTransformer(embed_model_name)
    embs = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    if normalize:
        faiss.normalize_L2(embs)

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embs)

    return index, texts, metadatas, dim


def retrieve(
    index: Any,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    query: str,
    embedder: Optional[Any] = None,
    embed_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    k: int = 4,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Retrieve top-k passages for a query. Returns list of (text, meta, score).

    You can pass a pre-instantiated embedder (SentenceTransformer) to avoid
    reconstructing it.
    """
    SentenceTransformer = _ensure_embeddings()
    faiss = _ensure_faiss()

    if embedder is None:
        embedder = SentenceTransformer(embed_model_name)

    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)
    results: List[Tuple[str, Dict[str, Any], float]] = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(texts):
            continue
        results.append((texts[idx], metadatas[idx], float(score)))
    return results


def generate_answer(
    model_name: str,
    prompt: str,
    device: int = -1,
    max_length: int = 256,
    do_sample: bool = False,
) -> str:
    """Generate answer text from a text2text model using transformers.pipeline.

    device: -1 for CPU, 0+ for GPU index
    """
    pipeline_fn = _ensure_transformers()

    # Try common generation tasks in order. Some transformer releases expose
    # 'text2text-generation' for seq2seq models (T5/flan), but other installs
    # may only provide 'text-generation'. Try both and return the first result.
    last_err = None
    for task in ("text2text-generation", "text-generation"):
        try:
            # Prepare generation args
            # Using max_new_tokens is preferred over max_length for text generation to avoid confusion
            gen_kwargs = {
                "max_new_tokens": max_length,
                "do_sample": do_sample,
                "truncation": True
            }
            if task == "text-generation":
                # For causal LM pipelines (or fallback), we must ensure we don't return the input prompt
                gen_kwargs["return_full_text"] = False
            
            gen = pipeline_fn(task, model=model_name, device=device)
            out = gen(prompt, **gen_kwargs)
            
            if isinstance(out, list) and len(out) > 0:
                print(f"DEBUG: Pipeline task '{task}' succeeded.")
                # Different pipelines return different keys
                text = out[0].get("generated_text") or out[0].get("text") or out[0].get("summary_text") or str(out[0])
                return text.strip()
            # If pipeline returned unexpected structure, stringify it
            return str(out)
        except Exception as e:
            print(f"DEBUG: Pipeline task '{task}' failed with error: {e}")
            last_err = e
            # try the next available task
            continue

    # If we reach here, no generation pipeline worked
    raise RuntimeError(
        "No suitable generation pipeline available (tried text2text-generation and text-generation). "
        f"Last error: {last_err}"
    )


# Small smoke test when executed directly
if __name__ == "__main__":
    print("hf_rag.py smoke test: building index with dummy chunk data (this will fail if optional deps not installed)")
    dummy = [
        {"text": "Policy A says the insured must notify within 30 days.", "meta": {"page": 0, "chunk_index": 0}},
        {"text": "Contract B requires arbitration for disputes.", "meta": {"page": 1, "chunk_index": 0}},
    ]
    try:
        idx, texts, metas, dim = build_faiss_index(dummy)
        print("Index built. dim=", dim)
        # retrieve
        embs = _ensure_embeddings()("sentence-transformers/all-mpnet-base-v2")
        res = retrieve(idx, texts, metas, "When must the insured notify?", embedder=embs)
        print("Retrieve result:", res)
    except Exception as e:
        print("Optional dependency missing or error during smoke test:", e)
