"""
test.py — Q2C RAG Evaluation Suite
====================================
Evaluates the LangGraph RAG pipeline on the dataset defined in dataset.py.

Metrics
-------
Retrieval : Precision@k, Recall@k, MRR
Generation: Exact Match (EM), Token-level F1
Quality   : Faithfulness proxy (answer-to-retrieved overlap)
Latency   : Per-query response time + average

Outputs
-------
* Summary metric tables (console)
* 5 sample query/answer/ground_truth cards (console)
* RAG vs Dummy Baseline comparison
"""

from __future__ import annotations

import os
import sys
import time
import random
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

# Seconds to sleep between RAG queries to avoid Mistral rate limits
QUERY_SLEEP_S: float = 4.0

# Retry config for 429 / service errors
MAX_RETRIES: int  = 5
RETRY_BASE_S: float = 10.0  # first retry waits 10 s, doubles each time


def _call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs) with exponential backoff on 429 / capacity errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "capacity" in msg.lower() or "rate" in msg.lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = RETRY_BASE_S * (2 ** attempt)
                print(f"\n[RETRY] Rate limit hit — waiting {wait:.0f}s before retry "
                      f"{attempt+1}/{MAX_RETRIES-1}…")
                time.sleep(wait)
            else:
                raise

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("MISTRAL_API_KEY")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
from dataset import DATASET          # List[Dict]: query, ground_truth, relevant_docs

# ---------------------------------------------------------------------------
# PDF directory
# ---------------------------------------------------------------------------
PDF_DIR = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------------
from ingest import ingest_pdf


def load_all_pdfs(pdf_dir: str) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Load every PDF in *pdf_dir*, tag each chunk with its source filename,
    and return (all_chunks, {filename: path}) for reference.
    """
    all_chunks: List[Dict] = []
    index: Dict[str, str] = {}

    for fname in sorted(os.listdir(pdf_dir)):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(pdf_dir, fname)
        print(f"[INGEST] Loading {fname} …")
        chunks = ingest_pdf(fpath)
        for c in chunks:
            c["meta"]["filename"] = fname
        all_chunks.extend(chunks)
        index[fname] = fpath
        print(f"         → {len(chunks)} chunks")

    print(f"[INGEST] Total chunks: {len(all_chunks)}\n")
    return all_chunks, index


# ---------------------------------------------------------------------------
# RAG pipeline initialisation
# ---------------------------------------------------------------------------
from graph_rag import RAGGraph


def build_rag(chunks: List[Dict]) -> RAGGraph:
    print("[RAG] Building knowledge graph …")
    rag = RAGGraph(chunks, api_key=API_KEY, k=6, search_type="mmr")
    print("[RAG] Ready.\n")
    return rag


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = Counter(_tokenise(prediction))
    gt_tokens   = Counter(_tokenise(ground_truth))
    common = sum((pred_tokens & gt_tokens).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_tokens.values())
    recall    = common / sum(gt_tokens.values())
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_tokenise(prediction) == _tokenise(ground_truth))


def precision_at_k(retrieved_names: List[str], relevant_names: List[str], k: int) -> float:
    top_k = retrieved_names[:k]
    hits  = sum(1 for n in top_k if n in relevant_names)
    return hits / k if k else 0.0


def recall_at_k(retrieved_names: List[str], relevant_names: List[str], k: int) -> float:
    top_k = retrieved_names[:k]
    hits  = sum(1 for n in top_k if n in relevant_names)
    return hits / len(relevant_names) if relevant_names else 0.0


def reciprocal_rank(retrieved_names: List[str], relevant_names: List[str]) -> float:
    for i, name in enumerate(retrieved_names, start=1):
        if name in relevant_names:
            return 1.0 / i
    return 0.0


def faithfulness_proxy(answer: str, retrieved_docs) -> float:
    """
    Overlap of answer tokens with the union of retrieved passage tokens.
    Range [0, 1].  A higher value means the answer is grounded in retrieved content.
    """
    if not retrieved_docs:
        return 0.0
    corpus_tokens = Counter()
    for doc in retrieved_docs:
        corpus_tokens.update(_tokenise(doc.page_content))
    answer_tokens = Counter(_tokenise(answer))
    common = sum((answer_tokens & corpus_tokens).values())
    denom  = sum(answer_tokens.values())
    return common / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Dummy Baseline
# ---------------------------------------------------------------------------

def dummy_baseline(query: str) -> Dict[str, Any]:
    """Returns an empty string — worst-case retrieval/generation baseline."""
    return {
        "generation": "",
        "documents":  [],
        "rewritten":  False,
        "question":   query,
    }


# ---------------------------------------------------------------------------
# Single-query evaluation
# ---------------------------------------------------------------------------
K = 5  # Precision / Recall @K


def evaluate_one(
    system_fn,
    item: Dict,
    k: int = K,
) -> Dict[str, Any]:
    query         = item["query"]
    ground_truth  = item["ground_truth"]
    relevant_docs = item["relevant_docs"]          # List[str] filenames

    t0      = time.perf_counter()
    result  = _call_with_retry(system_fn, query)
    latency = time.perf_counter() - t0

    answer     = result.get("generation", "")
    ret_docs   = result.get("documents", [])

    # Filenames from retrieved document metadata
    retrieved_names = [
        d.metadata.get("filename", "")
        for d in ret_docs
    ]

    return {
        "query":          query,
        "answer":         answer,
        "ground_truth":   ground_truth,
        "retrieved_names": retrieved_names,
        "retrieved_docs":  ret_docs,
        "relevant_docs":  relevant_docs,
        # Retrieval
        "precision_k":   precision_at_k(retrieved_names, relevant_docs, k),
        "recall_k":      recall_at_k(retrieved_names, relevant_docs, k),
        "mrr":           reciprocal_rank(retrieved_names, relevant_docs),
        # Generation
        "em":            exact_match(answer, ground_truth),
        "f1":            token_f1(answer, ground_truth),
        # Faithfulness
        "faithfulness":  faithfulness_proxy(answer, ret_docs),
        # Latency
        "latency":       latency,
    }


# ---------------------------------------------------------------------------
# Full evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(system_fn, label: str, dataset: List[Dict]) -> List[Dict]:
    print(f"\n{'='*60}")
    print(f" Evaluating: {label}")
    print(f"{'='*60}")
    results = []
    for i, item in enumerate(dataset, 1):
        print(f"  [{i:02d}/{len(dataset)}] {item['query'][:70]} …")
        rec = evaluate_one(system_fn, item)
        results.append(rec)
        if i < len(dataset):      # no need to sleep after the last query
            time.sleep(QUERY_SLEEP_S)
        print(f"         P@{K}={rec['precision_k']:.2f}  R@{K}={rec['recall_k']:.2f}"
              f"  MRR={rec['mrr']:.2f}  F1={rec['f1']:.2f}"
              f"  Latency={rec['latency']:.2f}s")
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _avg(results: List[Dict], key: str) -> float:
    vals = [r[key] for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def print_summary(rag_results: List[Dict], base_results: List[Dict]):
    K_str = f"@{K}"

    # ── Table 1: Retrieval ──────────────────────────────────────────────
    print("\n\n" + "─"*62)
    print(" RETRIEVAL PERFORMANCE")
    print("─"*62)
    print(f"{'Model':<20} {'Precision'+K_str:<16} {'Recall'+K_str:<16} {'MRR':<10}")
    print("─"*62)
    for label, results in [("Q2C RAG", rag_results), ("Dummy Baseline", base_results)]:
        p = _avg(results, "precision_k")
        r = _avg(results, "recall_k")
        m = _avg(results, "mrr")
        print(f"{label:<20} {p:<16.4f} {r:<16.4f} {m:<10.4f}")
    print("─"*62)

    # ── Table 2: Answer Quality ─────────────────────────────────────────
    print("\n" + "─"*62)
    print(" ANSWER QUALITY")
    print("─"*62)
    print(f"{'Model':<20} {'Exact Match':<16} {'Token F1':<16} {'Faithfulness':<14}")
    print("─"*62)
    for label, results in [("Q2C RAG", rag_results), ("Dummy Baseline", base_results)]:
        em  = _avg(results, "em")
        f1  = _avg(results, "f1")
        fth = _avg(results, "faithfulness")
        print(f"{label:<20} {em:<16.4f} {f1:<16.4f} {fth:<14.4f}")
    print("─"*62)

    # ── Table 3: Latency ────────────────────────────────────────────────
    print("\n" + "─"*62)
    print(" LATENCY")
    print("─"*62)
    print(f"{'Model':<20} {'Avg Time (s)':<16} {'Min (s)':<12} {'Max (s)':<12}")
    print("─"*62)
    for label, results in [("Q2C RAG", rag_results), ("Dummy Baseline", base_results)]:
        lats = [r["latency"] for r in results]
        print(f"{label:<20} {sum(lats)/len(lats):<16.3f} {min(lats):<12.3f} {max(lats):<12.3f}")
    print("─"*62)


def print_samples(results: List[Dict], n: int = 5):
    print("\n\n" + "═"*70)
    print(f" SAMPLE OUTPUTS  (first {n} queries — Q2C RAG)")
    print("═"*70)
    for i, r in enumerate(results[:n], 1):
        print(f"\n[{i}] QUERY        : {r['query']}")
        print(f"    ANSWER       : {r['answer'][:300].strip()}{'…' if len(r['answer'])>300 else ''}")
        print(f"    GROUND TRUTH : {r['ground_truth'][:300].strip()}{'…' if len(r['ground_truth'])>300 else ''}")
        print(f"    RETRIEVED    : {r['retrieved_names']}")
        print(f"    F1={r['f1']:.3f}  Faithfulness={r['faithfulness']:.3f}  Latency={r['latency']:.2f}s")
        print("    " + "·"*64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load documents
    all_chunks, _ = load_all_pdfs(PDF_DIR)

    # 2. Build RAG
    rag = build_rag(all_chunks)

    # 3. Run evaluations
    rag_results  = run_evaluation(rag.run,       "Q2C RAG",       DATASET)
    base_results = run_evaluation(dummy_baseline, "Dummy Baseline", DATASET)

    # 4. Print results
    print_summary(rag_results, base_results)
    print_samples(rag_results, n=5)

    print("\n\n✅ Evaluation complete.\n")


if __name__ == "__main__":
    main()
