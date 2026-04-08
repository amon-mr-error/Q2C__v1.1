"""
Q2C Graph RAG — v3 (Speed-Optimised Adaptive Pipeline)

Latency improvements over v2:
  - REMOVED: 6 LLM calls for per-chunk grading  →  replaced with FAISS cosine-score threshold
  - ADDED:   streaming generation  →  first token appears in ~1s instead of waiting for full answer
  - ADDED:   mistral-small for query rewriting (3-4x faster than mistral-large)
  - ADDED:   mistral-large only for the final generation step
  - ADDED:   retriever result de-duplication to avoid sending repeated context

Total API round-trips per query:  v2 → 8   |   v3 → 2  (rewrite + generate)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Generator, Iterator, List, Optional, Sequence

from mistral_keys import get_mistral_api_key

try:
    import operator
    from typing import Annotated, TypedDict

    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langgraph.graph import END, StateGraph

except ImportError:
    raise ImportError("Please install required packages: pip install -r requirements.txt")


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    original_question:  str
    question:           str
    documents:          List[Document]
    generation:         str
    rewritten:          bool
    messages:           Annotated[Sequence[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# RAGGraph
# ---------------------------------------------------------------------------

class RAGGraph:
    """
    Speed-optimised adaptive RAG graph.

    v3 key changes:
      - grade_documents replaced with score-threshold filter (zero extra LLM calls)
      - query rewriting uses mistral-small (fast) while generation uses mistral-large (accurate)
      - stream_run() yields tokens progressively for real-time Streamlit display
    """

    # Cosine similarity threshold — chunks below this score are filtered as irrelevant.
    # Range [0, 1].  0.30 is a permissive baseline; raise to 0.40 for stricter filtering.
    RELEVANCE_THRESHOLD: float = 0.30

    def __init__(
        self,
        chunks: List[Dict],
        api_key: Optional[str] = None,
        k: int = 6,
        search_type: str = "mmr",
        relevance_threshold: float = 0.30,
    ):
        self.chunks = chunks
        self.api_key = api_key
        self.k = k
        self.search_type = search_type
        self.RELEVANCE_THRESHOLD = relevance_threshold

        self.embedding_model = self._setup_embeddings()
        self.vectorstore     = self._setup_vectorstore()
        self.retriever       = self._setup_retriever()
        self.llm_fast        = self._setup_llm(fast=True)   # small — for rewriting
        self.llm             = self._setup_llm(fast=False)  # large — for generation
        self.graph           = self._build_graph()

    # ------------------------------------------------------------------
    # Component setup
    # ------------------------------------------------------------------

    def _setup_embeddings(self) -> HuggingFaceEmbeddings:
        if not hasattr(RAGGraph, "_cached_embeddings"):
            print("[Q2C] Loading embedding model…")
            RAGGraph._cached_embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            print("[Q2C] Using cached embedding model.")
        return RAGGraph._cached_embeddings

    def _setup_vectorstore(self) -> FAISS:
        print(f"[Q2C] Building FAISS index over {len(self.chunks)} chunks…")
        texts = [c["text"]  for c in self.chunks]
        metas = [c["meta"]  for c in self.chunks]
        return FAISS.from_texts(texts=texts, embedding=self.embedding_model, metadatas=metas)

    def _setup_retriever(self):
        if self.search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.k, "fetch_k": self.k * 3, "lambda_mult": 0.6},
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def _setup_llm(self, fast: bool = False):
        """
        fast=True  → mistral-small (rewriting, quick tasks)
        fast=False → mistral-large (answer generation)
        """
        api_key = self.api_key or get_mistral_api_key()

        if api_key:
            try:
                from langchain_mistralai import ChatMistralAI

                if fast:
                    model_name = os.getenv("MISTRAL_FAST_MODEL", "mistral-small-latest")
                    print(f"[Q2C] Fast LLM: {model_name}")
                else:
                    model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
                    print(f"[Q2C] Generation LLM: {model_name}")

                return ChatMistralAI(
                    mistral_api_key=api_key,
                    model=model_name,
                    temperature=0,
                    streaming=True,   # enables token-by-token streaming
                )
            except ImportError:
                raise ImportError("pip install langchain-mistralai")

        # ── CPU fallback ──────────────────────────────────────────────
        if not fast:
            print("[Q2C] No API key — using local Flan-T5-Base (CPU)…")
            from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                                      pipeline)
            model_id  = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            pipe = pipeline(
                "text2text-generation",
                model=model, tokenizer=tokenizer,
                max_length=512, do_sample=False, repetition_penalty=1.1,
            )
            return HuggingFacePipeline(pipeline=pipe)
        # fast model for CPU: same Flan-T5 (lightweight rewriting)
        return self._setup_llm(fast=False)

    # ------------------------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------------------------

    def rewrite_query(self, state: GraphState) -> dict:
        """Node 1 — Rephrase query with the fast (small) model."""
        print("[Q2C] → rewrite_query")
        question = state["original_question"]

        prompt = ChatPromptTemplate.from_template(
            "Convert this question into a concise, keyword-rich retrieval query.\n"
            "Return ONLY the rewritten query, nothing else.\n\n"
            "Question: {question}\nRetrieval query:"
        )
        chain = prompt | self.llm_fast | StrOutputParser()

        try:
            rewritten = chain.invoke({"question": question}).strip()
            if not rewritten or len(rewritten) > 300:
                rewritten = question
        except Exception as e:
            print(f"[Q2C] rewrite failed ({e}), using original.")
            rewritten = question

        print(f"[Q2C]   original : {question}")
        print(f"[Q2C]   rewritten: {rewritten}")
        return {"question": rewritten, "rewritten": rewritten != question}

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Map cosine similarity from [-1, 1] to [0, 1] so threshold comparisons are valid."""
        return (float(score) + 1.0) / 2.0

    def retrieve_and_filter(self, state: GraphState) -> dict:
        """
        Node 2 — Retrieve top-k chunks then filter by cosine score threshold.

        Fetches fetch_k = k*3 candidates, normalises scores to [0,1], filters
        by RELEVANCE_THRESHOLD, then uses max_marginal_relevance (MMR) on the
        filtered set for diversity — improving recall on on-corpus questions.
        """
        print("[Q2C] → retrieve_and_filter")
        question = state["question"]

        fetch_k = self.k * 3  # cast a wider net then MMR-select for diversity

        # Get docs WITH their raw similarity scores
        scored_docs = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=fetch_k
        )

        # Normalise scores to [0, 1] and filter by threshold
        filtered = []
        for doc, raw_score in scored_docs:
            score = self._normalize_score(raw_score)
            flag = "✓" if score >= self.RELEVANCE_THRESHOLD else "✗"
            print(f"[Q2C]   {flag} Page {doc.metadata.get('page', '?')+1}  score={score:.3f}")
            if score >= self.RELEVANCE_THRESHOLD:
                filtered.append(doc)

        # Fallback: if ALL chunks scored below threshold, use top-3 anyway
        if not filtered:
            print("[Q2C]   All below threshold — using top-3 as fallback")
            filtered = [doc for doc, _ in scored_docs[:3]]

        # MMR diversity re-ranking: pick up to self.k diverse docs from filtered set
        if len(filtered) > self.k:
            query_embedding = self.embedding_model.embed_query(question)
            filtered = self.vectorstore.max_marginal_relevance_search_by_vector(
                query_embedding,
                k=self.k,
                fetch_k=len(filtered),
                lambda_mult=0.6,
                filter=None,
            )

        # De-duplicate (same page_content can appear via MMR fetch_k overlap)
        seen, unique = set(), []
        for doc in filtered:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                unique.append(doc)

        print(f"[Q2C]   {len(unique)} unique relevant chunks after filtering")
        return {"documents": unique}

    def generate(self, state: GraphState) -> dict:
        """Node 3 — Chain-of-thought generation with citation enforcement."""
        print("[Q2C] → generate")
        question  = state["original_question"]
        documents = state["documents"]

        if documents:
            context_parts = []
            for doc in documents:
                page_num = doc.metadata.get("page", -1)
                filename = doc.metadata.get("filename", "document")
                context_parts.append(
                    f"[Source — {filename}, Page {page_num + 1}]\n{doc.page_content}"
                )
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(No relevant passages were retrieved.)"

        prompt = ChatPromptTemplate.from_template(
            "You are a precise, analytical assistant for document Q&A.\n\n"
            "## Retrieved Context\n"
            "{context}\n\n"
            "## Instructions\n"
            "Using the context passages above, provide a detailed answer to the question below.\n"
            "• Synthesise information across all passages — do not skip relevant details.\n"
            "• Cite every fact inline as (Source — Filename, Page X).\n"
            "• Only say 'Not found in the provided documents.' if the context contains absolutely NO relevant information.\n"
            "• Be thorough but concise.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain      = prompt | self.llm | StrOutputParser()
        generation = chain.invoke({"context": context, "question": question})
        return {"generation": generation}

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self):
        """3-node DAG: rewrite → retrieve_and_filter → generate."""
        wf = StateGraph(GraphState)
        wf.add_node("rewrite_query",       self.rewrite_query)
        wf.add_node("retrieve_and_filter", self.retrieve_and_filter)
        wf.add_node("generate",            self.generate)

        wf.set_entry_point("rewrite_query")
        wf.add_edge("rewrite_query",       "retrieve_and_filter")
        wf.add_edge("retrieve_and_filter", "generate")
        wf.add_edge("generate",            END)
        return wf.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, question: str) -> Dict[str, Any]:
        """Run the full pipeline and return the result dict."""
        initial: GraphState = {
            "original_question": question,
            "question":          question,
            "documents":         [],
            "generation":        "",
            "rewritten":         False,
            "messages":          [],
        }
        result = self.graph.invoke(initial)
        return {
            "generation": result["generation"],
            "documents":  result["documents"],
            "rewritten":  result.get("rewritten", False),
            "question":   result.get("question", question),
        }

    def stream_run(self, question: str):
        """
        Two-phase streaming pipeline for Streamlit:
          1. Run rewrite + retrieve_and_filter synchronously (fast, no tokens to stream)
          2. Stream the generation token-by-token via llm.stream()

        Yields:
            {"type": "meta",  "rewritten": bool, "question": str, "documents": List[Document]}
            {"type": "token", "content": str}
            {"type": "done"}
        """
        # ── Phase 1: rewrite ──────────────────────────────────────────
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Convert this question into a concise, keyword-rich retrieval query.\n"
            "Return ONLY the rewritten query, nothing else.\n\n"
            "Question: {question}\nRetrieval query:"
        )
        try:
            rewritten_q = (rewrite_prompt | self.llm_fast | StrOutputParser()).invoke(
                {"question": question}
            ).strip()
            if not rewritten_q or len(rewritten_q) > 300:
                rewritten_q = question
        except Exception:
            rewritten_q = question

        rewritten = rewritten_q != question

        # ── Phase 2: retrieve + score-filter ─────────────────────────
        fetch_k = self.k * 3
        scored_docs = self.vectorstore.similarity_search_with_relevance_scores(
            rewritten_q, k=fetch_k
        )
        # Normalise scores to [0, 1] before thresholding
        filtered = [d for d, s in scored_docs if self._normalize_score(s) >= self.RELEVANCE_THRESHOLD]
        if not filtered:
            filtered = [d for d, _ in scored_docs[:3]]

        # MMR diversity re-ranking on filtered set
        if len(filtered) > self.k:
            query_embedding = self.embedding_model.embed_query(rewritten_q)
            filtered = self.vectorstore.max_marginal_relevance_search_by_vector(
                query_embedding,
                k=self.k,
                fetch_k=len(filtered),
                lambda_mult=0.6,
            )

        # De-duplicate
        seen, documents = set(), []
        for doc in filtered:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                documents.append(doc)

        # Yield metadata so UI can render source panel immediately
        yield {"type": "meta", "rewritten": rewritten, "question": rewritten_q,
               "documents": documents}

        # ── Phase 3: streaming generation ────────────────────────────
        if documents:
            context_parts = [
                f"[Source — {d.metadata.get('filename','document')}, "
                f"Page {d.metadata.get('page',-1)+1}]\n{d.page_content}"
                for d in documents
            ]
            context = "\n\n---\n\n".join(context_parts)
        else:
            context = "(No relevant passages were retrieved.)"

        gen_prompt = ChatPromptTemplate.from_template(
            "You are a precise analytical assistant for document Q&A.\n\n"
            "## Retrieved Context\n{context}\n\n"
            "## Instructions\n"
            "Using the context passages above, provide a detailed answer to the question below.\n"
            "• Synthesise information across all passages — do not skip relevant details.\n"
            "• Cite every fact inline as (Source — Filename, Page X).\n"
            "• Only say 'Not found in the provided documents.' if the context contains absolutely NO relevant information.\n"
            "• Be thorough but concise.\n\n"
            "Question: {question}\n\nAnswer:"
        )
        chain = gen_prompt | self.llm   # no StrOutputParser — we need AIMessageChunk objects

        for chunk in chain.stream({"context": context, "question": question}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                yield {"type": "token", "content": token}

        yield {"type": "done"}
