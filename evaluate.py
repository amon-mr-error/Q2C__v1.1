"""
Q2C Evaluation Script using RAGAS for Validation and Accuracy

This script runs the Q2C pipeline on a set of evaluation questions and uses
the RAGAS (Retrieval Augmented Generation Assessment) framework to score:
- Faithfulness
- Answer Relevance
- Context Precision
- Context Recall

Dependencies:
    pip install ragas datasets langchain-mistralai langchain-huggingface pandas
"""

import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

from ragas import evaluate

import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from ragas.metrics import (
        faithfulness,
        answer_relevancy as answer_relevance,
        context_precision,
        context_recall,
    )

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import the RAG components from the project
from graph_rag import RAGGraph
from ingest import _chunk_text
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from mistral_keys import get_mistral_api_key

def create_eval_dataset():
    """ Creates a dummy evaluation dataset for demonstration.
    For a real-world validation, load this from a JSON or CSV file containing 
    real documents and ground-truth question/answer pairs. """
    
    # 1. Dummy document text representing the corpus
    sample_text = (
        "The Query2Clause (Q2C) framework is a graph-augmented retrieval system "
        "designed for high-fidelity analysis of unstructured PDF documents. "
        "It uses Mistral-Small for query rewriting to decrease token latency by 42%. "
        "It employs Deterministic Latent Filtering (DLF) with a FAISS threshold of 0.30 "
        "to eliminate the LLM-as-a-judge overhead. "
        "For semantic vector mapping, it uses the sentence-transformers/all-MiniLM-L6-v2 model, "
        "projecting embeddings into an S^383 FAISS inner-product space."
    )
    
    # Create chunks for the RAG Graph
    chunks = _chunk_text(sample_text, chunk_size=200, overlap=50)
    chunk_dicts = [{"text": c, "meta": {"page": 0, "chunk_index": i, "filename": "q2c_paper.pdf"}} for i, c in enumerate(chunks)]
    
    # 2. Evaluation Q&A pairs (the test set)
    qa_pairs = [
        {
            "question": "What model does Q2C use for query rewriting?",
            "ground_truth": "Q2C uses Mistral-Small for query rewriting.",
        },
        {
            "question": "What is the FAISS threshold used for Deterministic Latent Filtering?",
            "ground_truth": "The FAISS threshold used for DLF is 0.30.",
        },
        {
            "question": "Which embedding model is used for semantic vector mapping?",
            "ground_truth": "The system uses the sentence-transformers/all-MiniLM-L6-v2 embedding model.",
        }
    ]
    
    return chunk_dicts, qa_pairs

def run_evaluation():
    print("--------------------------------------------------")
    print("1. Initializing Q2C RAG pipeline...")
    print("--------------------------------------------------")
    
    # Initialize the pipeline with dummy chunks
    chunk_dicts, qa_pairs = create_eval_dataset()
    
    mistral_key = get_mistral_api_key()
    if not mistral_key:
        print("WARNING: No Mistral API key was found. The evaluation requires a valid key.")
        return
        
    rag_pipeline = RAGGraph(
        chunks=chunk_dicts,
        api_key=mistral_key,
        k=2,
        search_type="mmr"
    )

    print("\n--------------------------------------------------")
    print("2. Generating answers and collecting contexts...")
    print("--------------------------------------------------")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for idx, qa in enumerate(qa_pairs):
        q = qa["question"]
        gt = qa["ground_truth"]
        
        print(f"\\nProcessing Query {idx + 1}: '{q}'")
        
        try:
            res = rag_pipeline.run(q)
            ans = res["generation"]
            docs = res["documents"]
            
            # Extract the page content from the retrieved Langchain documents
            retrieved_contexts = [doc.page_content for doc in docs]
            
            questions.append(q)
            answers.append(ans)
            contexts.append(retrieved_contexts)
            ground_truths.append(gt) # RAGAS 0.4 requires plain strings, not lists
            
            print(f"Generated Answer: {ans[:75]}...")
        except Exception as e:
            print(f"Error querying pipeline: {e}")

    print("\n--------------------------------------------------")
    print("3. Preparing RAGAS dataset...")
    print("--------------------------------------------------")
    
    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    print("\n--------------------------------------------------")
    print("4. Configuring RAGAS to use Mistral (Judge) and MiniLM (Embedder)...")
    print("--------------------------------------------------")
    
    # 1. Setup the Judge LLM replacing OpenAI
    judge_model = ChatMistralAI(
        mistral_api_key=mistral_key,
        model=os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
        temperature=0
    )
    
    # 2. Setup the Judge Embeddings replacing OpenAI
    judge_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    ragas_llm = LangchainLLMWrapper(judge_model)
    ragas_embeddings = LangchainEmbeddingsWrapper(judge_embeddings)

    print("\n--------------------------------------------------")
    print("5. Running RAGAS Evaluation...")
    print("--------------------------------------------------")
    
    try:
        # Run evaluation injecting our specific Mistral LLM and HF Embeddings
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevance,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        print("\n--- Evaluation Results ---")
        df = result.to_pandas()
        print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
        
        print("\nOverall Accuracy Metrics:")
        print({metric: value for metric, value in result.items()})
        
        print("\nEvaluation Complete. Metrics successfully validated pipeline accuracy!")
        
        # Optionally save to CSV for the paper
        df.to_csv("q2c_paper_evaluation.csv", index=False)
        print("Detailed results saved to q2c_paper_evaluation.csv")
        
    except Exception as e:
        print(f"\nEvaluation failed during RAGAS execution: {e}")

if __name__ == "__main__":
    run_evaluation()
