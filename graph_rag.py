import os
from typing import List, Dict, Any, Optional

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, Annotated, Sequence
    import operator
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
except ImportError:
    raise ImportError("Please install required packages: pip install -r requirements.txt")

# Define the state for the graph
class GraphState(TypedDict):
    """
    State of the RAG graph.
    """
    question: str
    documents: List[str]
    generation: str
    messages: Annotated[Sequence[BaseMessage], operator.add]

class RAGGraph:
    def __init__(self, chunks: List[Dict], api_key: Optional[str] = None):
        """
        Initialize the RAG Graph.
        
        Args:
            chunks: List of chunk dictionaries from ingest.py
            api_key: Optional Mistral API key. If None, uses local HF pipeline (might be slow).
        """
        self.chunks = chunks
        self.api_key = api_key
        
        # Initialize components
        self.embedding_model = self._setup_embeddings()
        self.vectorstore = self._setup_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.llm = self._setup_llm()
        self.graph = self._build_graph()
        
    def _setup_embeddings(self):
        """Initialize local HuggingFace embeddings."""
        print("Initializing embeddings...")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    def _setup_vectorstore(self):
        """Build FAISS vectorstore from chunks."""
        print("Building vectorstore...")
        texts = [chunk["text"] for chunk in self.chunks]
        metadatas = [chunk["meta"] for chunk in self.chunks]
        
        # Create vectorstore
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )
        return vectorstore

    def _setup_llm(self):
        """Initialize LLM (Mistral API or Local HuggingFace)."""
        # Check for API key in args or environment
        api_key = self.api_key or os.getenv("MISTRAL_API_KEY")
        
        if api_key:
            print("Using Mistral API...")
            try:
                from langchain_mistralai import ChatMistralAI
                model_name = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
                return ChatMistralAI(
                    mistral_api_key=api_key,
                    model=model_name,
                    temperature=0
                )
            except ImportError:
                raise ImportError("Please install langchain-mistralai: pip install langchain-mistralai")
        else:
            print("Using Local HuggingFace Pipeline (Flan-T5)...")
            # Fallback to a smaller model if no API key, or try to load a big one
            # For demonstration, we'll use a smaller flan-t5 logic or similar if resources are constrained,
            # but user asked for Mistral.
            # Warning: Loading Mistral-7B on CPU without quantization is heavy.
            # We will use a quantized model if possible, or fallback to something lighter if it fails.
            
            try:
                # Attempt to use a quantized model or a smaller instruction tuned model
                # Ideally, integration logic should handle user warnings.
                # Here we default to a standard HF pipeline.
                
                # Use Flan-T5-Base for better instruction following on CPU than GPT-2
                # It is still small enough to run locally without major issues.
                model_id = "google/flan-t5-base"
                
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                
                pipe = pipeline(
                    "text2text-generation", 
                    model=model, 
                    tokenizer=tokenizer,
                    max_length=512,
                    temperature=0, # Deterministic
                    repetition_penalty=1.1
                )
                return HuggingFacePipeline(pipeline=pipe)
            except Exception as e:
                print(f"Failed to load local model: {e}")
                raise

    def retrieve(self, state: GraphState):
        """
        Retrieve documents based on the question.
        """
        print("---RETRIEVE---")
        question = state["question"]
        documents = self.retriever.get_relevant_documents(question)
        return {"documents": documents, "question": question}

    def generate(self, state: GraphState):
        """
        Generate answer using RAG.
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Create context string with source pages
        context_parts = []
        for i, doc in enumerate(documents):
            page_num = doc.metadata.get("page", "Unknown")
            source = f"Page {page_num + 1}"
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")
            
        context = "\n\n".join(context_parts)
        
        # Prompt template
        template = """Answer the question based only on the following context. 
        Start your answer by directly addressing the question.
        ALWAYS cite the 'Source: Page X' for every fact you state from the context.
        If you cannot find the answer in the context, say so.

        Context:
        {context}

        Question: {question}
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        generation = chain.invoke({"context": context, "question": question})
        return {"generation": generation}

    def _build_graph(self):
        """
        Build the LangGraph state machine.
        """
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        # Add edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, question: str):
        """
        Run the graph for a question.
        Returns a dictionary with "generation" and "documents".
        """
        inputs = {"question": question}
        result = self.graph.invoke(inputs)
        return {
            "generation": result["generation"],
            "documents": result["documents"]
        }
