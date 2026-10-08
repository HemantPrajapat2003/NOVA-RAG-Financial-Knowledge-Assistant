"""
Financial RAG Pipeline Module.
Implements the core Retrieval-Augmented Generation logic:
1. Semantic search with Cosine Similarity over ChromaDB (Top K = 5)
2. Prompt construction with strict anti-hallucination system prompt
3. Gemini 2.5 Flash invocation via langchain_google_genai
4. Source citations parsing with document name, page number, and document type
5. Graceful empty-retrieval handling and offline/fallback support
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from utils.embeddings import get_embedding_function

# Load environment variables from .env
load_dotenv()

SYSTEM_PROMPT = """You are an AI Financial Knowledge Assistant. Answer questions only using retrieved document context. If the answer is not present in the documents, respond: "I could not find this information in the uploaded financial documents."

Rules:
• Do not hallucinate
• Do not generate assumptions
• Always provide:
  - answer
  - source document
  - page number
• Keep responses professional and concise"""

COLLECTION_NAME = "financial_knowledge_base"


class FinancialRAGPipeline:
    """
    Production-ready Financial RAG Assistant Pipeline using ChromaDB and Gemini 2.5 Flash.
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = COLLECTION_NAME,
        top_k: int = 5,
        gemini_model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.gemini_model = gemini_model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        
        # Initialize embedding model (384-dimensional all-MiniLM-L6-v2)
        self.embedding_fn = get_embedding_function()
        
        # Connect to ChromaDB
        self._init_chroma()
        
        # Initialize Gemini LLM if key is present
        self.llm = None
        self._init_llm()

    def _init_chroma(self):
        """Initializes or connects to the local ChromaDB persistent collection."""
        os.makedirs(self.persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
        except Exception:
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

    def _init_llm(self):
        """Initializes the Gemini 2.5 Flash chat model if API key is available."""
        key = self.api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        if key and key != "your_gemini_api_key_here":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=self.gemini_model,
                    google_api_key=key,
                    temperature=0.0,  # Zero temperature for deterministic financial adherence
                    max_output_tokens=1024,
                )
            except Exception as e:
                print(f"[!] Warning: Could not initialize Gemini LLM: {e}")
                self.llm = None
        else:
            self.llm = None

    def set_api_key(self, api_key: str):
        """Allows updating the Gemini API key dynamically from Streamlit UI."""
        self.api_key = api_key.strip()
        os.environ["GEMINI_API_KEY"] = self.api_key
        self._init_llm()

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves top K chunks using cosine similarity search in ChromaDB.
        Returns a list of structured citation records with similarity score and metadata.
        """
        k = top_k or self.top_k
        if not self.collection or self.collection.count() == 0:
            return []

        # Convert question into 384-dimensional query embedding
        query_embedding = self.embedding_fn.embed_query(query)

        # Execute cosine similarity retrieval
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks: List[Dict[str, Any]] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc_text, meta, dist in zip(docs, metas, dists):
            # Chroma returns cosine distance in [0, 2] where 0 is identical
            # Cosine Similarity = 1 - cosine_distance
            sim_score = max(0.0, min(1.0, 1.0 - float(dist)))
            retrieved_chunks.append({
                "text": doc_text,
                "source_document": meta.get("source_document", "Unknown Document"),
                "page_number": int(meta.get("page_number", 1)),
                "document_type": meta.get("document_type", "Financial Document"),
                "distance": round(float(dist), 4),
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": round(sim_score * 100, 1)
            })

        return retrieved_chunks

    def build_prompt_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Constructs formatted context block with citations for LLM consumption."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_document", "Unknown")
            page = chunk.get("page_number", 1)
            dtype = chunk.get("document_type", "Financial Document")
            text = chunk.get("text", "").strip()
            context_parts.append(
                f"[DOCUMENT CHUNK {i}]\n"
                f"Source Document: {source}\n"
                f"Page Number: {page}\n"
                f"Document Type: {dtype}\n"
                f"Content:\n{text}\n"
            )
        return "\n----------------------------------------\n".join(context_parts)

    def answer_question(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        1. Retrieve top 5 relevant chunks via cosine similarity
        2. Validate retrieval relevance
        3. Formulate strict context-grounded prompt
        4. Query Gemini 2.5 Flash
        5. Return structured answer and citations
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k)

        # Step 2: Handle empty retrieval gracefully
        # If no chunks or highest similarity is below a reasonable semantic threshold
        if not chunks or (len(chunks) > 0 and chunks[0]["similarity_score"] < 0.20):
            return {
                "answer": "I could not find this information in the uploaded financial documents.",
                "citations": [],
                "retrieved_chunks": [],
                "is_empty_retrieval": True,
                "model_used": self.gemini_model,
                "status": "not_found"
            }

        # Step 3: Construct context
        context_block = self.build_prompt_context(chunks)

        user_content = (
            f"Retrieved Document Context:\n"
            f"========================================\n"
            f"{context_block}\n"
            f"========================================\n\n"
            f"User Question: {question}\n\n"
            f"Instructions:\n"
            f"Answer the question strictly using the retrieved context above.\n"
            f"Format your response clearly, followed by an explicit 'Source Citations' section listing:\n"
            f"- Source Document(s)\n"
            f"- Page Number(s)\n"
            f"If the retrieved context does not contain the answer, reply exactly:\n"
            f"\"I could not find this information in the uploaded financial documents.\""
        )

        # Step 4: Invoke Gemini LLM if available
        if self.llm is not None:
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            
            # Incorporate past conversation memory if provided
            if conversation_history:
                for msg in conversation_history[-4:]:  # Keep recent context window
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))
                        
            messages.append(HumanMessage(content=user_content))
            
            try:
                response = self.llm.invoke(messages)
                answer_text = response.content.strip()
            except Exception as e:
                # In case of API quota, network error, or invalid key
                err_msg = str(e)
                if "API_KEY_INVALID" in err_msg or "400" in err_msg:
                    answer_text = (
                        f"[!] Gemini API Key Error: Please verify your Gemini API key in the sidebar.\n\n"
                        f"Retrieved context was successfully found in {chunks[0]['source_document']} (Page {chunks[0]['page_number']}):\n\n"
                        f"{chunks[0]['text'][:400]}..."
                    )
                else:
                    answer_text = (
                        f"[!] LLM Generation Error: {err_msg}\n\n"
                        f"Context retrieved from {chunks[0]['source_document']} (Page {chunks[0]['page_number']}):\n"
                        f"{chunks[0]['text'][:400]}..."
                    )
        else:
            # Fallback mode when API key is not yet configured
            # Synthesizes a direct, helpful preview from the top retrieved chunks
            answer_text = self._synthesize_fallback_answer(question, chunks)

        return {
            "answer": answer_text,
            "citations": chunks,
            "retrieved_chunks": chunks,
            "is_empty_retrieval": False,
            "model_used": self.gemini_model if self.llm else "ChromaDB Semantic Engine (API Key Required for Gemini)",
            "status": "success"
        }

    def _synthesize_fallback_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Provides a graceful response when Gemini API key is not yet entered.
        Displays top retrieved context excerpts, page numbers, and instructions.
        """
        top_chunk = chunks[0]
        doc = top_chunk["source_document"]
        page = top_chunk["page_number"]
        dtype = top_chunk["document_type"]
        score = top_chunk["similarity_percentage"]
        
        return (
            f"**Information found in `{doc}` (Page {page}, {dtype})** with **{score}% semantic similarity**:\n\n"
            f"{top_chunk['text']}\n\n"
            f"> [!NOTE]\n"
            f"> *To generate full conversational answers with Gemini 2.5 Flash, enter your `GEMINI_API_KEY` in the sidebar or in the `.env` file.*"
        )


def query_rag(query: str, persist_dir: str = "./chroma_db", api_key: Optional[str] = None) -> Dict[str, Any]:
    """Helper function to run a one-shot query against the RAG pipeline."""
    pipeline = FinancialRAGPipeline(persist_dir=persist_dir, api_key=api_key)
    return pipeline.answer_question(query)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Query Financial RAG Assistant.")
    parser.add_argument("--query", type=str, required=True, help="Question to ask the financial assistant")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument("--persist", type=str, default="./chroma_db", help="Path to ChromaDB persist dir")
    args = parser.parse_args()

    pipeline = FinancialRAGPipeline(persist_dir=args.persist, top_k=args.top_k)
    print(f"\n[?] Question: {args.query}\n")
    result = pipeline.answer_question(args.query, top_k=args.top_k)
    print("=" * 60)
    print("AI ASSISTANT RESPONSE:")
    print("=" * 60)
    print(result["answer"])
    print("\n" + "=" * 60)
    print(f"RETRIEVED CITATIONS ({len(result['citations'])} chunks):")
    print("=" * 60)
    for i, c in enumerate(result["citations"], 1):
        print(f"[{i}] {c['source_document']} | Page {c['page_number']} | Type: {c['document_type']} | Similarity: {c['similarity_percentage']}%")
        print(f"    Excerpt: {c['text'][:120]}...\n")
