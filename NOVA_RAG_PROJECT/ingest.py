"""
Document Ingestion Module for Financial RAG Assistant.
Loads financial PDF documents, extracts text, performs chunking,
generates 384-dim embeddings via sentence-transformers/all-MiniLM-L6-v2,
and persists them into ChromaDB with comprehensive metadata.
"""

import os
import sys
import shutil
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from utils.embeddings import get_embedding_function

# Known mapping for standard enterprise banking documents
DOCUMENT_TYPE_MAP = {
    "Home_Loan_Policy.pdf": "Loan Policy",
    "Personal_Loan_Policy.pdf": "Loan Policy",
    "KYC_Guidelines.pdf": "Compliance Guideline",
    "RBI_Digital_Lending.pdf": "Regulatory Circular",
    "Insurance_Claim_Policy.pdf": "Insurance Policy",
    "Credit_Card_Terms.pdf": "Product Terms",
    "Banking_SOP.pdf": "Banking SOP",
    "Finance_FAQ.pdf": "FAQ Document",
}

COLLECTION_NAME = "financial_knowledge_base"


def infer_document_type(filename: str, sample_text: str = "") -> str:
    """Infer the document type from the filename or content keywords."""
    clean_name = os.path.basename(filename)
    if clean_name in DOCUMENT_TYPE_MAP:
        return DOCUMENT_TYPE_MAP[clean_name]
    
    lower_name = clean_name.lower()
    lower_text = sample_text.lower()[:500] if sample_text else ""
    combined = lower_name + " " + lower_text
    
    if "home loan" in combined or "housing" in combined:
        return "Loan Policy"
    elif "personal loan" in combined or "unsecured loan" in combined or "credit policy" in combined:
        return "Loan Policy"
    elif "kyc" in combined or "aml" in combined or "due diligence" in combined:
        return "Compliance Guideline"
    elif "rbi" in combined or "circular" in combined or "regulatory" in combined:
        return "Regulatory Circular"
    elif "insurance" in combined or "claim" in combined:
        return "Insurance Policy"
    elif "credit card" in combined or "tariff" in combined or "terms" in combined:
        return "Product Terms"
    elif "sop" in combined or "standard operating" in combined or "procedure" in combined:
        return "Banking SOP"
    elif "faq" in combined or "questions" in combined:
        return "FAQ Document"
    return "Financial Document"


def load_pdf_document(file_path: str) -> List[Document]:
    """
    Loads a single PDF document using PyPDFLoader with fallback to pypdf.
    Attaches standardized metadata: source_document, page_number (1-indexed), document_type.
    """
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    source_filename = pdf_path.name
    loaded_pages: List[Document] = []
    
    try:
        loader = PyPDFLoader(str(pdf_path))
        raw_pages = loader.load()
        doc_type = infer_document_type(source_filename, raw_pages[0].page_content if raw_pages else "")
        
        for i, page in enumerate(raw_pages):
            # Ensure page_number is 1-indexed as requested
            page_num = int(page.metadata.get("page", i)) + 1
            content = page.page_content.strip()
            if content:
                loaded_pages.append(Document(
                    page_content=content,
                    metadata={
                        "source_document": source_filename,
                        "page_number": page_num,
                        "document_type": doc_type,
                        "file_path": str(pdf_path.resolve())
                    }
                ))
    except Exception as e:
        print(f"[!] PyPDFLoader note for '{source_filename}': {e}. Attempting direct pypdf fallback...")
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            first_text = reader.pages[0].extract_text() if len(reader.pages) > 0 else ""
            doc_type = infer_document_type(source_filename, first_text)
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    loaded_pages.append(Document(
                        page_content=text.strip(),
                        metadata={
                            "source_document": source_filename,
                            "page_number": i + 1,
                            "document_type": doc_type,
                            "file_path": str(pdf_path.resolve())
                        }
                    ))
        except Exception as fallback_err:
            print(f"[!] Error reading '{source_filename}': {fallback_err}")
                
    return loaded_pages



def split_documents_into_chunks(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits loaded document pages into context-preserving chunks
    using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "• ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    # Ensure all chunks preserve the required metadata fields cleanly
    for chunk in chunks:
        chunk.metadata.setdefault("source_document", "Unknown.pdf")
        chunk.metadata.setdefault("page_number", 1)
        chunk.metadata.setdefault("document_type", "Financial Document")
        
    return chunks


def get_chroma_client(persist_dir: str = "./chroma_db") -> chromadb.PersistentClient:
    """Returns a PersistentClient for ChromaDB."""
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_or_create_collection(client: chromadb.PersistentClient, collection_name: str = COLLECTION_NAME):
    """
    Gets or creates a ChromaDB collection configured for Cosine Similarity.
    Metadata 'hnsw:space': 'cosine' guarantees Cosine Similarity as required.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def ingest_documents(
    docs_dir: str = "documents",
    persist_dir: str = "./chroma_db",
    force_reindex: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Main ingestion routine.
    Loads all PDFs in docs_dir, chunks them, embeds them, and stores in ChromaDB.
    If force_reindex is False and chunks already exist, skips reprocessing.
    """
    start_time = time.time()
    os.makedirs(docs_dir, exist_ok=True)
    os.makedirs(persist_dir, exist_ok=True)
    
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, COLLECTION_NAME)
    current_count = collection.count()
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"[!] Warning: No PDF files found in '{docs_dir}' directory.")
        return {
            "status": "warning",
            "message": f"No PDF files found in '{docs_dir}' directory.",
            "total_documents": 0,
            "total_chunks": current_count,
            "documents_indexed": []
        }
        
    if current_count > 0 and not force_reindex:
        # Check if all files in docs_dir are already recorded in metadata
        all_metadata = collection.get(include=["metadatas"])["metadatas"]
        indexed_files = set(m.get("source_document") for m in all_metadata if m)
        unindexed_files = [f for f in pdf_files if f not in indexed_files]
        
        if not unindexed_files:
            print(f"[OK] Database already populated with {current_count} chunks across {len(indexed_files)} files.")
            print("[OK] Reloading embeddings without reprocessing (use force_reindex=True to overwrite).")
            return {
                "status": "cached",
                "message": "Loaded existing embeddings from ChromaDB without reprocessing.",
                "total_documents": len(indexed_files),
                "total_chunks": current_count,
                "documents_indexed": sorted(list(indexed_files)),
                "elapsed_seconds": round(time.time() - start_time, 2)
            }
        else:
            print(f"[*] Found {len(unindexed_files)} new files to ingest incrementally: {unindexed_files}")
            files_to_process = unindexed_files
    else:
        if force_reindex and current_count > 0:
            print(f"[*] Clearing existing {current_count} chunks from collection...")
            client.delete_collection(COLLECTION_NAME)
            collection = get_or_create_collection(client, COLLECTION_NAME)
        files_to_process = pdf_files

    # 1. Load documents
    all_pages: List[Document] = []
    print(f"[*] Loading {len(files_to_process)} PDF file(s) from '{docs_dir}'...")
    for filename in sorted(files_to_process):
        pdf_path = os.path.join(docs_dir, filename)
        pages = load_pdf_document(pdf_path)
        all_pages.extend(pages)
        print(f"    - Loaded '{filename}': {len(pages)} page(s)")
        
    if not all_pages:
        return {
            "status": "empty",
            "message": "No text could be extracted from PDF documents.",
            "total_documents": len(files_to_process),
            "total_chunks": collection.count(),
            "documents_indexed": []
        }

    # 2. Chunk documents
    print(f"[*] Splitting text using RecursiveCharacterTextSplitter (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    chunks = split_documents_into_chunks(all_pages, chunk_size, chunk_overlap)
    print(f"    - Generated {len(chunks)} text chunks.")

    # 3. Generate embeddings
    print("[*] Generating 384-dimensional embeddings using all-MiniLM-L6-v2...")
    embedding_fn = get_embedding_function()
    chunk_texts = [c.page_content for c in chunks]
    embeddings = embedding_fn.embed_documents(chunk_texts)

    # 4. Prepare data for ChromaDB
    ids = []
    metadatas = []
    for idx, chunk in enumerate(chunks):
        doc_name = chunk.metadata.get("source_document", "doc").replace(".pdf", "")
        page_num = chunk.metadata.get("page_number", 1)
        chunk_id = f"{doc_name}_p{page_num}_c{idx}_{int(time.time()*1000)%1000000}"
        ids.append(chunk_id)
        metadatas.append({
            "source_document": str(chunk.metadata.get("source_document", "Unknown.pdf")),
            "page_number": int(chunk.metadata.get("page_number", 1)),
            "document_type": str(chunk.metadata.get("document_type", "Financial Document"))
        })

    # 5. Insert into ChromaDB
    print(f"[*] Storing {len(chunks)} chunks into ChromaDB at '{persist_dir}'...")
    # Add in batches of 100 for optimal memory and progress reporting
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end],
            embeddings=embeddings[i:end],
            documents=chunk_texts[i:end],
            metadatas=metadatas[i:end]
        )
        
    total_in_db = collection.count()
    elapsed = round(time.time() - start_time, 2)
    print(f"[OK] Ingestion complete in {elapsed}s! Total chunks in ChromaDB: {total_in_db}")

    # Collect indexed file list
    all_metadata = collection.get(include=["metadatas"])["metadatas"]
    all_files = sorted(list(set(m.get("source_document") for m in all_metadata if m)))

    return {
        "status": "success",
        "message": f"Successfully ingested {len(files_to_process)} document(s) ({len(chunks)} chunks).",
        "total_documents": len(all_files),
        "total_chunks": total_in_db,
        "documents_indexed": all_files,
        "elapsed_seconds": elapsed
    }


def ingest_single_uploaded_pdf(
    file_path: str,
    persist_dir: str = "./chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Dict[str, Any]:
    """
    Ingests a single newly uploaded PDF file into the existing ChromaDB collection.
    """
    start_time = time.time()
    filename = os.path.basename(file_path)
    client = get_chroma_client(persist_dir)
    collection = get_or_create_collection(client, COLLECTION_NAME)

    # Remove any prior chunks for this exact filename to prevent duplicate entries
    try:
        collection.delete(where={"source_document": filename})
    except Exception:
        pass

    pages = load_pdf_document(file_path)
    if not pages:
        return {"status": "error", "message": f"Could not extract text from '{filename}'"}

    chunks = split_documents_into_chunks(pages, chunk_size, chunk_overlap)
    embedding_fn = get_embedding_function()
    chunk_texts = [c.page_content for c in chunks]
    embeddings = embedding_fn.embed_documents(chunk_texts)

    ids = []
    metadatas = []
    doc_stem = filename.replace(".pdf", "")
    for idx, chunk in enumerate(chunks):
        page_num = chunk.metadata.get("page_number", 1)
        chunk_id = f"{doc_stem}_p{page_num}_c{idx}_{int(time.time()*1000)%1000000}"
        ids.append(chunk_id)
        metadatas.append({
            "source_document": filename,
            "page_number": int(chunk.metadata.get("page_number", 1)),
            "document_type": str(chunk.metadata.get("document_type", "Financial Document"))
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas
    )

    return {
        "status": "success",
        "message": f"Successfully indexed '{filename}' ({len(chunks)} chunks across {len(pages)} pages).",
        "chunks_added": len(chunks),
        "pages_processed": len(pages),
        "total_chunks": collection.count(),
        "elapsed_seconds": round(time.time() - start_time, 2)
    }


def get_database_stats(persist_dir: str = "./chroma_db") -> Dict[str, Any]:
    """
    Returns high-level statistics of the vector database.
    """
    if not os.path.exists(persist_dir):
        return {"exists": False, "total_chunks": 0, "documents": {}, "document_count": 0}
        
    try:
        client = get_chroma_client(persist_dir)
        collection = get_or_create_collection(client, COLLECTION_NAME)
        count = collection.count()
        if count == 0:
            return {"exists": True, "total_chunks": 0, "documents": {}, "document_count": 0}
            
        data = collection.get(include=["metadatas"])
        docs_summary: Dict[str, Dict[str, Any]] = {}
        for m in data.get("metadatas", []):
            if not m:
                continue
            doc = m.get("source_document", "Unknown")
            page = m.get("page_number", 1)
            dtype = m.get("document_type", "Financial Document")
            if doc not in docs_summary:
                docs_summary[doc] = {"chunk_count": 0, "max_page": 0, "document_type": dtype}
            docs_summary[doc]["chunk_count"] += 1
            docs_summary[doc]["max_page"] = max(docs_summary[doc]["max_page"], page)
            
        return {
            "exists": True,
            "total_chunks": count,
            "document_count": len(docs_summary),
            "documents": docs_summary
        }
    except Exception as e:
        return {"exists": False, "error": str(e), "total_chunks": 0, "documents": {}, "document_count": 0}


def clear_vector_database(persist_dir: str = "./chroma_db") -> bool:
    """
    Resets the ChromaDB collection and database directory.
    """
    try:
        client = get_chroma_client(persist_dir)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"[!] Error clearing vector database: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial PDFs into ChromaDB.")
    parser.add_argument("--docs", type=str, default="documents", help="Directory containing PDF files")
    parser.add_argument("--persist", type=str, default="./chroma_db", help="Directory to persist ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force reindexing and overwrite existing database")
    parser.add_argument("--clear", action="store_true", help="Clear the vector database and exit")
    parser.add_argument("--stats", action="store_true", help="Show database statistics and exit")
    args = parser.parse_args()

    if args.clear:
        success = clear_vector_database(args.persist)
        print("[OK] Vector database cleared." if success else "[!] Failed to clear database.")
        sys.exit(0)

    if args.stats:
        stats = get_database_stats(args.persist)
        print("--- ChromaDB Statistics ---")
        print(f"Total Chunks: {stats.get('total_chunks', 0)}")
        print(f"Total Documents: {stats.get('document_count', 0)}")
        for doc, details in stats.get("documents", {}).items():
            print(f"  - {doc} ({details.get('document_type')}) - {details.get('chunk_count')} chunks, {details.get('max_page')} page(s)")
        sys.exit(0)

    result = ingest_documents(docs_dir=args.docs, persist_dir=args.persist, force_reindex=args.force)
    print("\nSummary Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
