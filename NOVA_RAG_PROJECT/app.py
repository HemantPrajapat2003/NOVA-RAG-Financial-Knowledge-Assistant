"""
AI Financial Knowledge Assistant using RAG.
Interactive Streamlit application for enterprise financial document question-answering.
Powered by ChromaDB, sentence-transformers/all-MiniLM-L6-v2, and Google Gemini 2.5 Flash.
"""

import os
import time
import json
import streamlit as st
from datetime import datetime

from rag_pipeline import FinancialRAGPipeline, SYSTEM_PROMPT
from ingest import (
    ingest_documents,
    ingest_single_uploaded_pdf,
    get_database_stats,
    clear_vector_database,
    infer_document_type
)
from utils.pdf_generator import generate_all_sample_pdfs

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Knowledge Assistant (RAG)",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Financial Theme CSS
st.markdown("""
<style>
    /* Global styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-enterprise {
        background: linear-gradient(135deg, #1E3A8A, #2563EB);
        color: white;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Metrics block */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1E293B;
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
    }
    
    /* Citation Card */
    .citation-container {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 6px;
        padding: 0.8rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
    }
    .citation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 600;
        font-size: 0.85rem;
        color: #1E3A8A;
        margin-bottom: 0.4rem;
    }
    .citation-tag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .score-tag {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .citation-body {
        font-size: 0.82rem;
        color: #334155;
        line-height: 1.4;
        background: #FFFFFF;
        padding: 0.5rem;
        border-radius: 4px;
        border: 1px solid #E2E8F0;
        font-family: monospace;
        white-space: pre-wrap;
    }
    
    /* Document sidebar list */
    .doc-item {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization & Resource Caching
# ---------------------------------------------------------
@st.cache_resource
def load_rag_pipeline(persist_dir: str, gemini_model: str, api_key: str):
    return FinancialRAGPipeline(
        persist_dir=persist_dir,
        top_k=5,
        gemini_model=gemini_model,
        api_key=api_key
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if st.session_state.gemini_api_key == "your_gemini_api_key_here":
        st.session_state.gemini_api_key = ""

pipeline = load_rag_pipeline(
    persist_dir="./chroma_db",
    gemini_model="gemini-2.5-flash",
    api_key=st.session_state.gemini_api_key
)
st.session_state.pipeline = pipeline

# Synchronize API key if user updated in sidebar
if st.session_state.gemini_api_key and pipeline.api_key != st.session_state.gemini_api_key:
    pipeline.set_api_key(st.session_state.gemini_api_key)


# ---------------------------------------------------------
# Sidebar Component
# ---------------------------------------------------------
with st.sidebar:
    st.markdown('<span class="badge-enterprise">Enterprise GenAI Portfolio</span>', unsafe_allow_html=True)
    st.title("🏛️ Financial RAG")
    st.caption("Retrieval-Augmented Generation for Banking & Compliance")
    
    st.markdown("---")
    
    # 1. API Configuration
    st.subheader("🔑 Gemini LLM Settings")
    input_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Get a free key from https://aistudio.google.com/."
    )
    if input_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = input_key
        pipeline.set_api_key(input_key)
        st.success("API key updated!")
        
    model_choice = st.selectbox(
        "LLM Model",
        options=["gemini-2.5-flash", "gemini-1.5-flash"],
        index=0,
        help="Default model: Gemini 2.5 Flash as specified."
    )
    if model_choice != pipeline.gemini_model:
        pipeline.gemini_model = model_choice
        pipeline._init_llm()

    st.markdown("---")
    
    # 2. Database Status & Metrics
    st.subheader("📊 Vector DB Status")
    db_stats = get_database_stats("./chroma_db")
    total_chunks = db_stats.get("total_chunks", 0)
    total_docs = db_stats.get("document_count", 0)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_docs}</div>
            <div class="metric-lbl">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_chunks}</div>
            <div class="metric-lbl">Vector Chunks</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.caption("Embedding: `all-MiniLM-L6-v2` (384-dim) | Metric: `Cosine`")

    st.markdown("---")

    # 3. Document Management & Uploads
    st.subheader("📁 Upload Financial PDFs")
    uploaded_files = st.file_uploader(
        "Upload Financial PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports Home Loan, Personal Loan, KYC, RBI Circulars, Insurance, SOPs, FAQs"
    )
    
    if uploaded_files:
        if st.button("📥 Index Uploaded PDFs", use_container_width=True, type="primary"):
            os.makedirs("documents", exist_ok=True)
            with st.spinner("Processing & embedding uploaded files..."):
                added_count = 0
                for uploaded_file in uploaded_files:
                    save_path = os.path.join("documents", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    res = ingest_single_uploaded_pdf(save_path, persist_dir="./chroma_db")
                    if res.get("status") == "success":
                        added_count += 1
                pipeline._init_chroma()
                st.success(f"Indexed {added_count} new file(s) into ChromaDB!")
                time.sleep(1)
                st.rerun()

    # 4. Document List
    st.subheader("📚 Indexed Documents")
    if db_stats.get("documents"):
        for doc_name, details in db_stats["documents"].items():
            st.markdown(f"""
            <div class="doc-item">
                <b>📄 {doc_name}</b><br/>
                <span style="font-size:0.75rem; color:#64748B;">
                    {details.get('document_type')} • {details.get('chunk_count')} chunks ({details.get('max_page')} pages)
                </span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No documents currently indexed in ChromaDB.")
        
    st.markdown("---")
    
    # 5. Admin / Maintenance Controls
    st.subheader("⚙️ System Maintenance")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("🔄 Re-index All", use_container_width=True):
            with st.spinner("Re-indexing documents directory..."):
                res = ingest_documents("documents", "./chroma_db", force_reindex=True)
                pipeline._init_chroma()
                st.success(f"Re-indexed {res.get('total_chunks', 0)} chunks!")
                time.sleep(1)
                st.rerun()
    with col_a2:
        if st.button("🗑️ Clear DB", use_container_width=True):
            clear_vector_database("./chroma_db")
            pipeline._init_chroma()
            st.warning("Database cleared.")
            time.sleep(1)
            st.rerun()

    if st.button("⚡ Restore Standard 8 Sample PDFs", use_container_width=True):
        with st.spinner("Restoring authentic sample PDFs..."):
            generate_all_sample_pdfs("documents")
            ingest_documents("documents", "./chroma_db", force_reindex=True)
            pipeline._init_chroma()
            st.success("Sample PDFs restored and indexed!")
            time.sleep(1)
            st.rerun()

# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
st.markdown('<div class="main-title">🏛️ AI Financial Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enterprise RAG Engine powered by <b>sentence-transformers/all-MiniLM-L6-v2</b>, <b>ChromaDB</b>, and <b>Gemini 2.5 Flash</b>.</div>', unsafe_allow_html=True)

# System Architecture Info Box
with st.expander("ℹ️ System Architecture & Financial Grounding Rules", expanded=False):
    st.markdown("""
    **Core Pipeline Highlights:**
    - **Chunking Strategy:** `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200).
    - **Embeddings:** `all-MiniLM-L6-v2` generating 384-dimensional dense vectors.
    - **Vector Store:** ChromaDB with exact Cosine Similarity (`hnsw:space: cosine`).
    - **Retrieval:** Multi-document semantic search with Top K = 5.
    - **Anti-Hallucination Policy:** Responses are restricted strictly to retrieved context. Unfound information returns: *"I could not find this information in the uploaded financial documents."*
    - **Metadata Tracking:** Source Document, Page Number, and Document Type attached to every citation.
    """)

# Quick Question Action Pills
st.markdown("##### 💡 Example Financial Questions")
sample_questions = [
    "What documents are required for home loan?",
    "What is the minimum age for home loan?",
    "What is foreclosure charge for personal loan?",
    "What are RBI digital lending rules?",
    "How many days does insurance claim settlement take?",
    "What is KYC verification process?"
]

# Display quick action pill buttons
cols = st.columns(3)
clicked_sample = None
for i, q in enumerate(sample_questions):
    with cols[i % 3]:
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            clicked_sample = q

# Chat Action Toolbar (Clear Chat / Export)
col_tb1, col_tb2, col_tb3 = st.columns([6, 2, 2])
with col_tb2:
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col_tb3:
    if st.session_state.messages:
        # Prepare exportable chat history
        export_text = f"# AI Financial Knowledge Assistant Chat History\nExported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else "Assistant"
            export_text += f"### {role}:\n{m['content']}\n\n"
        st.download_button(
            "💾 Export Chat",
            data=export_text,
            file_name=f"financial_rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If citations exist, display rich collapsible citations
        if msg.get("citations"):
            with st.expander(f"📑 Retrieved Source Citations ({len(msg['citations'])} chunks)", expanded=False):
                for idx, c in enumerate(msg["citations"], 1):
                    st.markdown(f"""
                    <div class="citation-container">
                        <div class="citation-header">
                            <span><b>[{idx}] {c['source_document']}</b> — Page {c['page_number']}</span>
                            <div>
                                <span class="citation-tag">{c['document_type']}</span>
                                <span class="score-tag">Similarity: {c['similarity_percentage']}%</span>
                            </div>
                        </div>
                        <div class="citation-body">{c['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Determine Query to Run (from input or clicked sample button)
user_prompt = st.chat_input("Ask a question about loan policies, KYC, RBI circulars, insurance TAT, or credit terms...")
query_to_process = clicked_sample if clicked_sample else user_prompt

if query_to_process:
    # 1. Render and record User Message
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # 2. Run RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Searching financial knowledge base via ChromaDB & generating answer..."):
            # Pass recent conversation history for optional conversational memory
            history_tuples = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            
            result = pipeline.answer_question(
                question=query_to_process,
                conversation_history=history_tuples,
                top_k=5
            )

            answer = result["answer"]
            citations = result.get("citations", [])
            is_empty = result.get("is_empty_retrieval", False)

            # Display Answer
            st.markdown(answer)

            # Display Citations if available
            if citations and not is_empty:
                with st.expander(f"📑 Retrieved Source Citations ({len(citations)} chunks)", expanded=True):
                    for idx, c in enumerate(citations, 1):
                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-header">
                                <span><b>[{idx}] {c['source_document']}</b> — Page {c['page_number']}</span>
                                <div>
                                    <span class="citation-tag">{c['document_type']}</span>
                                    <span class="score-tag">Cosine Similarity: {c['similarity_percentage']}%</span>
                                </div>
                            </div>
                            <div class="citation-body">{c['text']}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Record in session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "citations": citations if not is_empty else []
            })
