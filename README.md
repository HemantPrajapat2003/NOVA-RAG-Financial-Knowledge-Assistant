# 🏛️ AI Financial Knowledge Assistant using RAG

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C?logo=chainlink)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange)](https://trychroma.com)
[![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2%20(384--dim)-7952B3)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![LLM](https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google)](https://aistudio.google.com/)

An enterprise-grade **Financial Retrieval-Augmented Generation (RAG) Assistant** designed for banking compliance, loan advisory, and customer support. The system allows users to search and query financial policy documents using semantic search and Google Gemini 2.5 Flash, strictly answering with verifiable source citations and zero hallucination.

---

## 📌 Project Objectives

1. **Accepts PDF financial documents** (Home Loans, Personal Loans, KYC, RBI Circulars, Insurance, SOPs, Credit Cards, FAQs).
2. **Extracts and processes text** from multi-page PDFs using `PyPDFLoader`.
3. **Splits documents into context-preserving chunks** using `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap).
4. **Generates 384-dimensional dense vectors** via `sentence-transformers/all-MiniLM-L6-v2`.
5. **Persists vectors in ChromaDB** with Cosine Similarity and rich metadata (`source_document`, `page_number`, `document_type`).
6. **Retrieves top-5 relevant chunks** using semantic similarity across multiple documents.
7. **Grounds answers strictly in context** via Google Gemini 2.5 Flash with an anti-hallucination system prompt.
8. **Displays verified source citations** with page numbers, document categories, and similarity scores in a modern Streamlit interface.

---

## 🏛️ System Architecture

```
                                      +--------------------------+
                                      | Financial PDF Documents  |
                                      |     (/documents)         |
                                      +-------------+------------+
                                                    |
                                                    v
                                       [ PyPDFLoader: Pages ]
                                                    |
                                                    v
                                 [ RecursiveCharacterTextSplitter ]
                                  (Chunk Size: 1000, Overlap: 200)
                                                    |
                                                    v
                              [ all-MiniLM-L6-v2 ONNX Embeddings ]
                                    (384-Dimensional Vectors)
                                                    |
                                                    v
                                      +--------------------------+
                                      | ChromaDB Vector Store    |
                                      | (./chroma_db | Cosine)   |
                                      +-------------+------------+
                                                    |
User Query ---> [ 384-dim Query Vector ]            |
      |                 |                           |
      |                 v                           |
      |          [ Cosine Similarity Top-K=5 ] <----+
      |                 |
      |                 v
      |       [ Retrieved Context Chunks ]
      |                 |
      +-----------------+
      |
      v
[ Strict Anti-Hallucination Prompt ]
      |
      v
[ Google Gemini 2.5 Flash LLM ]
      |
      v
[ Answer with Source Citations + Page Numbers ]
```

---

## 🛠️ Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Responsive, enterprise-styled UI with chat interface and citation accordions |
| **RAG Orchestration** | LangChain / LangChain Community | Modular document loaders, text splitters, and LLM chains |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Fast, accurate 384-dim dense embeddings run locally via ONNX runtime |
| **Vector Database** | ChromaDB (`./chroma_db`) | Embedded local vector database with cosine distance (`hnsw:space: cosine`) |
| **LLM** | Google Gemini 2.5 Flash | High-speed, context-faithful reasoning with structured citations |
| **Document Ingestion** | PyPDF / PyPDFLoader | Multi-page text extraction with 1-indexed page tracking |
| **PDF Generation** | ReportLab | Programmatic creation of 8 authentic multi-page banking PDF policies |

---

## 📂 Project Structure

```
finance-rag/
│
├── app.py                     # Streamlit frontend & chat interface
├── ingest.py                  # Document loader, chunking & ChromaDB ingestion
├── rag_pipeline.py            # Retrieval engine, prompt formatting & Gemini integration
├── test_rag.py                # Automated benchmark test suite (7/7 test cases)
├── requirements.txt           # Project dependencies
├── .env                       # API key configuration
├── .env.example               # Template for environment variables
│
├── documents/                 # Financial PDF documents repository
│   ├── Home_Loan_Policy.pdf
│   ├── Personal_Loan_Policy.pdf
│   ├── KYC_Guidelines.pdf
│   ├── RBI_Digital_Lending.pdf
│   ├── Insurance_Claim_Policy.pdf
│   ├── Credit_Card_Terms.pdf
│   ├── Banking_SOP.pdf
│   └── Finance_FAQ.pdf
│
├── chroma_db/                 # Persistent ChromaDB vector database
│
└── utils/
    ├── __init__.py
    ├── embeddings.py          # 384-dim all-MiniLM-L6-v2 LangChain Embeddings wrapper
    └── pdf_generator.py       # Authentic financial PDF document generator
```

---

## 📄 Financial Document Specifications

The system includes 8 authentic, multi-page financial policy documents with realistic clauses:

| Document | Category | Key Policy Information Covered |
| :--- | :--- | :--- |
| **`Home_Loan_Policy.pdf`** | Loan Policy | Min age: **21 years**, Max age: **65/70 years**, LTV tiers (90%, 80%, 75%), 0% floating foreclosure fee, required KYC, salary, and property deeds. |
| **`Personal_Loan_Policy.pdf`** | Loan Policy | Min age: **23 years**, 6-month lock-in period, tiered foreclosure fee (**4%** in yr 1, **3%** in yr 2, **2%** in yr 3, **1%** beyond yr 3), 24% penal interest. |
| **`KYC_Guidelines.pdf`** | Compliance Guideline | 4-step CDD process (OVD collection, PAN NSDL validation, CIP address check, V-CIP live video call), Re-KYC intervals (High: 2y, Med: 8y, Low: 10y). |
| **`RBI_Digital_Lending.pdf`** | Regulatory Circular | Direct disbursal mandate (no pass-through pool accounts), Key Fact Statement (KFS) APR disclosure, 3-day cooling-off period, zero contact/storage access. |
| **`Insurance_Claim_Policy.pdf`** | Insurance Policy | Cashless hospital pre-auth: **1 hour**, discharge: **2 hours**, reimbursement TAT: **15 days**, life death claims: **15 days** (investigation: 90+30 days), 2% penal interest on delay. |
| **`Credit_Card_Terms.pdf`** | Product Terms | Billing cycle (30-31 days), 20-50 day grace period, 3.65% monthly APR (43.8% p.a.), 5% Minimum Amount Due (MAD), 3-day zero fraud liability window. |
| **`Banking_SOP.pdf`** | Banking SOP | Vault dual custody, >₹50k PAN requirement, >₹10L CTR reporting, dormant account after **24 months**, 0 fee for reactivation, locker liability cap (100x rent). |
| **`Finance_FAQ.pdf`** | FAQ Document | Comprehensive consumer Q&A covering loan documents, interest rates, CKYC KIN numbers, FD TDS limits (₹40,000 / ₹50,000 senior citizens), Form 15G/15H. |

---

## ⚙️ Configuration Details

### 1. Chunking Configuration
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "• ", " ", ""]
)
```
- **Chunk Size (1000):** Captures complete financial clauses, definitions, and fee schedules.
- **Overlap (200):** Prevents cutting context between consecutive paragraphs.

### 2. Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimensions: **384**
- Metric: **Cosine Similarity**

### 3. Metadata Schema
Every chunk stored in ChromaDB contains:
```json
{
  "source_document": "Home_Loan_Policy.pdf",
  "page_number": 1,
  "document_type": "Loan Policy"
}
```

### 4. System Prompt
```
You are an AI Financial Knowledge Assistant. Answer questions only using retrieved document context. If the answer is not present in the documents, respond: "I could not find this information in the uploaded financial documents."

Rules:
• Do not hallucinate
• Do not generate assumptions
• Always provide:
  - answer
  - source document
  - page number
• Keep responses professional and concise
```

---

## 🚀 Quick Start Guide

### Step 1: Clone and Set Up Environment
```bash
# Ensure Python 3.10+ is installed
python --version

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Gemini API Key
Edit `.env` or set the environment variable:
```bash
# In .env:
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, enter your API key directly in the Streamlit sidebar at runtime).*

### Step 3: Ingest Documents into ChromaDB
```bash
# Ingest all 8 PDF documents into ChromaDB
python ingest.py

# Check vector database statistics
python ingest.py --stats
```

### Step 4: Run the Benchmark Test Suite
```bash
python test_rag.py
```
Expected output: **7/7 Passed (100.0%)**.

### Step 5: Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💬 Sample User Questions & Expected Answers

| Question | Primary Source Document | Target Page | Grounded Response Summary |
| :--- | :--- | :--- | :--- |
| **"What documents are required for home loan?"** | `Home_Loan_Policy.pdf` | Page 2 | KYC (PAN, Aadhaar), Income Proof (salary slips, Form 16, 6mo bank statement), Property title deeds, 30-yr EC, and approved municipal plan. |
| **"What is the minimum age for home loan?"** | `Home_Loan_Policy.pdf` | Page 1 | Minimum **21 years** at loan application; maximum 65 years (salaried) or 70 years (self-employed) at loan maturity. |
| **"What is foreclosure charge for personal loan?"** | `Personal_Loan_Policy.pdf` | Page 2 | 6-month lock-in (no foreclosure allowed). Months 7-12: **4%**, Months 13-24: **3%**, Months 25-36: **2%**, Beyond 36 months: **1%** + GST. |
| **"What are RBI digital lending rules?"** | `RBI_Digital_Lending.pdf` | Page 1 | Direct bank-to-bank disbursals, mandatory Key Fact Statement (KFS) APR, 3-day cooling-off period, zero access to phone contacts/storage. |
| **"How many days does insurance claim settlement take?"** | `Insurance_Claim_Policy.pdf` | Page 1 | Cashless: **1 hour** pre-auth, **2 hours** discharge; Reimbursement: **15 days**; Life death claims: **15 days** (investigation: 90+30 days). |
| **"What is KYC verification process?"** | `KYC_Guidelines.pdf` | Page 1 | 4 stages: OVD collection, PAN validation with IT database, CIP address verification, and V-CIP video call with live geo-tag and >90% face match. |

---

## 🌟 Advanced Features Implemented

- [x] **Multi-Document Retrieval:** Queries that require comparing policies automatically synthesize across relevant PDFs.
- [x] **Cosine Similarity Score Display:** Every citation includes real-time similarity metrics and color-coded badges.
- [x] **Dynamic PDF Uploader:** Upload and index custom financial PDFs directly from the UI without restart.
- [x] **Re-index & Database Clear:** One-click maintenance buttons in sidebar.
- [x] **Chat History Export:** Export conversations as formatted Markdown files.
- [x] **Strict Anti-Hallucination Fallback:** Out-of-domain queries return the standardized refusal without guesswork.
- [x] **Offline / Keyless Preview Mode:** If no API key is supplied, displays top retrieved chunks and semantic scores immediately.

---


# NOVA-RAG-Financial-Knowledge-Assistant
AI Financial Knowledge Assistant using RAG, ChromaDB, Sentence Transformers, LangChain, Gemini, and Streamlit.
