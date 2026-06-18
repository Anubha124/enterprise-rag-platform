Enterprise RAG Platform

A Retrieval-Augmented Generation (RAG) system for document question-answering, combining hybrid search, cross-encoder reranking, and a measured evaluation framework — built to avoid the most common RAG failure mode: shipping retrieval that feels good without proving it.

Key Results

MetricHybrid Retrieval (FAISS + BM25)+ Cross-Encoder RerankingHit Rate@5100%100%MRR0.3830.417 (+8.7%)

Retrieval coverage was already strong — the correct chunk was found for every test question. Reranking improved where that chunk was ranked, which is what actually drives downstream answer quality. Full methodology and scripts are in evaluation/.


Evaluated on a hand-labeled set of test questions — see evaluation/test_questions.py. (Fill in your exact question count here once finalized.)



Overview

Traditional LLMs only know what they were trained on — they can't answer questions about a user's own documents. This project solves that by ingesting documents, retrieving the most relevant sections, and generating answers grounded in — and cited to — the source material, reducing hallucination.

Architecture

User
  │
  ▼
Streamlit Frontend
  │  HTTP
  ▼
FastAPI Backend
  │
  ▼
Hybrid Retriever (FAISS + BM25)
  │
  ▼
Cross-Encoder Reranker
  │
  ▼
Groq · Llama 3.3 70B
  │
  ▼
Answer + Source Citations

Features


Multi-document upload (PDF, DOCX, TXT, CSV)
Chunking pipeline for retrieval-optimized text segments
Semantic search via BAAI/bge-small-en-v1.5 embeddings (Sentence Transformers)
Persistent FAISS vector store
BM25 keyword retrieval for exact-match terms (names, certifications, etc.)
Hybrid retrieval combining semantic + keyword search
Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
Source-grounded citations on every answer
Retrieval evaluation framework (Hit Rate@5, MRR) comparing retrieval strategies head-to-head
FastAPI backend with Swagger docs
Streamlit frontend for upload, querying, and source display


Tech Stack

AI / ML: Sentence Transformers, BGE embeddings, Hugging Face, Cross-Encoder reranker, Groq (Llama 3.3 70B)
Retrieval: FAISS, BM25 (rank-bm25)
Backend: FastAPI, Uvicorn
Frontend: Streamlit
Data processing: Pandas, PyPDF, python-docx

Project Structure

enterprise-rag-platform/
├── backend/
│   └── src/
│       ├── api/
│       ├── embeddings/
│       ├── ingestion/
│       ├── llm/
│       ├── retrieval/
│       └── utils/
├── frontend/
│   └── app.py
├── evaluation/
│   ├── test_questions.py
│   ├── evaluate.py
│   ├── mrr.py
│   └── reranker_eval.py
├── data/
├── vectorstore/
│   ├── faiss.index
│   └── chunks.pkl
├── tests/
├── notebooks/
├── requirements.txt
└── .env.example

Getting Started

bashgit clone https://github.com/<your-username>/enterprise-rag-platform.git
cd enterprise-rag-platform
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

Create a .env file:

GROQ_API_KEY=your_key_here

Run the backend:

bashuvicorn backend.src.api.main:app --reload

Swagger docs: http://127.0.0.1:8000/docs

Run the frontend:

bashstreamlit run frontend/app.py

Evaluation Methodology

Retrieval quality was measured against a hand-labeled test set mapping questions to their expected source chunk:


Hit Rate@5 — whether the correct chunk appears anywhere in the top 5 retrieved results.
MRR (Mean Reciprocal Rank) — how high the correct chunk is ranked, not just whether it's present.


Both metrics were computed for hybrid retrieval alone, then again after cross-encoder reranking, to isolate the reranker's actual contribution rather than assume it helped.

Roadmap


 Expand evaluation set with paraphrased / keyword-mismatched questions to stress-test retrieval further
 Live deployment with public demo link
 Containerize with Docker
 Swap FAISS for a managed vector store (e.g. pgvector, Pinecone) for multi-user scale
