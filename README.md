# OECD_RAG
Retrieval-Augmented Analysis System for OECD Economic Reports


### Overview
This project implements a retrieval-augmented generation (RAG) pipeline for automated analysis of OECD economic and financial reports. The system enables users to query large collections of unstructured OECD documents and receive grounded, source-backed answers using a local language model.

The primary goal is to reduce manual research time when working with long-form policy and economic reports, while maintaining transparency and reproducibility.


### Repository Structure
```
oecd-rag-system/
├── data/
│ ├── raw/ # Original OECD PDFs
│ ├── processed/ # Extracted text and chunks
│ └── embeddings/ # FAISS index and embedding metadata
│
├── src/
│ ├── ingestion/ # PDF parsing and text extraction
│ ├── embeddings/ # Embedding generation and vector store
│ ├── retrieval/ # Query-time document retrieval
│ ├── rag/ # Prompt construction and LLM generation
│ ├── evaluation/ # Retrieval and latency benchmarks
│ └── app/ # CLI / API entry points
│
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── requirements.txt
└── README.md
```

### Retrieval Latency Benchmark

We benchmarked semantic search latency on a corpus of 10,357 document chunk embeddings
(384-dimensional, all-MiniLM-L6-v2).

**Methodology**
- Baseline: brute-force cosine similarity over all vectors (NumPy)
- Optimized: FAISS ANN search (IndexFlatIP)
- Metric: p95 query latency across 50 sampled queries

**Results**

| Method        | p95 Latency |
|--------------|-------------|
| Brute force  | 4.21 ms     |
| FAISS        | 0.94 ms     |

FAISS achieved a **4.5× reduction in p95 retrieval latency** compared to brute-force search.

# Disclaimer

This project is for research and educational purposes and is not affiliated with or endorsed by the OECD.
