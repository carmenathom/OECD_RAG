"""
Benchmark semantic retrieval latency:
- Brute-force cosine similarity (NumPy)
- FAISS ANN search

Designed to be pipeline-friendly and resume-quantifiable.
"""

import os
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =====================
# Config
# =====================
EMBED_DIR = "../../data/embeddings"
EMBEDDINGS_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(EMBED_DIR, "metadata.json")
INDEX_PATH = os.path.join(EMBED_DIR, "faiss.index")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
N_QUERIES = 50
RANDOM_SEED = 42

# =====================
# Utilities
# =====================
def brute_force_search(query_vec, embeddings, top_k=5):
    scores = embeddings @ query_vec
    idx = np.argpartition(scores, -top_k)[-top_k:]
    return idx, scores[idx]

def faiss_search(index, query_vec, top_k=5):
    query_vec = query_vec.reshape(1, -1).astype(np.float32)
    scores, idx = index.search(query_vec, top_k)
    return idx[0], scores[0]

def benchmark(search_fn, queries):
    latencies = []
    for q in queries:
        start = time.perf_counter()
        search_fn(q)
        latencies.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }

# =====================
# Main
# =====================
def main():
    np.random.seed(RANDOM_SEED)

    print("[INFO] Loading embeddings + metadata")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"[INFO] {embeddings.shape[0]} vectors (dim={embeddings.shape[1]})")

    print("[INFO] Loading FAISS index")
    index = faiss.read_index(INDEX_PATH)

    print("[INFO] Loading encoder")
    model = SentenceTransformer(MODEL_NAME)

    print("[INFO] Sampling queries")
    sample_ids = np.random.choice(len(metadata), size=N_QUERIES, replace=False)
    query_texts = [metadata[i]["text"] for i in sample_ids]

    queries = model.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print("[INFO] Running benchmarks")

    bf_stats = benchmark(
        lambda q: brute_force_search(q, embeddings, TOP_K),
        queries,
    )

    faiss_stats = benchmark(
        lambda q: faiss_search(index, q, TOP_K),
        queries,
    )

    speedup = bf_stats["p95_ms"] / faiss_stats["p95_ms"]

    print("\n========== RESULTS ==========")
    print(f"Vectors: {embeddings.shape[0]}")
    print(f"Dim: {embeddings.shape[1]}")
    print(f"Queries: {N_QUERIES}")
    print("-----------------------------")
    print(f"Brute force p95: {bf_stats['p95_ms']:.2f} ms")
    print(f"FAISS p95:       {faiss_stats['p95_ms']:.2f} ms")
    print(f"Speedup (p95):   {speedup:.1f}×")
    print("=============================\n")

if __name__ == "__main__":
    main()
