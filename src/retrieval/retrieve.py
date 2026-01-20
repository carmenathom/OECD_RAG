import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

EMBED_DIR = "data/embeddings"
INDEX_PATH = os.path.join(EMBED_DIR, "faiss.index")
METADATA_PATH = os.path.join(EMBED_DIR, "metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError("FAISS index not found")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("metadata.json not found")

index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding = "utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer(MODEL_NAME)


def retrieve(
    query: str,
    k: int = 5
) -> List[Dict]:

    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk_meta = metadata[idx]

        results.append({
            "text": chunk_meta["text"],
            "score": float(score),
            "metadata": {
                "report_id": chunk_meta["report_id"],
                "chunk_id": chunk_meta["chunk_id"],
                "chunk_index": chunk_meta["chunk_index"],
                "token_count": chunk_meta["token_count"]
            }
        })

    return results


if __name__ == "__main__":
    query = "What factors are driving inflation across OECD economies?"
    results = retrieve(query, k = 5)

    for r in results:
        print(f"\nScore: {r['score']:.3f}")
        print(r["text"][:300], "...")
