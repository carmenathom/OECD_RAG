import os
import json
import faiss
import numpy as np

EMBED_DIR = "../../data/embeddings"
EMBEDDINGS_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(EMBED_DIR, "metadata.json")
INDEX_PATH = os.path.join(EMBED_DIR, "faiss.index")

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("embeddings.npy not found")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("metadata.json not found")

embeddings = np.load(EMBEDDINGS_PATH)

with open(METADATA_PATH, "r", encoding = "utf-8") as f:
    metadata = json.load(f)

if embeddings.ndim != 2:
    raise ValueError("Embeddings must be a 2D array")

num_vectors, dim = embeddings.shape

if num_vectors != len(metadata):
    raise ValueError(
        "Embeddings count does not match metadata length"
    )

print(f"[INFO] Loaded {num_vectors} vectors (dim={dim})")

index = faiss.IndexFlatIP(dim)

if embeddings.dtype != np.float32:
    embeddings = embeddings.astype(np.float32)

index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

print(
    f"[OK] FAISS index created and saved to {INDEX_PATH}"
)
