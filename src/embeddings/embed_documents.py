import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_DIR = "../../data/processed/final"
EMBED_DIR = "../../data/embeddings"
os.makedirs(EMBED_DIR, exist_ok = True)

EMBEDDINGS_PATH = os.path.join(EMBED_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(EMBED_DIR, "metadata.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

all_embeddings = []
all_metadata = []

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith("_final_corpus.json"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    with open(input_path, "r", encoding = "utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        continue

    print(f"[INFO] Embedding {filename} ({len(chunks)} chunks)")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size = 32,
        show_progress_bar = True,
        normalize_embeddings = True  
    )

    all_embeddings.append(embeddings)

    for chunk in chunks:
        all_metadata.append({
            "report_id": chunk["report_id"],
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "token_count": chunk["metadata"]["token_count"]
        })

if not all_embeddings:
    raise RuntimeError("No embeddings generated — check input directory")

final_embeddings = np.vstack(all_embeddings)

np.save(EMBEDDINGS_PATH, final_embeddings)

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print(
    f"[OK] Saved {final_embeddings.shape[0]} embeddings "
    f"({final_embeddings.shape[1]} dims)"
)
