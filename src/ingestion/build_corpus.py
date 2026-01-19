import os
import json
import hashlib
from typing import List, Dict

INPUT_DIR = "../../data/processed/chunked_text"
OUTPUT_DIR = "../../data/processed/final"
os.makedirs(OUTPUT_DIR, exist_ok = True)

MIN_TOKENS = 50
MAX_TOKENS = 350

def token_count(text: str) -> int:
    return len(text.split())

def hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def validate_chunk(chunk: Dict) -> bool:
    required_fields = {"report_id", "chunk_id", "chunk_index", "text"}
    if not required_fields.issubset(chunk.keys()):
        return False

    if not isinstance(chunk["chunk_index"], int):
        return False

    if not isinstance(chunk["text"], str):
        return False

    if not chunk["text"].strip():
        return False

    return True

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith("_CHUNKS.json"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    with open(input_path, "r", encoding = "utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list) or not chunks:
        print(f"[SKIP] {filename}: empty or invalid")
        continue

    report_id = chunks[0].get("report_id", os.path.splitext(filename)[0])

    seen_hashes = set()
    final_chunks = []

    for chunk in chunks:
        if not validate_chunk(chunk):
            continue

        text = chunk["text"].strip()
        n_tokens = token_count(text)

        if n_tokens < MIN_TOKENS or n_tokens > MAX_TOKENS:
            continue

        text_hash = hash_text(text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        final_chunks.append({
            "report_id": report_id,
            "chunk_id": chunk["chunk_id"],
            "chunk_index": chunk["chunk_index"],
            "text": text,
            "metadata": {
                "token_count": n_tokens,
                "text_hash": text_hash
            }
        })

    final_chunks.sort(key = lambda x: x["chunk_index"])
    for new_idx, chunk in enumerate(final_chunks):
        chunk["chunk_index"] = new_idx

    if not final_chunks:
        print(f"[WARN] {report_id}: no valid chunks after filtering")
        continue

    output_path = os.path.join(
        OUTPUT_DIR, f"{report_id}_final_corpus.json"
    )

    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(final_chunks, f, ensure_ascii = False, indent = 2)

    print(
        f"[OK] {report_id}: "
        f"{len(final_chunks)} chunks finalized"
    )