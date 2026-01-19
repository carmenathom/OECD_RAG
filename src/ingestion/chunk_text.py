import os
import json
import spacy
from typing import List

RAW_DIR = "../../data/processed/raw_text"
CHUNK_DIR = "../../data/processed/chunked_text"
os.makedirs(CHUNK_DIR, exist_ok = True)

TARGET_TOKENS = 250
MAX_TOKENS = 300
SENTENCE_OVERLAP = 2  

nlp = spacy.load("en_core_web_sm")

def sentence_tokenize(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def count_tokens(text: str) -> int:
    return len(nlp(text))

def build_chunks(sentences: List[str]) -> List[str]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        if current_chunk and current_tokens + sent_tokens > MAX_TOKENS:
            chunks.append(" ".join(current_chunk))

            current_chunk = current_chunk[-SENTENCE_OVERLAP:]
            current_tokens = sum(count_tokens(s) for s in current_chunk)

        current_chunk.append(sent)
        current_tokens += sent_tokens

        if current_tokens >= TARGET_TOKENS:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-SENTENCE_OVERLAP:]
            current_tokens = sum(count_tokens(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

for filename in os.listdir(RAW_DIR):
    if not filename.endswith("_RAW_TEXT.json"):
        continue

    input_path = os.path.join(RAW_DIR, filename)
    with open(input_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    report_id = data["report_id"]
    text = data["text"]

    sentences = sentence_tokenize(text)
    chunks = build_chunks(sentences)

    records = []
    for idx, chunk in enumerate(chunks):
        records.append({
            "report_id": report_id,
            "chunk_id": f"{report_id}_{idx:03d}",
            "chunk_index": idx,
            "text": chunk,
            "metadata": {
                "token_count": count_tokens(chunk)
            }
        })

    output_path = os.path.join(CHUNK_DIR, f"{report_id}_CHUNKS.json")
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(records, f, ensure_ascii = False, indent = 2)
