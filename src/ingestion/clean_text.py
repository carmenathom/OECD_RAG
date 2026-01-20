import os
import json
import re
import unicodedata
from collections import Counter

RAW_JSON_DIR = "../../data/processed/raw_text"
CLEAN_JSON_DIR = "../../data/processed/clean_text"
os.makedirs(CLEAN_JSON_DIR, exist_ok=True)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "−": "-",
        "…": "...",
        "\u00a0": " ",  
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def fix_hyphenated_line_breaks(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def remove_page_numbers(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*\d+\s*", line):
            continue
        lines.append(line)
    return "\n".join(lines)


def collapse_newlines(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def remove_repeated_boilerplate(text: str) -> str:
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

    counts = Counter(paragraphs)

    filtered = []
    for p in paragraphs:
        if counts[p] > 1 and any(
            key in p.lower()
            for key in [
                "oecd",
                "copyright",
                "secretary-general",
                "official views",
                "published under the responsibility",
            ]
        ):
            continue
        filtered.append(p)

    return "\n\n".join(filtered)


def clean_text(raw_text: str) -> str:
    text = normalize_unicode(raw_text)
    text = fix_hyphenated_line_breaks(text)
    text = remove_page_numbers(text)
    text = collapse_newlines(text)
    text = remove_repeated_boilerplate(text)
    text = normalize_whitespace(text)
    text = collapse_newlines(text)
    return text


for filename in os.listdir(RAW_JSON_DIR):
    if not filename.lower().endswith(".json"):
        continue

    input_path = os.path.join(RAW_JSON_DIR, filename)
    with open(input_path, "r", encoding = "utf-8") as f:
        data = json.load(f)

    report_id = data.get("report_id")
    raw_text = data.get("text", "")

    cleaned = clean_text(raw_text)

    output = {
        "report_id": report_id,
        "text": cleaned,
    }

    output_path = os.path.join(
        CLEAN_JSON_DIR, f"{report_id}_CLEAN_TEXT.json"
    )
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(output, f, ensure_ascii = False, indent = 2)
