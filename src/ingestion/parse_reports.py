import os
import json
from pypdf import PdfReader

RAW_DIR = "../../data/raw"
EXTRACT_DIR = "../../data/processed/raw_text"
os.makedirs(EXTRACT_DIR, exist_ok = True)

for filename in os.listdir(RAW_DIR):
    if not filename.lower().endswith(".pdf"):
        continue  

    pdf_path = os.path.join(RAW_DIR, filename)
    reader = PdfReader(pdf_path)

    full_text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            full_text += extracted + "\n"

    report_id = os.path.splitext(filename)[0]
    data = {
        "report_id": report_id,
        "text": full_text
    }

    output_path = os.path.join(EXTRACT_DIR, f"{report_id}_RAW_TEXT.json")
    with open(output_path, "w", encoding = "utf-8") as f:
        json.dump(data, f, ensure_ascii = False, indent = 2)
