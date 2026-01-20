from typing import List, Dict

SYSTEM_INSTRUCTION = (
    "You are an assistant answering questions about OECD economic reports. "
    "Answer the question using ONLY the sources provided. "
    "Cite sources using bracketed numbers (e.g., [1], [2]). "
    "Do not repeat yourself. If the sources do not contain the answer, say: "
    "'The provided sources do not contain sufficient information to answer this question.'"
)

def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    if not question.strip():
        raise ValueError("Question must be non-empty")
    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    sources_block = "\n\n".join([f"[{i}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks, start=1)])

    prompt = (
        f"SYSTEM:\n{SYSTEM_INSTRUCTION}\n\n"
        f"SOURCES:\n{sources_block}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
    return prompt
