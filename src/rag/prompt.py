from typing import List, Dict


SYSTEM_INSTRUCTION = (
    "You are an assistant answering questions about OECD economic reports. "
    "Answer the question using ONLY the sources provided below. "
    "If the answer cannot be found in the sources, say "
    "'The provided sources do not contain sufficient information to answer this question.' "
    "Cite sources using bracketed numbers (e.g., [1], [2])."
)


def build_prompt(
    question: str,
    retrieved_chunks: List[Dict]
) -> str:

    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string")

    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    sources = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        sources.append(
            f"[{i}] {chunk['text']}"
        )

    sources_block = "\n\n".join(sources)

    prompt = f"""SYSTEM:
{SYSTEM_INSTRUCTION}

SOURCES:
{sources_block}

QUESTION:
{question}
"""

    return prompt
