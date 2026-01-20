def main():
    import torch
    import torch.multiprocessing
    torch.multiprocessing.set_start_method("spawn", force=True)

    query = input("Question: ")

    from src.retrieval.retrieve import retrieve
    from src.rag.prompt import build_prompt
    from src.rag.generate import generate_answer

    retrieved = retrieve(query, k=5)
    prompt = build_prompt(query, retrieved)
    result = generate_answer(prompt)

    print("\nANSWER:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()
