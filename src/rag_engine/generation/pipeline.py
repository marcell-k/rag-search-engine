from rag_engine.hybrid.pipeline import build_pipeline, rrf_search


def augment(query: str, limit: int = 5, mode: str = "") -> None:
    components = build_pipeline()
    run = rrf_search(components, query, limit=limit)

    print("Search Results:")
    for i, res in enumerate(run.results, start=1):
        print(f"{i}. {res['title']}")

    docs = "\n\n".join(
        f"Document [{i}]:\nTitle: {res['title']}\nDescription: {res['description']}"
        for i, res in enumerate(run.results, start=1)
    )
    if not components.llm:
        raise RuntimeError("No LLM configured.")

    template = components.llm.load_prompt(mode)
    prompt = template.format(query=query, docs=docs)
    answer = components.llm.generate(prompt)

    answer = components.llm.generate(prompt)

    mode_key = mode.split("/")[-1]
    match mode_key:
        case "answer_question":
            print(f"\nRAG Response:\n{answer}\n")

        case "summarization":
            print(f"\nLLM Summary:\n{answer}\n")

        case "answer_with_citations":
            print(f"\nLLM Answer:\n{answer}")

        case "answer_question_detailed":
            print(f"\nAnswer:\n{answer}")

        case _:
            print(f"\nResponse:\n{answer}")
