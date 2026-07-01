from rag_engine.filing_search.pipeline import build_pipeline, search_hybrid


async def augment(query: str, limit: int = 5, mode: str = "") -> None:
    components = await build_pipeline()
    try:
        results = await search_hybrid(query, cik=None, sc=components.embedder, cr=components.repo, limit=limit)

        print("Search Results:")
        for i, res in enumerate(results, start=1):
            print(f"{i}. {res['sec_title']} — {res['company_name']} ({res['filing_type']})")

        docs = "\n\n".join(
            f"Document [{i}]:\n{res['company_name']} {res['filing_type']} — {res['sec_title']}\n{res['content']}"
            for i, res in enumerate(results, start=1)
        )
        if not components.llm:
            raise RuntimeError("No LLM configured.")

        template = components.llm.load_prompt(mode)
        prompt = template.format(query=query, docs=docs)

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
    finally:
        await components.client.disconnect()
