import argparse
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_engine.data_loader import load_data
from rag_engine.image.search import MultimodalSearch
from rag_engine.llm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe image")
    sub_parsers = parser.add_subparsers(dest="command")

    describe_parser = sub_parsers.add_parser("describe")
    describe_parser.add_argument("--image", type=str, help="Path to image")
    describe_parser.add_argument("--query", type=str, help="Query to rewrite based on the image")

    embed_parser = sub_parsers.add_parser("embed")
    embed_parser.add_argument("image", type=str, help="Path to image")

    image_parser = sub_parsers.add_parser("search")
    image_parser.add_argument("image", type=str, help="Path to image")
    image_parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    llm = LLM(client)
    documents = load_data()
    ms = MultimodalSearch(documents)

    match args.command:
        case "describe":
            mime = mimetypes.guess_type(args.image)[0] or "image/jpeg"
            with Path(args.image).open("rb") as f:
                img = f.read()

            system_prompt = llm.load_prompt("image/describe")
            parts = [system_prompt, types.Part.from_bytes(data=img, mime_type=mime), args.query.strip()]
            response = client.models.generate_content(model=llm.model, contents=parts)

            print(f"Rewritten query: {response.text.strip()}")
            if response.usage_metadata is not None:
                print(f"Total tokens:    {response.usage_metadata.total_token_count}")

        case "embed":
            ms.verify_image_embedding(args.image)

        case "search":
            res = ms.search(args.image, args.limit)
            for i, r in enumerate(res, start=1):
                print(f"{i}. {r['title']} (similarity: {r['score']:.3f}")
                print(f"\t{r['description'][:40]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
