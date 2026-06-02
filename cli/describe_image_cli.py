import argparse
import mimetypes
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from lib.utils import DEFAULT_K, DEFAULT_SEARCH_LIMIT
from lib.hybrid_search_commands import rrf_search_command
from lib.rag_commands import citation_command, question_command, rag_command, summarize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Describe Image CLI")
  
    parser.add_argument("--image", type=str, required=True, help="Image for context")
    parser.add_argument("--query", type=str, required=True, help="Query for search")
  

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()
    
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    model = "gemma-4-31b-it"
    model = "gemini-3.1-flash-lite"

    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]
    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                )
            ]
        )
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()