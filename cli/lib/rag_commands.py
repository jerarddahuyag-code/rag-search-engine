import os
from typing import Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"

def rag_command(query: str, docs: list[dict]):
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {docs}

    Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip().strip('"')
    return answer
