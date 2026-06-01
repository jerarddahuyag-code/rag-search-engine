import os
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"

def rerank_result_individual(result: dict, query):
    doc = result.get("movie")
    prompt = f"""Rate how well this movie matches the search query.
    
    Query: "{query}"
    Movie: {doc.get("title", "")} - {doc.get("description", "")}

    Consider:
    - Direct relevance to query
    - User intent (what they're looking for)
    - Content appropriateness

    Rate 0-10 (10 = perfect match).
    Output ONLY the number in your response, no other text or explanation.

    Score:"""
    response = client.models.generate_content(model=model, contents=prompt)
    new_score = (response.text or "").strip().strip('"')
    return float(new_score)

def rerank_results(results: list[dict], query: str) -> list[dict]:
    for result in results:
        new_score = rerank_result_individual(result, query)
        result['rerank_score'] = new_score
        time.sleep(3)
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)