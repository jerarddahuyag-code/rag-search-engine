import json
import os
import time
from typing import Literal, Optional

from lib.utils import DEFAULT_SEARCH_LIMIT
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
# model = "gemma-4-31b-it"
model = "gemini-3.1-flash-lite"

def individual_rerank(result: dict, query):
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

def batch_rerank(results: list[dict], query):
    prompt = f"""Rank the movies listed below by relevance to the following search query.

    Query: "{query}"

    Movies:
    {results}

    Return the movie IDs in order of relevance, best match first.

    Your response must be a raw JSON array of integers.
    Do not wrap the JSON in Markdown. Do not use a ```json code block.
    Do not include any explanatory text.

    For example:
    [75, 12, 34, 2, 1]

    Ranking:"""

    response = client.models.generate_content(model=model, contents=prompt)
    doc_ids = (response.text or "").strip().strip('"')
    return json.loads(doc_ids)

def cross_encoding(pairs: list[str, str]):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    scores = cross_encoder.predict(pairs)
    return scores

def rerank_results(results: list[dict], 
                   query: str,
                   rerank_method: Literal["individual", "batch", "cross_encoder"] | None = None,
                   limit:int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    
    match rerank_method:
        case "individual":
            for result in results:
                new_score = individual_rerank(result, query)
                result['rerank_score'] = new_score
                time.sleep(3)
                return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:limit]
        case "batch":
            ranked_ids = batch_rerank(results, query)
            new_ranking = []
            for i, doc_id in enumerate(ranked_ids):
                movie = next((mov for mov in results if mov["movie"]["id"] == doc_id),{})
                movie['rerank_score'] = i + 1
                new_ranking.append(movie)
            return new_ranking
        case "cross_encoder":
            pairs = []
            for result in results:
                doc = result.get("movie")
                pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])
            new_scores = cross_encoding(pairs)
            for i, result in enumerate(results):
                result['rerank_score'] = new_scores[i]
            return sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:limit]

    return results

