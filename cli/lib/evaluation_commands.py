import json
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"

def evaluate_results(query:str, results: list[dict], rerank_method: str) -> str:
    formatted_results = []
    for i, result in enumerate(results):
        title = f"{i + 1}. {result['movie']['title']}"
        rerank_score = ""
        rrf_score = f"RRF Score: {result['hybrid_score']}"
        indiv_scores = f"BM25: {result['keyword_score']} Semantic: {result['semantic_score']}"
        description = f"{result['movie']['description']}"
        if rerank_method:
            rerank_score = f"Re-rank score ({rerank_method}): {result['rerank_score']}"
        formatted_results.append("\n".join([title, rerank_score, rrf_score, indiv_scores, description]))
    
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {chr(10).join(formatted_results)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""
    response = client.models.generate_content(model=model, contents=prompt)
    evaluation = (response.text or "").strip().strip('"')
    evaluation = json.loads(evaluation)
    if evaluation:
        print("=================== Evaluation ====================")
        for i, result in enumerate(results):
            print(f"{i + 1}. {result['movie']['title']}: {evaluation[i]}/3")


