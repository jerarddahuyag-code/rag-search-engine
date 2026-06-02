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

def summarize_command(query: str, results: list[dict]):
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {results}

    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:"""
    response = client.models.generate_content(model=model, contents=prompt)
    answer = (response.text or "").strip().strip('"')
    return answer
