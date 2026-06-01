import json
import math
import os
from typing import TypedDict

DEFAULT_SEARCH_LIMIT = 5

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

DEFAULT_ALPHA = 0.5
DEFAULT_K = 60

SCORE_PRECISION = 4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")
CACHE_DIRECTORY = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75

class Movie(TypedDict):
    id: int
    title: str
    description: str

class SearchResult(TypedDict):
    movie: Movie
    score: float

class HybridSearchResult(TypedDict):
    movie: Movie
    keyword_score: float
    semantic_score: float
    hybrid_score: float

class ChunkMetaData(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int
    
class ChunkScore(TypedDict):
    movie_idx: int
    chunk_idx: int
    score: float

def get_spell_check_prompt(query: str) -> str:
    return f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
    """

def get_rewrite_query_prompt(query: str) -> str:
    return f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
    """

def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def format_search_result(doc_id: int, title: str, document: str, score: float, metadata=None) -> SearchResult:
    formatted_movie: Movie = {
        'id': doc_id,
        'title': title,
        'description': document[:100]
    }
    return {
        'movie': formatted_movie,
        'score': round(score, SCORE_PRECISION)
    }

def format_hybrid_search_result(doc_id: int, title: str, document: str, hybrid_score: float, bm25_score: float, semantic_score:float, metadata=None) -> HybridSearchResult:
    formatted_movie: Movie = {
        'id': doc_id,
        'title': title,
        'description': document[:100]
    }
    return {
        'movie': formatted_movie,
        'hybrid_score': round(hybrid_score, SCORE_PRECISION),
        'keyword_score': round(bm25_score, SCORE_PRECISION),
        'semantic_score': round(semantic_score, SCORE_PRECISION)
    }

