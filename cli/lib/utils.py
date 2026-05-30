import json
import math
import os
from typing import TypedDict

DEFAULT_SEARCH_LIMIT = 5

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

DEFAULT_ALPHA = 0.5

SCORE_PRECISION = 2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
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
