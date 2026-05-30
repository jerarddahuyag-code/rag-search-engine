import os

from lib.utils import DEFAULT_ALPHA, DEFAULT_SEARCH_LIMIT, HybridSearchResult, Movie, SearchResult, format_hybrid_search_result, format_search_result, load_movies

from .keyword_search_commands import InvertedIndex
from .semantic_search_commands import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridSearchResult]:
        new_limit = limit * 500
        keyword_results = self._bm25_search(query, new_limit)
        semantic_results = self.semantic_search.search_chunks(query, new_limit)
        hybrid_search_results: dict[int, HybridSearchResult] = {}
        normalized_keyword_scores = min_max_normalization([x['score'] for x in keyword_results])
        normalized_semantic_scores = min_max_normalization([x['score'] for x in keyword_results])
        for i, result in enumerate(keyword_results):
            doc_id = result['movie']['id']
            result['score'] = normalized_keyword_scores[i]
            if doc_id in hybrid_search_results:
                hybrid_search_results[doc_id]['keyword_score'] = result['score']
            else:
                hybrid_search_results[doc_id] = {
                        'movie': result['movie'],
                        'keyword_score': result['score'],
                        'semantic_score': 0.0,
                        'hybrid_score': 0.0
                    }
                
        for i, result in enumerate(semantic_results):
            doc_id = result['movie']['id']
            result['score'] = normalized_semantic_scores[i]
            if doc_id in hybrid_search_results:
                hybrid_search_results[doc_id]['semantic_score'] = result['score']
            else:
                hybrid_search_results[doc_id] = {
                        'movie': result['movie'],
                        'semantic_score': result['score'],
                        'hybrid_score': 0.0,
                        'keyword_score': 0.0
                    }

        results: list[HybridSearchResult] = []
        for hybrid_result in hybrid_search_results.values():
            keyword_score: float = hybrid_result['keyword_score']
            semantic_score: float = hybrid_result['semantic_score']
            hybrid_result['hybrid_score'] = hybrid_score(keyword_score, semantic_score, alpha)
            results.append(format_hybrid_search_result(
                doc_id=hybrid_result['movie']['id'],
                title=hybrid_result['movie']['title'],
                document=hybrid_result['movie']['description'],
                semantic_score=semantic_score,
                bm25_score=keyword_score,
                hybrid_score=hybrid_result['hybrid_score']
            ))
        sorted_results = sorted(results, key=lambda x: x['hybrid_score'], reverse=True)[:limit]
        return sorted_results
    
    def rrf_search(self, query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        new_limit = limit * 500
        keyword_results = self._bm25_search(query, new_limit)
        semantic_results = self.semantic_search.search_chunks(query, new_limit)
        hybrid_search_results: dict[int, HybridSearchResult] = {}  
        for i, result in enumerate(keyword_results):
            doc_id = result['movie']['id']
            if doc_id in hybrid_search_results:
                hybrid_search_results[doc_id]['keyword_score'] = i + 1
            else:
                hybrid_search_results[doc_id] = {
                        'movie': result['movie'],
                        'keyword_score': i + 1,
                        'semantic_score': 0.0,
                        'hybrid_score': 0.0
                    }
                
        for i, result in enumerate(semantic_results):
            doc_id = result['movie']['id']
            if doc_id in hybrid_search_results:
                hybrid_search_results[doc_id]['semantic_score'] = i + 1
            else:
                hybrid_search_results[doc_id] = {
                        'movie': result['movie'],
                        'semantic_score': i + 1,
                        'hybrid_score': 0.0,
                        'keyword_score': 0.0
                    }

        results: list[HybridSearchResult] = []
        for hybrid_result in hybrid_search_results.values():
            keyword_rank: float = hybrid_result['keyword_score']
            semantic_rank: float = hybrid_result['semantic_score']
            keyword_rrf = rrf_score(keyword_rank, k) if not keyword_rank == 0 else 0
            semantic_rrf = rrf_score(semantic_rank, k) if not semantic_rank == 0 else 0
            hybrid_result['hybrid_score'] = keyword_rrf + semantic_rrf
            results.append(format_hybrid_search_result(
                doc_id=hybrid_result['movie']['id'],
                title=hybrid_result['movie']['title'],
                document=hybrid_result['movie']['description'],
                semantic_score=semantic_rank,
                bm25_score=keyword_rank,
                hybrid_score=hybrid_result['hybrid_score']
            ))
        sorted_results = sorted(results, key=lambda x: x['hybrid_score'], reverse=True)[:limit]
        return sorted_results

def min_max_normalization(scores: list[float]) -> list[float]:
    if len(scores) == 0:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0 for _ in scores]
    return [(x - min_score)/(max_score - min_score) for x in scores]

def weighted_search_command(query: str, alpha:float=DEFAULT_ALPHA, limit: int=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['movie']['title']}")
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"BM25: {result['keyword_score']} Semantic: {result['semantic_score']}")
        print(f"{result['movie']['description']}")

def rrf_search_command(query: str, k: int, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    results = hybrid_search.rrf_search(query, k, limit)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['movie']['title']}")
        print(f"Hybrid Score: {result['hybrid_score']}")
        print(f"BM25: {result['keyword_score']} Semantic: {result['semantic_score']}")
        print(f"{result['movie']['description']}")

def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = 0.5
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(
    rank: int, k: float = 60
) -> float:
    return 1 / (k + rank)