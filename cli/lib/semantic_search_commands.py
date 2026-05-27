from collections import defaultdict
import os

from lib.utils import (
    DEFAULT_SEARCH_LIMIT,
    Movie,
    CACHE_DIRECTORY,
    SearchResult,
    load_movies
    )
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents: list[Movie] = None
        self.document_map: dict[int, Movie] = defaultdict(Movie)
        self.embeddings_path = os.path.join(CACHE_DIRECTORY, "movie_embeddings.npy")

    def build_embeddings(self, documents: list[Movie]):
        os.makedirs("cache", exist_ok=True)
        self.documents = documents
        doc_list: list[str] = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        self.embeddings  = self.model.encode(doc_list, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        doc_list: list[str] = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        embeddings = self.model.encode([text])
        return embeddings[0]
    
    def search(self, query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_orcreate_embedding`")
        embedded_query = self.generate_embedding(query)
        similarities:list[tuple] = []
        index = 0
        for doc_id in self.document_map:
            embedded_doc = self.embeddings[index]
            similarity_score = cosine_similarity(embedded_query, embedded_doc)
            similarities.append((similarity_score, self.document_map[doc_id]))
            index += 1
        most_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:limit]
        results: list[SearchResult] = []
        for result in most_similar:
            results.append({
                'movie': result[1],
                'score': result[0]
            })
        return results

def verify_model():
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def verify_embeddings():
    ss = SemanticSearch()
    docs = load_movies()
    embeddings = ss.load_or_create_embeddings(docs)
    print(f"Number of docs:   {len(docs)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT):
    ss = SemanticSearch()
    docs = load_movies()
    ss.load_or_create_embeddings(docs)
    results = ss.search(query, limit)
    for i, result in enumerate(results):
        movie = result['movie']
        print(f"{i + 1}. {movie['title']} ({result['score']})\n{movie['description']}")





