from collections import defaultdict
import json
import os
import re

from lib.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    ChunkMetaData,
    ChunkScore,
    Movie,
    CACHE_DIRECTORY,
    SearchResult,
    format_search_result,
    load_movies
    )
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self, model_name:str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIRECTORY, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIRECTORY, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        os.makedirs("cache", exist_ok=True)
        self.documents = documents
        chunks: list[str] = []
        chunks_metadata: list[ChunkMetaData] = []
        for i, doc in enumerate(documents):
            self.document_map[doc['id']] = doc
            if len(doc['description']) == 0:
                continue
            doc_chunks = semantic_chunking(doc['description'], 4, 1)
            for j, doc_chunk in enumerate(doc_chunks):
                chunks.append(doc_chunk)
                chunks_metadata.append({
                    'movie_idx': i,
                    'chunk_idx': j,
                    'total_chunks': len(doc_chunks)
                })
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(chunks)}, f, indent=2)
        return self.chunk_embeddings
    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json['chunks']
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
    def search_chunks(self, query: str, limit: int=DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        embedded_query = self.generate_embedding(query)
        chunk_scores: list[ChunkScore]= []
        for i, chunk in enumerate(self.chunk_embeddings):
            score = cosine_similarity(embedded_query, chunk)
            chunk_scores.append({
                'movie_idx': self.chunk_metadata[i]['movie_idx'],
                'chunk_idx': self.chunk_metadata[i]['chunk_idx'],
                'score': score
            })
        movie_scores: dict[int, float] = {}
        for chunk_score in chunk_scores:
            if movie_scores.get(chunk_score['movie_idx']):
                movie_scores[chunk_score['movie_idx']] = max(movie_scores[chunk_score['movie_idx']], chunk_score['score'])
            else:
                movie_scores[chunk_score['movie_idx']] = chunk_score['score']
        most_relevant = dict(sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)[:limit])
        results: list[SearchResult] = []
        for mov_idx, mov_score in most_relevant.items():
            movie = self.documents[mov_idx]
            results.append(format_search_result(movie['id'], movie['title'], movie['description'], mov_score))
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

def fixed_chunking(text: str, size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[list[str]]:
    splits = text.split()
    chunks: list[list[str]] = []
    chunk: list[str] = []
    for word in splits:
        if len(chunk) == size and not(len(chunk) == 0):
            chunks.append(chunk)
            chunk = chunk[max(0, len(chunk) - overlap):]
        chunk.append(word)
    if not(len(chunk) == 0):
        chunks.append(chunk)
    return chunks

def semantic_chunking(text: str, size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP) -> list[str]:
    splits = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    chunk: list[str] = []
    for sen in splits:
        if len(chunk) == size and not(len(chunk) == 0):
            chunks.append(" ".join(chunk))
            chunk = chunk[max(0, len(chunk) - overlap):]
        stripped_sentence = sen.strip()
        if not len(stripped_sentence) == 0:
            chunk.append(stripped_sentence)
    if not(len(chunk) == 0):
        chunks.append(chunk)
    return chunks

def chunk_command(text: str, size: int=DEFAULT_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP):
    chunks = fixed_chunking(text, size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, c in enumerate(chunks):
        print(f"{i + 1}. {" ".join(c)}")

def semantic_chunk_command(text: str, size: int=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int=DEFAULT_CHUNK_OVERLAP):
    chunks = semantic_chunking(text, size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, c in enumerate(chunks):
        print(f"{i + 1}. {" ".join(c)}")

def embed_semantic_chunk_command():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int=DEFAULT_SEARCH_LIMIT):
    css = ChunkedSemanticSearch()
    movies = load_movies()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, result in enumerate(results):
        print(f"\n{i + 1}. {result['movie']['title']} (score: {result['score']:.4f})")
        print(f"   {result['movie']['description']}...")