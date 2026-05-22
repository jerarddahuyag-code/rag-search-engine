import os
import pickle

from .utils import load_movies, PROJECT_ROOT
from .keyword_search_commands import tokenize

INDEX_PATH = os.path.join(PROJECT_ROOT, "cache", "index.pkl")
DOCMAP_PATH = os.path.join(PROJECT_ROOT, "cache", "docmap.pkl")

class InvertedIndex:
    def __init__(self, index: dict = {}, docmap: dict = {}) -> None:
        self.index = index
        self.docmap = docmap
    def __add_document(self, doc_id, text) -> None:
        tokens = tokenize(text)
        for word in tokens:
            if word not in self.index:
                self.index[word] = []
            if doc_id not in self.index[word]:
                self.index[word].append(doc_id)
    def get_documents(self, term) -> list[int]:
        return sorted(self.index[term])
    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie
    def save(self) -> None:
        os.makedirs("cache", exist_ok=True)
        with open(INDEX_PATH, "wb") as file:
            pickle.dump(self.index, file)
        with open(DOCMAP_PATH, "wb") as file:
            pickle.dump(self.docmap, file)