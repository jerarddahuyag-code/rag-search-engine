from collections import defaultdict
import string
import os
import pickle

from .utils import DEFAULT_SEARCH_LIMIT, CACHE_DIRECTORY, load_movies, load_stop_words
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict()
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIRECTORY, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIRECTORY, "docmap.pkl")
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
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)
        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

def build() -> None:
        index = InvertedIndex()
        index.build()
        index.save()
        docs = index.get_documents("Merida")
        print(f"First document for token 'merida' = {docs[0]}")

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    matchedMovies = []
    movies = load_movies()
    for movie in movies:
        if match(preprocess(query), preprocess(movie['title'])):
            matchedMovies.append(movie)
        if len(matchedMovies) >= limit:
            break
    return matchedMovies

def match(queryTokens: list[str], movieTokens: list[str]) -> bool:
    for token in queryTokens:
        for mov_token in movieTokens:
            if token in mov_token:
                return True
    return False

def preprocess(unprocessed:str) -> list[str]:
    no_punc = remove_punctuation(unprocessed).lower()
    tokens = tokenize(no_punc)
    filtered = remove_stop_words(tokens)
    stemmed = stemTokens(filtered)
    return stemmed

def remove_punctuation(unprocessed:str) -> str:
    mapping = {}
    for punctuation in string.punctuation:
        mapping[punctuation] = None
    processed = unprocessed.translate(str.maketrans(mapping))
    return processed

def tokenize(unprocessed:str) -> list[str]:
    split = unprocessed.split(" ")
    words = []
    for word in split:
        if word != " ":
            words.append(word)
    return words

def remove_stop_words(tokens: list[str]) -> list[str]:
    stop_words = load_stop_words()
    filtered = []
    for word in tokens:
        if word not in stop_words:
            filtered.append(word)
    return filtered

def stemTokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmedTokens = []
    for token in tokens:
        stemmedTokens.append(stemmer.stem(token))
    return stemmedTokens