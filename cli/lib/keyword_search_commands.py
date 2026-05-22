from collections import Counter, defaultdict
import string
import os
import pickle

from .utils import DEFAULT_SEARCH_LIMIT, CACHE_DIRECTORY, load_movies, load_stop_words
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIRECTORY, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIRECTORY, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIRECTORY, "term_frequencies.pkl")
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = preprocess(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]
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
        with open(self.term_frequencies_path, "wb") as file:
            pickle.dump(self.term_frequencies, file)
    def load(self) -> None:
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path):
            raise OSError("Index file does not exist")
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, "rb") as file:
            self.term_frequencies = pickle.load(file)

def tf(doc_id: int, term: str) -> int:
    validate_token_size(term)
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term)

def build() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try:
        idx = InvertedIndex()
        idx.load()
        query_tokens = preprocess(query)
        seen, results = set(), []
        for query_token in query_tokens:
            matching_doc_ids = idx.get_documents(query_token)
            for doc_id in matching_doc_ids:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc = idx.docmap[doc_id]
                results.append(doc)
                if len(results) >= limit:
                    return results
    except OSError as e:
        print(e)

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

def validate_token_size(term: str):
    result = preprocess(term)
    if len(result) > 1:
        raise ValueError("term is not a single token") 
    else:
        return result