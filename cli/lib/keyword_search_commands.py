from collections import Counter, defaultdict
import math
import string
import os
import pickle

from .utils import DEFAULT_SEARCH_LIMIT, CACHE_DIRECTORY, STOP_WORDS_PATH, load_movies
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
    
    def get_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        return math.log(float(total_doc_count + 1) / float(term_match_doc_count + 1))
    
    def get_tfidf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[term])
        term_missing_doc_count = total_doc_count - term_match_doc_count
        return math.log((term_missing_doc_count + 0.5)/(term_match_doc_count + 0.5) + 1)
   
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

def idf_command(term: str) -> float:
    validated_term = validate_token_size(term)
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(validated_term)

def tf_command(doc_id: int, term: str) -> int:
    validated_term = validate_token_size(term)
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, validated_term)

def tfidf_command(doc_id: int, term: str) -> float:
    validated_term = validate_token_size(term)
    idx = InvertedIndex()
    idx.load()
    return idx.get_tfidf(doc_id, validated_term)

def bm25_idf_command(term: str) -> float:
    validated_term = validate_token_size(term)
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(validated_term)

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
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
    no_punc = remove_punctuation(unprocessed.lower())
    tokens = tokenize(no_punc)
    filtered = remove_stop_words(tokens)
    stemmed = stemTokens(filtered)
    return stemmed

def remove_punctuation(unprocessed:str) -> str:
    return unprocessed.translate(str.maketrans("", "", string.punctuation))

def tokenize(unprocessed:str) -> list[str]:
    split = unprocessed.split()
    words = []
    for word in split:
        if word != " ":
            words.append(word)
    return words 

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        data = [remove_punctuation(word.lower()) for word in f.read().splitlines()]
    return data

STOP_WORDS = load_stop_words()

def remove_stop_words(tokens: list[str]) -> list[str]:
    filtered = []
    for word in tokens:
        if word not in STOP_WORDS:
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
        return result[0]