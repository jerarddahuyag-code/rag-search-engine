import string

from .utils import DEFAULT_SEARCH_LIMIT,load_movies, load_stop_words
from nltk.stem import PorterStemmer
from .inverted_index import InvertedIndex

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