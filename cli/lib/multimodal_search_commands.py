
from PIL import Image
from lib.semantic_search_commands import cosine_similarity
from lib.utils import Movie, SearchResult, load_movies
from sentence_transformers import SentenceTransformer

class MultimodalSearch():
    def __init__(self, documents: list[Movie], model_name="clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.document_map = {mov["id"]: mov for mov in documents}
        self.texts = [f"{mov["title"]} {mov['description']}" for mov in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def search_with_image(self, path: str):
        embedded_image = self.embed_image(path)
        similarities:list[tuple] = []
        index = 0
        for doc_id in self.document_map:
            embedded_doc = self.text_embeddings[index]
            similarity_score = cosine_similarity(embedded_image, embedded_doc)
            similarities.append((similarity_score, self.document_map[doc_id]))
            index += 1
        most_similar = sorted(similarities, key=lambda x: x[0], reverse=True)[:5]
        results: list[SearchResult] = []
        for result in most_similar:
            results.append({
                'movie': result[1],
                'score': result[0]
            })
        return results

    def embed_image(self, path: str):
        img = Image.open(path)
        embedding = self.model.encode([img])
        return embedding[0]
    
def verify_image_embedding_command(image_path: str):
    movies = load_movies()
    mms = MultimodalSearch(movies)
    embedding = mms.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path: str):
    movies = load_movies()
    mms = MultimodalSearch(movies)
    results = mms.search_with_image(image_path)
    for i, result in enumerate(results):
        print(f"{i + 1}. {result['movie']['title']} (similarity: {result['score']:.1f}) \n  {result['movie']['description']}")