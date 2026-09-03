from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore


class Retriever:

    def __init__(self, db_path):
        self.model = EmbeddingModel()

        self.db = FAISSVectorStore(db_path)
        self.db.load()

    def retrieve(self, query, k=3):
        vector = self.model.encode([query])
        return self.db.search(vector, k)
    
if __name__ == "__main__":
    from config import VECTOR_DB

    retriever = Retriever(VECTOR_DB)

    query = "शंखनादः कस्य भृत्यः आसीत्?"

    results = retriever.retrieve(query)

    for i, result in enumerate(results):
        print(f"\nResult {i + 1}")
        print("Page:", result["page"])
        print("Score:", result["score"])
        print(result["text"][:300])