from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


class EmbeddingModel:

    def __init__(self):
        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

    def encode(self, texts):
        return self.model.encode(
            texts,
            normalize_embeddings=True
        )

if __name__ == "__main__":
    model = EmbeddingModel()

    text = ["शंखनादः गोवर्धनदासस्य भृत्यः आसीत्।"]

    vector = model.encode(text)

    print("Embedding shape:", vector.shape)