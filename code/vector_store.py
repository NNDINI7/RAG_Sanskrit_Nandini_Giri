import faiss
import pickle
import numpy as np


class FAISSVectorStore:

    def __init__(self, path):
        self.path = path
        self.index = None
        self.documents = []

    def build(self, documents, embeddings):
        embeddings = np.array(
            embeddings,
            dtype="float32"
        )

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)
        self.documents = documents

        self.path.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(self.path / "index.faiss")
        )

        with open(self.path / "documents.pkl", "wb") as f:
            pickle.dump(documents, f)

    def load(self):
        self.index = faiss.read_index(
            str(self.path / "index.faiss")
        )

        with open(self.path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

    def search(self, embedding, k=3):
        scores, ids = self.index.search(
            np.array(embedding, dtype="float32"),
            k
        )

        return [
            {
                "page": self.documents[i]["page"],
                "text": self.documents[i]["text"],
                "score": float(score)
            }
            for score, i in zip(scores[0], ids[0])
            if i != -1
        ]