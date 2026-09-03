from config import PDF_PATH, VECTOR_DB
from document_loader import load_pdf
from preprocessing import clean_text, create_chunks
from embeddings import EmbeddingModel
from vector_store import FAISSVectorStore


documents = load_pdf(PDF_PATH)

chunks = []

for doc in documents:
    text = clean_text(doc["text"])

    for chunk in create_chunks(text):
        chunks.append({
            "page": doc["page"],
            "text": chunk
        })

print("Chunks:", len(chunks))

model = EmbeddingModel()

embeddings = model.encode(
    [x["text"] for x in chunks]
)

db = FAISSVectorStore(VECTOR_DB)
db.build(chunks, embeddings)

print("FAISS index created successfully!")