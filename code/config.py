from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "sanskrit_documents.pdf"
VECTOR_DB = BASE_DIR / "vector_db" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K = 3