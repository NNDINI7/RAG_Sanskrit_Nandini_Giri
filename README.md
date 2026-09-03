Project Summary

Sanskrit RAG System is a lightweight, CPU-based Retrieval-Augmented Generation system that answers questions from Sanskrit documents using semantic retrieval and a local LLM.

Objective
Extract and preprocess Sanskrit PDF text.
Create chunks and embeddings.
Store embeddings in FAISS.
Retrieve relevant chunks for a query.
Generate answers using a local LLM.
Evaluate similarity and latency.

Architecture

PDF → Preprocessing → Chunking → Embeddings → FAISS → Top-3 Retrieval → Qwen 0.5B → Answer + Sources

Tech Stack

Python – Development
pypdf – PDF extraction
Sentence Transformers + MiniLM – 384-D embeddings
FAISS – Vector similarity search
PyTorch + Transformers – LLM inference
Qwen 2.5 0.5B – Answer generation
Streamlit – UI

Algorithms

Semantic Embeddings – Converts text into vectors.
Inner Product/Cosine-style similarity – Finds relevant chunks.
Top-K Retrieval – Retrieves the top 3 chunks.
Chunking – 500 characters with 100-character overlap.

Evaluation

Similarity scores: 2.600, 2.248, 1.855.
These are not accuracy percentages.
Average latency: ~38.58 seconds.
Faithfulness was low in some cases because the 0.5B model sometimes misunderstood Sanskrit context.

Limitations

PDF extraction artifacts.
Small LLM has limited Sanskrit understanding.
High CPU latency.
Small evaluation dataset.
Occasional unfaithful answers.

Future Scope

Better Sanskrit OCR, stronger embeddings/LLMs, hybrid BM25 + vector retrieval, reranking, quantization, better prompts, larger evaluation dataset, multiple documents, Docker deployment, and multilingual support.
