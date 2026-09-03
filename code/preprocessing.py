import re


def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text, size=500, overlap=100):
    chunks = []

    for start in range(0, len(text), size - overlap):
        chunk = text[start:start + size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    from config import PDF_PATH
    from document_loader import load_pdf

    docs = load_pdf(PDF_PATH)

    chunks = []

    for doc in docs:
        text = clean_text(doc["text"])
        chunks.extend(create_chunks(text))

    print("Pages:", len(docs))
    print("Chunks:", len(chunks))
    print("\nFirst chunk:")
    print(chunks[0])