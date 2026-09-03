from pypdf import PdfReader


def load_pdf(path):
    reader = PdfReader(path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            documents.append({
                "page": i + 1,
                "text": text
            })

    return documents

if __name__ == "__main__":
    from config import PDF_PATH

    docs = load_pdf(PDF_PATH)

    print("Pages loaded:", len(docs))
    print(docs[0]["text"][:500])