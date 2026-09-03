from pathlib import Path

from config import VECTOR_DB, TOP_K
from retriever import Retriever
from generator import Generator


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever(VECTOR_DB)
        self.generator = Generator()

    def ask(self, question):

        results = self.retriever.retrieve(
            question,
            TOP_K
        )

        context = "\n".join(
            result["text"]
            for result in results
        )

        answer = self.generator.generate(
            question,
            context
        )

        return answer, results
if __name__ == "__main__":

    rag = RAGPipeline()

    question = "शंखनादः कस्य भृत्यः आसीत्?"

    answer, results = rag.ask(question)

    print("\n===== ANSWER =====")
    print(answer)

    print("\n===== SOURCES =====")

    for result in results:
        print("Page:", result["page"])