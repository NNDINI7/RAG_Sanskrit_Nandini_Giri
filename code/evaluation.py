import time
from sentence_transformers import SentenceTransformer

from rag_pipeline import RAGPipeline


TEST_DATA = [
    {
        "question": "शंखनादः कस्य भृत्यः आसीत्?",
        "answer": "शंखनादः गोवर्धनदासस्य भृत्यः आसीत्।"
    },

    {
        "question": "शंखनादः कुत्र गच्छति?",
        "answer": "शंखनादः आपणं गच्छति।"
    },

    {
        "question": "गोवर्धनदासः शंखनादं किम् आदिशत्?",
        "answer": "गोवर्धनदासः शंखनादं आपणं गत्वा शर्करां आनयितुम् आदिशत्।"
    }
]


def similarity(model, text1, text2):
    vectors = model.encode([text1, text2])
    return float(vectors[0] @ vectors[1])


def evaluate():
    rag = RAGPipeline()

    evaluator = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

    total_score = 0

    for i, item in enumerate(TEST_DATA):

        start = time.time()

        answer, results = rag.ask(item["question"])

        total_time = time.time() - start

        score = similarity(
            evaluator,
            answer,
            item["answer"]
        )

        total_score += score

        print("\n-------------------------")
        print("Question:", item["question"])
        print("Expected:", item["answer"])
        print("Generated:", answer)
        print("Similarity:", round(score, 3))
        print("Latency:", round(total_time, 2), "seconds")

        print("Retrieved Pages:",
              [r["page"] for r in results])

    average = total_score / len(TEST_DATA)

    print("\n=========================")
    print("Average Semantic Score:", round(average, 3))
    print("=========================")


if __name__ == "__main__":
    evaluate()