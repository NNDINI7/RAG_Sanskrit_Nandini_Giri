import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import LLM_MODEL


class Generator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            torch_dtype=torch.float32
        ).to("cpu")

    def generate(self, question, context):

        prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}

Answer:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        answer = output[0][inputs["input_ids"].shape[1]:]

        return self.tokenizer.decode(
            answer,
            skip_special_tokens=True
        ).strip()

if __name__ == "__main__":
    generator = Generator()

    context = """
    शंखनादः गोवर्धनदासस्य भृत्यः आसीत्।
    """

    question = "शंखनादः कस्य भृत्यः आसीत्?"

    answer = generator.generate(
        question,
        context
    )

    print("Answer:")
    print(answer)