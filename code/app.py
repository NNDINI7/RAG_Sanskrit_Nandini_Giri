import streamlit as st

from rag_pipeline import RAGPipeline


st.title("Sanskrit RAG System")

@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()

question = st.text_input(
    "Ask a question in Sanskrit or transliteration:"
)

if st.button("Ask") and question:

    answer, results = rag.ask(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for result in results:
        st.write(
            f"Page {result['page']} | "
            f"Score: {result['score']:.3f}"
        )