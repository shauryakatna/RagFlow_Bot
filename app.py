# app.py
import streamlit as st
from rag_crew import get_answer

st.title("📚 Gemini RAG Chatbot")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
query = st.text_input("Ask a question about your document:")

if uploaded_file and query:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    answer = get_answer(query, uploaded_file.name)
    st.subheader("Answer:")
    st.write(answer)
