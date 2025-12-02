# rag_crew.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama  # optional local LLM if available

load_dotenv()

def create_vector_store(file_path):
    """Loads a document, splits it, and creates a FAISS vector store with free local embeddings."""
    # Load the document
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Use HuggingFace embeddings (free)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build FAISS index
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store


def get_answer(query, file_path):
    """Retrieve relevant chunks and answer the question using a local model or template."""
    vector_store = create_vector_store(file_path)
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Simple offline text generation
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return f"Based on the document, here's what I found:\n\n{context[:700]}..."
