"""
graph.py — Self-corrective agentic RAG built on a LangGraph StateGraph.

The flow is a small loop that can *correct itself*:

    START -> retrieve -> grade_documents -> (decide) -> generate -> END
                              ^                            |
                              |                            |
                            rewrite <--- (no good docs) ---+

If the graded documents are not relevant, we rewrite the question and retrieve
again. A hard cap on rewrites (< 2) guarantees the loop terminates.

Everything is deliberately flat and heavily commented — no hidden abstraction —
because this is meant to be read and defended out loud.
"""

import os
from functools import lru_cache
from typing import List, TypedDict

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

# Load GOOGLE_API_KEY and DOC_PATH from a local .env if present.
load_dotenv()


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    """The single object passed between nodes; each node reads and updates it."""

    question: str       # the (possibly rewritten) query we retrieve/answer with
    documents: List     # docs kept after relevance grading
    generation: str     # the final grounded answer
    rewrites: int       # how many times we've rewritten — the loop guard


# ---------------------------------------------------------------------------
# Lazily-built, cached singletons (LLM + vector store)
# ---------------------------------------------------------------------------
# These are built on first use and cached, NOT at import time. That keeps
# `from graph import app` cheap and side-effect free (it just compiles the
# graph), while still building the FAISS index only ONCE for the process.

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Return the shared chat model (temperature 0 for deterministic grading)."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )
    # To swap to OpenAI instead of Gemini:
    #   from langchain_openai import ChatOpenAI
    #   return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    """Ingest DOC_PATH once and return an in-memory FAISS index over its chunks."""
    doc_path = os.environ["DOC_PATH"]

    # Loader depends on file type: PDFs via PyPDFLoader, everything else as text.
    loader = PyPDFLoader(doc_path) if doc_path.lower().endswith(".pdf") else TextLoader(doc_path)
    docs = loader.load()

    # Split into overlapping chunks so retrieval returns focused, self-contained text.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Local sentence-transformers embeddings (no API key needed for embedding).
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Build the index from the chunks — done once thanks to lru_cache.
    return FAISS.from_documents(chunks, embeddings)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve(state: GraphState) -> dict:
    """Single responsibility: fetch the top-k documents for the current question."""
    docs = get_vectorstore().similarity_search(state["question"], k=4)
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    """Single responsibility: keep only documents the LLM judges relevant (binary yes/no)."""
    llm = get_llm()
    kept = []
    for doc in state["documents"]:
        prompt = (
            "You are grading whether a retrieved document is relevant to a user question.\n"
            "Answer with a single word: 'yes' or 'no'.\n\n"
            f"Question: {state['question']}\n\n"
            f"Document: {doc.page_content}"
        )
        grade = llm.invoke(prompt).content.strip().lower()
        if "yes" in grade:
            kept.append(doc)
    return {"documents": kept}


def generate(state: GraphState) -> dict:
    """Single responsibility: write an answer grounded ONLY in the graded documents."""
    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in state["documents"])
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain enough information, say you couldn't find "
        "enough context to answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}"
    )
    return {"generation": llm.invoke(prompt).content}


def rewrite(state: GraphState) -> dict:
    """Single responsibility: rephrase the question into a better retrieval query."""
    llm = get_llm()
    prompt = (
        "Rewrite the user's question into a clearer, more specific query that will "
        "retrieve better documents. Return only the rewritten question.\n\n"
        f"Question: {state['question']}"
    )
    better = llm.invoke(prompt).content.strip()
    # Increment the counter so decide_to_generate can enforce the loop guard.
    return {"question": better, "rewrites": state["rewrites"] + 1}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def decide_to_generate(state: GraphState) -> str:
    """Route after grading: generate if we have relevant docs, else rewrite (bounded)."""
    if state["documents"]:
        # We have at least one relevant document — go answer.
        return "generate"
    if state["rewrites"] < 2:
        # No relevant docs yet, but we're still under the retry budget — try again.
        # This `< 2` guard is what PREVENTS AN INFINITE LOOP of rewrite->retrieve.
        return "rewrite"
    # Out of retries: generate anyway so the user gets a best-effort / "not found" answer.
    return "generate"


# ---------------------------------------------------------------------------
# Build & compile the graph
# ---------------------------------------------------------------------------
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"generate": "generate", "rewrite": "rewrite"},
)
workflow.add_edge("rewrite", "retrieve")  # rewritten question loops back to retrieval
workflow.add_edge("generate", END)

# `app` is the compiled, runnable graph imported by main.py.
app = workflow.compile()
