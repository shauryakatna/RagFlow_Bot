"""
main.py — FastAPI wrapper around the self-corrective RAG graph.

POST /query  {"question": "..."}  ->  {"answer": "...", "path": ["retrieve", ...]}

We stream the graph instead of calling .invoke() so we can record which nodes
actually ran. The `path` proves the self-correction happened (e.g. a
retrieve -> grade_documents -> rewrite -> retrieve loop shows up in the list).
"""

from fastapi import FastAPI
from pydantic import BaseModel

from graph import app

api = FastAPI(title="Self-Corrective Agentic RAG")


class Query(BaseModel):
    """Request body for /query."""

    question: str


@api.post("/query")
def query(body: Query) -> dict:
    """Run the graph for one question and return the answer plus the node path."""
    # rewrites starts at 0 so the loop guard (rewrites < 2) is well-defined.
    inputs = {"question": body.question, "documents": [], "generation": "", "rewrites": 0}

    path = []          # node names in execution order
    generation = ""    # filled by the generate node

    # stream() yields one {node_name: state_update} dict per node execution.
    for step in app.stream(inputs):
        for node_name, update in step.items():
            path.append(node_name)
            if "generation" in update and update["generation"]:
                generation = update["generation"]

    return {"answer": generation, "path": path}
