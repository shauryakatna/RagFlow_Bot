# Self-Corrective Agentic RAG (LangGraph)

A minimal Retrieval-Augmented Generation system that **grades its own retrieval
and retries** when the documents aren't good enough — built as a
[LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` and served
over FastAPI.

The code is intentionally flat and heavily commented (two files: `graph.py`,
`main.py`) so the whole flow can be read top to bottom.

## The 4-node graph

State (`TypedDict`): `question`, `documents`, `generation`, `rewrites`.

```
START -> retrieve -> grade_documents -> (decide) -> generate -> END
                          ^                            |
                          |                            |
                        rewrite <--- no good docs -----+
```

- **retrieve** — embed the question and run FAISS `similarity_search(k=4)`.
- **grade_documents** — ask the LLM a binary `yes`/`no` on each doc; keep the `yes` ones.
- **generate** — answer grounded **only** in the graded documents.
- **rewrite** — rephrase the question into a better query and increment `rewrites`.

**Conditional edge (`decide_to_generate`)**, after grading:
1. at least one relevant doc → `generate`
2. else if `rewrites < 2` → `rewrite` (loop back to retrieve)
3. else → `generate` (best-effort / "couldn't find enough context")

The `rewrites < 2` guard is what stops the rewrite→retrieve loop from running
forever.

## Stack

- **LLM:** `ChatGoogleGenerativeAI` — `gemini-2.0-flash`, temperature 0.
  (Swap comment in `graph.py` shows how to use `ChatOpenAI(model="gpt-4o-mini")`.)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local, no API key).
- **Vector store:** FAISS, built once at first use from `DOC_PATH`.
- **Ingestion:** `PyPDFLoader` for `.pdf` else `TextLoader`;
  `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GOOGLE_API_KEY (DOC_PATH already points at sample.pdf)
uvicorn main:api --reload --port 8000
```

The repo ships a placeholder `sample.pdf` and `.env.example` sets
`DOC_PATH=sample.pdf`, so it runs out of the box — just add your API key.
Swap in your own document by pointing `DOC_PATH` at it.

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the document about?"}'
```

Response — note `path` shows which nodes actually ran, so you can see the
self-correction loop when it triggers:

```json
{"answer": "...", "path": ["retrieve", "grade_documents", "generate"]}
```

## Run with Docker

```bash
docker build -t agentic-rag .
docker run --rm -p 8000:8000 --env-file .env agentic-rag
```

The bundled `sample.pdf` is copied into the image, so with the default
`DOC_PATH=sample.pdf` it works immediately. To use your own document instead,
mount it and point `DOC_PATH` at it, e.g. `-v /host/doc.pdf:/app/doc.pdf` and
set `DOC_PATH=/app/doc.pdf`.
