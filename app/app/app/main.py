from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from app.rag import PakistanLawRAG

app = FastAPI(
    title="Pakistan Law Assistant API",
    description="RAG-backed API serving Pakistani legal queries with source citations.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = PakistanLawRAG()

class QueryRequest(BaseModel):
    question: str

class SourceDetail(BaseModel):
    document: str
    page: str | int
    content_snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDetail]

@app.get("/")
def read_root():
    return {"message": "Pakistan Law RAG Assistant API is live."}

@app.post("/ingest")
def ingest_documents():
    """Trigger document ingestion from the data/ folder."""
    try:
        count = rag.ingest_documents()
        return {"status": "success", "message": f"Successfully ingested {count} legal text chunks into vector DB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """Query the legal assistant with a specific question."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question string cannot be empty.")
    try:
        result = rag.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
