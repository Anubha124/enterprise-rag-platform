from fastapi import FastAPI
from pydantic import BaseModel

from backend.src.retrieval.rag_qa import RAGQA


app = FastAPI(
    title="Enterprise RAG API"
)


rag = RAGQA()


class QuestionRequest(BaseModel):

    question: str


@app.get("/")
def root():

    return {
        "message": "Enterprise RAG API Running"
    }


@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    answer, sources = rag.ask(
        "data/resume.pdf",
        request.question
    )

    return {
        "answer": answer,
        "sources": sources
    }