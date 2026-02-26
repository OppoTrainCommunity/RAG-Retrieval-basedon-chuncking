from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_core import init_rag, ask_question, index_pdfs

app = FastAPI(title="Resume RAG API")

class AskRequest(BaseModel):
    question: str

@app.on_event("startup")
def startup():
    init_rag()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskRequest):
    try:
        answer = ask_question(req.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
def index():
    try:
        result = index_pdfs(reset=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))