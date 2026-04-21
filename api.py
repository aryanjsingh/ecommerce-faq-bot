from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid

from bot import get_llm, setup_kb, create_graph

app = FastAPI(title="E-Commerce FAQ Bot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm = None
_embedder = None
_collection = None
_agent = None


@app.on_event("startup")
async def startup():
    global _llm, _embedder, _collection, _agent
    _llm = get_llm()
    _embedder, _collection = setup_kb()
    _agent = create_graph(_llm, _embedder, _collection)


class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    thread_id: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    result = _agent.invoke({"question": req.question}, config=config)
    return ChatResponse(
        answer=result.get("answer", "Sorry, I couldn't find an answer."),
        sources=result.get("sources", []),
        thread_id=tid,
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "qwen2.5:3b", "project": "ecommerce-faq-bot"}
