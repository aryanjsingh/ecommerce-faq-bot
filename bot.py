import os
import datetime
import re
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def setup_kb():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="ecommerce_faq")

    documents = [
        {"id": "doc_001", "topic": "Returns Policy", "text": "Customers can return products within 30 days of delivery. Items must be unused and in original packaging."},
        {"id": "doc_002", "topic": "Shipping Times", "text": "Standard shipping: 5-7 days. Expedited: 2-3 days. Overnight available before 2 PM EST."},
        {"id": "doc_003", "topic": "Payment Methods", "text": "We accept Visa, MasterCard, PayPal, and Apple Pay."},
    ]
    if collection.count() == 0:
        collection.add(
            documents=[d["text"] for d in documents],
            metadatas=[{"topic": d["topic"]} for d in documents],
            ids=[d["id"] for d in documents]
        )
    return embedder, collection

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class CapstoneState(TypedDict):
    question: str
    messages: List[BaseMessage]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    faithfulness: float
    eval_retries: int
    user_name: str

def create_graph(llm, embedder, collection):
    g = StateGraph(CapstoneState)
    # nodes to be implemented
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)
