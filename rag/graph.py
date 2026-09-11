from typing import TypedDict, Annotated
from langgraph.graph import StateGraph,START,END,add_messages
from rag.node import rag_node

class RAGState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    rag_response: str
    retrieved_docs: list
    last_agent: str


builder = StateGraph(RAGState)

builder.add_node("rag",rag_node)

builder.add_edge(START,"rag")
builder.add_edge("rag",END)

rag_graph = builder.compile(name="rag_agent")