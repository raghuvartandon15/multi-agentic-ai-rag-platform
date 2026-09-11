from typing import TypedDict, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AppState(TypedDict, total=False):
    # Conversation
    messages: Annotated[list[BaseMessage], add_messages]
    # Input handling
    input_type: str
    raw_input: object
    extracted_content: str
    # Current user request
    query: str
    # Which specialist most recently produced a result
    last_agent: str
    # RAG output
    rag_response: str
    retrieved_docs: list
    # Research output
    research_response: str
    # Debug output
    debug_response: str
    # Critic output
    correctness: int
    faithfulness: int
    retrieval_relevance: int
    passed: bool
    critic_skipped: bool
    # Final answer
    final_response: str
    # retry_count
    retry_count: int
    input_blocked: bool
    # query enhancement
    should_decompose: bool
    sub_queries: list[str]
    processed_query: str
