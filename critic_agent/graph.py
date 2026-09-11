from langgraph.graph import StateGraph, START, END
from critic_agent.node import CriticState, correctness_node, faithfulness_node, retrieval_relevance_node, pass_fail_node

builder = StateGraph(CriticState)

builder.add_node("correctness", correctness_node)
builder.add_node("faithfulness", faithfulness_node)
builder.add_node("retrieval_relevance", retrieval_relevance_node)
builder.add_node("pass_fail", pass_fail_node)

builder.add_edge(START, "correctness")
builder.add_edge("correctness", "faithfulness")
builder.add_edge("faithfulness", "retrieval_relevance")
builder.add_edge("retrieval_relevance", "pass_fail")
builder.add_edge("pass_fail", END)

critic_graph = builder.compile()
