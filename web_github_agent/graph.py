from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

from web_github_agent.node import ResearchState,research_node

from web_github_agent.tools import research_tools

builder = StateGraph(ResearchState)

builder.add_node("research",research_node)
builder.add_node("tools",ToolNode(research_tools))

builder.add_edge(START,"research")
builder.add_conditional_edges("research",tools_condition)
builder.add_edge("tools","research")

research_graph = builder.compile(name="research_agent")