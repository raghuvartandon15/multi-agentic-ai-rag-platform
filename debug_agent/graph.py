from typing import TypedDict, Annotated
from langgraph.graph import (StateGraph,START,)
from langgraph.graph.message import add_messages
from langgraph.prebuilt import (ToolNode,tools_condition)
from debug_agent.agent import debug_agent
from debug_agent.tools import debug_tools


class DebugState(TypedDict):
    messages: Annotated[list,add_messages]

graph = StateGraph(DebugState)

graph.add_node("debug",debug_agent)
graph.add_node("tools",ToolNode(debug_tools))

graph.add_edge(START,"debug")
graph.add_conditional_edges("debug",tools_condition)
graph.add_edge("tools","debug")

debug_graph = graph.compile(name="debug_agent")