from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage
from langgraph.graph import add_messages

from web_github_agent.agent import llm_with_tools


class ResearchState(TypedDict):
    messages: Annotated[list,add_messages]


SYSTEM_PROMPT = """
You are a Web and GitHub research expert.

Your job is to research external information that
can help answer the user's question.

Use the web search tool when:

- the question requires current information
- the answer requires external documentation
- the question involves GitHub repositories,
  issues, discussions, or implementation details
- you need evidence to support your answer

When you have enough information, provide a concise
answer supported by the research you found.
"""


def research_node(state: ResearchState):
    messages = [SystemMessage(content=SYSTEM_PROMPT),*state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}