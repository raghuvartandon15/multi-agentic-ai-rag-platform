from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

tavily = TavilySearch()

@tool
def web_search(query: str):
    """
    Search the internet for current or external information.

    Use this tool when information is not available
    from the model's existing knowledge or when
    up-to-date information is required.
    """
    return tavily.invoke({"query": query})


research_tools = [
    web_search
]
