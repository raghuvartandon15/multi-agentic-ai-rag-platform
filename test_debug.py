from debug_agent.graph import debug_graph


result = debug_graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
I am getting this error:

ModuleNotFoundError: No module named 'web_github_agent'

I am running:

python graph.py

My project looks like:

final-rag-project/
    web_github_agent/
        __init__.py
        graph.py
        node.py
        tools.py

What is causing this problem and
how should I fix it?
"""
            }
        ]
    }
)


for message in result["messages"]:
    print("\n==============================")
    print(
        "Type:",
        type(message).__name__
    )
    print(
        "Content:",
        message.content
    )
    if hasattr(message, "tool_calls"):
        print(
            "Tool calls:",
            message.tool_calls
        )