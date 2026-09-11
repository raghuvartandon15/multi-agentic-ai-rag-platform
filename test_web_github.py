from web_github_agent.graph import research_graph

result = research_graph.invoke({"messages": [{"role": "user","content": "who is the 2025 f1 champion"}]})

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