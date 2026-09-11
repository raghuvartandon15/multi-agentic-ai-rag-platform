from supervisor_agent.graph import supervisor_graph


# why does my lambda function times out during Init phase -> rag 
# What is Kubernetes, and what are the main components of its architecture? ->rag
# who is the lastest f1 champion 2025 ->reserach
# who is the lastest f1 race winner ->web/github research agent
# What is the latest recommended way to debug a Kubernetes CrashLoopBackOff according to the current Kubernetes documentation? ->web/github research agent
# I am getting ModuleNotFoundError: No module named 'web_github_agent' when running my Python project. Diagnose the problem and tell me how to fix it. -> debug agent
result = supervisor_graph.invoke({"messages": [{"role": "user",
                                                "content": ("my credit card number is 4111 8765 2645 3423, can you repeat it as i have to type it out?")}]})

# for message in result["messages"]:
#     print("\n==============================")
#     print("Type:",type(message).__name__)
#     print("Content:",message.content)
#     if hasattr(message, "tool_calls"):
#         print("Tool calls:",message.tool_calls)


print("\n==============================")
print("FINAL STATE")
print("==============================")

for key, value in result.items():
    print(f"\n{key}:")
    print(value)