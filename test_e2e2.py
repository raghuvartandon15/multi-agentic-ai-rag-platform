from supervisor_agent.graph import supervisor_graph


config = {"configurable": {"thread_id": "user-1"}}

query_1 = """
who are you?
"""

result = supervisor_graph.invoke({
            "messages": [{"role": "user","content": query_1}],
            "raw_input": query_1,
            }, 
        config=config
    )

print("\n" + "#" * 70)
print("# FINAL RESULT")
print("#" * 70)

print("\nFinal response:")
print(result.get("final_response"))

print("\nFinal state keys:")
for key in result:
    print("-", key)

print("\n" + "#" * 70)
print("# MEMORY CHECK")
print("#" * 70)
stored_state = supervisor_graph.get_state(config)

for message in stored_state.values.get("messages", []):
    print(f"\n{message.type}:")
    print(message.content)