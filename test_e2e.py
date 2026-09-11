from supervisor_agent.graph import supervisor_graph


config = {"configurable": {"thread_id": "user-1"}}

query_1 = """
who won the lastest f1 championship 2025
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


# --------------------------------------------------
# Second turn
# --------------------------------------------------

query_2 = """
what was my last question?
"""

print("\n" + "#" * 70)
print("# TURN 2")
print("#" * 70)

result_2 = supervisor_graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": query_2
            }
        ],
        "raw_input": query_2,
    },
    config=config,
)

print("\nFinal response:")
print(result_2.get("final_response"))

stored_state = supervisor_graph.get_state(config)

print("\n" + "#" * 70)
print("# MEMORY CHECK")
print("#" * 70)

for message in stored_state.values.get("messages", []):
    print(f"\n{message.type}:")
    print(message.content)