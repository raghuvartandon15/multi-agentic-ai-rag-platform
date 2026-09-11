from pathlib import Path
from langchain_core.messages import HumanMessage
from supervisor_agent.graph import supervisor_graph


result = supervisor_graph.invoke({
    "messages": [
        HumanMessage(
            content="What technologies are shown in this architecture?"
        )
    ],
    "raw_input": Path("architecture.png")
})

print("\n==============================")
print("FINAL RESPONSE")
print("==============================")
print(result.get("final_response"))

print("\n==============================")
print("INPUT TYPE")
print("==============================")
print(result.get("input_type"))

print("\n==============================")
print("EXTRACTED CONTENT")
print("==============================")
print(result.get("extracted_content"))

print("\n==============================")
print("FINAL STATE")
print("==============================")

for key, value in result.items():
    print(f"\n{key}:")
    print(value)