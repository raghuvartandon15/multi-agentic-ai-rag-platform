from pathlib import Path
from input_handler.node import input_handler_node
from langchain_core.messages import HumanMessage

state = {
    "messages": [
        HumanMessage(
            content="What is this document about?"
        )
    ],
    "raw_input": Path("sample.pdf"),
}

result = input_handler_node(state)

print("\n==============================")
print("INPUT HANDLER RESULT")
print("==============================")

print("Input type:", result["input_type"])
print("Extracted content:")
print(result["extracted_content"][:2000])