from supervisor_agent.state import AppState
from input_handler.handler import process_input


def input_handler_node(appstate: AppState):
    print("[INPUT HANDLER] START")
    
    user_input = appstate.get("raw_input")

    if user_input is None:
        user_input = appstate["messages"][-1].content
    result = process_input(user_input)

    print(f"[INPUT HANDLER] Input type: {result['input_type']}")

    return {
        "input_type": result["input_type"],
        "raw_input": result["raw_input"],
        "extracted_content": result["extracted_content"],
        "query": result["extracted_content"],
    }