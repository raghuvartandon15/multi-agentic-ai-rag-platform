from langchain_core.messages import AIMessage
from supervisor_agent.state import AppState
from input_guardrails.guardrails import check_input


def input_guardrails_node(appstate: AppState):
    query = appstate["messages"][-1].content
    decision = check_input(query)
    print(f"[INPUT GUARDRAIL] Decision: {decision}")
    if decision == "UNSAFE":
        response = "I'm sorry, but I can't help with that request."
        return {
            "input_blocked": True,
            "final_response": response,
            "messages": [
                AIMessage(content=response)
            ],
        }

    return {
        "input_blocked": False,
        "query": query,
    }