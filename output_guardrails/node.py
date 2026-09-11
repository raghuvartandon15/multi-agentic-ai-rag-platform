from langchain_core.messages import AIMessage
from supervisor_agent.state import AppState
from output_guardrails.guardrails import apply_output_guardrails


def guardrails_node(appstate: AppState):
    print("\n" + "=" * 60)
    print("[OUTPUT GUARDRAILS] START")
    print("=" * 60)

    response_text = appstate["final_response"]

    print("\nBefore guardrails:")
    print(response_text)

    sanitized_response = apply_output_guardrails(response_text)

    print("\nAfter guardrails:")
    print(sanitized_response)

    return {
        "final_response": sanitized_response,
        "messages": [
            AIMessage(content=sanitized_response)
        ],
    }