from output_guardrails.node import guardrails_node


test_response = """
Sure! Your credit card number is 4111 8765 2645 3423.

Your API key is sk-123456789abcdef123456789.
"""


state = {
    "final_response": test_response
}

result = guardrails_node(state)

print("\n==============================")
print("ORIGINAL RESPONSE")
print("==============================")
print(test_response)

print("\n==============================")
print("GUARDED RESPONSE")
print("==============================")
print(result["final_response"])