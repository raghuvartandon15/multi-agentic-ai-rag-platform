from rag.node import rag_node


state = {
    "query": "Why is my Kubernetes pod stuck in CrashLoopBackOff?"
}


result = rag_node(state)


print("\n==============================")
print("ANSWER")
print("==============================\n")

print(result["rag_response"])


print("\n==============================")
print("RETRIEVED DOCUMENTS")
print("==============================\n")

for i, document in enumerate(
    result["retrieved_docs"],
    start=1
):
    print(f"---------- Document {i} ----------")

    print(
        "Technology:",
        document.get("technology")
    )

    print(
        "Title:",
        document.get("title")
    )

    print(
        "Score:",
        document.get("score")
    )

    print(
        "Source:",
        document.get("source_url")
    )