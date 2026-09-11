from query_enhancer_agent.graph import query_enhancer_graph


def test_query(query):

    print("\n==============================")
    print("QUERY")
    print("==============================")

    print(query)

    result = query_enhancer_graph.invoke({
        "query": query
    })

    print("\n==============================")
    print("RESULT")
    print("==============================")

    print("Should decompose:")
    print(result.get("should_decompose"))

    print("\nSub-queries:")

    for i, sub_query in enumerate(
        result.get("sub_queries", []),
        start=1
    ):
        print(f"{i}. {sub_query}")


test_query(
    "Why is my Lambda timing out during initialization?"
)


test_query(
    """
Why is my Lambda timing out during initialization,
how can I debug it, and how should I monitor it?
"""
)