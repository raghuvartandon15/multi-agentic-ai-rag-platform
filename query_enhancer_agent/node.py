from supervisor_agent.state import AppState
from query_enhancer_agent.analyzer import analyze_query
from query_enhancer_agent.decomposer import decompose_query

def query_analyzer_node(appstate: AppState):
    query = appstate["query"]
    print("\n" + "=" * 60)
    print("[QUERY ANALYZER] START")
    print("=" * 60)
    print(f"Input query:\n{query}")
    should_decompose = analyze_query(query)
    print(f"\nDecision: {should_decompose}")

    if not should_decompose:
        output = {
            "should_decompose": False,
            "processed_query": query,
        }
    else:
        output = {
            "should_decompose": True,
        }

    print(f"Output: {output}")
    return output


def query_decomposer_node(appstate: AppState):
    query = appstate["query"]

    print("\n" + "=" * 60)
    print("[QUERY DECOMPOSER] START")
    print("=" * 60)
    print(f"Input query:\n{query}")

    sub_queries = decompose_query(query)
    processed_query = "\n".join(
        f"- {sub_query}"
        for sub_query in sub_queries
    )
    print("\nGenerated sub-queries:")

    for i, sub_query in enumerate(sub_queries, start=1):
        print(f"  {i}. {sub_query}")

    print("\nProcessed query:")
    print(processed_query)

    return {
        "sub_queries": sub_queries,
        "processed_query": processed_query,
    }