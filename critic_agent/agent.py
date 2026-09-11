from critic_agent.graph import critic_graph

def critic_agent(state):
    if state.get("last_agent") != "rag":
        return {"critic_skipped": True}

    result = critic_graph.invoke(
        {
            "query": state["query"],
            "rag_response": state["rag_response"],
            "retrieved_docs": state["retrieved_docs"]
        }
    )

    return {
        "correctness": result["correctness"],
        "faithfulness": result["faithfulness"],
        "retrieval_relevance": result["retrieval_relevance"],
        "passed": result["passed"],
        "critic_skipped": False
    }