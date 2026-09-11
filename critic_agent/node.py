from typing import TypedDict

from critic_agent.evaluator import evaluate_correctness,evaluate_faithfulness,evaluate_retrieval_relevance

from critic_agent.config import CORRECTNESS_THRESHOLD,FAITHFULNESS_THRESHOLD,RETRIEVAL_RELEVANCE_THRESHOLD


class CriticState(TypedDict, total=False):
    # Information produced by the previous agent
    query: str
    last_agent: str

    # RAG-specific information
    rag_response: str
    retrieved_docs: list

    # Evaluation results
    correctness: int
    faithfulness: int
    retrieval_relevance: int

    # Final decision
    passed: bool


def correctness_node(state: CriticState):
    score = evaluate_correctness(query=state["query"],response=state["rag_response"])
    return {"correctness": score}


def faithfulness_node(state: CriticState):
    score = evaluate_faithfulness(retrieved_docs=state["retrieved_docs"],response=state["rag_response"])
    return {"faithfulness": score}


def retrieval_relevance_node(state: CriticState):
    score = evaluate_retrieval_relevance(query=state["query"],retrieved_docs=state["retrieved_docs"])
    return {"retrieval_relevance": score}


def pass_fail_node(state: CriticState):
    passed = (
        state["correctness"] >= CORRECTNESS_THRESHOLD
        and
        state["faithfulness"] >= FAITHFULNESS_THRESHOLD
        and
        state["retrieval_relevance"] >= RETRIEVAL_RELEVANCE_THRESHOLD
    )

    return {"passed": passed}


def should_evaluate(state: CriticState):
    if state.get("last_agent") == "rag":
        return "evaluate"

    return "skip"