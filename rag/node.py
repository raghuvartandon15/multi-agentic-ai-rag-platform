from rag.retriever import HybridRetriever
from rag.generator import generate_answer


retriever = HybridRetriever()


def rag_node(state):
    print("\n[RAG] START")

    query = state["query"]

    print(f"[RAG] Query: {query}")
    print("[RAG] Starting retrieval...")

    documents = retriever.retrieve(query=query)

    print(f"[RAG] Retrieval complete: {len(documents)} documents")

    print("[RAG] Starting generation...")

    response = generate_answer(
        query=query,
        documents=documents
    )

    print("[RAG] Generation complete")

    return {
        "rag_response": response,
        "retrieved_docs": documents,
        "last_agent": "rag",
    }