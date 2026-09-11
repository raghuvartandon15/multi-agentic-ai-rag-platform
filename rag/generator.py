from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from rag.config import MODEL_PROVIDER, MODEL
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL, timeout=60)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful technical assistant.

Answer the user's question using only the provided context.

Rules:

1. Use the retrieved context as the source of truth.
2. Do not invent information that is not supported by the context.
3. If the context does not contain enough information to answer
   the question, say that you don't know based on the available
   documentation.
4. Give a clear and concise technical explanation.
5. When useful, mention which technology or documentation the
   information came from.

Retrieved Context:
{context}
"""
        ),
        (
            "human",
            "{query}"
        )
    ]
)


def generate_answer(
    query: str,
    documents: list
):

    print("[GENERATOR] START")

    if not documents:
        print("[GENERATOR] No documents")
        return "I don't know based on the available documentation."

    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"""
--- Document {i} ---

Technology:
{document.get("technology")}

Title:
{document.get("title")}

Source:
{document.get("source_url")}

Content:
{document.get("content")}
"""
        )

    context = "\n".join(context_parts)

    print(f"[GENERATOR] Context length: {len(context)} characters")
    print("[GENERATOR] Calling LLM...")

    chain = rag_prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "context": context
        }
    )

    print("[GENERATOR] LLM returned")

    return response.content