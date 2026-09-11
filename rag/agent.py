from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from rag.config import MODEL_PROVIDER, MODEL
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL)


rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a technical documentation assistant.

Answer the user's question using only the
provided retrieved context.

If the retrieved context does not contain
enough information to answer the question,
say that you do not know.

Do not invent information.

When possible, use the source information
from the retrieved documents to make the
answer clear and specific.

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

    context = "\n\n".join(
        [
            f"""
Source: {doc["source_url"]}

Technology: {doc["technology"]}

Content:
{doc["content"]}
"""
            for doc in documents
        ]
    )

    chain = rag_prompt | llm

    response = chain.invoke(
        {
            "query": query,
            "context": context
        }
    )

    return response