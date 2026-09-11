from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from input_guardrails.config import MODEL, MODEL_PROVIDER
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL)

input_guardrail_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an input safety classifier for a technical AI assistant.

Your job is to classify the user's request.

The assistant is designed to help with:
- RAG
- Kubernetes
- AWS
- FastAPI
- MLflow
- NVIDIA Triton
- debugging
- deployment
- MLOps
- AIOps
- software engineering
- technical research
- General Questions

Classify the request into exactly one category:

SAFE
UNSAFE

Return ONLY one word:
SAFE
or
UNSAFE

Classify as UNSAFE if the user is requesting clearly harmful,
illegal, malicious, or dangerous assistance.

Normal technical questions, debugging questions,
deployment questions, security troubleshooting,
and general research questions should be considered SAFE.
"""
        ),
        (
            "human",
            "{query}"
        )
    ]
)


def check_input(query: str) -> str:
    chain = input_guardrail_prompt | llm

    response = chain.invoke({"query": query})
    decision = response.content.strip().upper()

    if "UNSAFE" in decision:
        return "UNSAFE"
    return "SAFE"