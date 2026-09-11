from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from critic_agent.config import MODEL_PROVIDER, MODEL

load_dotenv()


llm = init_chat_model(
    model_provider=MODEL_PROVIDER,
    model=MODEL,
    reasoning_effort="low",
    temperature=0,
    timeout=60
)


class ScoreOutput(BaseModel):
    score: int = Field(
        ge=0,
        le=10,
        description="Whole-number score from 0 to 10."
    )


def get_score(prompt: str) -> int:
    """
    Ask the evaluator LLM for a JSON score and validate it.
    """

    evaluator = llm.bind(
        response_format={
            "type": "json_object"
        }
    )

    response = evaluator.invoke(prompt)

    result = ScoreOutput.model_validate_json(response.content)

    return result.score


def format_documents(documents: list) -> str:
    return "\n\n".join(
        [
            f"""
Document {i + 1}
Technology: {doc.get("technology", "")}
Source: {doc.get("source_url", "")}

Content:
{doc.get("content", "")}
"""
            for i, doc in enumerate(documents)
        ]
    )


def evaluate_correctness(
    query: str,
    response: str
) -> int:

    prompt = PromptTemplate.from_template(
        """
        You are evaluating the correctness of a RAG answer.

        Determine how well the LLM response answers the user's query.

        Scoring:
        0 = completely incorrect
        10 = completely correct

        The score must be a whole-number integer from 0 to 10.

        Return ONLY valid JSON in exactly this format:

        {{"score": 0}}

        Replace 0 with your score.

        Do not return explanations.
        Do not use decimal values.
        Do not include markdown.

        Query:
        {query}

        LLM Response:
        {response}
        """
    )

    formatted_prompt = prompt.format(
        query=query,
        response=response
    )

    return get_score(formatted_prompt)


def evaluate_faithfulness(
    retrieved_docs: list,
    response: str
) -> int:

    prompt = PromptTemplate.from_template(
        """
        You are evaluating the faithfulness of a RAG answer.

        Determine whether the claims made in the LLM response
        are supported by the retrieved documents.

        Use ONLY the retrieved documents.
        Do not use outside knowledge.

        If the response contains claims that are not supported
        by the retrieved documents, lower the score.

        Scoring:
        0 = completely unsupported
        10 = all major claims are supported by the documents

        The score must be a whole-number integer from 0 to 10.

        Return ONLY valid JSON in exactly this format:

        {{"score": 0}}

        Replace 0 with your score.

        Do not return explanations.
        Do not use decimal values.
        Do not include markdown.

        Retrieved Documents:
        {retrieved_docs}

        LLM Response:
        {response}
        """
    )

    formatted_prompt = prompt.format(
        retrieved_docs=format_documents(retrieved_docs),
        response=response
    )

    return get_score(formatted_prompt)


def evaluate_retrieval_relevance(
    query: str,
    retrieved_docs: list
) -> int:

    prompt = PromptTemplate.from_template(
        """
        You are evaluating the retrieval quality of a RAG system.

        Determine how relevant the retrieved documents are
        for answering the user's query.

        Evaluate the retrieved documents as a group.

        Scoring:
        0 = completely irrelevant
        10 = highly relevant

        The score must be a whole-number integer from 0 to 10.

        Return ONLY valid JSON in exactly this format:

        {{"score": 0}}

        Replace 0 with your score.

        Do not return explanations.
        Do not use decimal values.
        Do not include markdown.

        Query:
        {query}

        Retrieved Documents:
        {retrieved_docs}
        """
    )

    formatted_prompt = prompt.format(
        query=query,
        retrieved_docs=format_documents(retrieved_docs)
    )

    return get_score(formatted_prompt)