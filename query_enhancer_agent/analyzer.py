from typing import Literal
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from query_enhancer_agent.config import MODEL, MODEL_PROVIDER


llm = init_chat_model(model=MODEL,model_provider=MODEL_PROVIDER,)

class QueryAnalysis(BaseModel):
    should_decompose: Literal["true", "false"]

def analyze_query(query: str) -> bool:
    prompt = PromptTemplate.from_template(
        """
You are a query complexity analyzer.

Determine whether the user's query contains multiple
distinct questions or tasks that would benefit from
being broken into separate sub-queries.

Return "true" if decomposition would be useful.

Return "false" if the query is simple, asks one
straightforward question, or does not benefit from
decomposition.

Query:
{query}
"""
    )

    structured_llm = llm.with_structured_output(
            QueryAnalysis,
            method="json_schema",
            strict=True,
        )
    chain = prompt | structured_llm
    response = chain.invoke({
        "query": query
    })

    return response.should_decompose == "true"