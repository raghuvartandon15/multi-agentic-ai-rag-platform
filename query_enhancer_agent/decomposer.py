from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from query_enhancer_agent.config import MODEL, MODEL_PROVIDER


llm = init_chat_model(model=MODEL,model_provider=MODEL_PROVIDER,)

def decompose_query(query: str):
    prompt = PromptTemplate.from_template(
        """
Break the following query into smaller independent
sub-queries.

Each sub-query should represent one distinct question
or task.

Only decompose the query.
Do not answer the questions.

Return one sub-query per line.

Query:
{query}
"""
    )

    chain = prompt | llm
    response = chain.invoke({
        "query": query
    })
    sub_queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    return sub_queries