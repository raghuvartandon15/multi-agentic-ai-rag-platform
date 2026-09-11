from langchain.chat_models import init_chat_model

from supervisor_agent.config import MODEL, MODEL_PROVIDER
from dotenv import load_dotenv
from rag.graph import rag_graph
from web_github_agent.graph import research_graph
from debug_agent.graph import debug_graph
from rag.node import rag_node as run_rag
from critic_agent.graph import critic_graph

load_dotenv()

llm = init_chat_model(model_provider=MODEL_PROVIDER, model=MODEL, timeout=60)

SUPERVISOR_PROMPT = """
You are the supervisor of a technical AI system.
Your name is 'RaghuvarGPT'

Your job is to understand the user's request and delegate it
to the most appropriate specialist.

You have three specialist agents:

1. CHAT:
    -Use for greetings, casual conversation, questions about the
    assistant itself, and simple conversational interaction that
    does not require RAG, web research, or debugging.

2. RAG Agent
   - Answers technical questions using the project's curated
     documentation and knowledge base.
   - Use this FIRST when the user is asking for an explanation,
     concept, procedure, configuration guidance, troubleshooting
     knowledge, or how something works.
   - Examples:
       "Why is Kubernetes CrashLoopBackOff happening?"
       "How do I troubleshoot a CrashLoopBackOff?"
       "What does Triton model control mode NONE do?"
       "How does MLflow tracking server work?"

3. Web/GitHub Research Agent
   - Searches external/current information.
   - Use this when the user explicitly needs current information,
     recent changes, external documentation, GitHub repositories,
     GitHub issues/discussions, or information that may not exist
     in the internal RAG knowledge base.
   - Examples:
       "What is the latest Kubernetes recommendation for this?"
       "Find recent GitHub issues about this error."
       "What changed in the latest version?"

4. Debug Agent
   - Investigates an actual software/project problem.
   - It can inspect project files, source code, dependencies,
     installed packages, and environment information.
   - Use this when the user asks you to investigate or diagnose
     THEIR actual project, code, environment, configuration,
     logs, or error.
   - Examples:
       "My project is throwing ModuleNotFoundError. Find the cause."
       "Inspect my code and tell me why this is failing."
       "My application crashes when I run it. Debug it."

IMPORTANT ROUTING RULES:

- Prefer RAG for general technical questions and troubleshooting
  knowledge.

- Do NOT route to Debug merely because the question contains an
  error name, exception, failure state, or troubleshooting term.

- Route to Debug only when the user is asking you to investigate
  their actual project, code, environment, configuration, logs,
  or runtime problem.

- Prefer Web/GitHub Research when the user asks for current,
  latest, recent, external, or GitHub-specific information.

- If a question can be answered from the internal documentation,
  prefer the RAG Agent over Web/GitHub Research.

- If the request genuinely requires multiple specialists, use
  multiple agents when necessary.

After receiving the specialist's result, provide the final answer
to the user.
"""

def supervisor_node(state):
    print("\n[SUPERVISOR] START")

    query = state["processed_query"]
    print(f"[SUPERVISOR] Query: {query}")

    prompt = f"""
{SUPERVISOR_PROMPT}

User request:

{query}

Respond with exactly ONE of these routing decisions:

RAG
RESEARCH
DEBUG
CHAT
"""

    print("[SUPERVISOR] Calling LLM...")

    response = llm.invoke(prompt)

    print("[SUPERVISOR] LLM returned")

    decision = response.content.strip().upper()

    print(f"[SUPERVISOR] Decision: {decision}")

    if "RESEARCH" in decision:
        route = "research"
    elif "DEBUG" in decision:
        route = "debug"
    elif "CHAT" in decision:
        route = "chat"
    elif "RAG" in decision:
        route = "rag"
    else:
        route = "rag"

    print(f"[SUPERVISOR] Route: {route}")

    return {
        # "query": query,
        "next_agent": route,
        "retry_count": state.get("retry_count", 0)
    }


def research_node(state):
    from web_github_agent.graph import research_graph
    result = research_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["query"],
                }
            ]
        }
    )
    response = result["messages"][-1].content

    return {
      "research_response": response,
      "final_response": response,
      "last_agent": "research",
  }

def debug_node(state):
    from debug_agent.graph import debug_graph
    result = debug_graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": state["query"],
                }
            ]
        }
    )
    response = result["messages"][-1].content

    return {
      "debug_response": response,
      "final_response": response,
      "last_agent": "debug",
  }

def critic_node(state):
    print("\n[CRITIC] START")

    if state.get("last_agent") != "rag":
        return {"critic_skipped": True}

    result = critic_graph.invoke(
        {
            "query": state["query"],
            "rag_response": state["rag_response"],
            "retrieved_docs": state["retrieved_docs"]
        }
    )

    retry_count = state.get("retry_count", 0)

    if not result["passed"]:
        retry_count += 1

    print(f"[CRITIC] Passed: {result['passed']}")
    print(f"[CRITIC] Retry count: {retry_count}")

    return {
        "correctness": result["correctness"],
        "faithfulness": result["faithfulness"],
        "retrieval_relevance": result["retrieval_relevance"],
        "passed": result["passed"],
        "critic_skipped": False,
        "retry_count": retry_count,
    }

def specialist_router(state):
    return state["next_agent"]


def critic_result_router(state):

    if state.get("passed"):
        return "final"

    if state.get("retry_count", 0) >= 2:
        print("[CRITIC ROUTER] Maximum retries reached → END")
        return "final"

    return "supervisor"

def rag_node(state):
    print("\n" + "=" * 60)
    print("[RAG] START")
    print("=" * 60)

    query = state["processed_query"]
    print(f"Input query:\n{query}")

    result = run_rag({"query": query})

    print("\nRAG response:")
    print(result["rag_response"])
    print(f"\nRetrieved documents: {len(result['retrieved_docs'])}")

    return {
        "rag_response": result["rag_response"],
        "retrieved_docs": result["retrieved_docs"],
        "last_agent": "rag",
        "final_response": result["rag_response"],
    }

def chat_node(state):
    query = state["processed_query"]

    response = llm.invoke([
        {
            "role": "system",
            "content": """
You are the general conversational assistant for an Agentic RAG application.

Your identity is RaghuvarGPT.

When the user asks for your name or identity, identify yourself
as RaghuvarGPT.

Do not identify yourself as ChatGPT, an OpenAI assistant, or any
other name.

You can help users with:
- RAG
- Agentic AI
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

For greetings, casual conversation, and questions about the assistant itself,
respond naturally and conversationally.

Do not pretend that casual conversation requires document retrieval.
"""
        },
        {
            "role": "user",
            "content": query
        }
    ])

    return {
        "final_response": response.content,
        "last_agent": "chat",
    }