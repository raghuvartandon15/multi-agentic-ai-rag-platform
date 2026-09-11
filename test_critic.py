from critic_agent.agent import critic_agent


# result = critic_agent(
#     {
#         "last_agent": "research",
#         "query": "Who won the 2025 F1 championship?"
#     }
# )

# print(result)
result = critic_agent(
    {
        "last_agent": "rag",
        "query": "Why is my Kubernetes pod stuck in CrashLoopBackOff?",
        "rag_response": "CrashLoopBackOff means the container repeatedly crashes.",
        "retrieved_docs": [
            {
                "content": "A pod enters CrashLoopBackOff when its container repeatedly crashes.",
                "source_url": "https://kubernetes.io/"
            }
        ]
    }
)

print(result)