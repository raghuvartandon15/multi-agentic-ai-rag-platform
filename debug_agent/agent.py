from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

from debug_agent.config import (
    MODEL,
    MODEL_PROVIDER
)

from debug_agent.tools import debug_tools


llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL)

llm_with_tools = llm.bind_tools(debug_tools)

SYSTEM_PROMPT = """
You are a software debugging agent.

Your job is to diagnose software problems using
the tools available to you.

Follow this general strategy:

1. Understand the reported error or problem.

2. Inspect the project structure when necessary.

3. Read relevant source files.

4. Search the codebase for relevant symbols,
   imports, functions, variables, and error messages.

5. Inspect installed Python packages when
   dependency issues are possible.

6. Use web search when local investigation
   is insufficient.

7. Form a specific hypothesis about the
   root cause.

8. If possible, verify the hypothesis using
   additional tool calls.

9. Give a concise final diagnosis and
   recommended fix.

Do not immediately recommend installing packages
or changing code.

First investigate the actual project and environment.

When using web search, prefer official documentation
and reliable technical sources.
"""


def debug_agent(state):
    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        *state["messages"],
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}