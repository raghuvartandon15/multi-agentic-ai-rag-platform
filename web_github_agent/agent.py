from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from web_github_agent.config import MODEL,MODEL_PROVIDER

from web_github_agent.tools import research_tools

load_dotenv()

llm = init_chat_model(model_provider=MODEL_PROVIDER,model=MODEL)

llm_with_tools = llm.bind_tools(research_tools)