import os
import dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent
from agent_tools import save_artifact, search_knowledge_base
from constants import AGENT_MAX_STEPS, LLM_MODEL_ID

dotenv.load_dotenv()

_model = OpenAIServerModel(
    model_id=LLM_MODEL_ID,
    api_base=os.getenv("UDACITY_BASE_URL"),
    api_key=os.getenv("UDACITY_OPENAI_API_KEY"),
)

_agent = ToolCallingAgent(
    tools=[search_knowledge_base, save_artifact],
    model=_model,
    max_steps=AGENT_MAX_STEPS,
)

def ask(question: str) -> str:
    return _agent.run(
        f"""
            You are a private knowledge base assistant. You must answer strictly from the
            knowledge base, NEVER from your own general knowledge, and NEVER from the
            internet.

            For the question below, call the search_knowledge_base tool exactly ONCE,
            passing the question exactly as written. Do not rephrase, reword, shorten,
            or guess at what the user "really" meant. Always search with their literal
            wording. Then follow the CONFIDENCE tier instructions returned by that tool
            exactly, to decide how to respond.

            If the user separately asks you to save or write out a summary or other
            artifact, use the save_artifact tool for that.

            Tools available to you: search_knowledge_base, save_artifact

            User question: "{question}"
        """,
        stream = True
    )
