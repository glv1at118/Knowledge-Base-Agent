import os
import dotenv
from smolagents import OpenAIServerModel, ToolCallingAgent
from agent_tools import save_artifact, search_knowledge_base
from constants import AGENT_MAX_STEPS, LLM_MODEL_ID, AGENT_LOG_LEVEL

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
    verbosity_level=AGENT_LOG_LEVEL, # debug or info here when I run local testing, put to off when in demo
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
            exactly, to decide how to respond. You MUST attach a confidence label at 
            the head of your response message, based on the confidence tier level.

            For example:
            - If the confidence tier is confident, then you MUST also attach 
            a <I_AM_CONFIDENT> label at the head of your answer.
            - If the confidence tier is ambiguous, then you MUST also attach a 
            <I_AM_NOT_VERY_SURE> label at the head of your answer.
            - If the confidence tier is no_match, then you MUST also attach a 
            <I_HAVE_NO_INFO> label at the head of your answer.

            If the user's question asks you to write, save, export, or create a file
            or document, for example "write me a summary", "save this as a doc",
            "create a file with...", "give me a report I can keep", and etc., you MUST call
            the save_artifact tool to actually create that file. Do not just describe
            the content in your reply without also saving it when this applies.

            When saving, choose the filename and extension based on the content: use
            ".md" for a written summary, report, or any multi-section document (so
            headings/formatting are preserved), and ".txt" for a short, single block
            of plain text. If the user names a specific format (e.g. "as a text
            file", "as markdown"), use that instead. Give the file a short,
            descriptive name related to its content (e.g. "northlight_summary.md").

            Tools available to you: search_knowledge_base, save_artifact

            User question: "{question}"
        """
    )
