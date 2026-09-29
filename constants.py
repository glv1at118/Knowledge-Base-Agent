from enum import Enum
from typing import Literal
from smolagents import LogLevel

# Embedding and retrieval related params
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.25
MAX_RESULT_DEFAULT = 6
Tier = Literal["confident", "ambiguous", "no_match"]

# Folder paths
KNOWLEDGE_FOLDER_NAME = "knowledge"
OUTPUT_FOLDER_NAME = "output"
LOGS_FOLDER_NAME = "logs"

# LLM agent related params
LLM_MODEL_ID = "gpt-4o-mini"
AGENT_MAX_STEPS = 4
AGENT_LOG_LEVEL = LogLevel.DEBUG

# Terminal interactivity
EXIT_COMMANDS = ("exit", "quit")