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
LLM_MODEL_ID = "gpt-4.1"
AGENT_MAX_STEPS = 4
AGENT_LOG_LEVEL = LogLevel.INFO

# Terminal interactivity & visuals
EXIT_COMMANDS = ("exit", "quit")
GREEN_COLOR = "\033[92m"
RED_COLOR = "\033[91m"
RESET_COLOR = "\033[0m"
BANNER = r"""
 _  __ ____      _                    _
| |/ /| __ )    / \   __ _  ___ _ __ | |_
| ' / |  _ \   / _ \ / _` |/ _ \ '_ \| __|
| . \ | |_) | / ___ \ (_| |  __/ | | | |_
|_|\_\|____/ /_/   \_\__, |\___|_| |_|\__|
                     |___/
    Private Knowledge Base Assistant
"""