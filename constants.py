from typing import Literal

# Embedding and retrieval related params
EMBEDDING_MODEL_NAME = "multi-qa-mpnet-base-dot-v1"
HIGH_CONFIDENCE_THRESHOLD = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.25
MAX_RESULT_DEFAULT = 3
Tier = Literal["confident", "ambiguous", "no_match"]

# Folder paths
KNOWLEDGE_FOLDER_NAME = "knowledge"
OUTPUT_FOLDER_NAME = "output"
LOGS_FOLDER_NAME = "logs"

# LLM agent related params
LLM_MODEL_ID = "gpt-4o-mini"
AGENT_MAX_STEPS = 3
