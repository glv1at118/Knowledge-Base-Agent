from datetime import datetime
from pathlib import Path
from typing import Optional
from constants import LOGS_FOLDER_NAME

# this represents the current active chat session's log path
_log_path: Optional[Path] = None

def init_logger_session() -> Path:
    """
        Creates a new timestamped log file for this running session. Returns its path.
        This method MUST be called ONCE before any call of log_msg.
    """
    logs_folder = Path(LOGS_FOLDER_NAME)
    logs_folder.mkdir(parents=True, exist_ok=True)

    global _log_path

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = "session_" + timestamp + ".txt"
    _log_path = logs_folder.joinpath(filename)

    return _log_path

def log_msg(question: str, answer: str) -> None:
    """
        Appends one question/answer turn to the current session's log file.
        This method CANNOT be called if init_logger_session is never called in a new session.
    """
    if _log_path is None:
        raise RuntimeError("Error: No available logger session exists! Please call init_logger_session() first!")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}]\nQ: {question}\nA: {answer}\n\n\n\n"

    with _log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)
