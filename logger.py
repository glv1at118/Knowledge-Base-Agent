from datetime import datetime
from pathlib import Path
from typing import Optional
from constants import LOGS_FOLDER_NAME, GREEN_COLOR, RESET_COLOR

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

def log_msg(info: str, print_terminal: bool = False) -> None:
    """
        Appends one log message which contains debug info, to the current session's log file.
        It will optionally output the logged message to the terminal, depending on the second argument.
        This method CANNOT be called if init_logger_session is never called in a new session.
    """
    if _log_path is None:
        raise RuntimeError("Error: No available logger session exists! Please call init_logger_session() first!")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp} - {info}\n"

    if print_terminal:
        print_colored(text=info, color=GREEN_COLOR)

    with _log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(entry)

def print_colored(text: str, color: str) -> None:
    print(f"{color}{text}{RESET_COLOR}")