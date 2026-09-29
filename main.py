import agent
import agent_tools
import logger
from constants import EXIT_COMMANDS, BANNER, RED_COLOR, GREEN_COLOR, RESET_COLOR

def print_colored(text: str, color: str) -> None:
    print(f"{color}{text}{RESET_COLOR}")

# The main entry of this knowledge RAG application as a terminal
def main() -> None:
    print_colored(BANNER, GREEN_COLOR)
    print_colored("Application starts: Initializing knowledge context...", GREEN_COLOR)

    try:
        agent_tools.initialize()
    except Exception as error:
        print_colored(f"FATAL: Could not start due to: {error}", RED_COLOR)
        return

    log_path = logger.init_logger_session()

    print_colored(f"Application is ready, logging this session to {log_path}", GREEN_COLOR)
    print_colored("Each question is answered independently, please mention the topic by name each time.", GREEN_COLOR)
    print_colored(f"Type {' or '.join(EXIT_COMMANDS)} to leave.\n", GREEN_COLOR)

    # Keep this terminal interactive CLI active
    while True:
        question = input(">>> ").strip()
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = agent.ask(question)
        except Exception as error:
            print_colored(f"\n[Error while answering: {error}]", RED_COLOR)
            continue

        print_colored(answer, GREEN_COLOR)
        logger.log_msg(question, answer)

if __name__ == "__main__":
    main()
