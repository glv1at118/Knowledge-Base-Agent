import agent
import agent_tools
from logger import init_logger_session, print_colored, log_msg
from constants import EXIT_COMMANDS, BANNER, RED_COLOR, GREEN_COLOR

# The main entry of this knowledge RAG application as a terminal
def main() -> None:
    print_colored(BANNER, GREEN_COLOR)
    log_path = init_logger_session()
    print_colored("Application starts: Initializing knowledge context...", GREEN_COLOR)

    try:
        agent_tools.initialize()
    except Exception as error:
        print_colored(f"FATAL: Could not start due to: {error}", RED_COLOR)
        return

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
        log_msg(f"User_Asked: {question}; Agent_Answered: {answer}", print_terminal=False)

if __name__ == "__main__":
    main()
