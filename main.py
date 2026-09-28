import agent
import agent_tools
import logger
from constants import EXIT_COMMANDS

# The main entry of this knowledge RAG application as a terminal
def main() -> None:
    print("Application starts: Initializing knowledge context...")

    try:
        agent_tools.initialize()
    except Exception as error:
        print(f"FATAL: Could not start due to: {error}")
        return

    log_path = logger.init_logger_session()

    print(f"Application is ready, logging this session to {log_path}")
    print("Each question is answered independently, please mention the topic by name each time.")
    print(f"Type {' or '.join(EXIT_COMMANDS)} to leave.\n")

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
            print(f"\n[Error while answering: {error}]")
            continue

        print(answer)
        logger.log_msg(question, answer)


if __name__ == "__main__":
    main()
