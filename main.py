"""AI Agent - CLI interface"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model import LLM
from core.agent import Agent
import config


def print_colored(text: str, color: str = "", end: str = "\n"):
    colors = {
        "green": "\033[92m", "cyan": "\033[96m",
        "yellow": "\033[93m", "red": "\033[91m",
        "bold": "\033[1m", "reset": "\033[0m",
    }
    c = colors.get(color, "")
    r = colors.get("reset", "")
    print(f"{c}{text}{r}", end=end)


def run_cli(args):
    backend = "ollama" if args.ollama else config.BACKEND
    agent = Agent(model=LLM(backend=backend))

    try:
        agent.model.load()
    except (FileNotFoundError, ConnectionError, ImportError) as e:
        print_colored(f"Error: {e}", "red")
        sys.exit(1)

    agent.memory.load()
    agent.memory.new_conversation()

    print_colored("\nAI Agent ready! Type your messages.", "cyan")
    print_colored("Commands: /new /tools /quit", "yellow")
    print_colored("─" * 50, "bold")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == "/quit":
            break
        elif user_input.lower() == "/new":
            agent.memory.new_conversation()
            print_colored("New conversation started.", "cyan")
            continue
        elif user_input.lower() == "/tools":
            print(agent.tools.format_for_prompt())
            continue

        print_colored("AI: ", "green", end="")
        for chunk in agent.chat(user_input, stream=True):
            print(chunk, end="", flush=True)
        print()
        print_colored("─" * 50, "bold")


def run_api(args):
    from web import run_server
    run_server(host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="AI Agent")
    parser.add_argument("--cli", action="store_true", help="CLI mode")
    parser.add_argument("--web", action="store_true", help="Web server mode")
    parser.add_argument("--host", default=config.WEB_HOST)
    parser.add_argument("--port", type=int, default=config.WEB_PORT)
    parser.add_argument("--model", default="", help="Model path (GGUF) or name (Ollama)")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama backend")

    args = parser.parse_args()

    if args.model:
        config.MODEL_PATH = args.model
        config.OLLAMA_MODEL = args.model

    if args.cli:
        run_cli(args)
    else:
        run_api(args)


if __name__ == "__main__":
    main()
