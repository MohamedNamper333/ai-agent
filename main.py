"""AI Agent - CLI interface"""

import logging
logger = logging.getLogger(__name__)

import argparse
import sys
import os

if sys.platform == "win32" and "pytest" not in sys.modules:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
    logger.info(f"{c}{text}{r}", end=end)


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

    print_colored("\nAI Agent ready! Type your messages. / الوكيل جاهز! اكتب رسالتك.", "cyan")
    print_colored("Commands / أوامر: /new /tools /fast /rag /quit", "yellow")
    print_colored("─" * 50, "bold")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info()
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "/quit":
            break
        elif cmd == "/new":
            agent.memory.new_conversation()
            print_colored("New conversation started.", "cyan")
            continue
        elif cmd == "/tools":
            print_colored("── Tools ──", "bold")
            logger.info(f"Enabled: {agent.tools.get_enabled_count()}/{len(agent.tools.list_all_tools())}")
            logger.info()
            cats = agent.tools.list_tools_by_category_all()
            for cat, tools in sorted(cats.items()):
                statuses = []
                for t in tools:
                    icon = "ON" if agent.tools.is_enabled(t.name) else "OFF"
                    statuses.append(f"  {t.name} [{icon}]")
                label = cat.title()
                logger.info(f" {label}:")
                for s in statuses:
                    logger.info(s)
                logger.info()
            logger.info("Usage: /tools enable <name> | /tools disable <name>")
            logger.info("       /tools on <category> | /tools off <category>")
            continue
        elif cmd.startswith("/tools enable "):
            name = cmd[14:].strip()
            if agent.tools.enable_tool(name):
                print_colored(f"Tool '{name}' enabled", "green")
            else:
                print_colored(f"Unknown tool: {name}", "red")
            continue
        elif cmd.startswith("/tools disable "):
            name = cmd[15:].strip()
            if agent.tools.disable_tool(name):
                print_colored(f"Tool '{name}' disabled", "yellow")
            else:
                print_colored(f"Unknown tool: {name}", "red")
            continue
        elif cmd.startswith("/tools on "):
            cat = cmd[10:].strip()
            count = agent.tools.enable_category(cat)
            print_colored(f"Enabled {count} tools in category '{cat}'", "green")
            continue
        elif cmd.startswith("/tools off "):
            cat = cmd[11:].strip()
            count = agent.tools.disable_category(cat)
            print_colored(f"Disabled {count} tools in category '{cat}'", "yellow")
            continue
        elif cmd == "/fast":
            if agent._fast_mode == "on":
                agent._fast_mode = "off"
                print_colored("Fast Mode: OFF", "yellow")
            else:
                agent._fast_mode = "on"
                print_colored("Fast Mode: ON", "green")
            continue
        elif cmd == "/rag":
            config.RAG_ENABLED = not config.RAG_ENABLED
            status = "ON" if config.RAG_ENABLED else "OFF"
            print_colored(f"RAG: {status}", "green" if config.RAG_ENABLED else "yellow")
            continue

        print_colored("AI: ", "green", end="")
        for chunk in agent.chat(user_input, stream=True):
            logger.info(chunk, end="", flush=True)
        logger.info()
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
