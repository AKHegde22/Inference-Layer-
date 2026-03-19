import sys
import argparse

from config import DEFAULT_CONFIG
from session import ChatSession
from client import LlamaClient
from commands import handle_command
from display import Display


def main():
    parser = argparse.ArgumentParser(description="LlamaCLI — Local AI Chatbot")
    parser.add_argument("--host",   type=str, default=DEFAULT_CONFIG["host"])
    parser.add_argument("--port",   type=int, default=DEFAULT_CONFIG["port"])
    parser.add_argument("--system", type=str, default=DEFAULT_CONFIG["system_prompt"])
    parser.add_argument("--model",  type=str, default=DEFAULT_CONFIG["model"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["host"]          = args.host
    config["port"]          = args.port
    config["system_prompt"] = args.system
    config["model"]         = args.model

    client  = LlamaClient(config["host"], config["port"])
    session = ChatSession(config["system_prompt"], config["max_history"])
    display = Display()

    if not client.health_check():
        print(f"ERROR: llama-server not running at {client.base_url}")
        print(f"Start it with: ./build/bin/llama-server -m model.gguf --port {config['port']}")
        sys.exit(1)

    display.banner(config)

    while True:
        try:
            user_input = input("\n\033[33mYou:\033[0m ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                handle_command(user_input, session, client, config, display)
            else:
                session.add("user", user_input)
                reply, stats = client.stream_chat(session.api_messages(), config)
                session.add("assistant", reply, stats["tokens"])
                display.show_stats(stats)

        except SystemExit:
            break
        except (KeyboardInterrupt, EOFError):
            display.info("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
