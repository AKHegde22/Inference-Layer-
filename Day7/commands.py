import json
from session import ChatSession


ALLOWED_PARAMS = {
    "temperature":    float,
    "top_p":          float,
    "top_k":          int,
    "max_tokens":     int,
    "repeat_penalty": float,
}


def handle_command(cmd_str, session, client, config, display):
    parts = cmd_str.strip().split(maxsplit=2)
    cmd   = parts[0].lower()
    args  = parts[1:] if len(parts) > 1 else []

    if cmd == "/help":
        display.show_help()

    elif cmd == "/system":
        if args:
            session.set_system(" ".join(args))
            display.info("System prompt updated.")
        else:
            display.info(f"Current: {session.system_prompt}")

    elif cmd == "/clear":
        session.clear()
        display.info("Conversation cleared.")

    elif cmd == "/history":
        for i, m in enumerate(session.messages):
            tok     = m.get("tokens", "?")
            preview = m["content"][:60].replace("\n", " ")
            display.info(f"[{i}] {m['role']:10} | {tok:>4} tokens | {preview}")

    elif cmd == "/save":
        fname = args[0] if args else "chat.json"
        session.save(fname)

    elif cmd == "/load":
        fname = args[0] if args else "chat.json"
        loaded = ChatSession.load(fname)
        session.__dict__.update(loaded.__dict__)

    elif cmd == "/set":
        if len(args) >= 2:
            key, val_str = args[0], args[1]
            if key not in ALLOWED_PARAMS:
                display.info(f"Unknown param '{key}'. Allowed: {list(ALLOWED_PARAMS)}")
                return
            try:
                config[key] = ALLOWED_PARAMS[key](val_str)
                display.info(f"Set {key} = {config[key]}")
            except ValueError:
                display.info(f"Invalid value '{val_str}' for {key}")
        else:
            display.info("Usage: /set <key> <value>")

    elif cmd == "/params":
        for k, v in config.items():
            display.info(f"  {k:20} = {v}")

    elif cmd == "/bench":
        display.info("Running quick benchmark (3 prompts)...")
        results = client.quick_bench(config)
        display.info(f"Avg latency: {results['avg_latency']:.2f}s")
        display.info(f"Avg tok/s:   {results['avg_tps']:.1f}")

    elif cmd == "/exit":
        display.info("Goodbye!")
        raise SystemExit

    else:
        display.info(f"Unknown command: {cmd}. Type /help for help.")
