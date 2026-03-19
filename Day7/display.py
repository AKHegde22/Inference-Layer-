class Display:
    def banner(self, config):
        print("\n" + "═" * 54)
        print("  🦙 LlamaCLI  —  Local AI Chat")
        print(f"  Host:  {config['host']}:{config['port']}")
        print(f"  Model: {config['model']}  |  CTX: {config['context_limit']}")
        print("═" * 54)
        print("  Type /help for commands, /exit to quit")
        print("═" * 54 + "\n")

    def info(self, msg):
        print(f"\033[90m{msg}\033[0m")

    def show_stats(self, stats):
        print(
            f"\033[90m  [{stats['tokens']} tokens | "
            f"{stats['elapsed']:.2f}s | "
            f"{stats['tps']:.1f} tok/s]\033[0m"
        )

    def show_help(self):
        lines = [
            "  /help              — show this message",
            "  /system <text>     — change system prompt",
            "  /clear             — clear conversation history",
            "  /history           — show message history with token counts",
            "  /save <file>       — save conversation to JSON",
            "  /load <file>       — load conversation from JSON",
            "  /params            — show current sampling parameters",
            "  /set <key> <val>   — change a parameter",
            "  /bench             — run a quick benchmark",
            "  /exit              — quit",
        ]
        for line in lines:
            self.info(line)
