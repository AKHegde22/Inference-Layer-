import json
import datetime


class ChatSession:
    def __init__(self, system_prompt, max_history=20):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages = [
            {"role": "system", "content": system_prompt, "tokens": len(system_prompt.split())}
        ]

    def add(self, role, content, tokens=None):
        if tokens is None:
            tokens = int(len(content.split()) * 1.3)
        self.messages.append({"role": role, "content": content, "tokens": tokens})
        self._trim_if_needed()

    def _trim_if_needed(self):
        non_sys = [m for m in self.messages if m["role"] != "system"]
        if len(non_sys) > self.max_history * 2:
            keep = non_sys[-(self.max_history * 2):]
            self.messages = [self.messages[0]] + keep

    def api_messages(self):
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = [self.messages[0]]

    def set_system(self, new_prompt):
        self.system_prompt = new_prompt
        self.messages[0]["content"] = new_prompt
        self.messages[0]["tokens"] = len(new_prompt.split())

    @property
    def token_count(self):
        return sum(m.get("tokens", 0) for m in self.messages)

    @property
    def turn_count(self):
        return sum(1 for m in self.messages if m["role"] == "user")

    def save(self, filepath):
        data = {
            "version":       "1.0",
            "system_prompt": self.system_prompt,
            "messages":      self.messages,
            "token_count":   self.token_count,
            "turn_count":    self.turn_count,
            "saved_at":      datetime.datetime.now().isoformat(),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.messages)} messages to {filepath}")

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            data = json.load(f)
        if "messages" not in data:
            raise ValueError("Invalid chat file: missing 'messages' key")
        session = cls(data.get("system_prompt", "You are helpful."))
        session.messages = data["messages"]
        ts = data.get("saved_at", "unknown")
        print(f"Loaded {len(session.messages)} messages from {filepath} (saved at {ts})")
        return session

    def to_dict(self):
        return {
            "system_prompt": self.system_prompt,
            "messages":      self.messages,
            "token_count":   self.token_count,
            "turn_count":    self.turn_count,
        }
