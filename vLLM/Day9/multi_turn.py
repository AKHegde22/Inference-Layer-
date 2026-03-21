"""
Day 9 — Multi-turn conversation with vLLM
Maintains a message history and sends the full context each turn,
exactly like the OpenAI chat API expects.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


class VLLMChat:
    """
    Stateful multi-turn chat session against the vLLM server.

    The vLLM server itself is stateless — it doesn't remember previous turns.
    The client is responsible for maintaining history and sending it every call.
    This is identical to how the OpenAI API works.
    """

    def __init__(self, system: str = "You are a helpful assistant.", max_history: int = 20):
        self.system      = system
        self.max_history = max_history          # max user+assistant pairs to keep
        self.history: list[dict] = []
        self.total_tokens = 0

    def chat(self, user_message: str, stream: bool = True) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim()

        messages = [{"role": "system", "content": self.system}] + self.history

        if stream:
            reply = self._stream(messages)
        else:
            reply = self._complete(messages)

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _complete(self, messages: list) -> str:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content

    def _stream(self, messages: list) -> str:
        full = ""
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full += delta
        print()
        return full

    def _trim(self):
        """Drop oldest pairs when history exceeds max_history turns."""
        max_msgs = self.max_history * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def show_history(self):
        print(f"\n  History ({len(self.history)} messages | ~{self.total_tokens} tokens used):")
        for i, msg in enumerate(self.history):
            role    = msg["role"].upper()
            preview = msg["content"][:60].replace("\n", " ")
            print(f"  [{i+1}] {role}: {preview}...")

    def clear(self):
        self.history.clear()
        self.total_tokens = 0


if __name__ == "__main__":
    session = VLLMChat(system="You are a concise AI tutor. Keep answers under 3 sentences.")

    turns = [
        "What is a transformer model?",
        "What is the attention mechanism?",
        "How does that relate to the KV cache?",
    ]

    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        print("Assistant: ", end="")
        session.chat(user_msg)

    session.show_history()
