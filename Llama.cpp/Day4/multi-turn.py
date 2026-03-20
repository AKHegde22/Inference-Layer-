from openai import OpenAI

class ChatBot:
    def __init__(self, system="You are helpful.", port=8080, max_history_turns=10):
        self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="none")
        self.system = system
        self.max_history_turns = max_history_turns
        self.history = [{"role": "system", "content": system}]

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        self._trim_history()
        response = self.client.chat.completions.create(
            model="local",
            messages=self.history,
            temperature=0.7,
            max_tokens=500
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _trim_history(self):
        max_msgs = self.max_history_turns * 2
        if len(self.history) - 1 > max_msgs:
            self.history = [self.history[0]] + self.history[-max_msgs:]

    def reset(self):
        self.history = [{"role": "system", "content": self.system}]

    @property
    def token_count(self):
        total_words = sum(len(m["content"].split()) for m in self.history)
        return int(total_words * 1.3)

bot = ChatBot(system="You are a Python tutor.", max_history_turns=5)
print(bot.chat("What is a decorator?"))
print(bot.chat("Show me a simple example"))
print(bot.chat("How is it different from a class?"))
print(f"Estimated tokens in history: {bot.token_count}")