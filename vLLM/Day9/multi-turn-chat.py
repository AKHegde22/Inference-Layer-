import time
from openai import OpenAI

class ChatClient:
    def __init__(self, host="localhost", port=8000, model="mistral-7b",
                 system_prompt="You are helpful.", max_retries=3, timeout=30):
        self.client      = OpenAI(base_url=f"http://{host}:{port}/v1", api_key="none")
        self.model       = model
        self.max_retries = max_retries
        self.timeout     = timeout
        self.history     = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        reply = self._send_with_retry()
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def _send_with_retry(self):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    max_tokens=512, temperature=0.7,
                    timeout=self.timeout,
                )
                return resp.choices[0].message.content
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(f"Retry {attempt+1}/{self.max_retries} after {wait}s... ({e})")
                time.sleep(wait)
        raise RuntimeError(f"All {self.max_retries} retries failed: {last_error}")

    def stream_chat(self, user_message):
        self.history.append({"role": "user", "content": user_message})
        stream = self.client.chat.completions.create(
            model=self.model, messages=self.history,
            max_tokens=512, temperature=0.7, stream=True,
        )
        full_reply = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full_reply += delta
            yield delta
        self.history.append({"role": "assistant", "content": full_reply})

    def reset(self):
        self.history = [self.history[0]]

    @property
    def context_usage(self):
        est_tokens = int(sum(len(m["content"].split()) for m in self.history) * 1.3)
        return {
            "messages":    len(self.history),
            "est_tokens":  est_tokens,
            "pct_of_4096": round(est_tokens / 4096 * 100, 1),
        }


client = ChatClient(system_prompt="You are a Python expert.")
print(client.chat("What is a generator?"))
print(client.chat("Show a simple example."))
print(client.chat("How does yield work?"))
print("Context:", client.context_usage)