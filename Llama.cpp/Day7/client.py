import time
import requests
from openai import OpenAI


class LlamaClient:
    def __init__(self, host="localhost", port=8080):
        self.base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(base_url=self.base_url, api_key="none")
        self.model = "local"

    def health_check(self):
        try:
            root = self.base_url.replace("/v1", "")
            r = requests.get(f"{root}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def stream_chat(self, messages, params):
        t_start = time.time()
        full_text = ""
        token_count = 0

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            stream=True,
        )

        print("\n\033[36mAssistant:\033[0m ", end="", flush=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_text += delta
            token_count += len(delta.split())

        print()  # newline after response
        elapsed = time.time() - t_start
        tps = token_count / elapsed if elapsed > 0 else 0

        return full_text, {"tokens": token_count, "elapsed": elapsed, "tps": tps}

    def quick_bench(self, params):
        prompts = ["What is 2+2?", "Name a color.", "Say hi."]
        times, toks = [], []
        for p in prompts:
            _, stats = self.stream_chat([{"role": "user", "content": p}], params)
            times.append(stats["elapsed"])
            toks.append(stats["tokens"])
        return {
            "avg_latency": sum(times) / len(times),
            "avg_tokens":  sum(toks) / len(toks),
            "avg_tps":     sum(t / e for t, e in zip(toks, times)) / len(times),
        }
